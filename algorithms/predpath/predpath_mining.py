"""
PredPath (PP) model building and prediction.

Source: 'Discriminative Predicate Path Mining for Fact-Checking 
in Knowledge Graphs' by Baoxu Shi and Tim Weninger.

Performs three things:
- Path extraction: Extracts anchored predicate paths as features, and constructs feature matrix 
for a given set of triples.
- Path selection: Computes mutual information / information gain between features and label 
for identifying discriminative predicate paths.
- Model building: Trains a logistic regression model that optimizes AUROC and empirically
sets a threshold 'delta' for retaining most informative feature paths.
"""
import os
import sys
import numpy as np
import argparse
import pandas as pd

from time import time
from os.path import exists, join, abspath, expanduser, basename, dirname, \
	isdir, splitext
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from datastructures.rgraph import make_graph, Graph
from datastructures.relationalpath import RelationalPath
from algorithms.predpath.pathenum import get_paths as c_get_paths

# ================ WRAPPER ================

def train_model(G, triples, use_interpretable_features=False, cv=10):
	"""
	Entry point for building a fact-checking classifier.
	Performs three steps:
	1. Path extraction (features)
	2a. Path selection using information gain
	2b. Filtering most informative discriminative predicate paths
	3. Building logistic regression model
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: dataframe
		A data frame consisting of at least four columns, including
		sid, pid, oid, class.
	use_interpretable_features: bool
		Whether or not to perform 2b.
	cv: int
		Number of cross-validation folds.
	
	Returns:
	--------
	vec: DictVectorizer
		Useful for preprocessing future triples.
	model: dict
		A dictionary containing 'clf' as the built model,
		and two other key-value pairs, including best parameter
		and best AUROC score.
	"""
	y = triples['class'] # ground truth
	triples = triples[['sid', 'pid', 'oid']].to_dict(orient='records')

	# Remove all edges in G corresponding to predicate p.
	pid = triples[0]['pid']
	print '=> Removing predicate {} from KG.'.format(pid)
	eraseedges_mask = ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid
	G.csr.data[eraseedges_mask] = 0 
	print ''

	# Path extraction
	print '=> Path extraction..(this can take a while)'
	t1 = time()
	features, pos_features, neg_features, measurements = extract_paths(G, triples, y)
	print 'P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features))
	vec = DictVectorizer()
	X = vec.fit_transform(measurements)
	n, m = X.shape
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''
	
	# Path selection
	print '=> Path selection..'
	t1 = time()
	pathselect = SelectKBest(mutual_info_classif, k=min(100, m))
	X_select = pathselect.fit_transform(X, y)
	selectidx = pathselect.get_support(indices=True) # selected feature indices
	vec = vec.restrict(selectidx, indices=True)
	select_pos_features, select_neg_features = set(), set()
	for feature in vec.get_feature_names():
		if feature in pos_features:
			select_pos_features.add(feature)
		if feature in neg_features:
			select_neg_features.add(feature)
	print 'D: +:{}, -:{}, tot:{}'.format(
		len(select_pos_features), len(select_neg_features), X_select.shape[1]
	)
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''

	# Fact interpretation
	if use_interpretable_features and len(select_neg_features) > 0:
		print '=> Fact interpretation..'
		t1 = time()
		theta = 10
		select_neg_idx = [i for i, f in enumerate(vec.get_feature_names()) if f in select_neg_features]
		removemask = np.where(np.sum(X_select[:, select_neg_idx], axis=0) >= theta)[0]
		restrictidx = select_neg_idx[removemask]
		keepidx = []
		for i, f in enumerate(vec.get_feature_names()):
			if i not in restrictidx:
				keepidx.append(i)
			else:
				select_neg_features.remove(f)
		vec = vec.restrictidx(keepidx, indices=True)
		X_select = X_select[:, keepidx]
		print 'D*: +:{}, -:{}, tot:{}'.format(
			len(select_pos_features), len(select_neg_features), X_select.shape[1]
		)
		print 'Time taken: {:.2f}s'.format(time() - t1)
		print ''
 
	# Model creation
	print '=> Model building..'
	t1 = time()
	model = find_best_model(X_select, y, cv=cv)
	print '#Features: {}, best-AUROC: {:.5f}'.format(X_select.shape[1], model['best_score'])
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''

	return vec, model


def predict(G, triples, vec, model):
	"""
	Predicts unseen triples using previously built PredPath (PP) model.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: dataframe
		A data frame consisting of at least four columns, including
		sid, pid, oid, class.
	vec: DictVectorizer
		For preprocessing triples.
	model: dict
		A dictionary containing 'clf' as the built model,
		and two other key-value pairs, including best parameter
		and best AUROC score.
	
	Returns:
	--------
	pred: array
		An array of predicttions, 1 or 0.
	"""
	y = triples['class'] # ground truth
	triples = triples[['sid', 'pid', 'oid']].to_dict(orient='records')

	# Path extraction
	print '=> Path extraction.. (this can take a while)'
	t1 = time()
	features, pos_features, neg_features, measurements = extract_paths(G, triples, y)
	print 'P: +:{}, -:{}, unique tot:{}'.format(len(pos_features), len(neg_features), len(features))
	X = vec.fit_transform(measurements)
	pred = model['clf'].predict(X) # array
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''
	return pred

# ================ PATH EXTRACTION ================

def extract_paths(G, triples, y, length=3, features=None):
	"""
	Extracts anchored predicate paths for a given sequence of triples.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: sequence
		A list of triples (sid, pid, oid).
	y: array
		A sequence of class labels.
	length: int
		Maximum length of any path.
	features: dict
		Features extracted earlier. A set of (feature_id, path) pairs.
		If None, it is assumed feature set and feature matrix are desired.
		If not None, only X (feature matrix) is returned.

	Returns:
	--------
	features: dict
		A set of (feature_id, path) pairs.
	X: dict
		A dictionary representation of feature matrix.
	"""
	return_features = False
	if features is None:
		return_features = True
		features, pos_features, neg_features = set(), set(), set()
	measurements = []
	for idx, triple in enumerate(triples):
		sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
		label = y[idx]

		# extract paths for a triple
		triple_feature = dict()
		for m in xrange(length + 1):
			if m in [0, 1]: # paths of length 0 and 1 mean nothing
				continue
			paths = c_get_paths(G, sid, pid, oid, length=m, maxpaths=200) # cythonized
			for pth in paths:
				ff = tuple(pth.relational_path) # feature
				if ff not in features:
					features.add(ff)
					if label == 1:
						pos_features.add(ff)
					elif label == 0:
						neg_features.add(ff)
					else:
						raise Exception('Unknown class label: {}'.format(label))
				triple_feature[ff] = triple_feature.get(ff, 0) + 1
		measurements.append(triple_feature)
		# print '(T:{}, F:{})'.format(idx+1, len(triple_feature))
		sys.stdout.flush()
	print ''
	if return_features:
		return features, pos_features, neg_features, measurements
	return measurements
	
def get_paths(G, s, p, o, length=3):
	"Returns all paths of length `length` starting at s and ending in o."
	path_stack = [[s]]
	relpath_stack = [[-1]]
	discoverd_paths = []
	while len(path_stack) > 0:
		# print 'Stack: {} {}'.format(path_stack, relpath_stack)
		curr_path = path_stack.pop()
		curr_relpath = relpath_stack.pop()
		node = curr_path[-1]
		# print 'Node: {}'.format(node)
		if len(curr_path) == length + 1:
			if node == o:
				# create a path
				path = RelationalPath(
					s, p, o, 0., length, curr_path, curr_relpath, np.ones(length+1)
				)
				discoverd_paths.append(path)
			continue
		relnbrs = G.get_neighbors(node)
		for i in xrange(relnbrs.shape[1]):
			rel, nbr = relnbrs[:, i]
			path_stack.append(curr_path + [nbr])
			relpath_stack.append(curr_relpath + [rel])
	return discoverd_paths


# ================ MODEL BUILDING & SELECTION ================

def find_best_model(X, y, scoring='roc_auc', cv=10):
	"""
	Fits a logistic regression classifier to the input data (X, y),
	and returns the best model that maximizes `scoring` (e.g. AUROC).
	
	Parameters:
	-----------
	X: sparse matrix
		Feature matrix.
	y: array
		A vector of ground truth labels.
	scoring: str
		A string indicating the evaluation criteria to use. e.g. ROC curve.
	cv: int
		No. of folds in cross-validation.
	
	Returns:
	--------
	best: dict
		Best model key-value pairs. e.g. classifier, best score on 
		left out data, optimal parameter.
	"""
	steps = [('clf', LogisticRegression())]
	pipe = Pipeline(steps)
	params = {'clf__C': [1, 5, 10, 15, 20]}
	grid_search = GridSearchCV(pipe, param_grid=params, cv=cv, refit=True, scoring=scoring)
	grid_search.fit(X, y)
	best = {
		'clf': grid_search.best_estimator_, 
		'best_score': grid_search.best_score_,
		'best_param': grid_search.best_params_
	}
	return best
