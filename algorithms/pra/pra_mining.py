"""
Path Ranking Algorithm (PRA) model building and prediction.

Performs three things:
- Path extraction: Extracts paths (or sequence of relations) using node pairs given as 
training examples. These relational paths (or so-called "path types") serve as features 
in the PRA model.
- Path/Feature selection: Since there can be a large number of such features, some sort of 
feature selection is performed, either via computing precision/recall or just using the 
frequency of each feature. We use the latter in this implementation.
- Model building: Trains a logistic regression model, and returns the weights 
to rank individual features.
"""
import os
import sys
import numpy as np
import argparse
import pandas as pd

from time import time
from os.path import exists, join, abspath, expanduser, basename, dirname, \
	isdir, splitext
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from datastructures.rgraph import make_graph, Graph
from datastructures.relationalpath import RelationalPath
from algorithms.pra.pra_helper \
	import extract_paths as c_extract_paths, \
	construct_feature_matrix as c_construct_feature_matrix

# ================ WRAPPER ================

def train_model(G, triples, maxfeatures=100, cv=10):
	"""
	Entry point for building a PRA model.
	Performs three steps:
	1. Path extraction (features)
	2. Path selection using most frequently seen features.
	3. Building logistic regression model
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: dataframe
		A data frame consisting of at least four columns, including
		sid, pid, oid, class.
	maxfeatures: int
		Maximum number of features to use in the model.
	cv: int
		Number of cross-validation folds.
	
	Returns:
	--------
	features: list 
		A list of features. Index represents the index in feature matrix.
	model: dict
		A dictionary containing 'clf' as the built model,
		and two other key-value pairs, including best parameter
		and best AUROC score.
	"""
	y = triples['class'] # ground truth
	triples = triples[['sid', 'pid', 'oid']].astype(np.int).to_dict(orient='records')

	# Remove all edges in G corresponding to predicate p.
	pid = triples[0]['pid']
	print '=> Removing predicate {} from KG.'.format(pid)
	eraseedges_mask = ((G.csr.indices - (G.csr.indices % G.N)) / G.N) == pid
	G.csr.data[eraseedges_mask] = 0 
	print ''

	# Path extraction
	print '=> Path extraction..'
	t1 = time()
	features = c_extract_paths(G, triples, maxpaths=200) # cythonized
	print '\n#Features: {}'.format(len(features))
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''
	
	# Path selection
	print '=> Path selection..'
	t1 = time()
	features = sorted(features.iteritems(), key=lambda x: x[1], reverse=True)[:maxfeatures]
	features = [f for f, _ in features] # frequency ignored henceforth
	print 'Selected {} features'.format(len(features))
	print ''

	# Feature matrix construction
	print '=> Constructing feature matrix..'
	t1 = time()
	X = c_construct_feature_matrix(G, features, triples, maxwalks=1000)
	print 'Time taken: {:.5f}s'.format(time() - t1)
	print ''
	
 
	# Model creation
	print '=> Model building..'
	t1 = time()
	model = find_best_model(X, y, cv=cv)
	print '#Features: {}, best-AUROC: {:.4f}'.format(X.shape[1], model['best_score'])
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''
	
	# weights = model['clf'].named_steps['clf'].coef_
	return features, model


def predict(G, features, model, triples):
	"""
	Predicts unseen triples using previously built PRA model.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	model: dict
		A dictionary containing 'clf' as the built model,
		and two other key-value pairs, including best parameter
		and best AUROC score.
	features: list
		List of features, per index in feature matrix.
	triples: dataframe
		A data frame consisting of at least four columns, including
		sid, pid, oid, class.
	
	
	Returns:
	--------
	pred: array
		An array of predicttions, 1 or 0.
	"""
	triples = triples[['sid', 'pid', 'oid']].astype(np.int).to_dict(orient='records')

	# Path extraction
	print '=> Path extraction..'
	t1 = time()
	X = c_construct_feature_matrix(G, features, triples)
	pred = model['clf'].predict(X) # array
	print 'Time taken: {:.2f}s'.format(time() - t1)
	print ''
	return pred

# ================ FEATURE MATRIX CONSTRUCTION ================

def construct_feature_matrix(G, features, triples, maxwalks=1000):
	"""
	Constructs a feature matrix where the paths extracted in previous
	step (input here) act as features. A value in the matrix corresponding 
	to a training instance (s, p, o) is the probability of arriving at 
	node o by a random walk starting at node s and a specific path (feature).

	There are exact and approximate approaches to compute this. Exact approach 
	is costly however: complexity is proportional to the average per-edge-label 
	(or per-relation) out-degree to the power of the path length per source node 
	per path. Hence, we compute this probability using the approximate method of 
	rejection sampling. It works as follows: a number of random walks are performed
	starting at node 's', each attempting to follow the path (feature). If a 
	node is reached where it is no longer possible to follow the path, the 
	random walk is restarted. The fraction of walks that land in node 'o' 
	represents the probability.

	Parameters:
	-----------
	G: rgraph
		Knowledge Graph.
	features: list
		List of features, per index in feature matrix.
	triples: list
		A list of training instance triples, each represented by a dict, and all 
		belonging to the same relation/predicate p.
	maxwalks: int
		Number of random walks to perform. The higher, the better the resulting 
		approximation. However, this also means increased computational time.
	
	Returns:
	--------
	mtx: sparse matrix
		A sparse matrix containing paths as features, node pairs as rows, and a 
		non-zero value as an entry corresponding to the probability computed 
		as above.
	"""
	mat = np.empty((len(triples), len(features)), dtype=np.float)
	for idx, triple in enumerate(triples):
		sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
		print 'Working on triple {}'.format(idx+1)
		t1 = time()
		for pthidx, path in enumerate(features):
			path = path[1:] # exclude -1 in front
			cnt = 0
			for walk in xrange(maxwalks):
				node = sid
				for r in path:
					relnbrs = G.get_neighbors(node, k=r)
					n_nbrs = relnbrs.shape[1]
					if n_nbrs == 0:
						break # restart random walk
					else:
						node = np.random.choice(relnbrs[1, :], 1)[0] # pick 1 nbr uniformly at random
				if node == oid:
					cnt += 1
			# print '\t{}. {}/{} for {}'.format(pthidx+1, cnt, maxwalks, path)
			mat[idx, pthidx] = cnt / float(maxwalks)
		print time() - t1
	return mat

# ================ PATH EXTRACTION ================

def extract_paths(G, triples, length=3):
	"""
	Extracts a dictionary of (feature, frequency) pairs,
	based on a set of training instances, that serves as 
	features in the model.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: sequence
		A list of triples (sid, pid, oid).
	length: int
		Maximum length of any path.

	Returns:
	--------
	features: dict
		A set of (feature, frequency) pairs.
	"""
	features = dict()
	for idx, triple in enumerate(triples):
		sid, pid, oid = triple['sid'], triple['pid'], triple['oid']
		print 'Working on triple {}'.format(idx+1)

		# extract paths for a triple
		for m in xrange(length + 1):
			if m in [0, 1]: # paths of length 0 and 1 mean nothing
				continue
			paths = c_get_paths(G, sid, pid, oid, length=m) # cythonized
			for pth in paths:
				ff = tuple(pth.relational_path) # feature
				features[ff] = features.get(ff, 0) + 1
	return features
	
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
