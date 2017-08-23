"""
Computes similarity between relations (relational similarity) based on their
co-occurrence counts in a knowledge graph (KG) as previously computed 
using a line graph representation (see relcooc.py).

input> relations.txt cooc*.npy
output> *_SCHEME_tfidf.npy *_idf.csv

Note: This is a working version of scratch code in 
the notebook 'cooc_mat_sym_analysis.ipynb'.
"""
import os
import sys
import numpy as np
import argparse
import re
import pandas as pd

from pandas import DataFrame, Series
from os.path import splitext, isdir, exists, join, abspath, \
	expanduser, dirname, basename
from datetime import date
from time import time
from scipy.spatial.distance import cdist

SCHEME = 'log-tf' # CHANGE THIS AS REQUIRED

def tfidf_relational_similarity_scheme1(d_org):
	"""
	SCHEME 1 ("log-tf"): Computes TF/IDF-like similarity between pair of relations
	based on co-occurrence of the relations.
	
	* Note: The values of similarity generally decrease *slowly* under this
	ranking.
	
	Parameters:
	-----------
	d_org: ndarray
		An square array containing number of pairwise co-occurrences
		of relations.
	
	Returns:
	--------
	idf: ndarray
		A vector of raw inverse-document frequency, where each document is a relation.
	tfidf_sim: ndarray
		An ndarray containing similarity between 
		relation i and relation j. Each entry is between 0 and 1, inclusive.
	"""   
	r = d_org.shape[0] # no of relations
	def idf_fn(x):
		"IDF of each relation."
		N = len(x)
		n = float(np.count_nonzero(x))
		res = np.log(N/n)
		return res
	idf = np.apply_along_axis(idf_fn, 0, d_org)
	idf_mat = np.tile(idf, r).reshape((r, len(idf)))
	tf = np.log(1 + d_org) # TF: raw co-occurence frequency
	tfidf = tf * idf_mat # TF-IDF: TF x IDF
	tfidf_sim = 1. - cdist(tfidf, tfidf, metric='cosine')
	return idf, tfidf_sim

def tfidf_relational_similarity_scheme2(d_org):
	"""
	SCHEME 2 ("max-tf"): Computes TF/IDF-like similarity between pair of relations
	based on co-occurrence of the relations.
	
	* Note: The values of similarity generally decrease *rapidly* under this
	ranking.
	
	Parameters:
	-----------
	d_org: ndarray
		An square array containing number of pairwise co-occurrences
		of relations.
	
	Returns:
	--------
	idf: ndarray
		A vector of raw inverse-document frequency, where each document is a relation.
	tfidf_sim: ndarray
		An ndarray containing similarity between 
		relation i and relation j. Each entry is between 0 and 1, inclusive.
	"""   
	r = d_org.shape[0] # no of relations
	term_freq = np.apply_along_axis(lambda x: np.max(x), 0, d_org)
	doc_freq = np.apply_along_axis(lambda x: np.count_nonzero(x), 1, d_org)
	def idf_fn(x):
		"IDF for each relation."
		N = len(x)
		n = float(np.count_nonzero(x))
		res = np.log(N/n)
		return res
	idf = np.apply_along_axis(idf_fn, 0, d_org)
	idf_mat = np.tile(idf, r).reshape((r, len(idf)))
	tf = np.apply_along_axis(lambda x: x/term_freq, 1, d_org) # TF: doc frequeny normalized
	tfidf = tf * idf_mat # TF-IDF: TF x IDF
	tfidf_sim = 1. - cdist(tfidf, tfidf, metric='cosine')
	return idf, tfidf_sim

def get_relational_similarity(d_org, scheme='log-tf', return_idf=False):
	"""
	Returns relational similarity derived from relation co-occurrence data.
	
	Parameters:
	-----------
	d_org: ndarray
		A square array of co-occurrence counts.
	scheme: str
		Indicative of specific TF-IDF scheme. See respective functions above.
		
	Returns:
	--------
	idf: ndarray
		A 1-d vector of IDF of relations.
	tfidf_sim: ndarray
		An (r x r) matrix of pairwise similarity between relations.
	"""
	if scheme == 'log-tf':
		print 'Using log-tf...'
		idf, tfidf_sim = tfidf_relational_similarity_scheme1(d_org)
	elif scheme == 'max-tf':
		print 'Using max-tf...'
		idf, tfidf_sim = tfidf_relational_similarity_scheme2(d_org)
	if return_idf:
		return idf, tfidf_sim
	return tfidf_sim

def top_similar_relations(d, relations, relidx=None, rel=None, top=10, display=True):
	"""
	Returns the (scores, indices) of the top 'top' similar relations for the 
	input relation. This is based on similarity matrix 'd', previously
	computed using TF/IDF inspired approach.
	
	Either the input relation index or its name needs
	to be specified. 
	"""
	if relidx is None and rel is None:
		raise Exception('One of relidx and rel needs to be specified')
	if rel is not None:
		relidx = [k for k, v in relations.iteritems() if v == rel][0]
	top_sim_rels = np.argsort(d[relidx,:])[::-1][:top+1]
	scores = [d[relidx, relid] for relid in top_sim_rels]
	if display:
		print '\n=> Relation: {} {}'.format(relidx, relations[relidx])
		for idx, (relid, sc) in enumerate(zip(top_sim_rels, scores)):
			print '  {}.\t{} [{}] {}'.format(idx+1, relations[relid], 
				relid, round(sc, 4)
			)
		print ''
	return scores, top_sim_rels

def main(args):
	# load co-occ counts
	d_org = np.load(args.cooc)
	print 'Cooc. mat: {}'.format(d_org.shape)

	# relsim
	idf, tfidf_sim = get_relational_similarity(d_org, scheme=SCHEME, return_idf=True)

	# read relations
	relations = pd.read_table(args.rels, sep=' ', header=None)
	relations = dict(zip(relations[0].values, relations[1].values))
	rels = [relations[i] for i in xrange(len(relations))]
	print 'Relations: {}'.format(len(relations))

	if args.top is not None or args.r is not None:
		if re.match(r'[0-9]+', args.r):
			relidx = int(args.r)
			top_similar_relations(tfidf_sim, relations, relidx=relidx, top=args.top)
		else:
			top_similar_relations(tfidf_sim, relations, rel=args.r, top=args.top)
	else:
		# save files
		_fname = splitext(basename(args.cooc))[0]
		rel_sim_fname = '{}_{}_tfidf.npy'.format(_fname, SCHEME)
		rel_sim_fname = join(dirname(args.cooc), rel_sim_fname)
		if exists(rel_sim_fname):
			print 'File already exists: %s' % rel_sim_fname
			ans = raw_input('Overwrite? ')
			if ans == 'n':
				sys.exit()
		np.save(rel_sim_fname, tfidf_sim)
		print 'Saved TFIDF-based rel. cooc: {}'.format(rel_sim_fname)

		# save IDF
		idf_fname = join(dirname(args.cooc), '{}_idf.csv'.format(_fname))
		cnts = np.apply_along_axis(lambda x: np.count_nonzero(x), 0, d_org)
		idf_df = DataFrame.from_dict({'relation': rels, 'idf': idf, 'cnts': cnts})
		idf_df = idf_df[['relation', 'idf', 'cnts']]
		idf_df = idf_df.sort_values(by='idf', ascending=False)
		idf_df['rank'] = np.arange(idf_df.shape[0]) + 1
		idf_df['idf_norm'] = idf_df['idf']/float(idf_df['idf'].max())
		idf_df.to_csv(idf_fname, sep=",", header=True, index=False)
		print 'Saved IDF info: {}'.format(idf_fname)

if __name__ == '__main__':
	"""
	Example:

	# generating tf-idf matrix and IDF vector
	python relsim.py 
		-rels ~/Projects/truthy_data/dbpedia/2016-04/processed/kg/relations.txt
		-cooc ~/Projects/truthy_data/dbpedia/2016-04/processed/relCooc/coo_mat_sym_2016-10-24.npy

	python relsim.py 
		-rels ~/Projects/truthy_data/semmeddb/processed/kg/relations.txt
		-cooc ~/Projects/truthy_data/semmeddb/processed/relCooc/coo_mat_sym_2017-04-23.npy

	# looking at top relations 
	python relsim.py 
		-rels ~/Projects/truthy_data/dbpedia/2016-04/processed/kg/relations.txt
		-cooc ~/Projects/truthy_data/dbpedia/2016-04/processed/relCooc/coo_mat_sym_2016-10-24.npy
		-top 10
		-r 'dbo:spouse'

	python relsim.py 
		-rels ~/Projects/truthy_data/semmeddb/processed/kg/relations.txt
		-cooc ~/Projects/truthy_data/semmeddb/processed/relCooc/coo_mat_sym_2017-04-23.npy
		-top 10
		-r 'causes'
	"""
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-rels', type=str, required=True, dest='rels', 
		help='Path to relations file.')
	parser.add_argument('-cooc', type=str, required=True, dest='cooc', 
		help='Path to co-occurrence counts .npy file.')
	parser.add_argument('-r', type=str, dest='r', 
		help='Relation index or relation (e.g. dbo:spouse)')
	parser.add_argument('-top', type=int, dest='top', 
		help='A flag to show top X similar relations for \
		the relation specified by flag -r')
	args = parser.parse_args()

	args.rels = abspath(expanduser(args.rels))
	args.cooc = abspath(expanduser(args.cooc))
	
	main(args)
	print '\nDone!\n'
