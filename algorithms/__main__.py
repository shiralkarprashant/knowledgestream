"""
Entry point for Knowledge Stream (KS) and 
Relational Knowledge Linker (KL-REL) algorithm.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import warnings
import ujson as json
import logging as log

from pandas import DataFrame, Series
from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from time import time
from datetime import date

from datastructures.rgraph import Graph, weighted_degree
from algorithms.mincostflow.ssp import succ_shortest_path, disable_logging
from algorithms.relklinker.rel_closure import relational_closure as relclosure

# KG - DBpedia
HOME = abspath(expanduser('~/Projects/knowledgestream/data/'))
if not exists(HOME):
	raise Exception('Please set HOME to data directory in algorithms/__main__.py')
PATH = join(HOME, 'kg/_undir/')
assert exists(PATH)
SHAPE = (6060993, 6060993, 663)
WTFN = 'logdegree'

# relational similarity using TF-IDF representation and cosine similarity
RELSIMPATH = join(HOME, 'relsim/coo_mat_sym_2016-10-24_log-tf_tfidf.npy') 
assert exists(RELSIMPATH)

# Date
DATE = '{}'.format(date.today())

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

# ================= KNOWLEDGE STREAM ALGORITHM ============

def compute_mincostflow(G, relsim, subs, preds, objs, flowfile):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	relsim: ndarray
		A square matrix containing relational similarity scores.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of 
		input triples.
	flowfile: str
		Absolute path of the file where flow will be stored as JSON,
		one line per triple.

	Returns:
	--------
	mincostflows: sequence
		A sequence containing total flow for each triple.
	times: sequence
		Times taken to compute stream of each triple. 
	"""
	# take graph backup
	G_bak = {
		'data': G.csr.data.copy(), 
		'indices': G.csr.indices.copy(),
		'indptr': G.csr.indptr.copy()
	}
	cost_vec_bak = np.log(G.indeg_vec).copy()

	# some set up
	G.sources = np.repeat(np.arange(G.N), np.diff(G.csr.indptr))
	G.targets = G.csr.indices % G.N
	cost_vec = cost_vec_bak.copy()
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN)
	specificity_wt = indegsim[G.targets] # specificity
	relations = (G.csr.indices - G.targets) / G.N
	mincostflows, times = [], []
	with open(flowfile, 'w', 0) as ff:
		for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
			s, p, o = [int(x) for x in (s, p, o)]
			ts = time()
			print '{}. Working on {} .. '.format(idx+1, (s, p, o)),
			sys.stdout.flush()

			# set weights
			relsimvec = np.array(relsim[p, :]) # specific to predicate p
			relsim_wt = relsimvec[relations]
			G.csr.data = np.multiply(relsim_wt, specificity_wt)
			
			# compute
			mcflow = succ_shortest_path(
				G, cost_vec, s, p, o, return_flow=False, npaths=5
			)
			mincostflows.append(mcflow.flow)
			ff.write(json.dumps(mcflow.stream) + '\n')
			tend = time()
			times.append(tend - ts)
			log.info('mincostflow: {:.5f}, #paths: {}, time: {:.2f}s.'.format(
				mcflow.flow, len(mcflow.stream['paths']), tend - ts
			))

			# reset state of the graph
			np.copyto(G.csr.data, G_bak['data'])
			np.copyto(G.csr.indices, G_bak['indices'])
			np.copyto(G.csr.indptr, G_bak['indptr'])
			np.copyto(cost_vec, cost_vec_bak)
	return mincostflows, times

# ================= RELATIONAL KNOWLEDGE LINKER ALGORITHM ============

def compute_relklinker(G, relsim, subs, preds, objs):
	"""
	Parameters:
	-----------
	G: rgraph
		See `datastructures`.
	relsim: ndarray
		A square matrix containing relational similarity scores.
	subs, preds, objs: sequence
		Sequences representing the subject, predicate and object of 
		input triples.

	Returns:
	--------
	scores, paths, rpaths, times: sequence
		One sequence each for the proximity scores, shortest path in terms of 
		nodes, shortest path in terms of relation sequence, and times taken.
	"""
	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight=WTFN).reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()

	# relation vector
	relations = (G.csr.indices - targets) / G.N

	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	scores, paths, rpaths, times = [], [], [], []
	for idx, (s, p, o) in enumerate(zip(subs, preds, objs)):
		print '{}. Working on {}..'.format(idx+1, (s, p, o)),
		ts = time()
		# set relational weight
		G.csr.data[targets == o] = 1 # no cost for target t => max. specificity.
		relsimvec = relsim[p, :] # specific to predicate p
		relsim_wt = relsimvec[relations] # graph weight
		G.csr.data = np.multiply(relsim_wt, G.csr.data)

		rp = relclosure(G, s, p, o, kind='metric', linkpred=True)
		tend = time()
		log.info('time: {:.2f}s'.format(tend - ts))
		times.append(tend - ts)
		scores.append(rp.score)
		paths.append(rp.path)
		rpaths.append(rp.relational_path)

		# reset graph
		G.csr.data = data.copy()
		G.csr.indices = indices.copy()
		G.csr.indptr = indptr.copy()
		sys.stdout.flush()
	log.info('')
	return scores, paths, rpaths, times


def main(args=None):
	# parse arguments
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-m', type=str, required=True,
			dest='method', help='Method to use: stream or relklinker.')
	parser.add_argument('-d', type=str, required=True,
			dest='dataset', help='Dataset to test on.')
	parser.add_argument('-o', type=str, required=True,
			dest='outdir', help='Path to the output directory.')
	args = parser.parse_args()

	# logging
	disable_logging(log.DEBUG)

	if args.method not in ('stream', 'relklinker'):
		raise Exception('Invalid method specified.')

	# ensure input file and output directory is valid.
	outdir = abspath(expanduser(args.outdir))
	assert exists(outdir)
	args.outdir = outdir
	datafile = abspath(expanduser(args.dataset))
	assert exists(datafile)
	args.dataset = datafile
	log.info('Launching {}..'.format(args.method))
	log.info('Dataset: {}'.format(basename(args.dataset)))
	log.info('Output dir: {}'.format(args.outdir))

	# read data
	df = pd.read_table(args.dataset, sep=',', header=0)
	log.info('Read data: {} {}'.format(df.shape, basename(args.dataset)))
	spo_df = df.dropna(axis=0, subset=['sid', 'pid', 'oid'])
	log.info('Note: Found non-NA records: {}'.format(spo_df.shape))
	df = spo_df[['sid', 'pid', 'oid']].values
	subs, preds, objs  = df[:,0].astype(_int), df[:,1].astype(_int), df[:,2].astype(_int)

	# load knowledge graph
	G = Graph.reconstruct(PATH, SHAPE, sym=True) # undirected
	assert np.all(G.csr.indices >= 0)

	# relational similarity
	relsim = np.load(RELSIMPATH)

	# execute
	base = splitext(basename(args.dataset))[0]
	t1 = time()
	if args.method == 'stream':
		# compute min. cost flow
		log.info('Computing KNOWLEDGE-STREAM for {} triples..'.format(spo_df.shape[0]))
		with warnings.catch_warnings():
			warnings.simplefilter("ignore")
			outjson = join(args.outdir, 'out_kstream_{}_{}.json'.format(base, DATE))
			outcsv = join(args.outdir, 'out_kstream_{}_{}.csv'.format(base, DATE))
			mincostflows, times = compute_mincostflow(G, relsim, subs, preds, objs, outjson)
			# save the results
			spo_df['score'] = mincostflows
			spo_df['time'] = times
			spo_df.to_csv(outcsv, sep=',', header=True, index=False)
			log.info('* Saved results: %s' % outcsv)
		log.info('Mincostflow computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))
	elif args.method == 'relklinker':
		log.info('Computing REL-KLINKER for {} triples..'.format(spo_df.shape[0]))
		scores, paths, rpaths, times = compute_relklinker(G, relsim, subs, preds, objs)
		# save the results
		spo_df['score'] = scores
		spo_df['path'] = paths
		spo_df['rpath'] = rpaths
		spo_df['time'] = times
		outcsv = join(args.outdir, 'out_relklinker_{}_{}.csv'.format(base, DATE))
		spo_df.to_csv(outcsv, sep=',', header=True, index=False)
		log.info('* Saved results: %s' % outcsv)
		log.info('Relklinker computation complete. Time taken: {:.2f} secs.\n'.format(time() - t1))

if __name__ == '__main__':
	"""
	Example calls: 

	cd ~/Projects/knowledgestream/
	python setup.py develop OR python setup.py install

	# Knowledge Stream:
	kstream -m 'stream' -d ./datasets/synthetic/Player_vs_Team_NBA.csv -o ./output/
	kstream -m 'stream' -d ./datasets/sample.csv -o ./output/

	# Relational Knowledge Linker (KL-REL)

	"""
	main()