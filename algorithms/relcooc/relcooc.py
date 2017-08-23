"""
Computing co-occurrence counts of relations in a knowledge graph (KG)
using a line graph representation.

"""
import os
import sys
import numpy as np
import argparse

from os.path import isdir, exists, join, abspath, expanduser, dirname, basename
from datetime import date
from time import time

from flow.datastructures.rgraph import make_graph, Graph
from _relcooc import compute_cooccurrence as c_relcooc

def get_outgoing_relations(G, i):
	"Returns a vector of relations directed out of node i."
	nbrs = G.csr.indices[G.csr.indptr[i]:G.csr.indptr[i+1]]
	rels = (nbrs - (nbrs % G.N)) / G.N
	return rels

def compute_cooccurrence(G, revG):
	"""Computes relational co-occurrence for input graph 
	using line-graph representation.
	
	Parameters:
	-----------
	G: rgraph
		A graph object representing a knowledge graph.
	revG: rgraph
		A reverse graph of G.

	Returns:
	--------
	cooc_mat: ndarray
		A matrix of co-occurrence counts for pairwise relations.
	"""
	N, R = G.N, G.R
	cooc_mat = np.zeros((R, R))
	z = np.zeros(R, dtype=np.int)
	for node in xrange(N):
		invec = get_outgoing_relations(revG, node)
		outvec = get_outgoing_relations(G, node)
		if len(invec) == 0 or len(outvec) == 0:
			continue
		in_uq, in_cnt = np.unique(invec, return_counts=True)
		out_uq, out_cnt = np.unique(outvec, return_counts=True)
		i, j = z.copy(), z.copy()
		i[in_uq], j[out_uq] = in_cnt, out_cnt
		u = np.outer(i, j)
		cooc_mat += u
	return cooc_mat

if __name__ == '__main__':
	"""
	Example call:

	# DBpedia
	python relcooc.py 
		-path ~/Projects/truthy_data/dbpedia/2016-04/processed/kg/_undir/ 
		-shape 6060993 6060993 663 
		-outpath ~/Projects/truthy_data/dbpedia/2016-04/processed/relCooc/

	# SemMedDB
	python relcooc.py 
		-path ~/Projects/truthy_data/semmeddb/processed/kg/_undir/ 
		-shape 282934 282934 58
		-outpath ~/Projects/truthy_data/semmeddb/processed/relCooc/

	"""
	parser = argparse.ArgumentParser(
		description=__doc__, 
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)
	parser.add_argument('-path', type=str, required=True,
			dest='path', help='Path to input graph directory. \
				Expects data, indices, indptr files.')
	parser.add_argument('-shape', type=int, nargs=3, required=True,
			dest='shape', help='Shape of KG (n x n x r)')
	parser.add_argument('-outpath', type=str, 
			dest='outpath', help='Absolute path to the output directory.')
	args = parser.parse_args()

	# arguments
	path = abspath(expanduser(args.path)) # adjacency or edge-list path
	undir = True if 'undir' in basename(path) else False
	shape = tuple(args.shape)
	suffix = 'sym' if undir else 'asym'
	outpath = dirname(path) if args.outpath is None else abspath(expanduser(args.outpath))
	outfname = join(outpath, 'coo_mat_{}_{}.npy'.format(suffix, date.today()))

	# checks & make output directory if not exists
	if not exists(outpath):
		os.mkdir(outpath)
	if not exists(path):
		raise Exception('Graph directory does not exist.')
	if not isdir(path):
		raise Exception('Graph directory expected as path.')
	if len(shape) != 3:
		raise Exception('Shape of multi-relational graph expected to be 3-tuple.')

	# log input parameters
	print '\n# INPUT:'
	print 'Graph path: {}'.format(path)
	print 'Graph type: {}'.format('undir' if undir else 'dir')
	print 'Output file: {}'.format(outfname)
	print ''

	# read graph
	t1 = time()
	G = Graph.reconstruct(path, shape, sym=undir)
	assert np.all(G.csr.indices >= 0)
	sys.stdout.flush()

	# read reverse graph
	t1 = time()
	revpath = join(dirname(path), '_revundir')
	revG = Graph.reconstruct(path, shape, sym=undir)
	assert np.all(revG.csr.indices >= 0)
	sys.stdout.flush()

	# compute co-occurrence of relation pairs.
	t1 = time()
	# cooc_mat = compute_cooccurrence(G, revG)
	cooc_mat = c_relcooc(G, revG)
	print 'Rel. cooccurrence matrix created: {:.4f} secs.'.format(time() - t1)
	np.save(outfname, cooc_mat)
	print 'Saved rel. cooccurrence matrix at: {}'.format(outfname)

	print '\nDone!\n'
