"""
A few tests for testing Knowledge Linker - RELATIONAL (KL-REL), 
the extention to PLOS ONE paper that uses line graph based 
approach to compute relational similarity.

* Note: The relational similarity for 'toy' graphs in these tests
is 1. and hence, the algorithm output is same as that from KL-BASIC.
"""
import os
import sys
import numpy as np
import logging as log
import cProfile

from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from datastructures.rgraph import make_graph, weighted_degree
from algorithms.relklinker.rel_closure import relational_closure as relclosure

allclose = lambda x, y: np.all([np.allclose(np.asarray(a), np.asarray(b)) for a, b in zip(x, y)])

def test_graph1():
	sym = True
	adj = np.array([
		[0, 2, 0, 10],
		[1, 2, 0, 30],
		[0, 1, 1, 20],
		[1, 3, 1, 10],
		[2, 3, 1, 20]
	])
	shape = (4, 4, 2)
	G = make_graph(adj[:,:3], shape, values=adj[:,3], sym=sym, display=False)
	print "Original graph:\n", G

	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight='degree').reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()

	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	# Closure
	expect = [ # s, p, o, path, rpath
		[0, 0, 1, 1.0, [0, 1], [-1, 1]],
		[0, 0, 2, 0.25, [0, 1, 2], [-1, 1, 0]],
		[0, 0, 3, 0.25, [0, 2, 3], [-1, 0, 1]],
		[0, 1, 1, 0.25, [0, 2, 1], [-1, 0, 0]],
		[0, 1, 2, 1.0, [0, 2], [-1, 0]],
		[0, 1, 3, 0.25, [0, 2, 3], [-1, 0, 1]],
		[1, 0, 0, 1.0, [1, 0], [-1, 1]],
		[1, 0, 2, 0.33333333333333331, [1, 3, 2], [-1, 1, 1]],
		[1, 0, 3, 1.0, [1, 3], [-1, 1]],
		[1, 1, 0, 0.25, [1, 2, 0], [-1, 0, 0]],
		[1, 1, 2, 1.0, [1, 2], [-1, 0]],
		[1, 1, 3, 0.25, [1, 2, 3], [-1, 0, 1]],
		[2, 0, 0, 0.25, [2, 1, 0], [-1, 0, 1]],
		[2, 0, 1, 0.33333333333333331, [2, 3, 1], [-1, 1, 1]],
		[2, 0, 3, 1.0, [2, 3], [-1, 1]],
		[2, 1, 0, 1.0, [2, 0], [-1, 0]],
		[2, 1, 1, 1.0, [2, 1], [-1, 0]],
		[2, 1, 3, 0.25, [2, 1, 3], [-1, 0, 1]],
		[3, 0, 0, 0.25, [3, 2, 0], [-1, 1, 0]],
		[3, 0, 1, 1.0, [3, 1], [-1, 1]],
		[3, 0, 2, 1.0, [3, 2], [-1, 1]],
		[3, 1, 0, 0.25, [3, 2, 0], [-1, 1, 0]],
		[3, 1, 1, 0.25, [3, 2, 1], [-1, 1, 0]],
		[3, 1, 2, 0.25, [3, 1, 2], [-1, 1, 0]]
	]
	results = []
	itr = 0
	for s in xrange(G.N):
		for p in xrange(G.R):
			for o in xrange(G.N):
				if s == o:
					continue
				G.csr.data[targets == o] = 1
				rp = relclosure(G, s, p, o, kind='metric', linkpred=True)
				tmp = [rp.source, rp.relation, rp.target, rp.score, rp.path, rp.relational_path]
				results.append(tmp)
				assert allclose(expect[itr], tmp)
				itr += 1
				G.csr.data = data.copy()
				G.csr.indices = indices.copy()
				G.csr.indptr = indptr.copy()
	
def test_graph2():
	sym = True
	adj = np.array([
		[0, 1, 0, 16],
		[2, 4, 0, 14],
		[4, 5, 0, 4],
		[0, 2, 1, 13],
		[2, 1, 1, 4],
		[3, 5, 1, 20],
		[1, 3, 2, 12],
		[3, 2, 2, 9],
		[4, 3, 2, 7]
	])
	shape = (6, 6, 3)
	G = make_graph(adj[:,:3], shape, values=adj[:,3], sym=sym, display=False)
	print "Original graph:\n", G

	# set weights
	indegsim = weighted_degree(G.indeg_vec, weight='degree').reshape((1, G.N))
	indegsim = indegsim.ravel()
	targets = G.csr.indices % G.N
	specificity_wt = indegsim[targets] # specificity
	G.csr.data = specificity_wt.copy()

	# back up
	data = G.csr.data.copy()
	indices = G.csr.indices.copy()
	indptr = G.csr.indptr.copy()

	# Closure
	expect = [
		[0, 0, 1, 0.20000000000000001, [0, 2, 1], [-1, 1, 1]],
		[0, 0, 2, 1.0, [0, 2], [-1, 1]],
		[0, 0, 3, 0.25, [0, 1, 3], [-1, 0, 2]],
		[0, 0, 4, 0.20000000000000001, [0, 2, 4], [-1, 1, 0]],
		[0, 0, 5, 0.125, [0, 1, 3, 5], [-1, 0, 2, 1]],
		[0, 1, 1, 1.0, [0, 1], [-1, 0]],
		[0, 1, 2, 0.25, [0, 1, 2], [-1, 0, 1]],
		[0, 1, 3, 0.25, [0, 1, 3], [-1, 0, 2]],
		[0, 1, 4, 0.20000000000000001, [0, 2, 4], [-1, 1, 0]],
		[0, 1, 5, 0.125, [0, 1, 3, 5], [-1, 0, 2, 1]],
		[0, 2, 1, 1.0, [0, 1], [-1, 0]],
		[0, 2, 2, 1.0, [0, 2], [-1, 1]],
		[0, 2, 3, 0.25, [0, 1, 3], [-1, 0, 2]],
		[0, 2, 4, 0.20000000000000001, [0, 2, 4], [-1, 1, 0]],
		[0, 2, 5, 0.125, [0, 1, 3, 5], [-1, 0, 2, 1]],
		[1, 0, 0, 0.20000000000000001, [1, 2, 0], [-1, 1, 1]],
		[1, 0, 2, 1.0, [1, 2], [-1, 1]],
		[1, 0, 3, 1.0, [1, 3], [-1, 2]],
		[1, 0, 4, 0.20000000000000001, [1, 3, 4], [-1, 2, 2]],
		[1, 0, 5, 0.20000000000000001, [1, 3, 5], [-1, 2, 1]],
		[1, 1, 0, 1.0, [1, 0], [-1, 0]],
		[1, 1, 2, 0.33333333333333331, [1, 0, 2], [-1, 0, 1]],
		[1, 1, 3, 1.0, [1, 3], [-1, 2]],
		[1, 1, 4, 0.20000000000000001, [1, 3, 4], [-1, 2, 2]],
		[1, 1, 5, 0.20000000000000001, [1, 3, 5], [-1, 2, 1]],
		[1, 2, 0, 1.0, [1, 0], [-1, 0]],
		[1, 2, 2, 1.0, [1, 2], [-1, 1]],
		[1, 2, 3, 0.20000000000000001, [1, 2, 3], [-1, 1, 2]],
		[1, 2, 4, 0.20000000000000001, [1, 3, 4], [-1, 2, 2]],
		[1, 2, 5, 0.20000000000000001, [1, 3, 5], [-1, 2, 1]],
		[2, 0, 0, 1.0, [2, 0], [-1, 1]],
		[2, 0, 1, 1.0, [2, 1], [-1, 1]],
		[2, 0, 3, 1.0, [2, 3], [-1, 2]],
		[2, 0, 4, 0.20000000000000001, [2, 3, 4], [-1, 2, 2]],
		[2, 0, 5, 0.25, [2, 4, 5], [-1, 0, 0]],
		[2, 1, 0, 0.25, [2, 1, 0], [-1, 1, 0]],
		[2, 1, 1, 0.33333333333333331, [2, 0, 1], [-1, 1, 0]],
		[2, 1, 3, 1.0, [2, 3], [-1, 2]],
		[2, 1, 4, 1.0, [2, 4], [-1, 0]],
		[2, 1, 5, 0.25, [2, 4, 5], [-1, 0, 0]],
		[2, 2, 0, 1.0, [2, 0], [-1, 1]],
		[2, 2, 1, 1.0, [2, 1], [-1, 1]],
		[2, 2, 3, 0.25, [2, 1, 3], [-1, 1, 2]],
		[2, 2, 4, 1.0, [2, 4], [-1, 0]],
		[2, 2, 5, 0.25, [2, 4, 5], [-1, 0, 0]],
		[3, 0, 0, 0.25, [3, 1, 0], [-1, 2, 0]],
		[3, 0, 1, 1.0, [3, 1], [-1, 2]],
		[3, 0, 2, 1.0, [3, 2], [-1, 2]],
		[3, 0, 4, 1.0, [3, 4], [-1, 2]],
		[3, 0, 5, 1.0, [3, 5], [-1, 1]],
		[3, 1, 0, 0.25, [3, 1, 0], [-1, 2, 0]],
		[3, 1, 1, 1.0, [3, 1], [-1, 2]],
		[3, 1, 2, 1.0, [3, 2], [-1, 2]],
		[3, 1, 4, 1.0, [3, 4], [-1, 2]],
		[3, 1, 5, 0.25, [3, 4, 5], [-1, 2, 0]],
		[3, 2, 0, 0.25, [3, 1, 0], [-1, 2, 0]],
		[3, 2, 1, 0.20000000000000001, [3, 2, 1], [-1, 2, 1]],
		[3, 2, 2, 0.25, [3, 4, 2], [-1, 2, 0]],
		[3, 2, 4, 0.33333333333333331, [3, 5, 4], [-1, 1, 0]],
		[3, 2, 5, 1.0, [3, 5], [-1, 1]],
		[4, 0, 0, 0.20000000000000001, [4, 2, 0], [-1, 0, 1]],
		[4, 0, 1, 0.20000000000000001, [4, 3, 1], [-1, 2, 2]],
		[4, 0, 2, 0.20000000000000001, [4, 3, 2], [-1, 2, 2]],
		[4, 0, 3, 1.0, [4, 3], [-1, 2]],
		[4, 0, 5, 0.20000000000000001, [4, 3, 5], [-1, 2, 1]],
		[4, 1, 0, 0.20000000000000001, [4, 2, 0], [-1, 0, 1]],
		[4, 1, 1, 0.20000000000000001, [4, 3, 1], [-1, 2, 2]],
		[4, 1, 2, 1.0, [4, 2], [-1, 0]],
		[4, 1, 3, 1.0, [4, 3], [-1, 2]],
		[4, 1, 5, 1.0, [4, 5], [-1, 0]],
		[4, 2, 0, 0.20000000000000001, [4, 2, 0], [-1, 0, 1]],
		[4, 2, 1, 0.20000000000000001, [4, 3, 1], [-1, 2, 2]],
		[4, 2, 2, 1.0, [4, 2], [-1, 0]],
		[4, 2, 3, 0.33333333333333331, [4, 5, 3], [-1, 0, 1]],
		[4, 2, 5, 1.0, [4, 5], [-1, 0]],
		[5, 0, 0, 0.125, [5, 4, 2, 0], [-1, 0, 0, 1]],
		[5, 0, 1, 0.20000000000000001, [5, 3, 1], [-1, 1, 2]],
		[5, 0, 2, 0.25, [5, 4, 2], [-1, 0, 0]],
		[5, 0, 3, 1.0, [5, 3], [-1, 1]],
		[5, 0, 4, 0.20000000000000001, [5, 3, 4], [-1, 1, 2]],
		[5, 1, 0, 0.125, [5, 4, 2, 0], [-1, 0, 0, 1]],
		[5, 1, 1, 0.20000000000000001, [5, 3, 1], [-1, 1, 2]],
		[5, 1, 2, 0.25, [5, 4, 2], [-1, 0, 0]],
		[5, 1, 3, 0.25, [5, 4, 3], [-1, 0, 2]],
		[5, 1, 4, 1.0, [5, 4], [-1, 0]],
		[5, 2, 0, 0.125, [5, 4, 2, 0], [-1, 0, 0, 1]],
		[5, 2, 1, 0.20000000000000001, [5, 3, 1], [-1, 1, 2]],
		[5, 2, 2, 0.25, [5, 4, 2], [-1, 0, 0]],
		[5, 2, 3, 1.0, [5, 3], [-1, 1]],
		[5, 2, 4, 1.0, [5, 4], [-1, 0]]
	]
	results = []
	itr = 0
	for s in xrange(G.N):
		for p in xrange(G.R):
			for o in xrange(G.N):
				if s == o:
					continue
				G.csr.data[targets == o] = 1
				rp = relclosure(G, s, p, o, kind='metric', linkpred=True)
				tmp = [rp.source, rp.relation, rp.target, rp.score, rp.path, rp.relational_path]
				results.append(tmp)
				assert allclose(expect[itr], tmp)
				itr += 1
				G.csr.data = data.copy()
				G.csr.indices = indices.copy()
				G.csr.indptr = indptr.copy()
	
if __name__ == '__main__':
	test_graph1()
	test_graph2()
