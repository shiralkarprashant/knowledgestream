"""
Tests co-occurrence count computation of relation pairs
by line graph representation. 

See relcooc.py for details.
"""
import os
import sys
import numpy as np
import pandas as pd
import cProfile

from scipy.sparse import csr_matrix
from pandas import DataFrame, Series

from datastructures.rgraph import Graph, make_graph
from _relcooc import compute_cooccurrence


def test_graph1():
	"""Test graph."""
	shape = (4, 4, 3)
	adj = np.array([
		[0, 1, 0],
		[1, 2, 0],
		[2, 3, 0],
		[0, 2, 1],
		[1, 2, 1],
		[1, 3, 1],
		[2, 1, 1],
		[2, 3, 1],
		[0, 1, 2],
		[1, 2, 2],
		[1, 3, 2]
	])
	revadj = adj.copy()
	ii = revadj[:,0].copy()
	revadj[:,0] = revadj[:,1]
	revadj[:,1] = ii

	# UNDIRECTED
	G = make_graph(adj, shape, sym=True, display=False)
	revG = make_graph(revadj, shape, sym=True, display=False)
	expect_cooc_mat_sym = np.array([
		[10, 13, 10],
		[13, 18, 12],
		[10, 12, 12]
	])
	cooc_mat_sym = compute_cooccurrence(G, revG)
	assert np.array_equal(expect_cooc_mat_sym, cooc_mat_sym)
	print ''

	# DIRECTED
	G = make_graph(adj, shape, sym=False, display=False)
	revG = make_graph(revadj, shape, sym=False, display=False)
	expect_cooc_mat_asym = np.array([
		[2, 4, 2],
		[3, 6, 2],
		[2, 4, 2]
	])
	cooc_mat_asym = compute_cooccurrence(G, revG)
	assert np.array_equal(expect_cooc_mat_asym, cooc_mat_asym)

def test_graph2():
	"""
	Co-occurrence count on a small graph, e.g. CNetS/IU graph.
	"""
	shape = (8, 8, 6)
	# s, o, p
	adj = np.array([ 
		[3, 5, 0],
		[4, 5, 0],
		[5, 3, 0],
		[5, 4, 0],
		[6, 7, 0],
		[7, 6, 0],
		[3, 0, 1],
		[3, 1, 1],
		[3, 2, 1],
		[4, 0, 1],
		[4, 1, 1],
		[5, 0, 1],
		[5, 1, 1],
		[5, 2, 1],
		[6, 0, 1],
		[7, 0, 1],
		[3, 2, 2],
		[3, 6, 3],
		[3, 7, 3],
		[4, 6, 3],
		[4, 7, 3],
		[1, 0, 4],
		[2, 0, 4],
		[0, 3, 5],
		[0, 4, 5],
		[0, 5, 5],
		[0, 6, 5],
		[0, 7, 5],
		[1, 3, 5],
		[1, 4, 5],
		[1, 5, 5],
		[2, 3, 5],
		[2, 4, 5],
		[2, 5, 5],
		[2, 6, 5],
		[2, 7, 5]
	])
	revadj = adj.copy()
	ii = revadj[:,0].copy()
	revadj[:,0] = revadj[:,1]
	revadj[:,1] = ii

	# UNDIRECTED cooccurrence matrix
	G = make_graph(adj, shape, sym=True, display=False)
	revG = make_graph(revadj, shape, sym=True, display=False)
	expect_cooc_mat_sym = np.array([
		[ 8, 13,  1,  8,  0, 16],
		[13, 62,  5, 14, 15, 72],
		[ 1,  5,  2,  2,  1,  8],
		[ 8, 14,  2, 16,  0, 20],
		[ 0, 15,  1,  0,  6, 18],
		[16, 72,  8, 20, 18, 94]
	])
	cooc_mat_sym = compute_cooccurrence(G, revG)
	assert np.array_equal(expect_cooc_mat_sym, cooc_mat_sym)
	print '\n', '=' * 25

	# DIRECTED cooccurrence matrix
	expect_cooc_mat_asym = np.array([
		[  8, 13,  1,  4,  0,  0],
		[  0,  0,  0,  0,  5, 44],
		[  0,  0,  0,  0,  1,  5],
		[  4,  4,  0,  0,  0,  0],
		[  0,  0,  0,  0,  0, 10],
		[ 16, 28,  3, 12,  0,  0]
	])
	G = make_graph(adj, shape, sym=False, display=False)
	revG = make_graph(revadj, shape, sym=False, display=False)
	cooc_mat_asym = compute_cooccurrence(G, revG)
	assert np.array_equal(expect_cooc_mat_asym, cooc_mat_asym)

if __name__ == '__main__':
	test_graph1()
	test_graph2()
	# cProfile.run('test_graph2()', sort='time')