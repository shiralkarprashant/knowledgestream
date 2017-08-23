#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for Successive Shortest Path (SSP) algorithm that iteratively finds the 
maxflow with minimum cost.
"""
import os
import sys
import numpy as np
import logging as log
import cProfile
import warnings

from os.path import expanduser, abspath, isfile, isdir, basename, splitext, \
	dirname, join, exists
from datastructures.rgraph import make_graph, Graph
from algorithms.mincostflow.ssp import succ_shortest_path, disable_logging


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
	G.sources = np.repeat(np.arange(G.N), np.diff(G.csr.indptr))
	G.targets = G.csr.indices % G.N
	cost_vec = G.indeg_vec
	print "Original graph:\n", G

	# Successive shortest path algorithm
	s, p, o = 0, 1, 3
	expect = 6.42857142857
	mincostflow = succ_shortest_path(G, cost_vec, s, p, o)
	print mincostflow
	assert np.allclose(mincostflow.flow, expect)

	print 'Recovered max-flow edges (i, j, r, flow)..'
	adj = np.zeros((len(mincostflow.edges), 4))
	for i, (k, v) in enumerate(mincostflow.edges.iteritems()):
		adj[i, :] = np.array([k[0], k[1], k[2], v])
	adj = adj[np.lexsort((adj[:,2], adj[:,1], adj[:,0])),:]
	print adj
	print ''

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
	G.sources = np.repeat(np.arange(G.N), np.diff(G.csr.indptr))
	G.targets = G.csr.indices % G.N
	cost_vec = G.indeg_vec
	print "Original graph:\n", G

	# Successive shortest path algorithm
	s, p, o = 0, 2, 5
	expect = 2.88888888889
	mincostflow = succ_shortest_path(G, cost_vec, s, p, o)
	print mincostflow
	assert np.allclose(mincostflow.flow, expect)

	print 'Recovered max-flow edges (i, j, r, flow)..'
	adj = np.zeros((len(mincostflow.edges), 4))
	for i, (k, v) in enumerate(mincostflow.edges.iteritems()):
		adj[i, :] = np.array([k[0], k[1], k[2], v])
	adj = adj[np.lexsort((adj[:,2], adj[:,1], adj[:,0])),:]
	print adj
	print ''

def test_dbpedia():
	dirpath = abspath(expanduser('./data/kg/_undir/'))
	shape = (6060993, 6060993, 663)
	G = Graph.reconstruct(dirpath, shape, sym=True)
	cost_vec = np.log(G.indeg_vec)
	
	s, p, o = 2145431, 178, 459128 # Gravity, Alfonso Cuarón
	mincostflow = succ_shortest_path(G, cost_vec, s, p, o)
	print mincostflow


if __name__ == '__main__':
	disable_logging(log.DEBUG)
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		test_graph1()
		test_graph2()
		# test_dbpedia()
		# cProfile.run('test_graph2()', sort='time')
