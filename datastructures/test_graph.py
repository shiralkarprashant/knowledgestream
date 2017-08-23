import os
import sys
import numpy as np
import argparse
import pandas as pd
import shutil

from time import time
from os.path import exists, join, abspath, expanduser, basename, dirname, isdir

from datastructures.rgraph import make_graph, Graph

def test_graph1_creation():
	shape = np.asarray([4, 4, 2], dtype=np.int32)
	adj = np.asarray([
		[0, 1, 0],
		[0, 2, 1],
		[1, 2, 0],
		[1, 2, 1],
		[1, 3, 1],
		[2, 3, 0],
		[2, 1, 1],
		[2, 3, 1],
	], dtype=np.int32)
	values = np.arange(adj.shape[0]) + 10.

	# create graph
	expect_G = np.asarray([
		[ 0., 1., 0., 0., 0., 0., 1., 0.],
		[ 1., 0., 1., 0., 0., 0., 1., 1.],
		[ 0., 1., 0., 1., 1., 1., 0., 1.],
		[ 0., 0., 1., 0., 0., 1., 1., 0.]
 	])
	G = make_graph(adj, shape, sym=True, save_csc=True)
	assert np.array_equal(G.csr.toarray(), G.csc.toarray())
	dirpath = join(abspath(expanduser(os.curdir)), '_undir')
	if not exists(dirpath):
		os.mkdir(dirpath)
	G.save_graph(dirpath)
	assert np.array_equal(G.indeg_vec, np.asarray([2, 3, 3, 2]))
	assert np.array_equal(expect_G, G.csr.toarray())

	# rebuild graph
	G = Graph.reconstruct(dirpath, shape, sym=True, save_csc=True)
	assert np.array_equal(expect_G, G.csr.toarray())
	if exists(dirpath):
		shutil.rmtree(dirpath)
		print 'Removed: %s' % dirpath

def test_graph2_creation():
	# shape of an example graph.
	shape = (8, 8, 6) 

	# adjacency
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
	A = np.asarray([
		[
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 1., 0., 0.],
			[ 0., 0., 0., 0., 0., 1., 0., 0.],
			[ 0., 0., 0., 1., 1., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 1.],
			[ 0., 0., 0., 0., 0., 0., 1., 0.]
		],

		[
			[ 0., 0., 0., 1., 1., 1., 1., 1.],
			[ 0., 0., 0., 1., 1., 1., 0., 0.],
			[ 0., 0., 0., 1., 0., 1., 0., 0.],
			[ 1., 1., 1., 0., 0., 0., 0., 0.],
			[ 1., 1., 0., 0., 0., 0., 0., 0.],
			[ 1., 1., 1., 0., 0., 0., 0., 0.],
			[ 1., 0., 0., 0., 0., 0., 0., 0.],
			[ 1., 0., 0., 0., 0., 0., 0., 0.]
		],

		[
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 1., 0., 0., 0., 0.],
			[ 0., 0., 1., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.]
		],

		[
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 1., 1.],
			[ 0., 0., 0., 0., 0., 0., 1., 1.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 1., 1., 0., 0., 0.],
			[ 0., 0., 0., 1., 1., 0., 0., 0.]
		],

		[
			[ 0., 1., 1., 0., 0., 0., 0., 0.],
			[ 1., 0., 0., 0., 0., 0., 0., 0.],
			[ 1., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.],
			[ 0., 0., 0., 0., 0., 0., 0., 0.]
		],

		[
			[ 0., 0., 0., 1., 1., 1., 1., 1.],
			[ 0., 0., 0., 1., 1., 1., 0., 0.],
			[ 0., 0., 0., 1., 1., 1., 1., 1.],
			[ 1., 1., 1., 0., 0., 0., 0., 0.],
			[ 1., 1., 1., 0., 0., 0., 0., 0.],
			[ 1., 1., 1., 0., 0., 0., 0., 0.],
			[ 1., 0., 1., 0., 0., 0., 0., 0.],
			[ 1., 0., 1., 0., 0., 0., 0., 0.]
		]
	])
	G = make_graph(adj, shape, sym=True, save_csc=True)
	assert np.array_equal(G.csr.toarray(), G.csc.toarray())
	assert np.array_equal(G.indeg_vec, np.asarray([7, 4, 6, 6, 6, 5, 5, 5]))
	for k in xrange(shape[2]):
		assert np.array_equal(G.getslice(k).toarray(), A[k, :, :])

def test_dbpedia():
	adjpath = abspath(expanduser('./data/kg/adjacency.npy'))
	shape = (6060993, 6060993, 663)
	adj = np.load(adjpath)
	adj = adj.astype(np.int32)
	T = Graph(adj, shape, sym=True)

	# save graph
	print 'Saving graph..'
	t1 = time()
	dirpath = join(dirname(adjpath), '_undir')
	if not exists(dirpath):
		os.makedirs(dirpath)
		print '* Created: %s' % dirpath
	T.save_graph(dirpath)
	print 'Graph saved in {:.4f} secs at: {} '.format(time() - t1, dirpath)

def test_dbpedia_loading():
	adjpath = abspath(expanduser('~/Projects/truthy_data/dbpedia/2016-04/processed/kg/adjacency.npy'))
	shape = (6060993, 6060993, 663)
	dirpath = join(dirname(adjpath), '_undir')
	G = Graph.reconstruct(dirpath, shape, sym=True)
	assert np.all(G.csr.indices >= 0)

	# reverse graph
	dirpath = join(dirname(adjpath), '_revundir')
	revG = Graph.reconstruct(dirpath, shape, sym=True)
	assert np.all(revG.csr.indices >= 0)



if __name__ == '__main__':
	# test_dbpedia()
	# test_dbpedia_loading()
	# test_graph2_creation()
	# test_graph1_creation()
	# import cProfile
	# cProfile.run('test_graph1_creation()', sort='time')