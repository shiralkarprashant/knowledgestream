"""
Similarity by SimRank (SR) score, using multi-step neighborhood of two nodes.

Source: Scalable Similarity Search for SimRank by Mitsuru Kusumoto et al.
"""
import os
import sys
import numpy as np

from numpy.random import choice

from algorithms.linkpred.simrank_helper import simrank_helper

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

def simrank(G, s, p, o, c=0.8, T=50, R=1000, linkpred=True):
	"""
	Returns SimRank (SR) score computed based on multi-step neighborhood 
	between s and o. 

	Note: Currently, only works for undirected graphs.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	s, p, o: int
		Subject, Predicate and Object identifiers. 
	c: float
		Decay factor, typically 0.6 or 0.8.
	T: int
		Number of terms in Eqn. (9) or (13) that determines score precision.
	R: int
		Number of random walks that are started from s and o.
	linkpred: bool
		Whether or not to perform link prediction.
	
	Returns:
	--------
	score: float
		A score >= 0.
	"""
	# link prediction: second condition is to avoid unnecessary introduction 
	# of a zero in the if clause
	if linkpred and G[o, s, p] != 0: 
		G[s, o, p] = 0
	score = 0.
	u = np.ones(R, dtype=_int) * s
	v = np.ones(R, dtype=_int) * o
	for t in xrange(T):
		# print 'U: {}, V: {}'.format(u, v)
		uu = dict(zip(*np.unique(u, return_counts=True)))
		vv = dict(zip(*np.unique(v, return_counts=True)))
		common_nbrs = set(uu.keys()) & set(vv.keys())
		for w in common_nbrs:
			score += (c ** t) * (1. - c) * uu[w] * vv[w] / float(R ** 2)
		# print '[{}] Score: {}'.format(t+1, score)
		for r in xrange(R):
			nbrs = np.unique(G.get_neighbors(u[r])[1, :])
			u[r] = choice(nbrs, size=1, replace=True)
			nbrs = np.unique(G.get_neighbors(v[r])[1, :])
			v[r] = choice(nbrs, size=1, replace=True)
	return score

def c_simrank(G, s, p, o, c=0.8, T=50, R=1000, linkpred=True):
	"""
	Same as simrank above, except this method is compiled
	(not completely cythonized however).
	"""
	if linkpred and G[o, s, p] != 0: 
		G[s, o, p] = 0
	u = np.ones(R, dtype=_int) * s
	v = np.ones(R, dtype=_int) * o
	score = simrank_helper(G, u, v, c, T, R)
	return score