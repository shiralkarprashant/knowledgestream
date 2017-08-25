# cython: profile=False
import numpy as np

from numpy.random import choice

# c imports
cimport cython
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.set cimport set as c_set

cpdef simrank_helper(G, int[:] u, int[:] v, double c, int T, int R):
	cdef:
		int t, w, r
		double score, deno, diag
		dict uu, vv, nbr_cache
		set common_nbrs
	score = 0.
	deno = float(R ** 2)
	diag = (1. - c)
	for t in xrange(T):
		# print 'U: {}, V: {}'.format(u, v)
		uu = dict(zip(*np.unique(u, return_counts=True)))
		vv = dict(zip(*np.unique(v, return_counts=True)))
		common_nbrs = set(uu.keys()) & set(vv.keys())
		for w in common_nbrs:
			score += (c ** t) * diag * uu[w] * vv[w] / deno
		# print '[{}] Score: {}'.format(t+1, score)
		nbr_cache = dict()
		for r in xrange(R):
			if u[r] not in nbr_cache:
				nbr_cache[u[r]] = np.unique(G.get_neighbors(u[r])[1, :])
			if v[r] not in nbr_cache:
				nbr_cache[v[r]] = np.unique(G.get_neighbors(v[r])[1, :])
			u[r] = choice(nbr_cache[u[r]], size=1, replace=True)
			v[r] = choice(nbr_cache[v[r]], size=1, replace=True)
	return score