"""
A Cython implementation of the code in 'relcooc.py' to 
static type variables and speed up computations in the loop.
"""
import os
import sys
from time import time

import numpy as np

# c imports
cimport cython
cimport numpy as np

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef _compute_cooccurrence(
		int R, np.ndarray indices, np.ndarray indptr,
		np.ndarray revindices, np.ndarray revindptr
	):
	cdef:
		int N, node, start, end, itr, ptr
		# ==== numpy matrices ====
		np.ndarray cooc_mat, u
		# ==== numpy vectors ====
		np.ndarray z, nbrs, invec, outvec, progress 
		np.ndarray in_uq, in_cnt, out_uq, out_cnt, i, j
	N = len(indptr) - 1
	cooc_mat = np.zeros((R, R))
	z = np.zeros(R, dtype=np.int)
	itr = 0
	progress = np.round(np.linspace(0, N, num=50, endpoint=True))[1:]
	ptr = 0
	t1 = time()
	for node in range(N):
		# invec
		start = revindptr[node]
		end = revindptr[node + 1]
		nbrs = revindices[start:end]
		invec = (nbrs - (nbrs % N)) / N
		
		# outvec
		start = indptr[node]
		end = indptr[node + 1]
		nbrs = indices[start:end]
		outvec = (nbrs - (nbrs % N)) / N
		
		if len(invec) == 0 or len(outvec) == 0:
			continue
		
		in_uq, in_cnt = np.unique(invec, return_counts=True)
		out_uq, out_cnt = np.unique(outvec, return_counts=True)
		i = z.copy()
		j = z.copy()
		i[in_uq] = in_cnt
		j[out_uq] = out_cnt
		u = np.outer(i, j)
		cooc_mat += u
		itr += 1
		if itr == progress[ptr]:
			print '{} complete: {:.3f}s.'.format(progress[ptr], time() - t1)
			ptr += 1
			t1 = time()
	return cooc_mat

cpdef compute_cooccurrence(G, revG):
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
	indices = G.csr.indices.astype(_int64)
	indptr = G.csr.indptr.astype(_int)

	revindices = revG.csr.indices.astype(_int64)
	revindptr = revG.csr.indptr.astype(_int)

	cooc_mat = _compute_cooccurrence(G.R, indices, indptr, revindices, revindptr)
	return cooc_mat