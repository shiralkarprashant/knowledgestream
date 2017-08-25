# cython: profile=False
import sys
import numpy as np

from datastructures.relationalpath import RelationalPath
from time import time

# c imports
cimport cython
cimport numpy as np

# http://hplgit.github.io/teamods/MC_cython/main_MC_cython.html
from libc.stdlib cimport rand, RAND_MAX
from libcpp.vector cimport vector
from libcpp.stack cimport stack

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float


# ================ PATH ENUMERATION PROCEDURE ================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cpdef extract_paths(G, triples, int length=3, int maxpaths=-1):
	"""
	Extracts a dictionary of (feature, frequency) pairs,
	based on a set of training instances, that serves as 
	features in the model.
	
	Parameters:
	-----------
	G: rgraph
		Knowledge graph.
	triples: sequence
		A list of triples (sid, pid, oid).
	length: int
		Maximum length of any path.
	maxpaths: int
		Maximum number of paths of length `length`.

	Returns:
	--------
	features: dict
		A set of (feature, frequency) pairs.
	"""
	cdef:
		int N, n, m, k, pthidx, idx
		long[:] indices
		int[:] indptr, sids, oids
		vector[vector[int]] relpaths
		dict features
		tuple path
	indices = G.csr.indices.astype(_int64)
	indptr = G.csr.indptr.astype(_int)
	N = len(indptr) - 1
	sids = np.asarray([t['sid'] for t in triples], dtype=_int)
	oids = np.asarray([t['oid'] for t in triples], dtype=_int)
	n = len(sids)
	length = length + 1
	features = dict()
	for idx in xrange(n):
		# extract paths for a triple
		sys.stdout.flush()
		# print '\nT{}'.format(idx+1),
		for m in xrange(length):
			if m in [0, 1]: # paths of length 0 and 1 mean nothing
				continue
			relpaths = enumerate_paths(
				indices, indptr, N, sids[idx], oids[idx], length=m, maxpaths=maxpaths
			) # cythonized
			k = relpaths.size()
			for pthidx in xrange(k):
				path = tuple(relpaths[pthidx])
				if path not in features:
					features[path] = 1
				else:
					features[path] += 1
	return features

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef vector[vector[int]] enumerate_paths(
		long[:] indices, int[:] indptr, int N,
		int s, int o, int length=3, int maxpaths=-1
	):
	"Workhorse function for path enumeration."
	cdef:
		# ===== basic types =====
		int i, node, nbr, rel, start, end, N_neigh
		stack[vector[int]] path_stack, relpath_stack
		vector[int] curr_path, curr_relpath, tmp
		vector[vector[int]] discovered_paths, discovered_relpaths
		long[:] neighbors
		np.ndarray paths_arr, relpaths_arr
	tmp.push_back(s)
	path_stack.push(tmp)
	tmp.clear()
	tmp.push_back(-1)
	relpath_stack.push(tmp)
	while path_stack.size() > 0:
		curr_path = path_stack.top()
		path_stack.pop()
		curr_relpath = relpath_stack.top()
		relpath_stack.pop()
		node = curr_path.back()
		if curr_path.size() == length + 1:
			if node == o:
				discovered_paths.push_back(curr_path)
				discovered_relpaths.push_back(curr_relpath)
				if maxpaths != -1 and discovered_paths.size() >= maxpaths:
					# print '[L:{}, maxpaths:{}]'.format(length, maxpaths),
					break
			continue
		start = indptr[node]
		end = indptr[node + 1]
		neighbors = indices[start:end] # nbrs in wide-CSR
		N_neigh = end - start
		for i in xrange(N_neigh):
			nbr = neighbors[i] % N # predecessor vec
			rel = (neighbors[i] - nbr) / N # relation vec
			curr_path.push_back(nbr)
			path_stack.push(curr_path)
			curr_path.pop_back()
			curr_relpath.push_back(rel)
			relpath_stack.push(curr_relpath)
			curr_relpath.pop_back()
	return discovered_relpaths

# ================ FEATURE MATRIX CONSTRUCTION BY RANDOM WALK ================

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cpdef construct_feature_matrix(G, list features, list triples, int maxwalks=1000):
	"""
	Constructs a feature matrix where the paths extracted in previous
	step (input here) act as features. A value in the matrix corresponding 
	to a training instance (s, p, o) is the probability of arriving at 
	node o by a random walk starting at node s and a specific path (feature).

	There are exact and approximate approaches to compute this. Exact approach 
	is costly however: complexity is proportional to the average per-edge-label 
	(or per-relation) out-degree to the power of the path length per source node 
	per path. Hence, we compute this probability using the approximate method of 
	rejection sampling. It works as follows: a number of random walks are performed
	starting at node 's', each attempting to follow the path (feature). If a 
	node is reached where it is no longer possible to follow the path, the 
	random walk is restarted. The fraction of walks that land in node 'o' 
	represents the probability.

	Parameters:
	-----------
	G: rgraph
		Knowledge Graph.
	features: list
		List of features, per index in feature matrix.
	triples: list
		A list of training instance triples, each represented by a dict, and all 
		belonging to the same relation/predicate p.
	maxwalks: int
		Number of random walks to perform. The higher, the better the resulting 
		approximation. However, this also means increased computational time.
	
	Returns:
	--------
	mtx: sparse matrix
		A sparse matrix containing paths as features, node pairs as rows, and a 
		non-zero value as an entry corresponding to the probability computed 
		as above.
	"""
	cdef:
		int n, m, i, N
		double deno
		long[:] indices
		int[:] indptr, sids, oids
		np.ndarray vec, mat
		vector[vector[int]] ff
		vector[int] cvec
	deno = float(maxwalks)
	indices = G.csr.indices.astype(_int64)
	indptr = G.csr.indptr.astype(_int)
	n = len(triples)
	m = len(features)
	N = len(indptr) - 1
	sids = np.asarray([t['sid'] for t in triples], dtype=_int)
	oids = np.asarray([t['oid'] for t in triples], dtype=_int)
	for i in xrange(m):
		ff.push_back(features[i])
	t1 = time()
	with nogil:
		cvec = _construct_feature_matrix(
			indices, indptr, sids, oids, ff, maxwalks, n, m, N
		)
	print 'Time taken at C level: {:.3f}s'.format(time() - t1)
	vec = np.asarray(cvec)
	vec = vec / deno
	mat = vec.reshape((n, m))
	return mat

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef vector[int] _construct_feature_matrix(
		long[:] indices, int[:] indptr, int[:] sids, int[:] oids, 
		vector[vector[int]] features, int maxwalks,
		int n, int m, int N
	) nogil:
	cdef:
		int pathlen, cnt, rel, n_nbrs, node
		int idx, pthidx, walk, r, start, end, k # indices
		vector[int] path, tmp, vec
		long[:] row
	for idx in range(n):
		for pthidx in xrange(m):
			cnt = 0
			path = features[pthidx]
			pathlen = path.size() - 1
			for walk in xrange(maxwalks):
				node = sids[idx]
				for r in xrange(pathlen):
					rel = path[r+1] # current relation on the path
					# find neighbors under this relation
					tmp.clear()
					start = indptr[node]
					end = indptr[node + 1]
					row = indices[start:end]
					for k in xrange(end-start):
						if row[k] >= rel * N and row[k] < (rel + 1) * N:
							tmp.push_back(row[k] % N)
					n_nbrs = tmp.size()
					if n_nbrs == 0:
						break # restart random walk
					else:
						k = int(rand() / float(RAND_MAX / n_nbrs + 1))
						node = tmp[k] # pick 1 nbr uniformly at random
				if node == oids[idx]:
					cnt += 1
			vec.push_back(cnt)
	return vec