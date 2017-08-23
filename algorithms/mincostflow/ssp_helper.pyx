"""
Cython implementation of shortest path finding procedure required for 
Successive Shortest Path (SSP) Algorithm.

References: 
* Ch 9: Network flows: Theory, Algorithms and Applications by R. Ahuja, T. Magnanti and J. Orlin.
"""
import os
import sys
from time import time

# cython: profile=False
import numpy as np

# c imports
cimport cython
cimport numpy as np

from numpy.math cimport INFINITY
from datastructures.heap cimport FastUpdateBinaryHeap

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float

# max int32 value
_maxint = np.iinfo(_int).max
_maxfloat = INFINITY

## =========== Python interfacing calls =========== 

cpdef object compute_shortest_path_distances(G, s, t, reduced_cost, delta=-1):
	"Python: BFS search to compute distance of each node from target t."
	data = G.csr.data.astype(_float, copy=False)
	indices = G.csr.indices.astype(_int64, copy=False)
	indptr = G.csr.indptr.astype(_int, copy=False)
	return _compute_shortest_path_distances(reduced_cost, data, indices, indptr, s, t, G.N, delta=delta)

## =========== Cython =========== 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef object _compute_shortest_path_distances(
		int[:] reduced_cost, double[:] data, long[:] indices, int[:] indptr, 
		int src, int tar, int N, double delta=-1.
	):
	cdef:
		FastUpdateBinaryHeap Q
		int node, N_neigh, i, j, pathlen, start, end, cost, size_int, size_double
		double residual_cap, bottleneck, dist
		long neighbor
		# ===== pointers =====
		int *_preds
		int *_rels
		int *_found
		int *_path
		int *_relpath
		int *_dist
		double *_caps
		# ===== memoryviews =====
		long[:] neighbors
		int[:] predecessors, relations, found, path, relpath, costs, distances
		double[:] capacities, residual_caps
		# ===== ndarrays =====
		np.ndarray path_arr, relpath_arr, found_arr, dist_arr
	Q = FastUpdateBinaryHeap(N, N)
	size_int = N * sizeof(int)
	size_double = N * sizeof(double)
	# distances
	_dist = <int*>malloc(size_int)
	if _dist == NULL:
		raise MemoryError()
	distances = <int[:N]>_dist
	distances[...] = _maxint # max distance
	# predecessors
	_preds = <int*>malloc(size_int)
	if _preds == NULL:
		raise MemoryError()
	predecessors = <int[:N]>_preds 
	predecessors[...] = -1
	# relations
	_rels = <int*>malloc(size_int)
	if _rels == NULL:
		raise MemoryError()
	relations = <int[:N]>_rels
	relations[...] = -1
	# found/visited/explored part of the graph
	_found = <int*>malloc(size_int)
	if _found == NULL:
		raise MemoryError()
	found = <int[:N]>_found
	found[...] = 0
	# bottlenecks for each node
	_caps = <double*>malloc(size_double)
	if _caps == NULL:
		raise MemoryError()
	capacities = <double[:N]>_caps
	capacities[...] = _maxfloat # random 

	# populate the priority queue / heap
	distances[src] = 0
	Q.push_fast(distances[src], src)

	# search path
	while Q.count:
		dist = Q.pop_fast()
		node = Q._popped_ref
		if found[node] == 0:
			found[node] = 1
		if node == tar:
			break
		start = indptr[node]
		end = indptr[node + 1]
		neighbors = indices[start:end] # nbrs in wide-CSR
		residual_caps = data[start:end]
		costs = reduced_cost[start:end]
		N_neigh = end - start
		for i in range(N_neigh):
			residual_cap = residual_caps[i] # residual capacity
			if residual_cap <= 0:
				continue
			if delta != -1 and residual_cap < delta:
				continue
			neighbor = neighbors[i] % N
			cost = costs[i] 
			if found[neighbor] == 0 and distances[node] + cost < distances[neighbor]:
				predecessors[neighbor] = node
				relations[neighbor] = (neighbors[i] - neighbor) / N # relation
				distances[neighbor] = distances[node] + cost
				capacities[neighbor] = residual_cap
				Q.push_if_lower_fast(distances[neighbor], neighbor) # heapify

	# make path
	if predecessors[tar] == -1:
		path_arr = None
		relpath_arr = None
		bottleneck = 0.
		found_arr = None
		dist_arr = None
		# free up memory
		free(<void*> _dist)
		free(<void*> _found)
	else:
		pathlen = 0
		i = tar
		while i != -1:
			pathlen += 1
			i = predecessors[i]
		size_int = pathlen * sizeof(int)
		# path
		_path = <int*>malloc(size_int)
		if _path == NULL:
			raise MemoryError()
		path = <int[:pathlen]>_path
		path[...] = -1
		# relational path
		_relpath = <int*>malloc(size_int)
		if _relpath == NULL:
			raise MemoryError()
		relpath = <int[:pathlen]>_relpath
		relpath[...] = -1
		i = tar
		bottleneck = -1
		while i != -1:
			path[pathlen-1] = i
			relpath[pathlen-1] = relations[i]
			if bottleneck == -1:
				bottleneck = capacities[i]
			elif capacities[i] < bottleneck:
				bottleneck = capacities[i]
			i = predecessors[i]
			pathlen -= 1
		path_arr = np.asarray(path)
		relpath_arr = np.asarray(relpath)
		found_arr = np.asarray(found)
		dist_arr = np.asarray(distances)
		# numpy owner frees up memory
		set_base(path_arr, _path, tag='path') 
		set_base(relpath_arr, _relpath, tag='relpath')
		set_base(found_arr, _found, tag='found')
		set_base(dist_arr, _dist, tag='dist')
	# free up memory
	free(<void*> _preds)
	free(<void*> _rels)
	free(<void*> _caps)
	return path_arr, relpath_arr, bottleneck, found_arr, dist_arr

## =========== Code to free up memory to avoid memory leaks =========== 
cdef class _finalizer:
	cdef void *_data
	cdef str tag
	def __cinit__(self, tag):
		self.tag = tag
	def __dealloc__(self):
		# print "_finalizer.__dealloc__ for {}".format(self.tag)
		if self._data is not NULL:
			free(self._data)

cdef void set_base(np.ndarray arr, void *carr, str tag=''):
	cdef _finalizer f = _finalizer(tag)
	f._data = <void*>carr
	np.set_array_base(arr, f)

