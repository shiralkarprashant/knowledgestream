# cython: profile=False
import numpy as np

from datastructures.relationalpath import RelationalPath

# c imports
cimport cython
cimport numpy as np

from libcpp.vector cimport vector
from libcpp.stack cimport stack

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float


# ================ PATH ENUMERATION PROCEDURE ================

cpdef get_paths(G, s, p, o, length=3, maxpaths=-1):
	"Returns all paths of length `length` starting at s and ending in o."
	cdef:
		double[:] data
		long[:] indices
		int[:] indptr
		list paths, relpaths, discovered_paths
	# graph vectors
	data = G.csr.data.astype(_float)
	indices = G.csr.indices.astype(_int64)
	indptr = G.csr.indptr.astype(_int)
	paths, relpaths = enumerate_paths(
		data, indices, indptr, s, p, o, length=length, maxpaths=maxpaths
	)
	# convert to Python objects
	discovered_paths = []
	for pth, rpth in zip(paths, relpaths):
		pp = RelationalPath(s, p, o, 0., length, pth, rpth, np.ones(length + 1))
		discovered_paths.append(pp)
	return discovered_paths


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef object enumerate_paths(
		double[:] data, long[:] indices, int[:] indptr, 
		int s, int p, int o, int length=3, int maxpaths=-1
	):
	"Workhorse function for path enumeration."
	cdef:
		# ===== basic types =====
		int i, N, node, nbr, rel, start, end, N_neigh
		stack[vector[int]] path_stack, relpath_stack
		vector[int] curr_path, curr_relpath, tmp
		vector[vector[int]] discovered_paths, discovered_relpaths
		long[:] neighbors
		np.ndarray paths_arr, relpaths_arr
	N = len(indptr) - 1
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
	return discovered_paths, discovered_relpaths

