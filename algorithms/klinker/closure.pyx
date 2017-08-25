# cython: profile=False
import numpy as np

from datastructures.relationalpath import RelationalPath

# c imports
cimport cython
cimport numpy as np
from libc.math cimport fmin, fmax
from libc.stdlib cimport malloc, abort, free, calloc
from klinker.closure.datastructures.heap cimport FastUpdateBinaryHeap

# data types for int and float
_short = np.int16
_int = np.int32
_int64 = np.int64
_float = np.float


# ================ CLOSURE PROCEDURES ================

cpdef closure(G, s, p, o, kind='metric', linkpred=True):
	"""
	Computes a relational closure using Dijkstra's algorithm.

	Parameters:
	-----------
	G: rgraph
		A knowledge graph with weights already set. E.g. An entry (i, j, k)
		may represent the product of relational similarity and specificity
		due to the incident node.
	s, p, o: int
		Indices corresponding to the subject, predicate and object of the triple.
	linkpred: bool
		Whether to perform link prediction.

	Returns:
	--------
	rp: RelationalPath
		A relational path found through closure.
	"""
	cdef:
		Closure closure
		double[:] data
		long[:] indices
		int[:] indptr
	# set the closure object 
	if kind == 'metric':
		closure.disjf = fmax
		closure.conjf = _dombit1
	elif kind == 'ultrametric':
		closure.disjf = fmax
		closure.conjf = fmin
	else:
		raise ValueError('unknown metric kind: {}'.format(kind))

	# link prediction
	if linkpred and G[s, o, p] != 0:
		G[s, o, p] = 0.

	# graph vectors
	data = G.csr.data.astype(_float)
	indices = G.csr.indices.astype(_int64)
	indptr = G.csr.indptr.astype(_int)
	
	# closure
	caps, preds, rels = cclosuress(data, indices, indptr, s, p, o, closure)

	# construct path from the returned vectors
	path = []
	rel_path = []
	shortcaps = []
	i = o
	while i != -1:
		path.append(i)
		rel_path.append(rels[i])
		shortcaps.append(caps[i])
		i = preds[i]
	path, rel_path = path[::-1], rel_path[::-1]
	pathlen = len(path) - 1
	rp = RelationalPath(s, p, o, caps[o], pathlen, path, rel_path, shortcaps)
	return rp


# Shortest path algorithm to include edge similarities in computation
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef cclosuress(
		double[:] data, long[:] indices, int[:] indptr, 
		int source, int predicate, int target, Closure closure
	):
	"""
	Single source closure via Dijkstra path finding.

	Parameters:
	-----------
	data, indices, indptr: array-like
		Arrays representing the weighted knowledge graph.
	source, predicate, target: int
		Source (subject), predicate and target (object) of closure computation. 
		Ensure index is within limits.

	Returns:
	--------
	caps, preds, rels: ndarray
		Arrays representing values for capacities, predecessor information and 
		relations under which they are connected. Note that the information is 
		accurate only for target and all nodes that are within the "radius".

	Note: cap = capacity of an edge
		  pred = predecessor of a node
		  rel = relation indicated by an edge
		  nbr = neighbor
	"""
	cdef:
		FastUpdateBinaryHeap Q  # min-heap
		# ===== basic types =====
		int N, node, N_neigh, i, start, end
		long neighbor
		double cap, new_cap, relational_cap
		int neigh_curr_rel 
		int neigh_cand_rel 
		double neigh_cand_cap, neigh_curr_cap # neighbor's current and candidate capacities (along current and candidate relations)
		# ===== pointers =====
		long *_preds
		int *_rels
		int *_found
		double *_caps
		# ===== memoryviews =====
		long[:] neighbors, predecessors
		int[:] relations, found
		double[:] capacities, nbr_caps
		# ===== ndarrays =====
		np.ndarray caps_arr, preds_arr, rels_arr
	# =======================================
	# Initialize arrays to hold results
	# =======================================
	N = len(indptr) - 1
	Q = FastUpdateBinaryHeap(N, N)
	# capacities
	_caps = <double*>malloc(N * sizeof(double))
	capacities = <double[:N]>_caps
	capacities[...] = 0.
	# predecessors
	_preds = <long*>malloc(N * sizeof(long))
	predecessors = <long[:N]>_preds 
	predecessors[...] = -1
	# relations
	_rels = <int*>malloc(N * sizeof(int))
	relations = <int[:N]>_rels
	relations[...] = -1
	# found/visited/explored part of the graph
	_found = <int*>malloc(N * sizeof(int))
	found = <int[:N]>malloc(N * sizeof(int))
	found[...] = 0

	# populate the queue
	for node in range(N):
		if node == source:
			cap = 1.
			# value, node, predecessor, relation index
		else:
			cap = 0.0
		capacities[node] = cap
		Q.push_fast(-cap, node)

	# compute path
	while Q.count:
		cap = - Q.pop_fast() # +ve
		node = Q._popped_ref
		if found[node] == 0:
			found[node] = 1
		if node == target:
			break # break when target has been extracted from the heap

		# continue search to node's neighbors
		start = indptr[node]
		end = indptr[node + 1]
		neighbors = indices[start:end] # nbrs in wide-CSR
		nbr_caps = data[start:end]
		N_neigh = end - start
		for i in xrange(N_neigh):
			neighbor = neighbors[i] % N # predecessor vec
			neigh_cand_rel = (neighbors[i] - neighbor) / N # relation vec
			relational_cap = nbr_caps[i] # weight of current edge (node, neighbor, neigh_cand_rel)
			if found[neighbor] == 0:
				neigh_curr_cap = capacities[neighbor] # current cap/dist to source
				neigh_curr_rel = relations[neighbor] # relation through which neighbor is connected to its predecessor
				neigh_cand_cap = closure.conjf(cap, relational_cap) # candidate capacity
				new_cap = closure.disjf(neigh_cand_cap, neigh_curr_cap)
				if new_cap > neigh_curr_cap:
					capacities[neighbor] = new_cap
					predecessors[neighbor] = node
					relations[neighbor] = neigh_cand_rel # candidate relation
					Q.push_if_lower_fast(-new_cap, neighbor) # heapify
	caps_arr = np.asarray(capacities)
	preds_arr = np.asarray(predecessors)
	rels_arr = np.asarray(relations)
	set_base(caps_arr, _caps)
	set_base(preds_arr, _preds)
	set_base(rels_arr, _rels)
	return caps_arr, preds_arr, rels_arr


## =========== Code to free up memory to avoid memory leaks =========== 
cdef class _finalizer:
	cdef void *_data
	def __dealloc__(self):
		if self._data is not NULL:
			free(self._data)

cdef void set_base(np.ndarray arr, void *carr):
	cdef _finalizer f = _finalizer()
	f._data = <void*>carr
	np.set_array_base(arr, f)

