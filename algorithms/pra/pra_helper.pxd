cimport cython

from libcpp.vector cimport vector

cpdef extract_paths(G, triples, int length=*, int maxpaths=*)

cdef vector[vector[int]] enumerate_paths(
		long[:] indices, int[:] indptr, int N, 
		int s, int o, int length=*, int maxpaths=*
	)

cpdef construct_feature_matrix(G, list features, list triples, int maxwalks=*)

cdef vector[int] _construct_feature_matrix(
		long[:] indices, int[:] indptr, int[:] sids, int[:] oids, 
		vector[vector[int]] features, int maxwalks,
		int n, int m, int N
	) nogil