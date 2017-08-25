cimport cython

cpdef get_paths(G, s, p, o, length=*, maxpaths=*)

cdef object enumerate_paths(
		double[:] data, long[:] indices, int[:] indptr, 
		int s, int p, int o, int length=*, int maxpaths=*
	)

