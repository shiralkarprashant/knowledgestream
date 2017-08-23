cimport cython
from libc.stdlib cimport malloc, abort
from libc.string cimport memset

ctypedef struct Closure:
	double conjf(double, double) nogil
	double disjf(double, double) nogil

cdef inline double _dombit1(double a, double b) nogil:
	""" Dombi T-conorm with lambda = 1.

	Returns a double precision float between 0 and 1.

	>>> _dombit1(0, 0)
	0.0
	>>> _dombit1(1, 1)
	1.0
	>>> dombit1(0.5, 0.5)
	0.3333333333333333

	"""
	if a == b == 0:
		return 0.0
	else:
		return (a * b) / (a + b - a * b)


cpdef relational_closure(G, s, p, o, kind=*, linkpred=*)

cdef cclosuress(
		double[:] data, long[:] indices, int[:] indptr, 
		int source, int predicate, int target, Closure closure
	)