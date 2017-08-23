""" This is the definition file for heap.pyx.
It contains the definitions of the heap classes, such that
other cython modules can "cimport heap" and thus use the
C versions of pop(), push(), and value_of(): pop_fast(), push_fast() and 
value_of_fast()
"""

# cython specific imports
import numpy as np
cimport numpy as np

# determine datatypes for heap
ctypedef double VALUE_T
ctypedef int REFERENCE_T


cdef class BinaryHeap:
    cdef readonly int count, levels, min_levels
    cdef VALUE_T *_values
    cdef REFERENCE_T *_references
    cdef int _popped_ref

    cdef void _add_or_remove_level(self, int add_or_remove) nogil
    cdef void _update(self) nogil
    cdef void _update_one(self, int i) nogil
    cdef void _remove(self, int i) nogil

    cdef int push_fast(self, double value, int reference) nogil
    cdef double pop_fast(self) nogil

cdef class FastUpdateBinaryHeap(BinaryHeap):
    cdef readonly int max_reference
    cdef REFERENCE_T *_crossref
    cdef int _invalid_ref
    cdef int _pushed

    cdef double value_of_fast(self, int reference) nogil
    cdef int push_if_lower_fast(self, double value, int reference) nogil
