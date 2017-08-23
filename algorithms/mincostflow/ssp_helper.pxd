cimport cython
cimport numpy as np
cimport numpy as cnp

from libc.stdlib cimport malloc, free

cdef void set_base(np.ndarray arr, void *carr, str tag=*)