cimport numpy as np
from scipy import sparse
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def extents_count(Py_ssize_t[::1] labels,
                  Py_ssize_t[::1] curr_loc, Py_ssize_t[::1] indices):
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t label
    cdef Py_ssize_t size = len(labels)
    for i in range(size):
        label = labels[i]
        indices[curr_loc[label]] = i
        curr_loc[label] += 1
        i += 1
