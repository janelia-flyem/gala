cimport numpy as np
include "dtypes_cy.pxi"  # definition of label_dtype
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def extents_count(label_dtype[::1] labels, Py_ssize_t[::1] curr_loc,
                  Py_ssize_t[::1] indices, label_dtype[::1] out):
    cdef Py_ssize_t i = 0
    cdef label_dtype label
    cdef Py_ssize_t size = len(labels)
    for i in range(size):
        label = labels[i]
        out[curr_loc[label]] = indices[i]
        curr_loc[label] += 1
