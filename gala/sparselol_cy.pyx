cimport numpy as np
from scipy import sparse

def extents_count(Py_ssize_t[::1] labels,
                  Py_ssize_t[::1] curr_loc, Py_ssize_t[::1] indices):
    cdef Py_ssize_t i
    for i, label in enumerate(labels):
        indices[curr_loc[label]] = i
        curr_loc[label] += 1
