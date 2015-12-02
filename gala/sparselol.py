import numpy as np
from numpy.lib import stride_tricks
from scipy import sparse
from .sparselol_cy import extents_count
from .dtypes import label_dtype

def extents(labels):
    """Compute the extents of every integer value in ``arr``.

    Parameters
    ----------
    labels : array of ints
        The array of values to be mapped.

    Returns
    -------
    locs : sparse.csr_matrix
        A sparse matrix in which the nonzero elements of row i are the
        indices of value i in ``arr``.
    """
    labels = labels.astype(label_dtype).ravel()
    counts = np.bincount(labels)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    indices = np.empty_like(labels)
    extents_count(labels.ravel(), indptr.copy(), out=indices)
    one = np.ones((1,), dtype=int)
    data = stride_tricks.as_strided(one, shape=indices.shape, strides=(0,))
    locs = sparse.csr_matrix((data, indices, indptr), dtype=int)
    return locs