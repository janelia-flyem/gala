import numpy as np
from scipy import sparse
from .sparselol_cy import extents_count

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
    labels = labels.ravel()
    counts = np.bincount(labels)
    indptr = np.concatenate([[0], np.cumsum(counts)])
    indices = np.empty(labels.size, int)
    extents_count(labels.ravel(), indptr.copy(), indices)
    locs = sparse.csr_matrix((indices, indices, indptr), dtype=int)
    return locs