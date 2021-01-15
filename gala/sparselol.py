import numpy as np
from numpy.lib import stride_tricks
from scipy import sparse
from .sparselol_cy import extents_count
from .dtypes import label_dtype

class SparseLOL:
    def __init__(self, csr):
        self.indptr = csr.indptr
        self.indices = csr.indices
        self.data = csr.data

    def __getitem__(self, item):
        if np.isscalar(item):  # get the column indices for the given row
            start, stop = self.indptr[item : item+2]
            return self.indices[start:stop]
        else:
            raise ValueError('SparseLOL can only be indexed by an integer.')

def extents(labels, input_indices=None):
    """Compute the extents of every integer value in ``arr``.

    Parameters
    ----------
    labels : array of int
        The array of values to be mapped.
    input_indices : array of int
        The indices corresponding to the label values passed. If `None`,
        we assume ``range(labels.size)``.

    Returns
    -------
    locs : sparse.csr_matrix
        A sparse matrix in which the nonzero elements of row i are the
        indices of value i in ``arr``.
    """
    labels = labels.astype(label_dtype).ravel()
    if input_indices is None:
        input_indices = np.arange(labels.size, dtype=np.intp)
    counts = np.bincount(labels)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)
    indices = np.empty_like(labels)
    extents_count(labels.ravel(), indptr.copy(), input_indices, out=indices)
    one = np.ones((1,), dtype=int)
    data = stride_tricks.as_strided(one, shape=indices.shape, strides=(0,))
    locs = sparse.csr_matrix((data, indices, indptr), dtype=int)
    return locs
