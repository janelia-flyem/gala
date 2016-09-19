import numpy as np
from scipy import sparse
from scipy.sparse.linalg import norm, eigs
from scipy.cluster import vq


def ncutW(W, num_eigs=10, kmeans_iters=10, offset = 0.5):
    """Run the normalized cut algorithm on the affinity matrix, W.

    (as implemented in Ng, Jordan, and Weiss, 2002)

    Parameters
    ----------
    W : scipy sparse matrix
        Square matrix with high values for edges to be preserved, and low
        values for edges to be cut.
    num_eigs : int, optional
        Number of eigenvectors of the affinity matrix to use for clustering.
    kmeans_iters : int, optional
        Number of iterations of the k-means algorithm to run when clustering
        eigenvectors.
    offset : float, optional
        Diagonal offset used to stabilise the eigenvector computation.

    Returns
    -------
    labels : array of int
        `labels[i]` is an integer value mapping node/row `i` to the cluster
        ID `labels[i]`.
    eigenvectors : list of array of float
        The computed eigenvectors of `W + offset * I`, where `I` is the
        identity matrix of same size as `W`.
    eigenvalues : array of float
        The corresponding eigenvalues.
    """
    
    n, m = W.shape
    # Add an offset in case some rows are zero
    # We also add the offset below to the diagonal matrix. See (Yu, 2001),
    # "Understanding Popout through Repulsion" for more information.  This
    # helps to stabilize the eigenvector computation.
    W = W + sparse.diags(np.full(n, offset))
    
    d = np.ravel(W.sum(axis=1))
    Dinv2 = sparse.diags(1 / (np.sqrt(d) + offset*np.ones(n)))
    P = Dinv2 @ W @ Dinv2
    
    # Get the eigenvectors and sort by eigenvalue
    eigvals, U = eigs(P, num_eigs, which='LR')
    eigvals = np.real(eigvals)  # it should be real anyway
    U = np.real(U)
    ind = np.argsort(eigvals)[::-1]
    eigvals = eigvals[ind]
    U = U[:, ind]
    
    # Normalize
    for i in range(n):
        U[i, :] /= norm(U[i, :])
    
    # Cluster them into labels, running k-means multiple times
    labels_list = []
    distortion_list = []
    for _iternum in range(kmeans_iters):
        # Cluster
        centroid, labels = vq.kmeans2(U, num_eigs, minit='points')
        # Calculate distortion
        distortion = 0
        for j in range(num_eigs):
            numvals = np.sum(labels == j)
            if numvals == 0:
                continue
            distortion += np.mean([norm(v - centroid[j])**2 for (i, v) in
                                   enumerate(U) if labels[i] == j])
        # Save values
        labels_list.append(labels)
        distortion_list.append(distortion)
    # Use lowest distortion
    labels = labels_list[np.argmin(distortion_list)]
    
    return labels, U, eigvals
