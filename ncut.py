import numpy
import agglo
import morpho
import scipy.sparse
import scipy.sparse.linalg
import scipy.cluster.vq
 
def ncutW(W, num_eigs=10, kmeans_iters=10, offset = 0.5, **kwargs):
    """Run the normalized cut algorithm
    
    (as implemented in Ng, Jordan, and Weiss, 2002)
    
    Return value: labels, eigenvectors, and eigenvalues
    """
    
    n,m = numpy.shape(W)
    # Add an offset in case some rows are zero
    # We also add the offset below to the diagonal matrix. See (Yu, 2001),
    #    "Understanding Popout through Repulsion" for more information.  This 
    #    helps to stabalize the eigenvector computation.
    W = W + scipy.sparse.spdiags(offset*numpy.ones(n),0,n,n)
    
    # Calculate matrix to take eigenvectors of
    rows, cols = W.nonzero()
    d = W.sum(1).transpose()
    Dinv2 = scipy.sparse.spdiags(1./(numpy.sqrt(d) + offset*numpy.ones(n)), 0, n, n)
    P = Dinv2*W*Dinv2;
    
    # Get the eigenvectors and sort by eigenvalue
    eval,U = scipy.sparse.linalg.eigs(P, num_eigs, which='LR')
    eval = numpy.real(eval) # it should be real anyway
    U = numpy.real(U)
    ind = numpy.argsort(eval)[::-1]
    eval = eval[ind]
    U = U[:,ind]
    
    # Normalize
    for i in range(n):
        u = U[i,:]
        U[i,:] = U[i,:]/scipy.linalg.norm(u)
    
    # Cluster them into labels, running k-means multiple times
    labels_list = []
    distortion_list = []
    for iter in range(kmeans_iters):
        # Cluster
        centroid, labels = scipy.cluster.vq.kmeans2(U, num_eigs, minit='points')
        # Calculate distortion
        distortion = 0
        for j in range(num_eigs):
            numvals = (sum(labels==j))
            if numvals==0: continue
            distortion = distortion + \
                1.0/float(numvals)* \
                sum([scipy.linalg.norm(v - centroid[j])**2 \
                    for (i,v) in enumerate(U) if labels[i]==j])
        # Save values
        labels_list.append(labels)
        distortion_list.append(distortion)
    # Use lowest distortion
    labels = labels_list[numpy.argmin(distortion_list)]
    
    return labels, U, eval
