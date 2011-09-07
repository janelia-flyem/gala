import numpy
from scipy.sparse import coo_matrix
from scipy.ndimage.measurements import label
from scipy.misc import comb as nchoosek

def pixel_wise_boundary_precision_recall(aseg, gt):
    gt = (1-gt.astype(numpy.bool)).astype(numpy.uint8)
    aseg = (1-aseg.astype(numpy.bool)).astype(numpy.uint8)
    tp = float((gt * aseg).sum())
    fp = (aseg * (1-gt)).sum()
    fn = (gt * (1-aseg)).sum()
    return tp/(tp+fp), tp/(tp+fn)

def edit_distance(aseg, gt, ws=None):
    if ws is None:
        return edit_distance_to_bps(aseg, gt)
    import agglo
    return edit_distance_to_bps(aseg, agglo.best_possible_segmentation(ws, gt))

def edit_distance_to_bps(aseg, bps):
    aseg = relabel_from_one(aseg)
    bps = relabel_from_one(bps)
    r = contingency_table(aseg, bps).astype(numpy.bool)
    if (bps==0).any(): r[:,0] = 0
    if (aseg==0).any(): r[0,:] = 0
    false_splits = (r.sum(axis=0)-1)[1:].sum()
    false_merges = (r.sum(axis=1)-1)[1:].sum()
    return (false_merges, false_splits)

def relabel_from_one(a):
    labels = numpy.unique(a)
    labels = labels[labels!=0]
    if labels.max() == len(labels):
        return a
    b = a.copy()
    for i, label in enumerate(labels):
        b[a==label] = i+1
    return b

def contingency_table(seg, gt):
    """Return the contingency table for all regions in matched segmentations."""
    gtr = gt.ravel()
    segr = seg.ravel() 
    ij = numpy.zeros((2,len(gtr)))
    ij[0,:] = segr
    ij[1,:] = gtr
    cont = coo_matrix((numpy.ones((len(gtr))), ij)).toarray()
    return cont

def xlogx(x, out=None):
    """Compute x * log_2(x) with 0 log(0) defined to be 0."""
    nz = x.nonzero()
    if out is None:
        y = x.copy()
    else:
        y = out
    y[nz] *= numpy.log2(y[nz])
    return y

def voi(X, Y, cont=None, weights=numpy.ones(2), ignore_seg_labels=[], ignore_gt_labels=[]):
    """Return the variation of information metric."""
    return numpy.dot(weights, split_voi(X,Y,cont, ignore_seg_labels, ignore_gt_labels))

def voi_tables(X, Y, cont=None, ignore_seg_labels=[], ignore_gt_labels=[]):
    """Return probability tables used for calculating voi."""
    if cont is None:
        cont = contingency_table(X, Y)

    # Zero out ignored labels
    cont[:, ignore_gt_labels] = 0
    cont[ignore_seg_labels,:] = 0
    n = cont.sum()

    # Calculate probabilities
    pxy = cont/float(n)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    # Remove zero rows/cols
    nzx = px.nonzero()[0]
    nzy = py.nonzero()[0]
    nzpx = px[nzx]
    nzpy = py[nzy]
    nzpxy = pxy[nzx,:][:,nzy]

    # Calculate log conditional probabilities and entropies
    ax = numpy.newaxis
    lpygx = numpy.zeros(numpy.shape(px))
    lpygx[nzx] = xlogx(nzpxy / nzpx[:,ax]).sum(axis=1) # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)
    
    lpxgy = numpy.zeros(numpy.shape(py))
    lpxgy[nzy] = xlogx(nzpxy / nzpy[ax,:]).sum(axis=0)
    hxgy = -(py*lpxgy)

    return pxy,px,py,hxgy,hygx, lpygx, lpxgy

def split_voi(X,Y,cont=None, ignore_seg_labels=[], ignore_gt_labels=[]):
    """Return the symmetric conditional entropies associated with the VOI.
    
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.
    """
    pxy,px,py,hxgy,hygx,lpygx,lpxgy = voi_tables(X,Y,cont,ignore_seg_labels, ignore_gt_labels)
    # false merges, false splits
    return numpy.array([hygx.sum(), hxgy.sum()])

def rand_values(cont_table):
    """Calculate values for rand indices."""
    n = cont_table.sum()
    sum1 = (cont_table*cont_table).sum()
    sum2 = (cont_table.sum(axis=1)**2).sum()
    sum3 = (cont_table.sum(axis=0)**2).sum()
    a = (sum1 - n)/2.0;
    b = (sum2 - sum1)/2
    c = (sum3 - sum1)/2
    d = (sum1 + n**2 - sum2 - sum3)/2
    return a, b, c, d

def rand_index(seg, gt, cont=None):
    """Return the unadjusted Rand index."""
    if cont is None:
        cont = contingency_table(seg, gt)
    a, b, c, d = rand_values(cont)
    return (a+d)/(a+b+c+d)
    
def adj_rand_index(seg, gt, cont=None):
    """Return the adjusted Rand index."""
    if cont is None:
        cont = contingency_table(seg, gt)
    a, b, c, d = rand_values(cont)
    nk = a+b+c+d
    return (nk*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(
        nk**2 - ((a+b)*(a+c) + (c+d)*(b+d)))

def fm_index(seg, gt, cont=None):
    """ Return the Fowlkes-Mallows index. """
    if cont is None:
        cont = contingency_table(seg, gt)
    a, b, c, d = rand_values(cont)
    return a/(numpy.sqrt((a+b)*(a+c)))

