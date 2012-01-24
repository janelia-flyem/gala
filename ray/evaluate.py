import numpy
import multiprocessing
from scipy.sparse import coo_matrix
from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist, cdist, squareform
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
    aseg = relabel_from_one(aseg)[0]
    bps = relabel_from_one(bps)[0]
    r = contingency_table(aseg, bps).astype(numpy.bool)
    if (bps==0).any(): r[:,0] = 0
    if (aseg==0).any(): r[0,:] = 0
    false_splits = (r.sum(axis=0)-1)[1:].sum()
    false_merges = (r.sum(axis=1)-1)[1:].sum()
    return (false_merges, false_splits)

def relabel_from_one(a):
    labels = numpy.unique(a)
    labels0 = labels[labels!=0]
    m = labels.max()
    if m == len(labels0):
        return a
    forward_map = numpy.zeros(m+1, int)
    forward_map[labels0] = numpy.arange(1, len(labels0)+1)
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map

def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations."""
    gtr = gt.ravel()
    segr = seg.ravel() 
    ij = numpy.zeros((2,len(gtr)))
    ij[0,:] = segr
    ij[1,:] = gtr
    cont = coo_matrix((numpy.ones((len(gtr))), ij)).toarray()
    cont[:, ignore_gt] = 0
    cont[ignore_seg,:] = 0
    if norm:
        cont /= float(cont.sum())
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

def vi(X, Y, cont=None, weights=numpy.ones(2), ignore_seg=[], ignore_gt=[]):
    """Return the variation of information metric."""
    return numpy.dot(weights, split_vi(X,Y,cont, ignore_seg, ignore_gt))

def simple_vi(X, Y):
    return vi(X, Y, None, numpy.ones(2), [0], [0])

def vi_pairwise_matrix(segs):
    """Compute the pairwise VI distances within a set of segmentations.
    
    0-labeled pixels are ignored.
    """
    return squareform(pdist(numpy.array([s.ravel() for s in segs]), simple_vi))

def split_vi_threshold(tup):
    """Compute VI with tuple input (to support multiprocessing).
    Tuple elements:
        - the UCM for the candidate segmentation,
        - the gold standard,
        - list of ignored labels in the segmentation,
        - list of ignored labels in the gold standard,
        - threshold to use for the UCM.
    Value:
        - array of length 2 containing the undersegmentation and 
        oversegmentation parts of the VI.
    """
    ucm, gt, ignore_seg, ignore_gt, t = tup
    return split_vi(label(ucm<t)[0], gt, None, ignore_seg, ignore_gt)

def vi_by_threshold(ucm, gt, ignore_seg=[], ignore_gt=[], npoints=None,
                                                            nprocessors=None):
    ts = numpy.unique(ucm)[1:]
    if npoints is None:
        npoints = len(ts)
    if len(ts) > 2*npoints:
        ts = ts[numpy.arange(1, len(ts), len(ts)/npoints)]
    if nprocessors == 1: # this should avoid pickling overhead
        result = [split_vi_threshold((ucm, gt, ignore_seg, ignore_gt, t))
                for t in ts]
    else:
        p = multiprocessing.Pool(nprocessors)
        result = p.map(split_vi_threshold, 
            ((ucm, gt, ignore_seg, ignore_gt, t) for t in ts))
    return numpy.concatenate(
                            (ts[numpy.newaxis, :], numpy.array(result).T), 
                            axis=0)

def rand_by_threshold(ucm, gt, npoints=None):
    ts = numpy.unique(ucm)[1:]
    if npoints is None:
        npoints = len(ts)
    if len(ts) > 2*npoints:
        ts = ts[numpy.arange(1, len(ts), len(ts)/npoints)]
    result = numpy.zeros((2,len(ts)))
    for i, t in enumerate(ts):
        seg = label(ucm<t)[0]
        result[0,i] = rand_index(seg, gt, None)
        result[1,i] = adj_rand_index(seg, gt, None)
    return ts, result

def vi_tables(X, Y, cont=None, ignore_seg=[], ignore_gt=[]):
    """Return probability tables used for calculating VI."""
    if cont is None:
        pxy = contingency_table(X, Y, ignore_seg, ignore_gt)
    else:
        cont[:, ignore_gt] = 0
        cont[ignore_seg,:] = 0
        pxy = cont/float(cont.sum())

    # Calculate probabilities
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
    lpygx[nzx] = xlogx(nzpxy / nzpx[:,ax]).sum(axis=1) 
                        # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)
    
    lpxgy = numpy.zeros(numpy.shape(py))
    lpxgy[nzy] = xlogx(nzpxy / nzpy[ax,:]).sum(axis=0)
    hxgy = -(py*lpxgy)

    return pxy, px, py, hxgy, hygx, lpygx, lpxgy

def split_vi(X,Y,cont=None, ignore_seg_labels=[], ignore_gt_labels=[]):
    """Return the symmetric conditional entropies associated with the VI.
    
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.
    """
    pxy,px,py,hxgy,hygx,lpygx,lpxgy = vi_tables(X,Y,cont,ignore_seg_labels,
                                                         ignore_gt_labels)
    # false merges, false splits
    return numpy.array([hygx.sum(), hxgy.sum()])

def split_components(idx, contingency, num_elems=4, axis=0):
    """Return the indices of the bodies most overlapping with body idx.

    Arguments:
        - idx: the body id being examined.
        - contingency: the normalized contingency table.
        - num_elems: the number of overlapping bodies desired.
        - axis: the axis along which to perform the calculations.
    Value:
        A list of tuples of (body_idx, overlap_int, overlap_ext).
    """
    if axis == 1:
        contingency = contingency.T
    cc = contingency / contingency.sum(axis=1)[:,numpy.newaxis]
    cct = contingency / contingency.sum(axis=0)[numpy.newaxis,:]
    idxs = (-cc[idx]).argsort()[:num_elems]
    probs = cc[idx][idxs]
    probst = cct[idx][idxs]
    return zip(idxs, probs, probst)

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
        cont = contingency_table(seg, gt, norm=False)
    a, b, c, d = rand_values(cont)
    return (a+d)/(a+b+c+d)
    
def adj_rand_index(seg, gt, cont=None):
    """Return the adjusted Rand index."""
    if cont is None:
        cont = contingency_table(seg, gt, norm=False)
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

