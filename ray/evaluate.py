import numpy as np
import multiprocessing
import itertools as it
import h5py
from scipy.sparse import coo_matrix
from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.misc import comb as nchoosek

def pixel_wise_boundary_precision_recall(aseg, gt):
    gt = (1-gt.astype(np.bool)).astype(np.uint8)
    aseg = (1-aseg.astype(np.bool)).astype(np.uint8)
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
    r = contingency_table(aseg, bps).astype(np.bool)
    if (bps==0).any(): r[:,0] = 0
    if (aseg==0).any(): r[0,:] = 0
    false_splits = (r.sum(axis=0)-1)[1:].sum()
    false_merges = (r.sum(axis=1)-1)[1:].sum()
    return (false_merges, false_splits)

def relabel_from_one(a):
    labels = np.unique(a)
    labels0 = labels[labels!=0]
    m = labels.max()
    if m == len(labels0): # nothing to do, already 1...n labels
        return a, labels, labels
    forward_map = np.zeros(m+1, int)
    forward_map[labels0] = np.arange(1, len(labels0)+1)
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map

def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations."""
    gtr = gt.ravel()
    segr = seg.ravel() 
    ij = np.zeros((2,len(gtr)))
    ij[0,:] = segr
    ij[1,:] = gtr
    cont = coo_matrix((np.ones((len(gtr))), ij)).toarray()
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
    y[nz] *= np.log2(y[nz])
    return y

def special_points_evaluate(eval_fct, coords, flatten=True, coord_format=True):
    if coord_format:
        coords = [coords[:,i] for i in range(coords.shape[1])]
    def special_eval_fct(x, y, *args, **kwargs):
        if flatten:
            for i in range(len(coords)):
                if coords[i][0] < 0:
                    coords[i] += x.shape[i]
            coords2 = np.ravel_multi_index(coords, x.shape)
        else:
            coords2 = coords
        sx = x.ravel()[coords2]
        sy = y.ravel()[coords2]
        return eval_fct(sx, sy, *args, **kwargs)
    return special_eval_fct

def make_synaptic_vi(fn):
    from syngeo import io as synio
    synapse_coords = \
        synio.raveler_synapse_annotations_to_coords(fn, 'arrays')
    synapse_coords = np.array(list(it.chain(*synapse_coords)))
    return special_points_evaluate(split_vi, synapse_coords)

def vi(x, y=None, weights=np.ones(2), ignore_x=[0], ignore_y=[0]):
    """Return the variation of information metric."""
    return np.dot(weights, split_vi(x, y, ignore_x, ignore_y))

def split_vi(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return the symmetric conditional entropies associated with the VI.
    
    The variation of information is defined as VI(X,Y) = H(X|Y) + H(Y|X).
    If Y is the ground-truth segmentation, then H(Y|X) can be interpreted
    as the amount of under-segmentation of Y and H(X|Y) is then the amount
    of over-segmentation.  In other words, a perfect over-segmentation
    will have H(Y|X)=0 and a perfect under-segmentation will have H(X|Y)=0.

    If y is None, x is assumed to be a contingency table.
    """
    _, _, _ , hxgy, hygx, _, _ = vi_tables(x, y, ignore_x, ignore_y)
    # false merges, false splits
    return np.array([hygx.sum(), hxgy.sum()])

def vi_pairwise_matrix(segs, split=False):
    """Compute the pairwise VI distances within a set of segmentations.
    
    If 'split' is set to True, two matrices are returned, one for each 
    direction of the conditional entropy.

    0-labeled pixels are ignored.
    """
    d = np.array([s.ravel() for s in segs])
    if split:
        def dmerge(x, y): return split_vi(x, y)[0]
        def dsplit(x, y): return split_vi(x, y)[1]
        merges, splits = [squareform(pdist(d, df)) for df in [dmerge, dsplit]]
        out = merges
        tri = np.tril(np.ones(splits.shape), -1).astype(bool)
        out[tri] = splits[tri]
    else:
        out = squareform(pdist(d, vi))
    return out

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
    return split_vi(label(ucm<t)[0], gt, ignore_seg, ignore_gt)

def vi_by_threshold(ucm, gt, ignore_seg=[], ignore_gt=[], npoints=None,
                                                            nprocessors=None):
    ts = np.unique(ucm)[1:]
    if npoints is None:
        npoints = len(ts)
    if len(ts) > 2*npoints:
        ts = ts[np.arange(1, len(ts), len(ts)/npoints)]
    if nprocessors == 1: # this should avoid pickling overhead
        result = [split_vi_threshold((ucm, gt, ignore_seg, ignore_gt, t))
                for t in ts]
    else:
        p = multiprocessing.Pool(nprocessors)
        result = p.map(split_vi_threshold, 
            ((ucm, gt, ignore_seg, ignore_gt, t) for t in ts))
    return np.concatenate((ts[np.newaxis, :], np.array(result).T), axis=0)

def rand_by_threshold(ucm, gt, npoints=None):
    ts = np.unique(ucm)[1:]
    if npoints is None:
        npoints = len(ts)
    if len(ts) > 2*npoints:
        ts = ts[np.arange(1, len(ts), len(ts)/npoints)]
    result = np.zeros((2,len(ts)))
    for i, t in enumerate(ts):
        seg = label(ucm<t)[0]
        result[0,i] = rand_index(seg, gt)
        result[1,i] = adj_rand_index(seg, gt)
    return np.concatenate((ts[np.newaxis, :], result), axis=0)

def vi_tables(x, y=None, ignore_x=[0], ignore_y=[0]):
    """Return probability tables used for calculating VI.
    
    If y is None, x is assumed to be a contingency table.
    """
    if y is not None:
        pxy = contingency_table(x, y, ignore_x, ignore_y)
    else:
        cont = x
        cont[:, ignore_y] = 0
        cont[ignore_x, :] = 0
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
    ax = np.newaxis
    lpygx = np.zeros(np.shape(px))
    lpygx[nzx] = xlogx(nzpxy / nzpx[:,ax]).sum(axis=1) 
                        # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)
    
    lpxgy = np.zeros(np.shape(py))
    lpxgy[nzy] = xlogx(nzpxy / nzpy[ax,:]).sum(axis=0)
    hxgy = -(py*lpxgy)

    return pxy, px, py, hxgy, hygx, lpygx, lpxgy

def sorted_vi_components(s1, s2, ignore1=[0], ignore2=[0], compress=True):
    """Return lists of the most entropic segments in s1|s2 and s2|s1.
    
    The 'compress' flag performs a remapping of the labels before doing the
    VI computation, resulting in massive memory savings when many labels are
    not used in the volume. (For example, if you have just two labels, 1 and
    1,000,000, 'compress=False' will give a VI contingency table having
    1,000,000 entries to a side, whereas 'compress=True' will have just size
    2.)
    """
    if compress:
        s1, forw1, back1 = relabel_from_one(s1)
        s2, forw2, back2 = relabel_from_one(s2)
    _, _, _, h1g2, h2g1, _, _ = vi_tables(s1, s2, ignore1, ignore2)
    i1 = (-h2g1).argsort()
    i2 = (-h1g2).argsort()
    ii1 = back1[i1] if compress else i1
    ii2 = back2[i2] if compress else i2
    return ii1, h2g1[i1], ii2, h1g2[i2]

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
    cc = contingency / contingency.sum(axis=1)[:,np.newaxis]
    cct = contingency / contingency.sum(axis=0)[np.newaxis,:]
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

def rand_index(x, y=None):
    """Return the unadjusted Rand index."""
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    return (a+d)/(a+b+c+d)
    
def adj_rand_index(x, y=None):
    """Return the adjusted Rand index."""
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    nk = a+b+c+d
    return (nk*(a+d) - ((a+b)*(a+c) + (c+d)*(b+d)))/(
        nk**2 - ((a+b)*(a+c) + (c+d)*(b+d)))

def fm_index(x, y=None):
    """ Return the Fowlkes-Mallows index. """
    cont = x if y is None else contingency_table(x, y, norm=False)
    a, b, c, d = rand_values(cont)
    return a/(np.sqrt((a+b)*(a+c)))

def reduce_vi(fn='testing/%i/flat-single-channel-tr%i-%i-%.2f.lzf.h5',
    iterable=[(ts, tr, ts) for ts, tr in it.combinations(range(8), 2)],
    thresholds=np.arange(0, 1.01, 0.01)):
    iterable = list(iterable)
    vi = np.zeros((3, len(thresholds), len(iterable)), np.double)
    for i, t in enumerate(thresholds):
        for j, v in enumerate(iterable):
            f = h5py.File(fn % (tuple(v) + (t,)), 'r')
            vi[:, i, j] += np.array(f['vi'])[:, 0]
            f.close()
    return vi

def sem(a, axis=None):
    if axis is None:
        a = a.ravel()
        axis = 0
    return np.std(a, axis=axis) / np.sqrt(a.shape[axis])

def vi_statistics(vi_table):
    return np.mean(vi_table, axis=-1), sem(vi_table, axis=-1), \
        np.median(vi_table, axis=-1)
