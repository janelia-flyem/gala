import numpy as np
import multiprocessing
import itertools as it
import collections as coll
from functools import partial
import logging
import h5py
import scipy.ndimage as nd
import scipy.sparse as sparse
from scipy.ndimage.measurements import label
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import precision_recall_curve

def bin_values(a, bins=255):
    if len(np.unique(a)) < 2*bins:
        return a.copy()
    b = np.zeros_like(a)
    m, M = a.min(), a.max()
    r = M - m
    step = r / bins
    lows = np.arange(m, M, step)
    highs = np.arange(m+step, M+step, step)
    for low, high in zip(lows, highs):
        locations = np.flatnonzero((low <= a) * (a < high))
        if len(locations) > 0:
            values = a.ravel()[locations]
            b.ravel()[locations] = values.mean()
    return b

def pixel_wise_boundary_precision_recall(aseg, gt):
    tp = float((gt * aseg).sum())
    fp = (aseg * (1-gt)).sum()
    fn = (gt * (1-aseg)).sum()
    return tp/(tp+fp), tp/(tp+fn)

def wiggle_room_precision_recall(pred, boundary, margin=2, connectivity=1):
    struct = nd.generate_binary_structure(boundary.ndim, connectivity)
    gtd = nd.binary_dilation(boundary, struct, margin)
    struct_m = nd.iterate_structure(struct, margin)
    pred_dil = nd.grey_dilation(pred, footprint=struct_m)
    missing = np.setdiff1d(np.unique(pred), np.unique(pred_dil))
    for m in missing:
        pred_dil.ravel()[np.flatnonzero(pred==m)[0]] = m
    prec, _, ts = precision_recall_curve(gtd.ravel(), pred.ravel())
    _, rec, _ = precision_recall_curve(boundary.ravel(), pred_dil.ravel())
    return zip(ts, prec, rec)

def get_stratified_sample(a, n):
    u = np.unique(a)
    if len(u) <= 2*n:
        return u
    else:
        return u[0:len(u):len(u)/n]

def edit_distance(aseg, gt, ws=None):
    if ws is None:
        return edit_distance_to_bps(aseg, gt)
    import agglo
    return edit_distance_to_bps(aseg, agglo.best_possible_segmentation(ws, gt))

def edit_distance_to_bps(aseg, bps):
    aseg = relabel_from_one(aseg)[0]
    bps = relabel_from_one(bps)[0]
    r = contingency_table(aseg, bps, [0], [0]).astype(np.bool)
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
    if not (labels == 0).any():
        labels = np.concatenate(([0], labels))
    inverse_map = labels
    return forward_map[a], forward_map, inverse_map

def contingency_table(seg, gt, ignore_seg=[0], ignore_gt=[0], norm=True):
    """Return the contingency table for all regions in matched segmentations."""
    segr = seg.ravel() 
    gtr = gt.ravel()
    ij = np.vstack((segr, gtr))
    data = np.ones(len(gtr))
    for i in ignore_seg:
        data[segr == i] = 0
    for j in ignore_gt:
        data[gtr == j] = 0
    cont = sparse.coo_matrix((data, ij)).tocsc()
    if norm:
        cont /= float(cont.sum())
    return cont

def xlogx(x, out=None, in_place=False):
    """Compute x * log_2(x).

    We define 0 * log_2(0) = 0

    Parameters
    ----------
    x : np.ndarray or scipy.sparse.csc_matrix or csr_matrix
        The input array.
    out : same type as x (optional)
        If provided, use this array/matrix for the result.
    in_place : bool (optional, default False)
        Operate directly on x.

    Returns
    -------
    y : same type as x
        Result of x * log_2(x).
    """
    if in_place:
        y = x
    elif out is None:
        y = x.copy()
    else:
        y = out
    if type(y) in [sparse.csc_matrix, sparse.csr_matrix]:
        z = y.data
    else:
        z = y
    nz = z.nonzero()
    z[nz] *= np.log2(z[nz])
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
    return make_synaptic_functions(fn, split_vi)

def make_synaptic_functions(fn, fncts):
    from syngeo import io as synio
    synapse_coords = \
        synio.raveler_synapse_annotations_to_coords(fn, 'arrays')
    synapse_coords = np.array(list(it.chain(*synapse_coords)))
    make_function = partial(special_points_evaluate, coords=synapse_coords)
    if not isinstance(fncts, coll.Iterable):
        return make_function(fncts)
    else:
        return map(make_function, fncts)

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

def calc_entropy(split_vals, count):
    col_count = 0
    for key, val in split_vals.items(): 
        col_count += val
    col_prob = float(col_count) / count 
    
    ent_val = 0
    for key, val in split_vals.items(): 
        val_norm = float(val)/count
        temp = (val_norm / col_prob)
        ent_val += temp * np.log2(temp) 
    return -(col_prob * ent_val)


def split_vi_mem(x, y):
    x_labels = np.unique(x)
    y_labels = np.unique(y)
    x_labels0 = x_labels[x_labels != 0]
    y_labels0 = y_labels[y_labels != 0]
 
    x_map = {}
    y_map = {}

    for label in x_labels0:
        x_map[label] = {}

    for label in y_labels0:
        y_map[label] = {}
    
    x_flat = x.ravel()
    y_flat = y.ravel()

    count = 0
    print "Analyzing similarities"
    for pos in range(0,len(x_flat)):
        x_val = x_flat[pos]
        y_val = y_flat[pos]

        if x_val != 0 and y_val != 0:
            x_map[x_val].setdefault(y_val, 0)
            y_map[y_val].setdefault(x_val, 0)
            (x_map[x_val])[y_val] += 1        
            (y_map[y_val])[x_val] += 1        
            count += 1
    print "Finished analyzing similarities"
     
    x_ents = {}
    y_ents = {}
    x_sum = 0.0
    y_sum = 0.0

    for key, vals in x_map.items():
        x_ents[key] = calc_entropy(vals, count)
        x_sum += x_ents[key]

    for key, vals in y_map.items():
        y_ents[key] = calc_entropy(vals, count)
        y_sum += y_ents[key]

    x_s = sorted(x_ents.items(), key=lambda x: x[1], reverse=True)
    y_s = sorted(y_ents.items(), key=lambda x: x[1], reverse=True)
    x_sorted = [ pair[0] for pair in x_s ]
    y_sorted = [ pair[0] for pair in y_s ]

    return x_sum, y_sum, x_sorted, x_ents, y_sorted, y_ents


def divide_rows(matrix, column, in_place=False):
    """Divide each row of `matrix` by the corresponding element in `column`.

    The result is as follows: out[i, j] = matrix[i, j] / column[i]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (M,)
        The column dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix or sparse.csr_matrix]:
        if type(out) == sparse.csr_matrix:
            convert_to_csr = True
            out = out.tocsc()
        else:
            convert_to_csr = False
        column_repeated = np.take(column, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= column_repeated[nz]
        if convert_to_csr:
            out = out.tocsr()
    else:
        out /= column[:, np.newaxis]
    return out


def divide_columns(matrix, row, in_place=False):
    """Divide each column of `matrix` by the corresponding element in `row`.

    The result is as follows: out[i, j] = matrix[i, j] / row[j]

    Parameters
    ----------
    matrix : np.ndarray, scipy.sparse.csc_matrix or csr_matrix, shape (M, N)
        The input matrix.
    column : a 1D np.ndarray, shape (N,)
        The row dividing `matrix`.
    in_place : bool (optional, default False)
        Do the computation in-place.

    Returns
    -------
    out : same type as `matrix`
        The result of the row-wise division.
    """
    if in_place:
        out = matrix
    else:
        out = matrix.copy()
    if type(out) in [sparse.csc_matrix or sparse.csr_matrix]:
        if type(out) == sparse.csc_matrix:
            convert_to_csc = True
            out = out.tocsr()
        else:
            convert_to_csc = False
        row_repeated = np.take(row, out.indices)
        nz = out.data.nonzero()
        out.data[nz] /= row_repeated[nz]
        if convert_to_csc:
            out = out.tocsc()
    else:
        out /= row[np.newaxis, :]
    return out

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
    px = np.array(pxy.sum(axis=1)).ravel()
    py = np.array(pxy.sum(axis=0)).ravel()
    # Remove zero rows/cols
    nzx = px.nonzero()[0]
    nzy = py.nonzero()[0]
    nzpx = px[nzx]
    nzpy = py[nzy]
    nzpxy = pxy[nzx, :][:, nzy]

    # Calculate log conditional probabilities and entropies
    lpygx = np.zeros(np.shape(px))
    lpygx[nzx] = xlogx(divide_rows(nzpxy, nzpx)).sum(axis=1) 
                        # \sum_x{p_{y|x} \log{p_{y|x}}}
    hygx = -(px*lpygx) # \sum_x{p_x H(Y|X=x)} = H(Y|X)
    
    lpxgy = np.zeros(np.shape(py))
    lpxgy[nzy] = xlogx(divide_columns(nzpxy, nzpy)).sum(axis=0)
    hxgy = -(py*lpxgy)

    return [pxy] + map(np.asarray, [px, py, hxgy, hygx, lpygx, lpxgy])

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
        iterable=[(ts, tr, ts) for ts, tr in it.permutations(range(8), 2)],
        thresholds=np.arange(0, 1.01, 0.01)):
    iterable = list(iterable)
    vi = np.zeros((3, len(thresholds), len(iterable)), np.double)
    current_vi = np.zeros(3)
    for i, t in enumerate(thresholds):
        for j, v in enumerate(iterable):
            current_fn = fn % (tuple(v) + (t,))
            try:
                f = h5py.File(current_fn, 'r')
            except IOError:
                logging.warning('IOError: could not open file %s' % current_fn)
            else:
                try:
                    current_vi = np.array(f['vi'])[:, 0]
                except IOError:
                    logging.warning('IOError: could not open file %s' 
                        % current_fn)
                except KeyError:
                    logging.warning('KeyError: could not find vi in file %s'
                        % current_fn)
                finally:
                    f.close()
            vi[:, i, j] += current_vi
    return vi

def sem(a, axis=None):
    if axis is None:
        a = a.ravel()
        axis = 0
    return np.std(a, axis=axis) / np.sqrt(a.shape[axis])

def vi_statistics(vi_table):
    return np.mean(vi_table, axis=-1), sem(vi_table, axis=-1), \
        np.median(vi_table, axis=-1)
