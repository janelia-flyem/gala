#!/usr/bin/env python

import numpy as np
from numpy import   reshape, \
                    array, zeros, zeros_like, ones, arange, \
                    double, \
                    int8, int16, int32, int64, uint8, uint16, uint32, uint64, \
                    iinfo, isscalar, \
                    unique, \
                    newaxis, \
                    minimum, bincount, dot, nonzero, concatenate, \
                    setdiff1d, flatnonzero
import itertools as it
from collections import defaultdict, deque as queue
from scipy.ndimage import grey_dilation, generate_binary_structure, \
        maximum_filter, minimum_filter
from scipy import ndimage as ndi
from scipy.ndimage import distance_transform_cdt
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_opening, binary_dilation
from . import iterprogress as ip
from skimage.segmentation import relabel_sequential
from skimage import measure, util, feature
import skimage.morphology

import joblib

zero3d = array([0,0,0])

def complement(a):
    return a.max()-a


def remove_merged_boundaries(labels, connectivity=1):
    """Remove boundaries in a label field when they separate the same region.

    By convention, the boundary label is 0, and labels are positive.

    Parameters
    ----------
    labels : array of int
        The label field to be processed.
    connectivity : int in {1, ..., labels.ndim}, optional
        The morphological connectivity for considering neighboring voxels.

    Returns
    -------
    labels_out : array of int
        The same label field, with unnecessary boundaries removed.

    Examples
    --------
    >>> labels = np.array([[1, 0, 1], [0, 1, 0], [2, 0, 3]], np.int)
    >>> remove_merged_boundaries(labels)
    array([[1, 1, 1],
           [0, 1, 0],
           [2, 0, 3]])
    """
    boundary = 0
    labels_out = labels.copy()
    is_boundary = (labels == boundary)
    labels_complement = labels.copy()
    labels_complement[is_boundary] = labels.max() + 1
    se = ndi.generate_binary_structure(labels.ndim, connectivity)
    smaller_labels = ndi.grey_erosion(labels_complement, footprint=se)
    bigger_labels = ndi.grey_dilation(labels, footprint=se)
    merged = is_boundary & (smaller_labels == bigger_labels)
    labels_out[merged] = smaller_labels[merged]
    return labels_out


def morphological_reconstruction(marker, mask, connectivity=1):
    """Perform morphological reconstruction of the marker into the mask.
    
    See the Matlab image processing toolbox documentation for details:
    http://www.mathworks.com/help/toolbox/images/f18-16264.html
    """
    sel = generate_binary_structure(marker.ndim, connectivity)
    diff = True
    while diff:
        markernew = grey_dilation(marker, footprint=sel)
        markernew = minimum(markernew, mask)
        diff = (markernew-marker).max() > 0
        marker = markernew
    return marker

def hminima(a, thresh):
    """Suppress all minima that are shallower than thresh.

    Parameters
    ----------
    a : array
        The input array on which to perform hminima.
    thresh : float
        Any local minima shallower than this will be flattened.

    Returns
    -------
    out : array
        A copy of the input array with shallow minima suppressed.
    """
    maxval = a.max()
    ainv = maxval-a
    return maxval - morphological_reconstruction(ainv-thresh, ainv)

imhmin = hminima

remove_small_connected_components = skimage.morphology.remove_small_objects

def regional_minima(a, connectivity=1):
    """Find the regional minima in an ndarray."""
    values = unique(a)
    delta = (values - minimum_filter(values, footprint=ones(3)))[1:].min()
    marker = complement(a)
    mask = marker+delta
    return marker == morphological_reconstruction(marker, mask, connectivity)

def impose_minima(a, minima, connectivity=1):
    """Transform 'a' so that its only regional minima are those in 'minima'.
    
    Parameters:
        'a': an ndarray
        'minima': a boolean array of same shape as 'a'
        'connectivity': the connectivity of the structuring element used in
        morphological reconstruction.
    Value:
        an ndarray of same shape as a with unmarked local minima paved over.
    """
    m = a.max()
    mask = m - a
    marker = zeros_like(mask)
    minima = minima.astype(bool)
    marker[minima] = mask[minima]
    return m - morphological_reconstruction(marker, mask, connectivity)

def minimum_seeds(current_seeds, min_seed_coordinates, connectivity=1):
    """Ensure that each point in given coordinates has its own seed."""
    seeds = current_seeds.copy()
    sel = generate_binary_structure(seeds.ndim, connectivity)
    if seeds.dtype == bool:
        seeds = label(seeds, sel)[0]
    new_seeds = grey_dilation(seeds, footprint=sel)
    overlap = new_seeds[min_seed_coordinates]
    seed_overlap_counts = bincount(concatenate((overlap, unique(seeds)))) - 1
    seeds_to_delete = (seed_overlap_counts > 1)[seeds]
    seeds[seeds_to_delete] = 0
    seeds_to_add = [m[overlap==0] for m in min_seed_coordinates]
    start = seeds.max() + 1
    num_seeds = len(seeds_to_add[0])
    seeds[seeds_to_add] = arange(start, start + num_seeds)
    return seeds

def split_exclusions(image, labels, exclusions, dilation=0, connectivity=1,
    standard_seeds=False):
    """Ensure that no segment in 'labels' overlaps more than one exclusion."""
    labels = labels.copy()
    cur_label = labels.max()
    dilated_exclusions = exclusions.copy()
    foot = generate_binary_structure(exclusions.ndim, connectivity)
    for i in range(dilation):
        dilated_exclusions = grey_dilation(exclusions, footprint=foot)
    hashed = labels * (exclusions.max() + 1) + exclusions
    hashed[exclusions == 0] = 0
    violations = bincount(hashed.ravel()) > 1
    violations[0] = False
    if sum(violations) != 0:
        offending_labels = labels[violations[hashed]]
        mask = zeros(labels.shape, dtype=bool)
        for offlabel in offending_labels:
            mask += labels == offlabel
        if standard_seeds:
            seeds = label(mask * (image == 0))[0]
        else:
            seeds = label(mask * dilated_exclusions)[0]
        seeds[seeds > 0] += cur_label
        labels[mask] = watershed(image, seeds, connectivity, mask)[mask]
    return labels


def watershed(a, seeds=None, connectivity=1, mask=None, smooth_thresh=0.0, 
        smooth_seeds=False, minimum_seed_size=0, dams=False,
        override_skimage=False, show_progress=False):
    """Perform the watershed algorithm of Vincent & Soille (1991).
    
    Parameters
    ----------
    a : np.ndarray, arbitrary shape and type
        The input image on which to perform the watershed transform.
    seeds : np.ndarray, int or bool type, same shape as `a` (optional)
        The seeds for the watershed. If provided, these are the only basins
        allowed, and the algorithm proceeds by flooding from the seeds.
        Otherwise, every local minimum is used as a seed.
    connectivity : int, {1, ..., a.ndim} (optional, default 1)
        The neighborhood of each pixel, defined as in `scipy.ndimage`.
    mask : np.ndarray, type bool, same shape as `a`. (optional)
        If provided, perform watershed only in the parts of `a` that are set
        to `True` in `mask`.
    smooth_thresh : float (optional, default 0.0)
        Local minima that are less deep than this threshold are suppressed,
        using `hminima`.
    smooth_seeds : bool (optional, default False)
        Perform binary opening on the seeds, using the same connectivity as
        the watershed.
    minimum_seed_size : int (optional, default 0)
        Remove seed regions smaller than this size.
    dams : bool (optional, default False)
        Place a dam where two basins meet. Set this to True if you require
        0-labeled boundaries between different regions.
    override_skimage : bool (optional, default False)
        skimage.morphology.watershed is used to implement the main part of the
        algorithm when `dams=False`. Use this flag to use the separate pure
        Python implementation instead.
    show_progress : bool (optional, default False)
        Show a cute little ASCII progress bar (using the progressbar package)

    Returns
    -------
    ws : np.ndarray, same shape as `a`, int type.
        The watershed transform of the input image.
    """
    seeded = seeds is not None
    sel = generate_binary_structure(a.ndim, connectivity)
    # various keyword arguments operate by modifying the input image `a`.
    # However, we operate on a copy of it called `b`, so that `a` can be used
    # to break ties.
    b = a
    if not seeded:
        seeds = regional_minima(a, connectivity)
    if minimum_seed_size > 0:
        seeds = remove_small_connected_components(seeds, minimum_seed_size)
        seeds = relabel_sequential(seeds)[0]
    if smooth_seeds:
        seeds = binary_opening(seeds, sel)
    if smooth_thresh > 0.0:
        b = hminima(a, smooth_thresh)
    if seeds.dtype == bool:
        seeds = label(seeds, sel)[0]
    if not override_skimage and not dams:
        return skimage.morphology.watershed(b, seeds, sel, None, mask)
    elif seeded:
        b = impose_minima(a, seeds.astype(bool), connectivity)
    levels = unique(b)
    a = pad(a, a.max()+1)
    b = pad(b, b.max()+1)
    ar = a.ravel()
    br = b.ravel()
    ws = pad(seeds, 0)
    wsr = ws.ravel()
    neighbors = build_neighbors_array(a, connectivity)
    level_pixels = build_levels_dict(b)
    if show_progress: wspbar = ip.StandardProgressBar('Watershed...')
    else: wspbar = ip.NoProgressBar()
    for i, level in ip.with_progress(enumerate(levels), 
                                            pbar=wspbar, length=len(levels)):
        idxs_adjacent_to_labels = queue([idx for idx in level_pixels[level] if
                                            any(wsr[neighbors[idx]])])
        while len(idxs_adjacent_to_labels) > 0:
            idx = idxs_adjacent_to_labels.popleft()
            if wsr[idx] > 0: continue # in case we already processed it
            nidxs = neighbors[idx] # neighbors
            lnidxs = nidxs[(wsr[nidxs] != 0).astype(bool)] # labeled neighbors
            adj_labels = unique(wsr[lnidxs])
            if len(adj_labels) == 1 or len(adj_labels) > 1 and not dams: 
                # assign a label
                wsr[idx] = wsr[lnidxs][ar[lnidxs].argmin()]
                idxs_adjacent_to_labels.extend(nidxs[((wsr[nidxs] == 0) * 
                                    (br[nidxs] == level)).astype(bool) ])
    return juicy_center(ws)


def multiscale_regular_seeds(off_limits, num_seeds):
    """Return evenly-spaced seeds, but thinned in areas with no boundaries.

    Parameters
    ----------
    off_limits : array of bool, shape (M, N)
        A binary array where `True` indicates the position of a boundary,
        and thus where we don't want to place seeds.
    num_seeds : int
        The desired number of seeds.

    Returns
    -------
    seeds : array of int, shape (M, N)
        An array of seed points. Each seed gets its own integer ID,
        starting from 1.
    """
    seeds_binary = np.zeros(off_limits.shape, dtype=bool)
    grid = util.regular_grid(off_limits.shape, num_seeds)
    seeds_binary[grid] = True
    seeds_binary &= ~off_limits
    seeds_img = seeds_binary[grid]
    thinned_equal = False
    step = 2
    while not thinned_equal:
        thinned = _thin_seeds(seeds_img, step)
        thinned_equal = np.all(seeds_img == thinned)
        seeds_img = thinned
        step *= 2
    seeds_binary[grid] = seeds_img
    return ndi.label(seeds_binary)[0]


def _thin_seeds(seeds_img, step):
    out = np.copy(seeds_img)
    m, n = seeds_img.shape
    for r in range(0, m, step):
        for c in range(0, n, step):
            window = (slice(r, min(r + 5 * step // 2, m), step // 2),
                      slice(c, min(c + 5 * step // 2, n), step // 2))
            if np.all(seeds_img[window]):
                out[window][1::2, :] = False
                out[window][:, 1::2] = False
    return out


def multiscale_seed_sequence(prob, l1_threshold=0, grid_density=10):
    npoints = ((prob.shape[1] // grid_density) *
               (prob.shape[2] // grid_density))
    seeds = np.zeros(prob.shape, dtype=int)
    for seed, p in zip(seeds, prob):
        hm = feature.hessian_matrix(p, sigma=3)
        l1, l2 = feature.hessian_matrix_eigvals(*hm)
        curvy = (l1 > l1_threshold)
        seed[:] = multiscale_regular_seeds(curvy, npoints)
    return seeds


def pipeline_compact_watershed(prob, *,
                               invert_prob=True,
                               l1_threshold=0,
                               grid_density=10,
                               compactness=0.01,
                               n_jobs=1):
    if invert_prob:
        prob = np.max(prob) - prob
    seeds = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(multiscale_seed_sequence)(p[np.newaxis, :],
                                                     l1_threshold=l1_threshold,
                                                     grid_density=grid_density)
            for p in prob)
    seeds = np.reshape(seeds, prob.shape)
    fragments = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(compact_watershed)(p, s, compactness=compactness)
        for p, s in zip(prob, seeds)
    )
    fragments = np.array(fragments)
    max_ids = fragments.max(axis=-1).max(axis=-1)
    to_add = np.concatenate(([0], np.cumsum(max_ids)[:-1]))
    fragments += to_add[:, np.newaxis, np.newaxis]
    return fragments


def _euclid_dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def compact_watershed(a, seeds, *, compactness=0.01, connectivity=1):
    try:
        a = np.copy(a)
        a[np.nonzero(seeds)] = np.min(a)
        result = skimage.morphology.watershed(a, seeds,
                                              connectivity=connectivity,
                                              compactness=compactness)
        return result
    except TypeError:  # old version of skimage
        import warnings
        warnings.warn('skimage prior to 0.13; compact watershed will be slow.')
    from .mergequeue import MergeQueue
    visiting_queue = MergeQueue()
    seeds = pad(seeds, 0).ravel()
    seed_coords = np.flatnonzero(seeds)
    visited = np.zeros(a.shape, dtype=bool)
    visited = pad(visited, True).ravel()
    ap = pad(a.astype(float), np.inf)
    apr = ap.ravel()
    neigh_sum = raveled_steps_to_neighbors(ap.shape, connectivity)
    result = np.zeros_like(seeds)
    for c in seed_coords:
        visiting_queue.push([0, True, c, seeds[c],
                             np.unravel_index(c, ap.shape)])
    while len(visiting_queue) > 0:
        _, _, next_coord, next_label, next_origin = visiting_queue.pop()
        if not visited[next_coord]:
            visited[next_coord] = True
            result[next_coord] = next_label
            neighbor_coords = next_coord + neigh_sum
            for coord in neighbor_coords:
                if not visited[coord]:
                    full_coord = np.array(np.unravel_index(coord, ap.shape))
                    cost = (apr[coord] +
                            compactness*_euclid_dist(full_coord, next_origin))
                    visiting_queue.push([cost, True, coord, next_label,
                                         next_origin])
    return juicy_center(result.reshape(ap.shape))


def watershed_sequence(a, seeds=None, mask=None, axis=0, n_jobs=1, **kwargs):
    """Perform a watershed on a plane-by-plane basis.

    See documentation for `watershed` for available kwargs.

    The watershed algorithm views image intensity as "height" and finds flood
    basins within it. These basins are then viewed as the different labeled
    regions of an image.

    This function performs watershed on an ndarray on each plane separately,
    then concatenate the results.

    Parameters
    ----------
    a : numpy ndarray, arbitrary type or shape.
        The input image on which to perform the watershed transform.
    seeds : bool/int numpy.ndarray, same shape as a (optional, default None)
        The seeds for the watershed.
    mask : bool numpy.ndarray, same shape as a (optional, default None)
        If provided, perform watershed only over voxels that are True in the
        mask.
    axis : int, {1, ..., a.ndim} (optional, default: 0)
        Which axis defines the plane sequence. For example, if the input image
        is 3D and axis=1, then the output will be the watershed on a[:, 0, :], 
        a[:, 1, :], a[:, 2, :], ... and so on.
    n_jobs : int, optional
        Use joblib to distribute each plane over given number of processing
        cores. If -1, `multiprocessing.cpu_count` is used.

    Returns
    -------
    ws : numpy ndarray, int type
        The labeled watershed basins.

    Other parameters
    ----------------
    **kwargs : keyword arguments passed through to the `watershed` function.
    """
    if axis != 0:
        a = a.swapaxes(0, axis).copy()
        if seeds is not None:
            seeds = seeds.swapaxes(0, axis)
        if mask is not None:
            mask = mask.swapaxes(0, axis)
    if seeds is None:
        seeds = it.repeat(None)
    if mask is None:
        mask = it.repeat(None)
    ws = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(watershed)(i, seeds=s, mask=m, **kwargs)
        for i, s, m in zip(a, seeds, mask))
    counts = list(map(np.max, ws[:-1]))
    counts = np.concatenate((np.array([0]), counts))
    counts = np.cumsum(counts)
    for c, w in zip(counts, ws):
        w += c
    ws = np.concatenate([w[np.newaxis, ...] for w in ws], axis=0)
    if axis != 0:
        ws = ws.swapaxes(0, axis).copy()
    return ws


def manual_split(probs, seg, body, seeds, connectivity=1, boundary_seeds=None):
    """Manually split a body from a segmentation using seeded watershed.

    Input:
        - probs: the probability of boundary in the volume given.
        - seg: the current segmentation.
        - body: the label to be split.
        - seeds: the seeds for the splitting (should be just two labels).
        [-connectivity: the connectivity to use for watershed.]
        [-boundary_seeds: if not None, these locations become inf in probs.]
    Value:
        - the segmentation with the selected body split.
    """
    struct = generate_binary_structure(seg.ndim, connectivity)
    body_pixels = seg == body
    bbox = find_objects(body_pixels)[0]
    body_pixels = body_pixels[bbox]
    body_boundary = binary_dilation(body_pixels, struct) - body_pixels
    non_body_pixels = True - body_pixels - body_boundary
    probs = probs.copy()[bbox]
    probs[non_body_pixels] = probs.min()-1
    if boundary_seeds is not None:
        probs[boundary_seeds[bbox]] = probs.max()+1
    probs[body_boundary] = probs.max()+1
    seeds = label(seeds.astype(bool)[bbox], struct)[0]
    outer_seed = seeds.max()+1 # should be 3
    seeds[non_body_pixels] = outer_seed
    seg_new = watershed(probs, seeds, 
        dams=(seg==0).any(), connectivity=connectivity, show_progress=True)
    seg = seg.copy()
    new_seeds = unique(seeds)[:-1]
    for new_seed, new_label in zip(new_seeds, [0, body, seg.max()+1]):
        seg[bbox][seg_new == new_seed] = new_label
    return seg


def relabel_connected(im, connectivity=1):
    """Ensure all labels in `im` are connected.

    Parameters
    ----------
    im : array of int
        The input label image.
    connectivity : int in {1, ..., `im.ndim`}, optional
        The connectivity used to determine if two voxels are neighbors.

    Returns
    -------
    im_out : array of int
        The relabeled image.

    Examples
    --------
    >>> image = np.array([[1, 1, 2],
    ...                   [2, 1, 1]])
    >>> im_out = relabel_connected(image)
    >>> im_out
    array([[1, 1, 2],
           [3, 1, 1]])
    """
    im_out = np.zeros_like(im)
    contiguous_segments = np.empty_like(im)
    structure = generate_binary_structure(im.ndim, connectivity)
    curr_label = 0
    labels = np.unique(im)
    if labels[0] == 0:
        labels = labels[1:]
    for label in labels:
        segment = (im == label)
        n_segments = ndi.label(segment, structure,
                               output=contiguous_segments)
        seg = segment.nonzero()
        contiguous_segments[seg] += curr_label
        im_out[seg] += contiguous_segments[seg]
        curr_label += n_segments
    return im_out


def smallest_int_dtype(number, signed=False, mindtype=None):
    if number < 0: signed = True
    if not signed:
        if number <= iinfo(uint8).max:
            return uint8
        if number <= iinfo(uint16).max:
            return uint16
        if number <= iinfo(uint32).max:
            return uint32
        if number <= iinfo(uint64).max:
            return uint64
    else:
        if iinfo(int8).min <= number <= iinfo(int8).max:
            return int8
        if iinfo(int16).min <= number <= iinfo(int16).max:
            return int16
        if iinfo(int32).min <= number <= iinfo(int32).max:
            return int32
        if iinfo(int64).min <= number <= iinfo(int64).max:
            return int64

def _is_container(a):
    try:
        n = len(a)
        return True
    except TypeError:
        return False

def pad(ar, vals, axes=None):
    if ar.size == 0:
        return ar
    if axes is None:
        axes = list(range(ar.ndim))
    if not _is_container(vals):
        vals = [vals]
    if not _is_container(axes):
        axes = [axes]
    padding_thickness = len(vals)
    newshape = array(ar.shape)
    for ax in axes:
        newshape[ax] += 2
    vals = array(vals)
    if ar.dtype == double or ar.dtype == float:
        new_dtype = double
    elif ar.dtype == bool:
        new_dtype = bool
    else:
        maxval = max([vals.max(), ar.max()])
        minval = min([vals.min(), ar.min()])
        if abs(minval) > maxval:
            signed = True
            extremeval = minval
        else:
            if minval < 0:
                signed = True
            else:
                signed = False
            extremeval = maxval
        new_dtype = max([smallest_int_dtype(extremeval, signed), ar.dtype])
    ar2 = zeros(newshape, dtype=new_dtype)
    center = ones(newshape, dtype=bool)
    for ax in axes:
        ar2.swapaxes(0,ax)[0,...] = vals[0]
        ar2.swapaxes(0,ax)[-1,...] = vals[0]
        center.swapaxes(0,ax)[0,...] = False
        center.swapaxes(0,ax)[-1,...] = False
    ar2[center] = ar.ravel()
    if padding_thickness == 1:
        return ar2
    else:
        return pad(ar2, vals[1:], axes)
        
def juicy_center(ar, skinsize=1):
    for i in range(ar.ndim):
        ar = ar.swapaxes(0,i)
        ar = ar[skinsize:-skinsize]
        ar = ar.swapaxes(0,i)
    return ar.copy()

def surfaces(ar, skinsize=1):
    s = []
    for i in range(ar.ndim):
        ar = ar.swapaxes(0, i)
        s.append(ar[0:skinsize].copy())
        s.append(ar[-skinsize:].copy())
        ar = ar.swapaxes(0, i)
    return s

def hollowed(ar, skinsize=1):
    """Return a copy of ar with the center zeroed out.

    'skinsize' determines how thick of a crust to leave untouched.
    """
    slices = (slice(skinsize, -skinsize),)*ar.ndim
    ar_out = ar.copy()
    ar_out[slices] = 0
    return ar_out

def build_levels_dict(a):
    d = defaultdict(list)
    for loc,val in enumerate(a.ravel()):
        d[val].append(loc)
    return d

def build_neighbors_array(ar, connectivity=1):
    idxs = arange(ar.size, dtype=uint32)
    return get_neighbor_idxs(ar, idxs, connectivity)


def raveled_steps_to_neighbors(shape, connectivity=1):
    """Compute the stepsize along all axes for given connectivity and shape.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array along which we are stepping.
    connectivity : int in {1, 2, ..., ``len(shape)``}
        The number of orthogonal steps we can take to reach a "neighbor".

    Returns
    -------
    steps : array of int64
        The steps needed to get to neighbors from a particular raveled
        index.

    Examples
    --------
    >>> shape = (5, 4, 9)
    >>> steps = raveled_steps_to_neighbors(shape)
    >>> sorted(steps)
    [-36, -9, -1, 1, 9, 36]
    >>> steps2 = raveled_steps_to_neighbors(shape, 2)
    >>> sorted(steps2)
    [-45, -37, -36, -35, -27, -10, -9, -8, -1, 1, 8, 9, 10, 27, 35, 36, 37, 45]
    """
    stepsizes = np.cumprod((1,) + shape[-1:0:-1])[::-1]
    steps = []
    steps.extend((stepsizes, -stepsizes))
    for nhops in range(2, connectivity + 1):
        prod = np.array(list(it.product(*([[1, -1]] * nhops))))
        multisteps = np.array(list(it.combinations(stepsizes, nhops))).T
        steps.append(np.dot(prod, multisteps).ravel())
    return np.concatenate(steps).astype(np.int64)


def get_neighbor_idxs(ar, idxs, connectivity=1):
    """Return indices of neighboring voxels given array, indices, connectivity.

    Parameters
    ----------
    ar : ndarray
        The array in which neighbors are to be found.
    idxs : int or container of int
        The indices for which to find neighbors.
    connectivity : int in {1, 2, ..., ``ar.ndim``}
        The number of orthogonal steps allowed to be considered a
        neighbor.

    Returns
    -------
    neighbor_idxs : 2D array, shape (nidxs, nneighbors)
        The neighbor indices for each index passed.

    Examples
    --------
    >>> ar = np.arange(16).reshape((4, 4))
    >>> ar
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> get_neighbor_idxs(ar, [5, 10], connectivity=1)
    array([[ 9,  6,  1,  4],
           [14, 11,  6,  9]])
    >>> get_neighbor_idxs(ar, 9, connectivity=2)
    array([[13, 10,  5,  8, 14, 12,  6,  4]])
    """
    if isscalar(idxs):  # in case only a single idx is given
        idxs = [idxs]
    idxs = array(idxs)  # in case a list or other array-like is given
    steps = raveled_steps_to_neighbors(ar.shape, connectivity)
    return idxs[:, np.newaxis] + steps


def orphans(a):
    """Find all the segments that do not touch the volume boundary.
    
    This function differs from agglo.Rag.orphans() in that it does not use the
    graph, but rather computes orphans directly from a volume.

    Parameters
    ----------
    a : array of int
        A segmented volume.

    Returns
    -------
    orph : 1D array of int
        The IDs of any segments not touching the volume boundary.

    Examples
    --------
    >>> segs = np.array([[1, 1, 1, 2],
    ...                  [1, 3, 4, 2],
    ...                  [1, 2, 2, 2]], int)
    >>> orphans(segs)
    array([3, 4])
    >>> orphans(segs[:2])
    array([], dtype=int64)
    """
    return setdiff1d(
            unique(a), unique(concatenate([s.ravel() for s in surfaces(a)]))
            )


def non_traversing_segments(a):
    """Find segments that enter the volume but do not leave it elsewhere.

    Parameters
    ----------
    a : array of int
        A segmented volume.

    Returns
    -------
    nt : 1D array of int
        The IDs of any segments not traversing the volume.

    Examples
    --------
    >>> segs = np.array([[1, 2, 3, 3, 4],
    ...                  [1, 2, 2, 3, 4],
    ...                  [1, 5, 5, 3, 4],
    ...                  [1, 1, 5, 3, 4]], int)
    >>> non_traversing_segments(segs)
    array([1, 2, 4, 5])
    """
    surface = hollowed(a)
    surface_ccs = measure.label(surface) + 1
    surface_ccs[surface == 0] = 0
    idxs = flatnonzero(surface)
    pairs = np.array(list(zip(surface.ravel()[idxs],
                              surface_ccs.ravel()[idxs])))
    unique_pairs = util.unique_rows(pairs)
    surface_singles = np.bincount(unique_pairs[:, 0]) == 1
    nt = np.flatnonzero(surface_singles)
    return nt


def damify(a, in_place=False):
    """Add dams to a borderless segmentation."""
    if not in_place:
        b = a.copy()
    b[seg_to_bdry(a)] = 0
    return b

def seg_to_bdry(seg, connectivity=1):
    """Given a borderless segmentation, return the boundary map."""
    strel = generate_binary_structure(seg.ndim, connectivity)
    return maximum_filter(seg, footprint=strel) != \
           minimum_filter(seg, footprint=strel)
    
def undam(seg):
    """ Assign zero-dams to nearest non-zero region. """
    bdrymap = seg==0
    k = distance_transform_cdt(bdrymap, return_indices=True)
    ind = nonzero(bdrymap.ravel())[0]
    closest_sub = concatenate([i.ravel()[:,newaxis] for i in k[1]],axis=1)
    closest_sub = closest_sub[ind,:]
    closest_ind = [
        dot(bdrymap.strides, i)/bdrymap.itemsize for i in closest_sub]
    sp = seg.shape
    seg = seg.ravel()
    seg[ind] = seg[closest_ind]
    seg = reshape(seg, sp)
    return seg

if __name__ == '__main__':
    pass
