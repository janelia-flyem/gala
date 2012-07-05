#!/usr/bin/env python

import sys, os, argparse
from numpy import   shape, reshape, \
                    array, zeros, zeros_like, ones, ones_like, arange, \
                    double, \
                    int8, int16, int32, int64, uint8, uint16, uint32, uint64, \
                    uint, \
                    iinfo, isscalar, \
                    unique, \
                    where, unravel_index, newaxis, \
                    ceil, floor, prod, cumprod, \
                    concatenate, \
                    ndarray, minimum, bincount, dot, nonzero, concatenate, \
                    setdiff1d, inf, flatnonzero
import itertools
import re
from collections import defaultdict, deque as queue
from scipy.ndimage import filters, grey_dilation, generate_binary_structure, \
        maximum_filter, minimum_filter
from scipy.ndimage import distance_transform_cdt
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_opening, binary_closing, \
    binary_dilation, grey_opening, grey_closing, \
    generate_binary_structure, iterate_structure
#from scipy.spatial.distance import cityblock as manhattan_distance
import iterprogress as ip

try:
    import skimage.morphology
    skimage_available = True
except ImportError:
    logging.warning('Unable to load skimage.')
    skimage_available = False

zero3d = array([0,0,0])

def manhattan_distance(a, b):
    return sum(abs(a-b))

def diamond_se(radius, dimension):
    se = generate_binary_structure(dimension, 1)
    return iterate_structure(se, radius)
    
def complement(a):
    return a.max()-a

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
    """Suppress all minima that are shallower than thresh."""
    maxval = a.max()
    ainv = maxval-a
    return maxval - morphological_reconstruction(ainv-thresh, ainv)

imhmin = hminima

def remove_small_connected_components(a, min_size=64, in_place=False):
    original_dtype = a.dtype
    if a.dtype == bool:
        a = label(a)[0]
    elif not in_place:
        a = a.copy()
    if min_size == 0: # shortcut for efficiency
        return a
    component_sizes = bincount(a.ravel())
    too_small = component_sizes < min_size
    too_small_locations = too_small[a]
    a[too_small_locations] = 0
    return a.astype(original_dtype)

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

def refined_seeding(a, maximum_height=0, grey_close_radius=1, 
    binary_open_radius=1, binary_close_radius=1, minimum_size=0):
    """Perform morphological operations to get good segmentation seeds."""
    if grey_close_radius > 0:
        strel = diamond_se(grey_close_radius, a.ndim)
        a = grey_closing(a, footprint=strel)
    s = (a <= maximum_height)
    if binary_open_radius > 0:
        strel = diamond_se(binary_open_radius, s.ndim)
        s = binary_opening(s, structure=strel)
    if binary_close_radius > 0:
        strel = diamond_se(binary_close_radius, s.ndim)
        s = binary_closing(s, structure=strel)
    s = remove_small_connected_components(s, minimum_size)
    return label(s)[0]

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

def split_exclusions(image, labels, exclusions, dilation=0, connectivity=1):
    """Ensure that no segment in 'labels' overlaps more than one exclusion."""
    labels = labels.copy()
    cur_label = labels.max() + 1
    dilated_exclusions = exclusions.copy()
    foot = generate_binary_structure(exclusions.ndim, connectivity)
    for i in range(dilation):
        dilated_exclusions = grey_dilation(exclusions, footprint=foot)
    while True:
        hashed = labels * (exclusions.max() + 1) + exclusions
        hashed[exclusions == 0] = 0
        violations = bincount(hashed.ravel()) > 1
        violations[0] = False
        if sum(violations) == 0:
            break
        offending_label = labels[violations[hashed]][0]
        offended_exclusion = exclusions[violations[hashed]][0]
        mask = labels == offending_label
        seeds, n = label(mask * (dilated_exclusions == offended_exclusion))
        seeds[seeds > 1] += cur_label
        cur_label += n-1
        seeds[seeds == 1] = offending_label
        labels[mask] = watershed(image, seeds, connectivity, mask)[mask]
    return labels


def watershed(a, seeds=None, connectivity=1, mask=None, smooth_thresh=0.0, 
        smooth_seeds=False, minimum_seed_size=0, dams=False,
        show_progress=False):
    """Perform the watershed algorithm of Vincent & Soille (1991)."""
    seeded = seeds is not None
    sel = generate_binary_structure(a.ndim, connectivity)
    if smooth_thresh > 0.0:
        b = hminima(a, smooth_thresh)
    if seeded:
        if smooth_seeds:
            seeds = binary_opening(seeds, sel)
        b = impose_minima(a, seeds.astype(bool), connectivity)
    else:
        seeds = regional_minima(a, connectivity)
        b = a
    if seeds.dtype == bool:
        ws = label(seeds, sel)[0]
    else:
        ws = seeds
    if skimage_available and not dams:
        return skimage.morphology.watershed(a, seeds, sel, None, mask)
    levels = unique(b)
    a = pad(a, a.max()+1)
    b = pad(b, b.max()+1)
    ar = a.ravel()
    br = b.ravel()
    ws = pad(ws, 0)
    wsr = ws.ravel()
    current_label = 0
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
        axes = range(ar.ndim)
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
    for i in xrange(ar.ndim):
        ar = ar.swapaxes(0,i)
        ar = ar[skinsize:-skinsize]
        ar = ar.swapaxes(0,i)
    return ar.copy()

def surfaces(ar, skinsize=1):
    s = []
    for i in xrange(ar.ndim):
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

def get_neighbor_idxs(ar, idxs, connectivity=1):
    if isscalar(idxs): # in case only a single idx is given
        idxs = [idxs]
    idxs = array(idxs) # in case a list or other array-like is given
    strides = array(ar.strides)/ar.itemsize
    if connectivity == 1: 
        steps = (strides, -strides)
    else:
        steps = []
        for i in range(1,connectivity+1):
            prod = array(list(itertools.product(*([[1,-1]]*i))))
            i_strides = array(list(itertools.combinations(strides,i))).T
            steps.append(prod.dot(i_strides).ravel())
    return idxs[:,newaxis] + concatenate(steps)

def orphans(a):
    """Find all the segments that do not touch the volume boundary.
    
    This function differs from agglo.Rag.orphans() in that it does not use the
    graph, but rather computes orphans directly from a volume.
    """
    return setdiff1d(
            unique(a), unique(concatenate([s.ravel() for s in surfaces(a)]))
            )

def non_traversing_segments(a):
    """Find segments that enter the volume but do not leave it elsewhere."""
    if a.all():
        a = damify(a)
    surface = hollowed(a)
    surface_ccs = label(surface)[0]
    idxs = flatnonzero(surface)
    pairs = unique(zip(surface.ravel()[idxs], surface_ccs.ravel()[idxs]))
    return flatnonzero(bincount(pairs.astype(int)[:,0])==1)

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
