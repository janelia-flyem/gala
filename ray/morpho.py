#!/usr/bin/env python

import sys, os, argparse
from numpy import   shape, reshape, \
                    array, zeros, zeros_like, ones, ones_like, arange, \
                    double, \
                    int8, int16, int32, int64, uint8, uint16, uint32, uint64, \
                    iinfo, isscalar, \
                    unique, \
                    where, unravel_index, newaxis, \
                    ceil, floor, prod, cumprod, \
                    concatenate, \
                    ndarray, minimum, bincount, dot, nonzero, concatenate, \
                    setdiff1d, inf
import itertools
import re
from collections import defaultdict, deque as queue
from scipy.ndimage import filters, grey_dilation, generate_binary_structure, \
        maximum_filter, minimum_filter
from scipy.ndimage import distance_transform_cdt
from scipy.ndimage.measurements import label, find_objects
from scipy.ndimage.morphology import binary_opening, binary_dilation, \
    generate_binary_structure, iterate_structure
#from scipy.spatial.distance import cityblock as manhattan_distance
import iterprogress as ip
import imio

zero3d = array([0,0,0])

arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Morphological operations options')
arggroup.add_argument('-S', '--save-watershed', metavar='FILE',
    help='Write the watershed result to FILE (overwrites).'
)
arggroup.add_argument('-w', '--watershed', metavar='WS_FN',
    type=imio.single_arg_read_image_stack,
    help='Use a precomputed watershed volume from file.'
)
arggroup.add_argument('--seed', metavar='FN', 
    type=imio.single_arg_read_image_stack,
    help='''use the volume in FN to seed the watershed. By default, connected
        components of 0-valued pixels will be used as the seeds.'''
)
arggroup.add_argument('--build-dams', default=True, action='store_true',
    help='''Build dams when two or more basins collide. (default)'''
)
arggroup.add_argument('--no-dams', action='store_false', dest='build_dams',
    help='''Don't build dams in the watershed. Every pixel has a label.'''
)

def manhattan_distance(a, b):
    return sum(abs(a-b))

def diamondse(radius, dimension):
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
    if a.dtype == bool:
        a = label(a)[0]
    elif not in_place:
        a = a.copy()
    component_sizes = bincount(a.ravel())
    too_small = component_sizes < min_size
    too_small_locations = too_small[a]
    a[too_small_locations] = 0
    return a

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

def watershed(a, seeds=None, smooth_thresh=0.0, smooth_seeds=False, 
        minimum_seed_size=0, dams=True, show_progress=False, connectivity=1):
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
    levels = unique(a)
    a = pad(a, a.max()+1)
    b = pad(b, b.max()+1)
    ar = a.ravel()
    br = b.ravel()
    ws = pad(ws, 0)
    wsr = ws.ravel()
    maxlabel = iinfo(ws.dtype).max
    current_label = 0
    neighbors = build_neighbors_array(a, connectivity)
    level_pixels = build_levels_dict(a)
    if show_progress: wspbar = ip.StandardProgressBar('Watershed...')
    else: wspbar = ip.NoProgressBar()
    for i, level in ip.with_progress(enumerate(levels), 
                                            pbar=wspbar, length=len(levels)):
        idxs_adjacent_to_labels = queue([idx for idx in level_pixels[level] if
                                            any(wsr[neighbors[idx]])])
        while len(idxs_adjacent_to_labels) > 0:
            idx = idxs_adjacent_to_labels.popleft()
            if wsr[idx]> 0: continue # in case we already processed it
            nidxs = neighbors[idx] # neighbors
            lnidxs = nidxs[
                ((wsr[nidxs] != 0) * (wsr[nidxs] != maxlabel)).astype(bool)
            ] # labeled neighbors
            adj_labels = unique(wsr[lnidxs])
            if len(adj_labels) > 1 and dams: # build a dam
                wsr[idx] = maxlabel 
            elif len(adj_labels) >= 1: # assign a label
                wsr[idx] = wsr[lnidxs][ar[lnidxs].argmin()]
                idxs_adjacent_to_labels.extend(nidxs[((wsr[nidxs] == 0) * 
                                    (br[nidxs] == level)).astype(bool) ])
    if dams:
        ws[ws==maxlabel] = 0
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

def seg_to_bdry(seg, connectivity=1):
    """Given a borderless segmentation, return the boundary map."""
    strel = generate_binary_structure(seg.ndim, connectivity)
    return maximum_filter(seg,footprint=strel) != minimum_filter(seg,footprint=strel)
    
def undam(seg):
    """ Assign zero-dams to nearest non-zero region. """
    bdrymap = seg==0
    k = distance_transform_cdt(bdrymap, return_indices=True)
    ind = nonzero(bdrymap.ravel())[0]
    closest_sub = concatenate([i.ravel()[:,newaxis] for i in k[1]],axis=1)
    closest_sub = closest_sub[ind,:]
    closest_ind = [dot(bdrymap.strides, i)/bdrymap.itemsize for i in closest_sub]
    sp = seg.shape
    seg = seg.ravel()
    seg[ind] = seg[closest_ind]
    seg = reshape(seg, sp)
    return seg

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        parents=[arguments],
        description='Watershed transform an image volume.'
    )
    parser.add_argument('fin', nargs='+',
        help='input image (png or h5 volume)'
    )
    parser.add_argument('fout', help='output filename (.h5)')
    parser.add_argument('-I', '--invert-image', action='store_true',
        help='invert the image before applying watershed'
    )
    parser.add_argument('-m', '--median-filter', action='store_true',
        help='Apply a median filter before watershed.'
    )
    parser.add_argument('-g', '--gaussian-filter', type=float, metavar='SIGMA',
        help='Apply a gaussian filter before watershed.'
    )
    parser.add_argument('-P', '--show-progress', action='store_true',
        help='Show a progress bar for the watershed transform.'
    )
    args = parser.parse_args()

    v = imio.read_image_stack(*args.fin)
    if args.invert_image:
        v = v.max() - v
    if args.median_filter:
        v = filters.median_filter(v, 3)
    if args.gaussian_filter is not None:
        v = filters.gaussian_filter(v, args.gaussian_filter)
    if args.seed is not None:
        args.seed, _ = label(args.seed == 0, diamondse(3, args.seed.ndim))
    ws = watershed(v, seeds=args.seed, dams=args.build_dams,
                                            show_progress=args.show_progress)
    if os.access(args.fout, os.F_OK):
        os.remove(args.fout)
    imio.write_h5_stack(ws, args.fout)
