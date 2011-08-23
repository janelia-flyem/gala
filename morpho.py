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
                    ndarray
import itertools
from collections import deque as queue
from scipy.ndimage import filters
from scipy.ndimage.measurements import label
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
    type=imio.read_image_stack_single_arg,
    help='Use a precomputed watershed volume from file.'
)
arggroup.add_argument('--seed', metavar='FN', 
    type=imio.read_image_stack_single_arg,
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

def diamondse(sz, dimension):
    d = floor(sz/2)+1 # ceil has problems if sz.dtype is int
    sz = sz * ones(dimension)
    se = zeros(sz)
    ctr = floor(sz/2)
    for i in [array(unravel_index(j,sz)) for j in range(se.size)]:
        if manhattan_distance(i, ctr) < d:
            se[tuple(i)] = 1
    return se


def watershed(a, seeds=None, dams=True, show_progress=False):
    seeded = seeds is not None
    if not seeded:
        ws = zeros(shape(a), uint32)
    else:
        if seeds.dtype == bool:
            seeds = label(seeds)[0]
        ws = seeds
    levels = unique(a)
    a = pad(a, a.max()+1)
    ar = a.ravel()
    arc = ar.copy() if seeded else ar
    ws = pad(ws, 0)
    wsr = ws.ravel()
    maxlabel = iinfo(ws.dtype).max
    sel = diamondse(3, a.ndim)
    current_label = 0
    neighbors = build_neighbors_array(a)
    level_pixels = build_levels_dict(a)
    if show_progress:
        def with_progress(it, *args, **kwargs): 
            return ip.with_progress(
                it, *args, pbar=ip.StandardProgressBar('Watershed...'), **kwargs
            )
    else:
        def with_progress(it, *args, **kwargs):
            return ip.with_progress(it, *args,pbar=ip.NoProgressBar(),**kwargs)
    for i, level in with_progress(enumerate(levels), length=len(levels)):
        idxs_adjacent_to_labels = queue([idx for idx in level_pixels[level] if
                                            any(wsr[neighbors[idx]])])
        while len(idxs_adjacent_to_labels) > 0:
            idx = idxs_adjacent_to_labels.popleft()
            nidxs = neighbors[idx] # neighbors
            lnidxs = nidxs[
                ((wsr[nidxs] != 0) * (wsr[nidxs] != maxlabel)).astype(bool)
            ] # labeled neighbors
            adj_labels = unique(wsr[lnidxs])
            if len(adj_labels) > 1 and dams: # build a dam
                wsr[idx] = maxlabel 
            else: # assign a label
                wsr[idx] = wsr[lnidxs][arc[lnidxs].argmin()]
                idxs_adjacent_to_labels.extend(nidxs[((wsr[nidxs] == 0) * 
                                    (ar[nidxs] == level)).astype(bool) ])
        if seeded:
            if i+1 < len(levels):
                not_adj = where((wsr == 0) * (ar == level))[0]
                level_pixels[levels[i+1]].extend(not_adj)
                ar[not_adj] = levels[i+1]
        else:
            new_labels, num_new = label((ws == 0) * (a == level), sel)
            new_labels = (current_label + new_labels) * (new_labels != 0)
            current_label += num_new
            ws += new_labels
    if dams:
        ws[ws==maxlabel] = 0
    return juicy_center(ws)

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
        return pad(ar2, vals[1:])
        
def juicy_center(ar, skinsize=1):
    for i in xrange(ar.ndim):
        ar = ar.swapaxes(0,i)
        ar = ar[skinsize:-skinsize]
        ar = ar.swapaxes(0,i)
    return ar

def build_levels_dict(a):
    return dict( ((l, list(where(a.ravel()==l)[0])) for l in unique(a)) )

def build_neighbors_array(ar):
    idxs = arange(ar.size, dtype=uint32)
    return get_neighbor_idxs(ar, idxs)

def get_neighbor_idxs(ar, idxs):
    if isscalar(idxs): # in case only a single idx is given
        idxs = [idxs]
    idxs = array(idxs) # in case a list or other array-like is given
    steps = array(ar.strides)/ar.itemsize
    steps = concatenate((steps, -steps))
    return idxs[:,newaxis] + steps


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
