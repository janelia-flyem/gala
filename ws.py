

from numpy import shape, reshape, zeros, ones, double, uint32, array, unique, where
import itertools
from collections import deque as queue
from scipy.ndimage import filters, measurements

zero3d = array([0,0,0])

def watershed3d(a):
    if len(a.shape) == 2:
        a = a.reshape([a.shape[0], a.shape[1], 1])
    sel = zeros([3,3,3])
    sel[:,1,1] = 1
    sel[1,:,1] = 1
    sel[1,1,:] = 1
    ws = zeros(shape(a), uint32)
    maxlabel = uint32(-1)
    current_label = 0
    neighbors = build_neighbors_dict(shape(a))
    level_pixels = build_levels_dict(a)
    for level in sorted(level_pixels.keys()):
        idxs_adjacent_to_labels = queue([idx for idx in level_pixels[level] if 
                                    any(ws[zip(*neighbors[idx])])])
        while len(idxs_adjacent_to_labels) > 0:
            idx = idxs_adjacent_to_labels.popleft()
            adj_labels = list(set([l for l in ws[zip(*neighbors[idx])]
                                                if l > 0 and l != maxlabel]))
            if len(adj_labels) > 1:
                ws[idx] = maxlabel # build a dam
            elif len(adj_labels) == 1 and ws[idx] == 0:
                ws[idx] = adj_labels[0]
                idxs_adjacent_to_labels.extend([p for p in neighbors[idx] if
                                                ws[p] == 0 and a[p] == level])
        new_labels, num_new = measurements.label((ws == 0) * (a == level), sel)
        new_labels = (current_label + new_labels) * (new_labels != 0)
        current_label += num_new
        ws += new_labels
    ws[ws==maxlabel] = 0
    return ws

def build_levels_dict(a):
    return dict( ((l, zip(*where(a==l))) for l in unique(a)) )

def build_neighbors_dict(arshape):
    xmax,ymax,zmax = arshape
    border_idxs = itertools.chain(
        ((x,y,z) for x in [0,xmax-1] for y in range(ymax) for z in range(zmax)),
        ((x,y,z) for x in range(1,xmax-1) for y in [0,ymax-1] for z in range(zmax)),
        ((x,y,z) for x in range(1,xmax-1) for y in range(1,ymax-1) for z in [0,zmax-1])
    )
    interior_idxs = ((x,y,z) for x in range(1,xmax-1) for y in range(1,ymax-1)
        for z in range(1,zmax-1))
    # precompute steps and arrayshape for efficiency inside loop
    steps = map(array, [(0,0,1),(0,1,0),(1,0,0)])
    d1 = dict( ((idx, neighbor_idxs(idx, steps, arshape)) 
                                                    for idx in border_idxs) )
    d2 = dict( ((idx, neighbor_idxs_no_check(idx, steps, arshape)) 
                                                    for idx in interior_idxs) )
    return dict(d1, **d2) # concatenate the two dictionaries


def neighbor_idxs(idx, steps, arrayshape):
    idx = array(idx)
    neighbors = itertools.chain(*[[idx+step, idx-step] for step in steps])
    return [tuple(n) for n in neighbors if all(n >= zero3d) and 
                                           all(n < arrayshape)]

def neighbor_idxs_no_check(idx, steps, arrayshape):
    idx = array(idx)
    neighbors = itertools.chain(*[[idx+step, idx-step] for step in steps])
    return map(tuple, neighbors)
