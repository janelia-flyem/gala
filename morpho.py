

from numpy import shape, reshape, zeros, ones, double, uint32, array, unique, where, unravel_index, ceil, floor, prod, cumprod, concatenate
import itertools
from collections import deque as queue
from scipy.ndimage import filters, measurements
#from scipy.spatial.distance import cityblock as manhattan_distance

zero3d = array([0,0,0])

def manhattan_distance(a, b):
    return sum(abs(a-b))

def diamondse(sz, dimension):
    d = floor(sz/2)+1
    sz = sz * ones(dimension)
    se = zeros(sz)
    ctr = floor(sz/2)
    for i in [array(unravel_index(j,sz)) for j in range(se.size)]:
        if manhattan_distance(i, ctr) < d:
            se[tuple(i)] = 1
    return se


def watershed3d(a):
    if len(a.shape) == 2:
        a = a.reshape([1, a.shape[0], a.shape[1]])
    padded_a = a.max()*ones(map(lambda x: x+2, a.shape), a.dtype)
    padded_a[1:-1,1:-1,1:-1] = a
    a = padded_a
    border_idxs = get_border_idxs(a.shape)
    sel = diamondse(3, len(a.shape))
    ws = zeros(shape(a), uint32)
    maxlabel = uint32(-1)
    ws.ravel()[border_idxs] = maxlabel
    current_label = 0
    neighbors = build_neighbors_array(a)
    level_pixels = build_levels_dict(a)
    for level in sorted(level_pixels.keys()):
        idxs_adjacent_to_labels = queue([idx for idx in level_pixels[level] if
                idx not in border_idxs and any(ws.ravel()[neighbors[idx]])])
        while len(idxs_adjacent_to_labels) > 0:
            idx = idxs_adjacent_to_labels.popleft()
            adj_labels = unique([l for l in ws.ravel()[neighbors[idx]]
                                                if l > 0 and l != maxlabel])
            if len(adj_labels) > 1:
                ws.ravel()[idx] = maxlabel # build a dam
            elif len(adj_labels) == 1 and ws.ravel()[idx] == 0:
                ws.ravel()[idx] = adj_labels[0]
                idxs_adjacent_to_labels.extend([p for p in neighbors[idx] if
                                ws.ravel()[p] == 0 and a.ravel()[p] == level])
        new_labels, num_new = measurements.label((ws == 0) * (a == level), sel)
        new_labels = (current_label + new_labels) * (new_labels != 0)
        current_label += num_new
        ws += new_labels
    ws[ws==maxlabel] = 0
    return ws[1:-1,1:-1,1:-1]

def get_border_idxs(ashape):
    y = zeros(ashape, dtype=bool)
    y[0,:,:] = y[:,0,:] = y[:,:,0] = y[ashape[0]-1,:,:] = y[:,ashape[1]-1,:] \
        = y[:,:,ashape[2]-1] = True
    return where(y.ravel())[0]

def build_levels_dict(a):
    return dict( ((l, where(a.ravel()==l)[0]) for l in unique(a)) )

def build_neighbors_array(ar):
    d = len(ar.shape)
    neighbors_array = zeros([ar.size, 2*d], uint32)
    steps = array(ar.strides)/ar.itemsize
    steps = concatenate((steps, -steps))
    for i in range(neighbors_array.shape[0]):
        neighbors_array[i,:] = i+steps
    return neighbors_array

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
