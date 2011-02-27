

from numpy import shape, reshape, zeros, double, array
import itertools

zero3d = array([0,0,0])

def watershed(a):
    try:
        xmax,ymax,zmax = shape(a)
    except ValueError:
        xmax,ymax = shape(a)
        zmax = 1
    ws = zeros(shape(a), dtype=double)
    labels = itertools.count(1)
    c = sorted([(a[i], i) for i in ((x,y,z) for x in range(xmax)
                                         for y in range(ymax)
                                         for z in range(zmax))])
    # precompute steps and arrayshape for efficiency inside loop
    steps = map(array, [(0,0,1),(0,1,0),(1,0,0)])
    arrayshape = array(a.shape) 
    for vox, idx in c:
        ns = neighbor_idxs(idx, steps, arrayshape)
        nlabels = list(set([l for l in [ws[n] for n in ns] if l > 0]))
        if len(nlabels) == 0:
            ws[idx] = labels.next()
        elif len(nlabels) == 1:
            ws[idx] = nlabels[0]
    return ws

def neighbor_idxs(idx, steps, arrayshape):
    idx = array(idx)
    neighbors = itertools.chain(*[[idx+step, idx-step] for step in steps])
    return [tuple(n) for n in neighbors if all(n >= zero3d) and 
                                           all(n < arrayshape)]
