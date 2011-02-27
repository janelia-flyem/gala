
import os

from fnmatch import filter as fnfilter
from os.path import split as split_path, join as join_path
from Image import open as imopen
from numpy import asarray, uint8, uint16, uint32, zeros

def read_image_stack(fn, *args):
    d, fn = split_path(fn)
    if len(d) == 0: d = '.'
    if fn[-4:] == '.png':
        fns = fnfilter(os.listdir(d), fn)
        ims = (imopen(join_path(d,fn)) for fn in sorted(fns))
        ars = (asarray(im) for im in ims)
        w, h = imopen(join_path(d,fns[0])).size
        dtype = asarray(imopen(join_path(d,fns[0]))).dtype
        stack = zeros([h,w,len(fns)], dtype)
        for i, im in enumerate(ars):
            stack[:,:,i] = im
    return stack
