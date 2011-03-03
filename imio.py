
import os

import h5py, Image, numpy

from fnmatch import filter as fnfilter
from os.path import split as split_path, join as join_path
from numpy import asarray, uint8, uint16, uint32, zeros

def read_image_stack(fn, *args):
    d, fn = split_path(fn)
    if len(d) == 0: d = '.'
    if fn.endswith('.png'):
        fns = fnfilter(os.listdir(d), fn)
        ims = (Image.open(join_path(d,fn)) for fn in sorted(fns))
        ars = (asarray(im) for im in ims)
        w, h = Image.open(join_path(d,fns[0])).size
        dtype = asarray(Image.open(join_path(d,fns[0]))).dtype
        stack = zeros([h,w,len(fns)], dtype)
        for i, im in enumerate(ars):
            stack[:,:,i] = im
    return stack

def write_image_stack(npy_vol, fn, **kwargs):
    fn = os.path.expanduser(fn)
    if fn.endswith('.png'):
        write_png_image_stack(npy_vol, fn, **kwargs)
    elif fn.endswith('.h5'):
        write_h5_stack(npy_vol, fn, **kwargs)
    elif fn.endswith('.npy'):
        write_npy_image_stack(npy_vol, fn, **kwargs)
    else:
        raise ValueError('Image format not supported: ' + fn + '\n')


def write_png_image_stack(npy_vol, fn, **kwargs):
    if numpy.max(npy_vol) < 2**16:
        mode = 'I'
        npy_vol = uint16(npy_vol)
    else:
        mode = 'RGBA'
        npy_vol = uint32(npy_vol)
    for z in range(npy_vol.shape[2]):
        Image.fromarray(npy_vol[:,:,z], mode).save(fn % z)

def write_h5_stack(npy_vol, fn, **kwargs):
    if not kwargs.has_key('compression'):
        kwargs['compression'] = 'lzf'
    try:
        group = kwargs['group']
    except KeyError:
        group = 'stack'
    fout = h5py.File(fn, 'w')
    fout.create_dataset(group, data=npy_vol, **kwargs)
