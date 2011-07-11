
import os
import argparse

import h5py, Image, numpy

from fnmatch import filter as fnfilter
from os.path import split as split_path, join as join_path
from numpy import array, asarray, uint8, uint16, uint32, zeros, squeeze, \
    fromstring, ndim
import numpy as np

arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Image IO options')
arggroup.add_argument('-I', '--invert-image', action='store_true',
    default=False,
    help='Invert the probabilities before segmenting.'
)
arggroup.add_argument('-m', '--median-filter', action='store_true', 
    default=False, help='Run a median filter on the input image.'
)

def read_image_stack(fn, *args, **kwargs):
    """Read a 3D volume of images in .png or .h5 format into a numpy.ndarray.

    The format is automatically detected from the (first) filename.

    A 'crop' keyword argument is supported, as a list of 
    [xmax, xmin, ymax, ymin, zmax, zmin].

    If reading in .h5 format, keyword arguments are passed through to
    read_h5_stack().
    """
    d, fn = split_path(fn)
    if len(d) == 0: d = '.'
    if kwargs.has_key('crop'):
        crop = kwargs['crop']
        if len(crop) == 4:
            crop.extend([None,None])
    else:
        crop = [None,None,None,None,None,None]
        kwargs['crop'] = crop
    if fn.endswith('.png'):
        xmin, xmax, ymin, ymax, zmin, zmax = crop
        if len(args) > 0 and type(args[0]) == str and args[0].endswith('png'):
            fns = [fn] + [split_path(f)[1] for f in args]
        else:
            fns = fnfilter(os.listdir(d), fn)
        fns.sort()
        fns = fns[zmin:zmax]
        ims = (Image.open(join_path(d,fn)) for fn in fns)
        ars = (asarray(im) for im in ims)
        w, h = asarray(Image.open(join_path(d,fns[0])))\
                                                [xmin:xmax,ymin:ymax].shape
        dtype = asarray(Image.open(join_path(d,fns[0]))).dtype
        stack = zeros([w,h,len(fns)], dtype)
        for i, im in enumerate(ars):
            stack[:,:,i] = im[xmin:xmax,ymin:ymax]
    if fn.endswith('.h5'):
        stack = read_h5_stack('/'.join([d,fn]), *args, **kwargs)
    return squeeze(stack)

def read_image_stack_single_arg(fn):
    """Read an image stack and print exceptions as they occur.
    
    argparse.ArgumentParser() subsumes exceptions when they occur in the 
    argument type, masking lower-level errors. This function prints out the
    error before propagating it up the stack.
    """
    try:
        return read_image_stack(fn)
    except Exception as err:
        print err
        raise

shiv_type_elem_dict = {
    0:np.int8, 1:np.uint8, 2:np.int16, 3:np.uint16,
    4:np.int32, 5:np.uint32, 6:np.int64, 7:np.uint64,
    8:np.float32, 9:np.float64
}

def read_shiv_raw_stack(ws_fn, sp2body_fn):
    ws = read_shiv_raw_array(ws_fn)
    sp2b = read_shiv_raw_array(sp2body_fn)[1]
    ar = sp2b[ws]
    return remove_merged_boundaries(ar)

import morpho
def remove_merged_boundaries(ar):
    arp = morpho.pad(ar, [0,ar.max()+1])
    arpr = arp.ravel()
    zero_idxs = (arpr == 0).nonzero()[0]
    ns = arpr[morpho.get_neighbor_idxs(arp, zero_idxs)]
    ns_compl = ns.copy()
    ns_compl[ns==0] = ns.max()+1
    merged_boundaries = (ns.max(axis=1) == ns_compl.min(axis=1)).nonzero()[0]
    arpr[zero_idxs[merged_boundaries]] = ns.max(axis=1)[merged_boundaries]
    return morpho.juicy_center(arp, 2)

def read_shiv_raw_array(fn):
    fin = open(fn, 'rb')
    type_elem_code = fromstring(fin.read(4), uint8)[1]
    ar_dtype = shiv_type_elem_dict[type_elem_code]
    ar_ndim = fromstring(fin.read(4), uint8)[0]
    ar_shape = fromstring(fin.read(ar_ndim*4), uint32)
    ar = fromstring(fin.read(), ar_dtype).reshape(ar_shape, order='F')
    return ar


def read_h5_stack(fn, *args, **kwargs):
    """Read a volume in HDF5 format into numpy.ndarray.

    Accepts keyword arguments 'group' (the group in the HDF5 file containing
    the array information; default: 'stack') and 'crop' (format as in 
    read_image_stack())
    """
    if len(args) > 0:
        group = args[0]
    elif kwargs.has_key('group'):
        group = kwargs['group']
    else:
        group = 'stack'
    if kwargs.has_key('crop'):
        crop = kwargs['crop']
    else:
        crop = [None,None,None,None,None,None]
    xmin, xmax, ymin, ymax, zmin, zmax = crop
    dset = h5py.File(fn, 'r')
    a = dset[group]
    if ndim(a) == 2:
        a = a[xmin:xmax,ymin:ymax]
    elif ndim(a) == 3:
        a = a[xmin:xmax,ymin:ymax,zmin:zmax]
    return array(a)

def write_image_stack(npy_vol, fn, **kwargs):
    """Write a numpy.ndarray 3D volume to a stack of images or an HDF5 file.
    """
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
    """Write a numpy.ndarray 3D volume to a stack of .png images.

    Only 8-bit and 16-bit single-channel images are currently supported.
    """
    if numpy.max(npy_vol) < 2**16:
        mode = 'I'
        npy_vol = uint16(npy_vol)
    else:
        mode = 'RGBA'
        npy_vol = uint32(npy_vol)
    for z in range(npy_vol.shape[2]):
        Image.fromarray(npy_vol[:,:,z], mode).save(fn % z)

def write_h5_stack(npy_vol, fn, **kwargs):
    """Write a numpy.ndarray 3D volume to an HDF5 file.

    The following keyword arguments are supported:
    - 'group': the group into which to write the array. (default: 'stack')
    - 'compression': The type of compression. (default: None)
    - 'chunks': Chunk size in the HDF5 file. (default: None)
    """
    if not kwargs.has_key('compression'):
        kwargs['compression'] = None
    if not kwargs.has_key('chunks'):
        kwargs['chunks'] = None
    try:
        group = kwargs['group']
        del kwargs['group']
    except KeyError:
        group = 'stack'
    fout = h5py.File(fn, 'a')
    if group in fout:
        del fout[group]
    fout.create_dataset(group, data=npy_vol, **kwargs)


#######################
# Image visualization #
#######################

def show_boundary(seg):
    """Show the boundary (0-label) in a 2D segmentation (labeling)."""
    seg = seg.squeeze()
    boundary = 255*(1-seg.astype(bool)).astype(uint8)
    Image.fromarray(boundary).show()

def show_image(ar):
    """Display an image from an array."""
    Image.fromarray(ar.squeeze()).show()
