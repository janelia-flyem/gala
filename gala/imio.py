# built-ins
import os
import json
from os.path import split as split_path, join as join_path
from fnmatch import filter as fnfilter
import logging
import itertools as it
import subprocess
import tempfile as tmp

# libraries
import h5py
try:
    import Image
except:
    from PIL import Image
try:
    from pylibtiff import TIFF
except:
    print "pylibtiff not available: http://www.lfd.uci.edu/~gohlke/pythonlibs/#pylibtiff"


from scipy.ndimage.measurements import label

from numpy import array, uint8, uint16, uint32, uint64, zeros, \
    zeros_like, squeeze, fromstring, ndim, concatenate, newaxis, swapaxes, \
    savetxt, unique, double, cumsum, ndarray
import numpy as np

from skimage.io.collection import alphanumeric_key
from skimage.io import imread

# local files
import evaluate
import morpho

### Auto-detect file format

supported_image_extensions = ['png', 'tif', 'tiff', 'jpg', 'jpeg']

def read_image_stack(fn, *args, **kwargs):
    """Read a 3D volume of images in image or .h5 format into a numpy.ndarray.

    This function attempts to automatically determine input file types and
    wraps specific image-reading functions.

    Parameters
    ----------
    fn : filename (string)
        A file path or glob pattern specifying one or more valid image files.
        The file format is automatically determined from this argument.

    *args : filenames (string, optional)
        More than one positional argument will be interpreted as a list of
        filenames pointing to all the 2D images in the stack.

    **kwargs : keyword arguments (optional)
        Arguments to be passed to the underlying functions. A 'crop'
        keyword argument is supported, as a list of length 6:
        [xmin, xmax, ymin, ymax, zmin, zmax]. Use 'None' for no crop in
        that coordinate.

    Returns
    -------
    stack : 3-dimensional numpy ndarray

    Notes
    -----
        If reading in .h5 format, keyword arguments are passed through to
        read_h5_stack().

        Automatic file type detection may be deprecated in the future.
    """
    # TODO: Refactor.  Rather than have implicit designation of stack format
    # based on filenames (*_boundpred.h5, etc), require explicit parameters
    # in config JSON files.
    if os.path.isdir(fn):
        fn += '/'
    d, fn = split_path(os.path.expanduser(fn))
    if len(d) == 0: d = '.'
    crop = kwargs.get('crop', [None]*6)
    if crop is None:
        crop = [None]*6
    if len(crop) == 4: crop.extend([None]*2)
    elif len(crop) == 2: crop = [None]*4 + crop
    kwargs['crop'] = crop
    if any([fn.endswith(ext) for ext in supported_image_extensions]):
        # image types, such as a set of pngs or a multi-page tiff
        xmin, xmax, ymin, ymax, zmin, zmax = crop
        if len(args) > 0 and type(args[0]) == str and args[0].endswith(fn[-3:]):
            # input is a list of filenames
            fns = [fn] + [split_path(f)[1] for f in args]
        else:
            # input is a filename pattern to match
            fns = fnfilter(os.listdir(d), fn)
        if len(fns) == 1 and fns[0].endswith('.tif'):
            stack = read_multi_page_tif(join_path(d,fns[0]), crop)
        else:
            fns.sort(key=alphanumeric_key) # sort filenames numerically
            fns = fns[zmin:zmax]
            im0 = imread(join_path(d, fns[0]))
            ars = (imread(join_path(d, fn)) for fn in fns)
            im0 = im0[xmin:xmax, ymin:ymax]
            dtype = im0.dtype
            stack = zeros((len(fns),)+im0.shape, dtype)
            for i, im in enumerate(ars):
                stack[i] = im[xmin:xmax,ymin:ymax]
    elif fn.endswith('_boundpred.h5') or fn.endswith('_processed.h5'):
        # Ilastik batch prediction output file
        stack = read_prediction_from_ilastik_batch(os.path.join(d,fn), **kwargs)
    elif fn.endswith('.h5'):
        # other HDF5 file
        stack = read_h5_stack(join_path(d,fn), *args, **kwargs)
    elif os.path.isfile(os.path.join(d, 'superpixel_to_segment_map.txt')):
        # Raveler export
        stack = raveler_to_labeled_volume(d, *args, **kwargs)
    return squeeze(stack)

def write_image_stack(npy_vol, fn, **kwargs):
    """Write a numpy.ndarray 3D volume to a stack of images or an HDF5 file.
    
    Parameters
    ----------
    npy_vol : numpy ndarray
        The volume to be written to disk.
    
    fn : string
        The filename to be written, or a format string when writing a 3D
        stack to a 2D format (e.g. a png image stack).
    
    **kwargs : keyword arguments
        Keyword arguments to be passed to wrapped functions. See
        corresponding docs for valid arguments.
    
    Returns
    -------
    out : None

    Examples
    --------
    >>> import numpy as np
    >>> from gala.imio import write_image_stack
    >>> im = 255 * np.array([
    ... [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
    ... [[1, 0, 1], [0, 1, 0], [1, 0, 1]]], dtype=uint8)
    >>> im.shape
    (2, 3, 3)
    >>> write_image_stack(im, 'image-example-%02i.png', axis=0)
    >>> import os
    >>> fns = sorted(filter(lambda x: x.endswith('.png'), os.listdir('.')))
    >>> fns # two 3x3 images
    ['image-example-00.png', 'image-example-01.png']
    >>> os.remove(fns[0]); os.remove(fns[1]) # doctest cleanup
    """
    fn = os.path.expanduser(fn)
    if fn.endswith('.png'):
        write_png_image_stack(npy_vol, fn, **kwargs)
    elif fn.endswith('.h5'):
        write_h5_stack(npy_vol, fn, **kwargs)
    elif fn.endswith('.vtk'):
        write_vtk(npy_vol, fn, **kwargs)
    else:
        raise ValueError('Image format not supported: ' + fn + '\n')

### Standard image formats (png, tiff, etc.)

def pil_to_numpy(img):
    """Convert an Image object to a numpy array.
    
    Parameters
    ----------
    img : Image object (from the Python Imaging Library)
    
    Returns
    -------
    ar : numpy ndarray
        The corresponding numpy array (same shape as the image)
    """
    ar = squeeze(array(img.getdata()).reshape((img.size[1], img.size[0], -1)))
    return ar

def read_multi_page_tif(fn, crop=[None]*6):
    """Read a multi-page tif file into a numpy array.
    
    Parameters
    ----------
    fn : string
        The filename of the image file being read.
    
    Returns
    -------
    ar : numpy ndarray
        The image stack in array format.

    Notes
    -----
        Currently, only grayscale images are supported.
    """
    xmin, xmax, ymin, ymax, zmin, zmax = crop
    img = Image.open(fn)
    pages = []
    if zmin is not None and zmin > 0:
        img.seek(zmin)
    eof = False
    while not eof and img.tell() != zmax:
        pages.append(pil_to_numpy(img)[...,newaxis])
        try:
            img.seek(img.tell()+1)
        except EOFError:
            eof = True
    return concatenate(pages, axis=-1)

def read_multi_page_tif_libtiff(fn):
    """Read a multi-page tif file into a numpy array.
    
    Parameters
    ----------
    fn : string
        The filename of the image file being read.
    
    Returns
    -------
    ar : numpy ndarray
        The image stack in array format.

    Notes
    -----
        Currently, only grayscale images are supported.
    """
    pages = []
    tif = TIFF.open(fn)
    for img in tif.iter_images():
        pages.append(img[..., newaxis])
    return concatenate(pages, axis=-1)

def write_png_image_stack(npy_vol, fn, axis=-1, bitdepth=None):
    """Write a numpy.ndarray 3D volume to a stack of .png images.

    Parameters
    ----------
    npy_vol : numpy ndarray, shape (M, N, P)
        The volume to be written to disk.

    fn : format string
        The file pattern to which to write the volume.

    axis : int, optional (default = -1)
        The axis along which output the images. If the input array has shape
        (M, N, P), and axis is 1, the function will write N images of shape
        (M, P) to disk. In keeping with Python convention, -1 specifies the
        last axis.

    Returns
    -------
    None : None
        No value is returned.

    Notes
    -----
        Only 8-bit and 16-bit single-channel images are currently supported.
    """
    npy_vol = swapaxes(npy_vol, 0, axis)
    fn = os.path.expanduser(fn)
    if 0 <= npy_vol.max() <= 1 and npy_vol.dtype == double:
        bitdepth = 16 if None else bitdepth
        imdtype = uint16 if bitdepth == 16 else uint8
        npy_vol = ((2**bitdepth-1)*npy_vol).astype(imdtype)
    if 1 < npy_vol.max() < 256 and bitdepth == None or bitdepth == 8:
        mode = 'L'
        mode_base = 'L'
        npy_vol = uint8(npy_vol)
    elif 256 <= np.max(npy_vol) < 2**16 and bitdepth == None or \
                                                bitdepth == 16:
        mode = 'I;16'
        mode_base = 'I'
        npy_vol = uint16(npy_vol)
    else:
        mode = 'RGBA'
        mode_base = 'RGBA'
        npy_vol = uint32(npy_vol)
    for z, pl in enumerate(npy_vol):
        im = Image.new(mode_base, pl.T.shape)
        im.fromstring(pl.tostring(), 'raw', mode)
        im.save(fn % z)

### VTK structured points array format

def extract_segments(seg, ids):
    """Get a uint8 volume containing only the specified segment ids.

    Parameters
    ----------
    seg : array of int
        The input segmentation.
    ids : list of int, maximum length 255
        A list of segments to extract from `seg`.

    Returns
    -------
    segs : array of uint8
        A volume with 1, 2, ..., ``len(ids)`` labels where the required
        segments were, and 0 elsewhere.

    Notes
    -----
    This function is designed to output volumes to VTK format for
    viewing in ITK-SNAP

    Examples
    --------
    >>> segments = array([[45, 45, 51, 51],
    ...                   [45, 83, 83, 51]])
    >>> extract_segments(segments, [83, 45])
    array([[2, 2, 0, 0],
           [2, 1, 1, 0]], dtype=uint8)
    """
    segs = np.zeros(seg.shape, dtype=np.uint8)
    for i, s in enumerate(ids):
        segs[seg == s] = i + 1
    return segs


numpy_type_to_vtk_string = {
    np.uint8:'unsigned_char', np.int8:'char', np.uint16:'unsigned_short',
    np.int16:'short', np.uint32:'unsigned_int', np.int32:'int',
    np.uint64:'unsigned_long', np.int64:'long', np.float32:'float',
    np.float64:'double'
}


vtk_string_to_numpy_type = \
    dict([(v,k) for k, v in numpy_type_to_vtk_string.items()])

def write_vtk(ar, fn, spacing=[1.0, 1.0, 1.0]):
    """Write 3D volume to VTK structured points format file.

    Code adapted from Erik Vidholm's writeVTK.m Matlab implementation.

    Parameters
    ----------
    ar : a numpy array, shape (M, N, P)
        The array to be written to disk.
    fn : string
        The desired output filename.
    spacing : iterable of float, optional (default: [1.0, 1.0, 1.0])
        The voxel spacing in x, y, and z.

    Returns
    -------
    None : None
        This function does not have a return value.
    """
    # write header
    f = open(fn, 'w')
    f.write('# vtk DataFile Version 3.0\n')
    f.write('created by write_vtk (Python implementation by JNI)\n')
    f.write('BINARY\n')
    f.write('DATASET STRUCTURED_POINTS\n')
    f.write(' '.join(['DIMENSIONS'] + map(str, ar.shape[-1::-1])) + '\n')
    f.write(' '.join(['ORIGIN'] + map(str, zeros(3))) + '\n')
    f.write(' '.join(['SPACING'] + map(str, spacing)) + '\n')
    f.write('POINT_DATA ' + str(ar.size) + '\n')
    f.write('SCALARS image_data ' +
                            numpy_type_to_vtk_string[ar.dtype.type] + '\n')
    f.write('LOOKUP_TABLE default\n');
    f.close()

    # write data as binary
    f = open(fn, 'ab')
    f.write(ar.data)
    f.close()

def read_vtk(fin):
    """Read a numpy volume from a VTK structured points file.

    Code adapted from Erik Vidholm's readVTK.m Matlab implementation.

    Parameters
    ----------
    fin : string
        The input filename.

    Returns
    -------
    ar : numpy ndarray
        The array contained in the file.
    """
    f = open(fin, 'r')
    num_lines_in_header = 10
    lines = [f.readline() for i in range(num_lines_in_header)]
    shape_line = [line for line in lines if line.startswith('DIMENSIONS')][0]
    type_line = [line for line in lines 
        if line.startswith('SCALARS') or line.startswith('VECTORS')][0]
    ar_shape = map(int, shape_line.rstrip('\n').split(' ')[1:])[-1::-1]
    ar_type = vtk_string_to_numpy_type[type_line.rstrip('\n').split(' ')[2]]
    ar = squeeze(fromstring(f.read(), ar_type).reshape(ar_shape+[-1]))
    return ar

### HDF5 format

def read_h5_stack(fn, group='stack', crop=[None]*6, **kwargs):
    """Read a volume in HDF5 format into numpy.ndarray.

    Parameters
    ----------
    fn : string
        The filename of the input HDF5 file.
    group : string, optional (default 'stack')
        The group within the HDF5 file containing the dataset.
    crop : list of int, optional (default '[None]*6', no crop)
        A crop to get of the volume of interest. Only available for 2D and 3D
        volumes.

    Returns
    -------
    stack : numpy ndarray
        The stack contained in fn, possibly cropped.
    """
    fn = os.path.expanduser(fn)
    dset = h5py.File(fn, 'r')
    if group not in dset:
        raise ValueError("HDF5 file (%s) doesn't have group (%s)!" % 
                            (fn, group))
    a = dset[group]
    if ndim(a) == 2:
        xmin, xmax, ymin, ymax = crop[:4]
        a = a[xmin:xmax, ymin:ymax]
    elif ndim(a) == 3:
        xmin, xmax, ymin, ymax, zmin, zmax = crop
        a = a[xmin:xmax, ymin:ymax, zmin:zmax]
    stack = array(a)
    dset.close()
    return stack

def compute_sp_to_body_map(sps, bodies):
    """Return unique (sp, body) pairs from a superpixel map and segmentation.

    Parameters
    ----------
    sps : numpy ndarray, arbitrary shape
        The superpixel (supervoxel) map.
    bodies : numpy ndarray, same shape as sps
        The corresponding segmentation.

    Returns
    -------
    sp_to_body : numpy ndarray, shape (NUM_SPS, 2)

    Notes
    -----
    No checks are made for sane inputs. This means that incorrect input,
    such as non-matching shapes, or superpixels mapping to more than one
    segment, will result in undefined behavior downstream with no warning.
    """
    sp_to_body = unique(zip(sps.ravel(), bodies.ravel())).astype(uint64)
    return sp_to_body

def write_mapped_segmentation(superpixel_map, sp_to_body_map, fn, 
                              sp_group='stack', sp_to_body_group='transforms'):
    """Write a mapped segmentation to an HDF5 file.

    Parameters
    ----------
    superpixel_map : numpy ndarray, arbitrary shape
    sp_to_body_map : numpy ndarray, shape (NUM_SPS, 2)
        A many-to-one map of superpixels to bodies (segments), specified as
        rows of (superpixel, body) pairs.
    fn : string
        The output filename.
    sp_group : string, optional (default 'stack')
        the group within the HDF5 file to store the superpixel map.
    sp_to_body_group : string, optional (default 'transforms')
        the group within the HDF5 file to store the superpixel to body map.

    Returns
    -------
    None
    """
    fn = os.path.expanduser(fn)
    fout = h5py.File(fn, 'w')
    fout.create_dataset(sp_group, data=superpixel_map)
    fout.create_dataset(sp_to_body_group, data=sp_to_body_map)
    fout.close()


def read_mapped_segmentation(fn, 
                             sp_group='stack', sp_to_body_group='transforms'):
    """Read a volume in mapped HDF5 format into a numpy.ndarray pair.

    Parameters
    ----------
    fn : string
        The filename to open.
    sp_group : string, optional (default 'stack')
        The group within the HDF5 file where the superpixel map is stored.
    sp_to_body_group : string, optional (default 'transforms')
        The group within the HDF5 file where the superpixel to body map is
        stored.

    Returns
    -------
    segmentation : numpy ndarray, same shape as 'superpixels', int type
        The segmentation induced by the superpixels and map.
    """
    sps, sp2body = read_mapped_segmentation_raw(fn, sp_group, sp_to_body_group)
    segmentation = apply_segmentation_map(sps, sp2body)
    return segmentation

def apply_segmentation_map(superpixels, sp_to_body_map):
    """Return a segmentation from superpixels and a superpixel to body map.

    Parameters
    ----------
    superpixels : numpy ndarray, arbitrary shape, int type
        A superpixel (or supervoxel) map (aka label field).
    sp_to_body_map : numpy ndarray, shape (NUM_SUPERPIXELS, 2), int type
        An array of (superpixel, body) map pairs.

    Returns
    -------
    segmentation : numpy ndarray, same shape as 'superpixels', int type
        The segmentation induced by the superpixels and map.
    """
    forward_map = np.zeros(sp_to_body_map[:, 0].max() + 1,
                           sp_to_body_map.dtype)
    forward_map[sp_to_body_map[:, 0]] = sp_to_body_map[:, 1]
    segmentation = forward_map[superpixels]
    return segmentation

def read_mapped_segmentation_raw(fn, 
                             sp_group='stack', sp_to_body_group='transforms'):
    """Read a volume in mapped HDF5 format into a numpy.ndarray pair.

    Parameters
    ----------
    fn : string
        The filename to open.
    sp_group : string, optional (default 'stack')
        The group within the HDF5 file where the superpixel map is stored.
    sp_to_body_group : string, optional (default 'transforms')
        The group within the HDF5 file where the superpixel to body map is
        stored.

    Returns
    -------
    sp_map : numpy ndarray, arbitrary shape
        The superpixel (or supervoxel) map.
    sp_to_body_map : numpy ndarray, shape (NUM_SUPERPIXELS, 2)
        The superpixel to body (segment) map, as (superpixel, body) pairs.
    """
    fn = os.path.expanduser(fn)
    dset = h5py.File(fn, 'r')
    if sp_group not in dset:
        raise ValueError(
            "HDF5 file (%s) doesn't have group (%s)!" % (fn, sp_group))
    if sp_to_body_group not in dset:
        raise ValueError(
            "HDF5 file (%s) doesn't have group (%s)!" % (fn, sp_to_body_group))
    sp_map = array(dset[sp_group])
    sp_to_body_map = array(dset[sp_to_body_group])
    dset.close()
    return sp_map, sp_to_body_map


def write_h5_stack(npy_vol, fn, group='stack', compression=None, chunks=None,
                   shuffle=None):
    """Write a numpy.ndarray 3D volume to an HDF5 file.

    Parameters
    ----------
    npy_vol : numpy ndarray
        The array to be saved to HDF5.
    fn : string
        The output filename.
    group : string, optional (default: 'stack')
        The group within the HDF5 file to write to.
    compression : {None, 'gzip', 'szip', 'lzf'}, optional (default: None)
        The compression to use, if any. Note that 'lzf' is only available
        through h5py, so implementations in other languages will not be able
        to read files created with this compression.
    chunks : tuple, True, or None (default: None)
        Whether to use chunking in the HDF5 dataset. Default is None. True
        lets h5py choose a chunk size automatically. Otherwise, use a tuple
        of int of the same length as `npy_vol.ndim`. From the h5py
        documentation: "In the real world, chunks of size 10kB - 300kB work
        best, especially for compression. Very small chunks lead to lots of
        overhead in the file, while very large chunks can result in 
        inefficient I/O."
    shuffle : bool, optional
        Shuffle the bytes on disk to improve compression efficiency.

    Returns
    -------
    None
    """
    fn = os.path.expanduser(fn)
    fout = h5py.File(fn, 'a')
    if group in fout:
        del fout[group]
    fout.create_dataset(group, data=npy_vol, compression=compression,
                        chunks=chunks, shuffle=shuffle)
    fout.close()

### Raveler format

def ucm_to_raveler(ucm, sp_threshold=0.0, body_threshold=0.1, **kwargs):
    """Return Raveler map from a UCM.
    
    Parameters
    ----------
    ucm : numpy ndarray, shape (M, N, P)
        An ultrametric contour map. This is a map of scored segment boundaries
        such that if A, B, and C are segments, then 
        score(A, B) = score(B, C) >= score(A, C), for some permutation of
        A, B, and C.
        A hierarchical agglomeration process produces a UCM.
    sp_threshold : float, optional (default: 0.0)
        The value for which to threshold the UCM to obtain the superpixels.
    body_threshold : float, optional (default: 0.1)
        The value for which to threshold the UCM to obtain the segments/bodies.
        The condition `body_threshold >= sp_threshold` should hold in order
        to obtain sensible results.
    **kwargs : dict, optional
        Keyword arguments to be passed through to `segs_to_raveler`.

    Returns
    -------
    superpixels : numpy ndarray, shape (M, N, P)
        The superpixel map. Non-zero superpixels are unique to each plane.
        That is, `np.unique(superpixels[i])` and `np.unique(superpixels[j])` 
        have only 0 as their intersection.
    sp_to_segment : numpy ndarray, shape (Q, 3)
        The superpixel to segment map. Segments are unique to each plane. The
        first number on each line is the plane number.
    segment_to_body : numpy ndarray, shape (R, 2)
        The segment to body map.
    """
    sps = label(ucm < sp_threshold)[0]
    bodies = label(ucm <= body_threshold)[0]
    return segs_to_raveler(sps, bodies, **kwargs)

def segs_to_raveler(sps, bodies, min_size=0, do_conn_comp=False, sps_out=None):
    """Return a Raveler tuple from 3D superpixel and body maps.
    
    Parameters
    ----------
    sps : numpy ndarray, shape (M, N, P)
        The supervoxel map.
    bodies : numpy ndarray, shape (M, N, P)
        The body map. Superpixels should not map to more than one body.
    min_size : int, optional (default: 0)
        Superpixels smaller than this size on a particular plane are blacked
        out.
    do_conn_comp : bool (default: False)
        Whether to do a connected components operation on each plane. This is
        required if we want superpixels to be contiguous on each plane, since
        3D-contiguous superpixels are not guaranteed to be contiguous along
        a slice.
    sps_out : numpy ndarray, shape (M, N, P), optional (default: None)
        A Raveler-compatible superpixel map, meaning that superpixels are
        unique to each plane along axis 0. (See `superpixels` in the return
        values.) If provided, this saves significant computation time.

    Returns
    -------
    superpixels : numpy ndarray, shape (M, N, P)
        The superpixel map. Non-zero superpixels are unique to each plane.
        That is, `np.unique(superpixels[i])` and `np.unique(superpixels[j])` 
        have only 0 as their intersection.
    sp_to_segment : numpy ndarray, shape (Q, 3)
        The superpixel to segment map. Segments are unique to each plane. The
        first number on each line is the plane number.
    segment_to_body : numpy ndarray, shape (R, 2)
        The segment to body map.
    """
    if sps_out is None:
        sps_out = raveler_serial_section_map(sps, min_size, do_conn_comp, False)
    segment_map = raveler_serial_section_map(bodies, min_size, do_conn_comp)
    segment_to_body = unique(zip(segment_map.ravel(), bodies.ravel()))
    segment_to_body = segment_to_body[segment_to_body[:,0] != 0]
    segment_to_body = concatenate((array([[0,0]]), segment_to_body), axis=0)
    sp_to_segment = []
    for i, (sp_map_i, segment_map_i, body_map_i) in \
                            enumerate(zip(sps_out, segment_map, bodies)):
        segment_map_i *= sp_map_i.astype(bool)
        valid = (sp_map_i != 0) + (segment_map_i == 0)
        sp_to_segment.append(
            unique(zip(it.repeat(i), sp_map_i[valid], segment_map_i[valid])))
        valid = segment_map != 0
        logging.debug('plane %i done'%i)
    logging.info('total superpixels before: ' + str(len(unique(sps))) +
                ' total superpixels after: ' + str(len(unique(sps_out))))
    sp_to_segment = concatenate(sp_to_segment, axis=0)
    return sps_out, sp_to_segment, segment_to_body

def raveler_serial_section_map(nd_map, min_size=0, do_conn_comp=False, 
                                                    globally_unique_ids=True):
    """Produce `serial_section_map` and label one corner of each plane as 0.

    Raveler chokes when there are no pixels with label 0 on a plane, so this
    function produces the serial section map as normal but then adds a 0 to
    the [0, 0] corner of each plane, IF the volume doesn't already have 0
    pixels.

    Notes
    -----
        See `serial_section_map` for more info.
    """
    nd_map = serial_section_map(nd_map, min_size, do_conn_comp, 
                                                        globally_unique_ids)
    if not (nd_map == 0).any():
        nd_map[:,0,0] = 0
    return nd_map

def serial_section_map(nd_map, min_size=0, do_conn_comp=False, 
                                                    globally_unique_ids=True):
    """Produce a plane-by-plane superpixel map with unique IDs.

    Raveler requires sps to be unique and different on each plane. This
    function converts a fully 3D superpixel map to a serial-2D superpixel
    map compatible with Raveler.

    Parameters
    ----------
    nd_map : np.ndarray, int, shape (M, N, P)
        The original superpixel map.
    min_size : int (optional, default 0)
        Remove superpixels smaller than this size (on each plane)
    do_conn_comp : bool (optional, default False)
        In some cases, a single supervoxel may result in two disconnected
        superpixels in 2D. Set to True to force these to have different IDs.
    globally_unique_ids : bool (optional, default True)
        If True, every plane has unique IDs, with plane n having IDs {i1, i2,
        ..., in} and plane n+1 having IDs {in+1, in+2, ..., in+ip}, and so on.

    Returns
    -------
    relabeled_planes : np.ndarray, int, shape (M, N, P)
        A volume equal to nd_map but with superpixels relabeled along axis 0.
        That is, the input volume is reinterpreted as M slices of shape (N, P).
    """
    if do_conn_comp:
        label_fct = label
    else:
        def label_fct(a):
            relabeled, fmap, imap = evaluate.relabel_from_one(a)
            return relabeled, len(imap)
    def remove_small(a):
        return morpho.remove_small_connected_components(a, min_size, False)
    mplanes = map(remove_small, nd_map)
    relabeled_planes, nids_per_plane = zip(*map(label_fct, mplanes))
    start_ids = concatenate((array([0], int), cumsum(nids_per_plane)[:-1])) \
        if globally_unique_ids else [0]*len(nids_per_plane)
    relabeled_planes = [(relabeled_plane + start_id)[newaxis, ...]
        for relabeled_plane, start_id in zip(relabeled_planes, start_ids)]
    return concatenate(relabeled_planes, axis=0)

def write_to_raveler(sps, sp_to_segment, segment_to_body, directory, gray=None,
                    raveler_dir='/usr/local/raveler-hdf', nproc_contours=16,
                    body_annot=None):
    """Output a segmentation to Raveler format. 

    Parameters
    ----------
    sps : np.ndarray, int, shape (nplanes, nx, ny)
        The superpixel map. Superpixels can only occur on one plane.
    sp_to_segment : np.ndarray, int, shape (nsps + nplanes, 3)
        Superpixel-to-segment map as a 3 column list of (plane number,
        superpixel id, segment id). Segments must be unique to a plane, and
        each plane must contain the map {0: 0}
    segment_to_body: np.ndarray, int, shape (nsegments, 2)
        The segment to body map.
    directory: string 
        The directory in which to write the stack. This directory and all
        necessary subdirectories will be created.
    gray: np.ndarray, uint8 or uint16, shape (nplanes, nx, ny) (optional)
        The grayscale images corresponding to the superpixel maps.
    raveler dir: string (optional, default `/usr/local/raveler-hdf`)
        Where Raveler is installed.
    nproc_contours: int (optional, default 16) 
        How many processes to use when generating the Raveler contours.
    body_annot: dict or np.ndarray (optional)
        Either a dictionary to write to JSON in Raveler body annotation
        format, or a numpy ndarray of the segmentation from which to compute
        orphans and non traversing bodies (which then get written out as body
        annotations).

    Returns
    -------
    None

    Notes
    -----
        Raveler is the EM segmentation proofreading tool developed in-house at
        Janelia for the FlyEM project.
    """
    sp_path = os.path.join(directory, 'superpixel_maps')
    im_path = os.path.join(directory, 'grayscale_maps')
    tile_path = os.path.join(directory, 'tiles')

    if not os.path.exists(directory):
        os.makedirs(directory)

    # write superpixel->segment->body maps
    savetxt(os.path.join(directory, 'superpixel_to_segment_map.txt'),
        sp_to_segment, '%i') 
    savetxt(os.path.join(directory, 'segment_to_body_map.txt'), 
        segment_to_body, '%i')

    # write superpixels
    if not os.path.exists(sp_path): 
        os.mkdir(sp_path)
    write_png_image_stack(sps, os.path.join(sp_path, 'sp_map.%05i.png'),
        bitdepth=16, axis=0)

    # write grayscale
    if gray is not None:
        if not os.path.exists(im_path): 
            os.mkdir(im_path)
        write_png_image_stack(gray, 
                              os.path.join(im_path, 'img.%05d.png'), axis=0)

    # body annotations
    if body_annot is not None:
        if type(body_annot) == ndarray:
            orphans = morpho.orphans(body_annot)
            non_traversing = morpho.non_traversing_segments(body_annot)
            body_annot = raveler_body_annotations(orphans, non_traversing)
        write_json(body_annot, os.path.join(directory, 'annotations-body.json'))

    # make tiles, bounding boxes, and contours, and compile HDF5 stack info.
    with tmp.TemporaryFile() as tmp_stdout:
        try: 
            def call(arglist):
                return subprocess.call(arglist, stdout=tmp_stdout)
            r1 = call(['createtiles', im_path, sp_path, tile_path])
            r2 = call(['bounds', directory])
            r3 = call(['compilestack', directory])
        except:
            logging.warning(
                'Error during Raveler export post-processing step. ' +
                'Possible causes are that you do not have Raveler installed ' +
                'or you did not specify the correct installation path.')
            logging.warning('Return codes: %i, %i, %i' % (r1, r2, r3))
#            with sys.exc_info() as ex:
#                logging.warning('Exception info:\n' + '\n'.join(map(str, ex)))
    # make permissions friendly for proofreaders.
    try:
        subprocess.call(['chmod', '-R', 'go=u', directory])
    except:
        logging.warning('Could not change Raveler export permissions.')

def raveler_output_shortcut(svs, seg, gray, outdir, sps_out=None):
    """Compute the Raveler format and write to directory, all at once.
    
    Parameters
    ----------
    svs : np.ndarray, int, shape (M, N, P)
        The supervoxel map.
    seg : np.ndarray, int, shape (M, N, P)
        The segmentation map. It is assumed that no supervoxel crosses
        any segment boundary.
    gray : np.ndarray, uint8, shape (M, N, P)
        The grayscale EM images corresponding to the above segmentations.
    outdir : string
        The export directory for the Raveler volume.
    sps_out : np.ndarray, int, shape (M, N, P) (optional)
        The precomputed serial section 2D superpixel map. Output will be
        much faster if this is provided.

    Returns
    -------
    sps_out : np.ndarray, int, shape (M, N, P)
        The computed serial section 2D superpixel map. Keep this when
        making multiple calls to `raveler_output_shortcut` with the
        same supervoxel map.
    """
    sps_out, sp2seg, seg2body = segs_to_raveler(svs, seg, sps_out=sps_out)
    write_to_raveler(sps_out, sp2seg, seg2body, outdir, gray, body_annot=seg)
    return sps_out

def raveler_body_annotations(orphans, non_traversing=None):
    """Return a Raveler body annotation dictionary of orphan segments.

    Orphans are labeled as body annotations with `not sure` status and
    a string indicating `orphan` in the comments field.

    Non-traversing segments have only one contact with the surface of
    the volume, and are labeled `does not traverse` in the comments.

    Parameters
    ----------
    orphans : iterable of int
        The ID numbers corresponding to orphan segments.
    non_traversing : iterable of int (optional, default None)
        The ID numbers of segments having only one exit point in the volume.

    Returns
    -------
    body_annotations : dict
        A dictionary containing entries for 'data' and 'metadata' as
        specified in the Raveler body annotations format [1, 2].

    References
    ----------
    [1] https://wiki.janelia.org/wiki/display/flyem/body+annotation+file+format
    and:
    [2] https://wiki.janelia.org/wiki/display/flyem/generic+file+format
    """
    data = [{'status': 'not sure', 'comment': 'orphan', 'body ID': int(o)}
        for o in orphans]
    if non_traversing is not None:
        data.extend([{'status': 'not sure', 'comment': 'does not traverse',
            'body ID': int(n)} for n in non_traversing])
    metadata = {'description': 'body annotations', 'file version': 2}
    return {'data': data, 'metadata': metadata}

def write_json(annot, fn='annotations-body.json', directory=None):
    """Write an annotation dictionary in Raveler format to a JSON file.
    
    The annotation file format is described in:
    https://wiki.janelia.org/wiki/display/flyem/body+annotation+file+format
    and:
    https://wiki.janelia.org/wiki/display/flyem/generic+file+format

    Parameters
    ----------
    annot : dict
        A body annotations dictionary (described in pages above).
    fn : string (optional, default 'annotations-body.json')
        The filename to which to write the file.
    directory : string (optional, default None, or '.')
        A directory in which to write the file.

    Returns
    -------
    None
    """
    if directory is not None:
        fn = join_path(directory, fn)
    with open(fn, 'w') as f:
        json.dump(annot, f, indent=2)


def raveler_rgba_to_int(im, ignore_alpha=True):
    """Convert a volume using Raveler's RGBA encoding to int. [1]

    Parameters
    ----------
    im : np.ndarray, shape (M, N, P, 4)
        The image stack to be converted.
    ignore_alpha : bool, optional
        By default, the alpha channel does not encode anything. However, if
        we ever need 32 bits, it would be used. This function supports that
        with `ignore_alpha=False`. (default is True.)

    Returns
    -------
    im_int : np.ndarray, shape (M, N, P)
        The label volume.

    References
    ----------
    [1] https://wiki.janelia.org/wiki/display/flyem/Proofreading+data+and+formats
    """
    if im.ndim == 4 and im.shape[3] == 4:
        if ignore_alpha:
            im = im[..., :3]
        im_int = (im * 255 ** np.arange(im.shape[3])).sum(axis=3)
    else:
        im_int = im
    return im_int


def raveler_to_labeled_volume(rav_export_dir, get_glia=False, 
                        use_watershed=False, probability_map=None, crop=None):
    """Import a raveler export stack into a labeled segmented volume.
    
    Parameters
    ----------
    rav_export_dir : string
        The directory containing the Raveler stack.
    get_glia : bool (optional, default False)
        Return the segment numbers corresponding to glia, if available.
    use_watershed : bool (optional, default False)
        Fill in 0-labeled voxels using watershed.
    probability_map : np.ndarray, same shape as volume to be read (optional)
        If `use_watershed` is True, use `probability_map` as the landscape. If
        this is not provided, it uses a flat landscape.
    crop : tuple of int (optional, default None)
        A 6-tuple of [xmin, xmax, ymin, ymax, zmin, zmax].

    Returns
    -------
    output_volume : np.ndarray, shape (Z, X, Y)
        The segmentation in the Raveler volume.
    glia : list of int (optional, only returned if `get_glia` is True)
        The IDs in the segmentation corresponding to glial cells.
    """
    import morpho
    spmap = read_image_stack(
        os.path.join(rav_export_dir, 'superpixel_maps', '*.png'), crop=crop)
    spmap = raveler_rgba_to_int(spmap)
    sp2seg_list = np.loadtxt(
        os.path.join(rav_export_dir, 'superpixel_to_segment_map.txt'), uint32)
    seg2bod_list = np.loadtxt(
        os.path.join(rav_export_dir, 'segment_to_body_map.txt'), uint32)
    sp2seg = {}
    max_sp = sp2seg_list[:,1].max()
    start_plane = sp2seg_list[:,0].min()
    for z, sp, seg in sp2seg_list:
        if not sp2seg.has_key(z):
            sp2seg[z] = zeros(max_sp+1, uint32)
        sp2seg[z][sp] = seg
    max_seg = seg2bod_list[:,0].max()
    seg2bod = zeros(max_seg+1, uint32)
    seg2bod[seg2bod_list[:,0]] = seg2bod_list[:,1]
    initial_output_volume = zeros_like(spmap)
    for i, m in enumerate(spmap):
        j = start_plane + i
        initial_output_volume[i] = seg2bod[sp2seg[j][m]]
    if use_watershed:
        probs = np.ones_like(spmap) if probability_map is None \
                                    else probability_map
        output_volume = morpho.watershed(probs, seeds=initial_output_volume)
    else:
        output_volume = initial_output_volume
    if (output_volume[:, 0, 0] == 0).all() and \
                        (output_volume == 0).sum() == output_volume.shape[0]:
        output_volume[:, 0, 0] = output_volume[:, 0, 1]
    if get_glia:
        annots = json.load(
            open(os.path.join(rav_export_dir, 'annotations-body.json'), 'r'))
        glia = [a['body ID'] for a in annots['data'] 
                                        if a.get('comment', None) == 'glia']
        return output_volume, glia
    else:
        return output_volume

### Ilastik formats

# obtained from Ilastik 0.5.4
ilastik_label_colors = \
    [0xffff0000, 0xff00ff00, 0xffffff00, 0xff0000ff, 
    0xffff00ff, 0xff808000, 0xffc0c0c0, 0xfff2022d] 

def write_ilastik_project(images, labels, fn, label_names=None):
    """Write one or more image volumes and corresponding labels to Ilastik.
    
    Parameters
    ----------
    images : np.ndarray or list of np.ndarray, shapes (M_i, N_i[, P_i])
        The grayscale images to be saved.
    labels : np.ndarray or list of np.ndarray, same shapes as `images`
        The label maps corresponding to the images.
    fn : string
        The filename to save the project in.
    label_names : list of string (optional)
        The names corresponding to each label in `labels`. (Not implemented!)

    Returns
    -------
    None

    Notes
    -----
    Limitations:
        Assumes the same labels are used for all images.
        Supports only grayscale images and volumes, and a maximum of 8 labels.
        Requires at least one unlabeled voxel in the label field.
    """
    f = h5py.File(fn, 'w')
    if type(images) != list:
        images = [images]
        labels = [labels]
    ulbs = unique(concatenate(map(unique, labels)))[1:]
    colors = array(ilastik_label_colors[:len(ulbs)])
    names = ['Label %i'%i for i in ulbs]
    names = array(names, '|S%i'%max(map(len, names)))
    label_attributes = {'color':colors, 'name':names, 'number':ulbs}
    for i, (im, lb) in enumerate(zip(images, labels)):
        if im.ndim == 2:
            new_shape = (1,1)+im.shape+(1,)
        elif im.ndim == 3:
            new_shape = (1,)+im.shape+(1,)
        else:
            raise ValueError('Unsupported number of dimensions in image.')
        im = im.reshape(new_shape)
        lb = lb.reshape(new_shape)
        root = 'DataSets/dataItem%02i/'%i
        f[root+'data'] = im
        f[root+'labels'] = lb
        for k, v in label_attributes.items():
            f[root+'labels'].attrs[k] = v
        f[root].attrs['Name'] = ''
        f[root].attrs['fileName'] = ''
    for subgroup in ['Description', 'Labeler', 'Name']:
        f['Project/%s'%subgroup] = array('', dtype='|S1')
    f['ilastikVersion'] = array(0.5)
    f.close()

def write_ilastik_batch_volume(im, fn):
    """Write a volume to an HDF5 file for Ilastik batch processing.
    
    Parameters
    ----------
    im : np.ndarray, shape (M, N[, P])
        The image volume to be saved.
    fn : string
        The filename in which to save the volume.

    Returns
    -------
    None
    """
    if im.ndim == 2:
        im = im.reshape((1,1)+im.shape+(1,))
    elif im.ndim == 3:
        im = im.reshape((1,)+im.shape+(1,))
    else:
        raise ValueError('Unsupported number of dimensions in image.')
    write_h5_stack(im, fn, group='/volume/data')

def read_prediction_from_ilastik_batch(fn, **kwargs):
    """Read the prediction produced by Ilastik from batch processing.
    
    Parameters
    ----------
    fn : string
        The filename to read from.
    group : string (optional, default '/volume/prediction')
        Where to read from in the HDF5 file hierarchy.
    single_channel : bool (optional, default True)
        Read only the 0th channel (final dimension) from the volume.

    Returns
    -------
    None
    """
    if not kwargs.has_key('group'):
        kwargs['group'] = '/volume/prediction'
    a = squeeze(read_h5_stack(fn, **kwargs))
    if kwargs.get('single_channel', True):
        a = a[..., 0]
    return a
