from gala import imio
import h5py
import numpy as np
from skimage._shared._tempfile import temporary_file


def test_cremi_roundtrip():
    raw_image = np.random.randint(256, size=(5, 100, 100), dtype=np.uint8)
    labels = np.random.randint(4096, size=raw_image.shape, dtype=np.uint64)
    for ax in range(labels.ndim):
        labels.sort(axis=ax)  # try to get something vaguely contiguous. =P
    with temporary_file('.hdf') as fout:
        imio.write_cremi({'/volumes/raw': raw_image,
                          '/volumes/labels/neuron_ids': labels}, fout)
        raw_in, lab_in = imio.read_cremi(fout)
        f = h5py.File(fout)
        stored_resolution = f['/volumes/raw'].attrs['resolution']
        f.close()
    np.testing.assert_equal(stored_resolution, (40, 4, 4))
    np.testing.assert_equal(raw_in, raw_image)
    np.testing.assert_equal(lab_in, labels)


def test_vtk_roundtrip():
    for i in range(4):
        ar_shape = np.random.randint(1, 50, size=3)
        dtypes = list(imio.numpy_type_to_vtk_string.keys())
        cur_dtype = np.random.choice(dtypes)
        if np.issubdtype(cur_dtype, np.integer):
            ar = np.random.randint(0, 127, size=ar_shape).astype(cur_dtype)
        else:
            ar = np.random.rand(*ar_shape).astype(cur_dtype)
        with temporary_file('.vtk') as fout:
            imio.write_vtk(ar, fout)
            ar_in = imio.read_vtk(fout)
        np.testing.assert_equal(ar, ar_in)
