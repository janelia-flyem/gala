import numpy as np
from scipy import ndimage as nd


def nd_sobel_magnitude(image, spacing=None):
    """Compute the magnitude of Sobel gradients along all axes.

    Parameters
    ----------
    image : array
        The input image.
    spacing : list of float, optional
        The voxel spacing along each dimension.

    Returns
    -------
    filtered : array
        The filtered image.
    """
    image = image.astype(np.float)
    filtered = np.zeros_like(image)
    if spacing is None:
        spacing = np.ones(image.ndim, np.float32)
    for ax, sp in enumerate(spacing):
        axsobel = nd.sobel(image, axis=ax) / sp
        filtered += axsobel * axsobel
    filtered = np.sqrt(filtered)
    return filtered

