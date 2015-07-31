from . import base, moments, histogram, graph, contact

def paper_em():
    """Return the feature manager used in the PLoS ONE paper.

    This manager was used both for the FIBSEM segmentation (with
    multi-channel probabilities) and the SNEMI3D segmentation. [1]_

    Returns
    -------
    comp : `base.Composite` feature manager
        The feature manager to use for graph agglomeration.

    References
    ----------
    .. [1] http://brainiac2.mit.edu/SNEMI3D/content/gala-serial-2d-watershed
    """
    fm = moments.Manager()
    fh = histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9])
    fg = graph.Manager()
    return base.Composite(children=[fm, fh, fg])


def snemi3d():
    """Return the best-performing feature manager for SNEMI3D.

    This correspond's to Neal Donnelly's last submission in 2014 [1]_.

    Returns
    -------
    comp : feature manager
        Same as the `paper_em` manager but including also a `contact`
        manager.

    References
    ----------
    .. [1] http://brainiac2.mit.edu/SNEMI3D/content/gala-new-watersheds
    """
    fm = moments.Manager()
    fh = histogram.Manager(25, 0, 1, [0.1, 0.5, 0.9])
    fg = graph.Manager()
    fc = contact.Manager()
    return base.Composite(children=[fm, fh, fg, fc])


