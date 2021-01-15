from collections.abc import Iterable

import numpy as np
from scipy import sparse, ndimage as ndi
import networkx as nx

from viridis import tree

from . import evaluate as ev
from . import sparselol


def edge_matrix(labels, connectivity=1):
    """Generate a COO matrix containing the coordinates of edge pixels.

    Parameters
    ----------
    labels : array of int
        An array of labeled pixels (or voxels).
    connectivity : int in {1, ..., labels.ndim}
        The square connectivity for considering neighborhood.

    Returns
    -------
    edges : sparse.coo_matrix
        A COO matrix where (i, j) indicate neighboring labels and the
        corresponding data element is the linear index of the edge pixel
        in the labels array.
    """
    conn = ndi.generate_binary_structure(labels.ndim, connectivity)
    eroded = ndi.grey_erosion(labels, footprint=conn).ravel()
    dilated = ndi.grey_dilation(labels, footprint=conn).ravel()
    labels = labels.ravel()
    boundaries0 = np.flatnonzero(eroded != labels)
    boundaries1 = np.flatnonzero(dilated != labels)
    labels_small = np.concatenate((eroded[boundaries0], labels[boundaries1]))
    labels_large = np.concatenate((labels[boundaries0], dilated[boundaries1]))
    n = np.max(labels_large) + 1
    data = np.concatenate((boundaries0, boundaries1))
    sparse_graph = sparse.coo_matrix((data, (labels_small, labels_large)),
                                     dtype=np.int_, shape=(n, n))
    return sparse_graph


def sparse_boundaries(coo_boundaries):
    """Use a sparselol to map edges to boundary extents.

    Parameters
    ----------
    coo_boundaries : sparse.coo_matrix
        The boundary locations encoded in ``(i, j, loc)`` form in a sparse COO
        matrix (scipy), where ``loc`` is the raveled index of a pixel that is
        part of the boundary between segments ``i`` and ``j``.

    Returns
    -------
    edge_to_idx : CSR matrix
        Maps each edge `[i, j]` to a unique index `v`.
    bounds : SparseLOL
        A map of edge indices to locations in the volume.
    """
    edge_to_idx = coo_boundaries.copy().tocsr()
    # edge_to_idx: CSR matrix that maps each edge to a unique integer
    # we don't use the ID 0 so that empty spots can be used to mean "no ID".
    edge_to_idx.data = np.arange(1, len(edge_to_idx.data) + 1, dtype=np.int_)
    edge_labels = np.ravel(edge_to_idx[coo_boundaries.row, coo_boundaries.col])
    bounds = sparselol.extents(edge_labels, input_indices=coo_boundaries.data)
    return edge_to_idx, sparselol.SparseLOL(bounds)


def fast_rag(labels, connectivity=1, out=None):
    """Build a data-free region adjacency graph quickly.

    Parameters
    ----------
    labels : array of int
        Image pre-segmentation or segmentation
    connectivity : int in {1, ..., labels.ndim}, optional
        Use square connectivity equal to `connectivity`. See
        `scipy.ndimage.generate_binary_structure` for more.
    out : networkx.Graph, optional
        Add edges into this graph object.

    Returns
    -------
    g : networkx Graph
        A graph where nodes represent regions in `labels` and edges
        indicate adjacency.

    Examples
    --------
    >>> labels = np.array([[1, 1, 1, 2, 2],
    ...                    [1, 1, 1, 2, 2],
    ...                    [3, 3, 4, 4, 4],
    ...                    [3, 3, 4, 4, 4]], dtype=np.int_)
    >>> sorted(fast_rag(labels).edges())
    [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

    Use the ``out`` parameter to build into an existing networkx graph.
    Warning: the existing graph contents will be cleared!

    >>> import networkx as nx
    >>> g = nx.Graph()
    >>> h = fast_rag(labels, out=g)
    >>> g is h
    True
    >>> sorted(h.edges())
    [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

    The edges contain the number of pixels counted in the boundary:

    >>> h[1][4]
    {'count': 2}
    >>> h[1][3]
    {'count': 4}

    ``fast_rag`` works on data of any dimension. For a 1D array:

    >>> labels = np.array([1, 1, 5, 5], dtype=np.int_)
    >>> list(fast_rag(labels).edges())
    [(1, 5)]
    """
    coo_graph = edge_matrix(labels, connectivity)
    # use a broadcast array of ones as data; these will get aggregated
    # into counts when the COO is converted to CSR
    coo_graph.data = np.broadcast_to(np.ones((1,), dtype=np.int_),
                                     coo_graph.row.shape)
    sparse_graph = coo_graph.tocsr()
    rag = nx.Graph() if out is None else out
    rag = nx.from_scipy_sparse_matrix(sparse_graph, create_using=rag,
                                      edge_attribute='count')
    return rag


class Rag(object):
    """A minimal region adjacency graph class.

    Parameters
    ----------
    labels : array of int
        An n-dimensional array of integer-labeled regions.
    connectivity : int in {1, ..., `labels.ndim`}, optional
        The square connectivity, determining which nearby pixels are
        considered neighbors.

    Attributes
    ----------
    labels : array of int
        Reference to the input. (ie if the input array is modified, so will
        the value in this attribute!)
    graph : networkx.Graph
        The region adjacency graph constructed from `labels`.
    tree : viridis.tree.Ultrametric
        The merge tree, for which the nodes in `graph` are leaves.
    """
    def __init__(self, labels : np.ndarray, connectivity: int = 1):
        self.labels = labels
        self.graph = fast_rag(labels, connectivity)
        self.tree = tree.Ultrametric(init_nodes=self.graph.nodes())

    def merge_subgraph(self, subgraph: Iterable = {},
                       source: int = None):
        """Merge nodes given by `subgraph`.

        Parameters
        ----------
        subgraph : iterable of int, optional
            A subset of nodes in `self.graph`.
        source : int, optional
            Merge subgraph starting at this node.
        """
        # first, turn node collection into graph (ie with corresponding edges)
        subgraph = self.graph.subgraph(subgraph)
        if len(subgraph) == 0:
            # do nothing given empty subgraph
            return
        for connected_subgraph in (
                self.graph.subgraph(c)
                for c in nx.connected_components(subgraph)
                ):
            ordered_nodes = nx.dfs_preorder_nodes(connected_subgraph, source)
            current_node = next(ordered_nodes)
            for next_node in ordered_nodes:
                current_node = self.tree.merge(current_node, next_node)

    def current_segmentation(self,
                             cut_threshold: float = np.inf) -> np.ndarray:
        """Return the segmentation implied by the graph and current merge tree.

        Parameters
        ----------
        cut_threshold : float, optional
            If provided, cut the merge tree at this threshold (virtually;
            the tree is not modified) before calculating the segmentation.

        Returns
        -------
        seg : array of int
            The segmentation.

        """
        label_map = self.tree.get_map(cut_threshold)
        return label_map[self.labels]


def best_segmentation(fragments: np.ndarray, ground_truth: np.ndarray,
                      random_seed: int = None) -> np.ndarray:
    """Return the best segmentation possible when only merging in `fragments`.

    Parameters
    ----------
    fragments : array of int
        An initial oversegmentation.
    ground_truth : array of int
        The true segmentation.
    random_seed : int, optional
        Seed `numpy.random` with this value.

    Returns
    -------
    segments : array of int
        The closest segmentation to `ground_truth` that can be obtained
        by only merging in `fragments`.

    Examples
    --------
    >>> fragments = np.array([[1, 1, 1, 2],
    ...                       [1, 1, 2, 2],
    ...                       [3, 4, 4, 4]], dtype=np.int_)
    >>> ground_truth = np.array([[1, 1, 2, 2]] * 3, dtype=np.int_)
    >>> best_segmentation(fragments, ground_truth)
    array([[5, 5, 5, 6],
           [5, 5, 6, 6],
           [5, 6, 6, 6]])
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    assignments = ev.assignment_table(fragments, ground_truth).tocsc()
    indptr = assignments.indptr
    rag = Rag(fragments)
    for i in range(len(indptr) - 1):
        rag.merge_subgraph(assignments.indices[indptr[i]:indptr[i+1]])
    return rag.current_segmentation()
