import collections

import numpy as np
from scipy import sparse, ndimage as ndi
import networkx as nx

from viridis import tree

from . import evaluate as ev

def fast_rag(labels, connectivity=1):
    """Build a data-free region adjacency graph quickly.

    Parameters
    ----------
    labels : array of int
        Image pre-segmentation or segmentation
    connectivity : int in {1, ..., labels.ndim}, optional
        Use square connectivity equal to `connectivity`. See
        `scipy.ndimage.generate_binary_structure` for more.

    Returns
    -------
    g : networkx Graph
        A graph where nodes represent regions in `labels` and edges
        indicate adjacency.

    Examples
    --------
    >>> labels = np.array([1, 1, 5, 5], dtype=np.int_)
    >>> fast_rag(labels).edges()
    [(1, 5)]
    >>> labels = np.array([[1, 1, 1, 2, 2],
    ...                    [1, 1, 1, 2, 2],
    ...                    [3, 3, 4, 4, 4],
    ...                    [3, 3, 4, 4, 4]], dtype=np.int_)
    >>> sorted(fast_rag(labels).edges())
    [(1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]
    """
    conn = ndi.generate_binary_structure(labels.ndim, connectivity)
    eroded = ndi.grey_erosion(labels, footprint=conn)
    dilated = ndi.grey_dilation(labels, footprint=conn)
    boundaries = (eroded != dilated)
    labels_small = eroded[boundaries]
    labels_large = dilated[boundaries]
    n = np.max(labels_large) + 1
    # use a dummy broadcast array as data for RAG
    data = np.broadcast_to(np.ones((1,), dtype=np.int_),
                           labels_small.shape)
    sparse_graph = sparse.coo_matrix((data, (labels_small, labels_large)),
                                     dtype=np.int_, shape=(n, n)).tocsr()
    rag = nx.from_scipy_sparse_matrix(sparse_graph, edge_attribute='count')
    return rag


class Rag(object):
    def __init__(self, labels, connectivity=1):
        self.labels = labels
        self.graph = fast_rag(labels, connectivity)
        self.tree = tree.Ultrametric(init_nodes=self.graph.nodes())

    def merge_subgraph(self, subgraph: collections.Iterable = {},
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
        for connected_subgraph in nx.connected_component_subgraphs(subgraph):
            ordered_nodes = nx.dfs_preorder_nodes(connected_subgraph, source)
            current_node = next(ordered_nodes)
            for next_node in ordered_nodes:
                current_node = self.tree.merge(current_node, next_node)

    def current_segmentation(self,
                             cut_threshold: float = np.inf) -> np.ndarray:
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
    ctable = ev.contingency_table(fragments, ground_truth,
                                  ignore_seg=[], ignore_gt=[],
                                  norm=False)
    # break ties randomly; since ctable is not normalised, it contains
    # integer values, so adding noise of standard dev 0.01 will not change
    # any existing ordering
    ctable.data += np.random.randn(ctable.data.size) * 0.01
    maxes = ctable.max(axis=1).toarray()
    maxes_repeated = np.take(maxes, ctable.indices)
    assignments = sparse.csc_matrix((ctable.data == maxes_repeated,
                                     ctable.indices, ctable.indptr),
                                    dtype=np.bool_)
    assignments.eliminate_zeros()
    indptr = assignments.indptr
    rag = Rag(fragments)
    for i in range(len(indptr) - 1):
        rag.merge_subgraph(assignments.indices[indptr[i]:indptr[i+1]])
    return rag.current_segmentation()
