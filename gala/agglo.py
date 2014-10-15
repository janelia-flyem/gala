# built-ins
from itertools import combinations, izip, repeat, product
import itertools as it
import argparse
import random
import logging
import json
from copy import deepcopy
from math import isnan
# libraries
from numpy import (array, mean, zeros, zeros_like, uint8, where, unique,
    double, newaxis, nonzero, median, exp, log2, float, ones, arange, inf,
    flatnonzero, sign, unravel_index, bincount)
import numpy as np
from scipy.stats import sem
from scipy.sparse import lil_matrix
from scipy.misc import comb as nchoosek
from scipy.ndimage.measurements import label
from networkx import Graph, biconnected_components
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes

from viridis import tree

# local modules
from . import morpho
from . import iterprogress as ip
from . import optimized as opt
from .ncut import ncutW
from .mergequeue import MergeQueue
from .evaluate import contingency_table as ev_contingency_table, split_vi, xlogx
from . import features
from . import classify
from .classify import get_classifier, \
    unique_learning_data_elements, concatenate_data_elements


def contingency_table(a, b):
    ct = ev_contingency_table(a, b)
    nx, ny = ct.shape
    ctout = np.zeros((2 * nx, ny), ct.dtype)
    ct.todense(out=ctout[:nx, :])
    return ctout


arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Agglomeration options')
arggroup.add_argument('-t', '--thresholds', nargs='+', default=[128],
    type=float, metavar='FLOAT',
    help='''The agglomeration thresholds. One output file will be written
        for each threshold.'''
)
arggroup.add_argument('-l', '--ladder', type=int, metavar='SIZE',
    help='Merge any bodies smaller than SIZE.'
)
arggroup.add_argument('-p', '--pre-ladder', action='store_true', default=True,
    help='Run ladder before normal agglomeration (default).'
)
arggroup.add_argument('-L', '--post-ladder',
    action='store_false', dest='pre_ladder',
    help='Run ladder after normal agglomeration instead of before (SLOW).'
)
arggroup.add_argument('-s', '--strict-ladder', type=int, metavar='INT',
    default=1,
    help='''Specify the strictness of the ladder agglomeration. Level 1
        (default): merge anything smaller than the ladder threshold as
        long as it's not on the volume border. Level 2: only merge smaller
        bodies to larger ones. Level 3: only merge when the border is
        larger than or equal to 2 pixels.'''
)
arggroup.add_argument('-M', '--low-memory', action='store_true',
    help='''Use less memory at a slight speed cost. Note that the phrase
        'low memory' is relative.'''
)
arggroup.add_argument('--disallow-shared-boundaries', action='store_false',
    dest='allow_shared_boundaries',
    help='''Watershed pixels that are shared between more than 2 labels are
        not counted as edges.'''
)
arggroup.add_argument('--allow-shared-boundaries', action='store_true',
    default=True,
    help='''Count every watershed pixel in every edge in which it participates
        (default: True).'''
)


def conditional_countdown(seq, start=1, pred=bool):
    """Count down from 'start' each time pred(elem) is true for elem in seq."""
    remaining = start
    for elem in seq:
        if pred(elem):
            remaining -= 1
        yield remaining


############################
# Merge priority functions #
############################

def oriented_boundary_mean(g, n1, n2):
    return mean(g.oriented_probabilities_r[list(g[n1][n2]['boundary'])])


def boundary_mean(g, n1, n2):
    return mean(g.probabilities_r[list(g[n1][n2]['boundary'])])


def boundary_median(g, n1, n2):
    return median(g.probabilities_r[list(g[n1][n2]['boundary'])])


def approximate_boundary_mean(g, n1, n2):
    """Return the boundary mean as computed by a MomentsFeatureManager.

    The feature manager is assumed to have been set up for g at construction.
    """
    return g.feature_manager.compute_edge_features(g, n1, n2)[1]


def make_ladder(priority_function, threshold, strictness=1):
    def ladder_function(g, n1, n2):
        s1 = g.node[n1]['size']
        s2 = g.node[n2]['size']
        ladder_condition = \
                (s1 < threshold and not g.at_volume_boundary(n1)) or \
                (s2 < threshold and not g.at_volume_boundary(n2))
        if strictness >= 2:
            ladder_condition &= ((s1 < threshold) != (s2 < threshold))
        if strictness >= 3:
            ladder_condition &= len(g[n1][n2]['boundary']) > 2

        if ladder_condition:
            return priority_function(g, n1, n2)
        else:
            return inf
    return ladder_function


def no_mito_merge(priority_function):
    def predict(g, n1, n2):
        if n1 in g.frozen_nodes or n2 in g.frozen_nodes \
        or (n1, n2) in g.frozen_edges:
            return np.inf
        else:
            return priority_function(g, n1, n2)
    return predict


def mito_merge():
    def predict(g, n1, n2):
        if n1 in g.frozen_nodes and n2 in g.frozen_nodes:
            return np.inf
        elif (n1, n2) in g.frozen_edges:
            return np.inf
        elif n1 not in g.frozen_nodes and n2 not in g.frozen_nodes:
            return np.inf
        else:
            if n1 in g.frozen_nodes:
                mito = n1
                cyto = n2
            else:
                mito = n2
                cyto = n1
            if g.node[mito]['size'] > g.node[cyto]['size']:
                return np.inf
            else:
                return 1.0 - (float(len(g[mito][cyto]["boundary"]))/
                sum([len(g[mito][x]["boundary"]) for x in g.neighbors(mito)]))
    return predict


def classifier_probability(feature_extractor, classifier):
    def predict(g, n1, n2):
        if n1 == g.boundary_body or n2 == g.boundary_body:
            return inf
        features = feature_extractor(g, n1, n2)
        try:
            prediction_arr = np.array(classifier.predict_proba(features))
            if prediction_arr.ndim > 2: prediction_arr = prediction_arr[0]
            try:
                prediction = prediction_arr[0][1]
            except (TypeError, IndexError):
                prediction = prediction_arr[0]
        except AttributeError:
            prediction = classifier.predict(features)[0]
        return prediction
    return predict


def ordered_priority(edges):
    d = {}
    n = len(edges)
    for i, (n1, n2) in enumerate(edges):
        score = float(i)/n
        d[(n1,n2)] = score
        d[(n2,n1)] = score
    def ord(g, n1, n2):
        return d.get((n1,n2), inf)
    return ord


def expected_change_vi(feature_extractor, classifier, alpha=1.0, beta=1.0):
    prob_func = classifier_probability(feature_extractor, classifier)
    def predict(g, n1, n2):
        p = prob_func(g, n1, n2) # Prediction from the classifier
        # Calculate change in VI if n1 and n2 should not be merged
        v = compute_local_vi_change(
            g.node[n1]['size'], g.node[n2]['size'], g.volume_size
        )
        # Return expected change
        return  (p*alpha*v + (1.0-p)*(-beta*v))
    return predict


def compute_local_vi_change(s1, s2, n):
    """Compute change in VI if we merge disjoint sizes s1,s2 in a volume n."""
    py1 = float(s1)/n
    py2 = float(s2)/n
    py = py1+py2
    return -(py1*log2(py1) + py2*log2(py2) - py*log2(py))


def compute_true_delta_vi(ctable, n1, n2):
    p1 = ctable[n1].sum()
    p2 = ctable[n2].sum()
    p3 = p1+p2
    p1g_log_p1g = xlogx(ctable[n1]).sum()
    p2g_log_p2g = xlogx(ctable[n2]).sum()
    p3g_log_p3g = xlogx(ctable[n1]+ctable[n2]).sum()
    return p3*log2(p3) - p1*log2(p1) - p2*log2(p2) - \
                                2*(p3g_log_p3g - p1g_log_p1g - p2g_log_p2g)


def expected_change_rand(feature_extractor, classifier, alpha=1.0, beta=1.0):
    prob_func = classifier_probability(feature_extractor, classifier)
    def predict(g, n1, n2):
        p = float(prob_func(g, n1, n2)) # Prediction from the classifier
        v = compute_local_rand_change(
            g.node[n1]['size'], g.node[n2]['size'], g.volume_size
        )
        return p*v*alpha + (1.0-p)*(-beta*v)
    return predict


def compute_local_rand_change(s1, s2, n):
    """Compute change in rand if we merge disjoint sizes s1,s2 in volume n."""
    return float(s1*s2)/nchoosek(n,2)


def compute_true_delta_rand(ctable, n1, n2, n):
    """Compute change in RI obtained by merging rows n1 and n2.

    This function assumes ctable is normalized to sum to 1.
    """
    localct = n*ctable[(n1,n2),]
    delta_sxy = 1.0/2*((localct.sum(axis=0)**2).sum()-(localct**2).sum())
    delta_sx = 1.0/2*(localct.sum()**2 - (localct.sum(axis=1)**2).sum())
    return (2*delta_sxy - delta_sx) / nchoosek(n,2)


def boundary_mean_ladder(g, n1, n2, threshold, strictness=1):
    f = make_ladder(boundary_mean, threshold, strictness)
    return f(g, n1, n2)


def boundary_mean_plus_sem(g, n1, n2, alpha=-6):
    bvals = g.probabilities_r[list(g[n1][n2]['boundary'])]
    return mean(bvals) + alpha*sem(bvals)


def random_priority(g, n1, n2):
    if n1 == g.boundary_body or n2 == g.boundary_body:
        return inf
    return random.random()


class Rag(Graph):
    """Region adjacency graph for segmentation of nD volumes."""

    def __init__(self, watershed=array([]), probabilities=array([]),
            merge_priority_function=boundary_mean,
            allow_shared_boundaries=True, gt_vol=None,
            feature_manager=features.base.Null(),
            show_progress=False, lowmem=False, connectivity=1,
            channel_is_oriented=None, orientation_map=array([]),
            normalize_probabilities=False, nozeros=False, exclusions=array([]),
            isfrozennode=None, isfrozenedge=None):
        """Create a graph from label and image/probability volumes.

        The label field can be complete (every pixel belongs to a
        region > 0), or it can have boundaries (regions are separated
        by pixels of label 0). Regions are considered adjacent if (a)
        they are adjacent to each other, or (b) they are both adjacent
        to a pixel of label 0.

        Parameters
        ----------
        watershed : array of int, shape (M, N, ..., P)
            The labeled regions of the image. Note: this is called
            `watershed` for historical reasons, but could refer to a
            superpixel map of any origin.
        probabilities : array of float, shape (M, N, ..., P[, Q])
            The probability of each pixel of belonging to a particular
            class. Typically, this has the same shape as `watershed`
            and represents the probability that the pixel is part of a
            region boundary, but it can also have an additional
            dimension for probabilities of belonging to other classes,
            such as mitochondria (in biological images) or specific
            textures (in natural images).
        merge_priority_function : callable function, optional
            This function must take exactly three arguments as input
            (a Rag object and two node IDs) and return a single float.
        allow_shared_boundaries : bool, optional
            If True, 0-pixels with three or more adjacent labels belong
            to the boundaries of each possible pair of labels.
            Otherwise, these pixels do not belong to any boundaries.
        feature_manager : ``features.base.Null`` object, optional
            A feature manager object that controls feature computation
            and feature caching.
        show_progress : bool, optional
            Whether to display an ASCII progress bar during long-
            -running graph operations.
        lowmem : bool, optional
            Use a lower-memory mode by not pre-caching the neighbors
            array. This trades off a 10% decrease in memory usage
            for a 10% slower runtime.
        connectivity : int in {1, ..., `watershed.ndim`}
            When determining adjacency, allow neighbors along
            `connectivity` dimensions.
        channel_is_oriented : array-like of bool, shape (Q,), optional
            For multi-channel images, some channels, for example some
            edge detectors, have a specific orientation. In conjunction
            with the `orientation_map` argument, specify which channels
            have an orientation associated with them.
        orientation_map : array-like of float, shape (Q,)
            Specify the orientation of the corresponding channel. (2D
            images only)
        normalize_probabilities : bool, optional
            Divide the input `probabilities` by their maximum to ensure
            a range in [0, 1].
        nozeros : bool, optional
            If you know your volume has no 0-labeled pixels, setting
            `nozeros` to ``True`` will speed up graph construction.
        exclusions : array-like of int, shape (M, N, ..., P), optional
            Volume of same shape as `watershed`. Mark points in the
            volume with the same label (>0) to prevent them from being
            merged during agglomeration. For example, if
            `exclusions[45, 92] == exclusions[51, 105] == 1`, then
            segments `watershed[45, 92]` and `watershed[51, 105]` will
            never be merged, regardless of the merge priority function.
        isfrozennode : function, optional
            Function taking in a Rag object and a node id and returning
            a bool. If the function returns ``True``, the node will not
            be merged, regardless of the merge priority function.
        isfrozenedge : function, optional
            As `isfrozennode`, but the function should take the graph
            and *two* nodes, to specify an edge that cannot be merged.

        Returns
        -------
        self : Rag object
            A region adjacency graph (Rag) object, containing all
            necessary information to perform agglomerative
            segmentation.
        """
        super(Rag, self).__init__(weighted=False)
        self.show_progress = show_progress
        self.nozeros = nozeros
        self.connectivity = connectivity
        self.pbar = (ip.StandardProgressBar() if self.show_progress
                     else ip.NoProgressBar())
        self.set_watershed(watershed, lowmem, connectivity)
        self.set_probabilities(probabilities, normalize_probabilities)
        self.set_orientations(orientation_map, channel_is_oriented)
        if watershed is None:
            self.ucm = None
        else:
            self.ucm = -inf*ones(self.watershed.shape, dtype=float)
            self.ucm[self.watershed==0] = inf
            self.ucm_r = self.ucm.ravel()
        self.merge_priority_function = merge_priority_function
        self.max_merge_score = -inf
        self.build_graph_from_watershed(allow_shared_boundaries,
                                        nozerosfast=self.nozeros)
        self.set_feature_manager(feature_manager)
        self.set_ground_truth(gt_vol)
        self.set_exclusions(exclusions)
        self.merge_queue = MergeQueue()
        self.tree = tree.Ultrametric(self.nodes())
        self.frozen_nodes = set()
        if isfrozennode is not None:
            for node in self.nodes():
                if isfrozennode(self, node):
                    self.frozen_nodes.add(node)
        self.frozen_edges = set()
        if isfrozenedge is not None:
            for n1, n2 in self.edges():
                if isfrozenedge(self, n1, n2):
                    self.frozen_edges.add((n1,n2))
        for nodeid in self.nodes():
            del self.node[nodeid]["extent"]


    def __copy__(self):
        """Return a copy of the object and attributes.
        """
        pr_shape = self.probabilities_r.shape
        g = super(Rag, self).copy()
        g.watershed_r = g.watershed.ravel()
        g.ucm_r = g.ucm.ravel()
        g.probabilities_r = g.probabilities.reshape(pr_shape)
        return g


    def copy(self):
        """Return a copy of the object and attributes.
        """
        return self.__copy__()


    def extent(self, nodeid):
        if 'extent' in self.node[nodeid]:
            return self.node[nodeid]['extent']
        extent_array = opt.flood_fill(self.watershed, 
                                      np.array(self.node[nodeid]['entrypoint']), 
                                      np.array(self.node[nodeid]['watershed_ids']))
        if len(extent_array) != self.node[nodeid]['size']:
            sys.stderr.write("Flood fill fail - found %d voxels but size expected %d\n" \
                                % (len(extent_array), self.node[nodeid]['size']))
        raveled_indices = np.ravel_multi_index((extent_array[:,0], 
                extent_array[:,1], extent_array[:,2]), self.watershed.shape)
        return set(raveled_indices)

    def real_edges(self, *args, **kwargs):
        """Return edges internal to the volume.

        The RAG actually includes edges to a "virtual" region that
        envelops the entire volume. This function returns the list of
        edges that are internal to the volume.

        Parameters
        ----------
        *args, **kwargs : arbitrary types
            Arguments and keyword arguments are passed through to the
            ``edges()`` function of the ``networkx.Graph`` class.

        Returns
        -------
        edge_list : list of tuples
            A list of pairs of node IDs, which are typically integers.

        See Also
        --------
        ``real_edges_iter``, ``networkx.Graph.edges``.
        """
        return [e for e in super(Rag, self).edges(*args, **kwargs) if
                                            self.boundary_body not in e[:2]]

    def real_edges_iter(self, *args, **kwargs):
        """Return iterator of edges internal to the volume.

        The RAG actually includes edges to a "virtual" region that
        envelops the entire volume. This function returns the list of
        edges that are internal to the volume.

        Parameters
        ----------
        *args, **kwargs : arbitrary types
            Arguments and keyword arguments are passed through to the
            ``edges()`` function of the ``networkx.Graph`` class.

        Returns
        -------
        edges_iter : iterator of tuples
            An iterator over pairs of node IDs, which are typically
            integers.
        """
        return (e for e in super(Rag, self).edges_iter(*args, **kwargs) if
                                            self.boundary_body not in e[:2])


    def build_graph_from_watershed_nozerosfast(self, idxs):
        """Build the graph object from the region labels.

        Parameters
        ----------
        idxs : array-like of int
            Build the graph considering only these indices (linear into
            the raveled array).

        Returns
        -------
        None

        Notes
        -----
        Always allow shared boundaries in this code.
        """
        if self.watershed.size == 0: return # stop processing for empty graphs
        if idxs is None:
            idxs = arange(self.watershed.size)
            self.add_node(self.boundary_body,
                    extent=set(flatnonzero(self.watershed==self.boundary_body)))
        inner_idxs = idxs[self.watershed_r[idxs] != self.boundary_body]
        for idx in inner_idxs:
            ns = self.neighbor_idxs(idx)
            adj_labels = self.watershed_r[ns]
            nodeid = self.watershed_r[idx]
            adj_labels = adj_labels[adj_labels != nodeid]
            edges = None
            if adj_labels.size > 0:
                adj_labels = unique(adj_labels)
                edges = zip(repeat(nodeid), adj_labels)
            if not self.has_node(nodeid):
                self.add_node(nodeid, extent=set())
            if 'entrypoint' not in self.node[nodeid]:
                entrypoint_tuple = np.unravel_index(idx, self.watershed.shape)
                self.node[nodeid]['entrypoint'] = np.array(entrypoint_tuple)
            if 'watershed_ids' not in self.node[nodeid]:
                self.node[nodeid]['watershed_ids'] = [nodeid]
            try:
                self.node[nodeid]['extent'].add(idx)
            except KeyError:
                self.node[nodeid]['extent'] = set([idx])
            try:
                self.node[nodeid]['size'] += 1
            except KeyError:
                self.node[nodeid]['size'] = 1

            if edges is not None:
                for l1,l2 in edges:
                    if self.has_edge(l1, l2):
                        self[l1][l2]['boundary'].add(idx)
                    else:
                        self.add_edge(l1, l2, boundary=set([idx]))


    def build_graph_from_watershed(self, allow_shared_boundaries=True,
                                   idxs=None, nozerosfast=False):
        """Build the graph object from the region labels.

        The region labels should have been set ahead of time using
        ``set_watershed()``.

        Parameters
        ----------
        allow_shared_boundaries : bool, optional
            Allow voxels that have three or more distinct neighboring
            labels to be included in all boundaries.
        idxs : array-like of int, optional
            Linear indices into raveled volume array. If provided, the
            graph is built only for these indices.
        nozerosfast : bool, optional
            Assume that there are no zero (boundary) labels in the
            volume. By removing this check, graph build time is
            reduced.

        Returns
        -------
        None
        """
        if nozerosfast:
            self.build_graph_from_watershed_nozerosfast(idxs)
            return

        if self.watershed.size == 0: return # stop processing for empty graphs
        if not allow_shared_boundaries:
            self.ignored_boundary = zeros(self.watershed.shape, bool)
        if idxs is None:
            idxs = arange(self.watershed.size)
            self.add_node(self.boundary_body,
                    extent=set(flatnonzero(self.watershed==self.boundary_body)))
        inner_idxs = idxs[self.watershed_r[idxs] != self.boundary_body]
        for idx in ip.with_progress(inner_idxs, title='Graph ', pbar=self.pbar):
            ns = self.neighbor_idxs(idx)
            adj_labels = self.watershed_r[ns]
            adj_labels = unique(adj_labels)
            adj_labels = adj_labels[adj_labels.nonzero()]
            nodeid = self.watershed_r[idx]
            if nodeid != 0:
                adj_labels = adj_labels[adj_labels != nodeid]
                edges = zip(repeat(nodeid), adj_labels)
                if not self.has_node(nodeid):
                    self.add_node(nodeid, extent=set())
                if 'entrypoint' not in self.node[nodeid]:
                    entrypoint_tuple = np.unravel_index(idx, self.watershed.shape)
                    self.node[nodeid]['entrypoint'] = np.array(entrypoint_tuple)
                if 'watershed_ids' not in self.node[nodeid]:
                    self.node[nodeid]['watershed_ids'] = [nodeid]
                try:
                    self.node[nodeid]['extent'].add(idx)
                except KeyError:
                    self.node[nodeid]['extent'] = set([idx])
                try:
                    self.node[nodeid]['size'] += 1
                except KeyError:
                    self.node[nodeid]['size'] = 1
            else:
                if len(adj_labels) == 0: continue
                if adj_labels[-1] != self.boundary_body:
                    edges = list(combinations(adj_labels, 2))
                else:
                    edges = list(product([self.boundary_body], adj_labels[:-1]))
            if allow_shared_boundaries or len(edges) == 1:
                for l1,l2 in edges:
                    if self.has_edge(l1, l2):
                        self[l1][l2]['boundary'].add(idx)
                    else:
                        self.add_edge(l1, l2, boundary=set([idx]))
            elif len(edges) > 1:
                self.ignored_boundary.ravel()[idx] = True


    def set_feature_manager(self, feature_manager):
        """Set the feature manager and ensure feature caches are computed.

        Parameters
        ----------
        feature_manager : ``features.base.Null`` object
            The feature manager to be used by this RAG.

        Returns
        -------
        None
        """
        self.feature_manager = feature_manager
        self.compute_feature_caches()


    def compute_feature_caches(self):
        """Use the feature manager to compute node and edge feature caches.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for n in ip.with_progress(
                    self.nodes(), title='Node caches ', pbar=self.pbar):
            self.node[n]['feature-cache'] = \
                            self.feature_manager.create_node_cache(self, n)
        for n1, n2 in ip.with_progress(
                    self.edges(), title='Edge caches ', pbar=self.pbar):
            self[n1][n2]['feature-cache'] = \
                            self.feature_manager.create_edge_cache(self, n1, n2)


    def get_neighbor_idxs_fast(self, idxs):
        """Retrieve a previously computed set of neighbors from an array.

        Parameters
        ----------
        idxs : int or iterable of int
            A linear index or set of indices into the padded array.

        Returns
        -------
        neighbors : array of int, shape `(len(idxs), N_neighbors)`
            An array of linear indices to the neighbors of each input
            index.

        Raises
        ------
        AttributeError
            If ``self.pixel_neighbors`` does not exist. It must be
            previously computed by
            ``self.set_watershed(..., lowmem=False)``.

        See Also
        --------
        ``self.set_watershed``
        """
        return self.pixel_neighbors[idxs]


    def get_neighbor_idxs_lean(self, idxs, connectivity=1):
        """Compute neighbor indices from input indices.

        Parameters
        ----------
        idxs : int or iterable of int
            A linear index or set of indices into the padded array.
        connectivity : int in {1, ..., ``self.watershed.ndim``}, optional
            The neighbor connectivity, defining which voxels are
            considered adjacent to the center. A connectivity of 1
            means voxels whose coordinates differ by 1 along only a
            single dimension, 2 along up to 2 dimensions, and so on.

        Returns
        -------
        neighbors : array of int, shape `(len(idxs), N_neighbors)`
            An array of linear indices to the neighbors of each input
            index.
        """
        return morpho.get_neighbor_idxs(self.watershed, idxs, connectivity)


    def set_probabilities(self, probs=array([]), normalize=False):
        """Set the `probabilities` attributes of the RAG.

        For various reasons, including removing the need for bounds
        checking when looking for neighboring pixels, the volume of
        pixel-level probabilities is padded on all faces. In addition,
        this function adds an attribute `probabilities_r`, a raveled
        view of the padded probabilities array for quick access to
        individual voxels using linear indices.

        Parameters
        ----------
        probs : array
            The input probabilities array.
        normalize : bool, optional
            If ``True``, the values in the array are scaled to be in
            [0, 1].

        Returns
        -------
        None
        """
        if len(probs) == 0:
            self.probabilities = zeros_like(self.watershed)
            self.probabilities_r = self.probabilities.ravel()
        probs = probs.astype(double)
        if normalize and len(probs) > 1:
            probs -= probs.min() # ensure probs.min() == 0
            probs /= probs.max() # ensure probs.max() == 1
        sp = probs.shape
        sw = tuple(array(self.watershed.shape, dtype=int)-\
                    2*self.pad_thickness*ones(self.watershed.ndim, dtype=int))
        p_ndim = probs.ndim
        w_ndim = self.watershed.ndim
        padding = [inf]+(self.pad_thickness-1)*[0]
        if p_ndim == w_ndim:
            self.probabilities = morpho.pad(probs, padding)
            self.probabilities_r = self.probabilities.ravel()[:,newaxis]
        elif p_ndim == w_ndim+1:
            if sp[1:] == sw:
                sp = sp[1:]+[sp[0]]
                probs = probs.transpose(sp)
            axes = range(p_ndim-1)
            self.probabilities = morpho.pad(probs, padding, axes)
            self.probabilities_r = self.probabilities.reshape(
                                                (self.watershed.size, -1))


    def set_orientations(self, orientation_map, channel_is_oriented):
        """Set the orientation map of the probability image.

        Parameters
        ----------
        orientation_map : array of float
            A map of angles of the same shape as the superpixel map.
        channel_is_oriented : 1D array-like of bool
            A vector having length the number of channels in the
            probability map.

        Returns
        -------
        None
        """
        if len(orientation_map) == 0:
            self.orientation_map = zeros_like(self.watershed)
            self.orientation_map_r = self.orientation_map.ravel()
        padding = [0]+(self.pad_thickness-1)*[0]
        self.orientation_map = morpho.pad(orientation_map, padding).astype(int)
        self.orientation_map_r = self.orientation_map.ravel()
        if channel_is_oriented is None:
            nchannels = 1 if self.probabilities.ndim==self.watershed.ndim \
                else self.probabilities.shape[-1]
            self.channel_is_oriented = array([False]*nchannels)
            self.max_probabilities_r = zeros_like(self.probabilities_r)
            self.oriented_probabilities_r = zeros_like(self.probabilities_r)
            self.non_oriented_probabilities_r = self.probabilities_r
        else:
            self.channel_is_oriented = channel_is_oriented
            self.max_probabilities_r = \
                self.probabilities_r[:, self.channel_is_oriented].max(axis=1)
            self.oriented_probabilities_r = \
                self.probabilities_r[:, self.channel_is_oriented]
            self.oriented_probabilities_r = \
                self.oriented_probabilities_r[
                    range(len(self.oriented_probabilities_r)),
                    self.orientation_map_r]
            self.non_oriented_probabilities_r = \
                self.probabilities_r[:, ~self.channel_is_oriented]


    def set_watershed(self, ws=array([]), lowmem=False, connectivity=1):
        """Set the initial segmentation volume (watershed).

        The initial segmentation is called `watershed` for historical
        reasons only.

        Parameters
        ----------
        ws : array of int
            The initial segmentation.
        lowmem : bool, optional
            Whether to use a low memory/high time mode. This usually
            results in about 10% less memory usage and 10% more time.
        connectivity : int in {1, ..., `ws.ndim`}, optional
            The pixel neighborhood.

        Returns
        -------
        None
        """
        try:
            self.boundary_body = ws.max()+1
        except ValueError: # empty watershed given
            self.boundary_body = -1
        self.volume_size = ws.size
        self.has_zero_boundaries = (ws==0).any()
        if self.has_zero_boundaries:
            self.watershed = morpho.pad(ws, [0, self.boundary_body])
        else:
            self.watershed = morpho.pad(ws, self.boundary_body)
        self.watershed_r = self.watershed.ravel()
        self.pad_thickness = 2 if (self.watershed == 0).any() else 1
        if lowmem:
            def neighbor_idxs(x):
                return self.get_neighbor_idxs_lean(x, connectivity)
            self.neighbor_idxs = neighbor_idxs
        else:
            self.pixel_neighbors = \
                morpho.build_neighbors_array(self.watershed, connectivity)
            self.neighbor_idxs = self.get_neighbor_idxs_fast


    def set_ground_truth(self, gt=None):
        """Set the ground truth volume.

        This is useful for tracking segmentation accuracy over time.

        Parameters
        ----------
        gt : array of int
            A ground truth segmentation of the same volume passed to
            ``set_watershed``.

        Returns
        -------
        None
        """
        if gt is not None:
            gtm = gt.max()+1
            gt_ignore = [0, gtm] if (gt==0).any() else [gtm]
            seg_ignore = [0, self.boundary_body] if \
                        (self.watershed==0).any() else [self.boundary_body]
            self.gt = morpho.pad(gt, gt_ignore)
            self.rig = contingency_table(self.watershed, self.gt,
                                         ignore_seg=seg_ignore,
                                         ignore_gt=gt_ignore)
        else:
            self.gt = None
            # null pattern to transparently allow merging of nodes.
            # Bonus feature: counts how many sp's went into a single node.
            try:
                self.rig = ones(2 * self.watershed.max() + 1)
            except ValueError:
                self.rig = ones(2 * self.number_of_nodes() + 1)


    def set_exclusions(self, excl):
        """Set an exclusion volume, forbidding certain merges.

        Parameters
        ----------
        excl : array of int
            Exclusions work as follows: the volume `excl` is the same
            shape as the initial segmentation (see ``set_watershed``),
            and consists of mostly 0s. Any voxels with *the same*
            non-zero label will not be allowed to merge during
            agglomeration (provided they were not merged in the initial
            segmentation).

            This allows manual separation *a priori* of difficult-to-
            -segment regions.

        Returns
        -------
        None
        """
        if excl.size != 0:
            excl = morpho.pad(excl, [0]*self.pad_thickness)
        for n in self.nodes():
            if excl.size != 0:
                eids = unique(excl.ravel()[list(self.extent(n))])
                eids = eids[flatnonzero(eids)]
                self.node[n]['exclusions'] = set(list(eids))
            else:
                self.node[n]['exclusions'] = set()


    def build_merge_queue(self):
        """Build a queue of node pairs to be merged in a specific priority.

        Parameters
        ----------
        None

        Returns
        -------
        mq : MergeQueue object
            A MergeQueue is a Python ``deque`` with a specific element
            structure: a list of length 4 containing:
                 - the merge priority (any ordered type)
                 - a 'valid' flag
                 - and the two nodes in arbitrary order
            The valid flag allows one to "remove" elements from the
            queue in O(1) time by setting the flag to ``False``. Then,
            one checks the flag when popping elements and ignores those
            marked as invalid.

            One other specific feature is that there are back-links from
            edges to their corresponding queue items so that when nodes
            are merged, affected edges can be invalidated and reinserted
            in the queue with a new priority.
        """
        queue_items = []
        for l1, l2 in self.real_edges_iter():
            w = self.merge_priority_function(self,l1,l2)
            qitem = [w, True, l1, l2]
            queue_items.append(qitem)
            self[l1][l2]['qlink'] = qitem
            self[l1][l2]['weight'] = w
        return MergeQueue(queue_items, with_progress=self.show_progress)


    def rebuild_merge_queue(self):
        """Build a merge queue from scratch and assign to self.merge_queue.

        See Also
        --------
        ``self.build_merge_queue``
        """
        self.merge_queue = self.build_merge_queue()


    def agglomerate(self, threshold=0.5, save_history=False):
        """Merge nodes hierarchically until given edge confidence threshold.

        This is the main workhorse of the ``agglo`` module!

        Parameters
        ----------
        threshold : float, optional
            The edge priority at which to stop merging.
        save_history : bool, optional
            Whether to save and return a history of all the merges made.

        Returns
        -------
        history : list of tuple of int, optional
            The ordered history of node pairs merged.
        scores : list of float, optional
            The list of merge scores corresponding to the `history`.
        evaluation : list of tuple, optional
            The split VI after each merge. This is only meaningful if
            a ground truth volume was provided at build time.

        Notes
        -----
            This function returns ``None`` when `save_history` is
            ``False``.
        """
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history, scores, evaluation = [], [], []
        while len(self.merge_queue) > 0 and \
                                        self.merge_queue.peek()[0] < threshold:
            merge_priority, _, n1, n2 = self.merge_queue.pop()
            self.update_frozen_sets(n1, n2)
            self.merge_nodes(n1, n2, merge_priority)
            if save_history:
                history.append((n1,n2))
                scores.append(merge_priority)
                evaluation.append(
                    (self.number_of_nodes()-1, self.split_vi())
                )
        if save_history:
            return history, scores, evaluation


    def agglomerate_count(self, stepsize=100, save_history=False):
        """Agglomerate until 'stepsize' merges have been made.

        This function is like ``agglomerate``, but rather than to a
        certain threshold, a certain number of merges are made,
        regardless of threshold.

        Parameters
        ----------
        stepsize : int, optional
            The number of merges to make.
        save_history : bool, optional
            Whether to save and return a history of all the merges made.

        Returns
        -------
        history : list of tuple of int, optional
            The ordered history of node pairs merged.
        scores : list of float, optional
            The list of merge scores corresponding to the `history`.
        evaluation : list of tuple, optional
            The split VI after each merge. This is only meaningful if
            a ground truth volume was provided at build time.

        Notes
        -----
            This function returns ``None`` when `save_history` is
            ``False``.

        See Also
        --------
        ``Rag.agglomerate``.
        """
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history, evaluation = [], []
        i = 0
        for i in range(stepsize):
            if len(self.merge_queue) == 0:
                break
            merge_priority, _, n1, n2 = self.merge_queue.pop()
            i += 1
            self.merge_nodes(n1, n2, merge_priority)
            if save_history:
                history.append((n1, n2))
                evaluation.append(
                    (self.number_of_nodes()-1, self.split_vi())
                )
        if save_history:
            return history, evaluation


    def agglomerate_ladder(self, min_size=1000, strictness=2):
        """Merge sequentially all nodes smaller than `min_size`.

        Parameters
        ----------
        min_size : int, optional
            The smallest allowable segment after ladder completion.
        strictness : {1, 2, 3}, optional
            `strictness == 1`: all nodes smaller than `min_size` are
            merged according to the merge priority function.
            `strictness == 2`: in addition to `1`, small nodes can only
            be merged to big nodes.
            `strictness == 3`: in addition to `2`, nodes sharing less
            than one pixel of boundary are not agglomerated.

        Returns
        -------
        None

        Notes
        -----
        Nodes that are on the volume boundary are not agglomerated.
        """
        original_merge_priority_function = self.merge_priority_function
        self.merge_priority_function = make_ladder(
            self.merge_priority_function, min_size, strictness
        )
        self.rebuild_merge_queue()
        self.agglomerate(inf)
        self.merge_priority_function = original_merge_priority_function
        self.merge_queue.finish()
        self.rebuild_merge_queue()
        max_score = max([qitem[0] for qitem in self.merge_queue.q])
        self.ucm -= max_score
        for n in self.tree.nodes():
            self.tree.node[n]['w'] -= max_score


    def learn_agglomerate(self, gts, feature_map,
                          min_num_samples=1,
                          learn_flat=True,
                          learning_mode='strict',
                          labeling_mode='assignment',
                          priority_mode='active',
                          memory=True,
                          unique=True,
                          random_state=None,
                          max_num_epochs=10,
                          min_num_epochs=2,
                          max_num_samples=np.inf,
                          classifier='random forest',
                          active_function=classifier_probability,
                          mpf=boundary_mean):
        """Agglomerate while comparing to ground truth & classifying merges.

        Parameters
        ----------
        gts : array of int or list thereof
            The ground truth volume(s) corresponding to the current
            probability map.
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector. This must
            consist either of uncached features or of the cache used
            when building the graph.
        min_num_samples : int, optional
            Continue training until this many training examples have
            been collected.
        learn_flat : bool, optional
            Do a flat learning on the static graph with no
            agglomeration.
        learning_mode : {'strict', 'loose'}, optional
            In 'strict' mode, if a "don't merge" edge is encountered,
            it is added to the training set but the merge is not
            executed. In 'loose' mode, the merge is allowed to proceed.
        labeling_mode : {'assignment', 'vi-sign', 'rand-sign'}, optional
            How to decide whether two nodes should be merged based on
            the ground truth segmentations. ``'assignment'`` means the
            nodes are assigned to the ground truth node with which they
            share the highest overlap. ``'vi-sign'`` means the the VI
            change of the switch is used (negative is better).
            ``'rand-sign'`` means the change in Rand index is used
            (positive is better).
        priority_mode : string, optional
            One of:
                ``'active'``: Train a priority function with the data
                              from previous epochs to obtain the next.
                ``'random'``: Merge edges at random.
                ``'mixed'``: Alternate between epochs of ``'active'``
                             and ``'random'``.
                ``'mean'``: Use the mean boundary value. (In this case,
                            training is limited to 1 or 2 epochs.)
                ``'custom'``: Use the function provided by `mpf`.
        memory : bool, optional
            Keep the training data from all epochs (rather than just
            the most recent one).
        unique : bool, optional
            Remove duplicate feature vectors.
        random_state : int, optional
            If provided, this parameter is passed to `get_classifier`
            to set the random state and allow consistent results across
            tests.
        max_num_epochs : int, optional
            Do not train for longer than this (this argument *may*
            override the `min_num_samples` argument).
        min_num_epochs : int, optional
            Train for no fewer than this number of epochs.
        max_num_samples : int, optional
            Train for no more than this number of samples.
        classifier : string, optional
            Any valid classifier descriptor. See
            ``gala.classify.get_classifier()``
        active_function : function (feat. map, classifier) -> function, optional
            Use this to create the next priority function after an
            epoch.
        mpf : function (Rag, node, node) -> float
            A merge priority function to use when ``priority_mode`` is
            ``'custom'``.

        Returns
        -------
        data : list of array
            Four arrays containing:
                - the feature vectors, shape ``(n_samples, n_features)``.
                - the labels, shape ``(n_samples, 3)``. A value of `-1`
                  means "should merge", while `1` means "should
                  not merge". The columns correspond to the three
                  labeling methods: assignment, VI sign, or RI sign.
                - the VI and RI change of each merge, ``(n_edges, 2)``.
                - the list of merged edges ``(n_edges, 2)``.
        alldata : list of list of array
            A list of lists like `data` above: one list for each epoch.

        Notes
        -----
        The gala algorithm [1] uses the default parameters. For the
        LASH algorithm [2], use:
            - `learning_mode`: ``'loose'``
            - `labeling_mode`: ``'rand-sign'``
            - `memory`: ``False``

        References
        ----------
        .. [1] Nunez-Iglesias et al, Machine learning of hierarchical
               clustering to segment 2D and 3D images, PLOS ONE, 2013.
        .. [2] Jain et al, Learning to agglomerate superpixel
               hierarchies, NIPS, 2011.

        See Also
        --------
        ``Rag.__init__``
        """
        learning_mode = learning_mode.lower()
        labeling_mode = labeling_mode.lower()
        priority_mode = priority_mode.lower()
        if priority_mode == 'mean' and unique:
            max_num_epochs = 2 if learn_flat else 1
        if priority_mode in ['random', 'mean'] and not memory:
            max_num_epochs = 1
        label_type_keys = {'assignment':0, 'vi-sign':1, 'rand-sign':2}
        if type(gts) != list:
            gts = [gts] # allow using single ground truth as input
        master_ctables = \
                [contingency_table(self.get_segmentation(), gt) for gt in gts]
        alldata = []
        data = [[],[],[],[]]
        for num_epochs in range(max_num_epochs):
            ctables = deepcopy(master_ctables)
            if len(data[0]) > min_num_samples and num_epochs >= min_num_epochs:
                break
            if learn_flat and num_epochs == 0:
                alldata.append(self.learn_flat(gts, feature_map))
                data = unique_learning_data_elements(alldata) if memory \
                    else alldata[-1]
                continue
            g = self.copy()
            if priority_mode == 'mean':
                g.merge_priority_function = boundary_mean
            elif num_epochs > 0 and priority_mode == 'active' or \
                num_epochs % 2 == 1 and priority_mode == 'mixed':
                if random_state == None:
                    cl = get_classifier(classifier)
                else:
                    cl = get_classifier(classifier, random_state=random_state)
                feat, lab = classify.sample_training_data(
                    data[0], data[1][:, label_type_keys[labeling_mode]],
                    max_num_samples)
                cl = cl.fit(feat, lab)
                g.merge_priority_function = active_function(feature_map, cl)
            elif priority_mode == 'random' or \
                (priority_mode == 'active' and num_epochs == 0):
                g.merge_priority_function = random_priority
            elif priority_mode == 'custom':
                g.merge_priority_function = mpf
            g.show_progress = False # bug in MergeQueue usage causes
                                    # progressbar crash.
            g.rebuild_merge_queue()
            alldata.append(g._learn_agglomerate(ctables, feature_map,
                                                learning_mode, labeling_mode))
            if memory:
                if unique:
                    data = unique_learning_data_elements(alldata)
                else:
                    data = concatenate_data_elements(alldata)
            else:
                data = alldata[-1]
            logging.debug('data size %d at epoch %d'%(len(data[0]), num_epochs))
        return data, alldata


    def learn_flat(self, gts, feature_map):
        """Learn all edges on the graph, but don't agglomerate.

        Parameters
        ----------
        gts : array of int or list thereof
            The ground truth volume(s) corresponding to the current
            probability map.
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector. This must
            consist either of uncached features or of the cache used
            when building the graph.

        Returns
        -------
        data : list of array
            Four arrays containing:
                - the feature vectors, shape ``(n_samples, n_features)``.
                - the labels, shape ``(n_samples, 3)``. A value of `-1`
                  means "should merge", while `1` means "should
                  not merge". The columns correspond to the three
                  labeling methods: assignment, VI sign, or RI sign.
                - the VI and RI change of each merge, ``(n_edges, 2)``.
                - the list of merged edges ``(n_edges, 2)``.

        See Also
        --------
        ``learn_agglomerate``.
        """
        if type(gts) != list:
            gts = [gts] # allow using single ground truth as input
        ctables = [contingency_table(self.get_segmentation(), gt) for gt in gts]
        assignments = [(ct == ct.max(axis=1)[:,newaxis]) for ct in ctables]
        return map(array, zip(*[
                self.learn_edge(e, ctables, assignments, feature_map)
                for e in self.real_edges()]))


    def learn_edge(self, edge, ctables, assignments, feature_map):
        """Determine whether an edge should be merged based on ground truth.

        Parameters
        ----------
        edge : (int, int) tuple
            An edge in the graph.
        ctables : list of array
            A list of contingency tables determining overlap between the
            current segmentation and the ground truth.
        assignments : list of array
            Similar to the contingency tables, but each row is thresholded
            so each segment corresponds to exactly one ground truth segment.
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector.


        Returns
        -------
        features : 1D array of float
            The feature vector for that edge.
        labels : 1D array of float, length 3
            The labels determining whether the edge should be merged.
            A value of `-1` means "should merge", while `1` means "should
            not merge". The columns correspond to the three labeling
            methods: assignment, VI sign, or RI sign.
        weights : 1D array of float, length 2
            The VI and RI change of the merge.
        nodes : tuple of int
            The given edge.
        """
        n1, n2 = edge
        features = feature_map(self, n1, n2).ravel()
        # Calculate weights for weighting data points
        s1, s2 = [self.node[n]['size'] for n in [n1, n2]]
        weights = \
            compute_local_vi_change(s1, s2, self.volume_size), \
            compute_local_rand_change(s1, s2, self.volume_size)
        # Get the fraction of times that n1 and n2 assigned to
        # same segment in the ground truths
        cont_labels = [
            [(-1)**(a[n1,:]==a[n2,:]).all() for a in assignments],
            [compute_true_delta_vi(ctable, n1, n2) for ctable in ctables],
            [-compute_true_delta_rand(ctable, n1, n2, self.volume_size)
                                                    for ctable in ctables]
        ]
        labels = [sign(mean(cont_label)) for cont_label in cont_labels]
        if any(map(isnan, labels)) or any([label == 0 for l in labels]):
            logging.debug('NaN or 0 labels found. ' +
                                    ' '.join(map(str, [labels, (n1, n2)])))
        labels = [1 if i==0 or isnan(i) or n1 in self.frozen_nodes or
            n2 in self.frozen_nodes or (n1, n2) in self.frozen_edges else
            i for i in labels]
        return features, labels, weights, (n1, n2)


    def _learn_agglomerate(self, ctables, feature_map, gt_dts,
                        learning_mode='strict', labeling_mode='assignment'):
        """Learn the agglomeration process using various strategies.

        Parameters
        ----------
        ctables : array of float or list thereof
            One or more contingency tables between own segments and gold
            standard segmentations
        feature_map : function (Rag, node, node) -> array of float
            The map from node pairs to a feature vector. This must
            consist either of uncached features or of the cache used
            when building the graph.
        learning_mode : {'strict', 'loose'}
            If ``'strict'``, don't proceed with a merge when it goes against
            the ground truth.
        labeling_mode : {'assignment', 'vi-sign', 'rand-sign'}
            Which label to use for `learning_mode`. Note that all labels
            are saved in the end.

        Returns
        -------
        data : list of array
            Four arrays containing:
                - the feature vectors, shape ``(n_samples, n_features)``.
                - the labels, shape ``(n_samples, 3)``. A value of `-1`
                  means "should merge", while `1` means "should
                  not merge". The columns correspond to the three
                  labeling methods: assignment, VI sign, or RI sign.
                - the VI and RI change of each merge, ``(n_edges, 2)``.
                - the list of merged edges ``(n_edges, 2)``.
        """
        label_type_keys = {'assignment':0, 'vi-sign':1, 'rand-sign':2}
        assignments = [(ct == ct.max(axis=1)[:,newaxis]) for ct in ctables]
        g = self
        data = []
        while len(g.merge_queue) > 0:
            merge_priority, valid, n1, n2 = g.merge_queue.pop()
            dat = g.learn_edge((n1,n2), ctables, assignments, feature_map)
            data.append(dat)
            label = dat[1][label_type_keys[labeling_mode]]
            if learning_mode != 'strict' or label < 0:
                node_id = g.merge_nodes(n1, n2, merge_priority)
                for ctable, assignment in zip(ctables, assignments):
                    ctable[node_id] = ctable[n1] + ctable[n2]
                    ctable[n1] = 0
                    ctable[n2] = 0
                    assignment[node_id] = (ctable[node_id] ==
                                           ctable[node_id].max())
                    assignment[n1] = 0
                    assignment[n2] = 0
        return map(array, zip(*data))


    def replay_merge_history(self, merge_seq, labels=None, num_errors=1):
        """Agglomerate according to a merge sequence, optionally labeled.

        Parameters
        ----------
        merge_seq : iterable of pair of int
            The sequence of node IDs to be merged.
        labels : iterable of int in {-1, 0, 1}, optional
            A sequence matching `merge_seq` specifying whether a merge
            should take place or not. -1 or 0 mean "should merge", 1
            otherwise.

        Returns
        -------
        n : int
            Number of elements consumed from `merge_seq`
        e : (int, int)
            Last merge pair observed.

        Notes
        -----
        The merge sequence and labels *must* be generators if you don't want
        to manually keep track of how much has been consumed. The merging
        continues until `num_errors` false merges have been encountered, or
        until the sequence is fully consumed.
        """
        if labels is None:
            labels1 = it.repeat(False)
            labels2 = it.repeat(False)
        else:
            labels1 = (label > 0 for label in labels)
            labels2 = (label > 0 for label in labels)
        counter = it.count()
        errors_remaining = conditional_countdown(labels2, num_errors)
        nodes = None
        for nodes, label, errs, count in \
                        izip(merge_seq, labels1, errors_remaining, counter):
            n1, n2 = nodes
            if not label:
                self.merge_nodes(n1, n2)
            elif errs == 0:
                break
        return count, nodes


    def update_ucm(self, n1, n2):
        """Update ultrametric contour map with the current max boundary value.

        Parameters
        ----------
        n1, n2 : int
            Nodes determining the edge for which to update the UCM.

        Returns
        -------
        None

        Notes
        -----
        Presently, the gala UCM is an approximation. A true UCM is a
        subpixel property of the edges *between* pixels (unless using
        pixel-thick boundaries). Gala, instead, uses the edges of
        segments. If using a boundary-less segmentation, it is best to
        avoid the UCM.
        """
        try:
            edge = self[n1][n2]
        except KeyError:
            return
        w = edge['weight'] if edge.has_key('weight') else -inf
        if self.ucm is not None:
            self.max_merge_score = max(self.max_merge_score, w)
            idxs = list(edge['boundary'])
            self.ucm_r[idxs] = self.max_merge_score


    def update_max_ucm(self, n1, n2):
        """Update the UCM locally with an infinite value.

        Parameters
        ----------
        n1, n2 : int
            Nodes determining the edge for which to update the UCM.

        Returns
        -------
        None
        """
        edge = self[n1][n2]
        if self.ucm is not None:
            self.ucm_r[list(edge['boundary'])] = inf


    def rename_node(self, old, new):
        """Rename node `old` to `new`, updating edges and weights.

        Parameters
        ----------
        old : int
            The node being renamed.
        new : int
            The new node id.
        """
        self.add_node(new, attr_dict=self.node[old])
        self.add_edges_from(
            [(new, v, self[old][v]) for v in self.neighbors(old)])
        for v in self.neighbors(new):
            qitem = self[new][v].get('qlink', None)
            if qitem is not None:
                if qitem[2] == old:
                    qitem[2] = new
                else:
                    qitem[3] = new
        self.remove_node(old)


    def merge_nodes(self, n1, n2, merge_priority=0.0):
        """Merge two nodes, while updating the necessary edges.

        Parameters
        ----------
        n1, n2 : int
            Nodes determining the edge for which to update the UCM.
        merge_priority : float, optional
            The merge priority of the merge.

        Returns
        -------
        node_id : int
            The id of the node resulting from the merge.

        Notes
        -----
        This updates the UCM with the maximum merge priority value
        encountered so far.

        Additionally, the RIG (region intersection graph), the
        contingency matrix to the ground truth (if provided) is
        updated.
        """
        if len(self.node[n1]['exclusions'] & self.node[n2]['exclusions']) > 0:
            self.update_max_ucm(n1, n2)
            return
        else:
            self.node[n1]['exclusions'].update(self.node[n2]['exclusions'])
        self.update_ucm(n1, n2)
        w = self[n1][n2].get('weight', merge_priority)
        self.node[n1]['size'] += self.node[n2]['size']
        self.node[n1]['watershed_ids'] += self.node[n2]['watershed_ids']

        self.feature_manager.update_node_cache(self, n1, n2,
                self.node[n1]['feature-cache'], self.node[n2]['feature-cache'])
        new_neighbors = [n for n in self.neighbors(n2)
                         if n not in [n1, self.boundary_body]]
        for n in new_neighbors:
            self.merge_edge_properties((n2, n), (n1, n))
        # this if statement enables merging of non-adjacent nodes
        if self.has_edge(n1,n2) and self.has_zero_boundaries:
            sp2segment = self.tree.get_map(w)
            self.refine_post_merge_boundaries(n1, n2, sp2segment)
        try:
            self.merge_queue.invalidate(self[n1][n2]['qlink'])
        except KeyError:
            pass
        node_id = self.tree.merge(n1, n2, w)
        self.remove_node(n2)
        self.rename_node(n1, node_id)
        self.rig[node_id] = self.rig[n1] + self.rig[n2]
        self.rig[n1] = 0
        self.rig[n2] = 0
        return node_id


    def refine_post_merge_boundaries(self, n1, n2, sp2segment):
        """Ensure boundary pixels are only counted once after a merge.

        Parameters
        ----------
        n1, n2 : int
            Nodes determining the edge for which to update the UCM.
        sp2segment : array of int
            The most recent map from superpixels to segments.
        """
        boundary = array(list(self[n1][n2]['boundary']))
        boundary_neighbor_pixels = sp2segment[self.watershed_r[
                                              self.neighbor_idxs(boundary)]]
        add = ((boundary_neighbor_pixels == 0) +
               (boundary_neighbor_pixels == n1) +
               (boundary_neighbor_pixels == n2)).all(axis=1)
        check = True - add
        self.feature_manager.pixelwise_update_node_cache(self, n1,
                        self.node[n1]['feature-cache'], boundary[add])
        boundaries_to_edit = {}
        for px in boundary[check]:
            px_neighbors = self.neighbor_idxs(px)
            labels = np.unique(sp2segment[self.watershed_r[px_neighbors]])
            for lb in labels:
                if lb not in [0, n1, self.boundary_body]:
                    try:
                        boundaries_to_edit[(n1,lb)].append(px)
                    except KeyError:
                        boundaries_to_edit[(n1,lb)] = [px]
        for u, v in boundaries_to_edit.keys():
            idxs = set(boundaries_to_edit[(u,v)])
            if self.has_edge(u, v):
                idxs = idxs - self[u][v]['boundary']
                self[u][v]['boundary'].update(idxs)
                self.feature_manager.pixelwise_update_edge_cache(self, u, v,
                                    self[u][v]['feature-cache'], list(idxs))
            else:
                self.add_edge(u, v, boundary=set(idxs))
                self[u][v]['feature-cache'] = \
                    self.feature_manager.create_edge_cache(self, u, v)
            self.update_merge_queue(u, v)
        for n in self.neighbors(n2):
            if not boundaries_to_edit.has_key((n1,n)) and n != n1:
                self.update_merge_queue(n1, n)


    def merge_subgraph(self, subgraph=None, source=None):
        """Merge a (typically) connected set of nodes together.

        Parameters
        ----------
        subgraph : agglo.Rag, networkx.Graph, or list of int (node id)
            A subgraph to merge.
        source : int (node id), optional
            Merge the subgraph to this node.

        Returns
        -------
        None
        """
        if type(subgraph) not in [Rag, Graph]: # input is node list
            subgraph = self.subgraph(subgraph)
        if len(subgraph) > 0:
            node_dfs = list(dfs_preorder_nodes(subgraph, source))
            # dfs_preorder_nodes returns iter, convert to list
            source_node, other_nodes = node_dfs[0], node_dfs[1:]
            for current_node in other_nodes:
                self.merge_nodes(source_node, current_node)


    def split_node(self, u, n=2, **kwargs):
        """Use normalized cuts [1] to split a node/segment.

        Parameters
        ----------
        u : int (node id)
            Which node to split.
        n : int, optional
            How many segments to split it into.

        Returns
        -------
        None

        References
        ----------
        .. [1] Shi, J., and Malik, J. (2000). Normalized cuts and image
               segmentation. Pattern Analysis and Machine Intelligence.
        """
        node_extent = list(self.extent(u))
        node_borders = set().union(
                        *[self[u][v]['boundary'] for v in self.neighbors(u)])
        labels = unique(self.watershed_r[node_extent])
        if labels[0] == 0:
            labels = labels[1:]
        self.remove_node(u)
        self.build_graph_from_watershed(
            idxs=array(list(set().union(node_extent, node_borders)))
        )
        self.ncut(num_clusters=n, nodes=labels, **kwargs)


    def merge_edge_properties(self, src, dst):
        """Merge the properties of edge src into edge dst.

        Parameters
        ----------
        src, dst : (int, int)
            Edges being merged.

        Returns
        -------
        None
        """
        u, v = dst
        w, x = src
        if not self.has_edge(u,v):
            self.add_edge(u, v, attr_dict=self[w][x])
        else:
            self[u][v]['boundary'].update(self[w][x]['boundary'])
            self.feature_manager.update_edge_cache(self, (u, v), (w, x),
                    self[u][v]['feature-cache'], self[w][x]['feature-cache'])
        try:
            self.merge_queue.invalidate(self[w][x]['qlink'])
        except KeyError:
            pass
        self.update_merge_queue(u, v)


    def update_merge_queue(self, u, v):
        """Update the merge queue item for edge (u, v). Add new by default.

        Parameters
        ----------
        u, v : int (node id)
            Edge being updated.

        Returns
        -------
        None
        """
        if self.boundary_body in [u, v]:
            return
        if self[u][v].has_key('qlink'):
            self.merge_queue.invalidate(self[u][v]['qlink'])
        if not self.merge_queue.is_null_queue:
            w = self.merge_priority_function(self,u,v)
            new_qitem = [w, True, u, v]
            self[u][v]['qlink'] = new_qitem
            self[u][v]['weight'] = w
            self.merge_queue.push(new_qitem)


    def get_segmentation(self):
        """Return the unpadded segmentation represented by the graph.

        Remember that the segmentation volume is padded with an
        "artificial" segment that envelops the volume. This function
        simply removes the wrapping and returns a segmented volume.

        Parameters
        ----------
        None

        Returns
        -------
        seg : array of int
            The segmentation of the volume presently represented by the
            graph.

        See Also
        --------
        ``agglo.Rag.get_ucm``
        """
        m = self.tree.get_map()
        seg = m[self.watershed]
        if self.pad_thickness > 1: # volume has zero-boundaries
            seg = morpho.remove_merged_boundaries(seg, self.connectivity)
        return morpho.juicy_center(seg, self.pad_thickness)


    def get_ucm(self):
        """Return the current, unpadded ultrametric contour map.

        The contour map is an approximation, because in the absence of
        boundaries, the true UCM is a subpixel property of the faces
        between pixels. However, in this case, we return all those
        pixels that touch a face, which can result in segments being
        disconnected in the UCM.

        In the case of "thick" boundaries where segments don't have
        very thin regions, this is a valid approximation.

        Parameters
        ----------
        None

        Returns
        -------
        ucm : array of float
            The map of boundary values between segments implied by the
            hierarchical agglomeration process.
        """
        if hasattr(self, 'ignored_boundary'):
            self.ucm[self.ignored_boundary] = self.max_merge_score
        ucm = morpho.juicy_center(self.ucm, self.pad_thickness)
        umin, umax = unique(ucm)[([1, -2],)]
        ucm[ucm==-inf] = umin-1
        ucm[ucm==inf] = umax+1
        return ucm


    def build_volume(self, nbunch=None):
        """Return the segmentation induced by the graph.

        Parameters
        ----------
        nbunch : iterable of int (node id), optional
            A list of nodes for which to build the volume. All nodes
            are used if this is not provided.

        Returns
        -------
        seg : array of int
            The segmentation implied by the graph.

        Notes
        -----
        This function is very similar to ``get_segmentation``, but it
        builds the segmentation from the bottom up, rather than using
        the currently-stored segmentation.
        """
        v = zeros_like(self.watershed)
        vr = v.ravel()
        if nbunch is None:
            nbunch = self.nodes()
        for n in nbunch:
            vr[list(self.extent(n))] = n
        return morpho.juicy_center(v,self.pad_thickness)


    def build_boundary_map(self, ebunch=None):
        """Return a map of the current merge priority.

        Parameters
        ----------
        ebunch : iterable of (int, int), optional
            The list of edges for which to build a map. Use all edges
            if not provided.

        Returns
        -------
        bm : array of float
            The image of the edge weights.
        """
        if len(self.merge_queue) == 0:
            self.rebuild_merge_queue()
        m = zeros(self.watershed.shape, double)
        mr = m.ravel()
        if ebunch is None:
            ebunch = self.real_edges_iter()
        ebunch = sorted([(self[u][v]['weight'], u, v) for u, v in ebunch])
        for w, u, v in ebunch:
            b = list(self[u][v]['boundary'])
            mr[b] = w
        if hasattr(self, 'ignored_boundary'):
            m[self.ignored_boundary] = inf
        return morpho.juicy_center(m, self.pad_thickness)


    def remove_obvious_inclusions(self):
        """Merge any nodes with only one edge to their neighbors."""
        for n in self.nodes():
            if self.degree(n) == 1:
                self.merge_nodes(self.neighbors(n)[0], n)


    def remove_inclusions(self):
        """Merge any segments fully contained within other segments.

        In 3D EM images, inclusions are not biologically plausible, so
        this function can be used to remove them.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        bcc = list(biconnected_components(self))
        if len(bcc) > 1:
            container = [i for i, s in enumerate(bcc) if
                         self.boundary_body in s][0]
            del bcc[container] # remove the main graph
            bcc = map(list, bcc)
            for cc in bcc:
                cc.sort(key=lambda x: self.node[x]['size'], reverse=True)
            bcc.sort(key=lambda x: self.node[x[0]]['size'])
            for cc in bcc:
                self.merge_subgraph(cc, cc[0])


    def orphans(self):
        """List all the nodes that do not touch the volume boundary.

        Parameters
        ----------
        None

        Returns
        -------
        orphans : list of int (node id)
            A list of node ids.

        Notes
        -----
        "Orphans" are not biologically plausible in EM data, so we can
        flag them with this function for further scrutiny.
        """
        return [n for n in self.nodes() if not self.at_volume_boundary(n)]


    def compute_orphans(self):
        """Find all the segments that do not touch the volume boundary.

        Parameters
        ----------
        None

        Returns
        -------
        orphans : list of int (node id)
            A list of node ids.

        Notes
        -----
        This function differs from ``orphans`` in that it does not use
        the graph, but rather computes orphans directly from the
        segmentation.
        """
        return morpho.orphans(self.get_segmentation())


    def is_traversed_by_node(self, n):
        """Determine whether a body traverses the volume.

        This is defined as touching the volume boundary at two distinct
        locations.

        Parameters
        ----------
        n : int (node id)
            The node being inspected.

        Returns
        -------
        tr : bool
            Whether the segment "traverses" the volume being segmented.
        """
        if not self.at_volume_boundary(n) or n == self.boundary_body:
            return False
        v = zeros(self.watershed.shape, uint8)
        v.ravel()[list(self[n][self.boundary_body]['boundary'])] = 1
        _, n = label(v, ones([3]*v.ndim))
        return n > 1


    def traversing_bodies(self):
        """List all bodies that traverse the volume."""
        return [n for n in self.nodes() if self.is_traversed_by_node(n)]


    def non_traversing_bodies(self):
        """List bodies that are not orphans and do not traverse the volume."""
        return [n for n in self.nodes() if self.at_volume_boundary(n) and
            not self.is_traversed_by_node(n)]


    def compute_non_traversing_bodies(self):
        """Same as agglo.Rag.non_traversing_bodies, but doesn't use graph."""
        return morpho.non_traversing_bodies(self.get_segmentation())


    def raveler_body_annotations(self, traverse=False):
        """Return JSON-compatible dict formatted for Raveler annotations."""
        orphans = self.compute_orphans()
        non_traversing_bodies = self.compute_non_traversing_bodies() \
                                if traverse else []
        data = \
            [{'status':'not sure', 'comment':'orphan', 'body ID':int(o)}
                for o in orphans] +\
            [{'status':'not sure', 'comment':'does not traverse',
                'body ID':int(n)} for n in non_traversing_bodies]
        metadata = {'description':'body annotations', 'file version':2}
        return {'data':data, 'metadata':metadata}


    def at_volume_boundary(self, n):
        """Return True if node n touches the volume boundary."""
        return self.has_edge(n, self.boundary_body) or n == self.boundary_body


    def should_merge(self, n1, n2):
        return self.rig[n1].argmax() == self.rig[n2].argmax()


    def get_pixel_label(self, n1, n2):
        boundary = array(list(self[n1][n2]['boundary']))
        min_idx = boundary[self.probabilities_r[boundary,0].argmin()]
        if self.should_merge(n1, n2):
            return min_idx, 2
        else:
            return min_idx, 1


    def pixel_labels_array(self, false_splits_only=False):
        ar = zeros_like(self.watershed_r)
        labels = [self.get_pixel_label(*e) for e in self.real_edges()]
        if false_splits_only:
            labels = [l for l in labels if l[1] == 2]
        ids, ls = map(array,zip(*labels))
        ar[ids] = ls.astype(ar.dtype)
        return ar.reshape(self.watershed.shape)


    def split_vi(self, gt=None):
        if self.gt is None and gt is None:
            return array([0,0])
        elif self.gt is not None:
            return split_vi(None, None, self.rig)
        else:
            return split_vi(self.get_segmentation(), gt, None, [0], [0])


    def boundary_indices(self, n1, n2):
        return list(self[n1][n2]['boundary'])


    def get_edge_coordinates(self, n1, n2, arbitrary=False):
        """Find where in the segmentation the edge (n1, n2) is most visible."""
        return get_edge_coordinates(self, n1, n2, arbitrary)


    def write(self, fout, output_format='GraphML'):
        if output_format == 'Plaza JSON':
            self.write_plaza_json(fout)
        else:
            raise ValueError('Unsupported output format for agglo.Rag: %s'
                % output_format)


    def write_plaza_json(self, fout, synapsejson=None, offsetz=0):
        """Write graph to Steve Plaza's JSON spec."""
        json_vals = {}
        if synapsejson is not None:
            synapse_file = open(synapsejson)
            json_vals1 = json.load(synapse_file)
            body_count = {}

            for item in json_vals1["data"]:
                bodyid = ((item["T-bar"])["body ID"])
                if bodyid in body_count:
                    body_count[bodyid] += 1
                else:
                    body_count[bodyid] = 1
                for psd in item["partners"]:
                    bodyid = psd["body ID"]
                    if bodyid in body_count:
                        body_count[bodyid] += 1
                    else:
                        body_count[bodyid] = 1

            json_vals["synapse_bodies"] = []
            for body, count in body_count.items():
                temp = [body, count]
                json_vals["synapse_bodies"].append(temp)

        edge_list = [
            {'location': map(int, self.get_edge_coordinates(i, j)[-1::-1]),
            'node1': int(i), 'node2': int(j),
            'edge_size': len(self[i][j]['boundary']),
            'size1': self.node[i]['size'],
            'size2': self.node[j]['size'],
            'weight': float(self[i][j]['weight'])}
            for i, j in self.real_edges()
        ]
        json_vals['edge_list'] = edge_list

        with open(fout, 'w') as f:
            json.dump(json_vals, f, indent=4)


    def ncut(self, num_clusters=10, kmeans_iters=5, sigma=255.0*20, nodes=None,
            **kwargs):
        """Run normalized cuts on the current set of superpixels.
           Keyword arguments:
               num_clusters -- number of clusters to compute
               kmeans_iters -- # iterations to run kmeans when clustering
               sigma -- sigma value when setting up weight matrix
           Return value: None
        """
        if nodes is None:
            nodes = self.nodes()
        # Compute weight matrix
        W = self.compute_W(self.merge_priority_function, nodes=nodes)
        # Run normalized cut
        labels, eigvec, eigval = ncutW(W, num_clusters, kmeans_iters, **kwargs)
        # Merge nodes that are in same cluster
        self.cluster_by_labels(labels, nodes)


    def cluster_by_labels(self, labels, nodes=None):
        """Merge all superpixels with the same label (1 label per 1 sp)"""
        if nodes is None:
            nodes = array(self.nodes())
        if not (len(labels) == len(nodes)):
            raise ValueError('Number of labels should be %d but is %d.',
                self.number_of_nodes(), len(labels))
        for l in unique(labels):
            inds = nonzero(labels==l)[0]
            nodes_to_merge = nodes[inds]
            node1 = nodes_to_merge[0]
            for node in nodes_to_merge[1:]:
                self.merge_nodes(node1, node)


    def compute_W(self, merge_priority_function, sigma=255.0*20, nodes=None):
        """ Computes the weight matrix for clustering"""
        if nodes is None:
            nodes = array(self.nodes())
        n = len(nodes)
        nodes2ind = dict(zip(nodes, range(n)))
        W = lil_matrix((n,n))
        for u, v in self.real_edges(nodes):
            try:
                i, j = nodes2ind[u], nodes2ind[v]
            except KeyError:
                continue
            w = merge_priority_function(self,u,v)
            W[i,j] = W[j,i] = exp(-w**2/sigma)
        return W


    def update_frozen_sets(self, n1, n2):
        self.frozen_nodes.discard(n1)
        self.frozen_nodes.discard(n2)
        for x, y in self.frozen_edges.copy():
            if n2 in [x, y]:
                self.frozen_edges.discard((x, y))
            if x == n2:
                self.frozen_edges.add((n1, y))
            if y == n2:
                self.frozen_edges.add((x, n1))


def get_edge_coordinates(g, n1, n2, arbitrary=False):
    """Find where in the segmentation the edge (n1, n2) is most visible."""
    boundary = g[n1][n2]['boundary']
    if arbitrary:
        # quickly get an arbirtrary point on the boundary
        idx = boundary.pop(); boundary.add(idx)
        coords = unravel_index(idx, g.watershed.shape)
    else:
        boundary_idxs = unravel_index(list(boundary), g.watershed.shape)
        coords = [bincount(dimcoords).argmax() for dimcoords in boundary_idxs]
    return array(coords) - g.pad_thickness


def is_mito_boundary(g, n1, n2, channel=2, threshold=0.5):
        return max(np.mean(g.probabilities_r[list(g[n1][n2]["boundary"]), c]) \
        for c in channel) > threshold


def is_mito(g, n, channel=2, threshold=0.5):
        return max(np.mean(g.probabilities_r[list(g.extent(n)), c]) \
        for c in channel) > threshold


def best_possible_segmentation(ws, gt):
    """Build the best possible segmentation given a superpixel map."""
    cnt = contingency_table(ws, gt)
    assignment = cnt == cnt.max(axis=1)[:,newaxis]
    hard_assignment = where(assignment.sum(axis=1) > 1)[0]
    # currently ignoring hard assignment nodes
    assignment[hard_assignment,:] = 0
    ws = Rag(ws)
    for gt_node in range(1,cnt.shape[1]):
        ws.merge_subgraph(where(assignment[:,gt_node])[0])
    return ws.get_segmentation()

