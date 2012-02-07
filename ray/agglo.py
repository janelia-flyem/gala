# built-ins
from itertools import combinations, izip, repeat, product
import argparse
import random
import sys
import logging
import json
from copy import deepcopy
from math import isnan

# libraries
from numpy import array, mean, zeros, zeros_like, uint8, int8, where, unique, \
    ones_like, finfo, size, double, transpose, newaxis, uint32, nonzero, \
    median, exp, ceil, dot, log2, float, ones, arange, inf, flatnonzero, \
    intersect1d, dtype, squeeze, sqrt, reshape, setdiff1d, argmin, sign, \
    concatenate, nan, __version__ as numpyversion, unravel_index, bincount
import numpy
from scipy.stats import sem
from scipy.sparse import lil_matrix
from scipy.ndimage import generate_binary_structure, iterate_structure, \
    distance_transform_cdt
from scipy.misc import comb as nchoosek
from scipy.ndimage.measurements import center_of_mass, label
from networkx import Graph, biconnected_components
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
from networkx.algorithms.components.connected import connected_components

# local modules
import morpho
import iterprogress as ip
from ncut import ncutW
from mergequeue import MergeQueue
from evaluate import contingency_table, split_vi, xlogx
from classify import NullFeatureManager, MomentsFeatureManager, \
    HistogramFeatureManager, RandomForest, unique_learning_data_elements, \
    concatenate_data_elements

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

class Rag(Graph):
    """Region adjacency graph for segmentation of nD volumes."""

    def __init__(self, watershed=array([]), probabilities=array([]), 
            merge_priority_function=None, allow_shared_boundaries=True,
            gt_vol=None, feature_manager=MomentsFeatureManager(), 
            show_progress=False, lowmem=False, connectivity=1,
            channel_is_oriented=None, orientation_map=array([]),
            normalize_probabilities=False):
        """Create a graph from a watershed volume and image volume.
        
        The watershed is assumed to have dams of label 0 in between basins.
        Then, each basin corresponds to a node in the graph and an edge is
        placed between two nodes if there are one or more watershed pixels
        connected to both corresponding basins.
        """
        super(Rag, self).__init__(weighted=False)
        self.show_progress = show_progress
        if merge_priority_function is None:
            self.merge_priority_function = boundary_mean
        else:
            self.merge_priority_function = merge_priority_function
        self.set_watershed(watershed, lowmem, connectivity)
        self.set_probabilities(probabilities, normalize_probabilities)
        self.set_orientations(orientation_map, channel_is_oriented)
        if watershed is None:
            self.ucm = None
        else:
            self.ucm = array(self.watershed==0, dtype=float)
            self.ucm[self.ucm==0] = -inf
            self.ucm_r = self.ucm.ravel()
        self.max_merge_score = -inf
        self.build_graph_from_watershed(allow_shared_boundaries)
        self.set_feature_manager(feature_manager)
        self.set_ground_truth(gt_vol)
        self.merge_queue = MergeQueue()

    def __copy__(self):
        if sys.version_info[:2] < (2,7):
            # Python versions prior to 2.7 don't handle deepcopy of function
            # objects well. Thus, keep a reference and remove from Rag object
            f = self.neighbor_idxs; del self.neighbor_idxs
            F = self.feature_manager; del self.feature_manager
        pr_shape = self.probabilities_r.shape
        g = super(Rag, self).copy()
        g.watershed_r = g.watershed.ravel()
        g.segmentation_r = g.segmentation.ravel()
        g.ucm_r = g.ucm.ravel()
        g.probabilities_r = g.probabilities.reshape(pr_shape)
        if sys.version_info[:2] < (2,7):
            g.neighbor_idxs = f
            self.neighbor_idxs = f
            g.feature_manager = F
            self.feature_manager = F
        return g

    def copy(self):
        return self.__copy__()

    def real_edges(self, *args, **kwargs):
        return [e for e in super(Rag, self).edges(*args, **kwargs) if
                                            self.boundary_body not in e[:2]]

    def real_edges_iter(self, *args, **kwargs):
        return (e for e in super(Rag, self).edges_iter(*args, **kwargs) if
                                            self.boundary_body not in e[:2])

    def build_graph_from_watershed(self, 
                                    allow_shared_boundaries=True, idxs=None):
        if self.watershed.size == 0: return # stop processing for empty graphs
        if not allow_shared_boundaries:
            self.ignored_boundary = zeros(self.watershed.shape, bool)
        if idxs is None:
            idxs = arange(self.watershed.size)
            self.add_node(self.boundary_body, 
                    extent=set(flatnonzero(self.watershed==self.boundary_body)))
        inner_idxs = idxs[self.watershed_r[idxs] != self.boundary_body]
        pbar = ip.StandardProgressBar() if self.show_progress \
                                        else ip.NoProgressBar()
        for idx in ip.with_progress(inner_idxs, title='Graph... ', pbar=pbar):
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
                try:
                    self.node[nodeid]['extent'].add(idx)
                except KeyError:
                    self.node[nodeid]['extent'] = set([idx])
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
        self.feature_manager = feature_manager
        if len(self.feature_manager) > 0:
            self.compute_feature_caches()

    def compute_feature_caches(self):
        for n in self.nodes_iter():
            self.node[n]['feature-cache'] = \
                            self.feature_manager.create_node_cache(self, n)
        for n1, n2 in self.edges_iter():
            self[n1][n2]['feature-cache'] = \
                            self.feature_manager.create_edge_cache(self, n1, n2)

    def get_neighbor_idxs_fast(self, idxs):
        return self.pixel_neighbors[idxs]

    def get_neighbor_idxs_lean(self, idxs, connectivity=1):
        return morpho.get_neighbor_idxs(self.watershed, idxs, connectivity)

    def set_probabilities(self, probs=array([]), normalize=False):
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
        if len(orientation_map) == 0:
            self.orientation_map = zeros_like(self.watershed)
            self.orientation_map_r = self.orientation_map.ravel()
        so = orientation_map.shape
        sw = tuple(array(self.watershed.shape, dtype=int)-\
                2*self.pad_thickness*ones(self.watershed.ndim, dtype=int))
        o_ndim = orientation_map.ndim
        w_ndim = self.watershed.ndim
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
        self.segmentation = self.watershed.copy()
        self.watershed_r = self.watershed.ravel()
        self.segmentation_r = self.segmentation.ravel() # reduce fct calls
        self.pad_thickness = 2 if (self.segmentation==0).any() else 1
        if lowmem:
            def neighbor_idxs(x): 
                return self.get_neighbor_idxs_lean(x, connectivity)
            self.neighbor_idxs = neighbor_idxs
        else:
            self.pixel_neighbors = \
                morpho.build_neighbors_array(self.watershed, connectivity)
            self.neighbor_idxs = self.get_neighbor_idxs_fast

    def set_ground_truth(self, gt=None):
        if gt is not None:
            gtm = gt.max()+1
            gt_ignore = [0, gtm] if (gt==0).any() else [gtm]
            seg_ignore = [0, self.boundary_body] if \
                        (self.segmentation==0).any() else [self.boundary_body]
            self.gt = morpho.pad(gt, gt_ignore)
            self.rig = contingency_table(self.segmentation, self.gt)
            self.rig[:, gt_ignore] = 0
            self.rig[seg_ignore, :] = 0
        else:
            self.gt = None
            # null pattern to transparently allow merging of nodes.
            # Bonus feature: counts how many sp's went into a single node.
            try:
                self.rig = ones(self.watershed.max()+1)
            except ValueError:
                self.rig = ones(self.number_of_nodes()+1)

    def build_merge_queue(self):
        """Build a queue of node pairs to be merged in a specific priority.
        
        The queue elements have a specific format in order to allow 'removing'
        of specific elements inside the priority queue. Each element is a list
        of length 4 containing:
            - the merge priority (any ordered type)
            - a 'valid' flag
            - and the two nodes in arbitrary order
        The valid flag allows one to "remove" elements by setting the flag to
        False. Then one checks the flag when popping elements and ignores those
        marked as invalid.

        One other specific feature is that there are back-links from edges to
        their corresponding queue items so that when nodes are merged,
        affected edges can be invalidated and reinserted in the queue.
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
        """Build a merge queue from scratch and assign to self.merge_queue."""
        self.merge_queue = self.build_merge_queue()

    def agglomerate(self, threshold=0.5, save_history=False):
        """Merge nodes sequentially until given edge confidence threshold."""
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history, scores, evaluation = [], [], []
        while len(self.merge_queue) > 0 and \
                                        self.merge_queue.peek()[0] < threshold:
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                self.update_ucm(n1,n2,merge_priority)
                self.merge_nodes(n1,n2)
                if save_history: 
                    history.append((n1,n2))
                    scores.append(merge_priority)
                    evaluation.append(
                        (self.number_of_nodes()-1, self.split_vi())
                    )
        if save_history:
            return history, scores, evaluation

    def agglomerate_count(self, stepsize=100, save_history=False):
        """Agglomerate until 'stepsize' merges have been made."""
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history, evaluation = [], []
        i = 0
        while len(self.merge_queue) > 0 and i < stepsize:
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                i += 1
                self.update_ucm(n1,n2,merge_priority)
                self.merge_nodes(n1,n2)
                if save_history: 
                    history.append((n1,n2))
                    evaluation.append(
                        (self.number_of_nodes()-1, self.split_vi())
                    )
        if save_history:
            return history, evaluation
        
    def agglomerate_ladder(self, threshold=1000, strictness=1):
        """Merge sequentially all nodes smaller than threshold.
        
        strictness = 1 only considers size of nodes
        strictness = 2 adds additional constraint: small nodes can only be 
        merged to large neighbors
        strictness = 3 additionally requires that the boundary between nodes
        be larger than 2 pixels
        Note: nodes that are on the volume boundary are not agglomerated.
        """
        original_merge_priority_function = self.merge_priority_function
        self.merge_priority_function = make_ladder(
            self.merge_priority_function, threshold, strictness
        )
        self.rebuild_merge_queue()
        self.agglomerate(inf)
        self.merge_priority_function = original_merge_priority_function
        self.merge_queue.finish()
        self.rebuild_merge_queue()
        
    def one_shot_agglomeration(self, threshold=0.5):
        g = self.copy()
        if len(g.merge_queue) == 0:
            g.rebuild_merge_queue()
        for u, v, d in g.edges(data=True):
            if g.boundary_body in [u,v] or d['weight'] > threshold:
                g.remove_edge(u, v)
        ccs = connected_components(g)
        for cc in ccs:
            g.merge_subgraph(cc)
        return g.get_segmentation()

    def assign_gt_to_ws(self, gt):
        ws_nopad = morpho.juicy_center(self.watershed, self.pad_thickness)
        bdrymap = morpho.pad(morpho.seg_to_bdry(ws_nopad), 
                                                    [0]*self.pad_thickness)
        gt_bdrymap_nopad = morpho.seg_to_bdry(gt)
        gt_bdrymap = morpho.pad(gt_bdrymap_nopad, [0]*self.pad_thickness)
        k = distance_transform_cdt(1-bdrymap, return_indices=True)
        ind = nonzero(gt_bdrymap.ravel())[0]
        closest_sub = numpy.concatenate(
                                [i.ravel()[:,newaxis] for i in k[1]],axis=1)
        closest_sub = closest_sub[ind,:]
        closest_ind = [dot(bdrymap.strides, i)/bdrymap.itemsize 
                                                        for i in closest_sub]
        M = zeros_like(bdrymap).astype(float)
        M.ravel()[closest_ind]=1.0
        bdrymap.ravel()[closest_ind] = False
        k = distance_transform_cdt(1-bdrymap, return_indices=True)
        ind = nonzero(gt_bdrymap.ravel())[0]
        closest_sub = numpy.concatenate(
                                [i.ravel()[:,newaxis] for i in k[1]],axis=1)
        closest_sub = closest_sub[ind,:]
        closest_ind = [dot(bdrymap.strides, i)/bdrymap.itemsize 
                                                        for i in closest_sub]
        M.ravel()[closest_ind]=1.0 
        return M 
        
    def learn_agglomerate(self, gts, feature_map, min_num_samples=1,
                                                            *args, **kwargs):
        """Agglomerate while comparing to ground truth & classifying merges."""
        learn_flat = kwargs.get('learn_flat', True)
        learning_mode = kwargs.get('learning_mode', 'strict').lower()
        labeling_mode = kwargs.get('labeling_mode', 'assignment').lower()
        priority_mode = kwargs.get('priority_mode', 'random').lower()
        memory = kwargs.get('memory', True)
        unique = kwargs.get('unique', True)
        max_numepochs = kwargs.get('max_numepochs', 10)
        if priority_mode == 'mean' and unique: 
            max_numepochs = 2 if learn_flat else 1
        if priority_mode in ['random', 'mean'] and not memory:
            max_numepochs = 1
        label_type_keys = {'assignment':0, 'vi-sign':1, 
                                                'rand-sign':2, 'boundary':3}
        if type(gts) != list:
            gts = [gts] # allow using single ground truth as input
        master_ctables = \
                [contingency_table(self.get_segmentation(), gt) for gt in gts]
        # Match the watershed to the ground truths
        ws_is_gt = zeros_like(self.watershed).astype(float)
        for gt in gts:
            ws_is_gt += self.assign_gt_to_ws(gt)
        ws_is_gt /= float(len(gts))
        ws_is_gt = ws_is_gt>0.5
        alldata = []
        data = [[],[],[],[]]
        for numepochs in range(max_numepochs):
            ctables = deepcopy(master_ctables)
            if len(data[0]) > min_num_samples:
                break
            if learn_flat and numepochs == 0:
                alldata.append(self.learn_flat(gts, feature_map, ws_is_gt))
                data = unique_learning_data_elements(alldata) if unique else \
                       alldata[-1]
                continue
            g = self.copy()
            if priority_mode == 'mean':
                g.merge_priority_function = boundary_mean
            elif numepochs > 0 and priority_mode == 'active' or \
                numepochs % 2 == 1 and priority_mode == 'mixed':
                cl = kwargs.get('classifier', RandomForest())
                cl = cl.fit(data[0], data[1][:,label_type_keys[labeling_mode]])
                if type(cl) == RandomForest:
                    logging.info('classifier oob error: %.2f'%cl.oob)
                g.merge_priority_function = \
                                        classifier_probability(feature_map, cl)
            elif priority_mode == 'random' or \
                (priority_mode == 'active' and numepochs == 0):
                g.merge_priority_function = random_priority
            elif priority_mode == 'custom':
                g.merge_priority_function = kwargs.get('mpf', boundary_mean)
            g.show_progress = False # bug in MergeQueue usage causes
                                    # progressbar crash.
            g.rebuild_merge_queue()
            alldata.append(g._learn_agglomerate(ctables, feature_map, ws_is_gt,
                                                learning_mode, labeling_mode))
            if memory:
                if unique:
                    data = unique_learning_data_elements(alldata) 
                else:
                    data = concatenate_data_elements(alldata)
            else:
                data = alldata[-1]
            logging.debug('data size %d at epoch %d'%(len(data[0]), numepochs))
        return data, alldata

    def learn_flat(self, gts, feature_map, ws_is_gt, *args, **kwargs):
        if type(gts) != list:
            gts = [gts] # allow using single ground truth as input
        ctables = [contingency_table(self.get_segmentation(), gt) for gt in gts]
        assignments = [(ct == ct.max(axis=1)[:,newaxis]) for ct in ctables]
        return map(array, zip(*[
                self.learn_edge(e, ctables, assignments, feature_map, ws_is_gt)
                for e in self.real_edges()]))

    def learn_edge(self, edge, ctables, assignments, feature_map, ws_is_gt,
            boundary_overlap_thresh=0.3):
        n1, n2 = edge
        features = feature_map(self, n1, n2).ravel()
        # Calculate weights for weighting data points
        s1, s2 = [len(self.node[n]['extent']) for n in [n1, n2]]
        weights = \
            compute_local_vi_change(s1, s2, self.volume_size), \
            compute_local_rand_change(s1, s2, self.volume_size)
        # Get the fraction of times that n1 and n2 assigned to
        # same segment in the ground truths
        cont_labels = [
            [(-1)**(a[n1,:]==a[n2,:]).all() for a in assignments],
            [compute_true_delta_vi(ctable, n1, n2) for ctable in ctables],
            [-compute_true_delta_rand(ctable, n1, n2, self.volume_size) 
                                                    for ctable in ctables],
            [(self.compute_boundary_overlap_with_gt(n1,n2, ws_is_gt)>
                                            boundary_overlap_thresh)*2 - 1] 
        ]
        labels = [sign(mean(cont_label)) for cont_label in cont_labels]
        if any(map(isnan, labels)) or any([label == 0 for l in labels]):
            logging.debug('NaN or 0 labels found. ' + 
                                    ' '.join(map(str, [labels, (n1, n2)])))
        labels = [1 if i==0 or isnan(i) else i for i in labels]
        return features, labels, weights, (n1,n2)

    def compute_boundary_overlap_with_gt(self, n1, n2, ws_is_gt):
        val = []
        val = ws_is_gt.ravel()[list(self[n1][n2]['boundary'])]
        return sum(val)/float(len(val)) 
    
    def _learn_agglomerate(self, ctables, feature_map, gt_dts, 
                        learning_mode='forbidden', labeling_mode='assignment'):
        """Learn the agglomeration process using various strategies.

        Arguments:
            - one or more contingency tables between own segments and gold
            standard segmentations
            - a feature map function {Graph, node1, node2} |--> array([float])
            [- a learning mode]

        Value:
            A learning data matrix of shape 
            [n_training_examples x (n_features + 5)]. The elements after the
            features are the label, the approximate magnitude of the variation
            of information (VI) change, the approximate magnitude of the Rand
            index (RI) change, and the two nodes that were sampled.

        Learning modes:
            - strict: use positive-boundary examples to learn but never
            merge
            - loose: merge regardless of label
        Labeling modes:
            - assignment: assign each node to a gold standard node and 
            - vi-sign: compute the vi change resulting from merging candidate
            regions. Use the sign of the change as the training label.
            - rand-sign: compute the rand change resulting from merging the
            candidate regions. Use the sign of the change as the training
            label.
        """
        label_type_keys = {'assignment':0, 'vi-sign':1, 'rand-sign':2, 
                                                                'boundary':3}
        assignments = [(ct == ct.max(axis=1)[:,newaxis]) for ct in ctables]
        g = self
        data = []
        while len(g.merge_queue) > 0:
            merge_priority, valid, n1, n2 = g.merge_queue.pop()
            if valid:
                dat = g.learn_edge(
                            (n1,n2), ctables, assignments, feature_map, gt_dts)
                data.append(dat)
                label = dat[1][label_type_keys[labeling_mode]]
                if learning_mode != 'strict' or label < 0:
                    for ctable, assignment in zip(ctables, assignments):
                        ctable[n1] += ctable[n2]
                        ctable[n2] = 0
                        assignment[n1] = ctable[n1] == ctable[n1].max()
                        assignment[n2] = 0
                    g.merge_nodes(n1, n2)
        return map(array, zip(*data))

    def replay_merge_history(self, merge_seq, labels=None, num_errors=1):
        """Agglomerate according to a merge sequence, optionally labeled.
        
        The merge sequence and labels _must_ be generators if you don't want
        to manually keep track of how much has been consumed. The merging
        continues until num_errors false merges have been encountered, or 
        until the sequence is fully consumed.
        
        labels are -1 or 0 for 'should merge', 1 for 'should not merge'.
        
        Return value: number of elements consumed from merge_seq, and last
        merge pair observed.
        """
        if labels is None:
            labels1 = itertools.repeat(False)
            labels2 = itertools.repeat(False)
        else:
            labels1 = (label > 0 for label in labels)
            labels2 = (label > 0 for label in labels)
        counter = itertools.count()
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

    def update_ucm(self, n1, n2, score=-inf):
        """Update ultrametric contour map."""
        if self.ucm is not None:
            self.max_merge_score = max(self.max_merge_score, score)
            idxs = list(self[n1][n2]['boundary'])
            self.ucm_r[idxs] = self.max_merge_score

    def merge_nodes(self, n1, n2):
        """Merge two nodes, while updating the necessary edges."""
        self.node[n1]['extent'].update(self.node[n2]['extent'])
        self.feature_manager.update_node_cache(self, n1, n2,
                self.node[n1]['feature-cache'], self.node[n2]['feature-cache'])
        self.segmentation_r[list(self.node[n2]['extent'])] = n1
        new_neighbors = [n for n in self.neighbors(n2)
                                        if n not in [n1, self.boundary_body]]
        for n in new_neighbors:
            self.merge_edge_properties((n2,n), (n1,n))
        # this if statement enables merging of non-adjacent nodes
        if self.has_edge(n1,n2) and self.has_zero_boundaries:
            self.refine_post_merge_boundaries(n1, n2)
        self.rig[n1] += self.rig[n2]
        self.rig[n2] = 0
        self.remove_node(n2)

    def refine_post_merge_boundaries(self, n1, n2):
        boundary = array(list(self[n1][n2]['boundary']))
        boundary_neighbor_pixels = self.segmentation_r[
            self.neighbor_idxs(boundary)
        ]
        add = ( (boundary_neighbor_pixels == 0) + 
            (boundary_neighbor_pixels == n1) + 
            (boundary_neighbor_pixels == n2) ).all(axis=1)
        check = True-add
        self.node[n1]['extent'].update(boundary[add])
        boundary_probs = self.probabilities_r[boundary[add]]
        self.feature_manager.pixelwise_update_node_cache(self, n1,
                        self.node[n1]['feature-cache'], boundary[add])
        self.segmentation_r[boundary[add]] = n1
        boundaries_to_edit = {}
        for px in boundary[check]:
            for lb in unique(
                        self.segmentation_r[self.neighbor_idxs(px)]):
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
        if type(subgraph) not in [Rag, Graph]: # input is node list
            subgraph = self.subgraph(subgraph)
        if len(subgraph) > 0:
            node_dfs = list(dfs_preorder_nodes(subgraph, source)) 
            # dfs_preorder_nodes returns iter, convert to list
            source_node, other_nodes = node_dfs[0], node_dfs[1:]
            for current_node in other_nodes:
                self.merge_nodes(source_node, current_node)

    def split_node(self, u, n=2, **kwargs):
        node_extent = list(self.node[u]['extent'])
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
        """Merge the properties of edge src into edge dst."""
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
        """Update the merge queue item for edge (u,v). Add new by default."""
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
        return morpho.juicy_center(self.segmentation, self.pad_thickness)

    def get_ucm(self):
        if hasattr(self, 'ignored_boundary'):
            self.ucm[self.ignored_boundary] = self.max_merge_score
        return morpho.juicy_center(self.ucm, self.pad_thickness)    

    def build_volume(self, nbunch=None):
        """Return the segmentation (numpy.ndarray) induced by the graph."""
        v = zeros_like(self.watershed)
        vr = v.ravel()
        if nbunch is None:
            nbunch = self.nodes()
        for n in nbunch:
            vr[list(self.node[n]['extent'])] = n
        return morpho.juicy_center(v,self.pad_thickness)

    def build_boundary_map(self, ebunch=None):
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
        """Merge any segments fully contained within other segments."""
        bcc = list(biconnected_components(self))
        container = [i for i, s in enumerate(bcc) if self.boundary_body in s][0]
        del bcc[container] # remove the main graph
        bcc = map(list, bcc)
        for cc in bcc:
            cc.sort(key=lambda x: len(self.node[x]['extent']), reverse=True)
        bcc.sort(key=lambda x: len(self.node[x[0]]['extent']))
        for cc in bcc:
            self.merge_subgraph(cc, cc[0])

    def orphans(self):
        """List of all the nodes that do not touch the volume boundary."""
        return [n for n in self.nodes() if not self.at_volume_boundary(n)]

    def is_traversed_by_node(self, n):
        """Determine whether a body traverses the volume.
        
        This is defined as touching the volume boundary at two distinct 
        locations.
        """
        if not self.at_volume_boundary(n) or n == self.boundary_body:
            return False
        v = zeros(self.segmentation.shape, uint8)
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

    def raveler_body_annotations(self, traverse=False):
        """Return JSON-compatible dict formatted for Raveler annotations."""
        orphans = self.orphans()
        non_traversing_bodies = self.non_traversing_bodies() if traverse else []
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
        
    def write_plaza_json(self, fout):
        """Write graph to Steve Plaza's JSON spec."""
        edge_list = [
            {'location': map(int, self.get_edge_coordinates(i, j)[-1::-1]),
            'node1': int(i), 'node2': int(j),
            'edge_size': len(self[i][j]['boundary']),
            'size1': len(self.node[i]['extent']), 
            'size2': len(self.node[j]['extent']),
            'weight': float(self[i][j]['weight'])}
            for i, j in self.real_edges()
        ]
        with open(fout, 'w') as f:
            json.dump({'edge_list': edge_list}, f, indent=4)

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
                self.merge_nodes(node1,node)
                
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
    n, sum_xs = g[n1][n2]['feature-cache'][0:2]
    return sum_xs/n

def make_ladder(priority_function, threshold, strictness=1):
    def ladder_function(g, n1, n2):
        s1 = len(g.node[n1]['extent'])
        s2 = len(g.node[n2]['extent'])
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

def classifier_probability(feature_extractor, classifier):
    def predict(g, n1, n2):
        if n1 == g.boundary_body or n2 == g.boundary_body:
            return inf
        features = feature_extractor(g, n1, n2)
        try:
            prediction = classifier.predict_proba(features)[0,1]
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
            len(g.node[n1]['extent']), len(g.node[n2]['extent']), g.volume_size
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
            len(g.node[n1]['extent']), len(g.node[n2]['extent']), g.volume_size
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
