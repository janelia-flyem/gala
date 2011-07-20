import itertools
combinations, izip = itertools.combinations, itertools.izip
from itertools import combinations, izip
import argparse
import random
import matplotlib.pyplot as plt
from heapq import heapify, heappush, heappop
from numpy import array, mean, zeros, zeros_like, uint8, int8, where, unique, \
    finfo, size, double, transpose, newaxis, uint32, nonzero, median, exp, \
    log2, float, ones, arange, inf
from scipy.stats import sem
from scipy.sparse import lil_matrix
from scipy.misc import comb as nchoosek
from scipy.ndimage.measurements import center_of_mass, label
from networkx import Graph
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
import morpho
import iterprogress as ip
from ncut import ncutW
from mergequeue import MergeQueue
from evaluate import contingency_table, split_voi

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

    def __init__(self, watershed=None, probabilities=None, 
            merge_priority_function=None, allow_shared_boundaries=True,
            gt_vol=None,
            edge_feature_init_fct=None, edge_feature_merge_fct=None, 
            node_feature_init_fct=None, node_feature_merge_fct=None,
            show_progress=False, lowmem=False):
        """Create a graph from a watershed volume and image volume.
        
        The watershed is assumed to have dams of label 0 in between basins.
        Then, each basin corresponds to a node in the graph and an edge is
        placed between two nodes if there are one or more watershed pixels
        connected to both corresponding basins.
        """
        super(Rag, self).__init__(weighted=False)
        self.boundary_probability = 10**100 #inconceivably high, but no overflow
        if probabilities is not None:
            self.set_probabilities(probabilities)
        self.show_progress = show_progress
        if merge_priority_function is None:
            self.merge_priority_function = boundary_mean
        else:
            self.merge_priority_function = merge_priority_function
        self.set_watershed(watershed, lowmem)
	if watershed is None:
	    self.ucm = None
	else:
            self.ucm = array(self.watershed==0, dtype=float)
	    self.ucm[self.ucm==0] = -inf
	self.max_merge_score = -inf
	self.build_graph_from_watershed(allow_shared_boundaries)
        self.set_ground_truth(gt_vol)
        self.merge_queue = MergeQueue()

    def build_graph_from_watershed(self, 
                                    allow_shared_boundaries=True, idxs=None):
        if self.watershed is None:
            return
        if idxs is None:
            idxs = arange(self.watershed.size)
        zero_idxs = idxs[self.watershed.ravel()[idxs] == 0]
        if self.show_progress:
            def with_progress(seq, length=None, title='Progress: '):
                return ip.with_progress(seq, length, title,
                                                ip.StandardProgressBar())
        else:
            def with_progress(seq, length=None, title='Progress: '):
                return ip.with_progress(seq, length, title, ip.NoProgressBar())
        if not hasattr(self, 'probabilities'):
            self.probabilities = zeros(self.watershed.shape, uint8)
        for idx in with_progress(zero_idxs, title='Building edges... '):
            ns = self.neighbor_idxs(idx)
            adj_labels = self.watershed.ravel()[ns]
            adj_labels = unique(adj_labels[adj_labels != 0])
            if len(adj_labels) > 2 and not allow_shared_boundaries:
                continue
            p = double(self.probabilities.ravel()[idx])
            for l1,l2 in combinations(adj_labels, 2):
                if self.has_edge(l1, l2): 
                    self[l1][l2]['boundary'].add(idx)
                    self[l1][l2]['sump'] += p
                    self[l1][l2]['sump2'] += p*p
                    self[l1][l2]['sump3'] += p*p*p
                    self[l1][l2]['n'] += 1
                else: 
                    self.add_edge(l1, l2, 
                        boundary=set([idx]), sump=p, sump2=p*p, sump3=p*p*p, n=1
                    )
        nonzero_idxs = idxs[self.watershed.ravel()[idxs] != 0]
        for idx in with_progress(nonzero_idxs, title='Building nodes... '):
            nodeid = self.watershed.ravel()[idx]
            if not allow_shared_boundaries and not self.has_node(nodeid):
                self.add_node(nodeid)
            p = double(self.probabilities.ravel()[idx])
            try:
                self.node[nodeid]['extent'].add(idx)
                self.node[nodeid]['sump'] += p
                self.node[nodeid]['sump2'] += p*p
                self.node[nodeid]['sump3'] += p*p*p
            except KeyError:
                self.node[nodeid]['extent'] = set([idx])
                self.node[nodeid]['sump'] = p
                self.node[nodeid]['sump2'] = p*p
                self.node[nodeid]['sump3'] = p*p*p
                self.node[nodeid]['absorbed'] = [nodeid]

    def get_neighbor_idxs_fast(self, idxs):
        return self.pixel_neighbors[idxs]

    def get_neighbor_idxs_lean(self, idxs):
        return morpho.get_neighbor_idxs(self.watershed, idxs)

    def set_probabilities(self, probs):
        self.probabilities = morpho.pad(probs, [self.boundary_probability, 0])

    def set_watershed(self, ws=None, lowmem=False):
        if ws is None:
            self.watershed = None
            return
        self.boundary_body = ws.max()+1
        self.size = ws.size
        self.watershed = morpho.pad(ws, [0, self.boundary_body])
        self.segmentation = self.watershed.copy()
        if lowmem:
            self.neighbor_idxs = self.get_neighbor_idxs_lean
        else:
            self.pixel_neighbors = morpho.build_neighbors_array(self.watershed)
            self.neighbor_idxs = self.get_neighbor_idxs_fast
        self.sum_body_sizes = double(self.get_segmentation().astype(bool).sum())

    def set_ground_truth(self, gt=None):
        if gt is not None:
            self.gt = morpho.pad(gt, [0, gt.max()+1])
            self.rig = contingency_table(self.watershed, self.gt)
            self.rig[[0,self.boundary_body],:] = 0
            self.rig[:,[0, gt.max()+1]] = 0
        else:
            self.gt = None
            # null pattern to transparently allow merging of nodes.
            # Bonus feature: counts how many sp's went into a single node.
            self.rig = ones(self.number_of_nodes())

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
        for l1, l2 in self.edges_iter():
            w = self.merge_priority_function(self,l1,l2)
            qitem = [w, True, l1, l2]
            queue_items.append(qitem)
            self[l1][l2]['qlink'] = qitem
            self[l1][l2]['weight'] = w
        return MergeQueue(queue_items, with_progress=self.show_progress)

    def rebuild_merge_queue(self):
        """Build a merge queue from scratch and assign to self.merge_queue."""
        self.merge_queue = self.build_merge_queue()

    def agglomerate(self, threshold=128, save_history=False, eval_function=None):
        """Merge nodes sequentially until given edge confidence threshold."""
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history = []
	evaluation = []
        while len(self.merge_queue) > 0 and \
                                        self.merge_queue.peek()[0] < threshold:
	    merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                self.merge_nodes(n1,n2,merge_score=merge_priority)
                if save_history: history.append((n1,n2))
		if eval_function is not None:
		    num_segs = len(unique(self.get_segmentation()))-1
		    val = eval_function(self.get_segmentation())
                    evaluation.append((num_segs, val))
	if (save_history) and (eval_function is not None):
	    return history, evaluation
	elif save_history:
	    return save_history
	elif eval_function is not None:
	    return evaluation

    def agglomerate_count(self, stepsize=100, save_history=False, eval_function=None):
        """Agglomerate until 'stepsize' merges have been made."""
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history = []
	evaluation = []
        i = 0
        while len(self.merge_queue) > 0 and i < stepsize:
	    merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                i += 1
                self.merge_nodes(n1, n2, merge_score=merge_priority)
                if save_history: history.append((n1,n2))
		if eval_function is not None: 
			num_segs = len(unique(self.get_segmentation()))-1
			val = eval_function(self.get_segmentation())
			evaluation.append((num_segs, val))
	if (save_history) and (eval_function is not None):
	    return history, evaluation
	elif save_history:
	    return history
  	elif eval_function is not None:
	    return evaluation
	
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
        self.agglomerate(self.boundary_probability/10)
        self.merge_priority_function = original_merge_priority_function
        self.merge_queue.finish()
        

    def learn_agglomerate(self, gt, feature_map_function, weight_type='voi'):
        """Agglomerate while comparing to ground truth & classifying merges."""
        gtg = Rag(gt)
        cnt = contingency_table(self.get_segmentation(), gtg.get_segmentation())
        assignment = cnt == cnt.max(axis=1)[:,newaxis]
        hard_assignment = where(assignment.sum(axis=1) > 1)[0]
        # 'hard assignment' nodes are nodes that have most of their overlap
        # with the 0-label in gt, or that have equal amounts of overlap between
        # two other labels
        if self.merge_queue.is_empty(): self.rebuild_merge_queue()
        features, labels, weights = [], [], []
        history, ave_size = [], []
        while self.number_of_nodes() > gtg.number_of_nodes():
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if merge_priority == self.boundary_probability or \
                                                self.boundary_body in [n1, n2]:
                print 'Warning: agglomeration done early...'
                break
            if valid:
                features.append(feature_map_function(self, n1, n2).ravel())
                if n2 in hard_assignment:
                    n1, n2 = n2, n1
		# Calculate weights for weighting data points
		n = self.size
		len1 = len(self.node[n1]['extent'])
		len2 = len(self.node[n2]['extent'])
		py1 = len1/float(n)
		py2 = len2/float(n)
		py = py1 + py2
		if weight_type == 'voi':
                    weight =  py*log2(py) - py1*log2(py1) - py2*log2(py2)
		elif weight_type == 'rand':
		    weight = (len1*len2)/float(nchoosek(n,2))
		elif weight_type == 'both':
		    weight = (py*log2(py) - py1*log2(py1) - py2*log2(py2), (len1*len2)/float(nchoosek(n,2)))
		else:
		    weight = 1.0		

                if n1 in hard_assignment and \
                                (assignment[n1,:] * assignment[n2,:]).any():
                    m = boundary_mean(self, n1, n2)
                    ms = [boundary_mean(self, n1, n) for n in 
                                                            self.neighbors(n1)]
                    if m == min(ms):
                        ave_size.append(self.sum_body_sizes / 
                                                (self.number_of_nodes()-1))
                        history.append([n2, n1])
                        self.merge_nodes(n2, n1)
                        labels.append(-1)
                        weights.append(weight)
                    else:
                        _ = features.pop() # remove last item
                else:
                    history.append([n1, n2])
                    ave_size.append(self.sum_body_sizes / 
                                                (self.number_of_nodes()-1))
                    if (assignment[n1,:] == assignment[n2,:]).all():
                        self.merge_nodes(n1, n2)
                        labels.append(-1)
                        weights.append(weight)
                    else:
                        labels.append(1)
                        weights.append(weight)
        return array(features).astype(double), array(labels), array(weights), \
                                            array(history), array(ave_size)

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

    def merge_nodes(self, n1, n2, merge_score=-inf):
        """Merge two nodes, while updating the necessary edges."""
        self.sum_body_sizes -= len(self.node[n1]['extent']) + \
                                len(self.node[n2]['extent'])
        # Update ultrametric contour map
	if self.ucm is not None:
	    self.max_merge_score = max(self.max_merge_score, merge_score)
	    try:
	        bdry = self[n1][n2]['boundary']
	        for i in bdry:
	            self.ucm[morpho.unravel_index(i, self.segmentation.shape)] = self.max_merge_score
	    except:
		pass
	new_neighbors = [n for n in self.neighbors(n2) if n != n1]
        for n in new_neighbors:
            if self.has_edge(n, n1):
                self[n1][n]['boundary'].update(self[n2][n]['boundary'])
                self[n1][n]['sump'] += self[n2][n]['sump']
                self[n1][n]['sump2'] += self[n2][n]['sump2']
                self[n1][n]['n'] += self[n2][n]['n']
            else:
                self.add_edge(n, n1, attr_dict=self[n2][n])
        self.node[n1]['extent'].update(self.node[n2]['extent'])
        self.node[n1]['sump'] += self.node[n2]['sump']
        self.node[n1]['sump2'] += self.node[n2]['sump2']
        self.segmentation.ravel()[list(self.node[n2]['extent'])] = n1
        for n in self.neighbors(n2):
            if n != n1:
                self.merge_edge_properties((n2,n), (n1,n))
        if self.has_edge(n1,n2):
            boundary = array(list(self[n1][n2]['boundary']))
            boundary_neighbor_pixels = self.segmentation.ravel()[
                self.neighbor_idxs(boundary)
            ]
            add = ( (boundary_neighbor_pixels == 0) + 
                (boundary_neighbor_pixels == n1) + 
                (boundary_neighbor_pixels == n2) ).all(axis=1)
            check = True-add
            self.node[n1]['extent'].update(boundary[add])
            boundary_probs = self.probabilities.ravel()[boundary[add]]
            self.node[n1]['sump'] += boundary_probs.sum()
            self.node[n1]['sump2'] += (boundary_probs*boundary_probs).sum()
            self.segmentation.ravel()[boundary[add]] = n1
            boundaries_to_edit = {}
            for px in boundary[check]:
                for lb in unique(
                            self.segmentation.ravel()[self.neighbor_idxs(px)]):
                    if lb != n1 and lb != 0:
                        try:
                            boundaries_to_edit[(n1,lb)].append(px)
                        except KeyError:
                            boundaries_to_edit[(n1,lb)] = [px]
            for u, v in boundaries_to_edit.keys():
                p = self.probabilities.ravel()[boundaries_to_edit[(u,v)]]\
                                                                .astype(double)
                if self.has_edge(u, v):
                    self[u][v]['boundary'].update(boundaries_to_edit[(u,v)])
                    self[u][v]['sump'] += p.sum()
                    self[u][v]['sump2'] += (p*p).sum()
                    self[u][v]['n'] += len(p)
                else:
                    self.add_edge(u, v, 
                        boundary=set(boundaries_to_edit[(u,v)]),
                        sump=p.sum(), sump2=(p*p).sum(), n=len(p)
                    )
                self.update_merge_queue(u, v)
            for n in new_neighbors:
                if not boundaries_to_edit.has_key((n1,n)):
                    self.update_merge_queue(n1, n)
        self.rig[n1] += self.rig[n2]
        self.rig[n2] = 0
        self.node[n1]['absorbed'].extend(self.node[n2]['absorbed'])
        self.remove_node(n2)
        self.sum_body_sizes += len(self.node[n1]['extent'])

    def split_node(self, u, n=2, **kwargs):
        node_extent = list(self.node[u]['extent'])
        node_borders = set().union(
                        *[self[u][v]['boundary'] for v in self.neighbors(u)])
        labels = unique(self.watershed.ravel()[node_extent])
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
            self[u][v]['sump'] += self[w][x]['sump']
            self[u][v]['sump2'] += self[w][x]['sump2']
            self[u][v]['n'] += self[w][x]['n']
        try:
            self.merge_queue.invalidate(self[w][x]['qlink'])
            self.update_merge_queue(u, v)
        except KeyError:
            pass

    def update_merge_queue(self, u, v):
        """Update the merge queue item for edge (u,v). Add new by default."""
        if self[u][v].has_key('qlink'):
            self.merge_queue.invalidate(self[u][v]['qlink'])
        if not self.merge_queue.is_null_queue:
            w = self.merge_priority_function(self,u,v)
            new_qitem = [w, True, u, v]
            self[u][v]['qlink'] = new_qitem
            self[u][v]['weight'] = w
            self.merge_queue.push(new_qitem)

    def show_merge_3D(self, n1, n2, **kwargs):
        """Show the 'best' view of a putative merge between given nodes."""
        im = self.image
        if kwargs.has_key('image'):
            im = kwargs['image']
        alpha = 0.7
        if kwargs.has_key('alpha'):
            alpha = kwargs['alpha']
        fignum = 1
        if kwargs.has_key('fignum'):
            fignum = kwargs['fignum']
        boundary = zeros(self.segmentation.shape, uint8)
        boundary_idxs = list(self[n1][n2]['boundary'])
        boundary.ravel()[boundary_idxs] = 3
        boundary.ravel()[list(self.node[n1]['extent'])] = 1
        boundary.ravel()[list(self.node[n2]['extent'])] = 2
        boundary = morpho.juicy_center(boundary, 2)
        x, y, z = array(center_of_mass(boundary==3)).round().astype(uint32)
        def imshow_grey(im):
            _ = plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        def imshow_jet_a(im):
            _ = plt.imshow(im, cmap=plt.cm.jet, 
                                        interpolation='nearest', alpha=alpha)
        fig = plt.figure(fignum)
        plt.subplot(221)
        imshow_grey(im[:,:,z])
        imshow_jet_a(boundary[:,:,z])
        plt.subplot(222)
        imshow_grey(im[:,y,:])
        imshow_jet_a(boundary[:,y,:])
        plt.subplot(223)
        imshow_grey(im[x,:,:])
        imshow_jet_a(boundary[x,:,:])
        plt.subplot(224)
        if kwargs.has_key('feature_map_function'):
            f = kwargs['feature_map_function']
            features = f(self, n1, n2)
            _ = plt.scatter(arange(len(features)), features)
        else:
            _ = plt.hist(self.probabilities.ravel()[boundary_idxs], bins=25)
        plt.title('feature vector. prob = %.4f' % 
                                self.merge_priority_function(self, n1, n2))
        return fig


    def get_segmentation(self):
        return morpho.juicy_center(self.segmentation, 2)

    def get_ucm(self):
	return morpho.juicy_center(self.ucm, 2)    

    def build_volume(self, nbunch=None):
        """Return the segmentation (numpy.ndarray) induced by the graph."""
        v = zeros_like(self.watershed)
        if nbunch is None:
            nbunch = self.nodes()
        for n in nbunch:
            v.ravel()[list(self.node[n]['extent'])] = n
        return morpho.juicy_center(v,2)

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

    def at_volume_boundary(self, n):
        """Return True if node n touches the volume boundary."""
        return self.has_edge(n, self.boundary_body) or n == self.boundary_body

    def split_voi(self, gt=None):
        if self.gt is None and gt is None:
            return array([0,0])
        elif self.gt is not None:
            return split_voi(None, None, self.rig)
        else:
            return split_voi(self.get_segmentation(), gt, [0], [0])

    def write(self, fout, format='GraphML'):
        pass
        
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
        for u, v in self.edges(nodes):
            try:
                i, j = nodes2ind[u], nodes2ind[v]
            except KeyError:
                continue
            w = merge_priority_function(self,u,v)
            W[i,j] = W[j,i] = exp(-w**2/sigma)
        return W
              
                    

############################
# Merge priority functions #
############################

def boundary_mean(g, n1, n2):
    return mean(g.probabilities.ravel()[list(g[n1][n2]['boundary'])])

def boundary_median(g, n1, n2):
    return median(g.probabilities.ravel()[list(g[n1][n2]['boundary'])])

def approximate_boundary_mean(g, n1, n2):
    return g[n1][n2]['sump'] / g[n1][n2]['n']

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
            return finfo(float).max / size(g.segmentation)
    return ladder_function

def classifier_probability(feature_extractor, classifier):
    def predict(g, n1, n2):
        if n1 == g.boundary_body or n2 == g.boundary_body:
            return g.boundary_probability
        features = feature_extractor(g, n1, n2)
        try:
            prediction = classifier.predict_proba(features)[0,1]
        except AttributeError:
            prediction = classifier.predict(features)[0]
        return prediction
    return predict

def expected_change_voi(feature_extractor, classifier):
    prob_func = classifier_probability(feature_extractor, classifier)
    def predict(g, n1, n2):
        p = float(prob_func(g, n1, n2)) # Prediction from the classifier
        n = g.size
        py1 = len(g.node[n1]['extent'])/float(n)
        py2 = len(g.node[n2]['extent'])/float(n)
        py = py1 + py2
        # Calculate change in VOI
        v = -float(py1*log2(py1) + py2*log2(py2) - py*log2(py))
        # Return expected change
        return  (p*v + (1.0-p)*(-v))
    return predict

def boundary_mean_ladder(g, n1, n2, threshold, strictness=1):
    f = make_ladder(boundary_mean, threshold, strictness)
    return f(g, n1, n2)

def boundary_mean_plus_sem(g, n1, n2, alpha=-6):
    bvals = g.probabilities.ravel()[list(g[n1][n2]['boundary'])]
    return mean(bvals) + alpha*sem(bvals)

def random_priority(g, n1, n2):
    if n1 == g.boundary_body or n2 == g.boundary_body:
        return g.boundary_probability
    return random.random()

def best_possible_segmentation(ws, gt):
    """Build the best possible segmentation given a superpixel map."""
    ws = Rag(ws)
    gt = Rag(gt)
    cnt = contingency_table(ws.get_segmentation(), gt.get_segmentation())
    assignment = cnt == cnt.max(axis=1)[:,newaxis]
    hard_assignment = where(assignment.sum(axis=1) > 1)[0]
    # currently ignoring hard assignment nodes
    assignment[hard_assignment,:] = 0
    for gt_node in range(1,cnt.shape[1]):
        sp_subgraph = ws.subgraph(where(assignment[:,gt_node])[0])
        if len(sp_subgraph) > 0:
            sp_dfs = list(dfs_preorder_nodes(sp_subgraph)) 
                    # dfs_preorder_nodes returns iter, convert to list
            source_sp, other_sps = sp_dfs[0], sp_dfs[1:]
            for current_sp in other_sps:
                ws.merge_nodes(source_sp, current_sp)
    return ws.get_segmentation()
