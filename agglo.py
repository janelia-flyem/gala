
import itertools
combinations, izip = itertools.combinations, itertools.izip
from itertools import combinations, izip
import argparse
import random
import matplotlib.pyplot as plt
from heapq import heapify, heappush, heappop
from numpy import array, mean, zeros, zeros_like, uint8, int8, where, unique, \
    finfo, float, size, double, transpose, newaxis, uint32, nonzero, median, exp
from scipy.stats import sem
from scipy.sparse import lil_matrix
from scipy.ndimage.measurements import center_of_mass
from networkx import Graph
from networkx.algorithms.traversal.depth_first_search import dfs_preorder_nodes
import morpho
import iterprogress as ip
from ncut import ncutW
from mergequeue import MergeQueue

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
            merge_priority_function=None, 
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
        if watershed is not None:
            self.set_watershed(watershed, lowmem)
            self.build_graph_from_watershed()
        self.merge_queue = MergeQueue()

    def build_graph_from_watershed(self):
        zero_idxs = where(self.watershed.ravel() == 0)[0]
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
            p = double(self.probabilities.ravel()[idx])
            for l1,l2 in combinations(adj_labels, 2):
                if self.has_edge(l1, l2): 
                    self[l1][l2]['boundary'].add(idx)
                    self[l1][l2]['sump'] += p
                    self[l1][l2]['sump2'] += p*p
                    self[l1][l2]['n'] += 1
                else: 
                    self.add_edge(l1, l2, 
                        boundary=set([idx]), sump=p, sump2=p*p, n=1
                    )
        nonzero_idxs = where(self.watershed.ravel() != 0)[0]
        for idx in with_progress(nonzero_idxs, title='Building nodes... '):
            nodeid = self.watershed.ravel()[idx]
            p = double(self.probabilities.ravel()[idx])
            try:
                self.node[nodeid]['extent'].add(idx)
                self.node[nodeid]['sump'] += p
                self.node[nodeid]['sump2'] += p*p
            except KeyError:
                self.node[nodeid]['extent'] = set([idx])
                self.node[nodeid]['sump'] = p
                self.node[nodeid]['sump2'] = p*p

    def get_neighbor_idxs_fast(self, idxs):
        return self.pixel_neighbors[idxs]

    def get_neighbor_idxs_lean(self, idxs):
        return morpho.get_neighbor_idxs(self.watershed, idxs)

    def set_probabilities(self, probs):
        self.probabilities = morpho.pad(probs, [self.boundary_probability, 0])

    def set_watershed(self, ws, lowmem=False):
        self.boundary_body = ws.max()+1
        self.watershed = morpho.pad(ws, [0, self.boundary_body])
        self.segmentation = self.watershed.copy()
        if lowmem:
            self.neighbor_idxs = self.get_neighbor_idxs_lean
        else:
            self.pixel_neighbors = morpho.build_neighbors_array(self.watershed)
            self.neighbor_idxs = self.get_neighbor_idxs_fast
        self.sum_body_sizes = double(self.get_segmentation().astype(bool).sum())

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
            qitem = [self.merge_priority_function(self,l1,l2), True, l1, l2]
            queue_items.append(qitem)
            self[l1][l2]['qlink'] = qitem
        return MergeQueue(queue_items, with_progress=self.show_progress)

    def rebuild_merge_queue(self):
        """Build a merge queue from scratch and assign to self.merge_queue."""
        self.merge_queue = self.build_merge_queue()

    def agglomerate(self, threshold=128, save_history=False):
        """Merge nodes sequentially until given edge confidence threshold."""
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history = []
        while len(self.merge_queue) > 0 and \
                                        self.merge_queue.peek()[0] < threshold:
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                self.merge_nodes(n1,n2)
                if save_history: history.append((n1,n2))
        if save_history: return history

    def agglomerate_count(self, stepsize=100, save_history=False):
        """Agglomerate until 'stepsize' merges have been made."""
        if self.merge_queue.is_empty():
            self.merge_queue = self.build_merge_queue()
        history = []
        i = 0
        while len(self.merge_queue) > 0 and i < stepsize:
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                i += 1
                self.merge_nodes(n1, n2)
                if save_history: history.append((n1,n2))
        if save_history: return history

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

    def learn_agglomerate(self, gt, feature_map_function):
        """Agglomerate while comparing to ground truth & classifying merges."""
        gtg = Rag(gt)
        rug = Rug(self.get_segmentation(), gtg.get_segmentation(), True, False)
        assignment = rug.overlaps == rug.overlaps.max(axis=1)[:,newaxis]
        hard_assignment = where(assignment.sum(axis=1) > 1)[0]
        # 'hard assignment' nodes are nodes that have most of their overlap
        # with the 0-label in gt, or that have equal amounts of overlap between
        # two other labels
        if self.merge_queue.is_empty(): self.rebuild_merge_queue()
        features, labels = [], []
        history, ave_size = [], []
        while self.number_of_nodes() > gtg.number_of_nodes():
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if merge_priority == self.boundary_probability or self.boundary_body in [n1, n2]:
                print 'Warning: agglomeration done early...'
                break
            if valid:
                features.append(feature_map_function(self, n1, n2).ravel())
                if n2 in hard_assignment:
                    n1, n2 = n2, n1
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
                    else:
                        _ = features.pop() # remove last item
                else:
                    history.append([n1, n2])
                    ave_size.append(self.sum_body_sizes / 
                                                (self.number_of_nodes()-1))
                    if (assignment[n1,:] == assignment[n2,:]).all():
                        self.merge_nodes(n1, n2)
                        labels.append(-1)
                    else:
                        labels.append(1)
        return array(features).astype(double), array(labels), \
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

    def merge_nodes(self, n1, n2):
        """Merge two nodes, while updating the necessary edges."""
        self.sum_body_sizes -= len(self.node[n1]['extent']) + \
                                len(self.node[n2]['extent'])
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
        self.remove_node(n2)
        self.sum_body_sizes += len(self.node[n1]['extent'])

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
            new_qitem = [self.merge_priority_function(self,u,v), True, u, v]
            self[u][v]['qlink'] = new_qitem
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

    def build_volume(self):
        """Return the segmentation (numpy.ndarray) induced by the graph."""
        v = zeros_like(self.watershed)
        for n in self.nodes():
            v.ravel()[list(self.node[n]['extent'])] = n
        return morpho.juicy_center(v,2)

    def at_volume_boundary(self, n):
        """Return True if node n touches the volume boundary."""
        return self.has_edge(n, self.boundary_body)

    def write(self, fout, format='GraphML'):
        pass
        
    def ncut(self, num_clusters=10, kmeans_iters=5, sigma=255.0*20):
        """Run normalized cuts on the current set of superpixels.
           Keyword arguments:
               num_clusters -- number of clusters to compute
               kmeans_iters -- # iterations to run kmeans when clustering
               sigma -- sigma value when setting up weight matrix
           Return value: None
        """
        W = self.compute_W(self.merge_priority_function) # Compute weight matrix
        labels, eigvec, eigval = ncutW(W, num_clusters, kmeans_iters) # Run ncut
        self.cluster_by_labels(labels) # Merge nodes that are in same cluster
    
    def cluster_by_labels(self, labels):
        """Merge all superpixels with the same label (1 label per 1 sp)"""
        if not (len(labels) == self.number_of_nodes()):
            raise ValueError('Number of labels should be %d but is %d.', 
                self.number_of_nodes(), len(labels))
        nodes = array(self.nodes())
        for label in unique(labels):
            ind = nonzero(labels==label)[0]
            nodes_to_merge = nodes[ind]
            node1 = nodes_to_merge[0]
            for node in nodes_to_merge[1:]:
                self.merge_nodes(node1,node)
                
    def compute_W(self, merge_priority_function, sigma=255.0*20):
        """ Computes the weight matrix for clustering"""
        nodes_list = self.nodes()
        n = len(nodes_list)
        W = lil_matrix((n,n))
        for e1, e2 in self.edges_iter():
            val = merge_priority_function(self,e1,e2)
            ind1 = nonzero(nodes_list==e1)[0][0]
            ind2 = nonzero(nodes_list==e2)[0][0]
            W[ind1, ind2] = exp(-val**2/sigma)
            W[ind2, ind1] = W[ind1, ind2]
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

# RUG #

class Rug(object):
    """Region union graph, used to compare two segmentations."""
    def __init__(self, s1=None, s2=None, progress=False, rem_zero_ovr=False):
        self.s1 = s1
        self.s2 = s2
        self.progress = progress
        if s1 is not None and s2 is not None:
            self.build_graph(s1, s2, rem_zero_ovr)

    def build_graph(self, s1, s2, remove_zero_overlaps=False):
        if s1.shape != s2.shape:
            raise RuntimeError('Error building region union graph: '+
                'volume shapes don\'t match. '+str(s1.shape)+' '+str(s2.shape))
        n1 = len(unique(s1))
        n2 = len(unique(s2))
        self.overlaps = zeros((n1,n2), double)
        self.sizes1 = zeros(n1, double)
        self.sizes2 = zeros(n2, double)
        if self.progress:
            def with_progress(seq):
                return ip.with_progress(seq, length=s1.size,
                            title='RUG...', pbar=ip.StandardProgressBar())
        else:
            def with_progress(seq): return seq
        for v1, v2 in with_progress(izip(s1.ravel(), s2.ravel())):
            self.overlaps[v1,v2] += 1
            self.sizes1[v1] += 1
            self.sizes2[v2] += 1
        if remove_zero_overlaps:
            self.overlaps[:,0] = 0
            self.overlaps[0,:] = 0
            self.overlaps[0,0] = 1

    def __getitem__(self, v):
        try:
            l = len(v)
        except TypeError:
            v = [v]
            l = 1
        v1 = v[0]
        v2 = Ellipsis
        do_transpose = False
        if l >= 2:
            v2 = v[1]
        if l >= 3:
            do_transpose = bool(v[2])
        if do_transpose:
            return transpose(self.overlaps)[v1,v2]/self.sizes2[v1,newaxis]
        else:
            return self.overlaps[v1,v2]/self.sizes1[v1,newaxis]

def best_possible_segmentation(ws, gt):
    """Build the best possible segmentation given a superpixel map."""
    ws = Rag(ws)
    gt = Rag(gt)
    rug = Rug(ws.get_segmentation(), gt.get_segmentation())
    assignment = rug.overlaps == rug.overlaps.max(axis=1)[:,newaxis]
    hard_assignment = where(assignment.sum(axis=1) > 1)[0]
    # currently ignoring hard assignment nodes
    assignment[hard_assignment,:] = 0
    for gt_node in range(1,len(rug.sizes2)):
        sp_subgraph = ws.subgraph(where(assignment[:,gt_node])[0])
        if len(sp_subgraph) > 0:
            sp_dfs = list(dfs_preorder_nodes(sp_subgraph)) 
                    # dfs_preorder_nodes returns iter, convert to list
            source_sp, other_sps = sp_dfs[0], sp_dfs[1:]
            for current_sp in other_sps:
                ws.merge_nodes(source_sp, current_sp)
    return ws.get_segmentation()
