
from itertools import combinations

from heapq import heapify, heappush, heappop
from numpy import array, mean, zeros, zeros_like, uint8, int8, where, unique
from networkx import Graph
import morpho
import iterprogress as ip
from mergequeue import MergeQueue

class Rag(Graph):
    """Region adjacency graph for segmentation of nD volumes."""

    def __init__(self, watershed, probabilities, 
                        merge_priority_function=None, show_progress=False):
        """Create a graph from a watershed volume and image volume.
        
        The watershed is assumed to have dams of label 0 in between basins.
        Then, each basin corresponds to a node in the graph and an edge is
        placed between two nodes if there are one or more watershed pixels
        connected to both corresponding basins.
        """
        super(Rag, self).__init__(weighted=False)
        self.show_progress = show_progress
        self.boundary_body = watershed.max()+1
        self.watershed = morpho.pad(watershed, array([0,self.boundary_body]))
        self.boundary_probability = probabilities.max()+1
        self.probabilities = morpho.pad(probabilities, 
                                        array([self.boundary_probability, 0]))
        self.segmentation = self.watershed.copy()
        if merge_priority_function is None:
            self.merge_priority_function = self.boundary_mean
        else:
            self.merge_priority_function = merge_priority_function
        self.pixel_neighbors = morpho.build_neighbors_array(self.watershed)
        zero_idxs = where(self.watershed.ravel() == 0)[0]
        if self.show_progress:
            def with_progress(seq, length=None, title='Progress: '):
                return ip.with_progress(seq, length, title,
                                                ip.StandardProgressBar())
        else:
            def with_progress(seq, length=None, title='Progress: '):
                return ip.with_progress(seq, length, title, ip.NoProgressBar())
        for idx in with_progress(zero_idxs, title='Building edges... '):
            ns = self.pixel_neighbors[idx]
            adj_labels = self.watershed.ravel()[ns]
            adj_labels = unique(adj_labels[adj_labels != 0])
            for l1,l2 in combinations(adj_labels, 2):
                if self.has_edge(l1, l2): 
                    self[l1][l2]['boundary'].add(idx)
                else: 
                    self.add_edge(l1, l2, boundary=set([idx]))
        nonzero_idxs = where(self.watershed.ravel() != 0)[0]
        for idx in with_progress(nonzero_idxs, title='Building nodes... '):
            try:
                self.node[self.watershed.ravel()[idx]]['extent'].add(idx)
            except KeyError:
                self.node[self.watershed.ravel()[idx]]['extent'] = set([idx])
        self.merge_queue = self.build_merge_queue()

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
            qitem = [self.merge_priority_function(l1,l2), True, l1, l2]
            queue_items.append(qitem)
            self[l1][l2]['qlink'] = qitem
        return MergeQueue(queue_items, with_progress=self.show_progress)

    def rebuild_merge_queue(self):
        """Build a merge queue from scratch and assign to self.merge_queue."""
        self.merge_queue.finish()
        self.merge_queue = self.build_merge_queue()

    def agglomerate(self, threshold=128):
        """Merge nodes sequentially until given edge confidence threshold."""
        while self.merge_queue.peek()[0] < threshold:
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                self.merge_nodes(n1,n2)

    def agglomerate_ladder(self, threshold=1000):
        """Merge sequentially all nodes smaller than threshold.
        
        Note: nodes that are on the volume boundary are not agglomerated.
        """
        while len(self.merge_queue) > 0:
            merge_priority, valid, n1, n2 = self.merge_queue.pop()
            if valid:
                if (len(self.node[n1]['extent']) < threshold and \
                            not self.has_edge(n1, self.boundary_body) or \
                            len(self.node[n2]['extent']) < threshold and \
                            not self.has_edge(n2, self.boundary_body)):
                    self.merge_nodes(n1,n2)
                else:
                    self[n1][n2]['qlink'][1] = False #invalidate but no cnt

    def merge_nodes(self, n1, n2):
        """Merge two nodes, while updating the necessary edges."""
        for n in self.neighbors(n2):
            if n != n1:
                if n in self.neighbors(n1):
                    self.merge_edge_properties((n2,n), (n1,n))
                else:
                    self.move_edge_properties((n2,n), (n1,n))
        self.node[n1]['extent'].update(self.node[n2]['extent'])
        self.segmentation.ravel()[list(self.node[n2]['extent'])] = n1
        boundary = array(list(self[n1][n2]['boundary']))
        boundary_neighbor_pixels = self.segmentation.ravel()[
            self.pixel_neighbors[boundary,:]
        ]
        add = ( (boundary_neighbor_pixels == 0) + 
            (boundary_neighbor_pixels == n1) + 
            (boundary_neighbor_pixels == n2) ).all(axis=1)
        check = True-add
        self.node[n1]['extent'].update(boundary[add])
        self.segmentation.ravel()[boundary[add]] = n1
        self.remove_node(n2)
        boundaries_to_edit = {}
        for px in boundary[check]:
            for lb in unique(
                        self.segmentation.ravel()[self.pixel_neighbors[px]]):
                if lb != n1 and lb != 0:
                    try:
                        boundaries_to_edit[(n1,lb)].append(px)
                    except KeyError:
                        boundaries_to_edit[(n1,lb)] = [px]
        for u, v in boundaries_to_edit.keys():
            if self.has_edge(u, v):
                self[u][v]['boundary'].update(boundaries_to_edit[(u,v)])
            else:
                self.add_edge(u, v, boundary=set(boundaries_to_edit[(u,v)]))
            self.update_merge_queue(u, v)

    def merge_edge_properties(self, src, dst):
        """Merge the properties of edge src into edge dst."""
        u, v = dst
        w, x = src
        self[u][v]['boundary'].update(self[w][x]['boundary'])
        self.merge_queue.invalidate(self[w][x]['qlink'])
        self.update_merge_queue(u, v)

    def move_edge_properties(self, src, dst):
        """Replace edge src with dst in the graph, maintaining properties."""
        u, v = dst
        w, x = src
        # this shouldn't happen in agglomeration, but check just in case:
        if self.has_edge(u,v):
            self.merge_queue.invalidate(self[u][v]['qlink'])
            self.remove_edge(u,v)
        self.add_edge(u, v, attr_dict=self[w][x])
        self.remove_edge(w, x)
        self[u][v]['qlink'][2:] = u, v

    def update_merge_queue(self, u, v):
        """Update the merge queue item for edge (u,v). Add new by default."""
        if self[u][v].has_key('qlink'):
            self.merge_queue.invalidate(self[u][v]['qlink'])
        new_qitem = [self.merge_priority_function(u,v), True, u, v]
        self[u][v]['qlink'] = new_qitem
        self.merge_queue.push(new_qitem)

    def build_volume(self):
        """Return the segmentation (numpy.ndarray) induced by the graph."""
        v = zeros_like(self.watershed)
        for n in self.nodes():
            v.ravel()[list(self.node[n]['extent'])] = n
        return morpho.juicy_center(v,2)

    def boundary_mean(self, n1, n2):
        return mean(self.probabilities.ravel()[list(self[n1][n2]['boundary'])])

    def write(self, fout, format='GraphML'):
        pass
