
from itertools import combinations

from heapq import heapify, heappush, heappop
from numpy import array, mean, zeros, zeros_like, uint8, int8, where, unique
from networkx import Graph
import morpho

class Rag(Graph):
    """Region adjacency graph for segmentation of nD volumes."""

    def __init__(self, watershed, probabilities, merge_priority_function=None):
        """Create a graph from a watershed volume and image volume.
        
        The watershed is assumed to have dams of label 0 in between basins.
        Then, each basin corresponds to a node in the graph and an edge is
        placed between two nodes if there are one or more watershed pixels
        connected to both corresponding basins.
        """
        super(Rag, self).__init__(weighted=False)
        self.boundary_body = watershed.max()+1
        self.boundary_probability = probabilities.max()+1
        self.watershed = morpho.pad(watershed, array([0,self.boundary_body]))
        self.probabilities = morpho.pad(probabilities, 
                                        array([self.boundary_probability, 0]))
        if merge_priority_function is None:
            self.merge_priority_function = self.boundary_mean
        else:
            self.merge_priority_function = merge_priority_function
        self.edge_idx_count = zeros(self.watershed.shape, int8)
        neighbors = morpho.build_neighbors_array(self.watershed)
        zero_idxs = where(self.watershed.ravel() == 0)[0]
        for idx in zero_idxs:
            ns = neighbors[idx]
            adj_labels = self.watershed.ravel()[ns]
            adj_labels = unique(adj_labels[adj_labels != 0])
            self.edge_idx_count.ravel()[idx] = len(adj_labels)-1
            for l1,l2 in combinations(adj_labels, 2):
                if self.has_edge(l1, l2):
                    self[l1][l2]['boundary'].add(idx)
                else:
                    self.add_edge(l1, l2, boundary=set([idx]))
        nonzero_idxs = where(self.watershed.ravel() != 0)[0]
        for idx in nonzero_idxs:
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
        merge_queue = []
        for l1, l2 in self.edges_iter():
            qitem = [self.merge_priority_function(l1,l2), True, l1, l2]
            merge_queue.append(qitem)
            self[l1][l2]['qlink'] = qitem
        heapify(merge_queue)
        return merge_queue

    def rebuild_merge_queue(self):
        """Build a merge queue from scratch and assign to self.merge_queue."""
        self.merge_queue = self.build_merge_queue()

    def agglomerate(self, threshold=128):
        """Merge nodes sequentially until given edge confidence threshold."""
        while self.merge_queue[0][0] < threshold:
            merge_priority, valid, n1, n2 = heappop(self.merge_queue)
            if valid:
                self.merge_nodes(n1,n2)

    def agglomerate_ladder(self, threshold=1000):
        """Merge sequentially all nodes smaller than threshold.
        
        Note: nodes that are on the volume boundary are not agglomerated.
        """
        while len(self.merge_queue) > 0:
            merge_priority, valid, n1, n2 = heappop(self.merge_queue)
            if valid and \
                        (len(self.node[n1]['extent']) < threshold and \
                        not self.has_edge(n1, self.boundary_body) or \
                        len(self.node[n2]['extent']) < threshold and \
                        not self.has_edge(n2, self.boundary_body)):
                self.merge_nodes(n1,n2)

    def merge_nodes(self, n1, n2):
        """Merge two nodes, while updating the necessary edges."""
        for n in self.neighbors(n2):
            if n != n1:
                if n in self.neighbors(n1):
                    self.merge_edge_properties((n2,n), (n1,n))
                else:
                    self.move_edge_properties((n2,n), (n1,n))
        self.node[n1]['extent'].update(self.node[n2]['extent'])
        self.edge_idx_count.ravel()[list(self[n1][n2]['boundary'])] -= 1
        self.node[n1]['extent'].update(
            [idx for idx in self[n1][n2]['boundary'] 
            if self.edge_idx_count.ravel()[idx] == 0]
        )
        self.remove_node(n2)

    def merge_edge_properties(self, src, dst):
        """Merge the properties of edge src into edge dst."""
        u, v = dst
        w, x = src
        self[u][v]['boundary'].update(self[w][x]['boundary'])
        self[u][v]['qlink'][1] = False
        self[w][x]['qlink'][1] = False
        new_qitem = [self.merge_priority_function(u,v), True, u, v]
        self[u][v]['qlink'] = new_qitem
        heappush(self.merge_queue, new_qitem)

    def move_edge_properties(self, src, dst):
        """Replace edge src with dst in the graph, maintaining properties."""
        u, v = dst
        w, x = src
        # this shouldn't happen in agglomeration, but check just in case:
        if self.has_edge(u,v):
            self[u][v]['qlink'][1] = False
            self.remove_edge(u,v)
        self.add_edge(u, v, attr_dict=self[w][x])
        self.remove_edge(w, x)
        self[u][v]['qlink'][2:] = u, v
        
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
