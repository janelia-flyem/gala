
from itertools import combinations

from heapq import heapify, heappush, heappop
from numpy import array, mean, zeros, zeros_like, uint8, int8
from networkx import Graph
import ws

class Rag(Graph):

    def __init__(self, watershed, probabilities):
        super(Rag, self).__init__(weighted=False)
        self.watershed = watershed
        self.probabilities = probabilities
        self.edge_idx_count = zeros(watershed.shape, int8)
        xmax, ymax, zmax = watershed.shape
        zero_idxs = ((x,y,z) for x in range(xmax) for y in range(ymax)
                    for z in range(zmax) if watershed[x,y,z] == 0)
        nonzero_idxs = ((x,y,z) for x in range(xmax) for y in range(ymax)
                    for z in range(zmax) if watershed[x,y,z] != 0)
        # precompute steps and arrayshape for efficiency inside loop
        steps = map(array, [(0,0,1),(0,1,0),(1,0,0)])
        arrayshape = array(watershed.shape) 
        for idx in zero_idxs:
            ns = ws.neighbor_idxs(idx, steps, arrayshape)
            nlabels = list(set([l for l in [watershed[n] for n in ns] if l>0]))
            self.edge_idx_count[idx] = len(nlabels)-1
            for l1,l2 in combinations(nlabels, 2):
                if self.has_edge(l1, l2):
                    self[l1][l2]['boundary'].append(idx)
                    self[l1][l2]['boundary_probs'].append(probabilities[idx])
                else:
                    self.add_edge(l1, l2, 
                        {'boundary': [idx], 
                        'boundary_probs': [probabilities[idx]]}
                    )
        for idx in nonzero_idxs:
            try:
                self.node[watershed[idx]]['extent'].append(idx)
            except KeyError:
                self.node[watershed[idx]]['extent'] = [idx]
        self.merge_queue = self.build_merge_queue()

    def build_merge_queue(self):
        merge_queue = []
        for l1, l2 in self.edges_iter():
            props = self[l1][l2]
            merge_queue.append([mean(props['boundary_probs']), True, l1, l2])
            props['qlink'] = merge_queue[-1]
        heapify(merge_queue)
        return merge_queue

    def rebuild_merge_queue(self):
        self.merge_queue = self.build_merge_queue()

    def agglomerate(self, threshold=128):
        while self.merge_queue[0][0] < threshold:
            mean_boundary, valid, n1, n2 = heappop(self.merge_queue)
            if valid:
                self.merge_nodes(n1,n2)

    def agglomerate_ladder(self, threshold=1000):
        while len(self.merge_queue) > 0:
            mean_boundary, valid, n1, n2 = heappop(self.merge_queue)
            if valid and \
                        (len(self.node[n1]['extent']) < threshold or \
                        len(self.node[n2]['extent']) < threshold):
                self.merge_nodes(n1,n2)

    def merge_nodes(self, n1, n2):
        for n in self.neighbors(n2):
            if n != n1:
                if n in self.neighbors(n1):
                    self.merge_edge_properties((n1,n), (n2,n))
                else:
                    self.move_edge_properties((n1,n), (n2,n))
        self.node[n1]['extent'].extend(self.node[n2]['extent'])
        self.edge_idx_count[zip(*self[n1][n2]['boundary'])] -= 1
        self.node[n1]['extent'].extend(
            [idx for idx in self[n1][n2]['boundary'] 
            if self.edge_idx_count[idx] == 0]
        )
        self.remove_node(n2)

    def merge_edge_properties(self, e1, e2):
        u, v = e1
        w, x = e2
        self[u][v]['boundary'].extend(self[w][x]['boundary'])
        self[u][v]['boundary_probs'].extend(self[w][x]['boundary_probs'])
        self[u][v]['qlink'][1] = False
        self[w][x]['qlink'][1] = False
        new_props = [mean(self[u][v]['boundary_probs']), True, u, v]
        self[u][v]['qlink'] = new_props
        heappush(self.merge_queue, new_props)

    def move_edge_properties(self, e1, e2):
        u, v = e1
        w, x = e2
        if self.has_edge(u,v):
            self[u][v]['qlink'][1] = False
            self.remove_edge(u,v)
        self.add_edge(u, v, attr_dict = self[w][x])
        self.remove_edge(w, x)
        self[u][v]['qlink'][2:] = u, v
        
    def build_volume(self):
        self.v = zeros_like(self.watershed)
        for n in self.nodes():
            self.v[zip(*self.node[n]['extent'])] = n

    def write(self, fout, format='GraphML'):
        pass
