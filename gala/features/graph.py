# external libraries
import numpy as np
import networkx as nx

# local imports
from . import base

class Manager(base.Null):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__()

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('graph')
        json_fm['graph'] = {}
        return json_fm

    def compute_node_features(self, g, n, cache=None):
        deg = g.degree(n)
        ndeg = nx.algorithms.average_neighbor_degree(g, nodes=[n])[n]
        return np.array([deg, ndeg])

    def compute_edge_features(self, g, n1, n2, cache=None):
        nn1, nn2 = g.neighbors(n1), g.neighbors(n2)
        common_neighbors = float(len(np.intersect1d(nn1, nn2)))
        return np.array([common_neighbors])

    def compute_difference_features(self, g, n1, n2, cache1=None, cache2=None):
        return self.compute_node_features(g, n1, cache1) - \
               self.compute_node_features(g, n2, cache2)
