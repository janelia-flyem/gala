import numpy as np
from . import base

class Manager(base.Null):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__()

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls()
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('inclusiveness')
        json_fm['inclusiveness'] = {} 
        return json_fm

    def compute_node_features(self, g, n, cache=None):
        bd_lengths = sorted([len(g[n][x]['boundary']) for x in g.neighbors(n)])
        ratio1 = float(bd_lengths[-1])/float(sum(bd_lengths))
        try:
            ratio2 = float(bd_lengths[-2])/float(bd_lengths[-1])
        except IndexError:
            ratio2 = 0.0
        return np.array([ratio1, ratio2])

    def compute_edge_features(self, g, n1, n2, cache=None):
        bd_lengths1 = sorted([len(g[n1][x]['boundary'])
                              for x in g.neighbors(n1)])
        bd_lengths2 = sorted([len(g[n2][x]['boundary'])
                              for x in g.neighbors(n2)])
        ratios1 = [float(len(g[n1][n2]["boundary"]))/float(sum(bd_lengths1)),
                   float(len(g[n1][n2]["boundary"]))/float(sum(bd_lengths2))]
        ratios1.sort()
        ratios2 = [float(len(g[n1][n2]["boundary"]))/float(max(bd_lengths1)),
                   float(len(g[n1][n2]["boundary"]))/float(max(bd_lengths2))]
        ratios2.sort()
        return np.concatenate((ratios1, ratios2))

    def compute_difference_features(self, g, n1, n2, cache1=None, cache2=None):
        return self.compute_node_features(g, n1, cache1) - \
               self.compute_node_features(g, n2, cache2)

