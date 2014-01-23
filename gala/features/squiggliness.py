import numpy as np
from . import base

def compute_bounding_box(indices, shape):
    d = len(shape)
    unraveled_indices = np.concatenate(
        np.unravel_index(list(indices), shape)).reshape((-1,d), order='F')
    m = unraveled_indices.min(axis=0)
    M = unraveled_indices.max(axis=0) + np.ones(d)
    return m, M

class Manager(base.Null):
    def __init__(self, ndim=3, *args, **kwargs):
        super(Manager, self).__init__()
        self.ndim = ndim

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('squiggliness')
        json_fm['squiggliness'] = {'ndim': self.ndim}
        return json_fm

    # cache is min and max coordinates of bounding box
    def create_edge_cache(self, g, n1, n2):
        edge_idxs = g[n1][n2]['boundary']
        return np.concatenate(
            compute_bounding_box(edge_idxs, g.watershed.shape))

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst[:self.ndim] = np.concatenate(
            (dst[np.newaxis,:self.ndim], src[np.newaxis,:self.ndim]),
            axis=0).min(axis=0)
        dst[self.ndim:] = np.concatenate(
            (dst[np.newaxis,self.ndim:], src[np.newaxis,self.ndim:]),
            axis=0).max(axis=0)

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if remove:
            pass
            # dst = self.create_edge_cache(g, n1, n2)
        if len(idxs) == 0: return
        b = np.concatenate(
            self.compute_bounding_box(idxs, g.watershed.shape))
        self.update_edge_cache(g, (n1,n2), None, dst, b)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        m, M = cache[:self.ndim], cache[self.ndim:]
        plane_surface = np.sort(M-m)[1:].prod() * (3.0-g.pad_thickness)
        return np.array([len(g[n1][n2]['boundary']) / plane_surface])

