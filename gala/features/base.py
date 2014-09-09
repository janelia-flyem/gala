import numpy as np

class Null(object):
    def __init__(self, *args, **kwargs):
        self.default_cache = 'feature-cache'

    def __call__(self, g, n1, n2=None):
        return self.compute_features(g, n1, n2)

    def write_fm(self, json_fm={}):
        return json_fm

    def compute_features(self, g, n1, n2=None):
        if n2 is None:
            c1 = g.node[n1][self.default_cache]
            return self.compute_node_features(g, n1, c1)
        if g.node[n1]['size'] > g.node[n2]['size']:
            n1, n2 = n2, n1 # smaller node first
        c1, c2, ce = [d[self.default_cache] for d in 
                            [g.node[n1], g.node[n2], g[n1][n2]]]
        return np.concatenate((
            self.compute_node_features(g, n1, c1),
            self.compute_node_features(g, n2, c2),
            self.compute_edge_features(g, n1, n2, ce),
            self.compute_difference_features(g, n1, n2, c1, c2)
        ))
    def create_node_cache(self, *args, **kwargs):
        return np.array([])
    def create_edge_cache(self, *args, **kwargs):
        return np.array([])
    def update_node_cache(self, *args, **kwargs):
        pass
    def update_edge_cache(self, *args, **kwargs):
        pass
    def pixelwise_update_node_cache(self, *args, **kwargs):
        pass
    def pixelwise_update_edge_cache(self, *args, **kwargs):
        pass
    def compute_node_features(self, *args, **kwargs):
        return np.array([])
    def compute_edge_features(self, *args, **kwargs):
        return np.array([])
    def compute_difference_features(self, *args, **kwargs):
        return np.array([])


class Composite(Null):
    def __init__(self, children=[], *args, **kwargs):
        super(Composite, self).__init__()
        self.children = children
 
    def write_fm(self, json_fm={}):
        for child in self.children:
            json_fm.update(child.write_fm(json_fm))
        return json_fm
   
    def create_node_cache(self, *args, **kwargs):
        return [c.create_node_cache(*args, **kwargs) for c in self.children]

    def create_edge_cache(self, *args, **kwargs):
        return [c.create_edge_cache(*args, **kwargs) for c in self.children]
    
    def update_node_cache(self, g, n1, n2, dst, src):
        for i, child in enumerate(self.children):
            child.update_node_cache(g, n1, n2, dst[i], src[i])
    
    def update_edge_cache(self, g, e1, e2, dst, src):
        for i, child in enumerate(self.children):
            child.update_edge_cache(g, e1, e2, dst[i], src[i])
    
    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        for i, child in enumerate(self.children):
            child.pixelwise_update_node_cache(g, n, dst[i], idxs, remove)

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        for i, child in enumerate(self.children):
            child.pixelwise_update_edge_cache(g, n1, n2, dst[i], idxs, remove)

    def compute_node_features(self, g, n, cache=None):
        if cache is None: cache = g.node[n][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_node_features(g, n, cache[i]))
        return np.concatenate(features)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: cache = g[n1][n2][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_edge_features(g, n1, n2, cache[i]))
        return np.concatenate(features)
    
    def compute_difference_features(self, g, n1, n2, cache1=None, cache2=None):
        if cache1 is None: cache1 = g.node[n1][self.default_cache]
        if cache2 is None: cache2 = g.node[n2][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_difference_features(
                                            g, n1, n2, cache1[i], cache2[i]))
        return np.concatenate(features)
    
