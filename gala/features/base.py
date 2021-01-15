import numpy as np

from .. import evaluate as ev

class Null(object):
    def __init__(self, *args, **kwargs):
        self.default_cache = 'feature-cache'

    def __call__(self, g, n1, n2=None):
        return self.compute_features(g, n1, n2)

    def write_fm(self, json_fm={}):
        return json_fm

    def compute_features(self, g, n1, n2=None):
        if n2 is None:
            c1 = g.nodes[n1][self.default_cache]
            return self.compute_node_features(g, n1, c1)
        if g.nodes[n1]['size'] > g.nodes[n2]['size']:
            n1, n2 = n2, n1 # smaller node first
        c1, c2, ce = [d[self.default_cache] for d in 
                            [g.nodes[n1], g.nodes[n2], g.edges[n1, n2]]]
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
    
    def compute_node_features(self, g, n, cache=None):
        if cache is None: cache = g.nodes[n][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_node_features(g, n, cache[i]))
        return np.concatenate(features)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: cache = g.edges[n1, n2][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_edge_features(g, n1, n2, cache[i]))
        return np.concatenate(features)
    
    def compute_difference_features(self, g, n1, n2, cache1=None, cache2=None):
        if cache1 is None: cache1 = g.nodes[n1][self.default_cache]
        if cache2 is None: cache2 = g.nodes[n2][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_difference_features(
                                            g, n1, n2, cache1[i], cache2[i]))
        return np.concatenate(features)


def _compute_delta_vi(ctable, fragments0, fragments1):
    c0 = np.sum(ctable[list(fragments0)], axis=0)
    c1 = np.sum(ctable[list(fragments1)], axis=0)
    cr = c0 + c1
    p0 = np.sum(c0)
    p1 = np.sum(c1)
    pr = np.sum(cr)
    p0g = np.sum(ev.xlogx(c0))
    p1g = np.sum(ev.xlogx(c1))
    prg = np.sum(ev.xlogx(cr))
    return (pr * np.log2(pr) - p0 * np.log2(p0) - p1 * np.log2(p1) -
            2 * (prg - p0g - p1g))


class Mock(Null):
    '''
    Mock feature manager to verify agglomerative learning works.

    This manager learns a different feature map for fragments vs
    agglomerated segments. It relies on knowing the ground truth for a
    given fragmentation.

    Parameters
    ----------
    frag, gt : array of int, same shape
        The fragmentation and ground truth volumes. Must have same shape.
    '''
    def __init__(self, frag, gt):
        super().__init__()
        self.ctable = ev.contingency_table(frag, gt, ignore_seg=[],
                                           ignore_gt=[]).toarray()
        self._std = 0.1  # standard deviation of feature computations

    def eps(self):
        return np.random.randn(2) * self._std

    def compute_features(self, g, n1, n2=None):
        if n2 is None:
            return np.array([])
        f1, f2 = g.nodes[n1]['fragments'], g.nodes[n2]['fragments']
        f1 -= {g.boundary_body}
        f2 -= {g.boundary_body}
        should_merge = _compute_delta_vi(self.ctable, f1, f2) < 0
        if should_merge:
            return np.array([0., 0.]) + self.eps()
        else:
            if len(f1) + len(f2) == 2:  # single-fragment merge
                return np.array([1., 0.]) + self.eps()
            else:  # multi-fragment merge
                return np.array([0., 1.]) + self.eps()
