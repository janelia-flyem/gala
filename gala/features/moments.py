import numpy as np
from scipy.misc import comb as nchoosek
from . import base

class Manager(base.Null):
    def __init__(self, nmoments=4, use_diff_features=True, oriented=False, 
            normalize=False, *args, **kwargs):
        super(Manager, self).__init__()
        self.nmoments = nmoments
        self.use_diff_features = use_diff_features
        self.oriented = oriented
        self.normalize = normalize

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['nmoments'], fm_info['use_diff'],
                    fm_info['oriented'], fm_info['normalize'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('moments')
        json_fm['moments'] = {
            'nmoments' : self.nmoments,
            'use_diff' : self.use_diff_features,
            'oriented' : self.oriented,
            'normalize' : self.normalize
        }
        return json_fm

    def compute_moment_sums(self, ar, idxs):
        values = ar[idxs][...,np.newaxis]
        return (values ** np.arange(self.nmoments+1)).sum(axis=0).T

    def create_node_cache(self, g, n):
        node_idxs = list(g.node[n]['extent'])
        if self.oriented:
            ar = g.max_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r
        return self.compute_moment_sums(ar, node_idxs)

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = list(g[n1][n2]['boundary'])
        if self.oriented:
            ar = g.oriented_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r
        return self.compute_moment_sums(ar, edge_idxs)

    def update_node_cache(self, g, n1, n2, dst, src):
        dst += src

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst += src

    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        if self.oriented:
            ar = g.max_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r
        dst += a * self.compute_moment_sums(ar, idxs)

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        if self.oriented:
            ar = g.max_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r
        dst += a * self.compute_moment_sums(ar, idxs)

    def compute_node_features(self, g, n, cache=None):
        if cache is None: 
            cache = g.node[n][self.default_cache]
        feat = central_moments_from_noncentral_sums(cache)
        if self.normalize:
            feat = ith_root(feat)
        n = feat.ravel()[0]
        return np.concatenate(([n], feat[1:].T.ravel()))

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        feat = central_moments_from_noncentral_sums(cache)
        if self.normalize:
            feat = ith_root(feat)
        n = feat.ravel()[0]
        return np.concatenate(([n], feat[1:].T.ravel()))

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None,
                                                            nthroot=False):
        if not self.use_diff_features:
            return np.array([])
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        m1 = central_moments_from_noncentral_sums(cache1)

        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        m2 = central_moments_from_noncentral_sums(cache2)
       
        if nthroot or self.normalize:
            m1, m2 = map(ith_root, [m1, m2])
        feat = abs(m1-m2)
        n = feat.ravel()[0]
        return np.concatenate(([n], feat[1:].T.ravel()))

def central_moments_from_noncentral_sums(a):
    """Compute moments about the mean from sums of x**i, for i=0, ..., len(a).

    The first two moments about the mean (1 and 0) would always be 
    uninteresting so the function returns n (the sample size) and mu (the 
    sample mean) in their place.
    """
    a = a.astype(np.double)
    if len(a) == 1:
        return a
    N = a.copy()[0]
    a /= N
    mu = a.copy()[1]
    ac = np.zeros_like(a)
    for n in range(2,len(a)):
        js = np.arange(n+1)
        if a.ndim > 1: js = js[:,np.newaxis]
        # Formula found in Wikipedia page for "Central moment", 2011-07-31
        ac[n] = (nchoosek(n,js) * 
                    (-1)**(n-js) * a[js.ravel()] * mu**(n-js)).sum(axis=0)
    ac[0] = N
    ac[1] = mu
    return ac

def ith_root(ar):
    """Get the ith root of the array values at ar[i] for i > 1."""
    if len(ar) < 2:
        return ar
    ar = ar.copy()
    ar[2:] = np.sign(ar[2:]) * \
        (abs(ar[2:]) ** (1.0/np.arange(2, len(ar)))[:, np.newaxis])
    return ar

