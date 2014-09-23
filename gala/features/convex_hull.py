# python standard library
import logging
import itertools as it

# numpy/scipy
import numpy as np
from scipy import ndimage as nd
from scipy.misc import factorial
from numpy.linalg import det
try:
    from scipy.spatial import Delaunay
except ImportError:
    logging.warning('Unable to load scipy.spatial.Delaunay. '+
        'Convex hull features not available.')

# local imports
from . import base

class Manager(base.Null):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__()

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('convex-hull')
        json_fm['convex-hull'] = {}
        return json_fm

    def convex_hull_ind(self, g, n1, n2=None):
        m = np.zeros_like(g.watershed); 
        if n2 is not None:
            m.ravel()[list(g[n1][n2]['boundary'])]=1
        else:
            m.ravel()[list(g.extent(n1))] = 1
        m = m - nd.binary_erosion(m) #Only need border
        ind = np.np.array(np.nonzero(m)).T
        return ind


    def convex_hull_vol(self, ind, g):
        # Compute the convex hull of the region
        try:
            tri = Delaunay(ind)
        except:
            # Just triangulate bounding box
            mins = ind.min(axis=0)
            maxes = ind.max(axis=0)
            maxes[maxes==mins] += 1
            ind = np.array(list(it.product(*tuple(np.array([mins,maxes]).T))))
            tri = Delaunay(ind)
        vol = 0
        for simplex in tri.vertices:
            pts = tri.points[simplex].T
            pts = pts - np.repeat(pts[:,0][:, np.newaxis], pts.shape[1], axis=1)
            pts = pts[:,1:]
            vol += abs(1/float(factorial(pts.shape[0])) * det(pts))
            return vol,tri 

    def create_node_cache(self, g, n):
        vol, tri = self.convex_hull_vol(self.convex_hull_ind(g,n), g)
        return np.array([tri,vol])

    def create_edge_cache(self, g, n1, n2):
        vol, tri = self.convex_hull_vol(self.convex_hull_ind(g,n1,n2), g)
        return np.array([tri,vol])

    def update_node_cache(self, g, n1, n2, dst, src):
        tri1 = src[0]
        tri2 = dst[0]
        ind1 = tri1.points[np.unique(tri1.convex_hull.ravel())]
        ind2 = tri2.points[np.unique(tri2.convex_hull.ravel())]
        allind = np.concatenate((ind1,ind2))
        vol, tri = self.convex_hull_vol(allind, g)
        dst = np.array([tri,vol])

    def update_edge_cache(self, g, e1, e2, dst, src):
        tri1 = src[0]
        tri2 = dst[0]
        ind1 = tri1.points[np.unique(tri1.convex_hull.ravel())]
        ind2 = tri2.points[np.unique(tri2.convex_hull.ravel())]
        allind = np.concatenate((ind1,ind2))
        vol, tri = self.convex_hull_vol(allind, g)
        dst = np.array([tri,vol])

    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        pass

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        pass

    def compute_node_features(self, g, n, cache=None):
        if cache is None: 
            cache = g.node[n][self.default_cache]
        convex_vol = cache[1]
        features = []
        features.append(convex_vol)
        features.append(convex_vol/float(g.node[n]['size']))
        return np.array(features)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        convex_vol = cache[1]

        features = []
        features.append(convex_vol)
        features.append(convex_vol/float(len(g[n1][n2]['boundary'])))
        return np.array(features)

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        tri1 = cache1[0]
        convex_vol1 = cache1[1]

        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        tri2 = cache2[0]
        convex_vol2 = cache2[1]

        ind1 = tri1.points[np.unique(tri1.convex_hull.ravel())]
        ind2 = tri2.points[np.unique(tri2.convex_hull.ravel())]
        allind = np.concatenate((ind1,ind2))
        convex_vol_both, tri_both = self.convex_hull_vol(allind, g)

        vol1 = float(g.node[n1]['size'])
        vol2 = float(g.node[n2]['size'])
        volborder = float(len(g[n1][n2]['boundary']))
        volboth = vol1+vol2

        features = []
        features.append(abs(convex_vol1/vol1 - convex_vol2/vol2))
        features.append(abs(convex_vol1/vol1 - convex_vol_both/volboth))
        features.append(abs(convex_vol2/vol2 - convex_vol_both/volboth))
        features.append(abs(convex_vol_both/volboth))
        features.append((convex_vol1*vol2)/(convex_vol2*vol1))
        features.append(volborder/vol1)
        features.append(volborder/vol2)
        features.append(volborder/volboth)

        return np.array(features)
