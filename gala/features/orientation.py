import numpy as np
from numpy.linalg import eig, norm
from . import base

class Manager(base.Null):
    def __init__(self, *args, **kwargs):
        super(Manager, self).__init__()

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('orientation')
        json_fm['orientation'] = {} 
        return json_fm

    def create_node_cache(self, g, n):
        # Get subscripts of extent (morpho.unravel_index was slow)
        M = np.zeros_like(g.watershed); 
        M.ravel()[list(g.extent(n))] = 1 
        ind = np.array(np.nonzero(M)).T
        # Get second moment matrix
        smm = np.cov(ind.T)/float(len(ind))
        try:
            # Get eigenvectors
            val,vec = eig(smm)
            idx = np.argsort(val)[::-1]
            val = val[idx]
            vec = vec[idx,:]
            return [val,vec,ind]
        except:
            n = g.watershed.ndim
            return [np.array([0]*n), np.zeros((n,n)), ind]
        return [val, vec, ind]

    def create_edge_cache(self, g, n1, n2):
        # Get subscripts of extent (morpho.unravel_index was slow)
        M = np.zeros_like(g.watershed); 
        M.ravel()[list(g[n1][n2]['boundary'])] = 1 
        ind = np.array(np.nonzero(M)).T
        # Get second moment matrix
        smm = np.cov(ind.T)/float(len(ind))
        try:
            # Get eigenvectors
            val,vec = eig(smm)
            idx = np.argsort(val)[::-1]
            val = val[idx]
            vec = vec[idx,:]
            return [val, vec, ind]
        except:
            n = g.watershed.ndim
            return [np.array([0]*n), np.zeros((n,n)), ind]

    def update_node_cache(self, g, n1, n2, dst, src):
        c = self.create_node_cache(g, n1)
        for i, e in enumerate(c):
            dst[i] = e

    def update_edge_cache(self, g, e1, e2, dst, src):
        c = self.create_edge_cache(g, *e1)
        for i, e in enumerate(c):
            dst[i] = e

    def compute_node_features(self, g, n, cache=None):
        if cache is None: 
            cache = g.node[n][self.default_cache]
        val = cache[0]
        features = []
        features.extend(val)
        # coherence measure
        if val[0]==0 and val[1]==0:
            features.append(0)
        else:
            features.append( ((val[0]-val[1])/(val[0]+val[1]))**2)
        return np.array(features)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        val = cache[0]
        features = []
        features.extend(val)
        # coherence measure
        if val[0]==0 and val[1]==0:
            features.append(0)
        else:
            features.append( ((val[0]-val[1])/(val[0]+val[1]))**2)
        return np.array(features)

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        vec1 = cache1[1]
        ind1 = cache1[2]

        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        vec2 = cache2[1]
        ind2 = cache2[2]

        v1 = vec1[:,0]
        v2 = vec2[:,0]
        # Line connecting centroids of regions
        m1 = ind1.mean(axis=0)
        m2 = ind2.mean(axis=0)
        v3 = m1 - m2 # move to origin

        # Features are angle differences
        if norm(v1) != 0: v1 /= norm(v1)
        if norm(v2) != 0: v2 /= norm(v2)
        if norm(v3) != 0: v3 /= norm(v3)

        features = []
        ang1 = np.arccos(min(max(np.dot(v1,v2),-1),1))
        if ang1>np.pi/2.0: ang1 = np.pi - ang1
        features.append(ang1)

        ang2 = np.arccos(min(max(np.dot(v1,v3),-1),1))
        if ang2>np.pi/2.0: ang2 = np.pi - ang2
        ang3 = np.arccos(min(max(np.dot(v2,v3),-1),1))
        if ang3>np.pi/2.0: ang3 = np.pi - ang3
        features.append(min([ang2,ang3]))
        features.append(max([ang2,ang3]))
        features.append(np.mean([ang2,ang3]))

        return np.array(features)
