#!/usr/bin/env python

# system modules
import sys, os, argparse
import cPickle
import logging
from math import sqrt
from abc import ABCMeta, abstractmethod

# libraries
import h5py
import time
import itertools
from numpy import bool, array, double, zeros, mean, random, concatenate, where,\
    uint8, ones, float32, uint32, unique, newaxis, zeros_like, arange, floor, \
    histogram, seterr, __version__ as numpy_version, unravel_index, diff, \
    nonzero, sort, log, inf, argsort, repeat, ones_like, cov, arccos, dot, \
    pi, bincount, isfinite, mean, median
seterr(divide='ignore')
from numpy.linalg import det, eig, norm
from scipy import arange
from scipy.misc.common import factorial
from scipy.ndimage import binary_erosion
try:
    from scipy.spatial import Delaunay
except ImportError:
    logging.warning('Unable to load scipy.spatial.Delaunay. '+
        'Convex hull features not available.')
from scipy.misc import comb as nchoosek
from scipy.stats import sem
try:
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, LinearRegression
except ImportError:
    logging.warning('scikits.learn not found. SVC, Regression not available.')
from evaluate import xlogx
try:
    from vigra.learning import RandomForest as VigraRandomForest
    from vigra.__version__ import version as vigra_version
    vigra_version = tuple(map(int, vigra_version.split('.')))
except ImportError:
    logging.warning(' vigra library is not available. '+
        'Cannot use random forest classifier.')
    pass

# local imports
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack, write_image_stack
from adaboost import AdaBoost

class NullFeatureManager(object):
    def __init__(self, *args, **kwargs):
        self.default_cache = 'feature-cache'
    def __len__(self, *args, **kwargs):
        return 0
    def __call__(self, g, n1, n2=None):
        return self.compute_features(g, n1, n2)

    def compute_features(self, g, n1, n2=None):
        if n2 is None:
            c1 = g.node[n1][self.default_cache]
            return self.compute_node_features(g, n1, c1)
        if len(g.node[n1]['extent']) > len(g.node[n2]['extent']):
            n1, n2 = n2, n1 # smaller node first
        c1, c2, ce = [d[self.default_cache] for d in 
                            [g.node[n1], g.node[n2], g[n1][n2]]]
        return concatenate((
            self.compute_node_features(g, n1, c1),
            self.compute_node_features(g, n2, c2),
            self.compute_edge_features(g, n1, n2, ce),
            self.compute_difference_features(g, n1, n2, c1, c2)
        ))
    def create_node_cache(self, *args, **kwargs):
        return array([])
    def create_edge_cache(self, *args, **kwargs):
        return array([])
    def update_node_cache(self, *args, **kwargs):
        pass
    def update_edge_cache(self, *args, **kwargs):
        pass
    def pixelwise_update_node_cache(self, *args, **kwargs):
        pass
    def pixelwise_update_edge_cache(self, *args, **kwargs):
        pass
    def compute_node_features(self, *args, **kwargs):
        return array([])
    def compute_edge_features(self, *args, **kwargs):
        return array([])
    def compute_difference_features(self, *args, **kwargs):
        return array([])
    

class MomentsFeatureManager(NullFeatureManager):
    def __init__(self, nmoments=4, use_diff_features=True, oriented=False, *args, **kwargs):
        super(MomentsFeatureManager, self).__init__()
        self.nmoments = nmoments
        self.use_diff_features = use_diff_features
        self.oriented = oriented

    def __len__(self):
        return self.nmoments+1

    def compute_moment_sums(self, ar, idxs):
        values = ar[idxs][...,newaxis]
        return (values ** arange(self.nmoments+1)).sum(axis=0).T

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
        return central_moments_from_noncentral_sums(cache).ravel()

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        return central_moments_from_noncentral_sums(cache).ravel()

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None,
                                                            nthroot=False):
        if not self.use_diff_features:
            return array([])
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        m1 = central_moments_from_noncentral_sums(cache1)

        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        m2 = central_moments_from_noncentral_sums(cache2)
       
        if m1.ndim==1:
            m1 = m1[:,newaxis]
            m2 = m2[:,newaxis]
        if nthroot:
            m1[2:] = sgn(m1[2:]) * (abs(m1[2:]) ** (1.0/arange(2, len(m1))))
            m2[2:] = sgn(m2[2:]) * (abs(m2[2:]) ** (1.0/arange(2, len(m2))))
        return abs(m1-m2).ravel()

def central_moments_from_noncentral_sums(a):
    """Compute moments about the mean from sums of x**i, for i=0, ..., len(a).

    The first two moments about the mean (1 and 0) would always be 
    uninteresting so the function returns n (the sample size) and mu (the 
    sample mean) in their place.
    """
    a = a.astype(double)
    if len(a) == 1:
        return a
    N = a.copy()[0]
    a /= N
    mu = a.copy()[1]
    ac = zeros_like(a)
    for n in range(2,len(a)):
        js = arange(n+1)
        if a.ndim > 1: js = js[:,newaxis]
        # Formula found in Wikipedia page for "Central moment", 2011-07-31
        ac[n] = (nchoosek(n,js) * 
                    (-1)**(n-js) * a[js.ravel()] * mu**(n-js)).sum(axis=0)
    ac[0] = N
    ac[1] = mu
    return ac

class OrientationFeatureManager(NullFeatureManager):
    def __init__(self, *args, **kwargs):
        super(OrientationFeatureManager, self).__init__()
 
    def __len__(self):
        return 1

    def create_node_cache(self, g, n):
        # Get subscripts of extent (morpho.unravel_index was slow)
        M = zeros_like(g.watershed); 
        M.ravel()[list(g.node[n]['extent'])] = 1 
        ind = array(nonzero(M)).T
        
        # Get second moment matrix
        smm = cov(ind.T)/float(len(ind))
        try:
            # Get eigenvectors
            val,vec = eig(smm)
            idx = argsort(val)[::-1]
            val = val[idx]
            vec = vec[idx,:]
            return [val,vec,ind]
        except:
            n = g.watershed.ndim
            return [array([0]*n), zeros((n,n)), ind]
        return [ val, vec, ind]
 
    def create_edge_cache(self, g, n1, n2):
        # Get subscripts of extent (morpho.unravel_index was slow)
        M = zeros_like(g.watershed); 
        M.ravel()[list(g[n1][n2]['boundary'])] = 1 
        ind = array(nonzero(M)).T
        
        # Get second moment matrix
        smm = cov(ind.T)/float(len(ind))
        try:
            # Get eigenvectors
            val,vec = eig(smm)
            idx = argsort(val)[::-1]
            val = val[idx]
            vec = vec[idx,:]
            return [val, vec, ind]
        except:
            n = g.watershed.ndim
            return [array([0]*n), zeros((n,n)), ind]

    def update_node_cache(self, g, n1, n2, dst, src):
        dst = self.create_node_cache(g,n1)

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst  = self.create_edge_cache(g,e1[0], e1[1])

    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        pass 
    
    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        pass

    def compute_node_features(self, g, n, cache=None):
        if cache is None: 
            cache = g.node[n][self.default_cache]
        val = cache[0]
        vec = cache[1]
        ind = cache[2]

        features = []
        features.extend(val)
        # coherence measure
        if val[0]==0 and val[1]==0:
            features.append(0)
        else:
            features.append( ((val[0]-val[1])/(val[0]+val[1]))**2)
        return array(features)
    
    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        val = cache[0]
        vec = cache[1]
        ind = cache[2]

        features = []
        features.extend(val)
        # coherence measure
        if val[0]==0 and val[1]==0:
            features.append(0)
        else:
            features.append( ((val[0]-val[1])/(val[0]+val[1]))**2)
        
        return array(features)

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        val1 = cache1[0]
        vec1 = cache1[1]
        ind1 = cache1[2]

        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        val2 = cache2[0]
        vec2 = cache2[1]
        ind2 = cache2[2]

        v1 = vec1[:,0]
        v2 = vec2[:,0]
        # Line connecting centroids of regions
        m1 = ind1.mean(axis=0)
        m2 = ind2.mean(axis=0)
        v3 = m1 - m2 # move to origin

        # Featres are angle differences
        if norm(v1) != 0: v1 /= norm(v1)
        if norm(v2) != 0: v2 /= norm(v2)
        if norm(v3) != 0: v3 /= norm(v3)
        

        features = []
        ang1 = arccos(min(max(dot(v1,v2),-1),1))
        if ang1>pi/2.0: ang1 = pi - ang1
        features.append(ang1)

        ang2 = arccos(min(max(dot(v1,v3),-1),1))
        if ang2>pi/2.0: ang2 = pi - ang2
        ang3 = arccos(min(max(dot(v2,v3),-1),1))
        if ang3>pi/2.0: ang3 = pi - ang3
        features.append(min([ang2,ang3]))
        features.append(max([ang2,ang3]))
        features.append(mean([ang2,ang3]))
        
        return array(features)



class ConvexHullFeatureManager(NullFeatureManager):
    def __init__(self, *args, **kwargs):
        super(ConvexHullFeatureManager, self).__init__()

    def __len__(self):
        return 1 
    
    def convex_hull_ind(self,g,n1,n2=None):
        M = zeros_like(g.watershed); 
        if n2 is not None:
            M.ravel()[list(g[n1][n2]['boundary'])]=1
        else:
            M.ravel()[list(g.node[n1]['extent'])] = 1
        M = M - binary_erosion(M) #Only need border
        ind = array(nonzero(M)).T
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
            ind = array(list(itertools.product(*tuple(array([mins,maxes]).T))))
            tri = Delaunay(ind)
        vol = 0
        for simplex in tri.vertices:
            pts = tri.points[simplex].T
            pts = pts - repeat(pts[:,0][:,newaxis], pts.shape[1],axis=1)
            pts = pts[:,1:]
            vol += abs(1/float(factorial(pts.shape[0])) * det(pts))
            return vol,tri 
    

    def create_node_cache(self, g, n):
        vol, tri = self.convex_hull_vol(self.convex_hull_ind(g,n), g)
        return array([tri,vol])

    def create_edge_cache(self, g, n1, n2):
        vol, tri = self.convex_hull_vol(self.convex_hull_ind(g,n1,n2), g)
        return array([tri,vol])

    def update_node_cache(self, g, n1, n2, dst, src):
        tri1 = src[0]
        tri2 = dst[0]
        ind1 = tri1.points[unique(tri1.convex_hull.ravel())]
        ind2 = tri2.points[unique(tri2.convex_hull.ravel())]
        allind = concatenate((ind1,ind2))
        vol, tri = self.convex_hull_vol(allind, g)
        dst = array([tri,vol])

    def update_edge_cache(self, g, e1, e2, dst, src):
        tri1 = src[0]
        tri2 = dst[0]
        ind1 = tri1.points[unique(tri1.convex_hull.ravel())]
        ind2 = tri2.points[unique(tri2.convex_hull.ravel())]
        allind = concatenate((ind1,ind2))
        vol, tri = self.convex_hull_vol(allind, g)
        dst = array([tri,vol])

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
        features.append(convex_vol/float(len(g.node[n]['extent'])))
    
        return array(features)
    
    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        convex_vol = cache[1]

        features = []
        features.append(convex_vol)
        features.append(convex_vol/float(len(g[n1][n2]['boundary'])))
        return array(features)

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        tri1 = cache1[0]
        convex_vol1 = cache1[1]

        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        tri2 = cache2[0]
        convex_vol2 = cache2[1]
 
        ind1 = tri1.points[unique(tri1.convex_hull.ravel())]
        ind2 = tri2.points[unique(tri2.convex_hull.ravel())]
        allind = concatenate((ind1,ind2))
        convex_vol_both, tri_both = self.convex_hull_vol(allind, g)

        vol1 = float(len(g.node[n1]['extent']))
        vol2 = float(len(g.node[n2]['extent']))
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

        return array(features)


 
class HistogramFeatureManager(NullFeatureManager):
    def __init__(self, nbins=4, minval=0.0, maxval=1.0, 
                    compute_percentiles=[], oriented=False, compute_histogram = True,
                    *args, **kwargs):
        super(HistogramFeatureManager, self).__init__()
        self.minval = minval
        self.maxval = maxval
        self.nbins = nbins
        self.oriented = oriented
        self.compute_histogram = compute_histogram
        try:
            _ = len(compute_percentiles)
        except TypeError: # single percentile value given
            compute_percentiles = [compute_percentiles]
        self.compute_percentiles = compute_percentiles

    def __len__(self):
        return self.nbins

    def histogram(self, vals):
        if vals.ndim == 1:
            return histogram(vals, bins=self.nbins,
                range=(self.minval,self.maxval))[0].astype(double)[newaxis,:]
        elif vals.ndim == 2:
            return concatenate([self.histogram(vals_i) for vals_i in vals.T], 0)
        else:
            raise ValueError('HistogramFeatureManager.histogram expects '+
                'either a 1-d or 2-d array of probabilities. Got %i-d array.'%
                vals.ndim)

    def percentiles(self, h, desired_percentiles):
        if h.ndim == 1 or any([i==1 for i in h.shape]): h = h.reshape((1,-1))
        h = h.T
        nchannels = h.shape[1]
        hcum = concatenate((zeros((1,nchannels)), h.cumsum(axis=0)), axis=0)
        bin_edges = zeros((self.nbins+1, nchannels))
        for i in range(nchannels):
            bin_edges[:,i] = arange(self.minval,self.maxval+1e-10,
                        (self.maxval-self.minval)/float(self.nbins))
        ps = zeros([len(desired_percentiles), h.shape[1]], dtype=double)
        for i, p in enumerate(desired_percentiles):
            b2 = (hcum>=p).argmax(axis=0)
            b1 = (b2-1, arange(nchannels,dtype=int))
            b2 = (b2, arange(nchannels,dtype=int))
            slope = (bin_edges[b2]-bin_edges[b1]) / (hcum[b2]-hcum[b1])
            delta = p - hcum[b1]
            estim = bin_edges[b1] + delta*slope
            error = slope==inf
            estim[error] = (bin_edges[b2]+bin_edges[b1])[error]/2
            ps[i] = estim
        return ps.T

    def normalized_histogram_from_cache(self, cache, desired_percentiles):
        s = cache.sum(axis=1)[:,newaxis]
        s[s==0] = 1
        h = cache/s
        ps = self.percentiles(h, desired_percentiles)
        return h, ps

    def create_node_cache(self, g, n):
        node_idxs = list(g.node[n]['extent'])
        if self.oriented:
            ar = g.max_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r

        return self.histogram(ar[node_idxs,:])

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = list(g[n1][n2]['boundary'])
        if self.oriented:
            ar = g.oriented_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r

        return self.histogram(ar[edge_idxs,:])

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

        dst += a * self.histogram(ar[idxs,:])

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        if self.oriented:
            ar = g.oriented_probabilities_r
        else:
            ar = g.non_oriented_probabilities_r

        dst += a * self.histogram(ar[idxs,:])

    def JS_divergence(self, p, q):
        m = (p+q)/2
        return (self.KL_divergence(p, m) + self.KL_divergence(q, m))/2
    def KL_divergence(self, p, q):
        """Return the Kullback-Leibler Divergence between two histograms."""
        kl = []
        if p.ndim == 1: 
            p = p[newaxis,:]
            q = q[newaxis,:]
        for i in range(len(p)):
            ind = nonzero(p[i]*q[i])
            if len(ind[0]) == 0:
                k = 1.0
            else:
                k = (p[i][ind] * log( p[i][ind]/q[i][ind])).sum() 
            kl.append(k)
        return array(kl)

    def compute_node_features(self, g, n, cache=None):
        if not self.compute_histogram:
            return array([])
        if cache is None: 
            cache = g.node[n1][self.default_cache]
        h, ps = self.normalized_histogram_from_cache(cache, 
                                                     self.compute_percentiles)
        return concatenate((h,ps), axis=1).ravel()

    def compute_edge_features(self, g, n1, n2, cache=None):
        if not self.compute_histogram:
            return array([])
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        h, ps = self.normalized_histogram_from_cache(cache, 
                                                    self.compute_percentiles)
        return concatenate((h,ps), axis=1).ravel()

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        h1, _ = self.normalized_histogram_from_cache(cache1, 
                                                    self.compute_percentiles)
        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        h2, _ = self.normalized_histogram_from_cache(cache2, 
                                                    self.compute_percentiles)
        return self.JS_divergence(h1, h2)

          
class SquigglinessFeatureManager(NullFeatureManager):
    def __init__(self, ndim=3, *args, **kwargs):
        super(SquigglinessFeatureManager, self).__init__()
        self.ndim = ndim
        # cache is min and max coordinates of bounding box
        if numpy_version < '1.6.0':
            self.compute_bounding_box = self.compute_bounding_box_old
            # uses older, slower version of numpy.unravel_index

    def __len__(self):
        return 1

    def compute_bounding_box(self, indices, shape):
        d = self.ndim
        unraveled_indices = concatenate(
            unravel_index(list(indices), shape)).reshape((-1,d), order='F')
        m = unraveled_indices.min(axis=0)
        M = unraveled_indices.max(axis=0)+ones(d)
        return m, M

    def compute_bounding_box_old(self, indices, shape):
        d = self.ndim
        unraveled_indices = concatenate(
            [unravel_index(idx, shape) for idx in indices]).reshape((-1,d))
        m = unraveled_indices.min(axis=0)
        M = unraveled_indices.max(axis=0)+ones(d)
        return m, M

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = g[n1][n2]['boundary']
        return concatenate(
            self.compute_bounding_box(edge_idxs, g.segmentation.shape)
        )

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst[:self.ndim] = \
            concatenate((dst[newaxis,:self.ndim], src[newaxis,:self.ndim]),
            axis=0).min(axis=0)
        dst[self.ndim:] = \
            concatenate((dst[newaxis,self.ndim:], src[newaxis,self.ndim:]),
            axis=0).max(axis=0)

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if remove:
            pass
            # dst = self.create_edge_cache(g, n1, n2)
        if len(idxs) == 0: return
        b = concatenate(self.compute_bounding_box(idxs, g.segmentation.shape))
        self.update_edge_cache(g, (n1,n2), None, dst, b)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        m, M = cache[:self.ndim], cache[self.ndim:]
        plane_surface = sort(M-m)[1:].prod() * (3.0-g.pad_thickness)
        return array([len(g[n1][n2]['boundary']) / plane_surface])

class CompositeFeatureManager(NullFeatureManager):
    def __init__(self, children=[], *args, **kwargs):
        super(CompositeFeatureManager, self).__init__()
        self.children = children
    
    def __len__(self, *args, **kwargs):
        return sum([len(child) for child in self.children])
    
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
        return concatenate(features)

    def compute_edge_features(self, g, n1, n2, cache=None):
        if cache is None: cache = g[n1][n2][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(child.compute_edge_features(g, n1, n2, cache[i]))
        return concatenate(features)
    
    def compute_difference_features(self, g, n1, n2, cache1=None, cache2=None):
        if cache1 is None: cache1 = g.node[n1][self.default_cache]
        if cache2 is None: cahce2 = g.node[n2][self.default_cache]
        features = []
        for i, child in enumerate(self.children):
            features.append(
                child.compute_difference_features(g, n1, n2, cache1[i], cache2[i])
            )
        return concatenate(features)

        
    
def mean_and_sem(g, n1, n2):
    bvals = g.probabilities_r[list(g[n1][n2]['boundary'])]
    return array([mean(bvals), sem(bvals)]).reshape(1,2)

def mean_sem_and_n_from_cache_dict(d):
    n, s1, s2 = d['feature-cache'][:3]
    m = s1/n
    v = 0 if n==1 else max(0, s2/(n-1) - n/(n-1)*m*m)
    s = sqrt(v/n)
    return m, s, n

def skew_from_cache_dict(d):
    n, s1, s2, s3 = d['feature-cache'][:4]
    m1 = s1/n
    k1 = m1
    m2 = s2/n
    k2 = m2 - m1*m1
    m3 = s3/n
    k3 = m3 - 3*m2*m1 + 2*m1*m1*m1
    return k3 * k2**(-1.5)

def feature_set_a(g, n1, n2):
    """Return the mean, SEM, and size of n1, n2, and the n1-n2 boundary in g.
    
    n1 is defined as the smaller of the two nodes, so the labels are swapped
    accordingly if necessary before computing the statistics.
    
    SEM: standard error of the mean, equal to sqrt(var/n)
    """
    if len(g.node[n1]['extent']) > len(g.node[n2]['extent']):
        n1, n2 = n2, n1
    mb, sb, lb = mean_sem_and_n_from_cache_dict(g[n1][n2])
    m1, s1, l1 = mean_sem_and_n_from_cache_dict(g.node[n1])
    m2, s2, l2 = mean_sem_and_n_from_cache_dict(g.node[n2])
    return array([mb, sb, lb, m1, s1, l1, m2, s2, l2]).reshape(1,9)

def node_feature_set_a(g, n):
    """Return the mean, standard deviation, SEM, size, and skewness of n.

    Uses the probability of boundary within n.
    """
    d = g.node[n]
    m, s, l = mean_sem_and_n_from_cache_dict(d)
    stdev = s*sqrt(l)
    skew = skew_from_cache_dict(d)
    return array([m, stdev, s, l, skew])

def h5py_stack(fn):
    try:
        a = array(h5py.File(fn, 'r')['stack'])
    except Exception as except_inst:
        print except_inst
        raise
    return a
    
class RandomForest(object):
    def __init__(self, ntrees=255, use_feature_importance=False, 
            sample_classes_individually=False):
        self.rf = VigraRandomForest(treeCount=ntrees, 
            sample_classes_individually=sample_classes_individually)
        self.use_feature_importance = use_feature_importance
        self.sample_classes_individually=sample_classes_individually

    def fit(self, features, labels, **kwargs):
        features = self.check_features_vector(features)
        labels = self.check_labels_vector(labels)
        if self.use_feature_importance:
            self.oob, self.feature_importance = \
                        self.rf.learnRFWithFeatureSelection(features, labels)
        else:
            self.oob = self.rf.learnRF(features, labels)
        return self

    def predict_proba(self, features):
        features = self.check_features_vector(features)
        return self.rf.predictProbabilities(features)

    def predict(self, features):
        features = self.check_features_vector(features)
        return self.rf.predictLabels(features)

    def check_features_vector(self, features):
        if features.dtype != float32:
            features = features.astype(float32)
        if features.ndim == 1:
            features = features[newaxis,:]
        return features

    def check_labels_vector(self, labels):
        if labels.dtype != uint32:
            if len(unique(labels[labels < 0])) == 1 and not (labels==0).any():
                labels[labels < 0] = 0
            else:
                labels = labels + labels.min()
            labels = labels.astype(uint32)
        labels = labels.reshape((labels.size, 1))
        return labels

    def save_to_disk(self, fn, rfgroupname='rf', overwrite=True):
        self.rf.writeHDF5(fn, rfgroupname, overwrite)
        attr_list = ['oob', 'feature_importance', 'use_feature_importance']
        f = h5py.File(fn)
        for attr in attr_list:
            if hasattr(self, attr):
                f[attr] = getattr(self, attr)

    def load_from_disk(self, fn, rfgroupname='rf'):
        self.rf = VigraRandomForest(fn, rfgroupname)
        f = h5py.File(fn, 'r')
        groups = []
        f.visit(groups.append)
        attrs = [g for g in groups if not g.startswith(rfgroupname)]
        for attr in attrs:
            setattr(self, attr, array(f[attr]))

def read_rf_info(fn):
    f = h5py.File(fn)
    return map(array, [f['oob'], f['feature_importance']])

def concatenate_data_elements(alldata):
    """Return one big learning set from a list of learning sets.
    
    A learning set is a list/tuple of length 4 containing features, labels,
    weights, and node merge history.
    """
    return map(concatenate, zip(*alldata))

def unique_learning_data_elements(alldata):
    if type(alldata[0]) not in (list, tuple): alldata = [alldata]
    f, l, w, h = concatenate_data_elements(alldata)
    af = f.view('|S%d'%(f.itemsize*(len(f[0]))))
    _, uids, iids = unique(af, return_index=True, return_inverse=True)
    bcs = bincount(iids) #DBG
    logging.debug( #DBG
        'repeat feature vec min %d, mean %.2f, median %.2f, max %d.' %
        (bcs.min(), mean(bcs), median(bcs), bcs.max())
    )
    def get_uniques(ar): return ar[uids]
    return map(get_uniques, [f, l, w, h])

def save_training_data_to_disk(data, fn, names=None, info='N/A'):
    if names is None:
        names = ['features', 'labels', 'weights', 'history']
    fout = h5py.File(fn, 'w')
    for data_elem, name in zip(data, names):
        fout[name] = data_elem
    fout.attrs['info'] = info
    fout.close()

def load_training_data_from_disk(fn, names=None, info='N/A'):
    if names is None:
        names = ['features', 'labels', 'weights', 'history']
    fin = h5py.File(fn, 'r')
    data = []
    for name in names:
        data.append(array(fin[name]))
    return data

def boundary_overlap_threshold(boundary_idxs, gt, tol_false, tol_true):
    """Return -1, 0 or 1 by thresholding overlaps between boundaries."""
    n = len(boundary_idxs)
    gt_boundary = 1-gt.ravel()[boundary_idxs].astype(bool)
    fraction_true = gt_boundary.astype(double).sum() / n
    if fraction_true > tol_true:
        return 1
    elif fraction_true > tol_false:
        return 0
    else:
        return -1

def make_thresholded_boundary_overlap_loss(tol_false, tol_true):
    """Return a merge loss function based on boundary overlaps."""
    def loss(g, n1, n2, gt):
        boundary_idxs = list(g[n1][n2]['boundary'])
        return \
            boundary_overlap_threshold(boundary_idxs, gt, tol_false, tol_true)
    return loss

def label_merges(g, merge_history, feature_map_function, gt, loss_function):
    """Replay an agglomeration history and label the loss of each merge."""
    labels = zeros(len(merge_history))
    number_of_features = feature_map_function(g, *g.edges_iter().next()).size
    features = zeros((len(merge_history), number_of_features))
    labeled_image = zeros(gt.shape, double)
    for i, nodes in enumerate(ip.with_progress(
                            merge_history, title='Replaying merge history...', 
                            pbar=ip.StandardProgressBar())):
        n1, n2 = nodes
        features[i,:] = feature_map_function(g, n1, n2)
        labels[i] = loss_function(g, n1, n2, gt)
        labeled_image.ravel()[list(g[n1][n2]['boundary'])] = 2+labels[i]
        g.merge_nodes(n1,n2)
    return features, labels, labeled_image

def select_classifier(cname, features=None, labels=None, **kwargs):
    if 'svm'.startswith(cname):
        del kwargs['class_weight']
        c = SVC(probability=True, **kwargs)
    elif 'logistic-regression'.startswith(cname):
        c = LogisticRegression()
    elif 'linear-regression'.startswith(cname):
        c = LinearRegression()
    elif 'random-forest'.startswith(cname):
        try:
            c = RandomForest()
        except NameError:
            logging.warning(' Tried to use random forest, but not available.'+
                ' Falling back on adaboost.')
            cname = 'ada'
    if 'adaboost'.startswith(cname):
        c = AdaBoost(**kwargs)
    if features is not None and labels is not None:
        c = c.fit(features, labels, **kwargs)
    return c


def pickled(fn):
    try:
        obj = cPickle.load(open(fn, 'r'))
    except cPickle.UnpicklingError:
        obj = RandomForest()
        obj.load_from_disk(fn)
    return obj

arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Classification options')
arggroup.add_argument('-c', '--classifier', default='ada', 
    help='''Choose the classifier to use. Default: adaboost. 
        Options: svm, logistic-regression, linear-regression,
        random-forest, adaboost'''
)
arggroup.add_argument('-k', '--load-classifier', 
    type=pickled, metavar='PCK_FILE',
    help='Load and use a pickled classifier as a merge priority function.'
)
arggroup.add_argument('-f', '--feature-map-function', metavar='FCT_NAME',
    default='feature_set_a',
    help='Use named function as feature map (ignored when -c is not used).'
)
arggroup.add_argument('-T', '--training-data', metavar='HDF5_FN', type=str,
    help='Load training data from file.'
)
arggroup.add_argument('-N', '--node-split-classifier', metavar='HDF5_FN',
    type=str,
    help='Load a node split classifier and split nodes when required.'
)


if __name__ == '__main__':
    from agglo import best_possible_segmentation, Rag, boundary_mean, \
                                classifier_probability, random_priority
    parser = argparse.ArgumentParser(
        parents=[arguments],
        description='Create an agglomeration classifier.'
    )
    parser.add_argument('ws', type=h5py_stack,
        help='Watershed volume, in HDF5 format.'
    )
    parser.add_argument('gt', type=h5py_stack,
        help='Ground truth volume, in HDF5 format also.'
    )
    parser.add_argument('probs', type=h5py_stack,
        help='''Probabilities volume, in HDF ... you get the idea.'''
    )
    parser.add_argument('fout', help='.pck filename to save the classifier.')
    parser.add_argument('-t', '--max-threshold', type=float, default=255,
        help='Agglomerate until this threshold'
    )
    parser.add_argument('-s', '--save-training-data', metavar='FILE',
        help='Save the generated training data to FILE (HDF5 format).'
    )
    parser.add_argument('-b', '--balance-classes', action='store_true',
        default=False, 
        help='Ensure both true edges and false edges are equally represented.'
    )
    parser.add_argument('-K', '--kernel', default='rbf',
        help='The kernel for an SVM classifier.'
    )
    parser.add_argument('-o', '--objective-function', metavar='FCT_NAME', 
        default='random_priority', help='The merge priority function name.'
    )
    parser.add_argument('--save-node-training-data', metavar='FILE',
        help='Save node features and labels to FILE.'
    )
    parser.add_argument('--node-classifier', metavar='FILE',
        help='Train and output a node split classifier.'
    )
    args = parser.parse_args()

    feature_map_function = eval(args.feature_map_function)
    if args.load_classifier is not None:
        mpf = classifier_probability(eval(args.feature_map_function), 
                                                        args.load_classifier)
    else:
        mpf = eval(args.objective_function)

    wsg = Rag(args.ws, args.probs, mpf)
    features, labels, weights, history, ave_sizes = \
                        wsg.learn_agglomerate(args.gt, feature_map_function)

    print 'shapes: ', features.shape, labels.shape

    if args.load_classifier is not None:
        try:
            f = h5py.File(args.save_training_data)
            old_features = array(f['samples'])
            old_labels = array(f['labels'])
            features = concatenate((features, old_features), 0)
            labels = concatenate((labels, old_labels), 0)
        except:
            pass
    print "fitting classifier of size, pos: ", labels.size, (labels==1).sum()
    if args.balance_classes:
        cw = 'auto'
    else:
        cw = {-1:1, 1:1}
    if args.save_training_data is not None:
        try:
            os.remove(args.save_training_data)
        except OSError:
            pass
        f = h5py.File(args.save_training_data)
        f['samples'] = features
        f['labels'] = labels
        f['history'] = history
        f['size'] = ave_sizes
    c = select_classifier(args.classifier, features=features, labels=labels, 
                                        class_weight=cw, kernel=args.kernel)
    print "saving classifier..."
    try:
        cPickle.dump(c, open(os.path.expanduser(args.fout), 'w'), -1)
    except RuntimeError:
        os.remove(os.path.expanduser(args.fout))
        c.save_to_disk(os.path.expanduser(args.fout))
        print 'Warning: unable to pickle classifier to :', args.fout
