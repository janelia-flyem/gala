import numpy as np
cimport numpy as np

from . import base
 
class Manager(base.Null):
    def __init__(self, nbins=4, minval=0.0, maxval=1.0, 
            compute_percentiles=[], oriented=False, 
            compute_histogram=True, use_neuroproof=False, *args, **kwargs):
        # Suggested starting parameters: 
        #     nbins: 25 
        #     compute_percentiles: [0.1, 0.5, 0.9]
        super(Manager, self).__init__()
        self.minval = minval
        self.maxval = maxval
        self.nbins = nbins
        self.oriented = oriented
        self.compute_histogram = compute_histogram
        self.use_neuroproof = use_neuroproof

        try:
            _ = len(compute_percentiles)
        except TypeError: # single percentile value given
            compute_percentiles = [compute_percentiles]
        self.compute_percentiles = compute_percentiles
   
    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(
            fm_info['nbins'],
            fm_info['minval'], 
            fm_info['maxval'],
            fm_info['compute_percentiles'],
            fm_info['oriented'],
            fm_info['compute_histogram'],
            fm_info['use_neuroproof'])
        return obj
 
    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('histogram')
        json_fm['histogram'] = {
            'minval' : self.minval, 
            'maxval' : self.maxval, 
            'nbins' : self.nbins, 
            'oriented' : self.oriented, 
            'compute_histogram' : self.compute_histogram, 
            'use_neuroproof' : self.use_neuroproof, 
            'compute_percentiles' : self.compute_percentiles
        } 
        return json_fm

    def histogram(self, vals):
        if vals.ndim == 1:
            h = np.histogram(vals, bins=self.nbins, 
                             range=(self.minval,self.maxval))
            return h[0].astype(np.double)[np.newaxis, :]
        elif vals.ndim == 2:
            return np.concatenate(
                    [self.histogram(vals_i) for vals_i in vals.T], axis=0)
        else:
            raise ValueError('HistogramFeatureManager.histogram expects '+
                'either a 1-d or 2-d np.array of probabilities. '+
                'Got %i-d np.array.'% vals.ndim)

    def percentiles_py(self, h, desired_percentiles):
        if h.ndim == 1 or any([i==1 for i in h.shape]): h = h.reshape((1,-1))
        h = h.T
        nchannels = h.shape[1]
        # reformulate histogram as CDF instead of PDF & prepend 0 to each channel
        hcum = np.concatenate( 
            (np.zeros((1, nchannels)), h.cumsum(axis=0)), axis=0)
        bin_edges = np.zeros((self.nbins+1, nchannels))
        for i in range(nchannels):
            bin_edges[:,i] = np.arange(self.minval,self.maxval+1e-10,
                        (self.maxval-self.minval)/float(self.nbins))
        ps = np.zeros([len(desired_percentiles), h.shape[1]], dtype=np.double)
        for i, p in enumerate(desired_percentiles):
            b2 = (hcum>=p).argmax(axis=0)
            b1 = (b2-1, np.arange(nchannels,dtype=int))
            b2 = (b2, np.arange(nchannels,dtype=int))
            slope = (bin_edges[b2]-bin_edges[b1]) / (hcum[b2]-hcum[b1])
            delta = p - hcum[b1]
            estim = bin_edges[b1] + delta*slope
            error = np.isinf(slope)
            estim[error] = (bin_edges[b2]+bin_edges[b1])[error]/2
            ps[i] = estim
        return ps.T

    def percentiles(self, h, desired_percentiles):
        desired_percentiles = np.array(desired_percentiles); h = np.array(h)
        if h.ndim == 1 or any([i==1 for i in h.shape]): h = h.reshape((1,-1))
        return _percentiles(h.T, desired_percentiles, self.minval, self.maxval, self.nbins)

    def normalized_histogram_from_cache(self, cache, desired_percentiles):
        s = cache.sum(axis=1)[:,np.newaxis]
        s[s==0] = 1
        h = cache/s
        ps = self.percentiles(h, desired_percentiles)
        return h, ps

    def create_node_cache(self, g, n):
        node_idxs = list(g.extent(n))
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
            p = p[np.newaxis,:]
            q = q[np.newaxis,:]
        for i in range(len(p)):
            ind = np.nonzero(p[i]*q[i])
            if len(ind[0]) == 0:
                k = 1.0
            else:
                k = (p[i][ind] * np.log( p[i][ind]/q[i][ind])).sum() 
            kl.append(k)
        return np.array(kl)

    def compute_node_features(self, g, n, cache=None):
        if not self.compute_histogram:
            return np.array([])
        if cache is None: 
            cache = g.node[n][self.default_cache]
        h, ps = self.normalized_histogram_from_cache(cache, 
                                                     self.compute_percentiles)
        if self.use_neuroproof:
            return ps.ravel()
        else:
            return np.concatenate((h,ps), axis=1).ravel()

    def compute_edge_features(self, g, n1, n2, cache=None):
        if not self.compute_histogram:
            return np.array([])
        if cache is None: 
            cache = g[n1][n2][self.default_cache]
        h, ps = self.normalized_histogram_from_cache(cache, 
                                                    self.compute_percentiles)
        if self.use_neuroproof:
            return ps.ravel()
        else:
            return np.concatenate((h,ps), axis=1).ravel()

    def compute_difference_features(self,g, n1, n2, cache1=None, cache2=None):
        if cache1 is None:
            cache1 = g.node[n1][self.default_cache]
        h1, _ = self.normalized_histogram_from_cache(cache1, 
                                                    self.compute_percentiles)
        if cache2 is None:
            cache2 = g.node[n2][self.default_cache]
        h2, _ = self.normalized_histogram_from_cache(cache2, 
                                                    self.compute_percentiles)
        
        if self.use_neuroproof:
            features = []
            return np.array(features)
        else:
            return self.JS_divergence(h1, h2)

cdef _percentiles(double[:,:] h, double[:] desired_percentiles,
                  double minval, double maxval, int nbins):
    cdef double p, slope, prev_h, delta, estim, step_size, prev_b
    cdef int cc,ii,jj,b2
    cdef int nchannels = h.shape[1]
    cdef np.ndarray[np.double_t, ndim=2] ps = np.zeros([nchannels, len(desired_percentiles)], dtype=np.double)
    step_size = (maxval-minval)/nbins
    for cc in range(nchannels):
        for ii in range(1,h.shape[0]):
            h[ii,cc] += h[ii-1,cc]
        for ii in range(desired_percentiles.shape[0]):
            p = desired_percentiles[ii]
            for jj in range(h.shape[0]):
                if h[jj, cc] >= p: 
                    b2 = jj
                    break
            if b2 > 0:
                prev_h = h[b2-1, cc]
                prev_b = b2 * step_size
            else:
                prev_h = 0; prev_b = 0
            if (h[b2, cc]-prev_h) == 0:
                ps[cc,ii] = ((b2+1*step_size)+prev_b)/2
            else:
                slope = step_size / (h[b2, cc]-prev_h)
                delta = p - prev_h
                ps[cc,ii] = prev_b + (slope*delta)
    return ps
