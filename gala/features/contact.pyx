import numpy as np
cimport numpy as np
from . import base

class Manager(base.Null):
    """ Feature comparing area of contact of two segments to each segment.

    For each segment, it computes both the fraction of the segment's 'dark' 
    pixels (ie less than a specified threshold) that appear in the contact 
    area, and that value normalized by the fraction of the segment's pixels
    the contact area represents. This is motivated by inputs in which 
    pixels represent probability of a cell membrane. For those datasets,
    this feature effectively computes how much of the cell's membrane is
    touching the other cell, and how much of the contact area is membrane.
    """

    def __init__(self, thresholds=[0.1, 0.5, 0.9], oriented=False, 
                 *args, **kwargs):
        """
        Parameters
        ----------
        threshold : array-like, optional
            The 'dark' values at which the contact ratios described above
            will be computed.
        oriented : bool, optional
            Whether to use oriented probabilities.
        """
        super(Manager, self).__init__()
        self.thresholds = np.array(thresholds)
        self.oriented = oriented

    @classmethod
    def load_dict(cls, fm_info):
        obj = cls(fm_info['thresholds'], fm_info['oriented'])
        return obj

    def write_fm(self, json_fm={}):
        if 'feature_list' not in json_fm:
            json_fm['feature_list'] = []
        json_fm['feature_list'].append('contact')
        json_fm['contact'] = {
            'thresholds' : list(self.thresholds),
            'oriented' : self.oriented
        }
        return json_fm

    def compute_edge_features(self, g, n1, n2, cache=None):
        volume_ratio_1 = float(len(g[n1][n2]['boundary'])) / g.node[n1]['size']
        volume_ratio_2 = float(len(g[n1][n2]['boundary'])) / g.node[n2]['size']
        if cache == None: cache = g[n1][n2][self.default_cache]
        contact_matrix = _compute_contact_matrix(cache, volume_ratio_1, 
                                                    volume_ratio_2)
        conlen = contact_matrix.size
        feature_vector = np.zeros(conlen*2 + 4)
        feature_vector[:conlen] = contact_matrix.ravel()
        feature_vector[conlen:2*conlen] = np.log(contact_matrix.ravel())
        feature_vector[2*conlen:2*conlen+2] = np.log(np.array([volume_ratio_1, volume_ratio_2]))
        feature_vector[-1] = volume_ratio_2
        feature_vector[-2] = volume_ratio_1
        return feature_vector

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = np.array(list(g[n1][n2]['boundary']))
        n1_idxs = np.array(list(g.extent(n1)))
        n2_idxs = np.array(list(g.extent(n2)))
        if self.oriented: ar = g.oriented_probabilities_r
        else: ar = g.non_oriented_probabilities_r
        return _compute_edge_cache(edge_idxs, n1_idxs, n2_idxs, ar, self.thresholds)

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst += src

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if len(idxs) == 0: return
        n1_idxs = np.array(list(g.extent(n1)))
        n2_idxs = np.array(list(g.extent(n2)))
        a = -1.0 if remove else 1.0
        if self.oriented: ar = g.oriented_probabilities_r
        else: ar = g.non_oriented_probabilities_r
        dst += a * _compute_edge_cache(idxs, n1_idxs, n2_idxs, ar, self.thresholds)


cdef _compute_contact_matrix(double[:,:,:] totals, double volume_ratio_1,
                             double volume_ratio_2):
    cdef int tt,cc, feature_count, nchannels, nthresholds
    nchannels = totals.shape[0]
    nthresholds = totals.shape[1]
    feature_count = 4
    cdef np.ndarray[np.double_t, ndim=3] scores = np.zeros([nchannels, nthresholds, 
                                                    feature_count], dtype=np.double)
    for cc in range(nchannels): # for each channel
        for tt in range(nthresholds): # for each threshold
            # for n1, ratio of dark pixels in boundary to rest of segment 1
            scores[cc, tt, 0] = totals[cc, tt, 0] / totals[cc, tt, 1]
            # same normalized by ratio of total pixels
            scores[cc, tt, 1] = scores[cc, tt, 0] / volume_ratio_1
            # same for segment 2
            scores[cc, tt, 2] = totals[cc, tt, 0] / totals[cc, tt, 2]
            scores[cc, tt, 3] = scores[cc, tt, 2] / volume_ratio_2
    return scores

cdef _compute_edge_cache(long[:] edge_idxs, long[:] n1_idxs, long[:] n2_idxs,
                    double[:,:] vals, double[:] thresholds):

    cdef int tt,cc, nchannels, nthresholds
    nchannels = vals.shape[1]
    nthresholds = thresholds.shape[0]
    cdef np.ndarray[np.double_t, ndim=3] totals = np.ones([nchannels, nthresholds, 3], 
                                                dtype=np.double)
    for cc in range(nchannels): # for each channel
        for tt in range(nthresholds): # for each threshold
            for nn in range(edge_idxs.shape[0]): # for each voxel
                if vals[edge_idxs[nn],cc] > thresholds[tt]: totals[cc, tt, 0] += 1
            for nn in range(n1_idxs.shape[0]): # for each voxel
                if vals[n1_idxs[nn],cc] > thresholds[tt]: totals[cc, tt, 1] += 1
            for nn in range(n2_idxs.shape[0]): # for each voxel
                if vals[n2_idxs[nn],cc] > thresholds[tt]: totals[cc, tt, 2] += 1
    return totals
