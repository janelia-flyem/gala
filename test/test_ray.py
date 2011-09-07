
import sys, os
import unittest
import time
import cPickle as pck
from copy import deepcopy as copy

import numpy
from scipy.ndimage.measurements import label

rundir = os.path.dirname(__file__)
sys.path.append(rundir)

import imio, morpho, agglo, classify

def time_me(function):
    def wrapped(*args, **kwargs):
        start = time.time()
        r = function(*args, **kwargs)
        end = time.time()
        return (end-start)*1000
    return wrapped

class TestMorphologicalOperations(unittest.TestCase):
    def setUp(self):
        test_idxs = range(1,5)
        self.num_tests = len(test_idxs)
        fns = [rundir+'/test-%02i-probabilities.h5'%i for i in test_idxs]
        self.probs = [ imio.read_h5_stack(fn) for fn in fns ]
        self.results = [
            imio.read_h5_stack(rundir+'/test-%02i-watershed.h5'%i)
            for i in test_idxs
        ]
        self.landscape = numpy.array([1,0,1,2,1,3,2,0,2,4,1,0])

    def test_watershed_images(self):
        wss = [morpho.watershed(self.probs[i]) for i in range(3)] + \
            [morpho.watershed(self.probs[3], dams=False)]
        for i in range(self.num_tests):
            self.assertTrue((wss[i]==self.results[i]).all(),
                                    'Watershed test number %i failed.'%(i+1))

    def test_watershed(self):
        regular_watershed_result = numpy.array([1,1,1,0,4,0,2,2,2,0,3,3])
        regular_watershed = morpho.watershed(self.landscape)
        self.assertTrue((regular_watershed == regular_watershed_result).all())

    def test_watershed_nodams(self):
        nodam_watershed_result = numpy.array([1,1,1,4,4,4,2,2,2,3,3,3])
        nodam_watershed = morpho.watershed(self.landscape, None, 0.0, False)
        self.assertTrue((nodam_watershed == nodam_watershed_result).all())

    def test_watershed_seeded(self):
        seeds_bool = self.landscape==0
        seeds_unique = label(seeds_bool)[0]
        seeded_watershed_result = numpy.array([1,1,1,1,1,0,2,2,2,0,3,3])
        seeded_watershed1 = morpho.watershed(self.landscape, seeds_bool)
        seeded_watershed2 = morpho.watershed(self.landscape, seeds_unique)
        self.assertTrue((seeded_watershed1 == seeded_watershed_result).all())
        self.assertTrue((seeded_watershed2 == seeded_watershed_result).all())

    def test_watershed_seeded_nodams(self):
        seeds_bool = self.landscape==0
        seeded_nodam_ws_result = numpy.array([1,1,1,1,1,1,2,2,2,3,3,3])
        seeded_nodam_ws = \
                morpho.watershed(self.landscape, seeds_bool, 0.0, False)
        self.assertTrue((seeded_nodam_ws == seeded_nodam_ws_result).all())
        
    def test_watershed_saddle_basin(self):
        saddle_landscape = numpy.array([[0,0,3],[2,1,2],[0,0,3]])
        saddle_result = numpy.array([[1,1,0],[0,0,3],[2,2,0]])
        saddle_ws = morpho.watershed(saddle_landscape)
        self.assertTrue((saddle_ws==saddle_result).all())

    def test_watershed_plateau_performance(self):
        """Test time taken by watershed on plateaus is acceptable.
        
        Versions prior to 2d319e performed redundant computations in the
        idxs_adjacent_to_labels queue which resulted in an explosion in 
        runtime on plateaus. This test checks against that behavior.
        """
        plat = numpy.ones((11,11))
        plat[5,5] = 0
        tws = time_me(morpho.watershed)
        self.assertTrue(tws(plat) < 100)

class TestAgglomeration(unittest.TestCase):
    def setUp(self):
        test_idxs = range(1,5)
        self.num_tests = len(test_idxs)
        pfns = [rundir+'/test-%02i-probabilities.h5'%i for i in test_idxs]
        wsfns = [rundir+'/test-%02i-watershed.h5'%i for i in test_idxs]
        self.probs = [imio.read_h5_stack(fn) for fn in pfns]
        self.wss = [imio.read_h5_stack(fn) for fn in wsfns]
        gtfns = [rundir+'/test-%02i-groundtruth.h5'%i for i in test_idxs]
        self.results = [imio.read_h5_stack(fn) for fn in gtfns]

    def test_8_connectivity(self):
        p = numpy.array([[0,0.5,0],[0.5,1.0,0.5],[0,0.5,0]])
        ws = numpy.array([[1,0,2],[0,0,0],[3,0,4]], numpy.uint32)
        g = agglo.Rag(ws, p, connectivity=2)
        self.assertTrue(agglo.boundary_mean(g, 1, 2) == 0.75)
        self.assertTrue(agglo.boundary_mean(g, 1, 4) == 1.0)

    def test_empty_rag(self):
        g = agglo.Rag()
        self.assertTrue(g.nodes() == [])
        self.assertTrue(g.copy().nodes() == [])

    def test_one_shot(self):
        i = 0
        g = agglo.Rag(self.wss[i], self.probs[i], agglo.boundary_mean)
        v = g.one_shot_agglomeration(0.76)
        self.assertTrue((v==self.results[i]).all(), 
                        'One shot agglomeration failed.')
        v = g.build_boundary_map()
        v = label(g.build_boundary_map()<0.76)[0]
        v[v==2] = 3
        self.assertTrue((v==self.results[i]).all(),
                        'Build boundary map failed.')

    def test_agglomeration(self):
        i = 1
        g = agglo.Rag(self.wss[i], self.probs[i], agglo.boundary_mean)
        g.agglomerate(0.51)
        self.assertTrue((g.get_segmentation()==self.results[i]).all(), 
                        'Mean agglomeration failed.')
                        
    def test_ladder_agglomeration(self):
        i = 2
        g = agglo.Rag(self.wss[i], self.probs[i], agglo.boundary_mean)
        g.agglomerate_ladder(2)
        g.agglomerate(0.5)
        self.assertTrue((g.get_segmentation()==self.results[i]).all(),
                        'Ladder agglomeration failed.')

    def test_no_dam_agglomeration(self):
        i = 3
        g = agglo.Rag(self.wss[i], self.probs[i], agglo.boundary_mean)
        g.agglomerate(0.75)
        self.assertTrue((g.get_segmentation()==self.results[i]).all(),
                        'No dam agglomeration failed.')



def feature_profile(g, f, n1=1, n2=2):
    out = []
    out.append(copy(g[n1][n2]['feature-cache']))
    out.append(copy(g.node[n1]['feature-cache']))
    out.append(copy(g.node[n2]['feature-cache']))
    out.append(f(g,n1,n2))
    out.append(f(g,n1))
    out.append(f(g,n2))
    return out

def list_of_feature_arrays(g, f, edges=[(1,2)], merges=[]):
    e1, edges = edges[0], edges[1:]
    out = feature_profile(g, f, *e1)
    for edge, merge in zip(edges, merges):
        g.merge_nodes(*merge)
        out.extend(feature_profile(g, f, *edge))
    return out

def equal_lists_or_arrays(a1, a2, eps=1e-15):
    """Return True if ls1 and ls2 are arrays equal within eps or equal lists.
    
    The equality relationship can be nested. For example, lists of lists of 
    arrays that have identical structure will match.
    """
    if type(a1) == list and type(a2) == list:
        return all([equal_lists_or_arrays(i1, i2, eps) for i1, i2 in zip(a1,a2)])
    elif type(a1) == numpy.ndarray and type(a2) == numpy.ndarray:
        return len(a1) == len(a2) and (abs((a1-a2)) < eps).all()
    elif type(a1) == float and type(a2) == float:
        return abs((a1-a2)) < eps
    else:
        try:
            return a1 == a2
        except ValueError:
            return False

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.probs2 = imio.read_h5_stack(rundir+'/test-05-probabilities.h5')
        self.probs1 = self.probs2[...,0]
        self.wss1 = imio.read_h5_stack(rundir+'/test-05-watershed.h5')
        self.f1, self.f2, self.f3 = classify.MomentsFeatureManager(2, False), \
            classify.HistogramFeatureManager(3,compute_percentiles=[0.5]),\
            classify.SquigglinessFeatureManager(ndim=2)
        self.f4 = classify.CompositeFeatureManager(
                                            children=[self.f1,self.f2,self.f3])

    def run_matched_test(self, f, fn, c=1,
                            edges=[(1,2),(1,3),(1,4)], merges=[(1,2),(1,3)]):
        if c == 1: p = self.probs1
        else: p = self.probs2
        g = agglo.Rag(self.wss1, p, feature_manager=f)
        o = list_of_feature_arrays(g, f, edges, merges)
        r = pck.load(open(fn, 'r'))
        self.assertTrue(equal_lists_or_arrays(o, r))

    def test_1channel_moment_features(self):
        f = self.f1
        self.run_matched_test(f, 'test/test-05-moments-1channel-12-13.pck')

    def test_2channel_moment_features(self):
        f = self.f1
        self.run_matched_test(f, 'test/test-05-moments-2channel-12-13.pck', 2)

    def test_1channel_histogram_features(self):
        f = self.f2
        self.run_matched_test(f, 'test/test-05-histogram-1channel-12-13.pck')

    def test_2channel_histogram_features(self):
        f = self.f2
        self.run_matched_test(f, 'test/test-05-histogram-2channel-12-13.pck', 2)

    def test_1channel_squiggliness_feature(self):
        f = self.f3
        self.run_matched_test(f, 'test/test-05-squiggle-1channel-12-13.pck')

    def test_1channel_composite_feature(self):
        f = self.f4
        self.run_matched_test(f, 'test/test-05-composite-1channel-12-13.pck')

    def test_2channel_composite_feature(self):
        f = self.f4
        self.run_matched_test(f, 'test/test-05-composite-2channel-12-13.pck', 2)


if __name__ == '__main__':
    unittest.main()
