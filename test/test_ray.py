
import sys, os
import unittest
import time

import numpy
from scipy.ndimage.measurements import label

rundir = os.path.dirname(os.path.realpath(sys.argv[0]))
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
        test_idxs = range(1,6)
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

    def test_1channel_prob_features(self):
        i = 4
        eps = 1e-15
        f1, f2, f3 = classify.MomentsFeatureManager(0,2), \
            classify.HistogramFeatureManager(0,3,compute_percentiles=[0.5]),\
            classify.SquigglinessFeatureManager()
        f4 = classify.CompositeFeatureManager(children=[f1,f2,f3])
        g1 = agglo.Rag(self.wss[i], self.probs[i], feature_manager=f1)
        fc = g1[1][2]['feature-cache']
        self.assertTrue(len(fc)==1, 'MomentsFeatureManager cache len is wrong.')
        self.assertTrue((fc[0]==numpy.array([3,2.7,2.45])).all(), 
                        'Boundary moments cache is wrong: %s'%fc[0])
        fc = g1.node[1]['feature-cache']
        self.assertTrue((fc[0]==numpy.array([5,0.5,0.07])).all(),
                        'Node moments cache is wrong: %s'%fc[0])
        f = f1(g1, 1, 2)
        self.assertTrue(abs((f-numpy.array(
            [4,0.4,0.01,5,0.1,0.004,3,0.9,0.02/3])).sum()) < eps,
            'Moments feature vector is wrong: %s'%f)


if __name__ == '__main__':
    unittest.main()
