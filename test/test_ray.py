
import sys, os
import unittest
import numpy
from scipy.ndimage.measurements import label

rundir = os.path.dirname(os.path.realpath(sys.argv[0]))
sys.path.append(rundir)

import imio, morpho, agglo

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

    def test_watershed(self):
        wss = [morpho.watershed(self.probs[i]) for i in range(3)] + \
            [morpho.watershed(self.probs[3], dams=False)]
        for i in range(self.num_tests):
            self.assertTrue((wss[i]==self.results[i]).all(),
                                    'Watershed test number %i failed.'%(i+1))

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

if __name__ == '__main__':
    unittest.main()
