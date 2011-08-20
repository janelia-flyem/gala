
import sys, os
import unittest
import numpy

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

if __name__ == '__main__':
    unittest.main()
