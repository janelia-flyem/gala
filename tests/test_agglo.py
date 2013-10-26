import numpy as np
from numpy.testing import assert_equal, assert_array_equal

from gala import agglo


test_idxs = range(4)
num_tests = len(test_idxs)
fns = ['toy-data/test-%02i-probabilities.txt' % i for i in test_idxs]
probs = map(np.loadtxt, fns)
fns = ['toy-data/test-%02i-watershed.txt' % i for i in test_idxs]
wss = map(np.loadtxt, fns)
fns = ['toy-data/test-%02i-groundtruth.txt' % i for i in test_idxs]
results = map(np.loadtxt, fns)

landscape = np.array([1,0,1,2,1,3,2,0,2,4,1,0])

def test_8_connectivity():
    p = np.array([[0,0.5,0],[0.5,1.0,0.5],[0,0.5,0]])
    ws = np.array([[1,0,2],[0,0,0],[3,0,4]], np.uint32)
    g = agglo.Rag(ws, p, connectivity=2)
    assert_equal(agglo.boundary_mean(g, 1, 2), 0.75)
    assert_equal(agglo.boundary_mean(g, 1, 4), 1.0)

def test_empty_rag():
    g = agglo.Rag()
    assert_equal(g.nodes(), [])
    assert_equal(g.copy().nodes(), [])

def test_agglomeration():
    i = 1
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean, 
        normalize_probabilities=True)
    g.agglomerate(0.51)
    assert_array_equal(g.get_segmentation(), results[i],
                       'Mean agglomeration failed.')

def test_ladder_agglomeration():
    i = 2
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean,
        normalize_probabilities=True)
    g.agglomerate_ladder(2)
    g.agglomerate(0.5)
    assert_array_equal(g.get_segmentation(), results[i],
                       'Ladder agglomeration failed.')

def test_no_dam_agglomeration():
    i = 3
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean, 
        normalize_probabilities=True)
    g.agglomerate(0.75)
    assert_array_equal(g.get_segmentation(), results[i],
                       'No dam agglomeration failed.')

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

