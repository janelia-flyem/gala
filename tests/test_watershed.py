import os
import time
import numpy as np
from scipy import ndimage as nd
from numpy.testing import assert_array_equal, assert_array_less

from gala import morpho

rundir = os.path.dirname(__file__)

def time_me(function):
    def wrapped(*args, **kwargs):
        start = time.time()
        r = function(*args, **kwargs)
        end = time.time()
        return r, (end-start)*1000
    return wrapped


test_idxs = list(range(4))
num_tests = len(test_idxs)
fns = [os.path.join(rundir, 'toy-data/test-%02i-probabilities.txt' % i)
       for i in test_idxs]
probs = list(map(np.loadtxt, fns))
fns = [os.path.join(rundir, 'toy-data/test-%02i-watershed.txt' % i)
       for i in test_idxs]
results = [np.loadtxt(fn, dtype=np.int32) for fn in fns]
landscape = np.array([1,0,1,2,1,3,2,0,2,4,1,0])


def test_watershed_images():
    wss = [morpho.watershed(probs[i], dams=(i == 0)) for i in range(2)]
    for i, (ws, res) in enumerate(zip(wss, results)):
        yield (assert_array_equal, ws, res,
               'Image watershed test %i failed.' % i)


def test_watershed():
    regular_watershed_result = np.array([1,1,1,0,2,0,3,3,3,0,4,4])
    regular_watershed = morpho.watershed(landscape, dams=True)
    assert_array_equal(regular_watershed, regular_watershed_result)


def test_watershed_nodams():
    nodam_watershed_result = np.array([1,1,1,2,2,2,3,3,3,4,4,4])
    nodam_watershed = morpho.watershed(landscape, dams=False)
    assert_array_equal(nodam_watershed, nodam_watershed_result)


def test_watershed_seeded():
    seeds_bool = (landscape == 0)
    seeds_unique = nd.label(seeds_bool)[0]
    seeded_watershed_result = np.array([1,1,1,1,1,0,2,2,2,0,3,3])
    seeded_watershed1 = morpho.watershed(landscape, seeds_bool, dams=True)
    seeded_watershed2 = morpho.watershed(landscape, seeds_unique, dams=True)
    assert_array_equal(seeded_watershed1, seeded_watershed_result)
    assert_array_equal(seeded_watershed2, seeded_watershed_result)


def test_watershed_seeded_nodams():
    seeds_bool = landscape==0
    seeded_nodam_ws_result = np.array([1,1,1,1,1,1,2,2,2,3,3,3])
    seeded_nodam_ws = morpho.watershed(landscape,
                seeds=seeds_bool, override_skimage=True, dams=False)
    assert_array_equal(seeded_nodam_ws, seeded_nodam_ws_result)


def test_watershed_saddle_basin():
    saddle_landscape = np.array([[0,0,3],[2,1,2],[0,0,3]])
    saddle_result = np.array([[1,1,1],[0,0,0],[2,2,2]])
    saddle_ws = morpho.watershed(saddle_landscape, dams=True)
    assert_array_equal(saddle_ws, saddle_result)


def test_watershed_plateau_performance():
    """Test time taken by watershed on plateaus is acceptable.
    
    Versions prior to 2d319e performed redundant computations in the
    idxs_adjacent_to_labels queue which resulted in an explosion in 
    runtime on plateaus. This test checks against that behavior.
    """
    plat = np.ones((11,11))
    plat[5,5] = 0
    timed_watershed = time_me(morpho.watershed)
    time_taken = timed_watershed(plat)[1]
    assert_array_less(time_taken, 100, 'watershed plateau too slow')


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
