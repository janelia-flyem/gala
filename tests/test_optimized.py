import numpy as np
from numpy.testing import (assert_allclose, assert_approx_equal,
                           assert_equal)
from gala import optimized as opt

def _flood_fill_example():
    return    np.array([[[0,1,2,5],
                         [0,0,2,4],
                         [1,0,1,2]],
                        [[0,0,5,5],
                         [1,1,1,5],
                         [1,2,1,5]]])

def test_flood_fill_basic():
    fail_message = 'Flood fill failed to find all matching labels.'
    t1 = opt.flood_fill(_flood_fill_example(), (0,0,0), [0], None, True)
    assert_equal(set(t1), set([0,4,5,9,12,13]), fail_message)
    t2 = opt.flood_fill(_flood_fill_example(), (0,0,3), [5], None, True)
    assert_equal(set(t2), set([3,14,15,19,23]), fail_message)


def test_flood_fill_with_no_hits():
    fail_message = 'Flood fill failed with mismatching first label.'
    t = opt.flood_fill(_flood_fill_example(), (0,1,3), [2], None, True)
    assert_equal(set(t), set([]), fail_message)


def test_flood_fill_with_coordinates():
    fail_message = 'Flood fill failed with coordinate return'
    t = opt.flood_fill(_flood_fill_example(), (0,0,0), [0], None, False)
    assert_equal(set(map(tuple, t.tolist())), set([(0,0,0), (0,1,0), (0,1,1), (0,2,1), (1,0,0),
                            (1,0,1)]), fail_message)


def test_flood_fill_multiple_acceptable():
    fail_message = 'Flood fill failed to flood with multiple acceptable labels'
    t1 = opt.flood_fill(_flood_fill_example(), (1,1,1), [1,4], None, True)
    assert_equal(set(t1), set([8,10,16,17,18,20,22]), fail_message)
    t2 = opt.flood_fill(_flood_fill_example(), (0,1,2), [2,5], None, True)
    assert_equal(set(t2), set([2,3,6,11,14,15,19,23]), fail_message)


def test_flood_fill_whole():
    fail_message = 'Flood fill failed to fill whole volume.'
    shape = (10,10,10)
    example2 = np.zeros(shape, dtype=np.int)
    example2[5,5,5] = 0
    t7 = opt.flood_fill(example2, (5,5,5), [0], None, False)
    assert_equal(len(t7), (example2==0).sum())


def test_flood_fill_pipes():
    fail_message = 'Flood fill failed with thin columns in large volume.'
    example3 = np.random.randint(6, size=(200,200,200))
    example3[2,2,:] = 6
    example3[:,2,150] = 6
    example3[45,:,:] = 6
    t8 = opt.flood_fill(example3, (2,2,0), [6], None, True)
    assert_equal(len(t8), (example3==6).sum())

def _despeckle_example():
    example = np.array( [[3,3,3,3,0,0,0,0,0,4,4,4,4,0,0,0,0,0,0,0,0,0,-1,10],
                         [3,2,2,3,0,0,0,0,0,0,4,4,0,0,0,0,0,0,0,0,0,0,-1,10],
                         [3,3,2,3,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,-1,-1],
                         [3,3,2,3,0,0,0,0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [3,3,3,3,0,0,0,0,0,0,4,4,0,0,0,0,0,0,0,0,0,0,0,0],
                         [3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [3,3,3,3,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [3,3,3,3,7,7,0,0,0,0,0,11,11,11, 0,0,0,0,0,0,0,0,0,0],
                         [3,3,3,3,7,7,7,0,0,0,0,11, 6,11,11,0,0,0,0,0,0,0,0,0],
                         [3,3,5,3,5,7,7,0,0,0,0,11, 6, 6,11,0,0,0,0,0,0,0,0,0],
                         [3,3,5,3,5,7,7,0,0,0,0,11,11,11,11,0,0,0,0,0,0,0,0,0],
                         [3,5,5,5,5,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [7,7,5,5,5,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                         [7,7,7,5,5,7,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                         [7,7,7,5,5,7,0,0,0,0,0,0,0,0,0,0,0,0,1,1,9,0,0,0],
                         [7,7,7,5,5,7,0,0,0,0,0,0,0,0,0,0,0,1,1,1,9,9,0,0],
                         [7,7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,1,1,1,9,9,0,0],
                         [7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,1,0,9,9,0,0,0],
                         [7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    expected = np.array([[ 3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1],
                         [ 3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1],
                         [ 3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-1],
                         [ 3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [ 3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [ 3,3,3,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [ 3,3,3,3,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [ 3,3,3,3,7,7,0,0,0,0,0,11,11,11,0,0,0,0,0,0,0,0,0,0],
                         [ 3,3,3,3,7,7,7,0,0,0,0,11,11,11,11,0,0,0,0,0,0,0,0,0],
                         [ 3,3,5,3,5,7,7,0,0,0,0,11,11,11,11,0,0,0,0,0,0,0,0,0],
                         [ 3,3,5,3,5,7,7,0,0,0,0,11,11,11,11,0,0,0,0,0,0,0,0,0],
                         [ 3,5,5,5,5,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [ 7,7,5,5,5,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                         [ 7,7,7,5,5,7,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
                         [ 7,7,7,5,5,7,0,0,0,0,0,0,0,0,0,0,0,0,1,1,9,0,0,0],
                         [ 7,7,7,5,5,7,0,0,0,0,0,0,0,0,0,0,0,1,1,1,9,9,0,0],
                         [ 7,7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,1,1,1,9,9,0,0],
                         [ 7,7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,1,0,9,9,0,0,0],
                         [ 7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                         [ 7,7,7,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])
    return (example, expected)


def test_despeckle_in_place():
    fail_message = "despeckle_watershed failed to corectly despeckle in place"
    example, expected = _despeckle_example()
    opt.despeckle_watershed(example, in_place=True)
    assert_equal(example, expected, fail_message)


def test_despeckle_not_in_place():
    fail_message = "despeckle_watershed failed to corectly despeckle not in place"
    example, expected = _despeckle_example()
    calculated = opt.despeckle_watershed(example, in_place=False)
    assert_equal(calculated, expected, fail_message)
    example_2, e = _despeckle_example()
    assert_equal(calculated, expected, "despeckle watershed modified original when not in place.")

def test_despeckle_stack():
    fail_message = "despeckle_watershed failed to corectly despeckle a stack of arrays"
    example_single, expected_single = _despeckle_example()
    example = np.dstack((np.rot90(example_single, 2), example_single,
        np.rot90(example_single, 2), example_single)).transpose(2,0,1)
    expected = np.dstack((np.rot90(expected_single, 2), expected_single,
        np.rot90(expected_single, 2), expected_single)).transpose(2,0,1)
    calculated = opt.despeckle_watershed(example)
    assert_equal(calculated, expected, fail_message)

if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()
