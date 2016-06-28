import os
import pytest
from scipy import ndimage as ndi

D = os.path.dirname(os.path.abspath(__file__)) + '/'

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from gala import agglo, agglo2
from gala import evaluate as ev


test_idxs = list(range(6))
num_tests = len(test_idxs)
fns = [D + 'toy-data/test-%02i-probabilities.txt' % i for i in test_idxs]
probs = list(map(np.loadtxt, fns))
fns = [D + 'toy-data/test-%02i-watershed.txt' % i for i in test_idxs]
wss = [np.loadtxt(fn, dtype=np.uint32) for fn in fns]
fns = [D + 'toy-data/test-%02i-groundtruth.txt' % i for i in test_idxs]
results = list(map(np.loadtxt, fns))

landscape = np.array([1,0,1,2,1,3,2,0,2,4,1,0])


def test_2_connectivity():
    p = np.array([[1., 0.], [0., 1.]])
    ws = np.array([[1, 2], [3, 4]], np.uint32)
    g = agglo.Rag(ws, p, connectivity=2, use_slow=True)
    assert_equal(agglo.boundary_mean(g, [[1, 2]]), [0.5])
    assert_equal(agglo.boundary_mean(g, [[1, 4]]), [1.0])
    assert_equal(agglo.boundary_mean(g, [[1, 2], [1, 4]]), [0.5, 1.0])

def test_float_watershed():
    """Ensure float arrays passed as watersheds don't crash everything."""
    p = np.array([[1., 0.], [0., 1.]])
    ws = np.array([[1, 2], [3, 4]], np.float32)
    g = agglo.Rag(ws, p, connectivity=2, use_slow=True)
    assert_equal(agglo.boundary_mean(g, [[1, 2]])[0], 0.5)
    assert_equal(agglo.boundary_mean(g, [[1, 4]])[0], 1.0)
    assert_equal(agglo.boundary_mean(g, [[1, 2], [1, 4]]), [0.5, 1.0])


def test_empty_rag():
    g = agglo.Rag()
    assert_equal(g.nodes(), [])
    assert_equal(g.copy().nodes(), [])


def test_agglomeration():
    i = 1
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean, 
                  normalize_probabilities=True)
    g.agglomerate(0.51)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='Mean agglomeration failed.')


def test_ladder_agglomeration():
    i = 2
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean,
                  normalize_probabilities=True, use_slow=True,
                  update_unchanged_edges=True)
    g.agglomerate_ladder(3)
    g.agglomerate(0.51)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='Ladder agglomeration failed.')

def test_no_dam_agglomeration():
    i = 3
    g = agglo.Rag(wss[i], probs[i], agglo.boundary_mean, 
        normalize_probabilities=True)
    g.agglomerate(0.75)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='No dam agglomeration failed.')


def test_mito():
    i = 5
    def frozen(g, i):
        "hardcoded frozen nodes representing mitochondria"
        return i in [3, 4]
    g = agglo.Rag(wss[i], probs[i], agglo.no_mito_merge(agglo.boundary_mean),
                  normalize_probabilities=True, isfrozennode=frozen,
                  use_slow=True)
    g.agglomerate(0.15)
    g.merge_priority_function = agglo.mito_merge
    g.rebuild_merge_queue()
    g.agglomerate(1.0)
    assert_allclose(ev.vi(g.get_segmentation(), results[i]), 0.0,
                    err_msg='Mito merge failed')


def test_mask():
    i = 1
    mask = np.array([[1, 1, 1, 1, 1],
                     [1, 0, 1, 1, 1],
                     [0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 1]], dtype=bool)
    g = agglo.Rag(wss[i], probs[i], mask=mask)
    assert 3 not in g
    assert (1, 2) in g.edges()
    assert (1, 5) in g.edges()
    assert (2, 4) in g.edges()


def test_traverse():
    labels = [[0, 1, 2],
              [0, 1, 2],
              [0, 1, 2]]
    g = agglo.Rag(np.array(labels))
    assert g.traversing_bodies() == [1]
    assert g.non_traversing_bodies() == [0, 2]


def test_thin_fragment_agglo2():
    labels = np.array([[1, 2, 3]] * 3)
    g = agglo2.Rag(labels)
    assert (1, 3) not in g.graph.edges()


def test_best_possible_segmentation():
    ws = np.array([[2,3],[4,5]], np.int32)
    gt = np.array([[1,2],[1,2]], np.int32)
    best = agglo.best_possible_segmentation(ws, gt)
    assert np.all(best[0,:] == best[1,:])


def test_set_ground_truth():
    labels = [[1, 0, 2],
              [1, 0, 2],
              [1, 0, 2]]
    g = agglo.Rag(np.array(labels))
    g.set_ground_truth(np.array(labels))


def test_split_vi():
    labels = [[1, 0, 2],
              [1, 0, 2],
              [1, 0, 2]]
    g = agglo.Rag(np.array(labels))
    vi0 = g.split_vi(np.array(labels))
    g.set_ground_truth(np.array(labels))
    vi1 = g.split_vi()
    assert np.all(vi0 == vi1)


@pytest.fixture
def dummy_data():
    frag = np.arange(1, 17, dtype=int).reshape((4, 4))
    gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3] * 4, [3] * 4], dtype=int)
    pr = 0.1 * np.array([[0, 1, 0, 9, 7, 0, 2, 0],
                         [0, 1, 0, 9, 7, 0, 2, 0],
                         [0, 1, 0, 9, 7, 0, 2, 0],
                         [9, 8, 7, 9, 7, 8, 9, 9],
                         [9, 8, 7, 9, 7, 8, 9, 9],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]])
    frag = ndi.zoom(frag, 2, order=0)
    gt = ndi.zoom(gt, 2, order=0)
    g = agglo.Rag(frag, pr, merge_priority_function=agglo.boundary_mean)
    return frag, gt, g


def test_manual_agglo_fast_rag(dummy_data):
    frag, gt, g = dummy_data
    assert agglo.boundary_mean(g, [[6, 7]])[0] == 0.8
    assert agglo.boundary_mean(g, [[6, 10]])[0] == 0.8
    original_ids_0 = [g[u][v]['boundary-ids'] for u, v in [(5, 9), (6, 10)]]
    original_ids_1 = [g[u][v]['boundary-ids'] for u, v in [(7, 11), (8, 12)]]
    original_ids_2 = [g[u][v]['boundary-ids'] for u, v in [(2, 3), (6, 7)]]
    g.merge_subgraph([1, 2, 5, 6])  # results in node ID 20
    assert agglo.boundary_mean(g, [[20, 10]])[0] == 0.8
    g.merge_subgraph(range(9, 17))
    assert g[20][27]['boundary-ids'] == set.union(*original_ids_0)
    assert np.allclose(agglo.boundary_mean(g, [[20, 27]])[0], 0.8, atol=0.02)
    g.merge_subgraph([3, 4, 7, 8])
    assert g[27][30]['boundary-ids'] == set.union(*original_ids_1)
    g.merge_nodes(27, 30)
    assert g[20][31]['boundary-ids'] == set.union(*(original_ids_0 +
                                                    original_ids_2))


def test_mean_agglo_fast_rag(dummy_data):
    frag, gt, g = dummy_data
    g.agglomerate(0.5)
    assert ev.vi(g.get_segmentation(), gt) == 0


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

