import sys, os
import pickle as pck
from copy import deepcopy as copy

import numpy as np
from numpy.testing import (assert_allclose, assert_approx_equal,
                           assert_equal)

rundir = os.path.dirname(__file__)
sys.path.append(rundir)


from gala import agglo, features


def feature_profile(g, f, n1=1, n2=2):
    out = []
    out.append(copy(g.edges[n1, n2]['feature-cache']))
    out.append(copy(g.nodes[n1]['feature-cache']))
    out.append(copy(g.nodes[n2]['feature-cache']))
    out.append(f(g, n1, n2))
    out.append(f(g, n1))
    out.append(f(g, n2))
    return out


def list_of_feature_arrays(g, f, edges=[(1,2)], merges=[]):
    e1, edges = edges[0], edges[1:]
    out = feature_profile(g, f, *e1)
    for edge, merge in zip(edges, merges):
        g.merge_nodes(*merge)
        out.extend(feature_profile(g, f, *edge))
    return out


def assert_equal_lists_or_arrays(a1, a2, eps=1e-3):
    """Return True if ls1 and ls2 are arrays equal within eps or equal lists.
    
    The equality relationship can be nested. For example, lists of lists of 
    arrays that have identical structure will match.
    """
    if type(a1) == list and type(a2) == list:
        [assert_equal_lists_or_arrays(i1, i2, eps) for i1, i2 in zip(a1,a2)]
    elif type(a1) == np.ndarray and type(a2) == np.ndarray:
        assert_allclose(a1, a2, atol=eps)
    elif type(a1) == float and type(a2) == float:
        assert_approx_equal(a1, a2, int(-np.log10(eps)))
    else:
        assert_equal(a1, a2)


probs2 = np.load(os.path.join(rundir, 'toy-data/test-04-probabilities.npy'))
probs1 = probs2[..., 0]
wss1 = np.loadtxt(os.path.join(rundir, 'toy-data/test-04-watershed.txt'),
                  np.uint32)
f1, f2, f3 = (features.moments.Manager(2, False),
              features.histogram.Manager(3, compute_percentiles=[0.5]),
              features.squiggliness.Manager(ndim=2))
f4 = features.base.Composite(children=[f1, f2, f3])


def run_matched(f, fn, c=1,
                edges=[(1, 2), (6, 3), (7, 4)],
                merges=[(1, 2), (6, 3)]):
    p = probs1 if c == 1 else probs2
    g = agglo.Rag(wss1, p, feature_manager=f, use_slow=True)
    o = list_of_feature_arrays(g, f, edges, merges)
    with open(fn, 'rb') as fin:
        r = pck.load(fin, encoding='bytes')
    assert_equal_lists_or_arrays(o, r)


def test_1channel_moment_features():
    f = f1
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-moments-1channel-12-13.pck'))

def test_2channel_moment_features():
    f = f1
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-moments-2channel-12-13.pck'), 2)

def test_1channel_histogram_features():
    f = f2
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-histogram-1channel-12-13.pck'))

def test_2channel_histogram_features():
    f = f2
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-histogram-2channel-12-13.pck'), 2)

def test_1channel_squiggliness_feature():
    f = f3
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-squiggle-1channel-12-13.pck'))

def test_1channel_composite_feature():
    f = f4
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-composite-1channel-12-13.pck'))

def test_2channel_composite_feature():
    f = f4
    run_matched(f, os.path.join(rundir,
                    'toy-data/test-04-composite-2channel-12-13.pck'), 2)


def test_convex_hull():
    ws = np.array([[1, 2, 2],
                   [1, 1, 2],
                   [1, 2, 2]], dtype=np.uint8)
    chull = features.convex_hull.Manager()
    g = agglo.Rag(ws, feature_manager=chull, use_slow=True)
    expected = np.array([0.5, 0.125, 0.5, 0.1, 1., 0.167, 0.025, 0.069,
                         0.44, 0.056, 1.25, 1.5, 1.2, 0.667])
    assert_allclose(chull(g, 1, 2), expected, atol=0.01, rtol=1.)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

