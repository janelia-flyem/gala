import os

D = os.path.dirname(os.path.abspath(__file__))

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from gala import agglo
from gala import evaluate as ev
from gala.features import default


fn_prob = os.path.join(D, 'toy-data/test-01-probabilities.txt')
prob = np.loadtxt(fn_prob)
fn_ws = os.path.join(D, 'toy-data/test-01-watershed.txt')
ws = np.loadtxt(fn_ws, dtype=np.uint32)
fn_gt = os.path.join(D, 'toy-data/test-01-groundtruth.txt')
results = np.loadtxt(fn_gt, dtype=np.uint32)


ans12 = np.array(
      [3.00, 2.00, 4.67, 6.00, 32.67,0.50, 0.00, 0.00, 0.00, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.50, 0.01, 0.04, 0.99,
       3.00, 4.00, 7.00, 1.29, 1.92, 2.05, 9.00, 0.75, 0.00, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.25, 0.01,
       0.03, 0.98, 4.00, 3.50, 2.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
       0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
       1.00, 0.96, 0.98, 1.00, 2.00, 4.00, 0.71, 2.75, 3.95, 23.67, 0.03,
       -1.00, 0.50])


contact = np.array(
      [1.00, 1.50, 0.60, 2.10, 1.00, 1.50, 0.60,
       2.10, 1.00, 1.50, 0.60, 2.10, 0.00, 0.41,
       -0.51,0.74, 0.00, 0.41,-0.51, 0.74, 0.00,
       0.41,-0.51, 0.74,-0.41,-1.25, 0.67, 0.29])


def test_paper_em():
    feat = default.paper_em()
    g = agglo.Rag(ws, prob, feature_manager=feat)
    assert_allclose(feat(g, 1, 2), ans12, atol=0.01)


def test_snemi():
    feat = default.snemi3d()
    g = agglo.Rag(ws, prob, feature_manager=feat)
    # contact are edge features, so they are inserted just before the 8
    # difference features in the base paper_em vector.
    expected = np.concatenate((ans12[:-8], contact, ans12[-8:]))
    assert_allclose(feat(g, 1, 2), expected, atol=0.01)


if __name__ == '__main__':
    from numpy import testing
    testing.run_module_suite()

