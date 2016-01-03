import os
import glob
from contextlib import contextmanager

import pytest

from numpy.testing import assert_allclose
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
import subprocess as sp
from gala import imio, features, agglo, evaluate as ev


@contextmanager
def tar_extract(fn):
    sp.call(['tar', '-xzf', fn + '.tar.gz'])
    ext_fn = os.path.basename(fn)
    yield ext_fn
    os.remove(ext_fn)
    for sub_fn in glob.glob(ext_fn + '_*'):
        os.remove(sub_fn)


rundir = os.path.dirname(__file__)

# prepare feature manager

def _compute_delta_vi(ctable, fragments0, fragments1):
    c0 = np.sum(ctable[list(fragments0)], axis=0)
    c1 = np.sum(ctable[list(fragments1)], axis=0)
    cr = c0 + c1
    p0 = np.sum(c0)
    p1 = np.sum(c1)
    pr = np.sum(cr)
    p0g = np.sum(ev.xlogx(c0))
    p1g = np.sum(ev.xlogx(c1))
    prg = np.sum(ev.xlogx(cr))
    return (pr * np.log2(pr) - p0 * np.log2(p0) - p1 * np.log2(p1) -
            2 * (prg - p0g - p1g))


class _f_mock(features.base.Null):
    '''
    Mock feature manager to verify agglomerative learning works.

    This manager learns a different feature map for fragments vs
    agglomerated segments. It relies on knowing the ground truth for a
    given fragmentation.

    Parameters
    ----------
    frag, gt : array of int, same shape
        The fragmentation and ground truth volumes. Must have same shape.
    '''
    def __init__(self, frag, gt):
        self.ctable = ev.contingency_table(frag, gt, ignore_seg=[],
                                           ignore_gt=[]).toarray()
        self._std = 0.1  # standard deviation of feature computations

    def eps(self):
        return np.random.randn(2) * self._std

    def compute_features(self, g, n1, n2=None):
        if n2 is None:
            return np.array([])
        f1, f2 = g.node[n1]['fragments'], g.node[n2]['fragments']
        should_merge = _compute_delta_vi(self.ctable, f1, f2) < 0
        if should_merge:
            return np.array([0., 0.]) + self.eps()
        else:
            if len(f1) + len(f2) == 2:  # single-fragment merge
                return np.array([1., 0.]) + self.eps()
            else:  # multi-fragment merge
                return np.array([0., 1.]) + self.eps()


### fixtures

@pytest.fixture
def dummy_data():
    frag = np.arange(1, 17, dtype=int).reshape((4, 4))
    gt = np.array([[1, 1, 2, 2], [1, 1, 2, 2], [3] * 4, [3] * 4], dtype=int)
    fman = _f_mock(frag, gt)
    g = agglo.Rag(frag, feature_manager=fman)
    return frag, gt, g, fman


### tests

def test_generate_flat_learning_edges(dummy_data):
    """Run a flat epoch and ensure all edges are correctly represented."""
    frag, gt, g, fman = dummy_data
    feat, target, weights, edges = g.learn_flat(gt, fman)
    assert feat.shape == (24, 2)
    assert tuple(edges[0]) == (1, 2)
    assert tuple(edges[-1]) == (15, 16)
    assert np.sum(target[:, 0] == 1) == 6  # number of non-merge edges


def test_generate_lash_examples(dummy_data):
    """Run a flat epoch and an active epoch of learning, compare learned sets.

    The mock feature manager places all merge examples at (0, 0) in feature
    space, and all non-merge examples at (1, 0), *in flat learning*. During
    agglomeration, non-merge examples go to (0, 1), which confuses the flat
    classifier (which has only learned the difference along the first feature
    dimension).

    This test checks for those differences in learning using a simple
    logistic regression.
    """
    frag, gt, g, fman = dummy_data
    np.random.seed(5)
    summary, allepochs = g.learn_agglomerate(gt, fman,
                                             learning_mode='permissive',
                                             classifier='logistic regression')
    feat, target, weights, edges = summary
    ffeat, ftarget, fweights, fedges = allepochs[0]  # flat
    lr = LR().fit(feat, target[:, 0])
    flr = LR().fit(ffeat, ftarget[:, 0])
    def pred(v):
        return lr.predict_proba([v])[0, 1]
    def fpred(v):
        return flr.predict_proba([v])[0, 1]
    assert len(allepochs[1][0]) == 15  # number of merges is |nodes| - 1

    # approx. same learning results at (0., 0.) and (1., 0.)
    assert_allclose(fpred([0, 0]), 0.2, atol=0.025)
    assert_allclose(pred([0, 0]), 0.2, atol=0.025)
    assert_allclose(fpred([1, 0]), 0.64, atol=0.025)
    assert_allclose(pred([1, 0]), 0.64, atol=0.025)

    # difference between agglomerative and flat learning in point (0., 1.)
    assert_allclose(fpred([0, 1]), 0.2, atol=0.025)
    assert_allclose(pred([0, 1]), 0.6, atol=0.025)


def test_generate_gala_examples(dummy_data):
    """As `test_generate_lash_examples`, but using strict learning. """
    frag, gt, g, fman = dummy_data
    np.random.seed(5)
    summary, allepochs = g.learn_agglomerate(gt, fman,
                                             learning_mode='strict',
                                             classifier='logistic regression')
    feat, target, weights, edges = summary
    ffeat, ftarget, fweights, fedges = allepochs[0]  # flat
    lr = LR().fit(feat, target[:, 0])
    flr = LR().fit(ffeat, ftarget[:, 0])
    def pred(v):
        return lr.predict_proba([v])[0, 1]
    def fpred(v):
        return flr.predict_proba([v])[0, 1]
    assert len(allepochs[1][0]) == 21 # number of merges is more than LASH

    # approx. same learning results at (0., 0.) and (1., 0.)
    assert_allclose(fpred([0, 0]), 0.2, atol=0.025)
    assert_allclose(pred([0, 0]), 0.2, atol=0.025)
    assert_allclose(fpred([1, 0]), 0.64, atol=0.025)
    assert_allclose(pred([1, 0]), 0.64, atol=0.025)

    # difference between agglomerative and flat learning in point (0., 1.);
    # greater separation than with LASH
    assert_allclose(fpred([0, 1]), 0.2, atol=0.025)
    assert_allclose(pred([0, 1]), 0.7, atol=0.025)


def test_segment_with_gala_classifer(dummy_data):
    frag, gt, g, fman = dummy_data
    np.random.seed(5)
    summary, allepochs = g.learn_agglomerate(gt, fman,
                                             learning_mode='strict',
                                             classifier='logistic regression')
    feat, target, weights, edges = summary
    ffeat, ftarget, fweights, fedges = allepochs[0]  # flat
    lr = LR().fit(feat, target[:, 0])
    gala_policy = agglo.classifier_probability(fman, lr)
    flr = LR().fit(ffeat, ftarget[:, 0])
    flat_policy = agglo.classifier_probability(fman, flr)

    gtest = agglo.Rag(frag, feature_manager=fman,
                      merge_priority_function=gala_policy)
    gtest.agglomerate(0.5)
    assert ev.vi(gtest.get_segmentation(), gt) == 0
    gtest_flat = agglo.Rag(frag, feature_manager=fman,
                           merge_priority_function=flat_policy)
    assert ev.vi(gtest_flat.get_segmentation(0.5), gt) == 1.5


def test_split_vi():
    ws_test = imio.read_h5_stack(
            os.path.join(rundir, 'example-data/test-ws.lzf.h5'))
    gt_test = imio.read_h5_stack(
            os.path.join(rundir, 'example-data/test-gt.lzf.h5'))
    seg_test1 = imio.read_h5_stack(
            os.path.join(rundir, 'example-data/test-seg1.lzf.h5'))
    seg_test4 = imio.read_h5_stack(
            os.path.join(rundir, 'example-data/test-seg4.lzf.h5'))
    result = np.vstack((
        ev.split_vi(ws_test, gt_test),
        ev.split_vi(seg_test1, gt_test),
        ev.split_vi(seg_test4, gt_test)
        ))
    expected = np.load(os.path.join(rundir, 'example-data/vi-results.npy'))
    assert_allclose(result, expected, atol=1e-6)


if __name__ == '__main__':
    np.random.RandomState(0)
    from numpy import testing
    testing.run_module_suite()
