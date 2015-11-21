from __future__ import absolute_import
import os
import glob
from contextlib import contextmanager

PYTHON_VERSION = sys.version_info[0]

from numpy.testing import assert_allclose, assert_array_equal
import numpy as np
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
import subprocess as sp
from gala import imio, classify, features, agglo, evaluate as ev
from six.moves import map, cPickle as pickle


@contextmanager
def tar_extract(fn):
    sp.call(['tar', '-xzf', fn + '.tar.gz'])
    ext_fn = os.path.basename(fn)
    yield ext_fn
    os.remove(ext_fn)
    for sub_fn in glob.glob(ext_fn + '_*'):
        os.remove(sub_fn)


rundir = os.path.dirname(__file__)

# load example data

train_list = ['example-data/train-gt.lzf.h5', 'example-data/train-p1.lzf.h5',
              'example-data/train-p4.lzf.h5', 'example-data/train-ws.lzf.h5']
train_list = [os.path.join(rundir, fn) for fn in train_list]
gt_train, pr_train, p4_train, ws_train = map(imio.read_h5_stack, train_list)
test_list = ['example-data/test-gt.lzf.h5', 'example-data/test-p1.lzf.h5',
             'example-data/test-p4.lzf.h5', 'example-data/test-ws.lzf.h5']
test_list = [os.path.join(rundir, fn) for fn in test_list]
gt_test, pr_test, p4_test, ws_test = map(imio.read_h5_stack, test_list)

# prepare feature manager
fm = features.moments.Manager()
fh = features.histogram.Manager()
fc = features.base.Composite(children=[fm, fh])

### helper functions


def load_pickle(fn):
    with open(fn, 'rb') as fin:
        if PYTHON_VERSION == 3:
            return pickle.load(fin, encoding='bytes', fix_imports=True)
        else:  # Python 2
            return pickle.load(fin)


def load_training_data(fn):
    io = np.load(fn)
    X, y = io['X'], io['y']
    if y.ndim > 1:
        y = y[:, 0]
    return X, y

def save_training_data(fn, X, y):
    np.savez(fn, X=X, y=y)

def train_and_save_classifier(training_data_file, filename,
                              classifier_kind='random forest'):
    X, y = load_training_data(training_data_file)
    cl = classify.get_classifier(classifier_kind)
    cl.fit(X, y)
    classify.save_classifier(cl, filename, use_joblib=False)

### tests

def test_generate_examples_1_channel():
    """Run a flat epoch and an active epoch of learning, compare learned sets.

    The *order* of the edges learned by learn_flat is not guaranteed, so we
    test the *set* of learned edges for the flat epoch. The learned epoch
    *should* have a fixed order, so we test array equality.

    Uses 1 channel probabilities.
    """
    g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
    _, alldata = g_train.learn_agglomerate(gt_train, fc,
                                           learning_mode='permissive',
                                           classifier='naive bayes')
    testfn = ('example-data/train-naive-bayes-merges1-py3.pck'
              if PYTHON_VERSION == 3 else
              'example-data/train-naive-bayes-merges1-py2.pck')
    exp0, exp1 = load_pickle(os.path.join(rundir, testfn))
    expected_edges = set(map(tuple, exp0))
    edges = set(map(tuple, alldata[0][3]))
    merges = alldata[1][3]
    assert edges == expected_edges
    # concordant is the maximum edges concordant in the Python 2.7 version.
    # The remaining edges diverge because of apparent differences
    # between Linux and OSX floating point handling.
    concordant = slice(None, 171) if PYTHON_VERSION == 2 else slice(None)
    assert_array_equal(merges[concordant], exp1[concordant])
    nb = GaussianNB().fit(alldata[0][0], alldata[0][1][:, 0])
    nbexp = joblib.load(os.path.join(rundir,
                                     'example-data/naive-bayes-1.joblib'))
    assert_allclose(nb.theta_, nbexp.theta_, atol=1e-10)
    assert_allclose(nb.sigma_, nbexp.sigma_, atol=1e-4)
    assert_allclose(nb.class_prior_, nbexp.class_prior_, atol=1e-7)


def test_segment_with_classifer_1_channel():
    if PYTHON_VERSION == 2:
        rf = classify.load_classifier(
            os.path.join(rundir, 'example-data/rf-1.joblib'))
    else:
        fn = os.path.join(rundir, 'example-data/rf1-py3.joblib')
        with tar_extract(fn) as fn:
            rf = joblib.load(fn)
    learned_policy = agglo.classifier_probability(fc, rf)
    g_test = agglo.Rag(ws_test, pr_test, learned_policy, feature_manager=fc)
    g_test.agglomerate(0.5)
    seg_test = g_test.get_segmentation()
    #imio.write_h5_stack(seg_test, 'example-data/test-seg-1.lzf.h5')
    seg_expected = imio.read_h5_stack(
        os.path.join(rundir, 'example-data/test-seg-1.lzf.h5'))
    assert_allclose(ev.vi(seg_test, seg_expected), 0.0)


def test_generate_examples_4_channel():
    """Run a flat epoch and an active epoch of learning, compare learned sets.

    The *order* of the edges learned by learn_flat is not guaranteed, so we
    test the *set* of learned edges for the flat epoch. The learned epoch
    *should* have a fixed order, so we test array equality.

    Uses 4 channel probabilities.
    """
    g_train = agglo.Rag(ws_train, p4_train, feature_manager=fc)
    _, alldata = g_train.learn_agglomerate(gt_train, fc,
                                           learning_mode='permissive',
                                           classifier='naive bayes')
    testfn = ('example-data/train-naive-bayes-merges4-py3.pck'
              if PYTHON_VERSION == 3 else
              'example-data/train-naive-bayes-merges4-py2.pck')
    exp0, exp1 = load_pickle(os.path.join(rundir, testfn))
    expected_edges = set(map(tuple, exp0))
    edges = set(map(tuple, alldata[0][3]))
    merges = alldata[1][3]
    assert edges == expected_edges
    assert_array_equal(merges, exp1)
    nb = GaussianNB().fit(alldata[0][0], alldata[0][1][:, 0])
    nbexp = joblib.load(os.path.join(rundir,
                                     'example-data/naive-bayes-4.joblib'))
    assert_allclose(nb.theta_, nbexp.theta_, atol=1e-10)
    assert_allclose(nb.sigma_, nbexp.sigma_, atol=1e-4)
    assert_allclose(nb.class_prior_, nbexp.class_prior_, atol=1e-7)


def test_segment_with_classifier_4_channel():
    if PYTHON_VERSION == 2:
        rf = classify.load_classifier(
            os.path.join(rundir, 'example-data/rf-4.joblib'))
    else:
        fn = os.path.join(rundir, 'example-data/rf4-py3.joblib')
        with tar_extract(fn) as fn:
            rf = joblib.load(fn)
    learned_policy = agglo.classifier_probability(fc, rf)
    g_test = agglo.Rag(ws_test, p4_test, learned_policy, feature_manager=fc)
    g_test.agglomerate(0.5)
    seg_test = g_test.get_segmentation()
    seg_expected = imio.read_h5_stack(
            os.path.join(rundir, 'example-data/test-seg-4.lzf.h5'))
    assert_allclose(ev.vi(seg_test, seg_expected), 0.0)


def test_split_vi():
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
