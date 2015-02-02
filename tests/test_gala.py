from __future__ import absolute_import
import os
import sys
import glob
from contextlib import contextmanager

PYTHON_VERSION = sys.version_info[0]

from numpy.testing import assert_allclose
import numpy as np
from sklearn.externals import joblib
import subprocess as sp
from gala import imio, classify, features, agglo, evaluate as ev
from six.moves import map


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
    g_train = agglo.Rag(ws_train, pr_train, feature_manager=fc)
    np.random.RandomState(0)
    (X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc, random_state=0)[0]
    X_expected, y_expected = load_training_data(
            os.path.join(rundir, 'example-data/train-set-1.npz'))
    assert_allclose(X, X_expected, atol=1e-6)
    assert_allclose(y, y_expected, atol=1e-6)


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
    g_train = agglo.Rag(ws_train, p4_train, feature_manager=fc)
    np.random.RandomState(0)
    (X, y, w, merges) = g_train.learn_agglomerate(gt_train, fc, random_state=0)[0]
    X_expected, y_expected = load_training_data(
            os.path.join(rundir, 'example-data/train-set-4.npz'))
    assert_allclose(X, X_expected, atol=1e-6)
    assert_allclose(y, y_expected, atol=1e-6)


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
