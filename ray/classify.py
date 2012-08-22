#!/usr/bin/env python

# system modules
import logging
from random import shuffle
import json

# libraries
import h5py
import numpy as np
np.seterr(divide='ignore')

try:
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, LinearRegression
except ImportError:
    logging.warning('scikits.learn not found. SVC, Regression not available.')

try:
    from vigra.learning import RandomForest as VigraRandomForest
    from vigra.__version__ import version as vigra_version
    vigra_version = tuple(map(int, vigra_version.split('.')))
except ImportError:
    logging.warning(' vigra library is not available. '+
        'Cannot use random forest classifier.')

# local imports
import iterprogress as ip
from .adaboost import AdaBoost


def h5py_stack(fn):
    try:
        a = np.array(h5py.File(fn, 'r')['stack'])
    except Exception as except_inst:
        print except_inst
        raise
    return a

class RandomForest(object):
    def __init__(self, ntrees=255, use_feature_importance=False, 
            sample_classes_individually=False):
        self.rf = VigraRandomForest(treeCount=ntrees, 
            sample_classes_individually=sample_classes_individually)
        self.use_feature_importance = use_feature_importance
        self.sample_classes_individually = sample_classes_individually

    def fit(self, features, labels, num_train_examples=None, **kwargs):
        idxs = range(len(features))
        shuffle(idxs)
        idxs = idxs[:num_train_examples]
        features = self.check_features_vector(features[idxs])
        labels = self.check_labels_vector(labels[idxs])
        if self.use_feature_importance:
            self.oob, self.feature_importance = \
                        self.rf.learnRFWithFeatureSelection(features, labels)
        else:
            self.oob = self.rf.learnRF(features, labels)
        return self

    def predict_proba(self, features):
        features = self.check_features_vector(features)
        return self.rf.predictProbabilities(features)

    def predict(self, features):
        features = self.check_features_vector(features)
        return self.rf.predictLabels(features)

    def check_features_vector(self, features):
        if features.dtype != np.float32:
            features = features.astype(np.float32)
        if features.ndim == 1:
            features = features[np.newaxis, :]
        return features

    def check_labels_vector(self, labels):
        if labels.dtype != np.uint32:
            if len(np.unique(labels[labels < 0])) == 1 \
                                                and not (labels==0).any():
                labels[labels < 0] = 0
            else:
                labels = labels + labels.min()
            labels = labels.astype(np.uint32)
        labels = labels.reshape((labels.size, 1))
        return labels

    def save_to_disk(self, fn, rfgroupname='rf', overwrite=True, json_fm=None):
        self.rf.writeHDF5(fn, rfgroupname, overwrite)
        attr_list = ['oob', 'feature_importance', 'use_feature_importance']
        f = h5py.File(fn)
        for attr in attr_list:
            if hasattr(self, attr):
                f[attr] = getattr(self, attr)
        if json_fm is not None:
            f['feature_description'] = json.dumps(json_fm)

    def load_from_disk(self, fn, rfgroupname='rf'):
        self.rf = VigraRandomForest(str(fn), rfgroupname)
        f = h5py.File(fn, 'r')
        groups = []
        f.visit(groups.append)
        attrs = [g for g in groups if not g.startswith(rfgroupname)]
        for attr in attrs:
            setattr(self, attr, np.array(f[attr]))
        json_data = None
        if 'feature_description' in groups:
            json_data = json.loads(str(np.array(f['feature_description'])))
        return json_data

def read_rf_info(fn):
    f = h5py.File(fn)
    return map(np.array, [f['oob'], f['feature_importance']])

def concatenate_data_elements(alldata):
    """Return one big learning set from a list of learning sets.
    
    A learning set is a list/tuple of length 4 containing features, labels,
    weights, and node merge history.
    """
    return map(np.concatenate, zip(*alldata))

def unique_learning_data_elements(alldata):
    if type(alldata[0]) not in (list, tuple): alldata = [alldata]
    f, l, w, h = concatenate_data_elements(alldata)
    af = f.view('|S%d'%(f.itemsize*(len(f[0]))))
    _, uids, iids = np.unique(af, return_index=True, return_inverse=True)
    bcs = np.bincount(iids)
    logging.debug(
        'repeat feature vec min %d, mean %.2f, median %.2f, max %d.' %
        (bcs.min(), np.mean(bcs), np.median(bcs), bcs.max())
    )
    def get_uniques(ar): return ar[uids]
    return map(get_uniques, [f, l, w, h])

def save_training_data_to_disk(data, fn, names=None, info='N/A'):
    if names is None:
        names = ['features', 'labels', 'weights', 'history']
    fout = h5py.File(fn, 'w')
    for data_elem, name in zip(data, names):
        fout[name] = data_elem
    fout.attrs['info'] = info
    fout.close()

def load_training_data_from_disk(fn, names=None, info='N/A'):
    if names is None:
        names = ['features', 'labels', 'weights', 'history']
    fin = h5py.File(fn, 'r')
    data = []
    for name in names:
        data.append(np.array(fin[name]))
    return data

def boundary_overlap_threshold(boundary_idxs, gt, tol_false, tol_true):
    """Return -1, 0 or 1 by thresholding overlaps between boundaries."""
    n = len(boundary_idxs)
    gt_boundary = 1-gt.ravel()[boundary_idxs].astype(bool)
    fraction_true = gt_boundary.astype(np.double).sum() / n
    if fraction_true > tol_true:
        return 1
    elif fraction_true > tol_false:
        return 0
    else:
        return -1

def make_thresholded_boundary_overlap_loss(tol_false, tol_true):
    """Return a merge loss function based on boundary overlaps."""
    def loss(g, n1, n2, gt):
        boundary_idxs = list(g[n1][n2]['boundary'])
        return \
            boundary_overlap_threshold(boundary_idxs, gt, tol_false, tol_true)
    return loss

def label_merges(g, merge_history, feature_map_function, gt, loss_function):
    """Replay an agglomeration history and label the loss of each merge."""
    labels = np.zeros(len(merge_history))
    number_of_features = feature_map_function(g, *g.edges_iter().next()).size
    features = np.zeros((len(merge_history), number_of_features))
    labeled_image = np.zeros(gt.shape, np.double)
    for i, nodes in enumerate(ip.with_progress(
                            merge_history, title='Replaying merge history...', 
                            pbar=ip.StandardProgressBar())):
        n1, n2 = nodes
        features[i,:] = feature_map_function(g, n1, n2)
        labels[i] = loss_function(g, n1, n2, gt)
        labeled_image.ravel()[list(g[n1][n2]['boundary'])] = 2+labels[i]
        g.merge_nodes(n1,n2)
    return features, labels, labeled_image

def select_classifier(cname, features=None, labels=None, **kwargs):
    if 'svm'.startswith(cname):
        del kwargs['class_weight']
        c = SVC(probability=True, **kwargs)
    elif 'logistic-regression'.startswith(cname):
        c = LogisticRegression()
    elif 'linear-regression'.startswith(cname):
        c = LinearRegression()
    elif 'random-forest'.startswith(cname):
        try:
            c = RandomForest()
        except NameError:
            logging.warning(' Tried to use random forest, but not available.'+
                ' Falling back on adaboost.')
            cname = 'ada'
    if 'adaboost'.startswith(cname):
        c = AdaBoost(**kwargs)
    if features is not None and labels is not None:
        c = c.fit(features, labels, **kwargs)
    return c

