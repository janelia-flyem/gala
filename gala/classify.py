#!/usr/bin/env python

# system modules
import os
import logging
import random
import cPickle as pck

# libraries
import h5py
import numpy as np
np.seterr(divide='ignore')

try:
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.externals import joblib
except ImportError:
    logging.warning('scikit-learn not found.')
    sklearn_available = False
else:
    sklearn_available = True

try:
    from vigra.learning import RandomForest as BaseVigraRandomForest
    from vigra.__version__ import version as vigra_version
    vigra_version = tuple(map(int, vigra_version.split('.')))
except ImportError:
    vigra_available = False
else:
    vigra_available = True

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

def default_classifier_extension(cl, use_joblib=True):
    """
    Return the default classifier file extension for the given classifier cl.

    Returns:
        String of file extension
    """
    if isinstance(cl, VigraRandomForest):
        return ".classifier.h5"
    elif use_joblib and sklearn_available:
        return ".classifier.joblib"
    else:
        return ".classifier"


def load_classifier(fn):
    """Load a classifier previously saved to disk, given a filename.
    
    Supported classifier types are:
    - scikit-learn classifiers saved using either pickle or joblib persistence
    - vigra random forest classifiers saved in HDF5 format

    Parameters
    ----------
    fn : string
        Filename in which the classifier is stored.

    Returns
    -------
    cl : classifier object
        cl is one of the supported classifier types; these support at least
        the standard scikit-learn interface of `fit()` and `predict_proba()`
    """
    if not os.path.exists(fn):
        raise IOError("No such file or directory: '%s'" % fn)
    try:
        with open(fn, 'r') as f:
            cl = pck.load(f)
        return cl
    except pck.UnpicklingError:
        pass
    if sklearn_available:
        try:
            cl = joblib.load(fn)
            return cl
        except KeyError:
            pass
    if vigra_available:
        cl = VigraRandomForest()
        try:
            cl.load_from_disk(fn)
            return cl
        except IOError as e:
            logging.error(e)
        except RuntimeError as e:
            logging.error(e)
    raise IOError("File '%s' does not appear to be a valid classifier file"
        % fn)

def save_classifier(cl, fn, use_joblib=True, **kwargs):
    """Save a classifier to disk.

    Parameters
    ----------
    cl : classifier object
        Pickleable object or a classify.VigraRandomForest object.
    fn : string
        Writeable path/filename.
    use_joblib : bool, optional
        Whether to prefer joblib persistence to pickle.
    kwargs : keyword arguments
        Keyword arguments to be passed on to either `pck.dump` or 
        `joblib.dump`.

    Returns
    -------
    None

    Notes
    -----
    For joblib persistence, `compress=3` is the default.
    """
    if isinstance(cl, VigraRandomForest):
        cl.save_to_disk(fn)
    elif use_joblib and sklearn_available:
        if not kwargs.has_key('compress'):
            kwargs['compress'] = 3
        joblib.dump(cl, fn, **kwargs)
    else:
        with open(fn, 'w') as f:
            pck.dump(cl, f, protocol=kwargs.get('protocol', -1))


def get_classifier(name='random forest', *args, **kwargs):
    name = name.lower()
    is_random_forest = name.find('random') > -1 and name.find('forest') > -1
    if vigra_available and is_random_forest:
        return VigraRandomForest(*args, **kwargs)
    elif sklearn_available and is_random_forest:
        return DefaultRandomForest(*args, **kwargs)
    else:
        raise NotImplementedError('Classifier "%s" is either not installed ' +
            'or not implemented in Gala.')

class DefaultRandomForest(RandomForestClassifier):
    def __init__(self, n_estimators=100, criterion='entropy', max_depth=20,
            bootstrap=False, random_state=None):
        super(DefaultRandomForest, self).__init__(
            n_estimators=n_estimators, criterion=criterion,
            max_depth=max_depth, bootstrap=bootstrap, random_state=random_state)


class VigraRandomForest(object):
    def __init__(self, ntrees=255, use_feature_importance=False, 
            sample_classes_individually=False):
        self.rf = BaseVigraRandomForest(treeCount=ntrees, 
            sample_classes_individually=sample_classes_individually)
        self.use_feature_importance = use_feature_importance
        self.sample_classes_individually = sample_classes_individually

    def fit(self, features, labels):
        features = self.check_features_vector(features)
        labels = self.check_labels_vector(labels)
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

    def save_to_disk(self, fn, rfgroupname='rf'):
        self.rf.writeHDF5(fn, rfgroupname)
        attr_list = ['oob', 'feature_importance', 'use_feature_importance',
            'feature_description']
        f = h5py.File(fn)
        for attr in attr_list:
            if hasattr(self, attr):
                f[rfgroupname].attrs[attr] = getattr(self, attr)

    def load_from_disk(self, fn, rfgroupname='rf'):
        self.rf = BaseVigraRandomForest(str(fn), rfgroupname)
        f = h5py.File(fn, 'r')
        for attr in f[rfgroupname].attrs:
            print("f[%s] = %s" % (attr, f[rfgroupname].attrs[attr]))
            setattr(self, attr, f[rfgroupname].attrs[attr])


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

def sample_training_data(features, labels, num_samples=None):
    """Get a random sample from a classification training dataset.

    Parameters
    ----------
    features: np.ndarray [M x N]
        The M (number of samples) by N (number of features) feature matrix.
    labels: np.ndarray [M] or [M x 1]
        The training label for each feature vector.
    num_samples: int, optional
        The size of the training sample to draw. Return full dataset if `None`
        or if num_samples >= M.

    Returns
    -------
    feat: np.ndarray [num_samples x N]
        The sampled feature vectors.
    lab: np.ndarray [num_samples] or [num_samples x 1]
        The sampled training labels
    """
    m = len(features)
    if num_samples is None or num_samples >= m:
        return features, labels
    idxs = random.sample(range(m), num_samples)
    return features[idxs], labels[idxs]

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
        if sklearn_available:
            c = DefaultRandomForest()
        elif vigra_available:
            c = VigraRandomForest()
        else:
            raise RuntimeError('tried to use random forest classifier, ' +
                'but neither scikit-learn nor vigra are available.')
    elif 'adaboost'.startswith(cname):
        c = AdaBoost(**kwargs)
    if features is not None and labels is not None:
        c = c.fit(features, labels, **kwargs)
    return c

