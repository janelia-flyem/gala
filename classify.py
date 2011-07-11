#!/usr/bin/env python

import sys, os, argparse
import cPickle
from math import sqrt

import h5py
from numpy import bool, array, double, zeros, mean, random, concatenate, where,\
    uint8, ones, float32, uint32, unique, newaxis
from scipy.stats import sem
from scikits.learn.svm import SVC
from scikits.learn.linear_model import LogisticRegression, LinearRegression
from vigra.learning import RandomForest as VigraRandomForest
from agglo import best_possible_segmentation, Rag, boundary_mean, \
    classifier_probability, random_priority
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack, write_image_stack

def mean_and_sem(g, n1, n2):
    bvals = g.probabilities.ravel()[list(g[n1][n2]['boundary'])]
    return array([mean(bvals), sem(bvals)]).reshape(1,2)

def mean_sem_and_n_from_cache_dict(d):
    try:
        n = d['n']
    except KeyError:
        n = len(d['extent'])
    m = d['sump']/n
    v = 0 if n==1 else max(0, d['sump2']/(n-1) - n/(n-1)*m*m)
    s = sqrt(v/n)
    return m, s, n

def skew_from_cache_dict(d):
    try:
        n = d['n']
    except KeyError:
        n = len(d['extent'])
    m1 = d['sump']/n
    k1 = m1
    m2 = d['sump2']/n
    k2 = m2 - m1*m1
    m3 = d['sump3']/n
    k3 = m3 - 3*m2*m1 + 2*m1*m1*m1
    return k3 * k2**(-1.5)

def feature_set_a(g, n1, n2):
    """Return the mean, SEM, and size of n1, n2, and the n1-n2 boundary in g.
    
    n1 is defined as the smaller of the two nodes, so the labels are swapped
    accordingly if necessary before computing the statistics.
    
    SEM: standard error of the mean, equal to sqrt(var/n)
    """
    if len(g.node[n1]['extent']) > len(g.node[n2]['extent']):
        n1, n2 = n2, n1
    mb, sb, lb = mean_sem_and_n_from_cache_dict(g[n1][n2])
    m1, s1, l1 = mean_sem_and_n_from_cache_dict(g.node[n1])
    m2, s2, l2 = mean_sem_and_n_from_cache_dict(g.node[n2])
    return array([mb, sb, lb, m1, s1, l1, m2, s2, l2]).reshape(1,9)

def node_feature_set_a(g, n):
    """Return the mean, standard deviation, SEM, size, and skewness of n.

    Uses the probability of boundary within n.
    """
    d = g.node[n]
    m, s, l = mean_sem_and_n_from_cache_dict(d)
    stdev = s*sqrt(l)
    skew = skew_from_cache_dict(d)
    return array([m, stdev, s, l, skew])

def h5py_stack(fn):
    try:
        a = array(h5py.File(fn, 'r')['stack'])
    except Exception as except_inst:
        print except_inst
        raise
    return a
    
class RandomForest(object):
    def __init__(self, ntrees=255, **kwargs):
        self.rf = VigraRandomForest(treeCount=ntrees)

    def fit(self, features, labels, with_progress=False, **kwargs):
        features = self.check_features_vector(features)
        labels = self.check_labels_vector(labels)
        self.oob, self.feature_importance = \
                        self.rf.learnRFWithFeatureSelection(features, labels)
        return self

    def predict_proba(self, features):
        features = self.check_features_vector(features)
        return self.rf.predictProbabilities(features)

    def check_features_vector(self, features):
        if features.dtype != float32:
            features = features.astype(float32)
        if features.ndim == 1:
            features = features[newaxis,:]
        return features

    def check_labels_vector(self, labels):
        if labels.dtype != uint32:
            if len(unique(labels[labels < 0])) == 1 and not (labels==0).any():
                labels[labels < 0] = 0
            else:
                labels = labels + labels.min()
            labels = labels.astype(uint32)
        labels = labels.reshape((labels.size, 1))
        return labels

    def save_to_disk(self, fn, rfgroupname='rf', overwrite=True):
        self.rf.writeHDF5(fn, rfgroupname, overwrite)
        f = h5py.File(fn)
        for attr in ['oob', 'feature_importance']:
            if hasattr(self, attr):
                f[attr] = getattr(self, attr)

    def load_from_disk(self, fn, rfgroupname='rf'):
        self.rf = VigraRandomForest(fn, rfgroupname)
        f = h5py.File(fn, 'r')
        groups = []
        f.visit(groups.append)
        attrs = [g for g in groups if g != rfgroupname]
        for attr in attrs:
            setattr(self, attr, array(f[attr]))


def boundary_overlap_threshold(boundary_idxs, gt, tol_false, tol_true):
    """Return -1, 0 or 1 by thresholding overlaps between boundaries."""
    n = len(boundary_idxs)
    gt_boundary = 1-gt.ravel()[boundary_idxs].astype(bool)
    fraction_true = gt_boundary.astype(double).sum() / n
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
    labels = zeros(len(merge_history))
    number_of_features = feature_map_function(g, *g.edges_iter().next()).size
    features = zeros((len(merge_history), number_of_features))
    labeled_image = zeros(gt.shape, double)
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
        c = RandomForest()
    if features is not None and labels is not None:
        c = c.fit(features, labels, **kwargs)
    return c


def pickled(fn):
    try:
        obj = cPickle.load(open(fn, 'r'))
    except cPickle.UnpicklingError:
        obj = RandomForest()
        obj.load_from_disk(fn)
    return obj

arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Classification options')
arggroup.add_argument('-c', '--classifier', default='svm', 
    help='''Choose the classifier to use. Default: svm (support vector 
        machine). Options: svm, logistic-regression, linear-regression,
        random-forest.'''
)
arggroup.add_argument('-k', '--load-classifier', 
    type=pickled, metavar='PCK_FILE',
    help='Load and use a pickled classifier as a merge priority function.'
)
arggroup.add_argument('-f', '--feature-map-function', metavar='FCT_NAME',
    default='feature_set_a',
    help='Use named function as feature map (ignored when -c is not used).'
)
arggroup.add_argument('-T', '--training-data', metavar='HDF5_FN', type=str,
    help='Load training data from file.'
)
arggroup.add_argument('-N', '--node-split-classifier', metavar='HDF5_FN',
    type=str,
    help='Load a node split classifier and split nodes when required.'
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        parents=[arguments],
        description='Create an agglomeration classifier.'
    )
    parser.add_argument('ws', type=h5py_stack,
        help='Watershed volume, in HDF5 format.'
    )
    parser.add_argument('gt', type=h5py_stack,
        help='Ground truth volume, in HDF5 format also.'
    )
    parser.add_argument('probs', type=h5py_stack,
        help='''Probabilities volume, in HDF ... you get the idea.'''
    )
    parser.add_argument('fout', help='.pck filename to save the classifier.')
    parser.add_argument('-t', '--max-threshold', type=float, default=255,
        help='Agglomerate until this threshold'
    )
    parser.add_argument('-s', '--save-training-data', metavar='FILE',
        help='Save the generated training data to FILE (HDF5 format).'
    )
    parser.add_argument('-b', '--balance-classes', action='store_true',
        default=False, 
        help='Ensure both true edges and false edges are equally represented.'
    )
    parser.add_argument('-K', '--kernel', default='rbf',
        help='The kernel for an SVM classifier.'
    )
    parser.add_argument('-o', '--objective-function', metavar='FCT_NAME', 
        default='random_priority', help='The merge priority function name.'
    )
    parser.add_argument('--save-node-training-data', metavar='FILE',
        help='Save node features and labels to FILE.'
    )
    parser.add_argument('--node-classifier', metavar='FILE',
        help='Train and output a node split classifier.'
    )
    args = parser.parse_args()

    feature_map_function = eval(args.feature_map_function)
    if args.load_classifier is not None:
        mpf = classifier_probability(eval(args.feature_map_function), 
                                                        args.load_classifier)
    else:
        mpf = eval(args.objective_function)

    wsg = Rag(args.ws, args.probs, mpf)
    features, labels, history, ave_sizes = \
                        wsg.learn_agglomerate(args.gt, feature_map_function)

    print 'shapes: ', features.shape, labels.shape

    if args.load_classifier is not None:
        try:
            f = h5py.File(args.save_training_data)
            old_features = array(f['samples'])
            old_labels = array(f['labels'])
            features = concatenate((features, old_features), 0)
            labels = concatenate((labels, old_labels), 0)
        except:
            pass
    print "fitting classifier of size, pos: ", labels.size, (labels==1).sum()
    if args.balance_classes:
        cw = 'auto'
    else:
        cw = {-1:1, 1:1}
    if args.save_training_data is not None:
        try:
            os.remove(args.save_training_data)
        except OSError:
            pass
        f = h5py.File(args.save_training_data)
        f['samples'] = features
        f['labels'] = labels
        f['history'] = history
        f['size'] = ave_sizes
    c = select_classifier(args.classifier, features=features, labels=labels, 
                                        class_weight=cw, kernel=args.kernel)
    print "saving classifier..."
    try:
        cPickle.dump(c, open(os.path.expanduser(args.fout), 'w'), -1)
    except RuntimeError:
        os.remove(os.path.expanduser(args.fout))
        c.save_to_disk(os.path.expanduser(args.fout))
        print 'Warning: unable to pickle classifier to :', args.fout
