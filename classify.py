#!/usr/bin/env python

import sys, os, argparse
import cPickle
from math import sqrt

import h5py
from numpy import bool, array, double, zeros, mean, random, concatenate, where,\
    uint8, ones
from scipy.stats import sem
from scikits.learn.svm import SVC
from scikits.learn.linear_model import LogisticRegression, LinearRegression
from milk.supervised import randomforest as rf
from agglo import best_possible_segmentation, Rag, boundary_mean, \
    classifier_probability
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack

def mean_and_sem(g, n1, n2):
    bvals = g.probabilities.ravel()[list(g[n1][n2]['boundary'])]
    return array([mean(bvals), sem(bvals)]).reshape(1,2)

def feature_set_a(g, n1, n2):
    lb = g[n1][n2]['n']
    mb = g[n1][n2]['sump']/lb
    try:
        vb = max(0, g[n1][n2]['sump2']/(lb-1) - lb/(lb-1)*mb*mb)
    except ZeroDivisionError:
        vb = 0
    sb = sqrt(vb/lb)
    l1 = len(g.node[n1]['extent'])
    m1 = g.node[n1]['sump']/l1
    try:
        v1 = max(0, g.node[n1]['sump2']/(l1-1) - l1/(l1-1)*m1*m1)
    except ZeroDivisionError:
        v1 = 0
    s1 = sqrt(v1/l1)
    l2 = len(g.node[n2]['extent'])
    m2 = g.node[n2]['sump']/l2
    try:
        v2 = max(0, g.node[n2]['sump2']/(l2-1) - l2/(l2-1)*m2*m2)
    except ZeroDivisionError:
        v2 = 0
    s2 = sqrt(v2/l2)
    return array([mb, sb, lb, m1, s1, l1, m2, s2, l2]).reshape(1,9)

def h5py_stack(fn):
    try:
        a = array(h5py.File(fn, 'r')['stack'])
    except Exception as except_inst:
        print except_inst
        raise
    return a
    
class RandomForest(object):
    def __init__(self, ntrees=255):
        self.learn = rf.rf_learner(rf=ntrees)

    def fit(self, features, labels, with_progress=False):
        self.model = self.learn.train(features, labels)

    def predict_proba(self, features):
        n = len(self.model.forest)
        result = zeros((len(features),2), double)
        for i in xrange(len(features)):
            votes = sum(t.apply(features[i]) for t in self.model.forest)
            result[i,1] = double(votes) / n
        result[:,0] = 1-result[:,1]
        return result

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
        labeled_image.ravel()[list(g[n1][n2]['boundary'])] = labels[i]
        g.merge_nodes(n1,n2)
    return features, labels, labeled_image

def pickled(fn):
    return cPickle.load(open(fn, 'r'))

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
    type=eval, default=feature_set_a,
    help='Use named function as feature map (ignored when -c is not used).'
)
arggroup.add_argument('-T', '--training-data', metavar='HDF5_FN', type=str,
    help='Load training data from file.'
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
    parser.add_argument('-T', '--max-threshold', type=float, default=255,
        help='Agglomerate until this threshold'
    )
    parser.add_argument('-E', '--true-tolerance', metavar='FLOAT', 
        type=float, default=0.9,
        help='''If and only if a boundary overlaps over more than fraction
            FLOAT of true boundary, use as a positive training example.'''
    )
    parser.add_argument('-e', '--false-tolerance', metavar='FLOAT',
        type=float, default=0.1,
        help='''If and only if a boundary overlaps over less than fraction
            FLOAT of a true boundary, use as a negative training example.'''
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
    parser.add_argument('-i', '--training-image', type=str,
        help='Save the positive and negative examples as an h5 image.'
    )
    parser.add_argument('-C', '--classifier-training', type=str,
        help='Use a previously trained classifier to train this one.'
    )
    args = parser.parse_args()

    feature_map_function = feature_set_a
    gt = best_possible_segmentation(args.ws, args.gt)
    gt = morpho.pad(gt, [0,0]) # put gt on same coordinate system as seg
    if args.classifier_training is not None:
        c = cPickle.load(open(args.classifier_training))
        mpf = classifier_probability(feature_map_function, c)
    else:
        mpf = boundary_mean
    g = Rag(args.ws, args.probs, show_progress=True,
                                                merge_priority_function=mpf)
    merge_history = g.agglomerate(args.max_threshold, save_history=True)
    g.merge_queue.finish()
    g = Rag(args.ws, args.probs, show_progress=True)
    loss = make_thresholded_boundary_overlap_loss(args.false_tolerance, 
                                                  args.true_tolerance)
    samples, labels, labeled_image = \
                                label_merges(g, merge_history, gt, loss)
    if args.training_image is not None:
        write_h5_stack(labeled_image, args.training_image)
    features = features[labels != 0,:]
    labels = labels[labels != 0]
    if args.classifier_training is not None:
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
    if args.classifier_training is not None:
        f = h5py.File(args.save_training_data)
    if 'svm'.startswith(args.classifier):
        c = SVC(kernel=args.kernel, probability=True).fit(features, labels,
                                                             class_weight=cw)
    elif 'logistic-regression'.startswith(args.classifier):
        c = LogisticRegression().fit(features, labels, class_weight=cw)
    elif 'linear-regression'.startswith(args.classifier):
        c = LinearRegression().fit(features, labels)
    elif 'random-forest'.startswith(args.classifier):
        c = RandomForest()
        c.fit(features, labels)
    print "saving classifier..."
    cPickle.dump(c, open(os.path.expanduser(args.fout), 'w'), -1)
    if args.save_training_data is not None:
        try:
            os.remove(args.save_training_data)
        except OSError:
            pass
        f = h5py.File(args.save_training_data)
        f['samples'] = features
        f['labels'] = labels
