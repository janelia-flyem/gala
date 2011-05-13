#!/usr/bin/env python

import sys, os, argparse

import cPickle

import h5py
from numpy import bool, array, double, zeros, mean, random, concatenate, where
from scipy.stats import sem
from scikits.learn.svm import SVC
from scikits.learn.linear_model import LogisticRegression, LinearRegression
from agglo import best_possible_segmentation, Rag
import morpho
import iterprogress as ip

def mean_and_sem(g, n1, n2):
    bvals = g.probabilities.ravel()[list(g[n1][n2]['boundary'])]
    return array([mean(bvals), sem(bvals)]).reshape(1,2)

def feature_set_a(g, n1, n2):
    bvals = g.probabilities.ravel()[list(g[n1][n2]['boundary'])]
    body1 = g.probabilities.ravel()[list(g.node[n1]['extent'])]
    body2 = g.probabilities.ravel()[list(g.node[n2]['extent'])]
    return array([mean(bvals), sem(bvals), bvals.size, 
                mean(body1), sem(body1), body1.size,
                mean(body2), sem(body2), body2.size]).reshape(1,9)

def h5py_stack(fn):
    try:
        a = array(h5py.File(fn, 'r')['stack'])
    except Exception as except_inst:
        print except_inst
        raise
    return a

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
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
    parser.add_argument('-E', '--true-tolerance', metavar='FLOAT', default=0.9,
        help='''If and only if a boundary overlaps over more than fraction
            FLOAT of true boundary, use as a positive training example.'''
    )
    parser.add_argument('-e', '--false-tolerance', metavar='FLOAT', default=0.1,
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
    parser.add_argument('-c', '--classifier', default='svm', 
        help='''Choose the classifier to use. Default: svm (support vector 
            machine). Options: svm, logistic-regression, linear-regression.'''
    )
    parser.add_argument('-k', '--kernel', default='rbf',
        help='The kernel for an SVM classifier.'
    )
    args = parser.parse_args()

    #feature_map_function = mean_and_sem
    feature_map_function = feature_set_a
    bps_boundaries = morpho.pad(
        1-best_possible_segmentation(args.ws, args.gt).astype(bool), [0,0]
    )
    g = Rag(args.ws, args.probs, show_progress=True)
    merge_history = list(g.agglomerate(args.max_threshold, generate=True))
    g.merge_queue.finish()
    g = Rag(args.ws, args.probs, show_progress=True)
    labels = zeros(len(merge_history))
    number_of_features = feature_map_function(g, *g.edges_iter().next()).size
    features = zeros((len(merge_history), number_of_features))
    print "generating features..."
    for i, nodes in enumerate(ip.with_progress(
            merge_history, title='Replaying merge history...', 
            pbar=ip.StandardProgressBar())):
        n1, n2 = nodes
        features[i,:] = feature_map_function(g, n1, n2)
        boundary_idxs = list(g[n1][n2]['boundary'])
        fraction_true = bps_boundaries.ravel()[boundary_idxs].\
            astype(double).sum()/len(boundary_idxs)
        if fraction_true > args.true_tolerance:
            labels[i] = 1
        elif fraction_true < args.false_tolerance:
            labels[i] = -1
        g.merge_nodes(n1,n2)
    features = features[labels != 0,:]
    labels = labels[labels != 0]
    print "fitting classifier of size: ", labels.size
    if args.balance_classes:
        cw = 'auto'
    else:
        cw = {-1:1, 1:1}
    if 'svm'.startswith(args.classifier):
        c = SVC(kernel=args.kernel, probability=True).fit(features, labels,
                                                             class_weight=cw)
    elif 'logistic-regression'.startswith(args.classifier):
        c = LogisticRegression().fit(features, labels, class_weight=cw)
    elif 'linear-regression'.startswith(args.classifier):
        c = LinearRegression().fit(features, labels)
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
