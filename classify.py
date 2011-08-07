#!/usr/bin/env python

# system modules
import sys, os, argparse
import cPickle
import logging
from math import sqrt
from abc import ABCMeta, abstractmethod

# libraries
import h5py
from numpy import bool, array, double, zeros, mean, random, concatenate, where,\
    uint8, ones, float32, uint32, unique, newaxis, zeros_like, arange, floor, \
    histogram, seterr
seterr(divide='ignore')
from scipy.misc import comb as nchoosek
from scipy.stats import sem
from scikits.learn.svm import SVC
from scikits.learn.linear_model import LogisticRegression, LinearRegression
try:
    from vigra.learning import RandomForest as VigraRandomForest
except ImportError:
    logging.warning(' vigra library is not available. '+
        'Cannot use random forest classifier.')
    pass

# local imports
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack, write_image_stack
from adaboost import AdaBoost

class NullFeatureManager(object):
    def __init__(self, begin_idx=0, cache_begin_idx=0, *args, **kwargs):
        self.begin_idx = begin_idx
        self.cache_begin_idx = cache_begin_idx
        self.cache_length = 0
        self.parent = None
    def __len__(self, *args, **kwargs):
        return 0
    def feature_range(self):
        return self.begin_idx, self_begin_idx + len(self)
    def cache_range(self):
        return self.cache_begin_idx, self.cache_begin_idx + self.cache_length
    def __call__(self, g, n1, n2):
        return self.compute_features(g, n1, n2)
    def compute_features(self, g, n1, n2):
        if len(g.node[n1]['extent']) > len(g.node[n2]['extent']):
            n1, n2 = n2, n1 # smaller node first
        return concatenate((
            self.compute_node_features(g, n1),
            self.compute_node_features(g, n2),
            self.compute_edge_features(g, n1, n2)
        ))
    def create_node_cache(self, *args, **kwargs):
        pass
    def create_edge_cache(self, *args, **kwargs):
        pass
    def update_node_cache(self, *args, **kwargs):
        pass
    def update_edge_cache(self, *args, **kwargs):
        pass
    def update_node_pixels(self, *args, **kwargs):
        pass
    def update_edge_pixels(self, *args, **kwargs):
        pass
    def compute_node_features(self, *args, **kwargs):
        return array([])
    def compute_edge_features(self, *args, **kwargs):
        return array([])

class MomentsFeatureManager(NullFeatureManager):
    def __init__(self, begin_idx=0, cache_begin_idx=0, nmoments=4, 
                                                            *args, **kwargs):
        super(MomentsFeatureManager, self).__init__(begin_idx, cache_begin_idx)
        self.nmoments = nmoments
        self.cache_length = nmoments+1 # we also include the 0th moment, aka n

    def __len__(self):
        return self.nmoments+1

    def compute_moment_sums(self, ar, idxs):
        values = ar.ravel()[idxs][:,newaxis]
        return (values ** arange(self.nmoments+1)).sum(axis=0)

    def create_node_cache(self, g, n):
        node_idxs = list(g.node[n]['extent'])
        return self.compute_moment_sums(g.probabilities, node_idxs)

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = list(g[n1][n2]['boundary'])
        return self.compute_moment_sums(g.probabilities, edge_idxs)

    def update_node_cache(self, g, n1, n2, dst, src):
        dst += src

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst += src

    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        dst += a * self.compute_moment_sums(g.probabilities, idxs)

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        dst += a * self.compute_moment_sums(g.probabilities, idxs)

    def compute_node_features(self, g, n):
        return central_moments_from_noncentral_sums(g.node[n]['feature-cache'])

    def compute_edge_features(self, g, n1, n2):
        return central_moments_from_noncentral_sums(g[n1][n2]['feature-cache'])

def central_moments_from_noncentral_sums(a):
    """Compute moments about the mean from sums of x**i, for i=0, ..., len(a).

    The first two moments about the mean (1 and 0) would always be 
    uninteresting so the function returns n (the sample size) and mu (the 
    sample mean) in their place.
    """
    a = a.astype(double)
    if len(a) == 1:
        return a
    N = a[0]
    a /= N
    mu = a[1]
    ac = zeros_like(a)
    for n in xrange(2,len(a)):
        js = arange(0,n+1)
        # Formula found in Wikipedia page for "Central moment", 2011-07-31
        ac[n] = (nchoosek(n,js) * (-1)**(n-js) * a[js] * mu**(n-js)).sum()
    ac[0] = N
    ac[1] = mu
    return ac

class HistogramFeatureManager(NullFeatureManager):
    def __init__(self, begin_idx=0, cache_begin_idx=0, nbins=4, 
                                        minval=0.0, maxval=1.0, *args, **kwargs):
        super(HistogramFeatureManager, self).__init__(begin_idx, 
                                                                cache_begin_idx)
        self.minval = minval
        self.maxval = maxval
        self.nbins = nbins
        self.cache_length = nbins

    def __len__(self):
        return self.nbins

    def histogram(self, vals):
        return histogram(vals, bins=self.nbins,
                            range=(self.minval,self.maxval))[0].astype(double)

    def create_node_cache(self, g, n):
        node_idxs = list(g.node[n]['extent'])
        return self.histogram(g.probabilities.ravel()[node_idxs])

    def create_edge_cache(self, g, n1, n2):
        edge_idxs = list(g[n1][n2]['boundary'])
        return self.histogram(g.probabilities.ravel()[edge_idxs])

    def update_node_cache(self, g, n1, n2, dst, src):
        dst += src

    def update_edge_cache(self, g, e1, e2, dst, src):
        dst += src

    def pixelwise_update_node_cache(self, g, n, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        dst += a * self.histogram(g.probabilities.ravel()[idxs])

    def pixelwise_update_edge_cache(self, g, n1, n2, dst, idxs, remove=False):
        if len(idxs) == 0: return
        a = -1.0 if remove else 1.0
        dst += a * self.histogram(g.probabilities.ravel()[idxs])

    def compute_node_features(self, g, n):
        h = g.node[n]['feature-cache']
        try:
            return h/h.sum()
        except ZeroDivisionError:
            return h

    def compute_edge_features(self, g, n1, n2):
        h = g[n1][n2]['feature-cache']
        try:
            return h/h.sum()
        except ZeroDivisionError:
            return h

def mean_and_sem(g, n1, n2):
    bvals = g.probabilities.ravel()[list(g[n1][n2]['boundary'])]
    return array([mean(bvals), sem(bvals)]).reshape(1,2)

def mean_sem_and_n_from_cache_dict(d):
    n, s1, s2 = d['feature-cache'][:3]
    m = s1/n
    v = 0 if n==1 else max(0, s2/(n-1) - n/(n-1)*m*m)
    s = sqrt(v/n)
    return m, s, n

def skew_from_cache_dict(d):
    n, s1, s2, s3 = d['feature-cache'][:4]
    m1 = s1/n
    k1 = m1
    m2 = s2/n
    k2 = m2 - m1*m1
    m3 = s3/n
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


def pickled(fn):
    try:
        obj = cPickle.load(open(fn, 'r'))
    except cPickle.UnpicklingError:
        obj = RandomForest()
        obj.load_from_disk(fn)
    return obj

arguments = argparse.ArgumentParser(add_help=False)
arggroup = arguments.add_argument_group('Classification options')
arggroup.add_argument('-c', '--classifier', default='ada', 
    help='''Choose the classifier to use. Default: adaboost. 
        Options: svm, logistic-regression, linear-regression,
        random-forest, adaboost'''
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
    from agglo import best_possible_segmentation, Rag, boundary_mean, \
                                classifier_probability, random_priority
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
