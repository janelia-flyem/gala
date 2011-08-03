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
    uint8, ones, float32, uint32, unique, newaxis, zeros_like, arange
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
from agglo import best_possible_segmentation, Rag, boundary_mean, \
    classifier_probability, random_priority
import morpho
import iterprogress as ip
from imio import read_h5_stack, write_h5_stack, write_image_stack
from adaboost import AdaBoost

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

class GraphFeatures(object):
    """Null class for graph features. Base of a Composite pattern."""
    def __init__(self, node_cache_begin_idx=0, edge_cache_begin_idx=0):
        self.node_cache_begin_idx = node_cache_begin_idx
        self.edge_cache_begin_idx = edge_cache_begin_idx
        self.parent = None

    def __call__(self, g, n1, n2=None):
        if n2 is None:
            return self.compute_node_features(g, n1)
        if len(g.node[n1]['extent']) > len(g.node[n2]['extent']):
            n1, n2 = n2, n1
        return self.compute_face_features(g, n1, n2)

    def __len__(self):
        return 0

    def cache_length(self):
        return 0

    def node_cache_address(self):
        return self.node_cache_begin_idx, \
                                self.node_cache_begin_idx+self.cache_length()

    def edge_cache_address(self):
        return self.edge_cache_begin_idx, \
                                self.edge_cache_begin_idx+self.cache_length()

    def compute_face_features(self, g, n1, n2):
        return concatenate(
            self.compute_edge_features(g, n1, n2),
            self.compute_node_features(g, n1),
            self.compute_node_features(g, n2)
        )

    def compute_edge_features(self, g, n1, n2):
        i, j = self.edge_cache_address()
        try:
            cache = g[n1][n2]['feature-cache'][i:j]
        except KeyError:
            cache = array([])
        return compute_edge_features_from_cache(self, g, n1, n2, cache)

    def compute_edge_features_from_cache(self, g, n1, n2, cache):
        return cache

    def compute_node_features(self, g, n1):
        i, j = self.node_cache_address()
        if not g.node[n1].haskey('feature-cache'):
            self.init_node_cache(g, n1)
        cache = g.node[n1]['feature-cache'][i:j]
        return compute_node_features_from_cache(self, g, n1, cache)

    def compute_node_features_from_cache(self, g, n1, cache):
        return cache

    def init_node_cache(self, g, n1, idx):
        pass

    def init_edge_cache(self, g, n1, n2, idx):
        pass

    def merge_node_caches(self, g, n1, n2):
        pass

    def merge_edge_caches(self, g, e1, e2):
        pass

class GraphHistogramFeatures(GraphFeatures):
    def __init__(self, node_cache_begin_idx=0, edge_cache_begin_idx=0,
                                                minp=0.0, maxp=1.0, nbins=20):
        super(GraphHistogramFeatures, self).__init__(
                                node_cache_begin_idx, edge_cache_begin_idx)
        self.minp = minp
        self.maxp = maxp
        self.nbins = nbins
        self.length = nbins

    def __len__(self):
        return self.nbins

    def root_length(self):
        if self.parent is None:
            return self.nbins
        else:
            return self.parent.root_length()

    def cache_length(self):
        return self.nbins

    def root_cache_length(self):
        if self.parent is None:
            return self.nbins
        else:
            return self.parent.root_cache_length()

    def compute_edge_features_from_cache(self, g, n1, n2, cache):
        return cache/len(g[n1][n2]['extent'])

    def compute_node_features_from_cache(self, g, n1, cache):
        return cache/len(g.node[n1]['extent'])

    def init_node_cache(self, g, n1, idx):
        p = g.probabilities.ravel()[idx]
        i, j = self.node_cache_address()
        bin_idx = floor((p-self.minp)/self.maxp * self.nbins)
        if not g.node[n1].haskey('feature-cache'):
            g.node[n1]['feature-cache'] = zeros(self.full_cache_length())
        g.node[n1]['feature-cache'][i:j][bin_idx] += 1

    def init_edge_cache(self, g, n1, n2, idx):
        p = g.probabilities.ravel()[idx]
        i, j = self.edge_cache_address()
        bin_idx = floor((p-self.minp)/self.maxp * self.nbins)
        if not g[n1][n2].haskey('feature-cache'):
            g[n1][n2]['feature-cache'] = zeros(self.root_cache_length())
        g[n1][n2]['feature-cache'][i:j][bin_idx] += 1
    
    def merge_node_caches(self, g, n1, n2):
        i, j = self.node_cache_address()
        g.node[n1]['feature-cache'][i:j] += g.node[n2]['feature-cache'][i:j]
        if g.has_boundaries:
            k, l = self.edge_cache_address()
            g.node[n1]['feature-cache'][i:j] += g[n1][n2]['feature-cache'][k:l]

    def merge_edge_caches(self, g, e1, e2):
        u, v = e1
        w, x = e2
        i, j = self.edge_cache_address()
        g[u][v]['feature-cache'][i:j] += g[w][x]['feature-cache'][i:j]

class GraphMomentsFeatures(object):
    """An attempt """
    def __init__(self, node_cache_begin_idx=0, edge_cache_begin_idx=0, 
                                                                nmoments=4):
        super(GraphMomentFeatures, self).__init__(
                                node_cache_begin_idx, edge_cache_begin_idx)
        self.nmoments = nmoments
    
    def __len__(self):
        return self.nmoments+1

    def compute_edge_features(self, 

    
    def auxdata_init_node(self,g,n1,vxl):
        # n1, n2 = idx of existing edge or node
        # vxl = idx of boundary or interior voxel
        memberset = set([vxl])
    
        pval=g.probabilities.ravel()[vxl]
        if pval >1:
            return
        #pdb.set_trace()
        # moments
        qtyvec = (pval*ones(self.nmoments+1))**arange(self.nmoments+1)
        #histograms
        qtyvec = concatenate((qtyvec,zeros(self.nhistbins)))
    
        bin= min(int(floor(pval/self.ival)),self.nhistbins-1)
        qtyvec[self.nmoments+1+bin] += 1
    
        g.add_node(n1, extent=memberset, cachevec= qtyvec)
      
    def auxdata_init_edge(self,g,n1,n2,vxl):
        # n1, n2 = idx of existing edge or node
        # vxl = idx of boundary or interior voxel
        memberset = set([vxl])
    
        pval=g.probabilities.ravel()[vxl]
        if pval >1:
            return

        #pdb.set_trace()
        # moments
        qtyvec = (pval*ones(self.nmoments+1))**arange(self.nmoments+1)
        #histograms
        qtyvec = concatenate((qtyvec,zeros(self.nhistbins)))
    
        bin= min(int(floor(pval/self.ival)),self.nhistbins-1)
        if bin>= self.nhistbins:
            pdb.set_trace()
            qtyvec[self.nmoments+1+bin] += 1
    
        g.add_edge(n1,n2, extent = memberset, cachevec = qtyvec)
      
    def auxdata_update_node(self,g,n1,vxl):
        # n1, n2 = idx of existing edge or node
        # vxl = idx of boundary or interior voxel

        pval = g.probabilities.ravel()[vxl]
        if pval >1:
            return

        memberset = g.node[n1]['extent']
        #pdb.set_trace()
    
        memberset.add(vxl)
    
    
        qtyvec = g.node[n1]['cachevec'];

    
        qtyvec[range(self.nmoments+1)] += (pval*ones(self.nmoments+1))**arange(self.nmoments+1)
    
        # computing histogram
        bin= min(int(floor(pval/self.ival)),self.nhistbins-1)
        qtyvec[self.nmoments+1+bin] += 1
    
    def auxdata_update_edge(self,g,n1,n2,vxl):
        # n1, n2 = idx of existing edge or node
        # vxl = idx of boundary or interior voxel
        pval = g.probabilities.ravel()[vxl]
        if pval >1:
            return

        memberset = g[n1][n2]['extent']
        #pdb.set_trace()
    
        memberset.add(vxl)
    
    
        qtyvec = g[n1][n2]['cachevec'];

        qtyvec[range(self.nmoments+1)] += (pval*ones(self.nmoments+1))**arange(self.nmoments+1)
    
        # computing histogram
        bin= min(int(floor(pval/self.ival)),self.nhistbins-1)
        qtyvec[self.nmoments+1+bin] += 1
      
    def auxdata_merge_nodes(self,g,n1,n2):
      
        #pdb.set_trace()
    
        memberset = g.node[n1]['extent'].union( g.node[n2]['extent'] )
    
        qtyvec1 = g.node[n1]['cachevec'];
        qtyvec2 = g.node[n1]['cachevec'];

        qtyvec = qtyvec1 + qtyvec2
    
        g.node[n1]['extent'] = memberset
        g.node[n1]['cachevec'] = qtyvec
    

        nbr_n1= g.neighbors(n1)
        nbr_n2= g.neighbors(n2)
    
        allnbrs= setdiff1d(union1d(nbr_n1,nbr_n2),set([n1,n2]))
        commonnbrs= intersect1d(nbr_n1,nbr_n2)
      
      
        for nn in allnbrs:
            if nn in commonnbrs:
            memberset=g[n1][nn]['extent']
            features=g[n1][nn]['cachevec']
            if g[n1][nn].has_key('qlink'):
                g.merge_queue.invalidate(g[n1][nn]['qlink'])
            g.remove_edge(n1,nn)          
        
            g.add_edge(n1,nn, extent=  memberset.union(g[n2][nn]['extent']),  cachevec = features + g[n2][nn]['cachevec'])
            g.update_merge_queue(n1,nn)
        
            if g[n2][nn].has_key('qlink'):
                g.merge_queue.invalidate(g[n2][nn]['qlink'])
            g.remove_edge(n2,nn)          
        
        elif g.has_edge(n2,nn):
        g.add_edge(n1,nn, extent=g[n2][nn]['extent'],cachevec=g[n2][nn]['cachevec']) 
        g.update_merge_queue(n1,nn)
        
        if g[n2][nn].has_key('qlink'):
            g.merge_queue.invalidate(g[n2][nn]['qlink'])
        g.remove_edge(n2,nn)
      #else if g.has_edge(nn,n2):
        #g.add_edge(n1,nn,members=g[nn][n2]['members'],features=g[nn][n2]['cachevec']) 
        #g.remove_edge(nn,n2)
    
    if g[n1][n2].has_key('qlink'):
        g.merge_queue.invalidate(g[n1][n2]['qlink'])
    g.remove_edge(n1,n2)
    
    g.remove_node(n2)
      
      


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
