# system modules
import sys, math, random, logging
import operator

# libraries
import numpy

# local modules
from decision_tree import DecisionTree
from iterprogress import with_progress, NoProgressBar, StandardProgressBar

class AdaBoost(object):
    """Class for an adaboost classifier, adapted from pyclassic. """
    def __init__(self, progress=False, **kwargs):
        if progress:
            self.progressbar = StandardProgressBar('AdaBoost training...')
        else:
            self.progressbar = NoProgressBar()

    def fit(self, X, Y, w=None, w_asymmetric=None, depth=1, T=100, **kwargs):
        self.X = X.copy()
        self.Y = Y.copy()
        N = len(self.Y)
        
        if w is None:
            w = (1.0/float(N))*numpy.ones(N)
        if w_asymmetric is None:
            w_asymmetric = (1.0/float(N))*numpy.ones(N)
        self.weights = w.copy()
        self.weights_asymmetric = numpy.array([i**(1.0/float(T)) 
                                                        for i in w_asymmetric])
        self.weights /= float(sum(self.weights))
        self.weak_classifier_ensemble = []
        self.alpha = []
        
        for t in with_progress(range(T), pbar=self.progressbar):
            # Apply asymmetric weights
            self.weights *= self.weights_asymmetric
            weak_learner = DecisionTree().fit(self.X,self.Y,self.weights, depth=depth)
            Y_pred = weak_learner.predict(self.X)
            e = sum(0.5*self.weights*abs(self.Y-Y_pred))/sum(self.weights)
            if e > 0.5:
                logging.warning(' ending training, no good weak classifiers.')
                break
            ee = (1.0-e)/float(e)
            alpha = 0.5*math.log(ee)
            # increase weights for wrongly classified:
            self.weights *= numpy.exp(-alpha*self.Y*Y_pred)
            self.weights /= sum(self.weights)
            self.weak_classifier_ensemble.append(weak_learner)
            self.alpha.append(alpha)
        return self

    def predict_score(self,X):
        X = numpy.array(X)
        Y = sum([alpha * weak_classifier.predict(X) for alpha, weak_classifier
                            in zip(self.alpha, self.weak_classifier_ensemble)])
        return Y
        
    def predict_proba(self, X):
        p = 1.0/(1.0 + numpy.exp(-2.0*self.predict_score(X)))
        return numpy.concatenate((numpy.array([1.0-p]), numpy.array([p])), axis=0).T

def measure_accuracy(Y, o, threshold=0):
    oo = o.copy()
    oo[numpy.where(o>threshold)[0]] = 1
    oo[numpy.where(o<threshold)[0]] = -1
    d = (oo - Y)
    return len(d[numpy.where(d==0)[0]])/float(len(Y))





