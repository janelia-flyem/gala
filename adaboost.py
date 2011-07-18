
from decision_stump import DecisionStump

import sys, math, random
import numpy
import operator

class AdaBoost(object):
    """ Class for an adaboost classifier, adapted from pyclassic. """

    def fit(self, X, Y, w=None, w_asymmetric=None, T=100):
        self.X = X.copy()
        self.Y = Y.copy()
        N = len(self.Y)
        
        if w is None:
            w = (1.0/float(N))*numpy.ones(N)
	if w_asymmetric is None:
	    w_asymmetric = (1.0/float(N))*numpy.ones(N)
        self.weights = w.copy()
	self.weights_asymmetric = numpy.array([i**(1.0/float(T)) for i in w_asymmetric])
	self.weights /= float(sum(self.weights))
        self.weak_classifier_ensemble = []
        self.alpha = []
        
        for t in range(T):
	    # Apply asymmetric weights
	    self.weights *= self.weights_asymmetric
            weak_learner = DecisionStump().fit(self.X,self.Y,self.weights)
            Y_pred = weak_learner.predict(self.X)
            e = sum(0.5*self.weights*abs(self.Y-Y_pred))/sum(self.weights)
	    if e > 0.5:
		sys.stdout.write('WARNING: ending training due to no good weak classifiers.')
                break
            ee = (1.0-e)/float(e)
            alpha = 0.5*math.log(ee)
            self.weights *= numpy.exp(-alpha*self.Y*Y_pred) # increase weights for wrongly classified
            self.weights /= sum(self.weights)
            self.weak_classifier_ensemble.append(weak_learner)
            self.alpha.append(alpha)
            self.T = t+1
            sys.stdout.write('.')
            sys.stdout.flush()
        print "\n"
        return self

    def predict(self,X):
        X = numpy.array(X)
        Y = sum([self.alpha[i]*self.weak_classifier_ensemble[i].predict(X) for i in xrange(len(self.alpha))])
        return Y
        
    def predict_proba(self, X):
        prob = 1.0/(1.0 + numpy.exp(-self.predict(X)))
        return numpy.concatenate((numpy.array([1.0-prob]), numpy.array([prob]))).T
        

def measure_accuracy(Y, o, threshold=0):
    oo = o.copy()
    oo[numpy.where(o>threshold)[0]] = 1
    oo[numpy.where(o<threshold)[0]] = -1
    d = (oo - Y)
    return len(d[numpy.where(d==0)[0]])/float(len(Y))





