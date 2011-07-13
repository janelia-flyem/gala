
from decision_stump import DecisionStump

import sys, math, random
import numpy
import operator

class AdaBoost(object):
    """ Class for an adaboost classifier, adapted from pyclassic. """

    def fit(self, X, Y, w=None, T=100):
        self.X = X
        self.Y = Y
        N = len(self.Y)
        
        if w is None:
            w = (1.0/float(N))*numpy.ones(N)
        self.weights = w
        self.weak_classifier_ensemble = []
        self.alpha = []
        
        for t in range(T):
            weak_learner = DecisionStump().fit(X,Y,w)
            Y_pred = weak_learner.predict(X)
            e = sum(0.5*w*abs((Y-Y_pred)))/sum(w)
            if e > 0.5:
                sys.stdout.write('WARNING: ending training due to no good weak classifiers.')
                break
            ee = (1-e)/float(e)
            alpha = 0.5*math.log(ee)
            w *= numpy.exp(-alpha*Y*Y_pred) # increase weights for wrongly classified
            w /= sum(w)
            self.weak_classifier_ensemble.append(weak_learner)
            self.alpha.append(alpha)
            self.T = t+1
            sys.stdout.write('.')
            sys.stdout.flush()
        print "\n"
        return self

    def predict(self,X):
        X = numpy.array(X)
        Y = sum([a*c.predict(X) for a,c in zip(self.alpha, self.weak_classifier_ensemble)])
        return Y
        
    def predict_proba(self, X):
        prob = 1.0/(1.0 + numpy.exp(-self.predict(X)))
        return numpy.array([[1.0-p, p] for p in prob])
        

def measure_accuracy(Y, o, threshold=0):
    oo = o.copy()
    oo[numpy.where(o>threshold)[0]] = 1
    oo[numpy.where(o<threshold)[0]] = -1
    d = (oo - Y)
    return len(d[numpy.where(d==0)[0]])/float(len(Y))





