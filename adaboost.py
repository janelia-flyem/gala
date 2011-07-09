
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
            w = (1.0/N)*numpy.ones(N)
        self.weights = w
        self.weak_classifier_ensemble = []
        self.alpha = []
        
        for t in range(T):
            weak_learner = DecisionStump()
            weak_learner.fit(X,Y,w)
            Y_pred = weak_learner.predict(X)
            e = sum(0.5*w*abs((Y-Y_pred)))/sum(w)
            if e > 0.5:
                sys.stdout.write('WARNING: ending training due to no good weak classifiers.')
                break
            ee = (1-e)/(e*1.0)
            alpha = 0.5*math.log(ee+0.00001)
            w *= numpy.exp(-alpha*Y*Y_pred) #*ww) # increase weights for wrongly classified
            w /= sum(w)
            self.weak_classifier_ensemble.append(weak_learner)
            self.alpha.append(alpha)
            self.T = t+1
            sys.stdout.write('.')
            sys.stdout.flush()
        print "\n"

    def predict(self,X):
        X = numpy.array(X)
        N, d = X.shape
        Y = numpy.zeros(N)
        for t in range(self.T):
            #sys.stdout.write('.')
            weak_learner = self.weak_classifier_ensemble[t]
            #print Y.shape, self.alpha[t], weak_learner.predict(X).shape
            Y += self.alpha[t]*weak_learner.predict(X)
        return Y
        
    def predict_proba(self, X):
        return 1.0/(1.0 + numpy.exp(-self.predict(X)))
        

    def measure_accuracy(self, Y, o, threshold=0):
        oo = o.copy()
        oo[numpy.where(o>threshold)[0]] = 1
        oo[numpy.where(o<threshold)[0]] = -1
        d = (oo - Y)
        return len(d[numpy.where(d==0)[0]])*1.0/len(Y)





