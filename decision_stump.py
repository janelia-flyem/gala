
import math
import numpy
import operator


class DecisionStump():
    """ Class for a decision stump, adapted from pyclassic. """

    def fit(self, X, Y, w):
        self.X = X
        self.Y = Y
        self.weights = w

        feature_index, stump = train_decision_stump(X,Y,w)
        self.feature_index = feature_index
        self.stump = stump
	return self	

    def predict(self,X):
        N, d = X.shape
        feature_index = self.feature_index
        threshold = self.stump.threshold
        s = self.stump.s

        Y = numpy.ones(N)
        Y[X[:,feature_index]<threshold] = -1
        return s*Y


class Stump:
    """1D stump"""
    def __init__(self, err, threshold, s):
        self.err = err
        self.threshold = threshold
        self.s = s

    def __cmp__(self, other):
        return cmp(self.err, other.err)


def train_decision_stump(X,Y,w):
    stumps = [build_stump_1d(x,Y,w) for x in X.T]
    feature_index, best_stump = min(enumerate(stumps), key=operator.itemgetter(1))
    best_threshold = best_stump.threshold
    return feature_index, best_stump


def build_stump_1d(x,y,w):
    idx = x.argsort()
    xsorted = x[idx]
    wy = y[idx]*w[idx]
    score_left = numpy.cumsum(wy)
    score_right = numpy.cumsum(wy[::-1])
    score = -score_left[0:-1:1] + score_right[-1:0:-1]
    Idec = numpy.where(xsorted[:-1]<xsorted[1:])[0]
    if len(Idec)>0:  # determine the boundary
        ind = Idec[numpy.argmax(abs(score[Idec]))]
	maxscore = abs(score[ind])
	err = 0.5-0.5*maxscore # compute weighted error
        threshold = (xsorted[ind] + xsorted[ind+1])/2 # threshold
        s = numpy.sign(score[ind]) # direction of -1 -> 1 change
    else:  # all identical; todo: add random noise?
        err = 0.5
        threshold = 0
        s = 1
    return Stump(err, threshold, s)

