
import math
import numpy
import operator


class DecisionStump():
    """ Class for a decision stump, adapted from pyclassic. """

    def fit(self, X, Y, w):
        feature_index, stump = train_decision_stump(X,Y,w)
        self.feature_index = feature_index
        self.stump = stump
        return self	

    def predict(self,X):
        if len(X.shape)==1:
            X = numpy.array([X])
        N, d = X.shape
        feature_index = self.feature_index
        threshold = self.stump.threshold
        s = self.stump.s
        return s*(2.0*(X[:,feature_index]>threshold).astype(numpy.uint8)-1)

class Stump:
    """1D stump"""
    def __init__(self, score, threshold, s):
        self.score = score
        self.threshold = threshold
        self.s = s

    def __cmp__(self, other):
        return cmp(self.err, other.err)


def train_decision_stump(X,Y,w):
    stumps = [build_stump_1d(x,Y,w) for x in X.T]
    feature_index = numpy.argmax([s.score for s in stumps])
    best_stump = stumps[feature_index]
    best_threshold = best_stump.threshold
    return feature_index, best_stump


def build_stump_1d(x,y,w):
    idx = x.argsort()
    xsorted = x[idx]
    wy = y[idx]*w[idx]
    wy_pos = numpy.clip(wy, a_min=0, a_max=numpy.inf)
    wy_neg = numpy.clip(wy, a_min=-numpy.inf, a_max=0)
    score_left_pos = numpy.cumsum(wy_pos)
    score_right_pos = numpy.cumsum(wy_pos[::-1])
    score_left_neg = numpy.cumsum(wy_neg)
    score_right_neg = numpy.cumsum(wy_neg[::-1])
    
    score1 = -score_left_pos[0:-1:1] + score_right_neg[-2::-1]
    score2 = -score_left_neg[0:-1:1] + score_right_pos[-2::-1]
    # using idx will ensure that we don't split between nodes with identical x values
    idx = numpy.nonzero((xsorted[:-1] < xsorted[1:]).astype(numpy.uint8))[0]
    if len(idx)==0:
        return Stump(-numpy.inf, 0, 0)

    score = numpy.where(abs(score1)>abs(score2), score1, score2)
    ind = idx[numpy.argmax(abs(score[idx]))]
    maxscore = abs(score[ind])
    threshold = (xsorted[ind] + xsorted[ind+1])/2.0
    s = numpy.sign(score[ind]) # direction of -1 -> 1 change
    return Stump(maxscore, threshold, s)

