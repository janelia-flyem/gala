from numpy import inf, unique, array, zeros
from decision_stump import DecisionStump

class DecisionTree():
    """ Class for a decision tree.
        The trees are grown until completion or up to a specified maximum depth.
        The splits are based on the imlementation of DecisionStump, which currently
            splits on weighted classification error.
    """    
    def fit(self, X, Y, w, depth=inf, curr_depth=0, curr_node=None):

        self.head = self.build_tree(X.copy(),Y.copy(),w.copy(),depth,curr_depth)
        self.weights = w.copy()
        return self

    def build_tree(self, X, Y, w, depth, curr_depth):
        # See if we can do any splitting at all
        tree = Node()
        yw = Y*w
        if len(X)<2 or len(unique(Y)) < 2 or curr_depth >= depth:
            tree.stump = 1.0 if abs(sum(yw[yw>=0]))>abs(sum(yw[yw<0])) else -1.0
            return tree
        # TODO: check for inconsistent data

        # Learn the decision stump
        stump = DecisionStump().fit(X,Y,w)
        side1 = stump.predict(X)>=0
        side2 = stump.predict(X)<0

        tree.stump = stump
        tree.left = self.build_tree(X[side1], Y[side1], w[side1], depth, curr_depth+1)
        tree.right = self.build_tree(X[side2], Y[side2], w[side2], depth, curr_depth+1)
        
        return tree

    def predict(self, X, curr_node=None):
        if len(X.shape)==1:
            X = array([X])
        
        if curr_node is None:
            curr_node = self.head

        pred = zeros(len(X))
        if not isinstance(curr_node.stump, DecisionStump):
            return curr_node.stump

        side1 = curr_node.stump.predict(X)>=0
        side2 = curr_node.stump.predict(X)<0
        
        if sum(side1)>0:
            pred[side1] = self.predict(X[side1], curr_node.left)
        if sum(side2)>0:
            pred[side2] = self.predict(X[side2], curr_node.right)

        return pred


class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.stump = None
