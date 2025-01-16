'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Kaiyuan Hou
'''
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import log


def calculate_split_score(y, criterion):
    """
    Given a numpy array of labels associated with a node, y, 
    calculate the score based on the crieterion specified.

    Parameters
    ----------
    y : numpy.1d array with shape n
        Array of labels associated with a node
    criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
    Returns
    -------
    score : float
        The gini or entropy associated with a node
    """
    '''
    label_number = len(y)
    label_count = {}
    for sample_class in y:
        if sample_class not in label_count.keys():
            label_count[sample_class] = 0
        label_count[sample_class] += 1

    entropy = 0
    for keys in label_count:
        probability = float(label_count[keys] / label_number)
        entropy -= probability * log(probability, 2)
    return entropy
    '''
    hist = np.bincount(y)
    ps = hist / len(y)
    if criterion == "entropy" or criterion == None:
        score = -np.sum([p * np.log2(p) for p in ps if p > 0])
    elif criterion == "gini":
        score = 1 - np.sum([p ** 2 for p in ps])
    else:
        print("Not supported\n")
    return score


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        # print

    def is_leaf_node(self):
        return self.value is not None


def most_common_label(y):
    counter = Counter(y)
    # print(most_common = counter.most_common(1)[0][0])
    most_common = counter.most_common(1)[0][0]
    return most_common


class DecisionTree(object):
    maxDepth = 0  # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of sampless in a leaf
    criterion = None  # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        self.n_feats = None
        self.root = None

    def train(self, xFeat, y):
        """
        Train the decision tree model.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape n x d
            Training data 
        y : numpy.1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        self.n_feats = xFeat.shape[1] if not self.n_feats else min(self.n_feats, xFeat.shape[1])
        self.root = self.grow_tree(xFeat, y)
        return self

    def grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if depth >= self.maxDepth or n_labels == 1 or n_samples < self.minLeafSample:
            leaf_value = most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.arange(self.n_feats)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self.best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self.split(X[:, best_feat], best_thresh)
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            # print(X_column)
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_score = calculate_split_score(y, self.criterion)

        # generate split
        left_idxs, right_idxs = self.split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l = len(left_idxs)
        n_r = len(right_idxs)

        e_l = calculate_split_score(y[left_idxs], self.criterion)
        e_r = calculate_split_score(y[right_idxs], self.criterion)
        child_score = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_score - child_score
        return ig

    '''
    def split(self,dataset, i, val):
    ret_data_set = []
    for feat_vector in dataset:
        if feat_vector[i] == val:
            reduced_feat_vector = np.concatenate((feat_vector[:i], feat_vector[i + 1:]))
            ret_data_set.append(reduced_feat_vector)
    return np.array(ret_data_set)
    '''

    def split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : numpy.nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : numpy.1d array with shape m
            Predicted class label per sample
        """
        yHat = np.array([])  # variable to store the estimated class label
        # print(type(yHat))
        if xFeat.ndim == 1:
            xFeat = np.expand_dims(xFeat, axis=0)

        yHat = np.array([self.traverse_tree(x, self.root) for x in xFeat])
        return yHat

    def traverse_tree(self, x, node):
        if node.is_leaf_node():  # to the button
            return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : numpy.nd-array with shape n x d
        Training data 
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data.
    xTest : numpy.nd-array with shape m x d
        Test data 
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain)
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain, yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest, yHatTest)
    # print(testAcc)
    # print(yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3xTrain.csv').to_numpy()
    yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3yTrain.csv').to_numpy().flatten()
    xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3xTest.csv').to_numpy()
    yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3yTest.csv').to_numpy().flatten()

    # print(xTrain,xTest)
    # print(yTrain,yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    maxDepth = 15
    minleaf = 15
    trainAccs = np.zeros((maxDepth - 1, minleaf - 3))
    testAccs = np.zeros((maxDepth - 1, minleaf - 3))

    for i in range(1, maxDepth):
        for j in range(3, minleaf):
            model_dt = DecisionTree('gini', i, j)
            trainAccs[i - 1, j - 3], testAccs[i - 1, j - 3] = dt_train_test(model_dt, xTrain, yTrain, xTest, yTest)

    # print(trainAccs)
    # print(testAccs)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    X = np.arange(1, maxDepth, 1)
    Y = np.arange(3, minleaf, 1)
    Y, X = np.meshgrid(Y, X)

    ax.plot_surface(X, Y, trainAccs)
    plt.xlabel(u"depth")
    plt.ylabel("samples")
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = fig.gca(projection='3d')
    # ax = fig.gca(projection='')
    X = np.arange(1, maxDepth, 1)
    Y = np.arange(3, minleaf, 1)
    Y, X = np.meshgrid(Y, X)

    ax.plot_surface(X, Y, testAccs)
    plt.xlabel(u"depth")
    plt.ylabel("samples")
    plt.show()


if __name__ == "__main__":
    main()
