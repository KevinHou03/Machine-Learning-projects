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
    freq = []
    freq_dict = {}
    for label in y:
        if label not in freq_dict.keys():
            freq_dict[label] = 0
        freq_dict[label] += 1
    for element in freq_dict:
        freq.append(freq_dict[element])
    freq = np.array(freq)
    possibility = freq / len(y)
    if criterion == "entropy" or criterion == None:
        score = -np.sum([p * np.log2(p) for p in possibility if p > 0])
    elif criterion == "gini":
        score = 1 - np.sum([p ** 2 for p in possibility])
    else:
        print("Not supported\n")
    return score


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        # print()

    def is_leaf_node(self):
        return self.value is not None


def most_common_label(y):
    """
    """
    counter = Counter(y)

    most = counter.most_common(1)[0][0]
    return most


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
        if not self.n_feats:
            self.n_feats = xFeat.shape[1]
        else:
            self.n_feats = min(self.n_feats, xFeat.shape[1])
        self.root = self.grow_tree(xFeat, y)
        return self

    def grow_tree(self, xFeat, y, depth=0):
        samples = xFeat.shape[0]
        features = xFeat.shape[1]
        labels = len(np.unique(y))

        # stop when
        if depth >= self.maxDepth \
                or labels <= 1 \
                or samples < self.minLeafSample:
            Class = most_common_label(y)
            return Node(value=Class)

        feat_index = np.arange(self.n_feats)  # 01234..
        # choose the best feature based on info gain
        best_feat, best_threshold = self.choose_best_feature(xFeat, y, feat_index)

        # recursively build the tree based on the splitting result.;
        left_index, right_index = self.split(xFeat[:, best_feat], best_threshold)
        left = self.grow_tree(xFeat[left_index, :], y[left_index], depth + 1)
        right = self.grow_tree(xFeat[right_index, :], y[right_index], depth + 1)
        return Node(best_feat, best_threshold, left, right)

    def choose_best_feature(self, xFeat, y, feat_indexs):  # finding the best feature and threshold combination
        initial_gain = 0
        best_gain = -1
        split_index = 0
        split_threshold = 0
        for feat_index in feat_indexs:
            thresholds = xFeat[:, feat_index]  # 11222333
            # print(X_column)
            unique_thresholds = set(thresholds)  # 123
            # print(unique_thresholds)
            for threshold in unique_thresholds:  # split according to every threshold
                info_gain = self.information_gain(y, thresholds, threshold)
                if info_gain > best_gain:
                    # 没有大于他就没必要更新
                    best_gain = info_gain
                    split_index = feat_index
                    split_threshold = threshold
        return split_index, split_threshold

    def information_gain(self, y, thresholds, split_threshold):
        # base loss
        base_score = calculate_split_score(y, self.criterion)

        # split
        left, right = self.split(thresholds, split_threshold)
        # high purity
        if len(left) == 0 or len(right) == 0:
            return 0
        # compute the weighted average
        n = len(y)
        n_left = len(left)
        n_right = len(right)

        left_probability = n_left / n
        right_probablity = n_right / n
        e_left = calculate_split_score(y[left], self.criterion)
        e_right = calculate_split_score(y[right], self.criterion)
        score = (left_probability * e_left) + (right_probablity * e_right)

        # information gain varies pre and post the split
        info_gain1 = 0
        new_score = score
        info_gain1 = np.abs(base_score - score)
        return info_gain1

    def split(self, thresholds, split_threshold):
        left_index = []
        right_index = []

        for i, value in enumerate(thresholds):
            if value <= split_threshold:
                left_index.append(i)
            else:
                right_index.append(i)

        return left_index, right_index

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
        N = xFeat.shape[0]
        # print(type(yHat))
        yHat = np.ones(N)  # Initialize an empty array to store predictions
        for i in range(0, N):
            yHat[i] = self.traverse_tree(xFeat[i], self.root)

        yHat = np.array(yHat)
        return yHat

    def traverse_tree(self, sample, node):
        if node.is_leaf_node():  # reach the buttom return the value
            return node.value
        if sample[node.feature] <= node.threshold:  # classify if smaller goes left
            return self.traverse_tree(sample, node.left)
        return self.traverse_tree(sample, node.right)  # if bigger goes right


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

            '''
            GINI Criterion ---------------
Training Acc: 0.974
Test Acc: 0.89
Entropy Criterion ---------------
Training Acc: 0.973
Test Acc: 0.887

            '''

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
