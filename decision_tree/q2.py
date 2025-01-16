'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Kaiyuan Hou
'''
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import time


def holdout(model, xFeat, y, testSize):
    """
    Split xFeat into random train and test based on the testSize and
    return the model performance on the training and test set.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset
    y : numpy.1d-array with shape n
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout.

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start_time = time.time()

    trainAuc = 0
    testAuc = 0
    time_elapsed = 0
    # TODO fill int

    xTrain, xTest, yTrain, yTest = train_test_split(
        xFeat, y, test_size=testSize, shuffle=True, random_state=1)
    # xTrain, xTest, yTrain, yTest = train_test_split(
    #         xFeat, y, test_size= testSize, shuffle=False, random_state=0)
    trainAuc, testAuc = sktree_train_test(model, xTrain, yTrain, xTest, yTest)
    end_time = time.time()

    time_elapsed = end_time - start_time

    return trainAuc, testAuc, time_elapsed


def kfold_cv(model, xFeat, y, k):
    """
    Split xFeat into k different groups, and then use each of the
    k-folds as a validation set, with the model fitting on the remaining
    k-1 folds. Return the model performance on the training and
    validation (test) set.


    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset
    y : numpy.1d-array with shape n
        Labels of the dataset
    k : int
        Number of folds or groups (approximately equal size)

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start_time = time.time()

    trainAuc = 0
    testAuc = 0
    time_elapsed = 0
    # TODO FILL IN

    ktrainAuc = np.zeros(k)
    ktestAuc = np.zeros(k)
    # print(ktrainAuc)
    # print(ktestAuc)
    for m in range(k):
        Xtrain = []
        Xtest = []
        for i in range(0, len(xFeat)):
            if i % k == m:
                Xtrain.append(i)
            else:
                Xtest.append(i)

        xTrain = xFeat[Xtrain, :]
        yTrain = y[Xtrain]
        xTest = xFeat[Xtest, :]
        yTest = y[Xtest]
        ktrainAuc[m], ktestAuc[m] = sktree_train_test(model, xTrain, yTrain, xTest, yTest)
    trainAuc = np.mean(ktrainAuc)
    testAuc = np.mean(ktestAuc)
    end_time = time.time()
    time_elapsed = end_time - start_time

    return trainAuc, testAuc, time_elapsed


def mc_cv(model, xFeat, y, testSize, s):
    """
    Evaluate the model using s samples from the
    Monte Carlo cross validation approach where
    for each sample you split xFeat into
    random train and test based on the testSize.
    Returns the model performance on the training and
    test datasets.

    Parameters
    ----------
    model : sktree.DecisionTreeClassifier
        Decision tree model
    xFeat : numpy.nd-array with shape n x d
        Features of the dataset
    y : numpy.1d-array with shape n
        Labels of the dataset
    testSize : float
        Portion of the dataset to serve as a holdout.

    Returns
    -------
    trainAuc : float
        Average AUC of the model on the training dataset
    testAuc : float
        Average AUC of the model on the validation dataset
    timeElapsed: float
        Time it took to run this function
    """
    start_time = time.time()

    trainAuc = 0
    testAuc = 0
    time_elapsed = 0
    # TODO fill int

    ktrainAuc = np.zeros(s)
    ktestAuc = np.zeros(s)
    for m in range(s):
        xTrain, xTest, yTrain, yTest = train_test_split(
            xFeat, y, test_size=testSize)
        ktrainAuc[m], ktestAuc[m] = sktree_train_test(model, xTrain, yTrain, xTest, yTest)

    trainAuc = np.mean(ktrainAuc)
    testAuc = np.mean(ktestAuc)
    end_time = time.time()
    time_elapsed = end_time - start_time

    return trainAuc, testAuc, time_elapsed


def sktree_train_test(model, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn tree model, train the model using
    the training dataset, and evaluate the model on the
    test dataset.

    Parameters
    ----------
    model : DecisionTreeClassifier object
        An instance of the decision tree classifier
    xTrain : numpy.nd-array with shape nxd
        Training data
    yTrain : numpy.1d array with shape n
        Array of labels associated with training data
    xTest : numpy.nd-array with shape mxd
        Test data
    yTest : numpy.1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    trainAUC : float
        The AUC of the model evaluated on the training data.
    testAuc : float
        The AUC of the model evaluated on the test data.
    """
    # fit the data to the training datasets
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain,
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                             yHatTest[:, 1])
    testAuc = metrics.auc(fpr, tpr)
    return trainAuc, testAuc


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
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
    # create the decision tree classifier
    dtClass = DecisionTreeClassifier(max_depth=15,
                                     min_samples_leaf=10)
    # use the holdout set with a validation size of 30 of training
    aucTrain1, aucVal1, time1 = holdout(dtClass, xTrain, yTrain, 0.30)
    # use 2-fold validation
    aucTrain2, aucVal2, time2 = kfold_cv(dtClass, xTrain, yTrain, 2)
    # use 5-fold validation
    aucTrain3, aucVal3, time3 = kfold_cv(dtClass, xTrain, yTrain, 5)
    # use 10-fold validation
    aucTrain4, aucVal4, time4 = kfold_cv(dtClass, xTrain, yTrain, 10)
    # use MCCV with 5 samples
    aucTrain5, aucVal5, time5 = mc_cv(dtClass, xTrain, yTrain, 0.30, 5)
    # use MCCV with 10 samples
    aucTrain6, aucVal6, time6 = mc_cv(dtClass, xTrain, yTrain, 0.30, 10)
    # train it using all the data and assess the true value
    '''
    0     Holdout  0.989804  0.965376  0.015013
1      2-fold  0.989192  0.967892  0.010542
2      5-fold  0.990060  0.960444  0.028093
3     10-fold  0.986060  0.957465  0.035642
4   MCCV w/ 5  0.988759  0.964623  0.020651
5  MCCV w/ 10  0.988014  0.973935  0.043935
6   True Test  0.988766  0.963088  0.000000
    '''
    trainAuc, testAuc = sktree_train_test(dtClass, xTrain, yTrain, xTest, yTest)
    perfDF = pd.DataFrame([['Holdout', aucTrain1, aucVal1, time1],
                           ['2-fold', aucTrain2, aucVal2, time2],
                           ['5-fold', aucTrain3, aucVal3, time3],
                           ['10-fold', aucTrain4, aucVal4, time4],
                           ['MCCV w/ 5', aucTrain5, aucVal5, time5],
                           ['MCCV w/ 10', aucTrain6, aucVal6, time6],
                           ['True Test', trainAuc, testAuc, 0]],
                          columns=['Strategy', 'TrainAUC', 'ValAUC', 'Time'])
    print(perfDF)


if __name__ == "__main__":
    main()
