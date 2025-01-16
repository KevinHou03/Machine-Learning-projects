'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Kaiyuan Hou
'''
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler


# from q3 import *
def standard_scale(xTrain, xTest):
    """
    Preprocess the training data to have zero mean and unit variance.
    The same transformation should be used on the test data. For example,
    if the mean and std deviation of feature 1 is 2 and 1.5, then each
    value of feature 1 in the test set is standardized using (x-2)/1.5.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with mean 0 and unit variance
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # TODO FILL IN
    '''
    train_mean = xTrain.mean()
    train_std = xTrain.std()
    test_mean = xTest.mean()
    test_std = xTest.std()
    xTrain = (xTrain - train_mean) / train_std
    xTest = (xTest - test_mean) / test_std
    '''

    scaler = StandardScaler()
    # xTrain = xTrain.reshape(-1,1)
    # xTest = xTest.reshape(-1,1)
    scaler.fit(xTrain)
    # scaler.fit(xTest)
    xTrain = scaler.transform(xTrain)
    xTest = scaler.transform(xTest)
    # np.set_printoptions(precision=5, suppress=True)

    return xTrain, xTest


def minmax_range(xTrain, xTest):
    """
    Preprocess the data to have minimum value of 0 and maximum
    value of 1.The same transformation should be used on the test data.
    For example, if the minimum and maximum of feature 1 is 0.5 and 2, then
    then feature 1 of test data is calculated as:
    (1 / (2 - 0.5)) * x - 0.5 * (1 / (2 - 0.5))

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data

    Returns
    -------
    xTrain : nd-array with shape n x d
        Transformed training data with min 0 and max 1.
    xTest : nd-array with shape m x d
        Transformed test data using same process as training.
    """
    # TODO FILL IN

    df_test_data = pd.DataFrame(xTest)
    df_train_data = pd.DataFrame(xTrain)

    test_information = df_test_data.describe()

    for test_column in df_test_data.columns:
        test_min = df_test_data[test_column].describe().loc['min']
        test_max = df_test_data[test_column].describe().loc['max']

        if test_max == test_min:
            df_test_data[test_column] = 0
        else:
            df_test_data[test_column] = (1 / (test_max - test_min)) * df_test_data[test_column] - test_min * (
                    1 / (test_max - test_min))

    for train_column in df_train_data.columns:
        train_min = df_train_data[train_column].describe().loc['min']
        train_max = df_train_data[train_column].describe().loc['max']

        if train_min == train_max:
            df_train_data[train_column] = 0
        else:
            df_train_data[train_column] = (1 / (train_max - train_min)) * df_train_data[train_column] - train_min * (
                    1 / (train_max - train_min))

    xTest = df_test_data.values
    xTrain = df_train_data.values

    #scaler_test = MinMaxScaler()
    # scaler_test.fit(xTest)

    scaler_train = MinMaxScaler()
    scaler_train.fit(xTrain)
    xTrain = scaler_train.transform(xTrain)
    xTest = scaler_train.transform(xTest)

    return xTrain, xTest
    # return df_train_data, df_test_data


'''
    return df_train_data,df_test_data
'''


def add_irr_feature(xTrain, xTest):
    """
    Add 2 features using Gaussian distribution with 0 mean,
    standard deviation of 1.

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    xTest : nd-array with shape m x d
        Test data

    Returns
    -------
    xTrain : nd-array with shape n x (d+2)
        Training data with 2 new noisy Gaussian features
    xTest : nd-array with shape m x (d+2)
        Test data with 2 new noisy Gaussian features
    """
    # TODO FILL IN
    df_x_test = pd.DataFrame(xTest)
    df_x_train = pd.DataFrame(xTrain)

    irr_feature1_test = np.random.normal(0, 1, df_x_test.shape[0])
    irr_feature2_test = np.random.normal(0, 1, df_x_test.shape[0])

    irr_feature1_train = np.random.normal(0, 1, df_x_train.shape[0])
    irr_feature2_train = np.random.normal(0, 1, df_x_train.shape[0])

    df_x_test.insert(df_x_test.shape[1], 'irr_feature1_test', irr_feature1_test)
    df_x_test.insert(df_x_test.shape[1], 'irr_feature2_test', irr_feature2_test)

    df_x_train.insert(df_x_train.shape[1], 'irr_feature1_train', irr_feature1_train)
    df_x_train.insert(df_x_train.shape[1], 'irr_feature2_train', irr_feature2_train)

    xTrain = df_x_train.values
    xTest = df_x_test.values

    return xTrain, xTest
    #return df_x_train, df_x_test


'''
return df_x_train,df_x_test
'''


def knn_train_test(k, xTrain, yTrain, xTest, yTest):
    """
    Given a specified k, train the knn model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    k : int
        The number of neighbors
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    '''
    model = knn(k)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = model.predict(xTest)
    '''

    model = KNeighborsClassifier()
    model.fit(xTrain, yTrain['label'])
    yHatTest = model.predict(xTest)

    return accuracy_score(yHatTest, yTest['label'])


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
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
    xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q4xTrain.csv')
    yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q4yTrain.csv')
    xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q4xTest.csv')
    yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q4yTest.csv')

    # no preprocessing
    acc1 = knn_train_test(args.k, xTrain, yTrain, xTest, yTest)
    print("Test Acc (no-preprocessing):", acc1)
    # preprocess the data using standardization scaling
    xTrainStd, xTestStd = standard_scale(xTrain, xTest)
    acc2 = knn_train_test(args.k, xTrainStd, yTrain, xTestStd, yTest)
    print("Test Acc (standard scale):", acc2)
    # preprocess the data using min max scaling
    xTrainMM, xTestMM = minmax_range(xTrain, xTest)

    '''

    print("xTrain:", xTrain)
    print("xTest:", xTest)
    print("*************")
    print("xTrainMM:", type(xTrainMM))
    print(xTrainMM)
    print("xTestMM：", type(xTestMM))
    print(xTestMM)
    
    '''

    acc3 = knn_train_test(args.k, xTrainMM, yTrain, xTestMM, yTest)
    print("Test Acc (min max scale):", acc3)
    # add irrelevant features
    xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)
    # print(xTrainIrr)
    # rint(yTrainIrr)
    acc4 = knn_train_test(args.k, xTrainIrr, yTrain, yTrainIrr, yTest)

    print("Test Acc (with irrelevant feature):", acc4)




if __name__ == "__main__":
    main()
