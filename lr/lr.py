import time
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


def mse(xFeat, y, beta):
    """
    """
    yHat = np.dot(xFeat, beta)
    return mean_squared_error(y, yHat)


class LinearRegression(ABC):
    """
    Base Linear Regression class from which all
    linear regression algorithm implementations are
    subclasses. Can not be instantiated.
    """
    beta = None  # Coefficients
    '''
    beta = [b_0,b_1] for y^ = b_0 + b_1*x
    '''

    @abstractmethod
    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        Train the linear regression and predict the values

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : dictionary
            key refers to the batch number
            value is another dictionary with time elapsed and mse
            :param xTrain:
        """
        result = {}
        start_time = time.time()
        # adding 1 col to xTrain
        ones_train = np.ones((xTrain.shape[0], 1))
        xTrain = np.hstack((ones_train, xTrain))

        # adding 1 col to xTest
        ones_test = np.ones((xTest.shape[0], 1))
        xTest = np.hstack((ones_test, xTest))

        # calculating beta B = (xTx)^-1 xTy using xtrain
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(xTrain.T, xTrain)), xTrain.T), yTrain)
        # self.beta = np.linalg.inv(xTrain.T @ xTrain) @ xTrain.T @ yTrain
        self.beta = np.linalg.inv(xTrain.T.dot(xTrain)).dot(xTrain).dot(yTrain)

        # predict
        test_pre_result = self.predict(xTest)
        train_pre_result = self.predict(xTrain)

        end_time = time.time()
        time_elapse = abs(start_time - end_time)

        # calculating mse
        train_mse = self.mse(xTrain, yTrain)
        test_mse = self.mse(xTest, yTest)

        # return a dic as required
        result = {
            0: {
                'time': time_elapse,
                'train-mse': train_mse,
                'test-mse': test_mse
            }
        }

        return result

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """

        yHat = []
        # TODO
        # the coefficient already given
        # ones = np.ones((xFeat.shape[0], 1))
        # xFeat = np.hstack((ones, xFeat))
        yHat = np.dot(xFeat, self.beta)
        return yHat

    def mse(self, xFeat, y):
        """
        """
        yHat = self.predict(xFeat)
        return mean_squared_error(y, yHat)


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()
