import argparse
import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        # TODO: DO SOMETHING
        start_time = time.time()
        # adding 1 col to xTrain
        ones_train = np.ones((xTrain.shape[0], 1))
        xTrain = np.hstack((ones_train, xTrain))

        # adding 1 col to xTest
        ones_test = np.ones((xTest.shape[0], 1))
        xTest = np.hstack((ones_test, xTest))

        # xTest 16770,10 yTest 10,10
        # xTest.T 10,16770

        # calculating beta B = (xTx)^-1 xTy using xtrain
        self.beta = np.dot(np.dot(np.linalg.inv(np.dot(xTrain.T, xTrain)), xTrain.T), yTrain)
        # self.beta = np.linalg.inv(xTrain.T @ xTrain) @ xTrain.T @ yTrain
        # self.beta = np.linalg.inv(xTrain.T.dot(xTrain)).dot(xTrain).dot(yTrain)

        # predict
        test_pre_result = self.predict(xTest)
        train_pre_result = self.predict(xTrain)

        end_time = time.time()
        time_elapse = abs(start_time - end_time)

        # calculating mse
        train_mse = self.mse(xTrain, yTrain)
        test_mse = self.mse(xTest, yTest)

        # return a dic as required
        trainStats = {
            0: {
                'time': time_elapse,
                'train-mse': train_mse,
                'test-mse': test_mse
            }
        }

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)





xTrain = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw1_template/new_csv/outTest.csv')
yTrain = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw3_template/energydata/eng_yTrain.csv')
xTest = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw1_template/new_csv/outTrain.csv')
yTest = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw3_template/energydata/eng_yTest.csv')


'''
{0: {'time': 0.0053250789642333984, 'train-mse': 0.35290543806079167, 'test-mse': 0.28247643762437485}}
{0: {'time': 0.0033180713653564453, 'train-mse': 0.35290543806079167, 'test-mse': 0.28247643762437485}}
'''

num_runs = 10
run_data = []

for _ in range(num_runs):
    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    run_data.append(trainStats[0])  # Append the results for this run

# Extract the runtime, train MSE, and test MSE from the results
runtimes = [run['time'] for run in run_data]
train_mses = [run['train-mse'] for run in run_data]
test_mses = [run['test-mse'] for run in run_data]

# Create a plot to visualize the relationship
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(runtimes, train_mses, marker='o', linestyle='-')
plt.title('Train MSE vs. Runtime')
plt.xlabel('Runtime (seconds)')
plt.ylabel('Train MSE')

plt.subplot(1, 2, 2)
plt.plot(runtimes, test_mses, marker='o', linestyle='-')
plt.title('Test MSE vs. Runtime')
plt.xlabel('Runtime (seconds)')
plt.ylabel('Test MSE')

plt.tight_layout()
plt.show()

if __name__ == "__main__":
    main()
