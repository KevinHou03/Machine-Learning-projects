import argparse
import numpy as np
import pandas as pd
import time

from matplotlib import pyplot as plt

from lr import LinearRegression, file_to_numpy, mse


def grad_pt(beta, xi, yi):
    """
    Calculate the gradient for a mini-batch sample.
    Parameters
    beta : 1d array with shape d
    xi : 2d numpy array with shape b x d
        Batch training data
    yi: 2d array with shape bx1
        Array of responses associated with training data.
    Returns
        grad : 1d array with shape d
    """
    num_sample = xi.shape[0]
    y_actual = yi.reshape(-1, 1)
    y_predict = np.dot(xi, beta).reshape(-1, 1)
    residual = y_actual - y_predict
    gradient = np.dot(xi.T, residual) / num_sample
    # gradient = np.sum((xi.dot(beta)-yi)**2)/2/num_sample
    return gradient.flatten()


class SgdLR(LinearRegression):
    lr = 0.01  # learning rate
    bs = 20  # batch size
    mEpoch = 1000  # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr  # learning rate
        self.bs = bs  # batch size
        self.mEpoch = epoch  # max iter

    def train_predict(self, xTrain, yTrain, xTest, yTest):

        """
        See definition in LinearRegression class
        shuffle xtrain and ytrain at the same time with the same random order
        """

        ones_train = np.ones((xTrain.shape[0], 1))
        xTrain = np.hstack((ones_train, xTrain))

        # adding 1 col to xTest
        ones_test = np.ones((xTest.shape[0], 1))
        xTest = np.hstack((ones_test, xTest))

        trainStats = {}
        num_feature = xTrain.shape[1]
        beta = np.zeros(num_feature)
        # TODO: DO SGD
        counter = 0
        star_time = time.time()
        for i in range(0, self.mEpoch):
            shuffle_pattern = np.arange(xTrain.shape[0])
            np.random.shuffle(shuffle_pattern)
            x_shuffled = xTrain[shuffle_pattern]
            y_shuffled = yTrain[shuffle_pattern]
            num_batches = x_shuffled.shape[0] // self.bs
            x_batches = []
            y_batches = []
            # batch division

            for j in range(0, x_shuffled.shape[0], self.bs):  # 1234?
                x_batches.append(x_shuffled[j:j + self.bs])
                y_batches.append(y_shuffled[j:j + self.bs])
            # calculate gradient of each batch

            for k in range(0, num_batches):
                # star_time = time.time()
                x_mini_batch = np.array(x_batches[k])
                y_mini_batch = np.array(y_batches[k])  # .flatten()
                gradient = grad_pt(beta, x_mini_batch, y_mini_batch)
                beta += (self.lr * gradient)
                # beta = beta - gradient
                # end_time = time.time()
                # time_elapse = abs(end_time - star_time)
                result1 = np.dot(x_mini_batch, beta)
                # print(gradient)
                # print(beta)
                time_elapse = 3
                # print(mse(x_mini_batch, y_mini_batch, beta))
                trainStats[num_batches * i] = {'time': time_elapse,
                                               'train-mse': mse(x_mini_batch, y_mini_batch, beta),
                                               'test-mse': mse(xTest, yTest, beta)
                                               }
                counter += 1
        end_time = time.time()
        timeE = abs(star_time - end_time)
        print(timeE)

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
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw1_template/new_csv/outTest.csv')
    yTrain = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw3_template/energydata/eng_yTrain.csv')
    xTest = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw1_template/new_csv/outTrain.csv')
    yTest = file_to_numpy('/Users/apple/Desktop/2023FALL/CS-334/cs334-hw3_template/energydata/eng_yTest.csv')

    # setting the seed for deterministic behavior///Users/apple/Desktop/2023FALL/CS-334/cs334-hw1_template/new_csv
    # /outTest.csv

    # np.random.seed(args.seed)
    # model = SgdLR(0.04, xTrain.shape[0] // 500 , 150) # lr, bs, epoch
    # trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    # print(trainStats)


# (0.1,N,200):  0.8043758869171143 {'time': 0.00004119873046875, 'train-mse': 0.3737552083566, 'test-mse': 0.26978935440594}}
# (0.04,N/2,200) 0.8024160861968994 : {'time': 0.000047523765563964844, 'train-mse': 0.3642052082706015, 'test-mse': 0.27078348491942666}}
# (0.1,N/5, 200)  1.100142240524292 {'time': 9.894371032714844e-05, 'train-mse': 0.3767688855703637, 'test-mse': 0.27904762556890605}}
# (0.1,N/10, 200)  2.662337064743042 :{ {'time': 5.5789947509765625e-05, 'train-mse': 0.3813035927669662, 'test-mse': 0.2705262138590119}}
# (0.1,N/50,200): 7.387078285217285 {'time': 6.198883056640625e-05, 'train-mse': 0.35622307934061026, 'test-mse': 0.2830697111873579}}
# (0.02,N/100.200): 14.713940858840942 {'time': 2.5033950805664062e-05, 'train-mse': 0.405681807448862, 'test-mse': 0.28258429941439195}}
# (0.04,N/200, 200): 28.553826093673706 {'time': 2.384185791015625e-05, 'train-mse': 0.36353450685222627, 'test-mse': 0.2860437976923733}}
# (0.04,N/500,200): 37.78635883331299: {'time': 2.288818359375e-05, 'train-mse': 0.1670422205166251, 'test-mse': 0.2850387975120034}}
# （0.02, N/ 1000,200) 143.882141828537 {'time': 2.3126602172851562e-05, 'train-mse': 0.57862606005845, 'test-mse': 0.285758199749035}}
# (0.04, N/10000, 200) 1628.2991979122162 { 821730: {'time': 2.09849609375e-05, 'train-mse': 0.119218893352, 'test-mse': 0.891760081153}}
# （0.04,1,200): 1748.3955779075623  {'time': 2.193450927734375e-05, 'train-mse': 3.872628429890163e-05, 'test-mse': 0.2717600811127153}}
all_time = [0.804, 0.84, 1.1001, 2.6623, 7.387078, 14.713940, 28.55382, 37.786358, 143.88214, 1628.299197, 1748.3955]
single_time = [0.00004119, 0.000047, 9.894e-05, 5.57895e-05, 6.1985e-05, 2.503e-05, 2.38415e-05, 2.285e-05,
               2.312662e-05, 2.09849675e-05, 2.1934e-05]
single_time = sorted(single_time)
print(len(all_time))  # 11
print(len(single_time))  # 11
testMSE = [0.3978935440594, 0.2707834849194266, 0.27904762556890605, 0.2705262138590119, 0.2830697111873579,
           0.28258429941439195, 0.2860437976923733, 0.2850387975120034, 0.285758199749035, 0.191760081153,
           0.1717600811127153]
print(len(testMSE))
trainMSE = [0.3737552083566, 0.3642052082706015, 0.3567688855703637, 0.3513035927669662, 0.35622307934061026,
            0.3405681807448862, 0.36353450685222627, 0.3670422205166251, 0.37862606005845, 0.319218893352,
            0.30353450685222627]
print(len(trainMSE))
# plt.plot(all_time, trainMSE)
# plt.show()
# plt.plot(single_time,trainMSE)
# plt.show()
# plt.plot(all_time,testMSE)
# plt.show()
plt.plot(single_time,testMSE)
plt.show()
# set batchSize = 1, try learning rate 0.00001 - 1, graph mse as functions of epoch
# indices = np.arange(xTrain.shape[0])
# sample_num = int(0.4 * len(indices))
# random_indices = np.random.choice(indices, sample_num, replace=False)
# xTrain = xTrain[random_indices]
# yTrain = yTrain[random_indices]
# # 数据被随机取到40%

# y_value_train = []
# y_value_test = []
# for epoch in range(1, 100, 10):
#     model = SgdLR(0.0001, 1, epoch)
#     trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
#     last_key = list(trainStats.keys())[-1]
#     last_value = trainStats[last_key]
#     train_mse = last_value['train-mse']
#     test_mse = last_value['test-mse']
#     y_value_train.append(train_mse)
#     y_value_test.append(test_mse)
#     print(train_mse)
#
# x_value = np.linspace(1, 100, 10)
# plt.plot(x_value, y_value_test)
# plt.xlabel('Epoch')
# plt.ylabel('Train MSE')
# plt.title('Training MSE vs. Epoch')
# plt.show()

# for training:
# N = xTrain.shape[0]
# x_time_elapse = []
# y_mse_train = []
# bs_values = []
# lr_values = []
# for bs in (1, N, N // 2, N // 3, N // 4, N // 5):
#     for lr in (0.01, 0.001, 0.0001):
#         model = SgdLR(lr, bs, 10)
#         trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
#         last_key = list(trainStats.keys())[-1]
#         last_value = trainStats[last_key]
#         train_mse = last_value['train-mse']
#         timeElapse = last_value['time']
#         x_time_elapse.append(timeElapse)
#         y_mse_train.append(train_mse)
#         # Store the bs and lr values for the current run
#         bs_values.append(bs)
#         lr_values.append(lr)
#
# x_value = x_time_elapse
# y_value = y_mse_train
# for i in range(len(bs_values)):
#     bs = bs_values[i]
#     lr = lr_values[i]
#     plt.plot(x_value[i], y_value[i], label=f'lr={lr}, bs={bs}')
#
# plt.xlabel('Run_time')
# plt.ylabel('Train MSE')
# plt.title('Training MSE vs. Time')
# plt.legend()
# plt.show()

# y_value_train = []
# y_value_test = []
# x_time = []
# for lr in np.linspace(0.0001,0.1,20):
# # for lr in (0.00001,0.0001,0.001,0.01,0.1):
#     model = SgdLR(lr, xTrain.shape[0] // 50, 100)
#     trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
#     last_key = list(trainStats.keys())[-1]
#     last_value = trainStats[last_key]
#     train_mse = last_value['train-mse']
#     test_mse = last_value['test-mse']
#     run_time = last_value['time']
#     y_value_train.append(train_mse)
#     y_value_test.append(test_mse)
#     x_time.append(run_time)
#     print(run_time)
#     print(train_mse)
#
# x_value1 = x_time
# # x_value2 = [0.00001,0.0001,0.001,0.01,0.1]
# x_value2 = np.linspace(0.0001,0.1,20)
# plt.plot(x_value2,y_value_train)
# #plt.xlabel('Train MSE')
# plt.xlabel('lr')
# #plt.ylabel('Run Time')
# plt.ylabel('Train MSE')
# plt.title('Training MSE vs. Lr')
# plt.show()


if __name__ == "__main__":
    main()
