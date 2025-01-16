'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Kaiyuan Hou
'''
import argparse
import numpy as np
import pandas as pd
from operator import itemgetter
from collections import Counter
import matplotlib.pyplot as plt
import sys

'''



def euclidean_distance_single_labeled(test_data, train, labels):
    # print(type(test_data), type(train))  # two numpy array
    # test_data = pd.DataFrame(test_data)
    train_dataset = pd.DataFrame(train)
    train_dataset_labeled = pd.DataFrame(train)
    train_dataset_labeled.insert(2, 'labels', labels)
    # print(train_dataset)
    result = []
    for j in range(0, train.shape[0]):
        train_data = train_dataset.iloc[j]  # train_data without label// train_data is a dataframe
        train_data_labeled = train_dataset_labeled.iloc[j]
        distance = np.sqrt(np.sum((np.array(test_data) - np.array(train_data)) ** 2))
        result.append([distance, train_data_labeled['labels']])
    return result


def euclidean_distance_single(test_data, train):
    # print(type(test_data), type(train))  # two numpy array
    train_dataset = pd.DataFrame(train)
    result = []
    for j in range(0, train.shape[0]):
        train_data = train_dataset.iloc[j]
        distance = np.sqrt(np.sum((np.array(test_data) - np.array(train_data)) ** 2))
        result.append(distance)
    return result





def euclidean_distance(test_array, train_array):
    # print(type(test_array), type(train_array))
    test = pd.DataFrame(test_array)
    train = pd.DataFrame(train_array)
    result = []
    # print(type(test),type(train))
    for i in range(0, test.shape[0]):
        distance_result = []
        test_target = test.iloc[i]
        for j in range(0, train.shape[0]):
            train_data = train.iloc[j]
            distance = np.sqrt(np.sum((np.array(test_target) - np.array(train_data)) ** 2))
            distance_result.append(distance)
        result.append(distance_result)
    # print(type(result))
    return result
'''


class Knn(object):
    k = 5  # number of neighbors to use

    def __init__(self, k):
        """
        knn constructor

        Parameters
        ----------
        k : int
            Number of neighbors to use.
        """
        self.x_train_data = None
        self.class_labels_train = None
        self.k = k

    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need
        self.x_train_data = xFeat
        self.class_labels_train = y
        return self

    '''
    def predict(self, xFeat):

        yHat = []  # variable to store the estimated class label
        xFeat = pd.DataFrame(xFeat)
        # TODO
        # for subject in xFeat.values:
        # xFeat.shape[0]-1
        for i in range(0, 4):
            subject = xFeat.values[i]
            # print(subject)
            distance_list = euclidean_distance_single_labeled(xFeat.values[i], self.x_train_data,
                                                              self.class_labels_train)

            distance_list_sorted = sorted(distance_list, key=itemgetter(0))
            # print(distance_list_sorted)
            train_data_labeled = [t[1] for t in distance_list_sorted[0:self.k]]
            print("train_data_labeled:", train_data_labeled)
            counter = Counter(train_data_labeled)
            most_common = counter.most_common(1)[0][0]
            # print("most common:", most_common)
            yHat.append(most_common)
            # print("yHat:", yHat)

        return yHat
        '''

    '''
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
        Predicted class label per sample
    """
    yHat = []  # variable to store the estimated class label
    xFeat = pd.DataFrame(xFeat)
    # TODO
    # for subject in xFeat.values:
    # xFeat.shape[0]-1
    for i in range(0, xFeat.shape[0] - 1):  # 这里是取出了test集里面的每一个元素，并拿test里的每一个元素和train中所有元素算距离
        subject = xFeat.values[i]
        # distance_list = euclidean_distance_single_labeled(xFeat.values[i], self.x_train_data,
        # self.class_labels_train)
        train_dataset = pd.DataFrame(self.x_train_data)
        train_dataset_labeled = pd.DataFrame(self.x_train_data)
        train_dataset_labeled.insert(2, 'labels', self.class_labels_train)
        result = []
        for j in range(0, self.x_train_data.shape[0]):
            train_data = train_dataset.iloc[j]
            train_data_labeled = train_dataset_labeled.iloc[j]
            distance = np.sqrt(np.sum((np.array(xFeat.values[i]) - np.array(train_data)) ** 2))
            result.append([distance, train_data_labeled['labels']])  # now the result stores a bunch of 1 and 0s
        result = sorted(result, key=itemgetter(0))
        # print(distance_list_sorted)
        marked_target = [t[1] for t in result[0:self.k]]
        print("train_data_labeled:", marked_target)
        counter = Counter(marked_target)
        most_common = counter.most_common(1)[0][0]
        # print("most common:", most_common)
        yHat.append(most_common)
        # print("yHat:", yHat)

    return yHat
    '''

    def predict(self, xFeat):
        xFeat = np.array(xFeat)
        # TODO
        # for subject in xFeat.values:
        # xFeat.shape[0]-1
        # print(xFeat.shape[0])
        # print(type(self.class_labels_train)) series
        row_number = xFeat.shape[0]
        # print(type(self.class_labels_train))
        # yHat = np.empty(row_number)
        yHat = []
        train_dataset = np.array(self.x_train_data)
        train_labels = np.array(self.class_labels_train)

        for new_data_feature in xFeat:
            # subject = xFeat.values[i]
            # distances = np.sqrt(np.sum((xFeat[i] - train_dataset) ** 2))
            distances = np.sqrt(np.sum((new_data_feature - train_dataset) ** 2, axis=1))
            # print(type(distances),distances) nparray
            sorted_indices = np.argsort(distances)[0:self.k]  # 获得到从小到大排列 index的顺序
            # print(type(sorted_indices)) ndarray fancy indexing?
            sorted_labels = train_labels[sorted_indices]

            sorted_labels = [int(label) for label in sorted_labels]
            # print(type(sorted_labels[0]))
            # print(type(sorted_labels),sorted_labels)
            # Count the most common label and store it in yHat
            counter = Counter(sorted_labels)
            most_common = counter.most_common(1)[0][0]
            # print("most common:", most_common)
            # yHat[i] = most_common
            yHat.append(most_common)
            # print("mostcommontype", type(most_common))
            # print(yHat[i])
            #print(yHat)

        return yHat


def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    # acc = 0
    # accuracy = correct labels / all labels
    # correct labels -> yHat[i] = yTrue[i]
    correctness_counter = 0
    for number, value in enumerate(yHat):
        if yHat[number] == yTrue[number]:
            correctness_counter += 1
    acc = correctness_counter / len(yHat)
    return acc


def draw_accuracy(xTrain, yTrain, xTest, yTest):
    k_value = list(range(1, 501))
    test_acc_list = []
    train_acc_list = []
    for k in k_value:
        knn = Knn(k)
        knn.train(xTrain, yTrain['label'])
        yHatTrain = knn.predict(xTrain)
        train_acc = accuracy(yHatTrain, yTrain['label'])
        # print("Training Acc:",train_acc)
        yHatTest = knn.predict(xTest)
        test_acc = accuracy(yHatTest, yTest['label'])
        #   print("Test acc:", test_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)
        # a list of q00 elements: test accuracy for every k

    fig, ax = plt.subplots()
    plt.plot(k_value, test_acc_list)
    plt.plot(k_value, train_acc_list)
    plt.xlabel('k value')
    plt.ylabel('test accuracy')
    plt.title('test accuracy for k')
    plt.show()


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3xTrain.csv')
    yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3yTrain.csv')
    xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3xTest.csv')
    yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3yTest.csv')
    # create an instance of the model
    knn = Knn(args.k)

    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    # sys.exit()
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)

    draw_accuracy(xTrain, yTrain, xTest, yTest)


if __name__ == "__main__":
    main()
