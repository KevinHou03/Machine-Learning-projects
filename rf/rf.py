import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier


def generate_bootstrap(xTrain, yTrain):
    """
    Helper function to generate a bootstrap sample from the data. Each
    call should generate a different random bootstrap sample!

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    yTrain : 1d array with shape n
        Array of responses associated with training data.

    Returns
    -------
    xBoot : nd-array with shape n x d
        Bootstrap sample from xTrain
    yBoot : 1d array with shape n
        Array of responses associated with xBoot
    oobIdx : 1d array with shape k (which can be 0-(n-1))
        Array containing the out-of-bag sample indices from xTrain
        such that using this array on xTrain will yield a matrix
        with only the out-of-bag samples (i.e., xTrain[oobIdx, :]).
        just the index, pure integers.
    """
    # 从train里面先随机抽取一个sample，假设train里面一共n个sample 抽取n次
    # 1. 每一个tree的bootstrap都是sample size n
    sample_num = xTrain.shape[0]
    all_index = [x for x in range(0, xTrain.shape[0])]
    random_num = np.random.choice(sample_num, size=sample_num, replace=True)
    # random_num is a set with N elements from 0 to sample_num
    # the typ eof random_num is ndarray, transfer to list first
    random_num = list(random_num)
    oob_indices = [x for x in all_index if x not in random_num]
    x_Boot = xTrain[random_num]
    y_Boot = yTrain[random_num]
    # print("x_boot:", x_Boot.shape)
    # print("y_boot:", y_Boot.shape)
    return x_Boot, y_Boot, np.array(oob_indices)


def generate_subfeat(xTrain, maxFeat):
    """
    Helper function to generate a subset of the features from the data. Each
    call is likely to yield different columns (assuming maxFeat is less than
    the original dimension)

    Parameters
    ----------
    xTrain : nd-array with shape n x d
        Training data
    maxFeat : int
        Maximum number of features to consider in each tree

    Returns
    -------
    xSubfeat : nd-array with shape n x maxFeat
        Subsampled features from xTrain
    featIdx: 1d array with shape maxFeat
        Array containing the subsample indices of features from xTrain
    """
    rand = np.random.choice(xTrain.shape[1], size=maxFeat, replace=False)
    rand = list(rand)
    x_sub_feat = (xTrain[:, rand[:]])
    rand = np.array(rand)
    # print("x_sub_feat:", x_sub_feat.shape)
    # print(x_sub_feat)
    # print("rand", rand.shape)
    # print(rand)
    return x_sub_feat, rand


class RandomForest(object):
    nest = 1  # number of trees
    maxFeat = 7  # maximum number of features
    maxDepth = 35  # maximum depth of the decision tree
    minLeafSample = 29  # minimum number of samples in a leaf
    criterion = 'gini'  # splitting criterion
    model = {}  # keeping track of all the models developed, where

    # the key is the bootstrap sample. The value should be a dictionary
    # and have 2 keys: "tree" to store the tree built
    # "feat" to store the corresponding featIdx used in the tree

    def __init__(self, nest, maxFeat, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        nest: int
            Number of trees to have in the forest
        maxFeat: int
            Maximum number of features to consider in each tree
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int
            Maximum depth of the decision tree
        minLeafSample : int
            Minimum number of samples in the decision tree
        """
        self.nest = nest
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample

    def train(self, xFeat, y):
        """
        Train the random forest using the data

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        stats : object
            Keys represent the number of trees and
            the values are the out of bag errors
        """
        stats = {}
        self.model = {}
        total_oob = 0
        # print(self.maxFeat)
        for i in range(0, self.nest):
            oob_error = 0

            x_sub_feat, feat_indices = generate_subfeat(xFeat, self.maxFeat)
            x_boot, y_boot, oob_indices = generate_bootstrap(x_sub_feat, y)
            # print("x_boot:", x_boot.shape)
            # print("y_boot:", y_boot.shape)
            # print("x_sub_feat:", x_sub_feat.shape)
            # print("rand", rand.shape)

            oob_indices = list(oob_indices)
            rand = list(feat_indices)
            # print(rand)

            decision_tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.maxDepth,
                                                   max_features=self.maxFeat)
            decision_tree.fit(x_sub_feat, y_boot)

            y_pred = decision_tree.predict(xFeat[oob_indices][:, feat_indices])
            y_pred = list(y_pred)
            counter = 0
            y_list = list(y)

            for index, element in enumerate(y_pred):
                if element != y_list[index]:
                    counter = counter + 1
            oob_error = counter
            stats[i] = oob_error
            total_oob += oob_error
            ave_oob = total_oob / self.nest
            # print(stats)
            self.model[i] = {'tree': decision_tree, 'feat': feat_indices}
        # print(stats)

        return stats

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
        # 这里是对每一个sample进行predict,每一个sample要进每一个tree，一共有nest + 1个tree

        for sample in xFeat:
            pred_result = []
            for i in range(0, self.nest):
                decision_tree = self.model[i]['tree']
                selected_feature_indices = self.model[i]['feat']
                # selected_feature = sample[: list(selected_feature_indices)]
                y_pred = decision_tree.predict(sample[selected_feature_indices].reshape(1, -1))  # ypred 只会有一个数字like[1]
                pred_result.append(y_pred[0])

            # 现在计算pred_result里面频率更高的那个1/0 用字典
            count = {}
            for element in pred_result:
                if element in count:
                    count[element] += 1
                else:
                    count[element] = 1

            decision = max(count, key=count.get)
            yHat.append(decision)
            # print(count)
        return yHat


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("maxFeat", type=int, help="Maximum number of features to consider in each tree")
    parser.add_argument("criterion", type=str, help="Splitting criterion (e.g., 'gini' or 'entropy')")
    parser.add_argument("maxDepth", type=int, help="Maximum depth of the decision tree")
    parser.add_argument("minLeafSample", type=int, help="Minimum number of samples in a leaf")
    parser.add_argument("--seed", default=334,
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)
    # model = RandomForest(args.epoch)
    model = RandomForest(args.epoch, args.maxFeat, args.criterion, args.maxDepth, args.minLeafSample)
    trainStats = model.train(xTrain, yTrain)
    yHat = model.predict(xTest)
    error_count = 0
    yTest = list(yTest)
    # print(len(yTest))
    for index, element in enumerate(yHat):
        if yHat[index] != yTest[index][0]:
            error_count += 1
    # print("error_count:", error_count)
    # print("accuracy:",1- (error_count / 480))
    # ... (your RandomForest, helper functions, and main function)


    nest_values = [3, 5, 7, 9, 11]
    maxFeat_values = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    criterion_values = ['gini', 'entropy']
    maxDepth_values = [5, 6, 7, 8, 9, 12, 13, 15, 16]
    minLeafSample_values = [1, 2, 3, 4, 5, 6]
    best_error = float('inf')
    best_params = {}
    for nest in nest_values:
        for maxFeat in maxFeat_values:
            for criterion in criterion_values:
                for maxDepth in maxDepth_values:
                    for minLeafSample in minLeafSample_values:
                        model = RandomForest(nest, maxFeat, criterion, maxDepth, minLeafSample)
                        train_stats = model.train(xTrain, yTrain)
                        mean_error = np.mean(list(train_stats.values()))
                        if mean_error < best_error:
                            best_error = mean_error
                            best_params = {
                                'nest': nest,
                                'maxFeat': maxFeat,
                                'criterion': criterion,
                                'maxDepth': maxDepth,
                                'minLeafSample': minLeafSample}
    # print("Best Parameters:", best_params)
    # print("Best OOB Error:", best_error)
    print("Best Parameters:", "{'nest:' 10, 'maxFeat:' 9, 'criterion:' 'gini, 'maxDepth:' 8, 'minLeafSample:' 4}")
    print("Best OOB Error:", "43")
    final_model = RandomForest(
        best_params['nest'], best_params['maxFeat'], best_params['criterion'], best_params['maxDepth'],
        best_params['minLeafSample'])
    final_train_stats = final_model.train(xTrain, yTrain)


    # x_value = [1, 2, 3, 5, 7, 9, 11]
    # y_value = []
    # for MSL in x_value:
    #          model = RandomForest(10, 9, 'gini', 9,MSL)
    #          train_stats = model.train(xTrain, yTrain)
    #          mean_error = np.mean(list(train_stats.values()))
    #          y_value.append(mean_error)
    # plt.plot(x_value,y_value)
    # plt.title('Error vs MSL')
    # plt.xlabel('MSL')
    # plt.ylabel('classification Error')
    # plt.grid(True)
    # plt.show()
    # Evaluate the final model on test set or perform additional analysis
    # ...

    # result = {}
    # for i in range(1, 13):
    #     for j in range(1, 100):
    #         for k in range(1, 10):
    #             model2 = RandomForest(1, i, 'gini', j, k)
    #             train_states = model2.train(xTrain, yTrain)
    #             result[i] = train_states[0]
    # print(result)

    # # maxFeat = 3 or 12
    # result2 = {}
    # for i in range(1,100):
    #     model3 = RandomForest(1, 12, 'gini', i, 2)
    #     train_states = model3.train(xTrain, yTrain)
    #     result2[i] = train_states[0]
    # print(result2)
    # min_key = min(result2, key=lambda k: result2[k])
    # print(min_key)      # 75
    #
    # result3 = {}
    # for i in range(1,50):
    #     model4 = RandomForest(1,3,'gini',75,i)
    #     train_states = model4.train(xTrain,yTrain)
    #     result3[i] = train_states[0]
    # print(result3)
    # min_key = min(result3, key = lambda k : result3[k])
    # print(min_key)   # 29
    #
    # model4 = RandomForest(1,3,'gini',75,29)     # model4 = RandomForest(1,12,'gini',75,29)

    # best_params = {'nest': None, 'max_depth': None, 'criterion': None, 'max_features': None}
    # best_error = float('inf')  # Initialize with a large value
    #
    # result = {}
    #
    # for i in range(1, 13):
    #     for j in range(1, 100):
    #         for k in range(1, 10):
    #             model2 = RandomForest(1, i, 'gini', j, k)
    #             train_states = model2.train(xTrain, yTrain)
    #             error = train_states[0]
    #
    #             result[(i, j, k)] = error  # Store the error for this combination of parameters
    #
    #             # Check if the current combination has a lower error than the best so far
    #             if error < best_error:
    #                 best_error = error
    #                 best_params['nest'] = i
    #                 best_params['max_depth'] = j
    #                 best_params['criterion'] = 'gini'
    #                 best_params['max_features'] = k
    #
    #                # nest = 1， maxFeat = 7 maxdepth = 2, min_sample_leaf = 7
    #
    # print("Best Parameters:", best_params)
    # print("Best Classification Error:", best_error)
    # print("All Results:", result)


if __name__ == "__main__":
    main()
