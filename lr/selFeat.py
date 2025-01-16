import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

'''
1a.the useful information would be year, month, days, and hours, particularly month,days, and hours. Year might not
matter
'''


def extract_features(df):
    """
    Given a pandas dataframe, extract the relevant features
    from the date column

    Parameters
    ----------
    df : pandas dataframe
        Training or test data
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with the new features
    """
    # TODO do more than this
    # 1/11/16 17:00
    date = df['date']
    date = pd.to_datetime(date, format='%m/%d/%y %H:%M')
    # year = date.dt.year
    month = date.dt.month
    day = date.dt.day
    hour = date.dt.hour
    minute = date.dt.minute

    df.insert(0, 'minute', minute)
    df.insert(1, 'Month', month)
    df.insert(2, 'Day', day)
    df.insert(3, 'hour', hour)
    # year/month/day/time/date
    df = df.drop(columns=['date'])
    print('remove date:', df)
    # year/month/day/time/
    return df


def cal_corr(df):
    """
    Given a pandas dataframe (include the target variable at the last column),
    calculate the correlation matrix (compute pairwise correlation of columns)

    Parameters
    ----------
    df : pandas dataframe
        Training or test data (with target variable)
    Returns
    -------
    corrMat : pandas dataframe
        Correlation matrix
    """
    # TODO
    corrMat = df.corr()

    # plt.figure(figsize=(8, 6))  # Set the size of the heatmap
    # sns.heatmap(corrMat, annot=True, cmap='coolwarm',
    #             fmt=".2f")  # annot=True to show correlation values, cmap for color map, fmt to format the values
    #
    # plt.title('Correlation Matrix Heatmap')  # Add a title to the heatmap
    # plt.show()

    return corrMat


def select_features(df):
    """
    Select the features to keep

    Parameters
    ----------
    df : pandas dataframe
        Training or test data
    Returns
    -------
    df : pandas dataframe
        The updated dataframe with a subset of the columns
    """
    # TODO

    Corr = df.corr()
    for columns, values in Corr.iloc[-1, :].items():
        if abs(values) < 0.3:
            df = df.drop(columns=[columns])  # get the last row which should includes corr between target and each feat
    print('remove irrelevant:', df)

    return df


def preprocess_data(trainDF, testDF):
    '''
    what else? handle missing data?

    '''
    """
    Preprocess the training data and testing data

    Parameters
    ----------
    trainDF : pandas dataframe
        Training data
    testDF : pandas dataframe
        Test data
    Returns
    -------
    trainDF : pandas dataframe
        The preprocessed training data
    testDF : pandas dataframe
        The preprocessed testing data
    """
    # TODO do something
    # scale = StandardScaler()
    # trainDF.iloc[:, 1:] = scale.fit_transform(trainDF.iloc[:, 1:])
    # # use eth same scaler
    # testDF.iloc[:, 1:] = scale.fit_transform(testDF.iloc[:, 1:])

    # scale_factor = StandardScaler()
    # scale_factor.fit(trainDF)
    # trainDF = scale_factor.transform(trainDF)
    # testDF = scale_factor.transform(testDF)
    #
    #
    #
    # return trainDF, testDF
    scale_factor1 = StandardScaler()
    scale_factor2 = StandardScaler()

    scaled_trainDF = scale_factor1.fit_transform(trainDF)
    scaled_testDF = scale_factor2.fit_transform(testDF)

    scaled_trainDF = pd.DataFrame(scaled_trainDF)
    scaled_testDF = pd.DataFrame(scaled_testDF)

    print('scaled train', scaled_trainDF)
    print('scaled test', scaled_testDF)

    return scaled_trainDF, scaled_testDF


def make_test_like_train(trainDF, testDF):
    for columns in testDF.columns:  # 取到testDF的每一列
        if columns not in trainDF.columns:
            testDF = testDF.drop(columns=[columns])
    return testDF


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("outTrain",
                        help="filename of the updated training data")
    parser.add_argument("outTest",
                        help="filename of the updated test data")
    parser.add_argument("--trainFile",
                        default="eng_xTrain.csv",
                        help="filename of the training data")
    parser.add_argument("--testFile",
                        default="eng_xTest.csv",
                        help="filename of the test data")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/cs334-hw3_template/energydata/eng_xTrain.csv')
    xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/cs334-hw3_template/energydata/eng_xTest.csv')
    # extract the new features
    xNewTrain = extract_features(xTrain)
    xNewTest1 = extract_features(xTest)
    # select the features
    xNewTrain = select_features(xNewTrain)
    xNewTest = make_test_like_train(xNewTrain, xNewTest1)

    # xNewTest = select_features(xNewTest)
    # preprocess the data
    xTrainTr, xTestTr = preprocess_data(xNewTrain, xNewTest)
    # save it to csv
    xTrainTr.to_csv('/Users/apple/Desktop/2023FALL/cs334-hw1_template/new_csv/outTrain.csv', index=False)
    xTestTr.to_csv('/Users/apple/Desktop/2023FALL/cs334-hw1_template/new_csv/outTest.csv', index=False)


if __name__ == "__main__":
    main()
