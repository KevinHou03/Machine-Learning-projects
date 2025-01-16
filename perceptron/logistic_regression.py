import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/binary_train.csv')
# yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTrain.csv').iloc[:, 1]
# xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/binary_test.csv')
# yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTest.csv').iloc[:, 1]

xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/count_train.csv')
yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTrain.csv').iloc[:, 1]
xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/count_test.csv')
yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTest.csv').iloc[:, 1]

LRclassifier = LogisticRegression(max_iter=200)
LRclassifier.fit(xTrain, yTrain.values.ravel())

y_pred = LRclassifier.predict(xTest)

mistakes_count = (y_pred != yTest.values.ravel()).sum()

accuracy = accuracy_score(yTest, y_pred)

print(f'Number of mistakes:', mistakes_count)
print(f'Accuracy:', accuracy)
