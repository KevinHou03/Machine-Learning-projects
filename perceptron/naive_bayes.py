import pandas as pd
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix

# xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/binary_train.csv')
# yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTrain.csv').iloc[:, 1]
# xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/binary_test.csv')
# yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTest.csv').iloc[:, 1]

xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/count_train.csv')
yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTrain.csv').iloc[:, 1]
xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/count_test.csv')
yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/CS-334/yTest.csv').iloc[:, 1]

print(xTrain.shape)
print(yTrain.shape)
print(xTest.shape)
print(yTest.shape)
NB = MultinomialNB()
NB.fit(xTrain, yTrain.values.ravel())

y_pred = NB.predict(xTest)
mistakes_count = (y_pred != yTest.values.ravel()).sum()
accuracy = accuracy_score(yTest, y_pred)
confusion_matrix = confusion_matrix(yTest, y_pred)

print('Number of mistakes:\n', mistakes_count)
print('Accuracy:\n', accuracy)
print('Confusion Matrix:\n')
print(confusion_matrix)
