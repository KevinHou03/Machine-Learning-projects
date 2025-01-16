from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

x_train = None
y_train = None

x_test = None
y_test = None
clf = GaussianNB()
clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)
y_actual = y_test

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

