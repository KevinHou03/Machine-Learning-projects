from stevedore.example.simple import Simple
from xgboost import XGBClassifier
import numpy as np
from sklearn.datasets import load_iris
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgb import *

wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)
xgb.fit(X_train, y_train)

predictions = xgb.predict(X_test)
acc = np.mean(predictions == y_test)
print(acc)