import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from q1 import *


X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the data into training and testing sets
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)
# Normalize features
normalized_train, normalized_test = normalize_feat(xTrain, xTest)

# Train an unregularized logistic regression model and calculate ROC
fpr, tpr, auc = unreg_log(normalized_train, yTrain, normalized_test, yTest)

logistic_regression = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
logistic_regression.fit(normalized_train, yTrain)
predicted_probabilities = logistic_regression.predict_proba(normalized_test)
positive_class_probabilities = predicted_probabilities[:, 1]

print("False Positive Rate (FPR):")
print(fpr)
print("\nTrue Positive Rate (TPR):")
print(tpr)
print("\nArea Under the Curve (AUC):", auc)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
