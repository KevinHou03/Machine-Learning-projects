import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

def normalize_feat(xTrain, xTest):
    scale = StandardScaler()
    normalized_train = scale.fit_transform(xTrain)
    normalized_test = scale.transform(xTest)
    return normalized_train, normalized_test

def unreg_log(xTrain, yTrain, xTest, yTest):
    logistic_regression = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    logistic_regression.fit(xTrain, yTrain)
    y_pred = logistic_regression.predict_proba(xTest)[:, 1]
    fpr, tpr, _ = roc_curve(yTest, y_pred)
    auc = roc_auc_score(yTest, y_pred)
    return fpr, tpr, auc

def run_pca(xTrain, xTest, k=None):
    pca = PCA(n_components=k)
    pca.fit(xTrain)
    pca_train = pca.transform(xTrain)
    pca_test = np.dot(xTest, pca.components_.T)  # Transform test data using PCA components
    return pca_train, pca_test

# Generate synthetic data
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
xTrain, xTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
normalized_train, normalized_test = normalize_feat(xTrain, xTest)

# Train unregularized logistic regression model and compute ROC
fpr, tpr, auc = unreg_log(normalized_train, yTrain, normalized_test, yTest)

# Run PCA
k_pca = 5
pca_train, pca_test = run_pca(normalized_train, normalized_test, k=k_pca)
fpr_pca, tpr_pca, auc_pca = unreg_log(pca_train, yTrain, pca_test, yTest)

# Plot ROC curves
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {auc:.2f})')
plt.plot(fpr_pca, tpr_pca, color='blue', lw=2, label=f'PCA (AUC = {auc_pca:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
