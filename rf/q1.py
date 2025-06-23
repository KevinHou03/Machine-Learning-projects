import numpy as np
import sklearn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


def normalize_feat(xTrain, xTest):
    scale = StandardScaler()

    normalized_train = scale.fit_transform(xTrain)
    normalized_test = scale.transform(xTest)

    return normalized_train, normalized_test


def unreg_log(xTrain, yTrain, xTest, yTest):
    logistic_regression = LogisticRegression(penalty='none', solver='lbfgs', max_iter=1000)
    logistic_regression.fit(xTrain, yTrain)

    # prediction:
    y_pred = logistic_regression.predict_proba(xTest)[:, 1]
    fpr, tpr, thresh = roc_curve(yTest, y_pred)
    auc = roc_auc_score(yTest, y_pred)

    return fpr, tpr, auc


def run_pca(xTrain, xTest):
    pca = PCA(n_components=None)
    pca.fit(xTrain)

    cumulative_sum = np.cumsum(pca.explained_variance_ratio_)
    components_num = np.argmax(cumulative_sum >= 0.95) + 1

    pca2 = PCA(n_components=components_num)
    pca_train = pca2.fit_transform(xTrain)
    pca_test = pca2.transform(xTest)

    components_set = pca2.components_

    top_3_components = components_set[:3]

    return pca_train, pca_test, top_3_components

