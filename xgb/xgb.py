import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class SimpleDicisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.value = None

    # def fit(self, X, y, depth=0):
    #     if depth == self.max_depth or len(np.unique(y)) == 1:  # 当前节点下只有一个类别
    #         self.value = np.argmax(np.bincount(y.astype(int)))  # return the index of t和predicted value（max）
    #         return

    def fit(self, X, grad, hess, depth=0):
        if depth == self.max_depth or len(grad) == 0:
            self.value = np.sum(grad) / (np.sum(hess) + 1e-6)  # Newton step
            return

        n_samples, n_features = X.shape
        best_gain = -np.inf

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])  # 该特征所有可能取值，在这里面找split point
            for threshold in thresholds:
                left_indices = X[:, feature] <= threshold
                right_indices = X[:, feature] > threshold

                # 跳过无效split，即所有样本都去了left or right
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                g_left = grad[left_indices]
                h_left = hess[left_indices]
                g_right = grad[right_indices]
                h_right = hess[right_indices]

                gain = self._calculate_gini(g_left, h_left, g_right, h_right)
                if gain > best_gain:
                    best_gain = gain
                    self.feature_index = feature
                    self.threshold = threshold

        if self.feature_index is not None:  # 如果找到一个合适的split point，递归的在左右sub tree继续创建
            left_indices = X[:, self.feature_index] <= self.threshold
            right_indices = X[:, self.feature_index] > self.threshold

            self.left = SimpleDicisionTree(max_depth=self.max_depth)  # 递归1
            self.left.fit(X[left_indices], grad[left_indices], hess[left_indices], depth + 1)

            self.right = SimpleDicisionTree(max_depth=self.max_depth)  # 递归2
            self.right.fit(X[right_indices], grad[right_indices], hess[right_indices], depth + 1)

        else:  # 没有找到分裂点呢，那就直接预测，用当前结果，并把当前节点设置为leaf node
            self.value = -np.sum(grad) / (np.sum(hess) + 1e-6)

    # def _calculate_gini(self, left_labels, right_labels):
    #     def gini(labels):
    #         _, counts = np.unique(labels, return_counts=True)
    #         probs = counts / counts.sum()
    #         return 1.0 - np.sum(probs ** 2)
    #     total = len(left_labels) + len(right_labels)
    #     return (len(left_labels) * gini(left_labels) + len(right_labels) * gini(right_labels)) / total

    def _calculate_gini(self, g_left, h_left, g_right, h_right):
        def calc_score(g, h):
            return (np.sum(g) ** 2) / (np.sum(h) + 1e-6)

        gain = calc_score(g_left, h_left) + calc_score(g_right, h_right)
        return gain  # 越大越好

    def predict(self, X):
        if self.feature_index is None:
            return np.array([self.value] * len(X))  # 构造一个array，长度为sample_n，每一项都是self.value。
        else:
            left_indices = X[:, self.feature_index] <= self.threshold
            right_indices = X[:, self.feature_index] > self.threshold

            predictions = np.zeros(len(X))
            predictions[left_indices] = self.left.predict(X[left_indices])
            predictions[right_indices] = self.right.predict(X[right_indices])

            return predictions


class SimpleXGBoost:
    def __init__(self, n_estimators=100, max_depth=3, lr=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.estimators = []  # 储存过程中生成的每一棵树

    def sigmoid(self, x):
        x = np.clip(x, -709, 709)  # 防止 np.exp 溢出，709 是 np.exp 能接受的最大值
        return 1 / (1 + np.exp(-x))
    # def fit(self, X, y):
    #     y = np.asarray(y, dtype=int)
    #     y_pred = np.zeros(len(y))
    #     for _ in range(self.n_estimators):
    #         residual = y - y_pred  # 这是梯度提升的关键，每一棵新树都在拟合残差
    #         tree = SimpleDicisionTree(max_depth=self.max_depth)
    #         tree.fit(X, residual)
    #         y_pred += self.lr * tree.predict(X)
    #         self.estimators.append(tree)

    def fit(self, X, y):
        y = np.asarray(y, dtype=int)
        y_pred = np.zeros(len(y))  # logit, not sigmoid yet

        for _ in range(self.n_estimators):
            p = self.sigmoid(y_pred)
            g = p - y  # 一阶导：dL/dy_pred
            h = p * (1 - p)  # 二阶导：d²L/dy_pred²

            tree = SimpleDicisionTree(max_depth=self.max_depth)
            tree.fit(X, g, h)
            y_pred += self.lr * tree.predict(X)  # 负梯度方向

            self.estimators.append(tree)

    def predict_proba(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.estimators:
            y_pred -= self.lr * tree.predict(X)
        return self.sigmoid(y_pred)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)



iris = load_iris()
mask = iris.target <= 1
X = iris.data[mask]
y = iris.target[mask]

# 然后再划分 train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

xgb = SimpleXGBoost(n_estimators=100, max_depth=3, lr=0.1)
xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)

accuracy = np.mean(predictions == y_test)
print("Accuracy:", accuracy)
