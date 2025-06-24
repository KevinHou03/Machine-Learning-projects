# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn import metrics
# from sklearn.datasets import make_classification
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import cross_val_score, train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# # Generate a synthetic dataset for demonstration
# X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Hyperparameter grids
# max_depth_values = [None, 5, 10, 15, 20]
# min_samples_leaf_values = [1, 2, 4, 8]
#
# # Initialize arrays to store results
# cv_errors = np.zeros((len(max_depth_values), len(min_samples_leaf_values)))
#
# # Perform cross-validation for each combination of max_depth and min_samples_leaf
# for i, max_depth in enumerate(max_depth_values):
#     for j, min_samples_leaf in enumerate(min_samples_leaf_values):
#         tree_model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
#         scores = cross_val_score(tree_model, X_train, y_train, cv=5)
#         cv_errors[i, j] = np.mean(scores)
#
# # Find the best combination
# best_combination_idx = np.unravel_index(np.argmin(cv_errors), cv_errors.shape)
# best_max_depth = max_depth_values[best_combination_idx[0]]
# best_min_samples_leaf = min_samples_leaf_values[best_combination_idx[1]]
# best_cv_error = cv_errors[best_combination_idx]
#
# # Plot the contour plot
# plt.figure(figsize=(8, 6))
# plt.contourf(max_depth_values, min_samples_leaf_values, cv_errors.T, cmap='viridis')
# plt.colorbar()
# plt.xlabel('Max Depth')
# plt.ylabel('Min Samples Leaf')
# plt.title('Cross-Validation Error vs. Max Depth and Min Samples Leaf')
#
# # Print the best combination and its CV error
# print("Best Max Depth:", best_max_depth)
# print("Best Min Samples Leaf:", best_min_samples_leaf)
# print("Best Cross-Validation Error:", best_cv_error)
#
# plt.show()
#
# xTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3xTrain.csv').to_numpy()
# yTrain = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3yTrain.csv').to_numpy().flatten()
# xTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3xTest.csv').to_numpy()
# yTest = pd.read_csv('/Users/apple/Desktop/2023FALL/hw1_template/q3yTest.csv').to_numpy().flatten()
# model_knn = KNeighborsClassifier(n_neighbors=1)  # # define with k = 1
# model_knn.fit(xTrain, yTrain)  # train the data
# y_true, y_pred = yTest, model_knn.predict(xTest)  # calculate accuracy
# acc_knn = []
# acc_knn.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_knn.predict_proba(xTest)  # calculate AUC
# fpr, tpr, thresholds = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_knn = []
# Auc_knn.append(metrics.auc(fpr, tpr))
#
# xTrain1, xTest1, yTrain1, yTest1 = train_test_split(  # Removed 5%
#     xTrain, yTrain, test_size=0.05)
# model_knn.fit(xTrain1, yTrain1)
#
# y_true, y_pred = yTest, model_knn.predict(xTest)  # calculate accuracy
# acc_knn.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_knn.predict_proba(xTest)  # calculate AUC
# fpr, tpr, thresholds2 = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_knn.append(metrics.auc(fpr, tpr))
#
# xTrain2, xTest2, yTrain2, yTest2 = train_test_split(  # removed 10%
#     xTrain, yTrain, test_size=0.10)
# model_knn.fit(xTrain2, yTrain2)
#
# y_true, y_pred = yTest, model_knn.predict(xTest)  # calculate accuracy
# acc_knn.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_knn.predict_proba(xTest)  # calculate AUC
# fpr, tpr, thresholds3 = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_knn.append(metrics.auc(fpr, tpr))
#
# xTrain3, xTest3, yTrain3, yTest3 = train_test_split(  # removed 20%
#     xTrain, yTrain, test_size=0.20)
# model_knn.fit(xTrain3, yTrain3)
#
# y_true, y_pred = yTest, model_knn.predict(xTest)  # calculate accuracy and AUC
# acc_knn.append(accuracy_score(y_true, y_pred))
# yHatTest = model_knn.predict_proba(xTest)
# fpr, tpr, thresholds4 = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_knn.append(metrics.auc(fpr, tpr))
#
# print(acc_knn)  # [0.903, 0.898, 0.906, 0.903]
# print(Auc_knn)  # [0.903, 0.8979999999999999, 0.9060000000000001, 0.9030000000000001]
#
# '''
# for the decision tree
# '''
# model_dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=3)  # define tree with parameters
# model_dt.fit(xTrain, yTrain)  # train the data
#
# acc_dt = []
# Auc_dt = []
#
# y_true, y_pred = yTest, model_dt.predict(xTest)  # calculate accuracy
# acc_dt.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_dt.predict_proba(xTest)
# fpr, tpr, threshold = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_dt.append(metrics.auc(fpr, tpr))
#
# model_dt.fit(xTrain1, yTrain1)  # removed 5%
# y_true, y_pred = yTest, model_dt.predict(xTest)  # calculate accuracy
# acc_dt.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_dt.predict_proba(xTest)  # calculate AUC
# fpr, tpr, threshold1 = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_dt.append(metrics.auc(fpr, tpr))
#
# model_dt.fit(xTrain2, yTrain2)  # removed 10%
# y_true, y_pred = yTest, model_dt.predict(xTest)  # calculate accuracy
# acc_dt.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_dt.predict_proba(xTest)  # calculate AUC
# fpr, tpr, threshold2 = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_dt.append(metrics.auc(fpr, tpr))
#
# model_dt.fit(xTrain3, yTrain3)  # removed 20%
# y_true, y_pred = yTest, model_dt.predict(xTest)  # calculate accuracy
# acc_dt.append(accuracy_score(y_true, y_pred))
#
# yHatTest = model_dt.predict_proba(xTest)  # calculate AUC
# fpr, tpr, threshold3 = metrics.roc_curve(yTest, yHatTest[:, 1])
# Auc_dt.append(metrics.auc(fpr, tpr))
#
# print(acc_dt) # [0.915, 0.913, 0.911, 0.904]
# print(Auc_dt) # [0.94983, 0.9503940000000001, 0.948764, 0.9398319999999999]
#
# conditions = ["Untouched", "5% Removed", "10% Removed", "20% Removed"]
# data = {
#     "k-NN Accuracy": acc_knn,
#     "k-NN AUC": Auc_knn,
#     "Decision Tree Accuracy": acc_dt,
#     "Decision Tree AUC": Auc_dt
# }
#
# # Create a DataFrame
# df = pd.DataFrame(data, index=conditions)
#
# # Display the DataFrame
# print(df)