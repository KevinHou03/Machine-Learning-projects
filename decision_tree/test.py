import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Parameter grid for k-NN
knn_param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}

# Parameter grid for decision tree
dt_param_grid = {'max_depth': [1, 3, 5, 7, 9], 'min_samples_leaf': [1, 2, 3, 4, 5]}

def cross_val_error(model, X, y, cv):
    cv_errors = []
    for train_index, val_index in cv.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        cv_errors.append(1 - accuracy_score(y_val, y_pred))
    return np.mean(cv_errors)

# Initialize k-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize k-NN and Decision Tree models
knn = KNeighborsClassifier()
dt = DecisionTreeClassifier()

# Initialize GridSearchCV with cross-validation and parameter grids
knn_grid_search = GridSearchCV(knn, knn_param_grid, cv=kf)
dt_grid_search = GridSearchCV(dt, dt_param_grid, cv=kf)

# Fit the models to the data
knn_grid_search.fit(X, y)
dt_grid_search.fit(X, y)

# Get the best hyperparameters
best_knn_params = knn_grid_search.best_params_
best_dt_params = dt_grid_search.best_params_

# Create a mesh grid of hyperparameters
depth_range = [1, 3, 5, 7, 9]
min_samples_leaf_range = [1, 2, 3, 4, 5]
depth_range, min_samples_leaf_range = np.meshgrid(depth_range, min_samples_leaf_range)

# Calculate cross-validation error for each combination of hyperparameters
cv_errors = np.zeros_like(depth_range, dtype=float)
for i in range(depth_range.shape[0]):
    for j in range(depth_range.shape[1]):
        depth = depth_range[i, j]
        min_samples_leaf = min_samples_leaf_range[i, j]
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=min_samples_leaf)
        cv_errors[i, j] = cross_val_error(dt, X, y, kf)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(depth_range, min_samples_leaf_range, cv_errors, cmap='viridis')
ax.set_xlabel('Max Depth')
ax.set_ylabel('Min Samples Leaf')
ax.set_zlabel('CV Error')
plt.title('Decision Tree Cross-Validation Error')
plt.show()

'''
    y_value_train = []
    y_value_test = []

    for epoch in range(1, 100, 10):
        model = SgdLR(0.0001, 1, epoch)
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
        last_key = list(trainStats.keys())[-1]
        last_value = trainStats[last_key]
        train_mse = last_value['train-mse']
        test_mse = last_value['test-mse']

        y_value_train.append(train_mse)
        y_value_test.append(test_mse)
        print(f"Epoch {epoch} - Train MSE: {train_mse}, Test MSE: {test_mse}")

    x_value = np.linspace(0.1, 100, 10)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)  # Create a subplot for training MSE
    plt.plot(x_value, y_value_train)
    plt.xlabel('Epoch')
    plt.ylabel('Train MSE')
    plt.title('Training MSE vs. Epoch')

    # Plot test MSE
    plt.subplot(1, 2, 2)  # Create a subplot for test MSE
    plt.plot(x_value, y_value_test)
    plt.xlabel('Epoch')
    plt.ylabel('Test MSE')
    plt.title('Test MSE vs. Epoch')

    plt.tight_layout()  # Adjust subplots for better visualization
    plt.show()
'''
