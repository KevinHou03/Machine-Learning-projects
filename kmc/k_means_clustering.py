import numpy as np
import matplotlib.pyplot as plt
from numpy import argwhere
'''
Step 1:初始化
随机在数据分布范围内生成k个聚类中心(centroids）

Step 2:分配数据点
对于每个数据点：
计算它到所有聚类中心的距离（euclidean）
将它分配到距离最近的中心

Step 3:更新聚类中心
对每个聚类：
计算当前簇中所有点的平均位置
这个平均点就是新的聚类中心

Step 4:重复直到收敛
重复 分配点+更新中心的过程直到：
聚类中心不再变化或
达到最大迭代次数或
聚类中心移动非常小(如 < 0.001）
'''

class KMeansClustering():

    def __init__(self, k = 3):
        self.k = k
        self.centroids = None

    def euclidian_distance(self, data_points, centroid):
         return np.sqrt(np.sum((data_points - centroid)**2, axis=1))


    def fit(self, X, max_iterations=300):
        # 随机生成centroids，同时保证随机生成的centroid在data range以内
        global y
        self.centroids = np.random.uniform(np.amin(X, axis = 0),
                                           np.amax(X, axis = 0),
                                           size = (self.k, X.shape[1])) #在每个特征维度的最小值到最大值之间，均匀随机地生成 k 个初始聚类中心向量
        for _ in range(max_iterations):
            y = [] # y里面放的是每个样本所属的聚类编号cluster index

            for data_point in X: # 对于每一个data point我都找到最近的那个centroid，并且放到y里面去
                # calculate the distance of every point to ALL the centroids
                distances = self.euclidian_distance(data_point, self.centroids) # list ， 计算它到每个质心的距离
                cluster_num = np.argmin(distances) # return the index of the min value找到距离最近的质心索引
                y.append(cluster_num)# 把这个样本分配到该簇

            y = np.array(y)
            cluster_indices = []#创建一个列表来保存每个簇对应的索引。

            for i in range(self.k):
                cluster_indices.append(argwhere(y == i))# 找出所有属于第i个簇的样本索引

            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis = 0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) <= 0.01:
                break
            else:
                self.centroids = np.array(cluster_centers)
        return y


random_points = np.random.randint(0, 100, (100, 2))
kmeans = KMeansClustering(k = 5)
labels = kmeans.fit(random_points)
plt.scatter(random_points[:, 0], random_points[:, 1], c = labels)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c = range(len(kmeans.centroids)),
            marker = '*', s = 200)
plt.show()


