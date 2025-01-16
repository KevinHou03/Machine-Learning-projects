import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 创建一个示例DataFrame
data = {'date_column': ["1/11/16 17:00", "1/12/16 18:30", "1/13/16 19:45"]}
df = pd.DataFrame(data)

# 使用pd.to_datetime将列转换为日期时间格式
date = df['date_column']
date = pd.to_datetime(date, format='%m/%d/%y %H:%M')
df['date_column'] = date

year = date.dt.year
hour = date.dt.hour
minute = date.dt.minute

# time = hour.astype(str) + ':' + minute.astype(str).str.zfill(2)
time = hour + minute
df.insert(0, 'time', time)
# df.insert(1,'year',year)
df.insert(2, 'hour', hour)

# 输出转换后的DataFrame
print(df)
print(year)
print(hour)
print(minute)
print(time)

df_co = df.corr()
print(df_co)

# # Create a heatmap using Seaborn
# plt.figure(figsize=(8, 6))  # Set the size of the heatmap
# sns.heatmap(df_co, annot=True, cmap='coolwarm', fmt=".2f")  # annot=True to show correlation values, cmap for color map, fmt to format the values
#
# plt.title('Correlation Matrix Heatmap')  # Add a title to the heatmap
# plt.show()

corr_feat_target = df_co.iloc[:, -1]
print('corr_feat_target', corr_feat_target)
features_to_remove1 = corr_feat_target[abs(corr_feat_target) < 1].index
print('features_to_remove1', features_to_remove1)
# print(corr_feat_target)

# for index, element in enumerate(corr_feat_target):
#     print(index,element)
#     if element < 1:
#         column_name = df.columns[index]
#         df.drop(column_name)

beta = [1, 1, 1]
feature = np.array([
    [1, 2, 1],
    [2, 3, 1],
    [3, 4, 1]])

list1 = np.dot(feature, beta)
print(list1)

feature2 = np.array([
    [1, 2, 1, 1],
    [2, 3, 1, 1],
    [3, 4, 1, 1]])
ones = np.ones((feature2.shape[0],1))
feature2 = np.hstack((feature2,ones))
print(feature2)

ones2 = np.ones(feature2.shape[0])
print(ones2) # [1. 1. 1.]
ones3 = np.ones((feature2.shape[0],1))
print(ones3)
'''
这两种用法不一样哈
[[1.]
 [1.]
 [1.]]
'''

bs = 2
xTrain = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],[4,5],[6,4],[2,3]])
np.random.shuffle(xTrain)
print(xTrain)
num_batches = xTrain.shape[0] / bs
print(num_batches)
batches = []
for j in range(0,xTrain.shape[0],bs): # 1234?
    batches.append(xTrain[j:j+bs])

print(np.array(batches))


my_list = [1, 2, 3]
result = my_list * 2

a = [4,6,7,8]
for i in a:
    print(i)