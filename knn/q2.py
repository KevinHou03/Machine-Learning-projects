'''THIS CODE IS MY OWN WORK, IT WAS WRITTEN WITHOUT CONSULTING CODE
WRITTEN BY OTHER STUDENTS OR LARGE LANGUAGE MODELS SUCH AS CHATGPT.
Kaiyuan Hou
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets

iris_data = datasets.load_iris()
# print(iris_data)
print(iris_data.keys())
'''
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
'''


# print(iris_data['data'])
# print(np.shape(iris_data['data']))  # (150, 4) four attributes, 150 as sample size
# print(iris_data['target'])
# print(iris_data['frame'])
# print(iris_data['target_names'])  # ['setosa' 'versicolor' 'virginica']


# print(iris_data['DESCR'])

def load_iris():
    data_set = datasets.load_iris(as_frame=True).frame
    # data_set = pd.DataFrame(data=iris_data['data'], columns=iris_data['feature_names'])
    # data_set.insert(4, column='target', value=iris_data['target'])
    return data_set


print("load_iris:", load_iris())
print(load_iris().describe())  # describe method gives you general statistical information of each column
print()

all_dataset = pd.DataFrame(data=iris_data['data'], columns=iris_data['feature_names'])
all_dataset.insert(2, column='target', value=iris_data['target'])  # # 所有sepal length/width petal length/width data
sepal_length_data_set = all_dataset[["sepal length (cm)", 'target']]  # 单独取到其中的sepal length set 但这里的sepal length是三种花一起的
print("所有花的sepal的数据：", sepal_length_data_set)

sepal_length_data_set_setosa = sepal_length_data_set[sepal_length_data_set['target'] == 0]
print("第一种花的所有sepal长度数据", sepal_length_data_set_setosa)
sepal_length_data_set_versicolor = sepal_length_data_set[sepal_length_data_set['target'] == 1]
print("第二种花的所有sepal长度数据", sepal_length_data_set_versicolor)
sepal_length_data_set_virginica = sepal_length_data_set[sepal_length_data_set['target'] == 2]
print("第三种花的所有sepal长度数据", sepal_length_data_set_virginica)


def plot_sepal_length_boxplot():
    '''
    fig, ax = plt.subplots()
    ax.boxplot(sepal_length_data_set_setosa, )
    ax.boxplot(sepal_length_data_set_versicolor)
    ax.boxplot(sepal_length_data_set_virginica)
    plt.legend()
    plt.show()
    '''
    data_1 = sepal_length_data_set_setosa['sepal length (cm)']
    data_2 = sepal_length_data_set_versicolor['sepal length (cm)']
    data_3 = sepal_length_data_set_virginica['sepal length (cm)']
    data_set = [data_1, data_2, data_3]

    plt.boxplot(data_set, labels=['setoso', 'versicolor', 'virginica'])

    plt.xlabel('class')
    plt.ylabel('sepal length (cm)')
    plt.title('sepal length boxplots')

    # plt.legend()
    plt.show()


plot_sepal_length_boxplot()

sepal_width_data_set = all_dataset[['sepal width (cm)', 'target']]  # 取到所有sepal width，包括所有三种花
print("所有花的sepal width 的数据：", sepal_width_data_set)
sepal_width_data_set_setosa = sepal_width_data_set[sepal_width_data_set['target'] == 0]
print("第一种花的所有sepal宽度数据", sepal_width_data_set_setosa)
sepal_width_data_set_versicolor = sepal_width_data_set[sepal_width_data_set['target'] == 1]
print("第二种花的所有sepal宽度数据", sepal_width_data_set_versicolor)
sepal_width_data_set_virginica = sepal_width_data_set[sepal_width_data_set['target'] == 2]
print("第三种花的所有sepal宽度数据", sepal_width_data_set_virginica)


def plot_sepal_width_boxplot():
    data_1 = sepal_width_data_set_setosa['sepal width (cm)']
    data_2 = sepal_width_data_set_versicolor['sepal width (cm)']
    data_3 = sepal_width_data_set_virginica['sepal width (cm)']
    data_set = [data_1, data_2, data_3]

    plt.boxplot(data_set, labels=['setoso', 'versicolor', 'virginica'])

    plt.xlabel('class')
    plt.ylabel('sepal width (cm)')
    plt.title('sepal width boxplots')

    # plt.legend()
    plt.show()


plot_sepal_width_boxplot()

petal_length_data_set = all_dataset[['petal length (cm)', 'target']]  # 取到所有sepal width，包括所有三种花
print("所有花的petal length 的数据：", petal_length_data_set)
petal_length_data_set_setosa = petal_length_data_set[petal_length_data_set['target'] == 0]
print("第一种花的所有petal length数据", petal_length_data_set_setosa)
petal_length_data_set_versicolor = petal_length_data_set[petal_length_data_set['target'] == 1]
print("第二种花的所有petal length数据", petal_length_data_set_versicolor)
petal_length_data_set_virginica = petal_length_data_set[petal_length_data_set['target'] == 2]
print("第三种花的所有petal length数据", petal_length_data_set_virginica)


def plot_petal_length_boxplot():
    data_1 = petal_length_data_set_setosa['petal length (cm)']
    data_2 = petal_length_data_set_versicolor['petal length (cm)']
    data_3 = petal_length_data_set_virginica['petal length (cm)']
    data_set = [data_1, data_2, data_3]

    plt.boxplot(data_set, labels=['setoso', 'versicolor', 'virginica'])

    plt.xlabel('class')
    plt.ylabel('petal length (cm)')
    plt.title('petal length boxplots')

    # plt.legend()
    plt.show()


plot_petal_length_boxplot()

petal_width_data_set = all_dataset[['petal width (cm)', 'target']]  # 取到所有sepal width，包括所有三种花
print("所有花的petal width 的数据：", petal_width_data_set)
petal_width_data_set_setosa = petal_width_data_set[petal_width_data_set['target'] == 0]
print("第一种花的所有petal width数据", petal_width_data_set_setosa)
petal_width_data_set_versicolor = petal_width_data_set[petal_width_data_set['target'] == 1]
print("第二种花的所有petal width数据", petal_width_data_set_versicolor)
petal_width_data_set_virginica = petal_width_data_set[petal_width_data_set['target'] == 2]
print("第三种花的所有petal width数据", petal_width_data_set_virginica)


def plot_petal_width_boxplot():
    data_1 = petal_width_data_set_setosa['petal width (cm)']
    data_2 = petal_width_data_set_versicolor['petal width (cm)']
    data_3 = petal_width_data_set_virginica['petal width (cm)']
    data_set = [data_1, data_2, data_3]

    plt.boxplot(data_set, labels=['setoso', 'versicolor', 'virginica'])

    plt.xlabel('class')
    plt.ylabel('petal width (cm)')
    plt.title('petal width boxplot')

    # plt.legend()
    plt.show()


plot_petal_width_boxplot()

'''
sepal_length_data_set
sepal_length_data_set_setosa - red
sepal_length_data_set_versicolor - blue 
sepal_length_data_set_virginica - green


sepal_width_data_set
sepal_width_data_set_setosa - red
sepal_width_data_set_versicolor - blue
sepal_width_data_set_virginica - green
'''


def scatter_plot_sepal():  # with the length on the x-axis and the width on the y-axis
    fig, ax = plt.subplots(1, figsize=(10, 8))

    plt.scatter(sepal_length_data_set_setosa, sepal_width_data_set_setosa, label="Setosa", color="red")
    plt.scatter(sepal_length_data_set_versicolor, sepal_width_data_set_versicolor, label="Versicolor", color="blue")
    plt.scatter(sepal_length_data_set_virginica, sepal_width_data_set_virginica, label="Virginica", color="green")


    plt.tight_layout()
    plt.title('scatter_plot_sepal')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    # plt.legend()
    print(plt.gcf())
    plt.show()


scatter_plot_sepal()


def scatter_plot_petal():  # with the length on the x-axis and the width on the y-axis
    fig, ax = plt.subplots(1)
    plt.scatter(petal_length_data_set_setosa, petal_width_data_set_setosa, label="Setosa", color="red")
    plt.scatter(petal_length_data_set_versicolor, petal_width_data_set_versicolor, label="Versicolor", color="blue")
    plt.scatter(petal_length_data_set_virginica, petal_width_data_set_virginica, label="Virginica", color="green")
    plt.tight_layout()
    plt.title('scatter_plot_petal')
    plt.xlabel('Length')
    plt.ylabel('Width')
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)
    # plt.legend()
    plt.show()


scatter_plot_petal()
