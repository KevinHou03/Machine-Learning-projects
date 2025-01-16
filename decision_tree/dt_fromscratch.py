import operator
from math import log

"""
dataset数据集，我们会递归的调用create_tree所以每一次传进去的dataset都不一样
第一次进去的是all dataset,选用一个特征作为根节点之后，第二次传进去的就是除去该
特征后剩下的特征了。

labels来判断purity,如果有一次分完后所有label都一样,那就停止，因为已经全部分好了
labels参数是一个存放feature name的列表，和dataset完全一一对应


featLabels, 按顺序排列放进node的feature，第一个是谁第二个是谁

"""
import operator


def create_tree(dataset, labels, featLabels):
    # 第一步，判断一个节点里所有标签是否一样，如果一样，purity拉满，不用再分了，先把所有标签存到classList里
    # creates a list called classList by extracting the last element from each example in the dataset list.
    classList = [example[-1] for example in dataset]  # No No Yes Yes Yes No.....
    # 先写出所有都一样的情况,这种情况可以结束了
    if classList.count(classList[0]) == len(classList):  # No No No No....No // or Yes Yes Yes ...Yes
        return classList[0]
    # 每把一个特征放进树里，就从原dataset去掉该特征，用剩下的特征递归的建树，当原dataset没有特征时，停止
    if len(dataset[0]) == 1:  # 所有特征都移除了
        return majority_count(classList)  # 该函数计算节点中出现次数最多的变量
    # 如果可以继续分： 继续选择最优特征
    # 下面这个best_feat得到的会是一个索引值
    best_feat = choose_best_feature_to_split(dataset)  # 此时的dataset可能已经去除一些feature了
    best_feat_label = labels[best_feat]  # 利用这个索引值找到这个feature的实际值是什么
    featLabels.append(best_feat_label)
    # my_tree是一个字典，key-value中key是目前选中的最好的feature，对应的value是里面的feature里面的012..所有可能的取值
    my_tree = {best_feat_label: {}}
    # 做完一个feature后，把它删掉
    del labels[best_feat]

    # 接下来，对于任意一个feature而言，看有多少种情况，有n种情况，那么那一个节点就要分n岔,分裂
    feat_values = [example[best_feat] for example in dataset]  # 取到feature那一列
    unique_value = set(feat_values)  # set构建集合，使所有value唯一 ： 11122233334 = 1234
    # 要分几条岔？ unique value这么多
    for value in unique_value:
        sub_labels = labels[:]
        # my_tree[best_feat_label][value] 就意味着对该feature的每一个可能的value做一个分支，每一个分支都要建一个树
        my_tree[best_feat_label][value] = create_tree(split_data_set(dataset, best_feat, value), sub_labels, featLabels)
    return my_tree  # 字典结构


# split_data_set就是把已选中的列去掉，无他


def majority_count(classList):  # 计算该节点哪一个类是最多的
    # majority_count 函数用于计算在一个节点内出现次数最多的类别（标签）
    class_count = {}  # 一个字典{a:b} a对应feature里的一个元素，b是他出现的次数
    for vote in classList:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]  # 返回出现最多的变量


'''
在信息增益的计算中，通常会有一个基础熵（Base Entropy），它是由整个训练数据集中的类别分布计算得出的熵值。
然后，针对每个特征，计算在该特征条件下的加权平均熵（Conditional Entropy），并将基础熵减去条件熵，得到信息增益。
信息增益表示了使用特定特征进行分割后，不确定性的减少程度，即分类的纯度提高了多少。
'''


def choose_best_feature_to_split(dataset):
    num_feat = len(dataset[0]) - 1  # 计算目前feature的数量 要减去最后一列把
    # 先找到原始数据的entropy
    base_entropy = calc_entropy(dataset)
    base_info_gain = 0
    best_info_gain = 0  # 最好的信息增益
    best_feat = -1  # index
    for i in range(num_feat):  # 第一个循环，遍历每一个特征 i = 0,1,2,3...
        # 一列一列拿到feature
        feat_list = [example[i] for example in dataset]
        unique_val = set(feat_list)  # 得到个数
        new_entropy = 0  # 这是针对每一个新特征的new entropy
        for val in unique_val:  # 第二个循环，遍历特征中每一个值，并计算每一个该特征内每一个值的entropy
            # 在这里你的val已经取到特征内每一个属性了
            sub_dataset = split_data_set(dataset, i, val)
            # 对于每一个属性会有一个分类，调用split函数 切分第i个特征into 他对应的所有属性，并计算每种情况的entropy
            # val对应的是该 属性！而sub_dataset是所有等于val的row的list
            probability = len(sub_dataset) / float(len(dataset))
            # 对于feature的entropy就等于 sum(其中每一个属性对应的概率*该属性对应的entropy)
            new_entropy += probability * calc_entropy(sub_dataset)
        # 算信息增益的结果,放在第一个循环内，因为每一个特征都要算一个entropy取最小值
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:  # 信息增益增加了，是我们想要看到的结果
            best_info_gain = info_gain  # 直接和最好的比
            best_feat = i
    return best_feat  # 返回feature的索引


def split_data_set(dataset, axis, val):
    ret_data_set = []
    for feat_vector in dataset:  # 拿到每一个特征对应的yesyes/nono，再把这一特征去掉
        if feat_vector[axis] == val:
            reduced_feat_vector = feat_vector[:axis]
            reduced_feat_vector.extend(feat_vector[axis + 1:])
            ret_data_set.append(reduced_feat_vector)
    return ret_data_set


def calc_entropy(dataset):
    num_examples = len(dataset)  # 所有test/train样本的个数
    label_count = {}  # 把yes和no传进字典，一一对应
    for feat_vector in dataset:  # 每一行
        current_label = feat_vector[-1]  # label 会是最后一个值
        if current_label not in label_count.keys():
            label_count[current_label] = 0  # 如果不在则创造一个这样的key并且对应的value为0
        label_count[current_label] += 1  # 现在我知道每个class有多少个了

    shannon_entropy = 0
    for key in label_count:
        probability = float(label_count[key] / num_examples)
        shannon_entropy -= probability * log(probability, 2)  # entropy计算出来了，是负的
    return shannon_entropy
