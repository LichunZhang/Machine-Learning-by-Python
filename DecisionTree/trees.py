#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : DecisionTree
# @File     : trees.py
# @Software : PyCharm
# @Date     : 2018/8/26 上午10:37
# @Descrip  : 创建决策树, ID3算法
# @License  : Copyright(c) Lichun Zhang. 
# @Contact  : lczhang2011@gmail.com
import operator
from math import log


def load_simple_data():
    """
    加载预设好的简单数据
    :return: 数据以及标签
    """
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def calc_info_entropy(data_set):
    """
    计算给定数据集的信息熵, 度量样本集合的纯度. 值越小, 纯度越高
    :param data_set: 输入的数据集，一个数据占一行, 最后一列为分类标签
    :return: 数据集的信息熵
    """
    num_data = len(data_set)
    label_count_dic = {}
    for feat_vec in data_set:
        # 最后一列为分类标签
        label = feat_vec[-1]
        if label not in label_count_dic.keys():
            label_count_dic[label] = 0
        label_count_dic[label] += 1
    entropy = 0
    # 某符号xi的信息: -log2p(xi)
    # 信息期望: -[求和到xn]p(xi)log2p(xi)
    for key in label_count_dic:
        prob = float(label_count_dic[key]) / num_data
        entropy -= prob * log(prob, 2)
    return entropy


def split_data_set(data_set, axis, value):
    """
    依照某一个维度的特征取值对数据集进行划分
    :param data_set: 待划分的数据集
    :param axis: 划分特征所在的维度
    :param value:特征划分的取值
    :return: 符合特征划分取值，同时去除了相应特征的数据的集合
    """
    splited_data = []
    for ft_vec in data_set:
        if ft_vec[axis] == value:
            reduced_data = ft_vec[:axis]
            reduced_data.extend(ft_vec[axis + 1:])
            splited_data.append(reduced_data)
    return splited_data


def choose_best_ft(data_set):
    """
    通过选取特征，划分数据集，通过比较划分之后的信息增益，得出最好的划分数据集的特征
    :param data_set:待划分数据集
    :return:能最好划分数据集特征所在维度
    """
    num_ft = len(data_set[0]) - 1
    base_entropy = calc_info_entropy(data_set)
    best_info_gain = 0.0
    best_ft = -1
    # 第一层循环：对每个特征维度
    for i in range(num_ft):
        # 获得数据集上所有数据在某一维度的特征值
        ft_list = [example[i] for example in data_set]
        # 去重，保留唯一值
        unique_ft_vals = set(ft_list)
        new_entropy = 0.0
        # 第二层循环：用当前维度属性的所有可能取值来划分数据集
        for val in unique_ft_vals:
            split_data = split_data_set(data_set, i, val)
            prob = len(split_data) / float(len(data_set))
            # 计算以此属性划分之后的数据集信息商, 可查阅<<机器学习 周志华>>p75-p77
            new_entropy += prob * calc_info_entropy(split_data)
        info_gain = base_entropy - new_entropy
        # 计算最好的信息增益,值越大, 意味用i维度的属性来进行划分所获得的"纯度提升越大"
        if (info_gain > best_info_gain):
            best_info_gain = info_gain
            best_ft = i
    return best_ft


def majority_cnt(class_list):
    """

    :param class_list:
    :return:
    """
    # 字典，统计类别出现次数
    class_count = {}
    for vote in class_list:
        if vote not in class_count.keys():
            class_count[vote] = 0
        class_count[vote] += 1
    # 根据类别出现次数从大到小排列
    sorted_class_cnt = sorted(class_count.items(), key = lambda d: d[1], reverse=True)
    # 返回出现次数最多的分类
    return sorted_class_cnt[0][0]


def create_tree(data_set, labels):
    """
    创建单棵决策树, 递归法
    :param data_set: 待分类数据
    :param labels: 特征的名字
    :return: 创建好的决策树, 字典的形式
    """
    class_list = [example[-1] for example in data_set]
    # 递归停止条件1: 类别完全相同, 停止继续划分
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 递归停止条件2: 遍历完所有特征时(数据只剩下类别标签了), 返回出现次数最多的类别
    if len(data_set[0]) == 1:
        return majority_cnt(class_list)
    # 在当前数据集中选取最好的特征
    best_ft_index = choose_best_ft(data_set)
    best_ft_label = labels[best_ft_index]
    my_tree = {best_ft_label: {}}
    # 因为choose_best_ft之后去除了相应列的特征, 因此标签列表也得调整
    del (labels[best_ft_index])
    # 得到最佳特征上的所有属性值
    ft_value = [example[best_ft_index] for example in data_set]
    unique_vals = set(ft_value)
    for value in unique_vals:
        sub_labels = labels[:]
        # 递归
        my_tree[best_ft_label][value] = create_tree( \
            split_data_set(data_set, best_ft_index, value), \
            sub_labels)
    return my_tree


def classify(input_tree, ft_labels, test_vec):
    """
    使用决策树来分类
    :param input_tree: 训练好的决策树
    :param ft_labels: 特征的标签(名字)
    :param test_vec: 一个需要分类的数据(只包含特征值)
    :return: 数据的类别标签
    """
    first_str = list(input_tree.keys())[0]
    second_dict = input_tree[first_str]
    # 将标签字符串转换为索引
    ft_index = ft_labels.index(first_str)
    for key in second_dict.keys():
        if test_vec[ft_index] == key:
            if (type(second_dict[key]).__name__ == 'dict'):
                class_label = classify(second_dict[key], ft_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


def store_tree(tree, file_name):
    """
    使用pickle序列化决策树进行存储
    :param tree: 需要存储的决策树
    :param file_name: 存储文件名字
    :return:
    """
    import pickle
    with open(file_name, 'wb') as fw:
        pickle.dump(tree, fw)


def grab_tree(file_name):
    """
    使用pickle读取序列化存储的决策树
    :param file_name: 存储有决策树的文件名
    :return: 存储的决策树
    """
    import pickle
    with open(file_name, 'rb') as fr:
        return pickle.load(fr)
