#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : KNN
# @File     : knn.py
# @Software : PyCharm
# @Date     : 2018/8/30 下午11:45
# @Descrip  : 
# @License  : Copyright(c) Lichun Zhang. 
# @Contact  : lczhang2011@gmail.com
from os import listdir

import numpy as np


def create_simple_data():
    """
    创建简单的数据
    :return:
    """
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def file2matrix(file_name):
    """
    解析, 将文本记录转换为Numpy矩阵
    :param file_name: 输入文件名，前三列为特征，最后一列为分类标签
    :return: 特征数据矩阵和分类标签
    """
    with open(file_name) as fr:
        lines_arr = fr.readlines()
    num_lines = len(lines_arr)
    res_mat = np.zeros((num_lines, 3))
    class_labels = []
    index = 0
    for line in lines_arr:
        data_list = line.strip().split('\t')
        res_mat[index, :] = data_list[0:3]
        # 仅适用于分类标签是数字的数据
        class_labels.append(int(data_list[-1]))
        index += 1
    return res_mat, class_labels


def img2vector(file_name):
    """
    将输入的图像文件转换为行向量
    :param file_name: 图像文件的地址
    :return: 转换后得到的行向量
    """
    fr = open(file_name)
    lines_origin = fr.readlines()
    fr.close()
    lines = [line.strip() for line in lines_origin]
    num_row = len(lines)
    num_col = len(lines[0])
    res_vec = np.zeros((1, num_row * num_col))
    for i in range(num_row):
        for j in range(num_col):
            res_vec[0, i * num_col + j] = int(lines[i][j])
        # res_vec[0, num_col * i:num_col * (i + 1)] = [lines[i][:]]
    return res_vec


def auto_norm(data_set):
    """
    数值归一化, 将取值范围处理为0-1间
    :param data_set: 待处理的数据
    :return: 归一化后的数据, 各特征的范围, 各特征的最小值,
    """
    # 从列中选取最小值
    min_vals = data_set.min(0)
    max_val = data_set.max(0)
    ranges = max_val - min_vals
    norm_data_set = np.zeros(np.shape(data_set))
    num_data = data_set.shape[0]
    norm_data_set = data_set - min_vals
    norm_data_set /= ranges
    return norm_data_set, ranges, min_vals


def classify0(in_vector, data_set, labels, num_neighbour):
    """
    KNN分类
    :param in_vector: 输入数据, 向量
    :param data_set: 训练数据集, 矩阵
    :param labels: 训练数据集的分类标签
    :param num_neighbour: 进行分类是参考的最近点的个数
    :return: 分类结果
    """
    num_data = data_set.shape[0]
    diff_mat = np.tile(in_vector, (num_data, 1)) - data_set
    diff_sqr_mat = diff_mat ** 2
    distance_sqr = diff_sqr_mat.sum(axis=1)
    distance = distance_sqr ** 0.5
    sorted_dist_indicies = distance.argsort()
    class_count = {}
    # 选择距离最小的k个点
    for i in range(num_neighbour):
        label_voted = labels[sorted_dist_indicies[i]]
        class_count[label_voted] = class_count.get(label_voted, 0) + 1
    sorted_class_count = sorted(class_count.items(), key=lambda d: d[1], reverse=True)
    return sorted_class_count[0][0]


def dating_class_test(test_ratio):
    """
    应用实例, 针对约会网站的数据进行测试
    :param test_ratio: 数据用来进行测试的比例
    :return: 测试的错误率
    """
    date_mat, date_labels = file2matrix('datingTestSet2.txt')
    norm_mat, ranges, min_vals = auto_norm(date_mat)
    num_data = norm_mat.shape[0]
    # test_ratio = 0.1
    num_test = int(num_data * test_ratio)
    count_err = 0.0
    for i in range(num_test):
        class_label = classify0(norm_mat[i, :], norm_mat[num_test:num_data, :], \
                                date_labels[num_test:num_data], 3)
        print("the classifer came back with %d, the real answer is %d" \
              % (class_label, date_labels[i]))
        if (class_label != date_labels[i]):
            count_err += 1.0
    print("the total error rate is %f" % (count_err / float(num_test)))
    return count_err / float(num_test)


def classify_person():
    """
    应用实例, 通过用户输入的数据使用KNN算法分类
    :return: 分类的结果(标签列表中的序号)
    """
    labels_list = ['not at all', 'in small dose', 'in large doses']
    time_percent = float(input("percentage of time spent playing video games?"))
    miles_fly = float(input("frequent flier miles earned per year?"))
    icecream = float(input("liters of ice cream consumed per year?"))
    in_arr = np.array([miles_fly, time_percent, icecream])
    dating_data, dating_labels = file2matrix('datingTestSet2.txt')
    nore_mat, ranges, min_vals = auto_norm(dating_data)
    result = classify0((in_arr - min_vals) / ranges, nore_mat, dating_labels, 3)
    print("You will probably like this person:", labels_list[result - 1])
    return result


def writing_class_test():
    """
    使用knn训练和测试手写数字
    :return: 测试的错误率
    """
    labels = []
    training_list = listdir('digits/trainingDigits')
    num_data = len(training_list)
    train_mat = np.zeros((num_data, 1024))
    for i in range(num_data):
        # 从文件名解析分类的数字
        file_name = training_list[i]
        file_str = file_name.split('.')[0]
        class_label = int(file_str.split('_')[0])
        labels.append(class_label)
        train_mat[i, :] = img2vector('digits/trainingDigits/%s' % (file_name))
    test_list = listdir('digits/testDigits')
    count_err = 0
    m_test = len(test_list)
    for i in range(m_test):
        file_name = test_list[i]
        file_str = file_name.split('.')[0]
        class_label = int(file_str.split('_')[0])
        data_test = img2vector('digits/testDigits/%s' % (file_name))
        class_result = classify0(data_test, train_mat, labels, 3)
        # print("the classifer came back with %d, the real answer is: %d" \
        #       % (class_result, class_label))
        if (class_result != class_label):
            count_err += 1
    print("\nthe total number of errors is: %d" % (count_err))
    print("\nthe total error rate is:%f" % (count_err / float(m_test)))
    return (count_err / float(m_test))
