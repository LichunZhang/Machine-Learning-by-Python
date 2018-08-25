#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : Adaboost
# @File     : adaboost.py
# @Software : PyCharm
# @Date     : 2018/8/22 下午7:00
# @Descrip  :
# @License  : Copyright(c) Lichun Zhang.
# @Contact  : lczhang2011@gmail.com
from math import log
import numpy as np


def load_simple_data():
    """
    导入简单的数据集及类标签
    :return: data_mat, labels_arr
    """
    data_mat = np.matrix([[1., 2.1], [2., 1.1], [1.3, 1.], [1., 1.], [2., 1.]])
    labels_arr = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_mat, labels_arr


def load_data_set(f_name):
    """
    从文件中加载数据(特征+标签)
    :param f_name: 文件名
    :return: 分类数据矩阵，标签数据矩阵
    """
    num_ft = len(open(f_name).readline().split('\t')) - 1
    data_mat = []
    label_mat = []
    data = open(f_name)
    # 行代表一个数据，列代表特征值，最后一列代表标签. 一行中的数据以'\t'隔开.
    for line in data.readlines():
        ft_arr = []
        line_dat = line.strip().split('\t')
        for i in range(num_ft):
            ft_arr.append(float(line_dat[i]))
        data_mat.append(ft_arr)
        label_mat.append(float(line_dat[-1]))
    return data_mat, label_mat


def stump_classify(data_mat, dim, thresh, thresh_ineq):
    """
    使用单层决策树（桩）通过阈值比较对数据进行分类
    :param data_mat:    待分类数据
    :param dim:         数据中进行阈值比较的某一维度
    :param thresh:      阈值
    :param thresh_ineq: 分类的运算标准（大于或小于）
    :return: res_arr  返回阈值分类后的类标签
    """
    res_arr = np.ones((np.shape(data_mat)[0], 1))
    if thresh_ineq == 'lt':
        res_arr[data_mat[:, dim] <= thresh] = -1.0
    else:
        res_arr[data_mat[:, dim] > thresh] = -1.0
    return res_arr


def build_stump(data_arr, labels_arr, w_data):
    """
    弱分类器，在数据集上找到最佳的单层决策树(桩)
    :param data_arr:    输入数据集（数组）
    :param labels_arr:  输入的类标签 （数组）
    :param w_data:      数据的权重向量
    :return: 返回得到的最佳分类器设置，最小加权误差以及预测结果
    """
    data_mat = np.mat(data_arr)
    labels_mat = np.mat(labels_arr).T
    num_data, num_features = np.shape(data_mat)
    # 预设搜索步数为10
    num_steps = 10.0
    # 存储得到的最佳单层决策树的信息, 字典类型
    best_stump = {}
    # 存储得到的最佳分类器的预测结果
    best_estimation = np.mat(np.zeros((num_data, 1)))
    min_error = np.inf
    # 第一层循环: 对数据集中的每个维度特征
    for i in range(num_features):
        range_min = data_mat[:, i].min()
        range_max = data_mat[:, i].max()
        step_size = (float)(range_max - range_min) / num_steps
        # 第二层循环: 对每个步长
        for j in range(-1, int(num_steps) + 1):
            # 第三层循环: 对每个运算符号
            # TODO: 这里两种运算符号得到的误差结果和为1，后期可以不用对每个运算符号进行运算，省略第三层循环
            for ineq in ['lt', 'gt']:
                thresh = (range_min + j * step_size)
                # 调用单层决策树，在当前步长下对数据的当前维度进行阈值分类
                predict_arr = stump_classify(data_mat, i, thresh, ineq)
                err_mat = np.mat(np.ones((num_data, 1)))
                err_mat[predict_arr == labels_mat] = 0
                # 这里并非是除以所有样本数目(平分)，而是考虑了权重. 和书中原理部分描述不一.
                weighted_error = w_data.T * err_mat
                # print("split: dim %d, thres: %.2f, thresh inequal: %s, the weighted error is % .3f" %\
                #    (i, thresh, ineq, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_estimation = predict_arr.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh
                    best_stump['ineq'] = ineq
    return best_stump, min_error, best_estimation


def train(data_arr, labels_arr, num_iter=40):
    """
    利用弱分类器，进行adaboost的迭代运算，即训练, 得到的一系列弱分类器可以构成强分类器
    :param data_arr:    输入数据
    :param labels_arr:  输入的类别标签
    :param num_iter:    设定的迭代次数
    :return:            返回迭代过程中使用到的弱分类器（一起构成强分类器）
    """
    weak_classifier_arr = []
    num_data = np.shape(data_arr)[0]
    w_data = np.mat(np.ones((num_data, 1)) / num_data)
    class_est_agg = np.mat(np.zeros((num_data, 1)))
    for i in range(num_iter):
        best_stump, error, class_est = build_stump(data_arr, labels_arr, w_data)
        # print("Data Weight:", w_data.T)
        # 根据错误率更新分类器的权重alpha, 同时确保分母在error为0时不发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 在弱分类器字典中包含分类器的权重信息alpha
        best_stump['alpha'] = alpha
        weak_classifier_arr.append(best_stump)
        # print("Class Estimation:", class_est.T)
        # 为下一次迭代更新分类器中的数据权重
        expon = np.multiply(-1 * alpha * np.mat(labels_arr).T, class_est)
        w_data = np.multiply(w_data, np.exp(expon))
        w_data = w_data / w_data.sum()
        # 记录在本轮迭代后，每个数据的类别估计累加值
        class_est_agg += alpha * class_est
        # print("Aggregation Class Estimation:", class_est_agg.T)
        # 错误率累加计算
        error_agg = np.multiply(np.sign(class_est_agg) != np.mat(labels_arr).T, np.ones((num_data, 1)))
        error_rate = error_agg.sum() / num_data
        print("Total Error:", error_rate)
        # 如果错误率为0，则提前停止迭代
        if error_rate == 0.0:
            break
    return weak_classifier_arr, class_est_agg


def classify(data_arr, classifier_arr):
    """
    利用训练出来的多个弱分类器进行分类
    :param data_arr:        待分类的数据
    :param classifier_arr:  弱分类器数组
    :return: 累加之后的类别估计值
    """
    data_mat = np.mat(data_arr)
    num_data = np.shape(data_mat)[0]
    class_est_agg = np.mat(np.zeros((num_data, 1)))
    for i in range(len(classifier_arr)):
        class_est = stump_classify(data_mat, \
                                   classifier_arr[i]['dim'], \
                                   classifier_arr[i]['thresh'], \
                                   classifier_arr[i]['ineq'])
        class_est_agg += classifier_arr[i]['alpha'] * class_est
        # print(class_est_agg)
    return np.sign(class_est_agg)


def plot_roc(pred_mat, labels_mat):
    """
    ROC曲线绘制及AUC计算. ROC曲线是选择不同阈值分类情况下, 绘制真阳性和假阳性
    :param pred_mat:      分类器预测值, 非二值，分类阈值并未确定
    :param labels_mat:    真实的分类标签
    :return: AUC的数值
    """
    import matplotlib.pyplot as plt
    y_sum = 0.0
    num_pos_class = sum(np.array(labels_mat) == 1.0)
    y_step = 1 / float(num_pos_class)
    x_step = 1 / float(len(labels_mat) - num_pos_class)
    # 按分类器的预测强度排序(从小到大),得到排序索引，从小到大排列
    # 代码实现是从预测强度排名最低的开始选择阈值，比阈值低的都分为负类，高的都为正。
    sorted_indicies = pred_mat.T.argsort().tolist()
    # 绘制光标位置
    # 一开始阈值极低, 把所有都判断为正，对应点为(1,1). 最终状态是阈值极高，所有都为负, 对应点(0,0)
    cur = (1.0, 1.0)
    fig = plt.figure()
    fig.clf()
    # 等同subplot(1,1,1)
    ax = plt.subplot(111)
    # x轴为假阳率 FP/(FP+TN) = 1 - 真阴率,
    # y轴为真阳率 TP/(TP+FN) = 1 - 假阴率
    # 将阈值选择为预测强度，逐步增高。阈值点本身分类为负
    for index in sorted_indicies[0]:
        # 阈值点真实标签为正，则为假阴，延y轴方向下降一个步长
        if labels_mat[index] == 1.0:
            del_x = 0
            del_y = y_step
        # 阈值点真实标签为负,即真阴， 则延x轴方向倒退一个步长
        else:
            del_x = x_step
            del_y = 0
            # 对矩形高度进行累加, 为计算总面积准备
            y_sum += cur[1]
        # 在当前点和新点之间画线段
        ax.plot([cur[0], cur[0] - del_x], [cur[1], cur[1] - del_y], 'b')
        # 更新当前点坐标
        cur = (cur[0] - del_x, cur[1] - del_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.axis([0, 1, 0, 1])
    plt.show()
    val_auc = y_sum * x_step
    print("the Area Under the Curve is:", val_auc)
    return val_auc
