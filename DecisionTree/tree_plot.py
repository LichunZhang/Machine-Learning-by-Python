#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : DecisionTree
# @File     : tree_plot.py
# @Software : PyCharm
# @Date     : 2018/8/27 下午1:38
# @Descrip  : 绘制词典类型的决策树
# @License  : Copyright(c) Lichun Zhang. 
# @Contact  : lczhang2011@gmail.com

import matplotlib.pyplot as plt

# 定义文本块和箭头格式
decision_node = dict(boxstyle="sawtooth", fc="0.8")
leaf_node = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plot_node(node_text, center_pt, parent_pt, node_type):
    """
    绘制带箭头注解
    :param node_text: 节点的注解文本
    :param center_pt: 注解文本的中心位置
    :param parent_pt: 箭头的出发位置
    :param node_type: 文本块的格式
    :return:
    """
    create_plot.ax1.annotate(node_text, \
                             xy=parent_pt, xycoords='axes fraction', \
                             xytext=center_pt, textcoords='axes fraction', \
                             va='center', ha='center', bbox=node_type, \
                             arrowprops=arrow_args)


def create_simple_plot():
    """
    绘制树型图
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # Python中所有变量默认全局有效
    create_plot.ax1 = plt.subplot(111, frameon=False)
    plot_node('Decision Node', (0.5, 0.1), (0.1, 0.5), decision_node)
    plot_node('Leaf Node', (0.8, 0.1), (0.3, 0.8), leaf_node)
    plt.show()


def get_num_leafs(tree):
    """
    画图所用函数, 获取决策树叶节点个数, 以便确定x轴长度
    :param tree: 字典类型的决策树
    :return: 叶节点个数
    """
    num_leafs = 0
    # python3改变了dict.keys,返回的是dict_keys对象,
    # 支持iterable 但不支持indexable，可将其明确的转化成list
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in list(second_dict.keys()):
        # 如果不是叶子节点, 即类型为字典
        if isinstance(second_dict[key], dict):
            # 递归
            num_leafs += get_num_leafs(second_dict[key])
        else:
            num_leafs += 1
    return num_leafs


def get_tree_depth(tree):
    """
    画图所用函数, 获得树的层数(深度), 以确定y轴高度
    :param tree: 字典类型的决策树
    :return: 树的深度
    """
    max_depth = 0
    first_str = list(tree.keys())[0]
    second_dict = tree[first_str]
    for key in second_dict.keys():
        if isinstance(second_dict[key], dict):
            # 递归
            cur_depth = 1 + get_tree_depth(second_dict[key])
        else:
            cur_depth = 1
        if cur_depth > max_depth:
            max_depth = cur_depth
    return max_depth


def retrieve_tree(i):
    """
    主要用于测试，返回预定义的树结构, 避免每次测试都重新创建树
    :param i: 获取列表中第i棵树的信息
    :return: 包含树信息的字典的列表
    """
    trees_list = [{'no surfacing': {0: 'no', 1: {'flippers': \
                                                     {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': \
                                                     {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return trees_list[i]


def plot_line_text(cur_pt, parent_pt, text):
    """
    在父子节点填充文本信息
    :param cur_pt: 子节点坐标
    :param parent_pt: 父节点坐标
    :param text: 文本信息
    :return:
    """
    x = (parent_pt[0] - cur_pt[0]) / 2 + cur_pt[0]
    y = (parent_pt[1] - cur_pt[1]) / 2 + cur_pt[1]
    create_plot.ax1.text(x, y, text)


def plot_tree(tree, parent_pt, node_text):
    """
    逻辑上的树绘制, 可根据图形比例绘制树型图, 无需关心实际输出图形大小
    :param tree: 决策树
    :param parent_pt: 父节点
    :param node_text: 节点文本
    :return:
    """
    # 计算宽与高
    num_leafs = get_num_leafs(tree)
    depth = get_tree_depth(tree)
    first_str = list(tree.keys())[0]
    # 1, 绘制自身
    # 使用全局变量xOff和yOff追踪已经绘制的节点位置
    cur_pt = (plot_tree.xOff + (1.0 + float(num_leafs)) / 2.0 / plot_tree.totalW, \
              plot_tree.yOff)
    plot_line_text(cur_pt, parent_pt, node_text)
    plot_node(first_str, cur_pt, parent_pt, decision_node)
    second_dict = tree[first_str]
    # 深入下一层, 减少y偏移
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in list(second_dict.keys()):
        # 2, 若子节点是非叶子节点, 递归
        if isinstance(second_dict[key],dict):
            plot_tree(second_dict[key], cur_pt, str(key))
        # 3, 若是叶子节点, 直接绘制
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), \
                      cur_pt, leaf_node)
            plot_line_text((plot_tree.xOff, plot_tree.yOff), cur_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(tree):
    """
    实际的绘制入口. 创建绘图区, 计算树形图的全部尺寸, 并调用plot_tree进行绘制
    :param tree: 决策树
    :return:
    """
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # 全局变量totalW和totalD存储树的宽度和深度
    plot_tree.totalW = float(get_num_leafs(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()
