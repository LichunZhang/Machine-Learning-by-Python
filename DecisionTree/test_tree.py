#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : DecisionTree
# @File     : test_tree.py
# @Software : PyCharm
# @Date     : 2018/8/26 上午10:50
# @Descrip  : 
# @License  : Copyright(c) Lichun Zhang. 
# @Contact  : lczhang2011@gmail.com
import unittest
import trees
import tree_plot


class MyTestCase(unittest.TestCase):
    def test_calc_entropy(self):
        data_set, labels = trees.load_simple_data()
        entropy = trees.calc_info_entropy(data_set)
        self.assertLessEqual(0.9709505945, round(entropy, 10))

    def test_split_data(self):
        data_set, labels = trees.load_simple_data()
        dat = trees.split_data_set(data_set, 0, 1)
        res_exp = [[1, 'yes'], [1, 'yes'], [0, 'no']]
        self.assertEqual(res_exp, dat)
        dat = trees.split_data_set(data_set, 0, 0)
        res_exp = [[1, 'no'], [1, 'no']]
        self.assertEqual(res_exp, dat)

    def test_choose_best_ft(self):
        data_set, labels = trees.load_simple_data()
        index_ft = trees.choose_best_ft(data_set)
        self.assertEqual(0, index_ft)

    def test_create_tree(self):
        data_set, labels = trees.load_simple_data()
        my_tree = trees.create_tree(data_set, labels)
        res_exp = {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
        self.assertEqual(res_exp, my_tree)

    def test_classify_simple(self):
        data_set, labels = trees.load_simple_data()
        my_tree = tree_plot.retrieve_tree(0)
        self.assertEqual('no', trees.classify(my_tree, labels, [1, 0]))
        self.assertEqual('yes', trees.classify(my_tree, labels, [1, 1]))

    def test_pickle_tree(self):
        my_tree = tree_plot.retrieve_tree(0)
        trees.store_tree(my_tree, 'tree_storage.txt')
        grab_tree = trees.grab_tree('tree_storage.txt')
        self.assertEqual(my_tree, grab_tree)

    def test_classify_loaded(self):
        with open('lenses.txt') as fr:
            lenses = [line.strip().split('\t') for line in fr.readlines()]
        lenses_labels = ['age', 'prescript', 'astigmatic', 'tearRate']
        lenses_tree = trees.create_tree(lenses, lenses_labels)
        tree_plot.create_plot(lenses_tree)

if __name__ == '__main__':
    unittest.main()
