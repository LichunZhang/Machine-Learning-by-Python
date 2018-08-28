#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : DecisionTree
# @File     : test_tree_plot.py
# @Software : PyCharm
# @Date     : 2018/8/27 下午1:48
# @Descrip  : 
# @License  : Copyright(c) Lichun Zhang. 
# @Contact  : lczhang2011@gmail.com
import unittest
import tree_plot


class MyTestCase(unittest.TestCase):
    def test_tree_plot(self):
        tree_plot.create_simple_plot()
        self.assertEqual(True, True)

    def test_retrieve_tree(self):
        tree = tree_plot.retrieve_tree(1)
        res = {'no surfacing': {0: 'no', 1: {'flippers': \
                                                 {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
        self.assertEqual(tree, res)
        tree = tree_plot.retrieve_tree(0)
        # self.assertEqual(tree_plot.get_num_leafs(tree), 3)
        self.assertEqual(tree_plot.get_tree_depth(tree), 2)

    def test_create_plot(self):
        tree = tree_plot.retrieve_tree(0)
        tree_plot.create_plot(tree)
        tree['no surfacing'][3]='maybe'
        tree_plot.create_plot(tree)
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main()
