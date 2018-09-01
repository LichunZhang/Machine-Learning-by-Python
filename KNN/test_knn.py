#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Project  : KNN
# @File     : test_knn.py
# @Software : PyCharm
# @Date     : 2018/8/30 下午11:49
# @Descrip  : 
# @License  : Copyright(c) Lichun Zhang. 
# @Contact  : lczhang2011@gmail.com
import unittest
import numpy as np
import matplotlib.pyplot as plt
import knn


class MyTestCase(unittest.TestCase):
    def test_create_simple_data(self):
        group_exp = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
        labels_exp = ['A', 'A', 'B', 'B']
        group, labels = knn.create_simple_data()
        self.assertEqual(True, (group_exp == group).all())
        self.assertEqual(labels_exp, labels)

    def test_classify0(self):
        group, labels = knn.create_simple_data()
        res = knn.classify0([0, 0], group, labels, 3)
        label_exp = 'B'
        self.assertEqual(label_exp, res)

    def test_file2matrix(self):
        date_mat, date_label = knn.file2matrix('datingTestSet2.txt')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(date_mat[:, 1], date_mat[:, 2], \
                   15.0 * np.array(date_label), 15.0 * np.array(date_label))
        plt.show()
        self.assertEqual((1000, 3), date_mat.shape)
        self.assertEqual(1000, len(date_label))
        labels_exp = [3, 2, 1, 1, 1, 1, 3, 3, 1, 3]
        self.assertEqual(True, labels_exp == date_label[0:10])

    def test_auto_norm(self):
        date_mat, date_label = knn.file2matrix('datingTestSet2.txt')
        norm_mat, ranges, min_val = knn.auto_norm(date_mat)
        min_exp = np.array([0., 0., 0.001156])
        ranges_exp = np.array([9.1273000e+04, 2.0919349e+01, 1.6943610e+00])
        self.assertEqual(True, (min_exp == min_val).all())
        self.assertEqual(True, (ranges == ranges_exp).all())

    def test_dating_test(self):
        err = knn.dating_class_test(0.1)
        self.assertEqual(0.05, err)

    def test_classify_person(self):
        res = knn.classify_person()

    def test_writing_test(self):
        err = knn.writing_class_test()
        self.assertEqual(0.011628, round(err, 6))


if __name__ == '__main__':
    unittest.main()
