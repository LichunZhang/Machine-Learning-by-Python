#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project  : Adaboost
# @File     : test_adaboost.py
# @Software : PyCharm
# @Date     : 2018/8/22 下午8:30
# @Descrip  :
# @License  : Copyright(c) Lichun Zhang.
# @Contact  : lczhang2011@gmail.com
import unittest
import numpy as np
import adaboost as adb


class MyTestCase(unittest.TestCase):
    def test_build_stump(self):
        print("Test Build Stump:")
        data_mat, labels_arr = adb.load_simple_data()
        w_data = np.mat(np.ones((5, 1)) / 5)
        best_stump, min_error, best_estimation = adb.build_stump(data_mat, labels_arr, w_data)
        self.assertEqual(best_stump, {'dim': 0, 'ineq': 'lt', 'thresh': 1.3})
        self.assertEqual(min_error, np.matrix([[0.2]]))
        res = np.array([[-1.],
                        [1.],
                        [-1.],
                        [-1.],
                        [1.]])
        self.assertEqual((best_estimation == res).all(), True)

    def test_train(self):
        print("Test Train:")
        data_mat, labels_arr = adb.load_simple_data()
        classifies_arr, est_agg = adb.train(data_mat, labels_arr, 9)
        self.assertEqual(len(classifies_arr), 3)
        self.assertEqual(classifies_arr[-1]['dim'], 0)
        self.assertEqual(classifies_arr[-1]['ineq'], 'lt')
        self.assertEqual(classifies_arr[-1]['thresh'] - 0.9, 0)
        self.assertEqual(classifies_arr[-1]['alpha'] - 0.89587973461402726, 0)

    def test_classify(self):
        print("Test Classify:")
        print("Classify Simple Data:")
        data_mat, labels_arr = adb.load_simple_data()
        classifies_arr, est_agg = adb.train(data_mat, labels_arr, 30)
        pred = adb.classify([[5, 5], [0, 0]], classifies_arr)
        res = np.matrix([[1.], [-1.]])
        self.assertEqual((pred == res).all(), True)

        print("Classify Loaded Data:")
        datArr, labelArr = adb.load_data_set('horseColicTraining2.txt')
        classiferArray, aggClassEst = adb.train(datArr, labelArr, 10)
        testArr, testLabelArr = adb.load_data_set('horseColicTest2.txt')
        prediction10 = adb.classify(testArr, classiferArray)
        errArr = np.mat(np.ones((67, 1)))
        err_rate = errArr[prediction10 != np.mat(testLabelArr).T].sum() / 67
        self.assertEqual(err_rate, 16.0 / 67)
        print("Test Error: %f%%" % (err_rate * 100))
        # 绘制ROC和计算AUC
        val_auc = adb.plot_roc(aggClassEst, labelArr)
        self.assertLessEqual(val_auc - 0.858296963506, 1e-10)


if __name__ == '__main__':
    unittest.main()
