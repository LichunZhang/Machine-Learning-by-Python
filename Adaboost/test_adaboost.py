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


class TestAdaboostCase(unittest.TestCase):
    def test(self):
        print("Train Simple Data:")
        data_mat, labels_arr = adb.load_simple_data()
        w_data = np.mat(np.ones((5, 1)) / 5)
        best_stump, min_error, best_estimation = adb.build_stump(data_mat, labels_arr, w_data)
        self.assertEqual(best_stump, {'dim': 0, 'ineq': 'lt', 'thresh': 1.3})
        self.assertEqual(min_error, np.matrix([[0.2]]))
        classifies_arr, est_agg = adb.train(data_mat, labels_arr, 9)
        adb.classify([[5,5],[0,0]], classifies_arr)

        print("Train Custom Data:")
        datArr, labelArr = adb.load_data_set('horseColicTraining2.txt')
        classiferArray, aggClassEst = adb.train(datArr, labelArr, 10)
        print("Test Custom Data:")
        testArr, testLabelArr = adb.load_data_set('horseColicTest2.txt')
        prediction10 = adb.classify(testArr,classiferArray)
        errArr = np.mat(np.ones((67,1)))
        err_rate = errArr[prediction10 != np.mat(testLabelArr).T].sum() / 67
        print("Test Error: %f%%"%(err_rate * 100))
        # 绘制ROC和计算AUC
        self.assertEqual(err_rate, 16.0 / 67)
        val_auc = adb.plot_roc(aggClassEst, labelArr)
        self.assertLessEqual(val_auc - 0.858296963506, 1e-10)

if __name__ == '__main__':
    unittest.main()
