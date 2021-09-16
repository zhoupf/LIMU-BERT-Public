# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 14/1/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : statistic.py
# @Description :
import numpy as np
from sklearn import metrics
from sklearn.metrics import f1_score

from plot import plot_matrix


def stat_acc_f1(label, results_estimated):
    # label = np.concatenate(label, 0)
    # results_estimated = np.concatenate(results_estimated, 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1


def stat_acc_f1_dual(label, results_estimated):
    label = np.concatenate(label, 0)
    results_estimated = np.concatenate([t[1] for t in results_estimated], 0)
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    return acc, f1


def stat_acc(label, results_estimated):
    label_estimated = np.argmax(results_estimated, 1)
    f1 = f1_score(label, label_estimated, average='macro')
    acc = np.sum(label == label_estimated) / label.size
    matrix = metrics.confusion_matrix(label, label_estimated) #, normalize='true'
    return acc, matrix, f1


def stat_acc_f1_tpn(label, label_estimated, task_num=5, threshold=0.5):
    label_new = []
    label_estimated_new = []
    for i in range(label.size):
        if label[i] == 0:
            label_new.append(np.zeros((task_num, 1)))
            label_estimated_new_temp = np.zeros((task_num, 1))
            label_estimated_new_temp[label_estimated[i, :] > threshold] = 1
            label_estimated_new.append(label_estimated_new_temp)
        else:
            label_new.append(np.ones((1, 1)))
            label_estimated_new_temp = np.zeros((1, 1))
            label_estimated_new_temp[label_estimated[i, label[i] - 1] > threshold] = 1
            label_estimated_new.append(label_estimated_new_temp)
    label_new = np.concatenate(label_new, 0)[:, 0]
    label_estimated_new = np.concatenate(label_estimated_new, 0)[:, 0]
    f1 = f1_score(label_new, label_estimated_new, average='macro')
    acc = np.sum(label_new == label_estimated_new) / label_new.size
    return acc, f1

# def stat_normalized_distance(input, target, norm_dimension=3, p=2):
#     sensor_num = input.shape[2] // norm_dimension
#     input_re = input.reshape(input.shape[0], input.shape[1], sensor_num, norm_dimension)
#     target_re = target.reshape(target.shape[0], target.shape[1], sensor_num, norm_dimension)
#     losses = []
#     # temps = []
#     norms = np.zeros(sensor_num, dtype=np.float32)
#     for i in range(sensor_num):
#         if p == 1:
#             losses.append(np.abs(input_re[:, :, i, :] - target_re[:, :, i, :]))
#             # temps.append(losses[i].mean())
#             norms[i] = np.sum(np.abs(target_re[:, :, i, :]))
#         elif p == 2:
#             losses.append(np.abs(input_re[:, :, i, :], input_re[:, :, i, :], reduction=self.reduction))
#             norms[i] = np.sum(torch.pow(torch.abs(target_re[:, :, i, :]), self.p))
#             # temps.append(losses[i].mean())
#         else:
#             raise RuntimeError(("Bad NormalizedLoss Parameter p = {}".format(self.p)))
#     loss = None
#     norms = np.sum(norms) / norms
#     factors = norms / np.sum(norms)
#     for i in range(sensor_num):
#         if i == 0:
#             loss = losses[i] * factors[i]
#         else:
#             loss += losses[i] * factors[i]
#     return loss

def illustrate_results(name, acc, f1, matrix, labels=['Facing', 'Pocket', 'Hand', 'Bag', 'Chest']):
    print('%s Accuracy %0.4f, %s F1-Score %0.4f' % (
        name, acc, name, f1))
    plot_matrix(matrix, labels)


def illustrate_sample_content(label, labels_name=None, target=""):
    label_unique = np.max(label)
    s = target + "sample: " + str(label.size)
    for i in range(int(label_unique) + 1):
        if labels_name is None:
            s += ", ({}, {})".format(i, np.sum(label == i))
        else:
            s += ", ({}, {}, {})".format(i, labels_name[i], np.sum(label == i))
    print(s)


def illustrate_classification_result(result_train, label_train, result_test, label_test, labels_name):
    acc, matrix, f1 = stat_acc(label_train, result_train)
    illustrate_sample_content(label_train, labels_name, target="Train ")
    print('Train Accuracy: %0.3f, F1-score: %0.3f' % (acc, f1))
    plot_matrix(matrix, labels_name=labels_name)
    acc, matrix, f1 = stat_acc(label_test, result_test)
    illustrate_sample_content(label_test, labels_name, target="Test ")
    print('Test Accuracy: %0.3f, F1-score: %0.3f' % (acc, f1))
    plot_matrix(matrix, labels_name=labels_name)