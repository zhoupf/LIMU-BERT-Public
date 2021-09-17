# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/5/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : merge.py
# @Description :

import os
import numpy as np

from utils import prepare_pretrain_dataset, set_seeds, partition_and_reshape


def merge_simple(datasets, versions, label_indexes, feature_num=6):
    data_merge = []
    label_merge = []
    for d in range(len(datasets)):
        path_data = os.path.join(datasets[d], "data_" + versions[d] + ".npy")
        path_label = os.path.join(datasets[d], "label_" + versions[d] + ".npy")
        data = np.load(path_data).astype(np.float32)
        label = np.load(path_label).astype(np.float32)
        data_merge.append(data[:, :, :feature_num])
        label_merge.append(label[:, :, label_indexes[d]])
    data_merge = np.concatenate(data_merge, 0)
    label_merge = np.concatenate(label_merge, 0)
    return data_merge, label_merge


def arange_dataset(data_train_merge, data_vali_merge, data_test_merge, label_train_merge, label_vali_merge, label_test_merge
                   , seed, training_rate, vali_rate):
    total_num = data_train_merge.shape[0] + data_vali_merge.shape[0] + data_test_merge.shape[0]
    data_merge = np.concatenate([data_train_merge, data_vali_merge, data_test_merge], 0)
    label_merge = np.concatenate([label_train_merge, label_vali_merge, label_test_merge], 0)
    data = np.zeros((total_num, data_train_merge.shape[1], data_train_merge.shape[2]))
    label = np.zeros((total_num, label_train_merge.shape[1], label_train_merge.shape[2]))
    set_seeds(seed)
    arr = np.arange(total_num)
    np.random.shuffle(arr)
    train_num = int(total_num * training_rate)
    vali_num = int(total_num * vali_rate)
    data[arr[:train_num], ...] = data_merge[:train_num, ...]
    data[arr[train_num:train_num + vali_num], ...] = data_merge[train_num:train_num + vali_num, ...]
    data[arr[train_num + vali_num:], ...] = data_merge[train_num + vali_num:, ...]
    label[arr[:train_num], ...] = label_merge[:train_num, ...]
    label[arr[train_num:train_num + vali_num], ...] = label_merge[train_num:train_num + vali_num, ...]
    label[arr[train_num + vali_num:], ...] = label_merge[train_num + vali_num:, ...]
    print(np.sum(data == 0.0))
    print(np.sum(data_merge == 0.0))
    return data, label




def merge(datasets, versions, label_indexes, feature_num=6, training_rate=0.8, seed=3431):
    data_train_merge = []
    data_vali_merge = []
    data_test_merge = []
    label_train_merge = []
    label_vali_merge = []
    label_test_merge = []
    for d in range(len(datasets)):
        path_data = os.path.join(datasets[d], "data_" + versions[d] + ".npy")
        path_label = os.path.join(datasets[d], "label_" + versions[d] + ".npy")
        data = np.load(path_data).astype(np.float32)
        label = np.load(path_label).astype(np.float32)
        set_seeds(seed)
        data_train, label_train, data_vali, label_vali, data_test, label_test \
            = partition_and_reshape(data, label, label_index=label_indexes[d], training_rate=training_rate, vali_rate=0.1, change_shape=False)
        data_train_merge.append(data_train[:, :, :feature_num])
        data_vali_merge.append(data_vali[:, :, :feature_num])
        data_test_merge.append(data_test[:, :, :feature_num])
        label_train_merge.append(label_train)
        label_vali_merge.append(label_vali)
        label_test_merge.append(label_test)
    data_train_merge = np.concatenate(data_train_merge, 0)
    data_vali_merge = np.concatenate(data_vali_merge, 0)
    data_test_merge = np.concatenate(data_test_merge, 0)
    label_train_merge = np.concatenate(label_train_merge, 0)
    label_vali_merge = np.concatenate(label_vali_merge, 0)
    label_test_merge = np.concatenate(label_test_merge, 0)
    data, label = arange_dataset(data_train_merge, data_vali_merge, data_test_merge, label_train_merge, label_vali_merge,
                   label_test_merge, seed, training_rate, 0.1)
    return data, label


if __name__ == "__main__":

    datasets = ["hhar", "uci", "motion", "shoaib"]
    versions = ["20_120"] * 4
    label_indexes = [[2], [0], [0], [0]]
    data, label = merge(datasets, versions, label_indexes)
    np.save(os.path.join("merge", "data_" + versions[0] + ".npy"), data)
    np.save(os.path.join("merge", "label_" + versions[0] + ".npy"), label)

