#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/10/4 9:16
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : benchmark.py
# @Description :
import argparse

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import train
from config import load_dataset_label_names
from models import ClassifierLSTM, ClassifierGRU, ClassifierCNN2D, fetch_classifier
from statistic import stat_acc_f1
from utils import get_device, merge_dataset, match_labels, handle_argv, \
    prepare_embedding_dataset_balance, IMUDataset, load_classifier_data_config, prepare_embedding_dataset, \
    FFTDataset, prepare_dataset, Preprocess4Normalization


def prepare_dataset_by_user(data_raw, labels_raw, training_rate, users, label_index=0, shuffle=True, merge=True):
    rids = np.unique(users)
    arr = np.arange(rids.size)
    train_rider_num = int(rids.size * training_rate)
    index = match_labels(users, arr[:train_rider_num])
    data_train = data_raw[index, :, :3]
    data_test = data_raw[~index, :, :3]
    label_train = labels_raw[index, ...][..., label_index]
    label_test = labels_raw[~index, ...][..., label_index]
    if merge:
        data_train, label_train = merge_dataset(data_train, label_train)
        data_test, label_test = merge_dataset(data_test, label_test)
    print('Train Size: %d, Test Size: %d' % (label_train.shape[0], label_test.shape[0]))
    return data_train, label_train, data_test, label_test


def main(args, label_index, training_rate, label_rate, balance=True, method=None):
    data, labels, train_cfg, model_cfg, dataset_cfg = load_classifier_data_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_dataset(data, labels
             , label_index=label_index, training_rate=training_rate, label_rate=label_rate
             , merge=model_cfg.seq_len, seed=train_cfg.seed, balance=balance) #, user_index=dataset_cfg.user_label_index
    pipeline = [Preprocess4Normalization(model_cfg.input)]
    if method != 'deepsense':
        data_set_train = IMUDataset(data_train, label_train, pipeline=pipeline)
        data_set_test = IMUDataset(data_test, label_test, pipeline=pipeline)
        data_set_vali = IMUDataset(data_vali, label_vali, pipeline=pipeline)
    else:
        data_set_train = FFTDataset(data_train, label_train, pipeline=pipeline)
        data_set_test = FFTDataset(data_test, label_test, pipeline=pipeline)
        data_set_vali = FFTDataset(data_vali, label_vali, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=True, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    model = fetch_classifier(method, model_cfg, input=model_cfg.input, output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer_train = train.Trainer(train_cfg, model, data_loader_train, optimizer, args.save_path, get_device(args.gpu))

    def get_loss(model, batch):
        inputs, labels = batch

        logits = model(inputs, True)
        loss = criterion(logits, labels)
        return loss

    trainer_train.train_eval(get_loss, stat_acc_f1, data_loader_test, data_loader_vali)


if __name__ == "__main__":
    train_rate = 0.8
    balance = True
    label_rate = 0.01
    method = "gru"
    # for label_rate in [0.01]:
    # for method in ["dcnn", "deepsense"]:
    #     args = handle_argv('bench_' + method, 'train.json', method)
    #     main(args, 0, train_rate, label_rate, balance=balance, method=method)
        # main(args, 1, train_rate, label_rate, balance=balance, method=method)
    args = handle_argv('bench_' + method, 'train.json', method)
    main(args, 0, train_rate, label_rate, balance=balance, method=method)
    # main(args, 1, train_rate, label_rate, balance=balance, method=method)
    # main(args, 2, train_rate, label_rate, balance=balance, method=method)
