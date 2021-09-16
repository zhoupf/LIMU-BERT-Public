# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 9/2/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : tpn.py
# @Description :
import numpy as np
import torch
from scipy.stats import special_ortho_group
from torch import nn
from torch.utils.data import DataLoader

import train
from config import load_dataset_label_names
from models import fetch_classifier, BenchmarkTPNPretrain, BenchmarkTPNClassifier
from statistic import stat_acc_f1, stat_acc_f1_tpn
from utils import load_classifier_data_config, prepare_embedding_dataset, prepare_embedding_dataset_balance, get_device, \
    IMUDataset, handle_argv, shuffle_data_label, prepare_dataset


def noised(data):
    noise = np.random.normal(0.0, 1, data.shape)
    return data + noise


def scaled(data):
    ope = np.random.randint(2, size=data.shape) * 2 - 1
    sca = np.random.random(data.shape) * 2
    return data * np.multiply(ope, sca)


def rotated(data):
    rm = special_ortho_group.rvs(3)
    re = []
    for i in range(data.shape[1] // 3):
        re.append(np.dot(rm, data[:, i*3:i*3+3].T).T)
    return np.concatenate(re, 1)


def flipped(data):
    return data[::-1, :]


def shuffled(data):
    index = np.array([0, 1, 2])
    index = np.random.permutation(index)
    sensor_num = data.shape[1] // 3
    index = np.repeat(index, sensor_num).reshape(3, sensor_num).T.reshape(data.shape[1])
    return data[:, index]


def generate_self_trained_dataset(dataset, funcs=None):
    if funcs is None:
        funcs = [noised, scaled, rotated, flipped, shuffled]
    dataset_new = []
    label_new = []
    for i in range(dataset.shape[0]):
        dataset_new.append(dataset[i])
        label_new.append(0)
        for j in range(len(funcs)):
            dataset_new.append(funcs[j](dataset[i]))
            label_new.append(1 + j)
    return np.array(dataset_new), np.array(label_new)


def self_train(args, training_rate, balance, task_num=5):
    data, labels, train_cfg, model_cfg, dataset_cfg = load_classifier_data_config(args)
    # label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_dataset(data, labels
                , label_index=label_index, training_rate=training_rate, label_rate=label_rate,
                merge=model_cfg.seq_len, change_shape=True, seed=train_cfg.seed, balance=balance)

    dataset_train_new, label_train_new = generate_self_trained_dataset(data_train)
    dataset_vali_new, label_vali_new = generate_self_trained_dataset(data_vali)
    dataset_test_new, label_test_new = generate_self_trained_dataset(data_test)

    data_set_train = IMUDataset(dataset_train_new, label_train_new)
    data_set_vali = IMUDataset(dataset_vali_new, label_vali_new)
    data_set_test = IMUDataset(dataset_test_new, label_test_new)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)


    model = BenchmarkTPNPretrain(model_cfg, task_num, input=data_train.shape[-1])
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.lambda2)
    trainer_train = train.Trainer(train_cfg, model, data_loader_train, optimizer, args.save_dir, get_device(args.gpu), save_name=args.save_model)

    def get_loss(model, batch):
        inputs, labels = batch

        logits = model(inputs, True)
        loss = []
        for i in range(labels.size()[0]):
            if labels[i] == 0:
                loss.append(-torch.log(1 - logits[i, :]).reshape(5, 1))
            else:
                loss.append(-torch.log(logits[i, labels[i].item() - 1]).reshape(1, 1))
        return torch.cat(loss, dim=0)

    trainer_train.train_eval(get_loss, stat_acc_f1_tpn, data_loader_vali=data_loader_vali, data_loader_test=data_loader_test)


def classify(args, label_index, training_rate, label_rate, balance):
    data, labels, train_cfg, model_cfg, dataset_cfg = load_classifier_data_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)

    data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_dataset(data, labels
        , label_index=label_index, training_rate=training_rate, label_rate=label_rate
        , merge=model_cfg.seq_len, seed=train_cfg.seed, balance=balance)
    data_set_train = IMUDataset(data_train, label_train)
    data_set_test = IMUDataset(data_test, label_test)
    data_set_vali = IMUDataset(data_vali, label_vali)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=True, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    model = BenchmarkTPNClassifier(model_cfg, input=data_train.shape[-1], output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer_train = train.Trainer(train_cfg, model, data_loader_train, optimizer, args.save_dir, get_device(args.gpu), save_name=args.save_model)

    def get_loss(model, batch):
        inputs, labels = batch

        logits = model(inputs, True)
        loss = criterion(logits, labels)
        return loss

    trainer_train.train_eval(get_loss, stat_acc_f1, data_loader_test, data_loader_vali, model_file=args.pretrain_model, load_self=True) #


if __name__ == "__main__":
    label_index = 0
    train_rate = 0.8
    # label_rate = 0.01 / self_train_rate
    label_rate = 0.01

    balance = True
    method = "tpn"
    # args = handle_argv('bench_' + method + "_pre", 'train_tpn.json', method)
    # self_train(args, train_rate, balance=balance)

    #  -s c1_1
    args = handle_argv('bench_' + method + "_classifier", 'train_tpn.json', method)
    # for label_rate in [0.002, 0.005, 0.01, 0.02, 0.05, 0.1]: #0.002, 0.005, 0.01, 0.02, 0.05, 0.1
    #     classify(args, 0, train_rate, label_rate, balance)
    #     classify(args, 1, train_rate,  label_rate, balance)
    classify(args, 0, train_rate, label_rate, balance)
    classify(args, 1, train_rate, label_rate, balance)



