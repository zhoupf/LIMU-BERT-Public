#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain_base.py
# @Description :
import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

import train
from config import  load_dataset_label_names
from embedding import load_embedding_label
from loss import FocalLoss, AngularPenaltySMLoss
from models import ClassifierLSTM, ClassifierGRU, ClassifierCNN2D, fetch_classifier
from plot import plot_matrix

from statistic import stat_acc_f1, stat_acc
from utils import get_device, handle_argv \
    , IMUDataset, load_classifier_config, prepare_dataset, Preprocess4Normalization, prepare_classifier_dataset


def generate_output(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_dataset(data, labels
                                                                                            , label_index=label_index,
                                                                                            training_rate=training_rate,
                                                                                            label_rate=label_rate
                                                                                            , merge=model_cfg.seq_len,
                                                                                            seed=train_cfg.seed,
                                                                                            balance=balance)
    data_set_test = IMUDataset(data_test, label_test)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
    trainer_train = train.Trainer(train_cfg, model, None, None, args.save_path, get_device(args.gpu))


    label_estimate_test = trainer_train.eval(None, model_file=args.save_path, data_loader_test=data_loader_test)
    return label_test, label_estimate_test, label_names


def classify_embeddings(args, data, labels, label_index, training_rate, label_rate, balance=False, method=None):
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    label_names, label_num = load_dataset_label_names(dataset_cfg, label_index)
    data_train, label_train, data_vali, label_vali, data_test, label_test = prepare_dataset(data, labels
           , label_index=label_index, training_rate=training_rate, label_rate=label_rate
           , merge=model_cfg.seq_len, seed=train_cfg.seed, balance=balance)
    data_set_train = IMUDataset(data_train, label_train)
    data_set_vali = IMUDataset(data_vali, label_vali)
    data_set_test = IMUDataset(data_test, label_test)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_vali = DataLoader(data_set_vali, shuffle=False, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)

    criterion = nn.CrossEntropyLoss()
    # criterion = FocalLoss()
    model = fetch_classifier(method, model_cfg, input=data_train.shape[-1], output=label_num)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)  # , weight_decay=0.95
    trainer_train = train.Trainer(train_cfg, model, data_loader_train, optimizer, args.save_path, get_device(args.gpu))

    def get_loss(model, batch):
        inputs, labels = batch

        logits = model(inputs, True)
        loss = criterion(logits, labels)
        return loss

    trainer_train.train_eval(get_loss, stat_acc_f1, data_loader_test, data_loader_vali)


if __name__ == "__main__":
    label_index = 0
    training_rate = 0.8
    label_rate = 0.01
    balance = True

    mode = "base"
    method = "gru"
    # for method in ["gru", "lstm", "cnn2", "attn"]:
    #     args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    #     embedding, labels = load_embedding_label(mode, args.model_file, args.dataset, args.dataset_version, type='t')
    #     classify_embeddings(args, embedding, labels, 0, training_rate, label_rate, balance=balance, method=method)
    #     classify_embeddings(args, embedding, labels, 1, training_rate, label_rate, balance=balance, method=method)
    # for label_rate in [0.002, 0.005, 0.02, 0.05, 0.1]:
    #     classify_embeddings(args, embedding, labels, 0, training_rate, label_rate, balance=balance, method=method)
    #     classify_embeddings(args, embedding, labels, 1, training_rate, label_rate, balance=balance, method=method)
    args = handle_argv('classifier_' + mode + "_" + method, 'train.json', method)
    embedding, labels = load_embedding_label(mode, args.model_file, args.dataset, args.dataset_version, type='t')
    # classify_embeddings(args, embedding, labels, 0, training_rate, label_rate, balance=balance, method=method)
    # classify_embeddings(args, embedding, labels, 1, training_rate, label_rate, balance=balance, method=method)
    # classify_embeddings(args, embedding, labels, 2, training_rate, label_rate, balance=balance, method=method)

    # lt, lte, label_names = generate_output(args, embedding, labels, 0, training_rate, label_rate, balance=balance, method=method)
    # acc, matrix, f1 = stat_acc(lt, lte)
    # matrix_norm = plot_matrix(matrix, label_names)


