#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain.py
# @Description :
import argparse
import sys

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import models, train
from config import MaskConfig, TrainConfig, PretrainModelConfig
from models import LIMUBertModel4Pretrain
from utils import set_seeds, get_device \
    , LIBERTDataset4Pretrain, handle_argv, load_pretrain_data_config, prepare_dataset, \
    prepare_pretrain_dataset, Preprocess4Normalization,  Preprocess4Mask


def main(args, training_rate):
    data, labels, train_cfg, model_cfg, mask_cfg, dataset_cfg = load_pretrain_data_config(args)

    pipeline = [Preprocess4Normalization(model_cfg.feature_num), Preprocess4Mask(mask_cfg)]
    # pipeline = [Preprocess4Mask(mask_cfg)]
    data_train, label_train, data_test, label_test = prepare_pretrain_dataset(data, labels, training_rate, seed=train_cfg.seed)

    data_set_train = LIBERTDataset4Pretrain(data_train, pipeline=pipeline)
    data_set_test = LIBERTDataset4Pretrain(data_test, pipeline=pipeline)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=train_cfg.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=train_cfg.batch_size)
    model = LIMUBertModel4Pretrain(model_cfg)

    criterion = nn.MSELoss(reduction='none')

    optimizer = torch.optim.Adam(params=model.parameters(), lr=train_cfg.lr)
    device = get_device(args.gpu)
    trainer = train.Trainer(train_cfg, model, data_loader_train, optimizer, args.save_path, device)

    writer = SummaryWriter(log_dir=args.log_dir)

    def get_loss(model, batch): # make sure loss is tensor
        mask_seqs, masked_pos, seqs = batch #, mask_norms

        logits_lm = model(mask_seqs, masked_pos) #, mask_norms
        loss_lm = criterion(logits_lm, seqs) # for masked LM
        # loss_lm = (loss_lm.float()).mean()
        return loss_lm

    def evaluate_mask(gt, es):
        es_t = torch.from_numpy(es)
        gt_t = torch.from_numpy(gt)
        loss_lm = criterion(es_t, gt_t)
        return loss_lm.mean().cpu().numpy()

    if hasattr(args, 'pretrain_model'):
        trainer.train(get_loss, evaluate=evaluate_mask, data_loader_test=data_loader_test
                      , model_file=args.pretrain_model)
    else:
        trainer.train(get_loss, evaluate=evaluate_mask, data_loader_test=data_loader_test, model_file=None)


if __name__ == "__main__":
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    training_rate = 0.8
    main(args, training_rate)