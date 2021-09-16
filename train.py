#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:22
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : train.py
# @Description :

import os
import time
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn

from utils import count_model_parameters


class Trainer(object):
    """Training Helper Class"""
    def __init__(self, cfg, model, data_loader, optimizer, save_path, device):
        self.cfg = cfg # config for training : see class Config
        self.model = model
        self.data_loader = data_loader # iterator to load data
        self.optimizer = optimizer
        self.save_path = save_path
        self.device = device # device name

    def train(self, get_loss, evaluate, model_file=None, data_parallel=False, data_loader_test=None):
        """ Train Loop """
        self.load(model_file)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        best_loss = 1e6
        for e in range(self.cfg.n_epochs):
            loss_sum = 0. # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(self.data_loader):
                batch = [t.to(self.device) for t in batch]
                start_time = time.time()
                self.optimizer.zero_grad()
                loss = get_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()
                time_sum += time.time() - start_time
                global_step += 1
                loss_sum += loss.item()

                if global_step % self.cfg.save_steps == 0: # save
                    self.save(global_step)

                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
                # print(i)

            loss_eva = self.eval(evaluate=evaluate, data_loader_test=data_loader_test)
            print('Epoch %d/%d : Average Loss %5.4f. Test Loss %5.4f'
                    % (e + 1, self.cfg.n_epochs, loss_sum / len(self.data_loader), loss_eva))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if loss_eva < best_loss:
                best_loss = loss_eva
                self.save(0)
        print('The Total Epoch have been reached.')
        # self.save(global_step)

    def eval(self, evaluate, model_file=None, data_loader_test=None, data_parallel=False):
        """ Evaluation Loop """
        self.model.eval() # evaluation mode
        self.load(model_file)
        # print(count_model_parameters(self.model))
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        dt = self.data_loader
        if data_loader_test:
            dt = data_loader_test
        results = [] # prediction results
        labels = []
        time_sum = 0.0
        for batch in dt:
            batch = [t.to(self.device) for t in batch]
            with torch.no_grad(): # evaluation without gradient calculation
                start_time = time.time()
                result = model(batch[:-1])
                time_sum += time.time() - start_time
                result = result.cpu().numpy()
                results.append(result)
                labels.append(batch[-1].cpu().numpy())
        # print("Eval execution time: %.5f seconds" % (time_sum / len(dt)))
        if evaluate:
            return evaluate(np.concatenate(labels, 0), np.concatenate(results, 0))
        else:
            return np.concatenate(results, 0)

    def train_eval(self, get_loss, evaluate, data_loader_test, data_loader_vali, model_file=None, data_parallel=False, load_self=False):
        """ Train Loop """
        self.load(model_file, load_self)
        model = self.model.to(self.device)
        if data_parallel: # use Data Parallelism with Multi-GPU
            model = nn.DataParallel(model)

        global_step = 0 # global iteration steps regardless of epochs
        vali_acc_best = 0.0
        best_stat = None

        for e in range(self.cfg.n_epochs):
            loss_sum = 0.0 # the sum of iteration losses to get average loss in every epoch
            time_sum = 0.0
            self.model.train()
            for i, batch in enumerate(self.data_loader):
                batch = [t.to(self.device) for t in batch]

                start_time = time.time()
                self.optimizer.zero_grad()
                loss = get_loss(model, batch)

                loss = loss.mean()# mean() for Data Parallelism
                loss.backward()
                self.optimizer.step()

                global_step += 1
                loss_sum += loss.item()
                time_sum += time.time() - start_time
                if self.cfg.total_steps and self.cfg.total_steps < global_step:
                    print('The Total Steps have been reached.')
                    return
            train_acc, train_f1 = self.eval(evaluate)
            test_acc, test_f1 = self.eval(evaluate, data_loader_test=data_loader_test)
            vali_acc, vali_f1 = self.eval(evaluate, data_loader_test=data_loader_vali)
            print('Epoch %d/%d : Average Loss %5.4f, Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f'
                  % (e+1, self.cfg.n_epochs, loss_sum / len(self.data_loader), train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1))
            # print("Train execution time: %.5f seconds" % (time_sum / len(self.data_loader)))
            if vali_acc > vali_acc_best:
                vali_acc_best = vali_acc
                best_stat = (train_acc, vali_acc, test_acc, train_f1, vali_f1, test_f1)
                self.save(0)
        print('The Total Epoch have been reached.')
        print('Best Accuracy: %0.3f/%0.3f/%0.3f, F1: %0.3f/%0.3f/%0.3f' % best_stat)

    def load(self, model_file, load_self=False):
        """ load saved model or pretrained transformer (a part of model) """
        if model_file:
            print('Loading the model from', model_file)
            if load_self:
                self.model.load_self(model_file + '.pt', map_location=self.device)
            else:
                self.model.load_state_dict(torch.load(model_file + '.pt', map_location=self.device))

    def save(self, i):
        """ save current model """
        if i != 0:
            torch.save(self.model.state_dict(), self.save_path + "_" + str(i) + '.pt')
        else:
            torch.save(self.model.state_dict(),  self.save_path + '.pt')

    def set_data_loader(self, data_loader_new):
        self.data_loader = data_loader_new
