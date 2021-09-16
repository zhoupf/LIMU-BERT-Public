# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19/5/2021
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : mobile.py
# @Description :
import os

import torch

from config import load_dataset_label_names
from models import LIMUBert4Embedding, fetch_classifier
from utils import handle_argv, load_pretrain_data_config, get_device, load_classifier_config, load_raw_data, \
    count_model_parameters


def output_bert_model(name):
    mode = "base"
    args = handle_argv('pretrain_' + mode, 'pretrain.json', mode)
    _, _, _, model_cfg, _, _ = load_pretrain_data_config(args)

    model = LIMUBert4Embedding(model_cfg, output_embed=True)
    model.load_self(args.pretrain_model, map_location=torch.device('cpu'))
    model.eval()
    print(count_model_parameters(model))
    example = torch.rand(1, 120, 6)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("mobile/" + name)


def output_classifier_model(method, name, label_index, hidden_dimension=72):
    args = handle_argv('classifier_base_' + method, 'train.json', method)
    train_cfg, model_cfg, dataset_cfg = load_classifier_config(args)
    _, label_num = load_dataset_label_names(dataset_cfg, label_index)
    model = fetch_classifier(method, model_cfg, input=hidden_dimension, output=label_num)
    model.load_state_dict(torch.load(os.path.join(args.save_dir, args.save_model + ".pt"), map_location=torch.device('cpu')))
    model.eval()
    print(count_model_parameters(model))
    example = torch.rand(1, model_cfg.seq_len, hidden_dimension)
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("mobile/" + name)



if __name__ == "__main__":
    # v2 uci 20_120 -f v1_35.pt
    output_bert_model("v1_38.pt")
    # v2 uci 20_120 -f v1_38.pt -s model

    output_classifier_model("gru", "v2.pt", label_index=0)
