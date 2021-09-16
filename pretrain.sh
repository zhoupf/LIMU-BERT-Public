#!/bin/bash
nohup python -u pretrain.py v23 motion 20_120 -g 0 -s v1_76 > log/pretrain_base_motion.log &
nohup python -u pretrain.py v23 uci 20_120 -g 0 -s v1_76 > log/pretrain_base_uci.log &
nohup python -u pretrain.py v23 hhar 20_120 -g 0 -s v1_76 > log/pretrain_base_hhar.log &
nohup python -u pretrain.py v23 shoaib 20_120 -g 0 -s v1_76 > log/pretrain_base_shoaib.log &
