# LIMU-BERT
Adopt lite BERT to analyze IMU data

## Pretrain Script Usage
usage: pretrain_base.py [-h] [-g GPU] [-f MODEL_FILE] [-t TRAIN_CFG] [-m MODEL_CFG] [-a MASK_CFG] {eleme,hhar,huawei,motion,uci,wisdm} {10_100,20_120}

PyTorch LIMU-BERT Model

positional arguments:
  {eleme,hhar,huawei,motion,uci,wisdm}
                        Dataset name
  {10_100,20_120}       Dataset version

optional arguments:

  -h, --help            show this help message and exit

  -g GPU, --gpu GPU     Set specific GPU

  -f MODEL_FILE, --model_file MODEL_FILE        Pretrain model file

  -t TRAIN_CFG, --train_cfg TRAIN_CFG       Training config json file path

  -m MODEL_CFG, --model_cfg MODEL_CFG       Model config json file path

  -a MASK_CFG, --mask_cfg MASK_CFG      Mask strategy json file path

## Docker
check containers: docker ps -a
check running containers: docker ps -s

