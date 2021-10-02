# LIMU-BERT
LIMU-BERT, a novel representation learning model that can make use of unlabeled IMU data and extract generalized rather than task-specific features. 
LIMU-BERT adopts the principle of self-supervised training of the natural language model BERT to effectively capture temporal relations and feature distributions in IMU sensor data. 
However, the original BERT is not adaptive to mobile IMU data. 
By meticulously observing the characteristics of IMU sensors, we propose a series of techniques and accordingly adapt LIMU-BERT to IMU sensing tasks. The designed models are lightweight and easily deployable on mobile devices. 
With the representations learned via LIMU-BERT, task-specific models trained with limited labeled samples can achieve superior performances. 

Please check our paper for more details.
## File Overview
This contains 9 python files.
- [`tokenization.py`](./tokenization.py) : Tokenizers adopted from the original Google BERT's code
- [`models.py`](./models.py) : Model classes for a general transformer
- [`optim.py`](./optim.py) : A custom optimizer (BertAdam class) adopted from Hugging Face's code
- [`train.py`](./train.py) : A helper class for training and evaluation
- [`utils.py`](./utils.py) : Several utility functions
- [`pretrain.py`](./pretrain.py) : An example code for pre-training transformer
- 
## Usage
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

## Cite
LIMU-BERT: Unleashing the Potential of Unlabeled Data for IMU Sensing Applications


