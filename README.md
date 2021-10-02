# LIMU-BERT
LIMU-BERT, a novel representation learning model that can make use of unlabeled IMU data and extract generalized rather than task-specific features. 
LIMU-BERT adopts the principle of self-supervised training of the natural language model BERT to effectively capture temporal relations and feature distributions in IMU sensor data. 
However, the original BERT is not adaptive to mobile IMU data. 
By meticulously observing the characteristics of IMU sensors, we propose a series of techniques and accordingly adapt LIMU-BERT to IMU sensing tasks. The designed models are lightweight and easily deployable on mobile devices. 
With the representations learned via LIMU-BERT, task-specific models trained with limited labeled samples can achieve superior performances. 

Please check our paper for more details.
## File Overview
This contains 9 python files.
- [`config`](./config) : config json files of models and training hyper-parameters.
- [`dataset`](./dataset) : the scripts for preprocessing four open datasets and a config file of key attributes of those datasets.
- [`benchmark.py`](./benchmark.py) : run DCNN, DeepSense, and R-GRU.
- [`classifier.py`](./classifier.py) : run LIMU-GRU that inputs representations learned by LIMU-BERT and output labels for target applications.
- [`classifier_bert.py`](./classifier_bert.py) : run LIMU-GRU that inputs raw IMU readings and output labels for target applications.
- [`config.py`](./config.py) : some helper functions for loading settings.
- [`embedding.py`](./embedding.py) : generates representation or embeddings for raw IMU readings given a pre-trained LIMU-BERT.
- [`models.py`](./models.py) : the implementations of LIMU-BERT, LIMU-GRU, and other baseline models.
- [`plot.py`](./plot.py) : some helper function for plotting IMU sensor data or learned representations.
- [`pretrain.py`](./pretrain.py) : pretrain LIMU-BERT.
- [`statistic.py`](./statistic.py) : some helper functions for evaluation.
- [`tpn.py`](./tpn.py) : run TPN.
- [`train.py`](./train.py) : several helper functions for training models.
- [`utils.py`](./utils.py) : some helper functions for preprocessing data or separating dataset.

## Prepare dataset
In the [`dataset`](./dataset) folder, we provide four scripts that preprocess the corresponding datasets. Those datasets are widely adopted in the previous studies:
- [HHAR](http://archive.ics.uci.edu/ml/datasets/heterogeneity+activity+recognition)
- [UCI](http://archive.ics.uci.edu/ml/datasets/Smartphone-Based+Recognition+of+Human+Activities+and+Postural+Transitions)
- [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [Shoaib](https://www.mdpi.com/1424-8220/14/6/10146)

Each script has a kernel function which transform the raw IMU data and output preprocessed data and label. You can set the sampling rate and window size (sequence length).
- Data: a numpy array with the shape of (N*W*F), N is the number of samples, W is the windows size, and F is the number of features (6 or 9).
- Label: a numpy array with the shape of (N*W*L), N is the number of samples, W is the windows size, and L is the number of label types (e.g., activity and user label). The detailed label information is provied in [`data_config.json`](./dataset/data_config.json).
The two numpy arrays are saved as "data_X_Y.npy" and "label_X_Y.npy", where X represents the sampling rate and Y is the window size. For example, all data and label are saved as 
"data_20_120.npy" and "label_20_120.npy" in our experiments.
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


