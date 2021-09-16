#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/9/16 11:20
# @Author  : Huatao
# @Email   : 735820057@qq.com
# @File    : pretrain_base.py
# @Description :
import argparse

import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot

SENSORS = ["Accelerometer", "Gyroscope", "Magnetometer"]
SENSOR_NAMES = ['ACC-X', 'ACC-Y', 'ACC-Z', 'GYRO-X', 'GYRO-Y', 'GYRO-Z', 'MAG-X', 'MAG-Y', 'MAG-Z']
COLOR_BLUE = 'tab:blue'; COLOR_ORANGE = 'tab:orange'; COLOR_GREEN = 'tab:green'
COLOR_RED = 'tab:red'; COLOR_PURPLE = 'tab:purple'
COLOR_LIST = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED]
LINE_STYLES = ['solid', 'dotted']


def plot_tsne(data, labels, dimension=2, label_names=None):
    tsne = TSNE(n_components=dimension)
    data_ = tsne.fit_transform(data)
    ls = np.unique(labels)
    plt.figure()
    bwith = 2
    TK = plt.gca()
    TK.spines['bottom'].set_linewidth(bwith)
    TK.spines['left'].set_linewidth(bwith)
    TK.spines['top'].set_linewidth(bwith)
    TK.spines['right'].set_linewidth(bwith)
    for i in range(ls.size):
        index = labels == ls[i]
        x = data_[index, 0]
        y = data_[index, 1]
        if label_names is None:
            plt.scatter(x, y, label=str(int(ls[i])))
        else:
            plt.scatter(x, y, label=label_names[int(ls[i])])
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='lower right') #, prop={'size': 20, 'weight':'bold'}
    plt.show()
    return data_


def plot_pca(data, labels, dimension=2):
    pca = PCA(n_components=dimension)
    data_ = pca.fit_transform(data)
    ls = np.unique(labels)
    plt.figure()
    for i in range(ls.size):
        index = labels == ls[i]
        x = data_[index, 0]
        y = data_[index, 1]
        plt.scatter(x, y, label=str(ls[i]))
    plt.show()
    # plt.close()


def plot_matrix(matrix, labels_name=None):
    plt.figure()
    row_sum = matrix.sum(axis=1)
    matrix_per = np.copy(matrix).astype('float')
    for i in range(row_sum.size):
        if row_sum[i] != 0:
            matrix_per[i] = matrix_per[i] / row_sum[i]
    # plt.figure(figsize=(10, 7))
    if labels_name is None:
        labels_name = "auto"
    sn.heatmap(matrix_per, annot=True, fmt='.2f', xticklabels=labels_name, yticklabels=labels_name)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    # plt.savefig()
    return matrix


def plot_embedding(embeddings, labels, label_index=0, reduce=1000, label_names=None):
    embeddings = embeddings.reshape(embeddings.shape[0], embeddings.shape[1] * embeddings.shape[2])
    index_rand = np.arange(embeddings.shape[0])
    np.random.shuffle(index_rand)
    index_rand = index_rand[:reduce]
    if isinstance(label_index, list):
        label_composite = np.zeros(labels.shape[0])
        for i in range(len(label_index)):
            label_composite += labels[:, 0, label_index[i]] * pow(10, len(label_index) - 1 - i)
        plot_tsne(embeddings[index_rand, :], label_composite[index_rand])
        return None
    else:
        data_tsne = plot_tsne(embeddings[index_rand, :], labels[index_rand, 0, label_index], label_names=label_names)
        return data_tsne, labels[index_rand, 0, label_index]
        # plot_pca(embeddings[index_rand, :], labels[index_rand, label_index])


def plot_reconstruct_sensor(sensors, sensors_re, sensor_dimen=3):
    sensor_num = sensors.shape[1] // 3
    fig, axs = plt.subplots(sensor_num)
    fig.suptitle('IMU Sensor Data') #Sensor Reconstruction Comparison
    x = np.arange(sensors.shape[0])
    for i in range(sensor_num):
        index_start = i * sensor_dimen
        axs[i].set_xlabel("Index")
        axs[i].set_ylabel(SENSORS[i])
        for j in range(sensor_dimen):
            dimen = index_start + j
            axs[i].plot(x, sensors[:, dimen], label=SENSOR_NAMES[dimen], linestyle=LINE_STYLES[0], color=COLOR_LIST[j]) #
            axs[i].plot(x, sensors_re[:, dimen], label=SENSOR_NAMES[dimen], linestyle=LINE_STYLES[1], color=COLOR_LIST[j])
    plt.show()


def plot_nlp_training_method():
    m_sensor_wp = np.array([[95.3, 90.0, 88.7], [95.5, 90.0, 88.2]])
    m_sensor_p = np.array([[95.2, 82.6, 83.6], [95.4, 82.5, 83.1]])
    m_all_wp = np.array([[95.0, 84.6, 84.5], [95.2, 84.4, 84.0]])
    m_all_p = np.array([[95.1, 82.8, 82.1], [94.8, 82.6, 81.4]])
    data = np.array([m_sensor_wp, m_sensor_p, m_all_wp, m_all_p])
    index = 1
    labels = ['HAR', 'DPC', 'UI']
    labels_data = ['P-WP', 'P-P', 'A-WP', 'A-P']
    men_means = [20, 34, 30, 35, 27]
    women_means = [25, 32, 34, 20, 25]

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars

    fig, ax = plt.subplots()
    for i in range(len(labels)):
        for j in range(len(labels_data)):
            xp = x[i] + (j - np.mean(x)) * width
            if i == 0:
                ax.bar(xp, data[j, index, i], width, label=labels_data[j], color=COLOR_LIST[j])
            else:
                ax.bar(xp, data[j, index, i], width, color=COLOR_LIST[j])
    # rects1 = ax.bar(x - width / 2, men_means, width, label='Men')
    # rects2 = ax.bar(x + width / 2, women_means, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim([80, 100])
    ax.set_ylabel('Scores', fontsize=14, fontweight='bold')
    ax.set_title('F1-score', fontsize=18, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, fontweight='bold')
    # ax.tick_params(axis='y', which='minor', labelsize=20)
    ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()


def plot_roc_auc(y_pred, y_true):
    auc = metrics.roc_auc_score(y_true, y_pred)
    print('ROC AUC=%.3f' % (auc))
    fpr, tpr, thre = metrics.roc_curve(y_true, y_pred)
    pyplot.plot(fpr, tpr, marker='.')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.show()
    return fpr, tpr, thre




if __name__ == "__main__":
    plot_nlp_training_method()