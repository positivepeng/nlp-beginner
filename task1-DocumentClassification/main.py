#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Created on 2020/4/15 21:24
@author: phil
"""

import numpy as np
from collections import Counter
from data_preprocess import read_data
from feature_extraction import BagOfWord, NGram
from softmax_regerssion import SoftmaxRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    debug = 1
    # 读入数据
    X_data, y_data = read_data()

    if debug == 1:
        # index = np.arange(len(X_data))
        # np.random.shuffle(index)
        # X_data = X_data[index[:2000]]
        # y_data = y_data[index[:2000]]
        X_data = X_data[:2000]
        y_data = y_data[:2000]
    y = np.array(y_data).reshape(len(y_data), 1)

    # 数据集划分
    bag_of_word_model = BagOfWord()
    ngram_model = NGram(ngram=(1, 2))
    X_Bow = bag_of_word_model.fit_transform(X_data)
    X_Gram = ngram_model.fit_transform(X_data)

    X_train_Bow, X_test_Bow, y_train_Bow, y_test_Bow = train_test_split(X_Bow, y, test_size=0.33, random_state=42, stratify=y)
    X_train_Gram, X_test_Gram, y_train_Gram, y_test_Gram = train_test_split(X_Gram, y, test_size=0.33, random_state=42, stratify=y)

    # 训练模型 不同特征的差别
    epoch = 100
    bow_learning_rate = 1
    gram_learning_rate = 0.5
    # 梯度下降
    model1 = SoftmaxRegression()
    history = model1.fit(X_train_Bow, y_train_Bow, epoch=epoch, learning_rate=bow_learning_rate, print_loss_steps=epoch//10, update_strategy="stochastic_batch")
    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()
    print("Bow train {} test {}".format(model1.score(X_train_Bow, y_train_Bow), model1.score(X_test_Bow, y_test_Bow)))

    model2 = SoftmaxRegression()
    history = model2.fit(X_train_Gram, y_train_Gram, epoch=epoch, learning_rate=gram_learning_rate, print_loss_steps=epoch//10, update_strategy="stochastic_batch")
    plt.plot(np.arange(len(history)), np.array(history))
    plt.show()
    print("Gram train {} test {}".format(model2.score(X_train_Gram, y_train_Gram), model2.score(X_test_Gram, y_test_Gram)))

    # Bow  train 0.8552238805970149 test 0.5045454545454545
    # Gram train 1.0                test 0.49696969696969695