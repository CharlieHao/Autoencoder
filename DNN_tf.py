#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: DNN:
#               pretrainig
# IDEA: use unsupervised learnng moel to pre-trainig supervised model

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from utils import get_MINST_data
from Autoencoder_tf import Autoencoder, error_rate	


class DNN(object):
    def __init__(self, D, hidden_layer_sizes, K, UnsupervisedModel=Autoencoder):
        self.hidden_layers = []
        count = 0
        input_size = D
        for output_size in hidden_layer_sizes:
            ae = UnsupervisedModel(input_size, output_size, count)
            self.hidden_layers.append(ae)
            count += 1
            input_size = output_size
        self.build_final_layer(D, hidden_layer_sizes[-1], K)

    def set_session(self, session):
        self.session = session
        for layer in self.hidden_layers:
            layer.init_session(session)

    def build_final_layer(self, D, M, K):
        # initialize logistic regression layer
        self.W = tf.Variable(tf.random_normal(shape=(M, K)))
        self.b = tf.Variable(np.zeros(K).astype(np.float32))

        self.X = tf.placeholder(tf.float32, shape=(None, D))
        labels = tf.placeholder(tf.int32, shape=(None,))
        self.Y = labels
        logits = self.forward(self.X)

        self.cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels
            )
        )
        self.train_op = tf.train.AdamOptimizer(1e-2).minimize(self.cost)
        self.prediction = tf.argmax(logits, 1)

    def fit(self, X, Y, Xtest, Ytest, pretrain=True, epochs=1, batch_sz=100):
        N = len(X)

        # greedy layer-wise training of autoencoders
        pretrain_epochs = 1
        if not pretrain:
            pretrain_epochs = 0

        current_input = X
        for ae in self.hidden_layers:
            ae.fit(current_input, epochs=pretrain_epochs)

            # create current_input for the next layer
            current_input = ae.transform(current_input)

        n_batches = N // batch_sz
        costs = []
        print("supervised training...")
        for i in range(epochs):
            print("epoch:", i)
            X, Y = shuffle(X, Y)
            for j in range(n_batches):
                Xbatch = X[j*batch_sz:(j*batch_sz + batch_sz)]
                Ybatch = Y[j*batch_sz:(j*batch_sz + batch_sz)]
                self.session.run(
                    self.train_op,
                    feed_dict={self.X: Xbatch, self.Y: Ybatch}
                )
                c, p = self.session.run(
                    (self.cost, self.prediction),
                    feed_dict={self.X: Xtest, self.Y: Ytest
                })
                error = error_rate(p, Ytest)
                if j % 10 == 0:
                    print("num of batches:", j, "cost:", c, "error:", error)
                costs.append(c)
        plt.plot(costs)
        plt.show()

    def forward(self, X):
        current_input = X
        for ae in self.hidden_layers:
            Z = ae.forward_hidden(current_input)
            current_input = Z

        # logistic layer
        logits = tf.matmul(current_input, self.W) + self.b
        return logits