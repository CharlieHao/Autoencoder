#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

# Descriptions: Autoencoder:
#               encoder is use a network to project high dimensional data onto 
# 				a nonlinear manifold in a low dimentional space
# Method: sparse represenation --> defferent method:
#               denoising autoencoder (DAE)
#				sparse autoencoder (SAE)			
#				contractive autoencoder (CAE)

import utils
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class Autoencoder2(object):
  def __init__(self, D, M):
    # represents a batch of training data
    self.X = tf.placeholder(tf.float32, shape=(None, D))

    # input -> hidden
    self.W = tf.Variable(tf.random_normal(shape=(D, M)) * np.sqrt(2.0 / M))
    self.b = tf.Variable(np.zeros(M).astype(np.float32))

    # hidden -> output
    self.V = tf.Variable(tf.random_normal(shape=(M, D)) * np.sqrt(2.0 / D))
    self.c = tf.Variable(np.zeros(D).astype(np.float32))

    # construct the reconstruction
    self.Z = tf.nn.relu(tf.matmul(self.X, self.W) + self.b)
    logits = tf.matmul(self.Z, self.V) + self.c
    self.X_hat = tf.nn.sigmoid(logits)

    # compute the cost
    self.cost = tf.reduce_sum(
      tf.nn.sigmoid_cross_entropy_with_logits(
        labels=self.X,
        logits=logits
      )
    )

    # make the trainer
    self.train_op = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(self.cost)

    # set up session and variables for later
    self.init_op = tf.global_variables_initializer()
    self.sess = tf.InteractiveSession()
    self.sess.run(self.init_op)

  def fit(self, X, epochs=30, batch_sz=64):
    costs = []
    n_batches = len(X) // batch_sz
    print("n_batches:", n_batches)
    for i in range(epochs):
      print("epoch:", i)
      np.random.shuffle(X)
      for j in range(n_batches):
        batch = X[j*batch_sz:(j+1)*batch_sz]
        _, c, = self.sess.run((self.train_op, self.cost), feed_dict={self.X: batch})
        c /= batch_sz # just debugging
        costs.append(c)
        if j % 100 == 0:
          print("iter: %d, cost: %.3f" % (j, c))
    plt.plot(costs)
    plt.show()

  def predict(self, X):
    return self.sess.run(self.X_hat, feed_dict={self.X: X})

















