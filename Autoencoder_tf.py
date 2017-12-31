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

import numpy as np
import tensorflow as tf 
import matplotlib.pyplot as plt 

from sklearn.utils import shuffle
from utils import get_MINST_data, error_rate

class Autoencoder(object):
	def __init__(self,D,M,an_id):
		self.M = M 
		self.D = D
		self.id = an_id
		self.build_structure(D,M)

	def init_session(self,session):
		self.session = session

	def build_structure(self,D,M):
		self.W = tf.Variable(tf.random_normal(shape=(D,M)))
		self.b_en = tf.Variable(np.zeros(M).astype(np.float32))
		self.b_de = tf.Variable(np.zeros(D).astype(np.float32))

		self.tfX = tf.placeholder(tf.float32,shape=(None,D),name='X')
		self.Z = self.forward_hidden(self.tfX)
		self.X_hat = self.forward_output(self.tfX)

		# cost_op:
		# this is a non sparse type. we can use regulizer like L1, student-t and KL-divergebcy
		# to add sparsity on hidden layer 
		logits = self.forward_logits(self.tfX)
		self.cost_op = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(
				logits = logits,
				labels = self.tfX,
			)
		)
		# train_op
		self.train_op = tf.train.MomentumOptimizer(10e-4, momentum=0.9).minimize(self.cost_op)
		# self.train_op = tf.train.AdamOptimizer(1e-1).minimize(self.cost_op)


	def fit(self,X,epochs=3,batch_size=100,show_fig=False):
		N,D = X.shape
		n_batches = int(N/batch_size)

		costs = []
		for i in range(epochs):
			epo = i
			print("epoch:", epo)
			X = shuffle(X)
			for j in range(n_batches):
				Xbatch = X[j*batch_size:(j*batch_size + batch_size)]
				_, c = self.session.run((self.train_op, self.cost_op), feed_dict={self.tfX: Xbatch})
				if j % 10 == 0:
					print("num of batches:", j,  "cost:", c)
				costs.append(c)
		if show_fig:
			plt.plot(costs)
			plt.show()

	def transform(self,X):
		# generate encoded data
		return self.session.run(self.Z,feed_dict={self.tfX:X})

	def predict(self,X):
		# generate transformed data
		return self.session.run(self.X_hat,feed_dict={self.tfX:X})

	def forward_hidden(self,X):
		return tf.nn.sigmoid(tf.matmul(X,self.W)+self.b_en)

	def forward_logits(self,X):
		Z = self.forward_hidden(X)
		return tf.matmul(Z,tf.transpose(self.W))+self.b_de

	def forward_output(self,X):
		return tf.nn.sigmoid(self.forward_logits(X))

















