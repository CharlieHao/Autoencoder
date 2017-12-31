#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: zehao
"""

import numpy as np 
import pandas as pd 
from sklearn.utils import shuffle
import os

def init_weights(shape):
	return np.random.randn(*shape)/np.sqrt(sum(shape))

def get_MINST_data():
	data = pd.read_csv('/Users/zehaodong/research/Autoencoder/data/train.csv').as_matrix().astype(np.float64)
	data = shuffle(data)

	Xtrain = data[:-1000,1:]/255.0
	Ytrain = data[:-1000,0].astype(np.int64)

	Xtest = data[-1000:,1:]/255.0
	Ytest = data[-1000:,0].astype(np.int64)

	return Xtrain,Ytrain,Xtest,Ytest


def get_mnist(limit=None):
	if not os.path.exists('../data'):
		print("create a folder called data adjacent to the class folder")
	if not os.path.exists('../data/train.csv'):
		print("please download the dataset.")
		print("from https://www.kaggle.com/c/digit-recognizer")

	print("Reading in and transforming data...")
	df = pd.read_csv('../data/train.csv')
	data = df.as_matrix()
	np.random.shuffle(data)
	X = data[:, 1:] / 255.0 # data is from 0..255
	Y = data[:, 0]
	if limit is not None:
		X, Y = X[:limit], Y[:limit]
	return X, Y

def error_rate(x,y):
	return np.mean(x != y)