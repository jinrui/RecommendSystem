# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:30:50 2018

@author: jimmyjin
"""

# http://blog.csdn.net/m0_37744293/article/details/69950262 代码参考
# tf.contrib.learn.DNNLinearCombinedClassifier  wide&deep

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn
import numpy as np;
import pandas as pd;
from numpy import loadtxt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from time import time

TIME = int(time())

#其中hidden_dim就是我们矩阵分解的隐含维度k。user_emb_w对应矩阵W, item_emb_w对应矩阵H
class BprModel():
    def __init__(self,user_count, item_count, hidden_dim):
        u = tf.placeholder(tf.int32, [None])
        i = tf.placeholder(tf.int32, [None])
        j = tf.placeholder(tf.int32, [None])
    
        with tf.device("/cpu:0"):
            user_emb_w = tf.get_variable("user_emb_w", [user_count+1, hidden_dim], 
                                initializer=tf.random_normal_initializer(0, 0.1))
            item_emb_w = tf.get_variable("item_emb_w", [item_count+1, hidden_dim], 
                                    initializer=tf.random_normal_initializer(0, 0.1))
            
            u_emb = tf.nn.embedding_lookup(user_emb_w, u)
            i_emb = tf.nn.embedding_lookup(item_emb_w, i)
            j_emb = tf.nn.embedding_lookup(item_emb_w, j)
        
        # MF predict: u_i > u_j
        x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)
        
        # AUC for one user:
        # reasonable iff all (u,i,j) pairs are from the same user
        # 
        # average AUC = mean( auc for each user in test set)
        mf_auc = tf.reduce_mean(tf.to_float(x > 0))
        
        l2_norm = tf.add_n([
                tf.reduce_sum(tf.multiply(u_emb, u_emb)), 
                tf.reduce_sum(tf.multiply(i_emb, i_emb)),
                tf.reduce_sum(tf.multiply(j_emb, j_emb))
            ])
        
        regulation_rate = 0.0001
        #这个是经典公式
        bprloss = tf.reduce_mean(regulation_rate * l2_norm - tf.reduce_sum(tf.log(tf.sigmoid(x))))
        
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)
        return u, i, j, mf_auc, bprloss, train_op