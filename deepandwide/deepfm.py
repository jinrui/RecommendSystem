# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 23:30:50 2018

@author: jimmyjin
"""

#http://blog.csdn.net/m0_37744293/article/details/69950262 代码参考
#tf.contrib.learn.DNNLinearCombinedClassifier  wide&deep

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.contrib import rnn


#deepfm 包括 fm和dnn

class DeepFmModel():
    def __init__(self,is_training,batch_size,feature_size,class_num):
        '''fm部分 '''
        x = tf.placeholder(shape=[batch_size,feature_size],name='input')
        y = tf.placeholder(shape=[batch_size,class_num])
        self.deep_layers=[256,128,64]
        with tf.name_score('fm'):
            #n*k的embedding
            k=10
            embedding = tf.get_variable('embedding',shape=[feature_size,k],initializer=tf.random_normal_initializer(0.0, 0.01))
            #lr部分
            weight = tf.get_variable('weight',shape = [feature_size,1])
            bias = tf.get_variable('bias',shape=[1])
            left = tf.matmul(x,weight)+bias   #batch_size,1
            
            #fm右边
            sum_square_part = tf.square(tf.matmul(x,embedding))
            squared_sum_part = tf.matmul(tf.square(x),tf.square(embedding))
            right = 0.5*tf.reduce_sum(tf.substrract(sum_square_part,squared_sum_part),axis=1) #batch_size,1
            
        with tf.name_scope('dnn'):
            ydeep = tf.reshape(embedding,shape=[-1,feature_size*k])
            for i in range(0, len(self.deep_layers)):
                ydeep = tf.contrib.layers.fully_connected(ydeep,  
                         self.deep_layers[i], activation_fn= 
                         tf.nn.relu, scope = 'fc%d' % i)  #batch_size,64
                #drop out
                ydeep = tf.contrib.layers.dropout(ydeep, keep_prob=0.5, is_training=is_training, scope='dropout%d'%i)
        
        with tf.name_score('deepfm'):
            deepfm = tf.concat([left,right,ydeep],axis=1)
            deepfm    = tf.contrib.layers.fully_connected(deepfm, class_num, \
            activation_fn=tf.nn.softmax, scope = 'deepfm_out')
            
        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(deepfm),reduction_indices=[1]))
            self.train_op=tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(loss)
            
        