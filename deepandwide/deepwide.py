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

#常用参数
lr=1e-5
batch_size=16
input_size=100
timestep_size = 20
hidden_size = 100
layer_num=2
epoch = 100

##注意歌曲embedding提前用word2vec算好,利用听歌历史和歌单列表计算

#获取batch_size听歌序列，以及接下来1~10的预测歌曲，选择1~到10 是因为只选择接下来一首受上一首影响太大
def getSongVec():
    pass

#定义一个lstm模型
xinput = tf.placeholder(tf.float32,batch_size,timestep_size,input_size)
yout = tf.placeholder(tf.float32,hidden_size)
keep_prob = tf.placeholder(tf.float32)
lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size,state_is_tuple=True)
lstm_cell = rnn.DropoutWrapper(cell=lstm_cell, input_keep_prob=1.0, output_keep_prob=keep_prob)
mlstm_cell = rnn.MultiRNNCell([lstm_cell] * layer_num, state_is_tuple=True)
init_state = mlstm_cell.zero_state(batch_size, dtype=tf.float32)
outputs, state = tf.nn.dynamic_rnn(mlstm_cell, inputs=xinput, initial_state=init_state, time_major=False)
h_state = outputs[:, -1, :]
rmseloss = tf.sqrt(tf.reduce_mean(tf.square(h_state-yout)))
train = tf.train.AdamOptimizer(lr).minimize(rmseloss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        x_train,y_train=getSongVec()
        global_step=i
        #print(x_train,y_train)
      
        loss,_=sess.run([rmseloss,train],feed_dict={xinput:x_train,yout:y_train,keep_prob: 0.8})
        print('epoch=%d,loss=%f'%(i,loss))