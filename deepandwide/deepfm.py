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
import numpy as np;
import pandas as pd;
from numpy import loadtxt
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
import matplotlib.pylab as plt
import random
from sklearn.preprocessing import MinMaxScaler
#deepfm 包括 fm和dnn

class DeepFmModel():
	def __init__(self,is_training,batch_size,feature_size,class_num):
		'''fm部分 '''
		self.x = tf.placeholder(shape=[batch_size,feature_size],name='input',dtype=tf.float32)
		self.y = tf.placeholder(shape=[batch_size,class_num],name='yout',dtype=tf.float32)
		self.deep_layers=[256,128,64]
		self.is_training=is_training
		with tf.variable_scope('fm'):
			#n*k的embedding
			k=10
			embedding = tf.get_variable('embedding',shape=[feature_size,k],initializer=tf.random_normal_initializer(0.0, 0.01))
			#lr部分
			weight = tf.get_variable('weight',shape = [feature_size,1])
			bias = tf.get_variable('bias',shape=[1])
			left = tf.matmul(self.x,weight)+bias   #batch_size,1
			
			#fm右边
			sum_square_part = tf.square(tf.matmul(self.x,embedding))
			squared_sum_part = tf.matmul(tf.square(self.x),tf.square(embedding))
			print('sum_square_part.shape:',sum_square_part.get_shape(),squared_sum_part.get_shape())
			right = 0.5*tf.reduce_sum(tf.subtract(sum_square_part,squared_sum_part),axis=1) #batch_size,1
			right = tf.reshape(right,shape=[batch_size,1])
		with tf.variable_scope('dnn'):
			#embedding和input的交叉
			embedding = tf.matmul(self.x,embedding)
			ydeep = tf.reshape(embedding,shape=[-1,k])
			print('ydeep.shape:',ydeep.get_shape())
			for i in range(0, len(self.deep_layers)):
				ydeep = tf.contrib.layers.fully_connected(ydeep,  
						 self.deep_layers[i], activation_fn= 
						 tf.nn.relu, scope = 'fc%d' % i)  #batch_size,64
				#drop out
				ydeep = tf.contrib.layers.dropout(ydeep, keep_prob=0.7, is_training=is_training, scope='dropout%d'%i)
		
		with tf.variable_scope('deepfm'):
			print(left.get_shape(),right.get_shape(),ydeep.get_shape())
			deepfm = tf.concat([left,right,ydeep],axis=1)
			deepfm	= tf.contrib.layers.fully_connected(deepfm, class_num, \
			activation_fn=tf.nn.softmax, scope = 'deepfm_out')
			
		with tf.variable_scope('loss'):
			self.loss = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(deepfm),reduction_indices=[1]))
			self.train_op=tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.loss)
			correct_prediction = tf.equal(tf.argmax(deepfm,1), tf.argmax(self.y,1)) # 计算准确度
			self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			self.auc=tf.contrib.metrics.streaming_auc(deepfm,self.y)

#调用deepfm模型

fd = open('diss_data.txt')
data = fd.readlines()
dataset = np.array([line.strip().split(',') for line in data])

print(dataset[:, :])
Y = np.array(dataset[:, 0], dtype=float)
yout=[]
for i in Y:
	if i ==0:
		yout.append([0,1])
	else:
		yout.append([1,0])
Y = np.array(yout)
X = np.array(dataset[:, 1:], dtype=float)
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=19910825)
print('Y.shape',Y.shape)

batch_size=64
epoch_size = 10000

feature_size = X.shape[-1]

def getBatchData(xdata,ydata,dataIndex):
	random.shuffle(dataIndex)
	return xdata[dataIndex[:batch_size]],ydata[dataIndex[:batch_size]]

def runEpoch(session,model,xdata,ydata,epoch_size):
	totalloss = 0.0
	totalacc = 0.0
	dataIndex = [i for i in range(len(xdata))]
	for step in range(epoch_size):
		x,y=getBatchData(xdata,ydata,dataIndex)
		feed_dict = {}
		feed_dict[model.x]=x
		print('xy.shape',x.shape,y.shape)
		feed_dict[model.y]=y
		loss,acc,auc,_=session.run([model.loss,model.accuracy,model.auc,model.train_op],feed_dict=feed_dict)
		totalloss += loss
		totalacc += acc
		print('step=%d,loss=%f,acc=%f'%(step,loss,acc))
		print('auc=',auc)
	avgloss=totalloss/epoch_size
	avgacc = totalacc/epoch_size
	print('run finish,avgloss=%f,avgacc=%f'%(avgloss,avgacc))
	return avgloss,avgacc
	
	
	

with tf.variable_scope('trainfm'):
	trainmodel = DeepFmModel(True,batch_size,feature_size,2)
	
with tf.variable_scope('testfm'):
	testmodel = DeepFmModel(False,batch_size,feature_size,2)

config = tf.ConfigProto()	
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	sess.run(tf.local_variables_initializer())
	trainavgloss,trainavgacc = runEpoch(sess,trainmodel,X_train,y_train,epoch_size)
	#testavgloss,testavgacc = runEpoch(sess,testmodel,X_test,y_test,epoch_size//10)
			
		