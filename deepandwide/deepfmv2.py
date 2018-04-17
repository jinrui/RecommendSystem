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
	def __init__(self,is_training,batch_size,feature_size,class_num,deep_layers=[256,128,64],dropout_keep_deep=[1,1,1,1],dropout_keep_fm=[1,1]):
		self.x = tf.placeholder(shape=[batch_size,feature_size],name='input',dtype=tf.float32)
		self.y = tf.placeholder(shape=[batch_size,class_num],name='yout',dtype=tf.float32)
		embedding_size=10		
		# embeddings
		embeddings = tf.Variable(
			tf.random_normal([feature_size,embedding_size], 0.0, 0.01),
			name="feature_embeddings")

		# model
		#self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],self.feat_index)
		# None * F * Ks						
		feat_value = tf.reshape(self.x, shape=[-1, feature_size, 1])
		embeddings = tf.multiply(embeddings, feat_value)

		# ---------- first order term ----------
		feature_bias = tf.Variable(tf.random_uniform([feature_size, 1], 0.0, 1.0), name="feature_bias_0")  # feature_size * 1
		y_first_order = feature_bias
		y_first_order = tf.reduce_sum(tf.multiply(y_first_order, feat_value), 2)  # None * F
		y_first_order = tf.nn.dropout(y_first_order, dropout_keep_fm[0]) # None * F

		# ---------- second order term ---------------
		# sum_square part
		summed_features_emb = tf.reduce_sum(embeddings, 1)  # None * K
		summed_features_emb_square = tf.square(summed_features_emb)  # None * K

		# square_sum part
		squared_features_emb = tf.square(embeddings)
		squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

		# second order
		y_second_order = 0.5 * tf.subtract(summed_features_emb_square, squared_sum_features_emb)  # None * K
		y_second_order = tf.nn.dropout(y_second_order, dropout_keep_fm[1])  # None * K
	
		with tf.name_scope("deep"):
			
			# ---------- Deep component ----------
			y_deep = tf.reshape(embeddings, shape=[-1, feature_size*embedding_size]) # None * (F*K)
			y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[0])
			
			weights = dict()
		
			input_size = feature_size * embedding_size
			glorot = np.sqrt(2.0 / (input_size + deep_layers[0]))
			weights["layer_0"] = tf.Variable(
				np.random.normal(loc=0, scale=glorot, size=(input_size, deep_layers[0])), dtype=np.float32, name="weights_layer0")
			weights["bias_0"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[0])),
															dtype=np.float32,name="weights_bias0")

			num_layer = len(deep_layers)
			for i in range(1, num_layer): 
				glorot = np.sqrt(2.0 / (deep_layers[i-1] + deep_layers[i]))
				weights["layer_%d" % i] = tf.Variable(
					np.random.normal(loc=0, scale=glorot, size=(deep_layers[i-1], deep_layers[i])),
					dtype=np.float32 ,name="weights_layer"+str(i))  # layers[i-1] * layers[i]
				weights["bias_%d" % i] = tf.Variable(
					np.random.normal(loc=0, scale=glorot, size=(1, deep_layers[i])),
					dtype=np.float32 ,name="weights_bias"+str(i))  # 1 * layer[i]
			
			for i in range(0, len(deep_layers)):		   
				y_deep = tf.add(tf.matmul(y_deep, weights["layer_%d" %i]), weights["bias_%d"%i]) # None * layer[i] * 1
					#if self.batch_norm:
					#	self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
				y_deep = tf.nn.relu(y_deep)
				y_deep = tf.nn.dropout(y_deep, dropout_keep_deep[1+i]) # dropout at each Deep layer

		# ---------- DeepFM ----------
		MODETYPE=0
		with tf.name_scope("deepfm"):
			concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
			if MODETYPE==0:
				concat_input = tf.concat([y_first_order, y_second_order, y_deep], axis=1)
				input_size = feature_size + embedding_size + deep_layers[-1]
			elif MODETYPE==1:
				concat_input = tf.concat([y_first_order, y_second_order], axis=1)
				input_size = feature_size + embedding_size 
			elif MODETYPE==2:
				concat_input = y_deep   
				input_size =  deep_layers[-1]
			
			glorot = np.sqrt(2.0 / (input_size + 1))
			weights["concat_projection"] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(input_size, 1)),
							dtype=np.float32 ,name="concat_projection0")  # layers[i-1]*layers[i]
			weights["concat_bias"] = tf.Variable(tf.constant(0.01), dtype=np.float32 ,name="concat_bias0")	
			print('concat_input.shape',concat_input.get_shape(),weights["concat_projection"].get_shape(),weights["concat_bias"].get_shape())
			out = tf.add(tf.matmul(concat_input, weights["concat_projection"]), weights["concat_bias"],name='out')

		score=tf.nn.sigmoid(out,name='score')
		##观看变量
		with tf.name_scope('estimate_name'):
		# 损失函数的定义：均方差
			self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - score),reduction_indices=[1]))
			##观看常量
			tf.summary.scalar('loss',self.loss)
			#correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(self.y,1)) # 计算准确度
			#self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			y=tf.cast(self.y, tf.int32)
			score = tf.round(score)
			score=tf.cast(score, tf.int32)
			self.accuracy  = tf.contrib.metrics.accuracy(score, y)
			self.auc = tf.contrib.metrics.streaming_auc(score,tf.convert_to_tensor(y))   
			##观看常量
			tf.summary.scalar('auc1',self.auc[0])
			tf.summary.scalar('auc2',self.auc[1])


		with tf.name_scope("train"):
			self.train_op = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999,epsilon=1e-8).minimize(self.loss)

#调用deepfm模型

fd = open('diss_data.txt')
data = fd.readlines()
dataset = np.array([line.strip().split(',') for line in data])
batch_size=64
print(dataset[:, :])
Y = np.array(dataset[:, 0], dtype=float)
yout=[]
for i in Y:
	if i ==0:
		yout.append([0,1])
	else:
		yout.append([1,0])
#Y = np.array(yout)
X = np.array(dataset[:, 1:], dtype=float)
scaler = MinMaxScaler()
scaler.fit(X)
X=scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=19910825)
print('Y.shape',Y.shape)


epoch_size = 1000
dataIndex = [i for i in range(len(X_train))]
feature_size = X.shape[-1]

def getBatchData(xdata,ydata):
	random.shuffle(dataIndex)
	return xdata[dataIndex[:batch_size]],ydata[dataIndex[:batch_size]].reshape([batch_size,1])

def runEpoch(session,model,xdata,ydata,epoch_size):
	totalloss = 0.0
	totalacc = 0.0
	for step in range(epoch_size):
		x,y=getBatchData(xdata,ydata)
		feed_dict = {}
		feed_dict[model.x]=x
		print('xy.shape',x.shape,y.shape)
		feed_dict[model.y]=y
		loss,acc,_=session.run([model.loss,model.accuracy,model.train_op],feed_dict=feed_dict)
		totalloss += loss
		totalacc += acc
		print('step=%d,loss=%f,acc=%f'%(step,loss,acc))
	avgloss=totalloss/epoch_size
	avgacc = totalacc/epoch_size
	print('run finish,avgloss=%f,avgacc=%f'%(avgloss,avgacc))
	return avgloss,avgacc
	
	
	

with tf.variable_scope('trainfm'):
	trainmodel = DeepFmModel(True,batch_size,feature_size,1)
	
with tf.variable_scope('testfm'):
	testmodel = DeepFmModel(False,batch_size,feature_size,1)

config = tf.ConfigProto()	
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	trainavgloss,trainavgacc = runEpoch(sess,trainmodel,X_train,y_train,epoch_size)
	testavgloss,testavgacc = runEpoch(sess,testmodel,X_test,y_test,epoch_size//10)
			
		