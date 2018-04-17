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
dataIndex = [i for i in range(len(X_train))]
feature_size = X.shape[-1]

def getBatchData(xdata,ydata):
	random.shuffle(dataIndex)
	return xdata[dataIndex[:batch_size]],ydata[dataIndex[:batch_size]]

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
	trainmodel = DeepFmModel(True,batch_size,feature_size,2)
	
with tf.variable_scope('testfm'):
	testmodel = DeepFmModel(False,batch_size,feature_size,2)

config = tf.ConfigProto()	
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	trainavgloss,trainavgacc = runEpoch(sess,trainmodel,X_train,y_train,epoch_size)
	testavgloss,testavgacc = runEpoch(sess,testmodel,X_test,y_test,epoch_size/10)
			
		