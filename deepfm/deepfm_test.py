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
from sklearn.preprocessing import MinMaxScaler,StandardScaler
#deepfm 包括 fm和dnn
from time import time
import json
from tensorflow.contrib.session_bundle import session_bundle
#获取均值方差
#with open("./meanandstand.json",'r') as load_f:
#	usr_list_obj=json.load(load_f)
#	print(usr_list_obj[0],usr_list_obj[1])
	#json_usr_data_str=json.dumps(usr_list_obj)


fd = open('diss_data.txt')
data = fd.readlines()
dataset = np.array([line.strip().split(',') for line in data])

print(dataset[:, :])
Y = np.array(dataset[:, 0], dtype=float).reshape((-1,1))

X = np.array(dataset[:, 1:], dtype=float)
export_dir='save_erjinzhi/1525340333/'
#sess, meta_graph_def = tf.contrib.session_bundle.session_bundle.load_session_bundle_from_path(export_dir)
#for op in sess.graph.get_operations():  
#	print(op.name)  
#tf.reset_default_graph()
#tf.train.import_meta_graph(meta_graph_def)
#sess.run(tf.global_variables_initializer()) 
#saver=tf.train.import_meta_graph(meta_graph_def)
#saver.restore(sess,export_dir+'checkpoint')  
#input_graph_def = meta_graph_def.graph_def
#collection_def = meta_graph_def.collection_def
#signatures_any = collection_def['inputs'].any_list.value
#input_x = sess.graph.get_tensor_by_name('trainfm/input:0')
#op = sess.graph.get_operation_by_name('trainfm/deepfm_sigmoid/deepfm_out/Sigmoid').outputs[0]
#op=tf.get_collection("outputs")
#print(collection_def)

with tf.Session() as sess:
	new_saver=tf.train.import_meta_graph(export_dir+'export.meta')
	new_saver.restore(sess, tf.train.latest_checkpoint(export_dir))
	graph = tf.get_default_graph()
	input_x=graph.get_operation_by_name('trainfm/input').outputs[0]
	for op in tf.get_collection("means"):
		print(op)
		print('hehe')
	op=tf.get_collection("outputs")[0]
	mean=tf.get_collection("means")[0].eval()
	var=tf.get_collection("vars")[0].eval()
	print(mean,var)
	ip = X[0:64]
	print(ip)
	ret = sess.run(op,  feed_dict={input_x: ip})
	print(ret)

#print(meta_graph_def)
#print(signatures_any)
#print(type(meta_graph_def))
#result=sess.run(['outputs'],feed_dict={'inputs':X[0:2]})
#print(result)
			