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
# deepfm 包括 fm和dnn
from time import time

TIME = int(time())


class DeepFmModel():
    def __init__(self, is_training, batch_size, feature_size, class_num):
        '''fm部分 '''
        self.x = tf.placeholder(shape=[None, feature_size], name='input', dtype=tf.float32)
        self.y = tf.placeholder(shape=[None, class_num], name='yout', dtype=tf.float32)
        self.mean = tf.constant([1.41900000e-01,2.76489000e+00,1.45494000e+00,3.41649000e+00
,8.91310100e+01,1.20244229e+04,4.96493263e+04,1.32830602e+03
,1.64125260e+04,1.27164917e+03,2.52711540e+03,1.18072681e+03
,3.49986033e+03,1.20622909e+03,1.68196160e+02,1.97378500e+01
,6.55865000e+00,1.07977000e+00,7.17740023e-01,5.31860485e-01
,4.01318657e-01,1.08059124e+04,1.37007229e+03,3.34992789e+03
,1.31444132e+03,5.09621880e+02,1.18217997e+03,8.98539500e+02
,2.83424320e+02,3.99004400e+01,5.16797000e+00,1.35941000e+00
,0.00000000e+00,2.60931421e+04,1.33432233e+03,8.18998762e+03
,1.27679907e+03,1.25178771e+03,1.18321788e+03,1.84239671e+03
,5.97872900e+02,8.19491100e+01,1.09542800e+01,3.03818000e+00
,0.00000000e+00,3.90053132e+04,1.33483476e+03,1.22466687e+04
,1.27837867e+03,1.85710692e+03,1.18523939e+03,2.94579968e+03
,9.55618020e+02,1.31514920e+02,1.76710400e+01,4.90241000e+00
,0.00000000e+00,8.77204224e+03,1.34380718e+03,2.75554004e+03
,1.28690541e+03,4.21939710e+02,1.18658134e+03,6.11670310e+02
,1.99503420e+02,2.74738300e+01,3.62413000e+00,1.01161000e+00
,0.00000000e+00])
        self.var = tf.constant([1.21764390e-01,3.30861329e+00,2.74649596e-01,2.60266608e+00
,1.70415836e+04,4.65662345e+08,7.53146170e+09,2.54878274e+05
,1.67620245e+09,2.25115355e+05,5.42565576e+07,1.92928286e+05
,8.33060945e+07,1.16723183e+07,5.76424342e+05,3.06725143e+03
,4.43014820e+02,5.88126267e+01,1.37220857e-01,1.10856912e-01
,1.02406882e-01,4.78093522e+08,4.15389629e+05,7.18939274e+07
,4.00137346e+05,3.07986371e+06,4.01758750e+05,1.04799743e+07
,9.31638616e+05,5.63944716e+04,4.07459716e+02,3.10449945e+01
,0.00000000e+00,2.01437102e+09,2.72069440e+05,4.03989580e+08
,2.53318847e+05,1.28937703e+07,2.63315748e+05,2.31048831e+07
,2.88986074e+06,1.32676908e+05,9.98526170e+02,1.04564102e+02
,0.00000000e+00,5.17493938e+09,2.64553258e+05,9.91883118e+08
,2.49011134e+05,3.34741539e+07,2.31743158e+05,6.46396828e+07
,7.93600155e+06,3.87193985e+05,2.81508071e+03,2.98611746e+02
,0.00000000e+00,2.36089138e+08,3.16765399e+05,4.89131104e+07
,3.03993144e+05,1.55968920e+06,3.51114369e+05,2.57318677e+06
,3.35763263e+05,1.58302489e+04,1.14234212e+02,1.39003552e+01
,0.00000000e+00])
        self.xx = (self.x-self.mean)/tf.sqrt(self.var)
        self.xx=tf.where(tf.is_nan(self.xx), tf.zeros_like(self.xx), self.xx)
        #self.xx = (self.x-self.mean)/self.var
        # self.jimmy_deepfm_out = tf.Variable(0.0,name='jimmy_deepfm_out')
        self.deep_layers = [256, 128, 64]
        self.is_training = tf.placeholder_with_default(False, shape=(), name="is_training")
        with tf.variable_scope('fm'):
            # n*k的embedding
            k = 10
            embedding = tf.get_variable('embedding', shape=[feature_size, k],
                                        initializer=tf.random_normal_initializer(0.0, 0.01))
            # lr部分
            weight = tf.get_variable('weight', shape=[feature_size, 1])
            bias = tf.get_variable('bias', shape=[1])
            left = tf.matmul(self.xx, weight) + bias  # batch_size,1

            # fm右边
            sum_square_part = tf.square(tf.matmul(self.xx, embedding))
            squared_sum_part = tf.matmul(tf.square(self.xx), tf.square(embedding))
            print('sum_square_part.shape:', sum_square_part.get_shape(), squared_sum_part.get_shape())
            right = 0.5 * tf.reduce_sum(tf.subtract(sum_square_part, squared_sum_part), axis=1)  # batch_size,1
            right = tf.reshape(right, shape=[-1, 1])
        with tf.variable_scope('dnn'):
            # embedding和input的交叉
            embedding = tf.matmul(self.xx, embedding)
            ydeep = tf.reshape(embedding, shape=[-1, k])
            print('ydeep.shape:', ydeep.get_shape())
            for i in range(0, len(self.deep_layers)):
                ydeep = tf.contrib.layers.fully_connected(ydeep,
                                                          self.deep_layers[i], activation_fn=
                                                          tf.nn.relu, scope='fc%d' % i)  # batch_size,64
                # drop out
                ydeep = tf.contrib.layers.dropout(ydeep, keep_prob=0.5, is_training=self.is_training, scope='dropout%d' % i)

        with tf.variable_scope('deepfm_sigmoid'):
            print(left.get_shape(), right.get_shape(), ydeep.get_shape())
            deepfm = tf.concat([left, right, ydeep], axis=1)
            self.jimmy_deepfm_out = tf.contrib.layers.fully_connected(deepfm, class_num, \
                                                                      activation_fn=tf.nn.sigmoid, scope='deepfm_out')
            tf.add_to_collection('outputs', self.jimmy_deepfm_out)  # 用于加载模型获取要预测的网络结构

        with tf.variable_scope('loss'):
            # self.loss = tf.reduce_mean(-tf.reduce_sum(self.y*tf.log(deepfm),reduction_indices=[1]))
            self.loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.y - self.jimmy_deepfm_out), reduction_indices=[1]))
            self.train_op = tf.no_op
            if is_training:
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                                       epsilon=1e-8).minimize(self.loss)
            ##观看常量
            tf.summary.scalar('loss', self.loss)
            # correct_prediction = tf.equal(tf.argmax(score,1), tf.argmax(self.y,1)) # 计算准确度
            # self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            y = tf.cast(self.y, tf.int32)
            deepfm1 = tf.round(self.jimmy_deepfm_out)
            deepfm1 = tf.cast(deepfm1, tf.int32)
            self.accuracy = tf.contrib.metrics.accuracy(deepfm1, y)
            self.auc = tf.contrib.metrics.streaming_auc(self.jimmy_deepfm_out, tf.convert_to_tensor(self.y))

            # 调用deepfm模型


fd = open('diss_data.txt')
data = fd.readlines()
dataset = np.array([line.strip().split(',') for line in data])

print(dataset[:, :])
Y = np.array(dataset[:, 0], dtype=float).reshape((-1, 1))
# print(Y)
yout = np.zeros((Y.shape[0], 1))
for i in range(len(Y)):
    if Y[i] < 100:
        yout[i][0] = 0
    if Y[i] > 1000:
        yout[i][0] = 1
# Y = np.array(yout)
X = np.array(dataset[:, 1:], dtype=float)
scaler = MinMaxScaler()
stand = StandardScaler()
stand.fit(X)
#X = stand.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=19910825)
# print('Y.shape',Y.shape)
# print(X_train[0:2])
batch_size = 64
epoch_size = 10000
print(stand.mean_)
print(stand.var_)
tf_mean = tf.constant(stand.mean_)
tf_var = tf.constant(stand.var_)
feature_size = X.shape[-1]
tf.add_to_collection('means', tf_mean)  # 用于加载模型获取要预测的网络结构
tf.add_to_collection('vars', tf_var)  # 用于加载模型获取要预测的网络结构

def getBatchData(xdata, ydata, dataIndex):
    random.shuffle(dataIndex)
    return xdata[dataIndex[:batch_size]], ydata[dataIndex[:batch_size]]


def runEpoch(is_training,session, model, xdata, ydata, epoch_size):
    totalloss = 0.0
    totalacc = 0.0
    totalauc = 0.0
    dataIndex = [i for i in range(len(xdata))]
    for step in range(epoch_size):
        x, y = getBatchData(xdata, ydata, dataIndex)
        feed_dict = {}
        feed_dict[model.x] = x
        print('xy.shape', x.shape, y.shape)
        feed_dict[model.y] = y
        feed_dict[model.is_training] = is_training
        loss, acc, auc,xx, _ = session.run([model.loss, model.accuracy, model.auc,model.xx,model.train_op], feed_dict=feed_dict)
        totalloss += loss
        totalacc += acc
        totalauc += auc[0]
        print('step=%d,loss=%f,acc=%f' % (step, loss, acc))
        print('auc=', auc)
        print('model.x=', xx)
    # print('deep_out=',deep_out)
    avgloss = totalloss / epoch_size
    avgacc = totalacc / epoch_size
    avgauc = totalauc / epoch_size
    print('run finish,avgloss=%f,avgacc=%f,avgauc=%f' % (avgloss, avgacc, avgauc))
    return avgloss, avgacc


with tf.variable_scope('trainfm'):
    trainmodel = DeepFmModel(True, batch_size, feature_size, 1)

#with tf.variable_scope('testfm'):
 #   testmodel = DeepFmModel(False, batch_size, feature_size, 1)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    trainavgloss, trainavgacc = runEpoch(True, sess, trainmodel, X_train, y_train, epoch_size)
    # builder = tf.saved_model.builder.SavedModelBuilder('./model')
    # builder.add_meta_graph_and_variables(sess, ['tag_string'])
    # builder.save()
    # 按时间戳保存模型文件，提供给tesorflow serving
    model_dir = './save' + "/" + str(TIME)
    export_path = './save_erjinzhi/'
    builder = tf.saved_model.builder.SavedModelBuilder(model_dir)
    builder.add_meta_graph_and_variables(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            "serving_default": tf.saved_model.signature_def_utils.predict_signature_def(
                # inputs is feature vector
                inputs={"inputs": trainmodel.x},
                # outputs is score
                outputs={"results": trainmodel.jimmy_deepfm_out})
        })
    builder.save()
    saver = tf.train.Saver()

    model_exporter = tf.contrib.session_bundle.exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': tf.contrib.session_bundle.exporter.generic_signature({'inputs': trainmodel.x}),
            'outputs': tf.contrib.session_bundle.exporter.generic_signature({'outputs': trainmodel.jimmy_deepfm_out})
        }),

    model_version = TIME
    print('Exporting trained model to', export_path)
    model_exporter.export(export_path, tf.constant(model_version), sess)
# builder.save(as_text=True)
# testavgloss,testavgacc = runEpoch(sess,testmodel,X_test,y_test,epoch_size//10)

