import tensorflow as tf 
import tensorflow.keras as ks
from tensorflow.keras.layers import Dense,Input, Embedding, Dropout,concatenate,Dot,Flatten
from tensorflow.keras.models import Model
from sklearn import cross_validation as cv

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from core_layers import DNN_layers
VOCAB_SIZE = 100000
EMBEDDING_OUT_DIM = 100
train_file = 'train_sample.txt'
test_file = 'test_sample.txt'
train_data = []
train_label = []
test_data = []
words = {}
main_data = []
forward_data = []
like_data = []
comment_data = []

filetxt = np.loadtxt('sample_fea.txt', delimiter='\t',skiprows=1)
print filetxt.shape
train_data, test_data = cv.train_test_split(filetxt, test_size=0.3)
train_x = train_data[:,3:]
train_x_user = train_data[:,[1]]
train_x_item = train_data[:,[2]]
train_y = train_data[:,0]
test_x = test_data[:,3:]
test_y = test_data[:,0]
test_x_user = test_data[:,[1]]
test_x_item = test_data[:,[2]]
 
def dssm():
          #test_label = train_file[:,-1]
    train_x_user = train_data[:,4:36]
    train_x_item = train_data[:,36:55]
    train_x_user_id = train_data[:,[1]]
    train_x_item_id = train_data[:,[2]]
    train_y = train_data[:,0]
    test_x_user = test_data[:,4:36]
    test_x_item = test_data[:,36:55]
    test_x_user_id = test_data[:,[1]]
    test_x_item_id = test_data[:,[2]]
    test_y = test_data[:,0]
    user_input = Input((32,), dtype='float', name='user_input' )
    item_input = Input((19,), dtype='float', name='item_input' )
    user_id_input = Input((1,), dtype='float', name='user_id_input' )
    item_id_input = Input((1,), dtype='float', name='item_id_input' )
    embeddings_matrix_user = np.random.rand(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM)
    embedding_user = Embedding(input_dim = VOCAB_SIZE + 1, # 字典长度
                                output_dim = EMBEDDING_OUT_DIM, # 词向量 长度（100）
                                weights=[embeddings_matrix_user], # 重点：预训练的词向量系数
                                input_length=1, # 每句话的 最大长度（必须padding） 
                                trainable=True # 是否在 训练的过程中 更新词向量
                                )
    embeddings_matrix_item = np.random.rand(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM)
    embedding_item = Embedding(input_dim = VOCAB_SIZE + 1, # 字典长度
                                output_dim = EMBEDDING_OUT_DIM, # 词向量 长度（100）
                                weights=[embeddings_matrix_item], # 重点：预训练的词向量系数
                                input_length=1, # 每句话的 最大长度（必须padding） 
                                trainable=True # 是否在 训练的过程中 更新词向量
                                )
    user_x = embedding_user(user_id_input)
    item_x = embedding_item(item_id_input)
    user_x = Flatten()(user_x)
    item_x = Flatten()(item_x)
    x =  concatenate([user_input,user_x])
    x = ks.layers.BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    y =  concatenate([item_input,item_x])
    y = ks.layers.BatchNormalization()(y)
    y = Dense(256, activation='relu')(y)
    y = Dense(128, activation='relu')(y)
    y = Dense(64, activation='relu')(y)
    y = Dense(32, activation='relu')(y)
    y = Dropout(0.3)(y)

    forward_out = Dot(axes = 1,name='forward_out')([x, y])

    model = Model(inputs=[user_input,item_input,user_id_input,item_id_input], outputs=[forward_out])
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001),loss={'forward_out': 'mean_squared_error'})
    model.fit(x={'user_input': train_x_user, 'user_id_input': train_x_user_id, 'item_input': train_x_item, 'item_id_input': train_x_item_id},
            y={'forward_out': train_y},
            batch_size=64, epochs=10,verbose=1)
    #model.fit(x_train, y_train, epochs = 1000, batch_size=32)
    print(model.evaluate(x={'user_input': test_x_user, 'user_id_input': test_x_user_id, 'item_input': test_x_item, 'item_id_input': test_x_item_id},y = test_y , batch_size=32))

def dnn():
    #test_label = train_file[:,-1]
    main_input = Input((53,), dtype='float', name='main_input' )
    user_id_input = Input((1,), dtype='float', name='user_id_input' )
    item_id_input = Input((1,), dtype='float', name='item_id_input' )
    embeddings_matrix_user = np.random.rand(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM)
    embedding_user = Embedding(input_dim = VOCAB_SIZE + 1, # 字典长度
                                output_dim = EMBEDDING_OUT_DIM, # 词向量 长度（100）
                                weights=[embeddings_matrix_user], # 重点：预训练的词向量系数
                                input_length=1, # 每句话的 最大长度（必须padding） 
                                trainable=True # 是否在 训练的过程中 更新词向量
                                )
    embeddings_matrix_item = np.random.rand(VOCAB_SIZE + 1, EMBEDDING_OUT_DIM)
    embedding_item = Embedding(input_dim = VOCAB_SIZE + 1, # 字典长度
                                output_dim = EMBEDDING_OUT_DIM, # 词向量 长度（100）
                                weights=[embeddings_matrix_item], # 重点：预训练的词向量系数
                                input_length=1, # 每句话的 最大长度（必须padding） 
                                trainable=True # 是否在 训练的过程中 更新词向量
                                )
    user_x = embedding_user(user_id_input)
    item_x = embedding_item(item_id_input)
    user_x = Flatten()(user_x)
    item_x = Flatten()(item_x)
    x =  concatenate([main_input,user_x,item_x])
    x = ks.layers.BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    #x = DNN_layers([256,128,64,32])(x)
    x = Dropout(0.3)(x)
    forward_out = Dense(1, activation=None, name='forward_out')(x)
    model = Model(inputs=[main_input,user_id_input,item_id_input], outputs=[forward_out])
    model.compile(optimizer=tf.train.AdamOptimizer(0.0001),loss={'forward_out': 'mean_squared_error'})
    model.fit(x={'main_input': train_x, 'user_id_input': train_x_user, 'item_id_input': train_x_item},
            y={'forward_out': train_y},
            batch_size=64, epochs=5,verbose=1)
    #model.fit(x_train, y_train, epochs = 1000, batch_size=32)
    print(model.evaluate(x={'main_input': test_x, 'user_id_input': test_x_user, 'item_id_input': test_x_item},y = test_y , batch_size=32))
    #result = model.predict(test_data)
    #np.savetxt('result.txt', result)
#dssm()
dnn()
