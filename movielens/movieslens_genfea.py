#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 23:47:40 2018

@author: jinrui06
"""

import sklearn
import numpy as np
import pandas as pd
from sklearn import cross_validation as cv
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler 
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.grid_search import GridSearchCV
import time
import math

#读取用户对item打分到pd
user_item_pd = pd.read_csv('../data/ml-100k/u.data',sep='\t',names=['user_id', 'item_id', 'rating', 'timestamp'])

#读取用户特征到pd
user_feature_pd = pd.read_csv('../data/ml-100k/u.user',sep='\|',names=['user_id', 'age', 'gender', 'occupation','zipcode'])

#读取item特征到pd
item_feature_pd = pd.read_csv('../data/ml-100k/u.item',sep='\|',names=['mvid', 'mvtitle', 'releasedate', 'vdreleasedate','imdburl','unknow','action','adventure',
                                                                      'animation','children','comedy','crime','Documentary','Drama',
                                                                       'Fantasy','film_noir','horror','musical','mystery','romance','sci_fi','thriller','war','western'])
#print(len(user_feature_pd['zipcode'].unique()))
item_feature_pd = item_feature_pd.rename(columns={"mvid":"item_id"})
print(user_feature_pd.head(1))
#特征工程
#user_feature_pd 年龄分桶，职业分桶，
#object转数字
le = LabelEncoder()
for column in user_feature_pd.columns:
    if(user_feature_pd[column].dtype == 'object'):
        #print(column)
        le.fit(user_feature_pd[column])
        user_feature_pd[column]=le.transform(user_feature_pd[column])
#print(user_feature_pd.head(10))        
enc = OneHotEncoder(sparse = False)
def checkage(age):
    if age <6:
        return '0'
    elif age < 12:
        return '1'
    elif age < 15:
        return '2'
    elif age < 18:
        return '3'
    elif age < 22:
        return '4'
    elif age < 30:
        return '5'
    elif age < 40:
        return '6'
    elif age < 50:
        return '7'
    elif age < 60:
        return '8'
    elif age <70:
        return '9'
    else:
        return '10'
user_feature_pd['age'] = user_feature_pd['age'] .apply(lambda x: checkage(x))
user_feature_pd['occupation']=user_feature_pd['occupation'].apply(str)
user_feature_pd['zipcode']=user_feature_pd['zipcode'].apply(str)
user_feature_pd = user_feature_pd.drop(['zipcode'],axis=1)

user_feature_pd=pd.get_dummies(user_feature_pd)
print(user_feature_pd.head(1))

#print(item_feature_pd['unknow'])

#item特征工程麻烦一点，需要对title变成词袋模型，releasedate，vdreleasedate变成年月日3个字段
#imdburl 去掉 
def change(x,leng,vocab):
    arr = np.zeros(leng,dtype=np.int)
    words = [w.strip('()[]\'').lower() for w in x.split(' ')]
    nums = [vocab[wod] for wod in words]
    for num in nums:
        arr[num] = 1
    arrlist = [str(i) for i in arr.tolist()]
    return ','.join(arrlist)

def  string_toTimestamp(st):
    if type(st) != type('a') :
        return 0
    st = st.replace('Jan','1').replace('Feb','2').replace('Mar','3').replace('Apr','4').replace('May','5').replace('Jun','6').replace('Jul','7').replace('Aug','8').replace('Sep','9').replace('Oct','10').replace('Nov','11').replace('Dec','12')

    return  time.mktime(time.strptime(st, "%d-%m-%Y"))
item_feature_pd = item_feature_pd.drop(['imdburl'],axis=1)
#print(item_feature_pd.head(1))
vocab = item_feature_pd['mvtitle'].values.tolist()
vocab = [title.split(' ') for title in vocab]
vocab = sorted(list(set(reduce(lambda x,y:x+y,vocab))))
vocab = list(set([vo.strip('()[]\'').lower() for vo in vocab]))
#print(len(vocab))
vocab = dict(zip(vocab,range(len(vocab))))
#print(len(vocab))
#item_feature_pd['mvtitle'] = item_feature_pd['mvtitle'].apply(lambda x : change(x,len(vocab),vocab))
#names=item_feature_pd['mvtitle'].str.split(',',expand=True)#多名字分列
#releasedate转换成时间戳
#print(item_feature_pd['releasedate'].unique())
item_feature_pd['releasedate']=item_feature_pd['releasedate'].apply(string_toTimestamp)
item_feature_pd = item_feature_pd.drop(['mvtitle'],axis=1)
#item_feature_pd = item_feature_pd.drop(['releasedate'],axis=1)
item_feature_pd = item_feature_pd.drop(['vdreleasedate'],axis=1)
item_feature_pd = item_feature_pd.drop(['unknow'],axis=1)

#item_feature_pd = item_feature_pd.join(names)
print(item_feature_pd.head(2))
item_feature_pd.to_csv('item_fea.txt', sep='\t', index=False)
user_feature_pd.to_csv('user_fea.txt', sep='\t', index=False)
user_item_pd = pd.merge(user_item_pd, user_feature_pd, on='user_id')
user_item_pd = pd.merge(user_item_pd, item_feature_pd, on='item_id')
user_item_pd['rate_age'] = user_item_pd['timestamp']  - user_item_pd['releasedate'] 
user_item_pd['releasedate'] = (user_item_pd['releasedate'] - user_item_pd['releasedate'].min()) / (user_item_pd['releasedate'].max() - user_item_pd['releasedate'].min())
user_item_pd['timestamp'] = (user_item_pd['timestamp'] - user_item_pd['timestamp'].min()) / (user_item_pd['timestamp'].max() - user_item_pd['timestamp'].min())
user_item_pd['rate_age'] = (user_item_pd['rate_age'] - user_item_pd['rate_age'].min()) / (user_item_pd['rate_age'].max() - user_item_pd['rate_age'].min())

haha = user_item_pd['rating']
user_item_pd = user_item_pd.drop('rating',axis=1)
user_item_pd.insert(0,'rating',haha)
user_item_pd.to_csv('sample_fea.txt', sep='\t', index=False)



