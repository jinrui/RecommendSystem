import numpy as np;
import xgboost as xgb;
import pandas as pd;
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV
import matplotlib.pylab as plt

 
bst = xgb.Booster({'nthread':4}) #init model
bst.load_model("model.bin") # load data

fd = open('D:/works/qqmusic/testxgboost.txt')
data = fd.readlines()
dataset = np.array([line.strip().split(',') for line in data])



Y = np.array(dataset[:, 0], dtype=float)
X = xgb.DMatrix(np.array(dataset[:, 1:], dtype=float))
print(X)
dtest_predictions = bst.predict(X)
#dtest_predprob = bst.predict_proba(X)[:,1]
print(dtest_predictions)





