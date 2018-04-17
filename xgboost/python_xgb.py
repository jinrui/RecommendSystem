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



def modelfit(alg, dtrain, dlabel, x_test, y_test, useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
    '''
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain, label=dlabel)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds,show_stdv=True)
        print('n_estimators:',cvresult.shape[0])
        alg.set_params(n_estimators=cvresult.shape[0])
    '''
   
    #Fit the algorithm on the data
    alg.fit(dtrain, dlabel,eval_metric='auc')
    
    
    #Predict training set:
    dtrain_predictions = alg.predict(dtrain)
    dtrain_predprob = alg.predict_proba(dtrain)[:,1]
    
    #Predict test set:
    dtest_predictions = alg.predict(x_test)
    dtest_predprob = alg.predict_proba(x_test)[:,1]
    
    #Print model report:
    print("\nModel Report(train)")
    print("Accuracy : %.4g" % metrics.accuracy_score(dlabel, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(dlabel, dtrain_predprob))
    
    #Print model report:
    print("\nModel Report(test)")
    print("Accuracy : %.4g" % metrics.accuracy_score(y_test, dtest_predictions))
    print("AUC Score (test): %f" % metrics.roc_auc_score(y_test, dtest_predprob))
                    
    feat_imp = pd.Series(alg.feature_importances_).sort_values(ascending=False)
    feat_imp.head(20).plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')

    
    
xgb1 = XGBClassifier(
 learning_rate = 0.2,
 n_estimators=46,
 max_depth=3,
 min_child_weight=3,
 gamma=0.0,
 subsample=0.9,
 colsample_bytree=0.7,
 objective= 'binary:logistic',
 nthread=20,
 reg_alpha=0.01,
 scale_pos_weight=1,
 seed=19911118)


fd = open('D:/works/qqmusic/xgboost_data.txt')
data = fd.readlines()
dataset = np.array([line.strip().split(',') for line in data])

print(dataset[:, :])



Y = np.array(dataset[:, 0], dtype=float)
X = np.array(dataset[:, 1:], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=19910825)
print(X_test[0])
'''param_test1 = {
 'max_depth':[i for i in range(3,10,2)],
 'min_child_weight':[i for i in range(1,6,2)]
}'''
'''
param_test3 = {
 'gamma':[i/10.0 for i in range(0,5)]
}'''
'''param_test4 = {
 'subsample':[i/10.0 for i in range(6,10)],
 'colsample_bytree':[i/10.0 for i in range(6,10)]
}'''
'''
param_test5 = {
 'learning_rate':[i/100.0 for i in range(1,20)]
}'''
#gsearch1 = GridSearchCV(xgb1, param_grid = param_test4,scoring='roc_auc')
#Fit the algorithm on the data
#gsearch1.fit(X_train, y_train)

#print(gsearch1.grid_scores_, gsearch1.best_params_,     gsearch1.best_score_)

modelfit(xgb1, X_train, y_train, X_test, y_test, useTrainCV=True)

#xgb.Booster.save_model(xgb1.get_booster(), fname='model.bin')





