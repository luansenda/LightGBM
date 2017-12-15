# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 15:00:47 2017

@author: cteam
"""

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split

print('Load data...')
data1 = pd.read_excel(r'E:\didi_data\2016080_O.xlsx')
data2 = pd.read_excel(r'E:\didi_data\2016080_D.xlsx')
data1.drop(['ID1','ID2','ID3','CITY','O_lon','O_lat','D_lon','D_lat','O_time','D_time','TYPE','FLAG'],axis = 1, inplace = True)
data2.drop(['ID1','ID2','ID3','CITY','O_lon','O_lat','D_lon','D_lat','O_time','D_time','TYPE','FLAG'],axis = 1, inplace = True)
data2.columns = data1.columns
data = data1 + data2
#
data['travel_time'] = data['travel_time']/2
data['week_flag'] = data['week_flag']/2
data['hour_flag'] = data['hour_flag']/2
data['dist'] = data['dist']/2
del data1
del data2
data = pd.read_excel(r'E:\didi_data\2016080_OD.xlsx')
## 按travel time 分组
data = data[data['travel_time']>15]
data = data[data['travel_time']<61]
df_train,df_test = train_test_split(data, random_state = 11, test_size = 0.25) # randomly split the data
y_train = df_train['travel_time'].values # take 'travel_time' as y
y_test = df_test['travel_time'].values
x_train = df_train.drop('travel_time', axis = 1).values #extract first column from data source and save as dependent variable
x_test = df_test.drop('travel_time', axis = 1).values
del data
del df_train
del df_test

'''
调参数，固定estimator内的参数1，param_grid给出可选参数值列
'''
## 选择learming_rate,n_estimators
estimator = lgb.LGBMRegressor(objective = 'regression',
                              colsample_bytree = 0.8,
                              subsample = 0.9,
                              subsample_freq = 1)
param_grid = {
        'learning_rate' : [0.01, 0.05, 0.1],
        'n_estimators' : [200, 400, 600],
        'num_leaves' : [300, 600, 1000]
}

gbm = GridSearchCV(estimator, param_grid)
gbm.fit(x_train,y_train)
print('Best parameters found by grid search are:', gbm.best_params_)


'''
根据上述选定的optimal params,建立模型
'''                     
print('Start training...')
# train
gbm = lgb.LGBMRegressor(objective = 'regression',
                        colsample_bytree = 0.8,
                        subsample = 0.9,
                        subsample_freq = 1,
                        learning_rate = 0.01,
                        n_estimators = 1000,
                        num_leaves = 300)
gbm.fit(x_train, y_train,
        eval_set = [(x_test, y_test)],
        eval_metric = 'l1',
        early_stopping_rounds = 5)

print('Start predicting...')
# predict
y_pred = gbm.predict(x_test, num_iteration = gbm.best_iteration)
# eval
print ('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
print('The MAPE of prediction is:', np.mean(abs(y_pred-y_test)/y_test))

# feature importances
print('Feature importances:', list(gbm.feature_importances_))

print('Plot feature importances...')
ax = lgb.plot_importance(gbm)
# plt.savefig(r'C:\Users\cteam\Desktop\proj_images\travel_time_0-60_importance.jpg', format = 'jpg')
plt.show()
