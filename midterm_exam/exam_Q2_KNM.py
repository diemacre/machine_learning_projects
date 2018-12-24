"""
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split

import itertools
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statsmodels.api as api
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as LA

autos = pd.read_csv('policy_2001.csv', delimiter=',')

train, test = train_test_split(autos, test_size=0.3,
                               random_state=20181010, stratify=autos['CLAIM_FLAG'])
print('len de train:', len(train.CLAIM_FLAG))
print('len de test', len(test.CLAIM_FLAG))
train1 = train.loc[train['CLAIM_FLAG'] == 1]
train0 = train.loc[train['CLAIM_FLAG'] == 0]
test1 = test.loc[test['CLAIM_FLAG'] == 1]
test0 = test.loc[test['CLAIM_FLAG'] == 0]
claimTest = len(test1.CLAIM_FLAG)/len(test.CLAIM_FLAG)
claimTrain = len(train1.CLAIM_FLAG)/len(train.CLAIM_FLAG)
print('claimtrain:', claimTrain)
print('claimtest:', claimTest)

dumTrain = pd.get_dummies(train[['CREDIT_SCORE_BAND']].astype('category'))
train_inputs = dumTrain.join(
    train[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
Y_target_train = train.CLAIM_FLAG

dumTest = pd.get_dummies(
    test[['CREDIT_SCORE_BAND']].astype('category'))
test_inputs = dumTest.join(
    test[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
Y_target_test = test.CLAIM_FLAG

kNNSpec = KNeighborsClassifier(
    n_neighbors=3, algorithm='brute', metric='euclidean' )
nbrs = kNNSpec.fit(train_inputs, Y_target_train)
Y_predict = nbrs.predict(test_inputs)
Y_probab = nbrs.predict_proba(test_inputs)

sel = 1-metrics.accuracy_score(Y_target_test,
                               (Y_probab[:, 1] > claimTrain).astype(int))
missclassification = sel
print('Misclasification = ', missclassification)



msse = math.sqrt(metrics.mean_squared_error(Y_target_test, Y_predict))

score_result = nbrs.score(test_inputs, Y_target_test)
auc = metrics.roc_auc_score(Y_target_test, Y_probab[:, 1])


#print(miss_classi)
print('RMSE =', msse)
print('AUC =',auc)
print('Accuracy: ', score_result)
