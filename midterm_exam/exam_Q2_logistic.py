"""
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import itertools
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statsmodels.api as api
import math

autos = pd.read_csv('policy_2001.csv', delimiter=',')

print('len total:',len(autos.CLAIM_FLAG))

train, test = train_test_split(autos, test_size=0.3,
                         random_state=20181010, stratify=autos['CLAIM_FLAG'])
print('len de train:', len(train.CLAIM_FLAG))
print('len de test',len(test.CLAIM_FLAG))
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
train_inputs = api.add_constant(train_inputs, prepend=True)
Y_target_train = train.CLAIM_FLAG

dumTest = pd.get_dummies(
    test[['CREDIT_SCORE_BAND']].astype('category'))
test_inputs = dumTest.join(
    test[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
test_inputs = api.add_constant(test_inputs, prepend=True)
Y_target_test = test.CLAIM_FLAG

y_target_train = Y_target_train.astype('category')
y_target_train_category = y_target_train.cat.categories

y_target_test = Y_target_test.astype('category')
y_target_test_category = y_target_test.cat.categories

logic = api.MNLogit(y_target_train, train_inputs)
logicModel = logic.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
thisParameter = logicModel.params

y_predProb = logicModel.predict(test_inputs)
y_miss = y_predProb.loc
y_predict = pd.to_numeric(y_predProb.idxmax(axis=1))

Y_predict = y_target_test_category[y_predict]

Y_predict_aux = np.asarray(Y_predict).reshape(-1, 1)
Y_predict_aux = pd.DataFrame(Y_predict_aux)

sel = 1-metrics.accuracy_score(Y_target_test,
                               (y_predProb[1] > claimTrain).astype(int))
missclassification = sel
print('Misclasification = ', missclassification)

y_accuracy = metrics.accuracy_score(y_target_test, Y_predict)
msse = math.sqrt(metrics.mean_squared_error(y_target_test, Y_predict))
auc = metrics.roc_auc_score(Y_target_test, y_predProb[1])


print('RMSE=',msse)
print('AUC=', auc)
print("Accuracy= ", y_accuracy)
