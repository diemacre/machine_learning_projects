"""
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sklearn.cluster as cluster
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import metrics
import math

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

classTree = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=10, random_state=20181010)
Cluster_DT = classTree.fit(train_inputs, Y_target_train)

Y_predict = Cluster_DT.predict(test_inputs)
Y_probab = Cluster_DT.predict_proba(test_inputs)
'''
#conditions misclassified: test != prediction and probPredicted >= threshold
matrix = pd.DataFrame({'Target': Y_target_test, 'Predicted': Y_predict,
                       'Probability': Y_probab[:, 1]}, index=Y_target_test.index.copy())
selected = matrix.loc[((matrix['Probability'] >= claimTrain) & (matrix.Predicted!=matrix.Target))]
'''
sel= 1-metrics.accuracy_score(Y_target_test,(Y_probab[:, 1] > claimTrain).astype(int))
missclassification= sel
print('Misclasification = ', missclassification)



msse = metrics.mean_squared_error(Y_target_test, Y_predict)
auc = metrics.roc_auc_score(Y_target_test, Y_probab[:, 1])


#print(miss_classi)
print('RMSE =', msse)
print('AUC =', auc)
#print(thresholds)
print('Accuracy= {:.6f}' .format(
    classTree.score(train_inputs, Y_target_train)))

