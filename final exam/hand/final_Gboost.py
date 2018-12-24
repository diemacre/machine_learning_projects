#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:25:43 2018
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas


import sklearn.metrics as metrics
import scipy

import sklearn.ensemble as ensemble
import math


fleet_train = pandas.read_csv('fleet_train.csv', delimiter=',')
target_train= fleet_train[['Maintenance_flag']]
fleet_train= fleet_train.drop(fleet_train[['fleetid','truckid','record_id','Measurement_timestamp']], axis=1)
train_region1= fleet_train.loc[fleet_train['Region'] == 1]
train_region1= train_region1.drop(train_region1[['Region']], axis=1)
train_region2= fleet_train.loc[fleet_train['Region'] == 2]
train_region2= train_region2.drop(train_region2[['Region']], axis=1)
train_region3= fleet_train.loc[fleet_train['Region'] == 3]
train_region3= train_region3.drop(train_region3[['Region']], axis=1)

fleet_test = pandas.read_csv('fleet_monitor_notscored_2.csv', delimiter=',')
target_test= fleet_test[['Maintenance_flag']]
fleet_test= fleet_test.drop(fleet_test[['fleetid','truckid','record_id','Measurement_timestamp', 'period']], axis=1)
test_region1= fleet_test.loc[fleet_test['Region'] == 1]
test_region1= test_region1.drop(test_region1[['Region']], axis=1)
test_region2= fleet_test.loc[fleet_test['Region'] == 2]
test_region2= test_region2.drop(test_region2[['Region']], axis=1)
test_region3= fleet_test.loc[fleet_test['Region'] == 3]
test_region3= test_region3.drop(test_region3[['Region']], axis=1)


atributes_list1= train_region1.columns.values.tolist()
print(atributes_list1)
atributes_list2= train_region2.columns.values.tolist()
print(atributes_list2)
atributes_list3= train_region3.columns.values.tolist()
print(atributes_list3)

print("-----------------------------------Region 1-----------------------------------------")
for atr in atributes_list1:    
        train_region1.boxplot(column=atr, by='Maintenance_flag', vert=False, figsize=(10,5))
        plt.xlabel(atr)
        plt.ylabel('Maintenance_flag1')
        plt.title('')
plt.show()
print("\n\n")

print("-----------------------------------Region 2-----------------------------------------")

for atr in atributes_list2:    
        train_region2.boxplot(column=atr, by='Maintenance_flag', vert=False, figsize=(10,5))
        plt.xlabel(atr)
        plt.ylabel('Maintenance_flag2')
        plt.title('')
plt.show()
print("\n\n")
print("-----------------------------------Region 3-----------------------------------------")

for atr in atributes_list3:    
        train_region3.boxplot(column=atr, by='Maintenance_flag', vert=False, figsize=(10,5))
        plt.xlabel(atr)
        plt.ylabel('Maintenance_flag3')
        plt.title('')
plt.show()

print("\n\n")


print("\n\n")
print("-----------------------------------Region 1-----------------------------------------")

for atr in train_region1:
    au1= train_region1.loc[train_region1['Maintenance_flag'] == 0, [atr]]
    au2= train_region1.loc[train_region1['Maintenance_flag'] == 1, [atr]]
    #statistic, pvalue= scipy.stats.ttest_ind(atributes[[atr]], wine[['quality_grp']])
    statistic, pvalue= scipy.stats.ttest_ind(au1, au2)
    print(atr,',',statistic[0],',', pvalue[0])
    
print('\n')


yTrain1 = train_region1[['Maintenance_flag']]
train_region1= train_region1.drop(train_region1[['Maintenance_flag']], axis=1)
train_region1=train_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS','Throttle_Pos_Manifold','Mass_Air_Flow_Rate']]

yTest1 = test_region1[['Maintenance_flag']]
test_region1= test_region1.drop(test_region1[['Maintenance_flag']], axis=1)
test_region1=test_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS','Throttle_Pos_Manifold','Mass_Air_Flow_Rate']]

train1 = yTrain1.loc[yTrain1['Maintenance_flag'] == 1]
claimTrain1 = len(train1.Maintenance_flag)/len(yTrain1.Maintenance_flag)

Y_target_train1 = yTrain1.Maintenance_flag
Y_target_test1 = yTest1.Maintenance_flag

gbm1 = ensemble.GradientBoostingClassifier(loss='deviance', criterion='mse', n_estimators = 1500,
                                         max_leaf_nodes = 5, verbose=1)
fit_gbm1 = gbm1.fit(train_region1, yTrain1)
Y_predict1 = fit_gbm1.predict(test_region1)
Y_predProb1= fit_gbm1.predict_proba(test_region1)

predict1 = pandas.DataFrame({'Maintenance_flag': Y_predict1})
aux1 = predict1.loc[predict1.Maintenance_flag == 1]
claimPred1 = len(aux1.Maintenance_flag)/len(Y_predict1)
print('\nMaintenance Rate1 before:', claimTrain1)
print('Maintenance Rate1 predicted:', claimPred1)

sel1 = 1-metrics.accuracy_score(Y_target_test1,
                               (Y_predProb1[:, 1] > claimTrain1).astype(int))
missclassification1 = sel1
print('\nMisclasification1= ', missclassification1)

y_accuracy1 = metrics.accuracy_score(Y_target_test1, Y_predict1)
msse1 = math.sqrt(metrics.mean_squared_error(Y_target_test1, Y_predict1))
auc1 = metrics.roc_auc_score(Y_target_test1, Y_predProb1[:, 1])

print('RMSE1=',msse1)
print('AUC1=', auc1)



print("\n\n")
print("-----------------------------------Region 2-----------------------------------------")

for atr in train_region2:
    au1= train_region2.loc[train_region2['Maintenance_flag'] == 0, [atr]]
    au2= train_region2.loc[train_region2['Maintenance_flag'] == 1, [atr]]
    #statistic, pvalue= scipy.stats.ttest_ind(atributes[[atr]], wine[['quality_grp']])
    statistic, pvalue= scipy.stats.ttest_ind(au1, au2)
    print(atr,',',statistic[0],',', pvalue[0])
    
print('\n')


yTrain2 = train_region2[['Maintenance_flag']]
train_region2= train_region2.drop(train_region2[['Maintenance_flag']], axis=1)
train_region2=train_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude','Engine_Coolant_Temp']]

yTest2 = test_region2[['Maintenance_flag']]
test_region2= test_region2.drop(test_region2[['Maintenance_flag']], axis=1)
test_region2=test_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude','Engine_Coolant_Temp']]

train2 = yTrain2.loc[yTrain2['Maintenance_flag'] == 1]
claimTrain2 = len(train2.Maintenance_flag)/len(yTrain2.Maintenance_flag)

Y_target_train2 = yTrain2.Maintenance_flag
Y_target_test2 = yTest2.Maintenance_flag

gbm2 = ensemble.GradientBoostingClassifier(loss='deviance', criterion='mse', n_estimators = 1500,
                                         max_leaf_nodes = 5, verbose=1)
fit_gbm2 = gbm2.fit(train_region2, yTrain2)
Y_predict2 = fit_gbm2.predict(test_region2)
Y_predProb2= fit_gbm2.predict_proba(test_region2)

predict2 = pandas.DataFrame({'Maintenance_flag': Y_predict2})
aux2 = predict2.loc[predict2.Maintenance_flag == 1]
claimPred2 = len(aux2.Maintenance_flag)/len(Y_predict2)
print('\nMaintenance Rate2 before:', claimTrain2)
print('Maintenance Rate2 predicted:', claimPred2)

sel2 = 1-metrics.accuracy_score(Y_target_test2,
                               (Y_predProb2[:, 1] > claimTrain2).astype(int))
missclassification2 = sel2
print('\nMisclasification2= ', missclassification2)

y_accuracy2 = metrics.accuracy_score(Y_target_test2, Y_predict2)
msse2 = math.sqrt(metrics.mean_squared_error(Y_target_test2, Y_predict2))
auc2 = metrics.roc_auc_score(Y_target_test2, Y_predProb2[:, 1])

print('RMSE2=',msse2)
print('AUC2=', auc2)

print("\n\n")
print("-----------------------------------Region 3-----------------------------------------")

for atr in train_region3:
    au1= train_region3.loc[train_region3['Maintenance_flag'] == 0, [atr]]
    au2= train_region3.loc[train_region3['Maintenance_flag'] == 1, [atr]]
    #statistic, pvalue= scipy.stats.ttest_ind(atributes[[atr]], wine[['quality_grp']])
    statistic, pvalue= scipy.stats.ttest_ind(au1, au2)
    print(atr,',',statistic[0],',', pvalue[0])
    
print('\n')


yTrain3 = train_region3[['Maintenance_flag']]
train_region3= train_region3.drop(train_region3[['Maintenance_flag']], axis=1)
train_region3=train_region3[['Intake_Air_Temp', 'Speed_GPS']]

yTest3 = test_region3[['Maintenance_flag']]
test_region3= test_region3.drop(test_region3[['Maintenance_flag']], axis=1)
test_region3=test_region3[['Intake_Air_Temp', 'Speed_GPS']]

train3 = yTrain3.loc[yTrain3['Maintenance_flag'] == 1]
claimTrain3 = len(train3.Maintenance_flag)/len(yTrain3.Maintenance_flag)

Y_target_train3 = yTrain3.Maintenance_flag
Y_target_test3 = yTest3.Maintenance_flag

gbm3 = ensemble.GradientBoostingClassifier(loss='deviance', criterion='mse', n_estimators = 1500,
                                         max_leaf_nodes = 5, verbose=1)
fit_gbm3 = gbm3.fit(train_region3, yTrain3)
Y_predict3 = fit_gbm3.predict(test_region3)
Y_predProb3= fit_gbm3.predict_proba(test_region3)

predict3 = pandas.DataFrame({'Maintenance_flag': Y_predict3})
aux3 = predict3.loc[predict3.Maintenance_flag == 1]
claimPred3 = len(aux3.Maintenance_flag)/len(Y_predict3)
print('\nMaintenance Rate3 before:', claimTrain3)
print('Maintenance Rate3 predicted:', claimPred3)


sel3 = 1-metrics.accuracy_score(Y_target_test3,
                               (Y_predProb3[:, 1] > claimTrain3).astype(int))
missclassification3 = sel3
print('\nMisclasification3= ', missclassification3)

y_accuracy3 = metrics.accuracy_score(Y_target_test3, Y_predict3)
msse3 = math.sqrt(metrics.mean_squared_error(Y_target_test3, Y_predict3))
auc3 = metrics.roc_auc_score(Y_target_test3, Y_predProb3[:, 1])

print('RMSE3=',msse3)
print('AUC3=', auc3)
