#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:25:43 2018
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""


import pandas
import numpy as np
from sklearn.model_selection import train_test_split
import scipy
import itertools
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statsmodels.api as api
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

#--------------------------------------training and testing data for region 1------------------------------------------------#
yTrain1 = train_region1[['Maintenance_flag']]
train_region1= train_region1.drop(train_region1[['Maintenance_flag']], axis=1)
train_region1=train_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS']]

yTest1 = test_region1[['Maintenance_flag']]
test_region1= test_region1.drop(test_region1[['Maintenance_flag']], axis=1)
test_region1=test_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS']]


train1 = yTrain1.loc[yTrain1['Maintenance_flag'] == 1]
claimTrain1 = len(train1.Maintenance_flag)/len(yTrain1.Maintenance_flag)

Y_target_train1 = yTrain1.Maintenance_flag
Y_target_test1 = yTest1.Maintenance_flag

y_target_train1 = Y_target_train1.astype('category')
y_target_train_category1 = y_target_train1.cat.categories

y_target_test1 = Y_target_test1.astype('category')
y_target_test_category1 = y_target_test1.cat.categories

logic1 = api.MNLogit(y_target_train1, train_region1)
logicModel1 = logic1.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
thisParameter1 = logicModel1.params

y_predProb1 = logicModel1.predict(test_region1)
y_miss1 = y_predProb1.loc
y_predict1 = pandas.to_numeric(y_predProb1.idxmax(axis=1))

predict1 = pandas.DataFrame({'Maintenance_flag': y_predict1})
aux1 = predict1.loc[predict1.Maintenance_flag == 1]
claimPred1 = len(aux1.Maintenance_flag)/len(y_predict1)
print('\nMaintenance Rate1 before:', claimTrain1)
print('Maintenance Rate1 predicted:', claimPred1)

Y_predict1 = y_target_test_category1[y_predict1]

Y_predict_aux1 = np.asarray(Y_predict1).reshape(-1, 1)
Y_predict_aux1 = pandas.DataFrame(Y_predict_aux1)

sel1 = 1-metrics.accuracy_score(Y_target_test1,
                               (y_predProb1[1] > claimTrain1).astype(int))
missclassification1 = sel1
print('\nMisclasification1= ', missclassification1)

y_accuracy1 = metrics.accuracy_score(y_target_test1, Y_predict1)
msse1 = math.sqrt(metrics.mean_squared_error(y_target_test1, Y_predict1))
auc1 = metrics.roc_auc_score(Y_target_test1, y_predProb1[1])


print('RMSE1=',msse1)
print('AUC1=', auc1)
print("Accuracy1= ", y_accuracy1)
print("\n\n")
print("-----------------------------------Region 2-----------------------------------------")

for atr in train_region2:
    au1= train_region2.loc[train_region2['Maintenance_flag'] == 0, [atr]]
    au2= train_region2.loc[train_region2['Maintenance_flag'] == 1, [atr]]
    #statistic, pvalue= scipy.stats.ttest_ind(atributes[[atr]], wine[['quality_grp']])
    statistic, pvalue= scipy.stats.ttest_ind(au1, au2)
    print(atr,',',statistic[0],',', pvalue[0])
    
print('\n')

#--------------------------------------training and testing data for region 2------------------------------------------------#
yTrain2 = train_region2[['Maintenance_flag']]
train_region2= train_region2.drop(train_region2[['Maintenance_flag']], axis=1)
train_region2=train_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude','Engine_Coolant_Temp']]

yTest2 = test_region2[['Maintenance_flag']]
test_region2= test_region2.drop(test_region2[['Maintenance_flag']], axis=1)
test_region2=test_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude', 'Engine_Coolant_Temp']]

train2 = yTrain2.loc[yTrain2['Maintenance_flag'] == 1]
claimTrain2 = len(train2.Maintenance_flag)/len(yTrain2.Maintenance_flag)

Y_target_train2 = yTrain2.Maintenance_flag
Y_target_test2 = yTest2.Maintenance_flag

y_target_train2 = Y_target_train2.astype('category')
y_target_train_category2 = y_target_train2.cat.categories

y_target_test2 = Y_target_test2.astype('category')
y_target_test_category2 = y_target_test2.cat.categories

logic2 = api.MNLogit(y_target_train2, train_region2)
logicModel2 = logic2.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
thisParameter2 = logicModel2.params

y_predProb2 = logicModel2.predict(test_region2)
y_miss2 = y_predProb2.loc
y_predict2 = pandas.to_numeric(y_predProb2.idxmax(axis=1))


predict2 = pandas.DataFrame({'Maintenance_flag': y_predict2})
aux2 = predict2.loc[predict2.Maintenance_flag == 1]
claimPred2 = len(aux2.Maintenance_flag)/len(y_predict2)
print('\nMaintenance Rate2 before:', claimTrain2)
print('Maintenance Rate2 predicted:', claimPred2)


Y_predict2 = y_target_test_category2[y_predict2]

Y_predict_aux2 = np.asarray(Y_predict2).reshape(-1, 1)
Y_predict_aux2 = pandas.DataFrame(Y_predict_aux2)

sel2 = 1-metrics.accuracy_score(Y_target_test2,
                               (y_predProb2[1] > claimTrain2).astype(int))
missclassification2 = sel2
print('\nMisclasification2= ', missclassification2)

y_accuracy2 = metrics.accuracy_score(y_target_test2, Y_predict2)
msse2 = math.sqrt(metrics.mean_squared_error(y_target_test2, Y_predict2))
auc2 = metrics.roc_auc_score(Y_target_test2, y_predProb2[1])


print('RMSE2=',msse2)
print('AUC2=', auc2)
print("Accuracy2= ", y_accuracy2)
print("\n\n")
print("-----------------------------------Region 3-----------------------------------------")

for atr in train_region3:
    au1= train_region3.loc[train_region3['Maintenance_flag'] == 0, [atr]]
    au2= train_region3.loc[train_region3['Maintenance_flag'] == 1, [atr]]
    #statistic, pvalue= scipy.stats.ttest_ind(atributes[[atr]], wine[['quality_grp']])
    statistic, pvalue= scipy.stats.ttest_ind(au1, au2)
    print(atr,',',statistic[0],',', pvalue[0])
    
print('\n')

#--------------------------------------training and testing data for region 3------------------------------------------------#
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

y_target_train3 = Y_target_train3.astype('category')
y_target_train_category3 = y_target_train3.cat.categories

y_target_test3 = Y_target_test3.astype('category')
y_target_test_category3 = y_target_test3.cat.categories

logic3 = api.MNLogit(y_target_train3, train_region3)
logicModel3 = logic3.fit(method='newton', full_output=True, maxiter=100, tol=1e-8)
thisParameter3 = logicModel3.params

y_predProb3 = logicModel3.predict(test_region3)
y_miss3 = y_predProb3.loc
y_predict3 = pandas.to_numeric(y_predProb3.idxmax(axis=1))

predict3 = pandas.DataFrame({'Maintenance_flag': y_predict3})
aux3 = predict3.loc[predict3.Maintenance_flag == 1]
claimPred3 = len(aux3.Maintenance_flag)/len(y_predict3)
print('\nMaintenance Rate1 before:', claimTrain3)
print('Maintenance Rate1 predicted:', claimPred3)

Y_predict3 = y_target_test_category3[y_predict3]

Y_predict_aux3 = np.asarray(Y_predict3).reshape(-1, 1)
Y_predict_aux3 = pandas.DataFrame(Y_predict_aux3)

sel3 = 1-metrics.accuracy_score(Y_target_test3,
                               (y_predProb3[1] > claimTrain3).astype(int))
missclassification3 = sel3
print('\nMisclasification3= ', missclassification3)

y_accuracy3 = metrics.accuracy_score(y_target_test3, Y_predict3)
msse3 = math.sqrt(metrics.mean_squared_error(y_target_test3, Y_predict3))
auc3 = metrics.roc_auc_score(Y_target_test3, y_predProb3[1])


print('RMSE3=',msse3)
print('AUC3=', auc3)
print("Accuracy3= ", y_accuracy3)
print("\n\n")



