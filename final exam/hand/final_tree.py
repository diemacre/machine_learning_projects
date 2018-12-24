#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:25:43 2018
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

import numpy as np
import sklearn.cluster as cluster
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn import metrics
import graphviz
from sklearn.model_selection import train_test_split


fleet_train = pd.read_csv('fleet_train.csv', delimiter=',')
target_train= fleet_train[['Maintenance_flag']]
fleet_train= fleet_train.drop(fleet_train[['fleetid','truckid','record_id','Measurement_timestamp']], axis=1)
train_region1= fleet_train.loc[fleet_train['Region'] == 1]
train_region1= train_region1.drop(train_region1[['Region']], axis=1)
train_region2= fleet_train.loc[fleet_train['Region'] == 2]
train_region2= train_region2.drop(train_region2[['Region']], axis=1)
train_region3= fleet_train.loc[fleet_train['Region'] == 3]
train_region3= train_region3.drop(train_region3[['Region']], axis=1)

fleet_test = pd.read_csv('fleet_monitor_notscored_2.csv', delimiter=',')
target_test= fleet_test[['Maintenance_flag']]
fleet_test= fleet_test.drop(fleet_test[['fleetid','truckid','record_id','Measurement_timestamp', 'period']], axis=1)
test_region1= fleet_test.loc[fleet_test['Region'] == 1]
test_region1= test_region1.drop(test_region1[['Region']], axis=1)
test_region2= fleet_test.loc[fleet_test['Region'] == 2]
test_region2= test_region2.drop(test_region2[['Region']], axis=1)
test_region3= fleet_test.loc[fleet_test['Region'] == 3]
test_region3= test_region3.drop(test_region3[['Region']], axis=1)


atributes_list1= train_region1.columns.values.tolist()
#print(atributes_list1)
atributes_list2= train_region2.columns.values.tolist()
#print(atributes_list2)
atributes_list3= train_region3.columns.values.tolist()
#print(atributes_list3)

#--------------------------------------training and testing data for region 1------------------------------------------------#
print("\n\n")
print("-----------------------------------Region 1-----------------------------------------")

yTrain1 = train_region1[['Maintenance_flag']]
train_region1= train_region1.drop(train_region1[['Maintenance_flag']], axis=1)
#train_region1=train_region1[['Speed_GPS', 'Trip_Distance', 'Trip_Time_journey']]
train_region1=train_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS','Throttle_Pos_Manifold','Mass_Air_Flow_Rate']]
atributes_list1= train_region1.columns.values.tolist()
yTest1 = test_region1[['Maintenance_flag']]
test_region1= test_region1.drop(test_region1[['Maintenance_flag']], axis=1)
#test_region1=test_region1[['Speed_GPS', 'Trip_Distance', 'Trip_Time_journey']]
test_region1=test_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS','Throttle_Pos_Manifold','Mass_Air_Flow_Rate']]


train1 = yTrain1.loc[yTrain1['Maintenance_flag'] == 1]
claimTrain1 = len(train1.Maintenance_flag)/len(yTrain1.Maintenance_flag)


Y_target_train1 = yTrain1.Maintenance_flag
Y_target_test1 = yTest1.Maintenance_flag

classTree1 = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=4, random_state=20181010)
Cluster_DT1 = classTree1.fit(train_region1, Y_target_train1)

Y_predict1 = Cluster_DT1.predict(test_region1)
Y_probab1 = Cluster_DT1.predict_proba(test_region1)

predict1 = pd.DataFrame({'Maintenance_flag': Y_predict1})
aux1 = predict1.loc[predict1.Maintenance_flag == 1]
claimPred1 = len(aux1.Maintenance_flag)/len(Y_predict1)
print('\nMaintenance Rate1 before:', claimTrain1)
print('Maintenance Rate1 predicted:', claimPred1)


sel1= 1-metrics.accuracy_score(Y_target_test1,(Y_probab1[:, 1] > claimTrain1).astype(int))
missclassification1= sel1
print('\nMisclasification1= ', missclassification1)

msse1 = metrics.mean_squared_error(Y_target_test1, Y_predict1)
auc1 = metrics.roc_auc_score(Y_target_test1, Y_probab1[:, 1])

#print(miss_classi)
print('RMSE1=', msse1)
print('AUC1=', auc1)
#print(thresholds)
#print('Accuracy= {:.6f}' .format(classTree1.score(train_region1, Y_target_train1)))

#X_name1 = np.asarray(train_region1[['Speed_GPS', 'Trip_Distance', 'Trip_Time_journey']].columns.values.tolist()).flat
X_name1= np.asarray(atributes_list1).flat
dot_data1 = tree.export_graphviz(Cluster_DT1,
                                out_file=None,
                                impurity=True, filled=True,
                                feature_names=X_name1,
                                class_names=['0', '1'])

graph1 = graphviz.Source(dot_data1)
print(graph1)

graph1.render('Tree Region 1')

print("\n\n")
print("-----------------------------------Region 2-----------------------------------------")


yTrain2 = train_region2[['Maintenance_flag']]
train_region2= train_region2.drop(train_region2[['Maintenance_flag']], axis=1)
#train_region2=train_region2[['Trip_Time_journey','Ambient_air_temp']]
#train_region2=train_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude','Engine_Coolant_Temp']]
atributes_list2= train_region2.columns.values.tolist()

yTest2 = test_region2[['Maintenance_flag']]
test_region2= test_region2.drop(test_region2[['Maintenance_flag']], axis=1)
#test_region2=test_region2[['Trip_Time_journey','Ambient_air_temp']]
#test_region2=test_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude','Engine_Coolant_Temp']]


train2 = yTrain2.loc[yTrain2['Maintenance_flag'] == 1]
claimTrain2 = len(train2.Maintenance_flag)/len(yTrain2.Maintenance_flag)


Y_target_train2 = yTrain2.Maintenance_flag
Y_target_test2 = yTest2.Maintenance_flag

classTree2 = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=7, random_state=20181010)
Cluster_DT2 = classTree2.fit(train_region2, Y_target_train2)

Y_predict2 = Cluster_DT2.predict(test_region2)
Y_probab2 = Cluster_DT2.predict_proba(test_region2)


predict2 = pd.DataFrame({'Maintenance_flag': Y_predict2})
aux2 = predict2.loc[predict2.Maintenance_flag == 1]
claimPred2 = len(aux2.Maintenance_flag)/len(Y_predict2)
print('\nMaintenance Rate2 before:', claimTrain2)
print('Maintenance Rate2 predicted:', claimPred2)


sel2= 1-metrics.accuracy_score(Y_target_test2,(Y_probab2[:, 1] > claimTrain2).astype(int))
missclassification2= sel2
print('\nMisclasification2= ', missclassification2)

msse2 = metrics.mean_squared_error(Y_target_test2, Y_predict2)
auc2 = metrics.roc_auc_score(Y_target_test2, Y_probab2[:, 1])

#print(miss_classi)
print('RMSE2=', msse2)
print('AUC2=', auc2)
#print(thresholds)
#print('Accuracy= {:.6f}' .format(classTree2.score(train_region2, Y_target_train2)))

#X_name2 = np.asarray(train_region2[['Trip_Time_journey','Ambient_air_temp']].columns.values.tolist()).flat
X_name2= np.asarray(atributes_list2).flat
dot_data2 = tree.export_graphviz(Cluster_DT2,
                                out_file=None,
                                impurity=True, filled=True,
                                feature_names=X_name2,
                                class_names=['0', '1'])

graph2 = graphviz.Source(dot_data2)
print(graph2)

graph2.render('Tree Region 2')

print("\n\n")
print("-----------------------------------Region 3-----------------------------------------")


yTrain3 = train_region3[['Maintenance_flag']]
train_region3= train_region3.drop(train_region3[['Maintenance_flag']], axis=1)
#train_region3=train_region3[['GPS_Latitude','Vehicle_speed_sensor', 'Accel_Ssor_Total']]
train_region3=train_region3[['Intake_Air_Temp', 'Speed_GPS']]
atributes_list3= train_region3.columns.values.tolist()

yTest3 = test_region3[['Maintenance_flag']]
test_region3= test_region3.drop(test_region3[['Maintenance_flag']], axis=1)
#test_region3=test_region3[['GPS_Latitude','Vehicle_speed_sensor', 'Accel_Ssor_Total']]
test_region3=test_region3[['Intake_Air_Temp', 'Speed_GPS']]

train3 = yTrain3.loc[yTrain3['Maintenance_flag'] == 1]
claimTrain3 = len(train3.Maintenance_flag)/len(yTrain3.Maintenance_flag)


Y_target_train3 = yTrain3.Maintenance_flag
Y_target_test3 = yTest3.Maintenance_flag

classTree3 = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=2, random_state=20181010)
Cluster_DT3 = classTree3.fit(train_region3, Y_target_train3)

Y_predict3 = Cluster_DT3.predict(test_region3)
Y_probab3 = Cluster_DT3.predict_proba(test_region3)

predict3 = pd.DataFrame({'Maintenance_flag': Y_predict3})
aux3 = predict3.loc[predict3.Maintenance_flag == 1]
claimPred3 = len(aux3.Maintenance_flag)/len(Y_predict3)
print('\nMaintenance Rate3 before:', claimTrain3)
print('Maintenance Rate3 predicted:', claimPred3)


sel3= 1-metrics.accuracy_score(Y_target_test3,(Y_probab3[:, 1] > claimTrain3).astype(int))
missclassification3= sel3
print('\nMisclasification3= ', missclassification3)

msse3 = metrics.mean_squared_error(Y_target_test3, Y_predict3)
auc3 = metrics.roc_auc_score(Y_target_test3, Y_probab3[:, 1])

#print(miss_classi)
print('RMSE3=', msse3)
print('AUC3=', auc3)
#print(thresholds)
#print('Accuracy= {:.6f}' .format(classTree3.score(train_region3, Y_target_train3)))

#X_name3 = np.asarray(train_region3[['GPS_Latitude','Vehicle_speed_sensor','Accel_Ssor_Total']].columns.values.tolist()).flat
X_name3= np.asarray(atributes_list3).flat
dot_data3 = tree.export_graphviz(Cluster_DT3,
                                out_file=None,
                                impurity=True, filled=True,
                                feature_names=X_name3,
                                class_names=['0', '1'])

graph3 = graphviz.Source(dot_data3)
print(graph3)

graph3.render('Tree Region 3')

