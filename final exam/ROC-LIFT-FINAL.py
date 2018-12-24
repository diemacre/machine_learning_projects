#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 09:32:30 2018

@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""


# Load the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as stats
import statsmodels.api as api
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from numpy import linalg as LA
from sklearn import tree
import sklearn.metrics as metrics
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp
import sklearn.ensemble as ensemble
# Define a function to compute the coordinates of the Lift chart


def compute_lift_coordinates(
        DepVar,          # The column that holds the dependent variable's values
        EventValue,      # Value of the dependent variable that indicates an event
        EventPredProb,   # The column that holds the predicted event probability
        Debug='N'):    # Show debugging information (Y/N)

    # Find out the number of observations
    nObs = len(DepVar)

    # Get the quantiles
    quantileCutOff = np.percentile(EventPredProb, np.arange(0, 100, 10))
    nQuantile = len(quantileCutOff)

    quantileIndex = np.zeros(nObs)
    for i in range(nObs):
        iQ = nQuantile
        EPP = EventPredProb.iloc[i]
        for j in range(1, nQuantile):
            if (EPP > quantileCutOff[-j]):
                iQ -= 1
        quantileIndex[i] = iQ

    # Construct the Lift chart table
    countTable = pd.crosstab(quantileIndex, DepVar)
    decileN = countTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = countTable[EventValue]
    totalNResponse = gainN.sum(0)
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    overallResponsePct = 100 * (totalNResponse / nObs)
    lift = responsePct / overallResponsePct

    LiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                    axis=1, ignore_index=True)
    LiftCoordinates = LiftCoordinates.rename({0: 'Decile N',
                                              1: 'Decile %',
                                              2: 'Gain N',
                                              3: 'Gain %',
                                              4: 'Response %',
                                              5: 'Lift'}, axis='columns')

    # Construct the Accumulative Lift chart table
    accCountTable = countTable.cumsum(axis=0)
    decileN = accCountTable.sum(1)
    decilePct = 100 * (decileN / nObs)
    gainN = accCountTable[EventValue]
    gainPct = 100 * (gainN / totalNResponse)
    responsePct = 100 * (gainN / decileN)
    lift = responsePct / overallResponsePct

    accLiftCoordinates = pd.concat([decileN, decilePct, gainN, gainPct, responsePct, lift],
                                       axis=1, ignore_index=True)
    accLiftCoordinates = accLiftCoordinates.rename({0: 'Acc. Decile N',
                                                    1: 'Acc. Decile %',
                                                    2: 'Acc. Gain N',
                                                    3: 'Acc. Gain %',
                                                    4: 'Acc. Response %',
                                                    5: 'Acc. Lift'}, axis='columns')

    if (Debug == 'Y'):
        print('Number of Quantiles = ', nQuantile)
        print(quantileCutOff)
        _u_, _c_ = np.unique(quantileIndex, return_counts=True)
        print('Quantile Index: \n', _u_)
        print('N Observations per Quantile Index: \n', _c_)
        print('Count Table: \n', countTable)
        print('Accumulated Count Table: \n', accCountTable)

    return(LiftCoordinates, accLiftCoordinates)


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




##############################################################################################################################################
print("\n\n")
print("-----------------------------------Region 1-----------------------------------------")
#GCBC MODEL
yTrain1 = train_region1[['Maintenance_flag']]
train_region1= train_region1.drop(train_region1[['Maintenance_flag']], axis=1)
train_region1=train_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS','Vehicle_speed_sensor','Throttle_Pos_Manifold','Mass_Air_Flow_Rate']]

yTest1 = test_region1[['Maintenance_flag']]
test_region1= test_region1.drop(test_region1[['Maintenance_flag']], axis=1)
test_region1=test_region1[['Vibration','Engine_RPM','Speed_OBD','Ambient_air_temp','Speed_GPS','Vehicle_speed_sensor','Throttle_Pos_Manifold','Mass_Air_Flow_Rate']]

train1 = yTrain1.loc[yTrain1['Maintenance_flag'] == 1]
claimTrain1 = len(train1.Maintenance_flag)/len(yTrain1.Maintenance_flag)

Y_target_train1 = yTrain1.Maintenance_flag
Y_target_test1 = yTest1.Maintenance_flag

gbm1 = ensemble.GradientBoostingClassifier(loss='deviance', criterion='mse', n_estimators = 1500,
                                         max_leaf_nodes = 5, verbose=1)
fit_gbm1 = gbm1.fit(train_region1, yTrain1)
Y_predict1 = fit_gbm1.predict(test_region1)
Y_predProb1= fit_gbm1.predict_proba(test_region1)

Y_probab1 = pd.DataFrame(
    {0: Y_predProb1[:, 0], 1: Y_predProb1[:, 1]}, index=Y_target_test1.index.copy())
# roc values
fpr1, tpr1, threshold1 = metrics.roc_curve(Y_target_test1, Y_probab1[1])

# Score the test partition

score_test1 = pd.concat([Y_target_test1, Y_probab1], axis=1)

# Get the Lift chart coordinates
lift_coordinates_logistic, acc_lift_coordinates_logistic = compute_lift_coordinates(
    DepVar=score_test1['Maintenance_flag'],
    EventValue=1,
    EventPredProb=score_test1[1],
    Debug='Y')

##############################################################################################################################################
print("\n\n")
print("-----------------------------------Region 2-----------------------------------------")
#GBC MODEL
yTrain2 = train_region2[['Maintenance_flag']]
train_region2= train_region2.drop(train_region2[['Maintenance_flag']], axis=1)
train_region2=train_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude']]

yTest2 = test_region2[['Maintenance_flag']]
test_region2= test_region2.drop(test_region2[['Maintenance_flag']], axis=1)
test_region2=test_region2[['Engine_Oil_Temp','Trip_Distance','Trip_Time_journey','GPS_Altitude']]

train2 = yTrain2.loc[yTrain2['Maintenance_flag'] == 1]
claimTrain2 = len(train2.Maintenance_flag)/len(yTrain2.Maintenance_flag)

Y_target_train2 = yTrain2.Maintenance_flag
Y_target_test2 = yTest2.Maintenance_flag

gbm2 = ensemble.GradientBoostingClassifier(loss='deviance', criterion='mse', n_estimators = 1500,
                                         max_leaf_nodes = 5, verbose=1)
fit_gbm2 = gbm2.fit(train_region2, yTrain2)
Y_predict2 = fit_gbm2.predict(test_region2)
Y_predProb2= fit_gbm2.predict_proba(test_region2)

Y_probab2 = pd.DataFrame(
    {0: Y_predProb2[:, 0], 1: Y_predProb2[:, 1]}, index=Y_target_test2.index.copy())

# roc values
fpr2, tpr2, threshold2 = metrics.roc_curve(Y_target_test2, Y_probab2[1])

# Score the test partition
score_test2 = pd.concat([Y_target_test2, Y_probab2], axis=1)


# Get the Lift chart coordinates
lift_coordinates_tree, acc_lift_coordinates_tree = compute_lift_coordinates(
    DepVar=score_test2['Maintenance_flag'],
    EventValue=1,
    EventPredProb=score_test2[1],
    Debug='Y')


##############################################################################################################################################
print("\n\n")
print("-----------------------------------Region 3-----------------------------------------")
#Tree model

yTrain3 = train_region3[['Maintenance_flag']]
train_region3= train_region3.drop(train_region3[['Maintenance_flag']], axis=1)
#train_region3=train_region3[['GPS_Latitude','Vehicle_speed_sensor']]

yTest3 = test_region3[['Maintenance_flag']]
test_region3= test_region3.drop(test_region3[['Maintenance_flag']], axis=1)
#test_region3=test_region3[['GPS_Latitude','Vehicle_speed_sensor']]


train3 = yTrain3.loc[yTrain3['Maintenance_flag'] == 1]
claimTrain3 = len(train3.Maintenance_flag)/len(yTrain3.Maintenance_flag)


Y_target_train3 = yTrain3.Maintenance_flag
Y_target_test3 = yTest3.Maintenance_flag

classTree3 = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=2, random_state=20181010)
Cluster_DT3 = classTree3.fit(train_region3, Y_target_train3)

Y_predict3 = Cluster_DT3.predict(test_region3)
Y_predProb3 = Cluster_DT3.predict_proba(test_region3)

Y_probab3 = pd.DataFrame(
    {0: Y_predProb3[:, 0], 1: Y_predProb3[:, 1]}, index=Y_target_test3.index.copy())

# roc values
fpr3, tpr3, threshold3 = metrics.roc_curve(Y_target_test3, Y_probab3[1])

# Score the test partition
score_test3 = pd.concat([Y_target_test3, Y_probab3], axis=1)

# Get the Lift chart coordinates
lift_coordinates_KNM, acc_lift_coordinates_KNM = compute_lift_coordinates(
    DepVar=score_test3['Maintenance_flag'],
    EventValue=1,
    EventPredProb=score_test3[1],
    Debug='Y')

##############################################################################################################################################
####ROC



roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc3 = metrics.auc(fpr3, tpr3)


plt.figure(figsize=(10,5))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label='Region1: GBC', markersize=6, marker='o')
plt.plot(fpr2, tpr2, 'r', label='Region2: GBC', markersize=6, marker='o')
plt.plot(fpr3, tpr3, 'g', label='Region3: Tree', markersize=6, marker='o')
plt.legend()
plt.plot([0, 1], [0, 1], 'y--')
plt.xlim([-0.05, 1])
plt.ylim([0, 1.05])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
##############################################################################################################################################

#LIFT


#plot all lift
y1 = lift_coordinates_logistic['Lift']
y2 = lift_coordinates_tree['Lift']
y3 = lift_coordinates_KNM['Lift']

plt.figure(figsize=(10,5))
plt.plot(acc_lift_coordinates_logistic.index, y1, marker='o',
         color='b', linestyle='solid', linewidth=2, markersize=6, label='Region1: GBC')
plt.plot(acc_lift_coordinates_tree.index, y2, marker='o',
         color='r', linestyle='solid', linewidth=2, markersize=6, label='Region2: GBC')
plt.plot(acc_lift_coordinates_KNM.index, y3, marker='o',
         color='g', linestyle='solid', linewidth=2, markersize=6, label='Region3: Tree')
plt.legend()
plt.title("Lift Testing Partition")
plt.grid(True)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Lift")
plt.show()


#plot all acumulative lift
y1 = acc_lift_coordinates_logistic['Acc. Lift']
y2 = acc_lift_coordinates_tree['Acc. Lift']
y3 = acc_lift_coordinates_KNM['Acc. Lift']

plt.figure(figsize=(10,5))
plt.plot(acc_lift_coordinates_logistic.index, y1, marker='o',
          color='b', linestyle='solid', linewidth=2, markersize=6, label='Region1: GBC')
plt.plot(acc_lift_coordinates_tree.index, y2, marker='o',
         color='r', linestyle='solid', linewidth=2, markersize=6, label='Region2: GBC')
plt.plot(acc_lift_coordinates_KNM.index, y3, marker='o',
         color='g', linestyle='solid', linewidth=2, markersize=6, label='Region3: Tree')
plt.title("Accumulatie Lift Testing Partition")
plt.grid(True)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.legend()
plt.show()
