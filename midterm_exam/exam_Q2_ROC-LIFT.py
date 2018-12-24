"""
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


# Read the HMEQ data
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


#######################################################################
#######################################################################
#LOGISTIC MODEL
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
y_target_test = Y_target_test.astype('category')

logic = api.MNLogit(y_target_train, train_inputs)
logicModel = logic.fit(method='newton', full_output=True,
                       maxiter=100, tol=1e-8)

y_predProb = logicModel.predict(test_inputs)
# roc values
fpr1, tpr1, threshold1 = metrics.roc_curve(Y_target_test, y_predProb[1])

# Score the test partition

score_test1 = pd.concat([Y_target_test, y_predProb], axis=1)
# Get the Lift chart coordinates
lift_coordinates_logistic, acc_lift_coordinates_logistic = compute_lift_coordinates(
    DepVar=score_test1['CLAIM_FLAG'],
    EventValue=1,
    EventPredProb=score_test1[1],
    Debug='Y')

#######################################################################
#######################################################################
#TREE MODEL
dumTrain2 = pd.get_dummies(train[['CREDIT_SCORE_BAND']].astype('category'))
train_inputs = dumTrain2.join(
    train[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
#train_inputs = api.add_constant(train_inputs, prepend=True)
Y_target_train = train.CLAIM_FLAG

dumTest2 = pd.get_dummies(
    test[['CREDIT_SCORE_BAND']].astype('category'))
test_inputs = dumTest2.join(
    test[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
#test_inputs = api.add_constant(test_inputs, prepend=True)
Y_target_test = test.CLAIM_FLAG
classTree = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=10, random_state=20181010)
Cluster_DT = classTree.fit(train_inputs, Y_target_train)

Y_predict = Cluster_DT.predict(test_inputs)
Y_probab = Cluster_DT.predict_proba(test_inputs)

Y_probab = pd.DataFrame(
    {0: Y_probab[:, 0], 1: Y_probab[:, 1]}, index=Y_target_test.index.copy())

# roc values
fpr2, tpr2, threshold2 = metrics.roc_curve(Y_target_test, Y_probab[1])

# Score the test partition
score_test2 = pd.concat([Y_target_test, Y_probab], axis=1)


# Get the Lift chart coordinates
lift_coordinates_tree, acc_lift_coordinates_tree = compute_lift_coordinates(
    DepVar=score_test2['CLAIM_FLAG'],
    EventValue=1,
    EventPredProb=score_test2[1],
    Debug='Y')


#######################################################################
#######################################################################
#KNM model

dumTrain = pd.get_dummies(train[['CREDIT_SCORE_BAND']].astype('category'))
train_inputs = dumTrain.join(
    train[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
#train_inputs = api.add_constant(train_inputs, prepend=True)
Y_target_train = train.CLAIM_FLAG

dumTest = pd.get_dummies(
    test[['CREDIT_SCORE_BAND']].astype('category'))
test_inputs = dumTest.join(
    test[['BLUEBOOK_1000', 'CUST_LOYALTY', 'MVR_PTS', 'TIF', 'TRAVTIME']])
#test_inputs = api.add_constant(test_inputs, prepend=True)
Y_target_test = test.CLAIM_FLAG

kNNSpec = KNeighborsClassifier(
    n_neighbors=3, algorithm='brute', metric='euclidean')
nbrs = kNNSpec.fit(train_inputs, Y_target_train)
Y_predict = nbrs.predict(test_inputs)
Y_probab = nbrs.predict_proba(test_inputs)

Y_probab = pd.DataFrame(
    {0: Y_probab[:, 0], 1: Y_probab[:, 1]}, index=Y_target_test.index.copy())

# roc values
fpr3, tpr3, threshold3 = metrics.roc_curve(Y_target_test, Y_probab[1])

# Score the test partition
score_test3 = pd.concat([Y_target_test, Y_probab], axis=1)

# Get the Lift chart coordinates
lift_coordinates_KNM, acc_lift_coordinates_KNM = compute_lift_coordinates(
    DepVar=score_test3['CLAIM_FLAG'],
    EventValue=1,
    EventPredProb=score_test3[1],
    Debug='Y')

#######################################################################
#######################################################################
####ROC



roc_auc1 = metrics.auc(fpr1, tpr1)
roc_auc2 = metrics.auc(fpr2, tpr2)
roc_auc3 = metrics.auc(fpr3, tpr3)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr1, tpr1, 'b', label='logistic', markersize=6, marker='o',)
plt.plot(fpr2, tpr2, 'r', label='Tree', markersize=6, marker='o',)
plt.plot(fpr3, tpr3, 'g', label='KNC', markersize=6, marker='o',)
plt.legend()
plt.plot([0, 1], [0, 1], 'y--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
#######################################################################
#######################################################################

#LIFT


#plot all lift
y1 = lift_coordinates_logistic['Lift']
y2 = lift_coordinates_tree['Lift']
y3 = lift_coordinates_KNM['Lift']


plt.plot(acc_lift_coordinates_logistic.index, y1, marker='o',
         color='blue', linestyle='solid', linewidth=2, markersize=6, label='Logistic')
plt.plot(acc_lift_coordinates_tree.index, y2, marker='o',
         color='yellow', linestyle='solid', linewidth=2, markersize=6, label='Tree')
plt.plot(acc_lift_coordinates_KNM.index, y3, marker='o',
         color='red', linestyle='solid', linewidth=2, markersize=6, label='KNM')
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


plt.plot(acc_lift_coordinates_logistic.index, y1, marker='o',
          color='blue', linestyle='solid', linewidth=2, markersize=6, label='Logistic')
plt.plot(acc_lift_coordinates_tree.index, y2, marker='o',
         color='red', linestyle='solid', linewidth=2, markersize=6, label='Tree')
plt.plot(acc_lift_coordinates_KNM.index, y3, marker='o',
         color='green', linestyle='solid', linewidth=2, markersize=6, label='KNM')
plt.title("Accumulatie Lift Testing Partition")
plt.grid(True)
plt.xticks(np.arange(1, 11, 1))
plt.xlabel("Decile Group")
plt.ylabel("Accumulated Lift")
plt.legend()
plt.show()
