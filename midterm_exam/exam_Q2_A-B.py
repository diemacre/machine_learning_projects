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

autos = pd.read_csv('policy_2001.csv', delimiter=',')

print(len(autos.CLAIM_FLAG))

claim0 = autos.loc[autos['CLAIM_FLAG'] == 0]
claim1 = autos.loc[autos['CLAIM_FLAG'] == 1]

train0, test0 = train_test_split(claim0, test_size=0.3)
test1, train1 = train_test_split(claim1, test_size=0.7)

train = pd.concat([train0,train1], ignore_index=True)
test = pd.concat([test0, test1], ignore_index=True)
print('There are :', len(train.CLAIM_FLAG), ' observations in the train dataset')
print(len(train.CLAIM_FLAG)/len(autos.CLAIM_FLAG))
print('There are :', len(test.CLAIM_FLAG), ' observations in the test dataset')
print(len(test.CLAIM_FLAG)/len(autos.CLAIM_FLAG))

claimTest = len(test1.CLAIM_FLAG)/len(test.CLAIM_FLAG)
print('claim percent for test:',claimTest)
claimTrain = len(train1.CLAIM_FLAG)/len(train.CLAIM_FLAG)
print('claim percent for train:', claimTrain)
