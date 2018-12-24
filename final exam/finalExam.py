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
import sklearn.svm as svm
import statsmodels.api as sm
import statsmodels.stats.weightstats as st

fleet_train = pandas.read_csv('fleet_train.csv', delimiter=',')
target_train= fleet_train[['Maintenance_flag']]
fleet_train= fleet_train.drop(fleet_train[['Maintenance_flag','fleetid','truckid','record_id','Measurement_timestamp']], axis=1)

fleet_test = pandas.read_csv('fleet_monitor_notscored_2.csv', delimiter=',')
target_test= fleet_test[['Maintenance_flag']]
fleet_test= fleet_test.drop(fleet_test[['Maintenance_flag','fleetid','truckid','record_id','Measurement_timestamp', 'period']], axis=1)

