ve#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:44:06 2018

@author: diego
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas


import sklearn.metrics as metrics
import scipy
import sklearn.svm as svm
import statsmodels.api as sm
import statsmodels.stats.weightstats as st

wineQuality = pandas.read_csv('WineQuality.csv',
                          delimiter=',')
wineQuality= wineQuality.drop(wineQuality[['quality','type']], axis=1)
wine=wineQuality

#wine.loc[wine['type'] == 'white', ['type']] = 0
#wine.loc[wine['type'] == 'red', ['type']]= 1

target= wine[['quality_grp']]
atributes= wine.drop(wine[['quality_grp']], axis=1)

atributes_list= atributes.columns.values.tolist()
'''
for atr in atributes_list:
    
    if atr == 'type':
        wine.boxplot(column=atr, by='quality_grp', vert=False, figsize=(10,5), layout=(1,1))
        plt.xticks(np.arange(2), ('white', 'red'))
        plt.ylabel('quality_gpr')
        plt.xlabel(atr)
        plt.title('')
    else:        
    
    wine.boxplot(column=atr, by='quality_grp', vert=False, figsize=(10,5))
    plt.xlabel(atr)
    plt.ylabel('quality_grp')
    plt.title('')
plt.show()
'''
#print(scipy.stats.ttest_ind(atributes, wine[['quality_grp']]))
#au1= wine[wine['quality_grp'] == 0]['type']
#au2= wine[wine['quality_grp'] == 0]['type']

#print('au1=:', au1)
'''
for atr in atributes_list:
    au1= wine.loc[wine['quality_grp'] == 0, [atr]]
    au2= wine.loc[wine['quality_grp'] == 1, [atr]]
    #statistic, pvalue= scipy.stats.ttest_ind(atributes[[atr]], wine[['quality_grp']])
    statistic, pvalue= scipy.stats.ttest_ind(au1, au2)
    print(atr,',',statistic[0],',', pvalue[0])
    
print('\n')
'''
# Try the sklearn.svm.LinearSVC
#atributes2= atributes.drop(atributes[['citric_acid','volatile_acidity','chlorides','fixed_acidity','residual_sugar','free_sulfur_dioxide','total_sulfur_dioxide']], axis=1)
trainData = atributes[['volatile_acidity', 'chlorides', 'density', 'alcohol']]
#trainData =atributes
#trainData = atributes2
yTrain = target

svm_Model = svm.LinearSVC(verbose = 1, random_state = 20181111, max_iter = 10000)
thisFit = svm_Model.fit(trainData, yTrain)

print('\nIntercept:\n', thisFit.intercept_)
print('\nWeight Coefficients:\n', thisFit.coef_)

y_predictClass = thisFit.predict(trainData)

print('\nMean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
trainData['_PredictedClass_'] = y_predictClass

svm_Mean = trainData.groupby('_PredictedClass_').mean()
print(svm_Mean)

svm_Mean25 = trainData.groupby('_PredictedClass_').quantile(0.25)
print(svm_Mean25)

svm_Mean75 = trainData.groupby('_PredictedClass_').quantile(0.75)
print(svm_Mean75)


'''
carray = ['red', 'green']
plt.figure(figsize=(10,10))
for i in range(2):
    subData = trainData[trainData['_PredictedClass_'] == i]
    plt.scatter(x = subData['alcohol'],
                y = subData['pH'], 
                c = carray[i], 
                label = i, 
                s = 10)
    
plt.scatter(x = svm_Mean['alcohol'], y = svm_Mean['pH'], c = ['black','blue'], marker = 'X', s = 150)
#plt.plot([12.95, 10.75], [0, 40], color = 'black', linestyle = ':')
plt.grid(True)
plt.title('Support Vector Machines')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(title = 'Predicted Class', loc = 'best', bbox_to_anchor = (1, 1), fontsize = 14)
plt.show()
'''