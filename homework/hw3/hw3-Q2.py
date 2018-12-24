#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@Title: Homework 3: QUESTION 2
@author: diego
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 1:
print('QUESTION 2: \n\n' )

import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import statsmodels.api as api



# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominal (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pd.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100

   print("Frequency Table: \n")
   print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

   return

# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominalN (
   targetVar,       # target variable
   predictor):      # nominal predictor

   countTable = pd.crosstab(index = predictor, columns = targetVar, margins = True, dropna = True)
   x = countTable.drop('All', 1)
   percentTable = countTable.div(x.sum(1), axis='index')*100

   print("Frequency Table: \n")
   print(countTable)
   print( )
   print("Percent Table: \n")
   print(percentTable)

# Define a function to visualize the percent of a particular target category by a nominal predictor
def TargetPercentByNominal (
   targetVar,       # target variable
   targetCat,       # target category
   predictor,       # nominal predictor
   val4na):         # imputed value for NaN

   crossTable = pd.crosstab(index = predictor.fillna(val4na), columns = targetVar, margins = True, dropna = True)
   crossTable['Percent'] = 100 * (crossTable[targetCat] / crossTable['All'])
   print(crossTable)

   plotTable = crossTable[crossTable.index != 'All']
   plt.bar(plotTable.index, plotTable['Percent'])
   plt.xlabel(predictor.name)
   plt.ylabel('Percent of ' + targetVar.name + ' = ' + str(targetCat))
   plt.grid(True, axis='y')
   plt.show()

   return(crossTable)

# Define a function to visualize the percent of a particular target category by an interval predictor
def TargetPercentByInterval (
   targetVar,       # target variable
   targetCat,       # target category
   predictor,       # nominal predictor
   val4na):         # imputed value for NaN

   crossTable = pd.crosstab(index = predictor.fillna(val4na), columns = targetVar, margins = True, dropna = True)
   crossTable['Percent'] = 100 * (crossTable[targetCat] / crossTable['All'])
   print(crossTable)

   plotTable = crossTable[crossTable.index != 'All']
   plt.scatter(plotTable.index, plotTable['Percent'])
   plt.xlabel(predictor.name)
   plt.ylabel('Percent of ' + targetVar.name + ' = ' + str(targetCat))
   plt.grid(True, axis='both')
   plt.show()

   return(crossTable)

hmeq = pd.read_csv('Purchase_Likelihood.csv', delimiter=',')






### A) 
print('A): \n' )

print(' \n \n')

### B) 
print('B): \n' )

print(' \n \n')

### C) 
print('C): \n' )
nTotal = len(hmeq)
# Generate the frequency table and the bar chart for the BAD target variable
crossTable = pd.crosstab(index = hmeq['A'], columns = ["Count"], margins = True, dropna = False)
crossTable['Percent'] = 100 * (crossTable['Count'] / nTotal)
crossTable = crossTable.drop(columns = ['All'])

print(crossTable, '\n')


plotTable = crossTable[crossTable.index != 'All']
plt.bar(plotTable.index, plotTable['Percent'])
plt.xticks([[0], [1], [2]])
plt.xlabel('A')
plt.ylabel('Percent')
plt.grid(True, axis='y')
plt.show()

# Cross-tabulate A by group_size
resultTable = TargetPercentByNominal(hmeq['A'], 1, hmeq['group_size'], val4na = -1)

# Cross-tabulate A by homeowner
resultTable = TargetPercentByNominal(hmeq['A'], 1, hmeq['homeowner'], val4na = -1)

# Cross-tabulate A by married_couple
resultTable = TargetPercentByNominal(hmeq['A'], 1, hmeq['married_couple'], val4na = -1)

# Model is Origin = Intercept + DriveTrain

model = hmeq['A'].astype('category')
y = model
y_category = y.cat.categories

X = pd.get_dummies(hmeq[['group_size','homeowner', 'married_couple']].astype('category'))


X = pd.get_dummies(hmeq[['group_size']].astype('category'))
X = X.join(hmeq[['homeowner', 'married_couple']])
X = api.add_constant(X, prepend=True)

logit = api.MNLogit(y, X)
print("Name of Target Variable:", logit.endog_names)
print("Name(s) of Predictors:", logit.exog_names)

thisFit = logit.fit(method='newton', full_output = True, maxiter = 100, tol = 1e-7)
thisParameter = thisFit.params

print("Model Parameter Estimates:\n", thisFit.params)
print("Model Log-Likelihood Value:\n", logit.loglike(thisParameter.values))


y_predProb = thisFit.predict(X)
y_predict = pd.to_numeric(y_predProb.idxmax(axis=1))

y_predictClass = y_category[y_predict]

y_confusion = metrics.confusion_matrix(y, y_predictClass)
print("Confusion Matrix (Row is True, Column is Predicted) = \n")
print(y_confusion)

y_accuracy = metrics.accuracy_score(y, y_predictClass)
print("Accuracy Score = ", y_accuracy)


print(' \n \n')

### D) 
print('D): \n' )

print(' \n \n')

### E) 
print('E): \n' )

print(' \n \n')

### F) 
print('F): \n' )

print(' \n \n')

### G) 
print('G): \n' )

print(' \n \n')

### H) 
print('H): \n' )

print(' \n \n')