#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@Title: Homework 4: QUESTION 2
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 1:
print('QUESTION 1: \n\n' )

import numpy
import pandas

import sklearn.naive_bayes as NB

purchases = pandas.read_csv('Purchase_Likelihood.csv',
                          delimiter=',')
y=purchases['A']
X=purchases[['group_size', 'homeowner', 'married_couple']]


classifier = NB.MultinomialNB().fit(X, y)


#print('Log Class Probability:\n', classifier.class_log_prior_ )
#print('Feature Count (after adding alpha):\n', classifier.feature_count_)
#print('Log Feature Probability:\n', classifier.feature_log_prob_)





#X_test = numpy.array([[3,0,0,0,1,1],
#                      [0,1,1,0,1,1]])


### A) 
print('A): \n' )

print('Class Count:', classifier.class_count_)

print(' \n \n')

### B) 
print('B): \n' )

predProb = classifier.predict_proba([[1,0,0]])
print(predProb)

print(' \n \n')

### C) 
print('C): \n' )

predProb = classifier.predict_proba([[2,1,1]])
print(predProb)

print(' \n \n')

### D) 
print('D): \n' )

predProb = classifier.predict_proba([[3,1,1]])
print(predProb)

print(' \n \n')

### E) 
print('E): \n' )

predProb = classifier.predict_proba([[4,0,0]])
print(predProb)

print(' \n \n')

### F) 
print('F): \n' )

predProb = classifier.predict_proba(X)
predProb =pandas.DataFrame(predProb)

maximum= predProb[1].max()
index= predProb[1].idxmax()
print(maximum)
number= X.loc[predProb[1] == maximum]


print(' \n \n')

### G) 
print('G): \n' )

result = pandas.concat([predProb, X], axis=1, sort=False)

predProb = classifier.predict_proba([[1,1,1]])
print(predProb)
predProb = classifier.predict_proba([[1,0,1]])
print(predProb)
predProb = classifier.predict_proba([[1,1,0]])
print(predProb)
predProb = classifier.predict_proba([[1,0,0]])
print(predProb)

predProb = classifier.predict_proba([[2,1,1]])
print(predProb)
predProb = classifier.predict_proba([[2,0,1]])
print(predProb)
predProb = classifier.predict_proba([[2,1,0]])
print(predProb)
predProb = classifier.predict_proba([[2,0,0]])
print(predProb)

predProb = classifier.predict_proba([[3,1,1]])
print(predProb)
predProb = classifier.predict_proba([[3,0,1]])
print(predProb)
predProb = classifier.predict_proba([[3,1,0]])
print(predProb)
predProb = classifier.predict_proba([[3,0,0]])
print(predProb)

predProb = classifier.predict_proba([[4,1,1]])
print(predProb)
predProb = classifier.predict_proba([[4,0,1]])
print(predProb)
predProb = classifier.predict_proba([[4,1,0]])
print(predProb)
predProb = classifier.predict_proba([[4,0,0]])
print(predProb)

print(' \n \n')

