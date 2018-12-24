#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@Title: Homework 3: QUESTION 1
@author: diego
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 1: You will use the CART algorithm to build profiles of credit card holders.  The data is the CustomerSurveyData.csv.
"""
Specifications:

Target Variable
•	CreditCard. The type of credit card held.  This variable has five categories which are American Express, Discover, MasterCard, Others, and Visa.
•	Drop all missing values in the target variable.

Nominal Predictors
•	CarOwnership. The type of car ownership.  This variable has three non-missing categories which are Leased, None, and Own.
•	JobCategory. The category of the job held.  This variable has six non-missing categories which are Agriculture, Crafts, Labor, Professional, Sales, and Service.
•	Recode all the missing values into the Missing category.
"""
print('QUESTION 1: \n\n' )

import pandas as pd
import numpy as np
import itertools

customerData = pd.read_csv('CustomerSurveyData.csv', delimiter=',')
customerData.dropna()

### A) 
print('A): What is the Gini metric for the root node? \n' )

tableCounts= pd.value_counts(customerData.CreditCard.values)
print('These table shows the number of each category of CreditCard:\n')
print(tableCounts, '\n')  
def gini(target, dataset):
    total= dataset.CreditCard.size
    tableCounts= pd.value_counts(customerData[target].values)
    suma=0
    for i in tableCounts:
        suma = suma + (i/total)**2
    gini= 1-suma
    return gini
print('The entropy of the root node is: ', gini('CreditCard', customerData))
print(' \n \n')

### B) 
print('B): How many possible binary-splits that you can generate from the CarOwnership predictor? \n' )
k1 = customerData.CarOwnership.nunique()
kposibleCos= 2**(k1-1)-1
print('There are: ', kposibleCos, 'possible binary-splits from CarOwnership predictor')

print(' \n \n')

### C) 
print('C): Calculate the Gini metric for each possibly binary split that you can generate from the CarOwnership predictor. \n' )


def GiniIntervalSplit (inData, split):
   dataTable = inData   
   dataTable['LE_Split'] = (dataTable.iloc[:,0] == split)

   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   print(crossTable)

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableGini = 0
   for iRow in range(nRows-1):
       suma = 0
       rowGini = 0
       for iColumn in range(nColumns-1):
           p2 = (crossTable.iloc[iRow,iColumn]/crossTable.iloc[iRow,(nColumns-1)])**2
           suma += p2
       rowGini= 1- suma
       print('Row = ', iRow, 'Gini =', rowGini)
       print(' ')
       tableGini += rowGini * crossTable.iloc[iRow,(nColumns-1)]
   tableGini = tableGini / crossTable.iloc[(nRows-1),(nColumns-1)]
   return(tableGini, crossTable)

    

inputCarOwn = customerData[['CarOwnership', 'CreditCard']].replace(np.nan, 'Missing', regex=True)
valuesTableCarOwnership = ['None','Lease','Own']
#it is assigned a number to each category of the CarOwnership (0,1,2) with ('None','Lease','Own')
for i in range(len(valuesTableCarOwnership)):
    inputCarOwn = inputCarOwn.replace(valuesTableCarOwnership[i] , i)
inputCarOwn.groupby('CarOwnership').size()
inputCarOwn.groupby('CreditCard').size()

crossTable1 = pd.crosstab(index = inputCarOwn['CarOwnership'], columns = inputCarOwn['CreditCard'], margins = True, dropna = True) 
print(crossTable1, '\n \n')

# Split (Own), (None, Leased)
print('Split (None)->TRUE, (Lease, Own)->FALSE: \n')
ev0, crossOwn = GiniIntervalSplit(inputCarOwn, 0)
print('Split Gini Metric for (None), (Lease, Own) =', ev0, '\n \n')
# Split (Leased), (None, Own)
print('Split (Lease)->TRUE, (None, Own)->FALSE: \n')
ev1, crossLeased = GiniIntervalSplit(inputCarOwn, 1 )
print('Split Gini Metric for (Lease), (None, Own)= ', ev1, '\n \n')
# Split (None), (Leased, Own)
print('Split (Own)->TRUE, (None, Lease)->FALSE: \n')
ev2, crossNone = GiniIntervalSplit(inputCarOwn, 2)
print('Split Gini Metric for (Own), (None, Lease) = ', ev2, '\n \n')

table1 = {'SeqIndexSplit': ['(0), (1, 2)','(1), (0, 2)','(2), (0, 1)'],'ContentBranches': [ '(None), (Lease, Own)', '(Lease), (None, Own)', '(Own), (None, Lease)' ], 'Gini': [ev0, ev1, ev2]}
table1 = pd.DataFrame(data=table1)
print(table1)

print(' \n \n')

### D) 
print('D): What is the optimal split for the CarOwnership predictor? \n' )
print('The optimal split for CarOwnership will be (Lease), (None, Own) as it has the lowest Gini score of 0.765709.')
print(' \n \n')

### E) 
print('E): How many possible binary-splits that you can generate from the JobCategory predictor? \n' )
k2 = customerData.JobCategory.nunique()
kposibleJob= 2**(k2)-1
print('There are: ', kposibleJob, 'possible binary-splits from JobCategory predictor.')
print(' \n \n')

### F) 
print('F): Calculate the Gini metric for each possibly binary split that you can generate from the JobCategory predictor. \n' )
posItem1= np.array(list(itertools.combinations(['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],1)))
posItem2= np.array(list(itertools.combinations(['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],2)))
posItem3= np.array(list(itertools.combinations(['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],3)))

def GiniIntervalSplit1 (inData, split):
   dataTable = inData   
   dataTable['LE_Split'] = (dataTable.iloc[:,0] == split)

   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableGini = 0
   for iRow in range(nRows-1):
       suma = 0
       rowGini = 0
       for iColumn in range(nColumns-1):
           p2 = (crossTable.iloc[iRow,iColumn]/crossTable.iloc[iRow,(nColumns-1)])**2
           suma += p2
       rowGini= 1- suma
       tableGini += rowGini * crossTable.iloc[iRow,(nColumns-1)]
   tableGini = tableGini / crossTable.iloc[(nRows-1),(nColumns-1)]
   return(tableGini, crossTable)
   
def GiniIntervalSplit2 (inData, split1, split2):
   dataTable = inData   
   dataTable['LE_Split'] = ((dataTable.iloc[:,0] == split1) | (dataTable.iloc[:,0] == split2))

   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   

   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableGini = 0
   for iRow in range(nRows-1):
       suma = 0
       rowGini = 0
       for iColumn in range(nColumns-1):
           p2 = (crossTable.iloc[iRow,iColumn]/crossTable.iloc[iRow,(nColumns-1)])**2
           suma += p2
       rowGini= 1- suma
       tableGini += rowGini * crossTable.iloc[iRow,(nColumns-1)]
   tableGini = tableGini / crossTable.iloc[(nRows-1),(nColumns-1)]
   return(tableGini, crossTable)

def GiniIntervalSplit3 (inData, split1, split2, split3):
   dataTable = inData   
   dataTable['LE_Split'] = ((dataTable.iloc[:,0] == split1) | (dataTable.iloc[:,0] == split2) | (dataTable.iloc[:,0] == split3 ))

   crossTable = pd.crosstab(index = dataTable['LE_Split'], columns = dataTable.iloc[:,1], margins = True, dropna = True)   
   nRows = crossTable.shape[0]
   nColumns = crossTable.shape[1]
   
   tableGini = 0
   for iRow in range(nRows-1):
       suma = 0
       rowGini = 0
       for iColumn in range(nColumns-1):
           p2 = (crossTable.iloc[iRow,iColumn]/crossTable.iloc[iRow,(nColumns-1)])**2
           suma += p2
       rowGini= 1- suma
       tableGini += rowGini * crossTable.iloc[iRow,(nColumns-1)]
   tableGini = tableGini / crossTable.iloc[(nRows-1),(nColumns-1)]
   return(tableGini, crossTable)
   
inputJobCat = customerData[['JobCategory', 'CreditCard']].replace(np.nan, 'Missing', regex=True)
inputJobCat.groupby('JobCategory').size()
inputJobCat.groupby('CreditCard').size()

evs1=[]
for i in range(len(posItem1)):
    evr1, cross1= GiniIntervalSplit1(inputJobCat, posItem1[i,0])
    #print('Split Gini Metric of(',posItem1[i,0] ,') and rest', evr1, '\n \n')    
    evs1.append(evr1)

evs2=[]
for i in range(len(posItem2)):
    evr2, cross2= GiniIntervalSplit2(inputJobCat, posItem2[i,0], posItem2[i,1])
    #print('Split Gini Metric of(',posItem2[i,0], posItem2[i,1] ,') and rest', evr2, '\n \n')    
    evs2.append(evr2)

evs3=[]
for i in range(len(posItem3)):
    evr3, cross3= GiniIntervalSplit3(inputJobCat, posItem3[i,0], posItem3[i,1], posItem3[i,2])
    #print('Split Gini Metric of(',posItem3[i,0], posItem3[i,1], posItem3[i,2],') and rest', evr3, '\n \n')    
    evs3.append(evr3)

evsfinal=evs1 + evs2 + evs3
print(evsfinal)
print(len(evsfinal))

pIlist1= list(itertools.combinations(['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],1))
pIlist2= list(itertools.combinations(['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],2))
pIlist3= list(itertools.combinations(['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],3))
posfinal= pIlist1 + pIlist2 + pIlist3
pI1= list(itertools.combinations([0,1,2,3,4,5,6],1))
pI2= list(itertools.combinations([0,1,2,3,4,5,6],2))
pI3= list(itertools.combinations([0,1,2,3,4,5,6],3))
pt=pI1+pI2+pI3

jobs= ['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing']
ijos= [0,1,2,3,4,5,6]


print(len(posfinal))
final= {'Index_Split': pt,'Split': posfinal, 'Gini_Value': evsfinal}
finalTable= pd.DataFrame(final)

posfinalrest=[['Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Labor', 'Professional', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Crafts', 'Professional', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Professional', 'Service', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service'],
     ['Labor', 'Professional', 'Sales', 'Service', 'Missing'],
     ['Crafts', 'Professional', 'Sales', 'Service', 'Missing'],
     ['Crafts','Labor', 'Sales', 'Service', 'Missing'],
     ['Crafts','Labor', 'Professional', 'Service', 'Missing'],
     ['Crafts','Labor', 'Professional', 'Sales', 'Missing'],
     ['Crafts','Labor', 'Professional', 'Sales', 'Service'],
     ['Agriculture', 'Professional', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Labor', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Labor', 'Professional', 'Service', 'Missing'],
     ['Agriculture','Labor', 'Professional', 'Sales', 'Missing'],
     ['Agriculture','Labor', 'Professional', 'Sales', 'Service'],
     ['Agriculture','Crafts', 'Sales', 'Service', 'Missing'],
     ['Agriculture','Crafts', 'Professional', 'Service', 'Missing'],
     ['Agriculture','Crafts', 'Professional', 'Sales', 'Missing'],
     ['Agriculture','Crafts', 'Professional', 'Sales', 'Service'],
     ['Agriculture','Crafts','Labor', 'Service', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Sales', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Sales', 'Service'],
     ['Agriculture','Crafts','Labor', 'Professional', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Professional', 'Service'],
     ['Agriculture','Crafts','Labor', 'Professional', 'Sales'],
     ['Professional', 'Sales', 'Service', 'Missing'],
     ['Labor', 'Sales', 'Service', 'Missing'],
     ['Labor', 'Professional',  'Service', 'Missing'],
     ['Labor', 'Professional', 'Sales',  'Missing'],
     ['Labor', 'Professional', 'Sales', 'Service'],
     ['Crafts', 'Sales', 'Service', 'Missing'],
     ['Crafts', 'Professional', 'Service', 'Missing'],
     ['Crafts', 'Professional', 'Sales',  'Missing'],
     ['Crafts', 'Professional', 'Sales', 'Service'],
     ['Crafts','Labor', 'Service', 'Missing'],
     ['Crafts','Labor', 'Sales', 'Missing'],
     ['Crafts','Labor', 'Sales', 'Service'],
     ['Crafts','Labor', 'Professional', 'Missing'],
     ['Crafts','Labor', 'Professional', 'Service'],
     ['Crafts','Labor', 'Professional', 'Sales'],
     ['Agriculture', 'Sales', 'Service', 'Missing'],
     ['Agriculture', 'Professional', 'Service', 'Missing'],
     ['Agriculture', 'Professional', 'Sales', 'Missing'],
     ['Agriculture', 'Professional', 'Sales', 'Service'],
     ['Agriculture','Labor', 'Service', 'Missing'],
     ['Agriculture','Labor', 'Sales',  'Missing'],
     ['Agriculture','Labor', 'Sales', 'Service'],
     ['Agriculture','Labor', 'Professional', 'Missing'],
     ['Agriculture','Labor', 'Professional', 'Service'],
     ['Agriculture','Labor', 'Professional', 'Sales'],
     ['Agriculture','Crafts', 'Service', 'Missing'],
     ['Agriculture','Crafts', 'Sales', 'Missing'],
     ['Agriculture','Crafts', 'Sales', 'Service'],
     ['Agriculture','Crafts', 'Professional', 'Missing'],
     ['Agriculture','Crafts', 'Professional', 'Service'],
     ['Agriculture','Crafts', 'Professional', 'Sales'],
     ['Agriculture','Crafts','Labor', 'Missing'],
     ['Agriculture','Crafts','Labor', 'Service'],
     ['Agriculture','Crafts','Labor', 'Sales'],
     ['Agriculture','Crafts','Labor', 'Professional'],  
     ]

posIrest2=[[1,2,3,4,5,6],
          [0,2,3,4,5,6],
          [0,1,3,4,5,6],
          [0,1,2,4,5,6],
          [0,1,2,3,5,6],
          [0,1,2,3,4,6],
          [0,1,2,3,4,5],
          [2,3,4,5,6],
          [1,3,4,5,6],
          [1,2,4,5,6],
          [1,2,3,5,6],
          [1,2,3,4,6],
          [1,2,3,4,5],
          [0,3,4,5,6],
          [0,2,4,5,6],
          [0,2,3,5,6],
          [0,2,3,4,6],
          [0,2,3,4,5],
          [0,1,4,5,6],
          [0,1,3,5,6],
          [0,1,3,4,6],
          [0,1,3,4,5],
          [0,1,2,5,6],
          [0,1,2,4,6],
          [0,1,2,4,5],
          [0,1,2,3,6],
          [0,1,2,3,5],
          [0,1,2,3,4],
          [3,4,5,6],
          [2,4,5,6],
          [2,3,5,6],
          [2,3,4,6],
          [2,3,4,5],
          [1,4,5,6],
          [1,3,5,6],
          [1,3,4,6],
          [1,3,4,5],
          [1,2,5,6],
          [1,2,4,6],
          [1,2,4,5],
          [1,2,3,6],
          [1,2,3,5],
          [1,2,3,4],
          [0,4,5,6],
          [0,3,5,6],
          [0,3,4,6],
          [0,3,4,5],
          [0,2,5,6],
          [0,2,4,6],
          [0,2,4,5],
          [0,2,3,6],
          [0,2,3,5],
          [0,2,3,4],
          [0,1,5,6],
          [0,1,4,6],
          [0,1,4,5],
          [0,1,3,6],
          [0,1,3,5],
          [0,1,3,4],
          [0,1,2,6],
          [0,1,2,5],
          [0,1,2,4],
          [0,1,2,3],
     ]
for i in range(len(finalTable.Split)):
    finalTable.Split[i] = [finalTable.Split[i], posfinalrest[i]]
    finalTable.Index_Split[i]=[finalTable.Index_Split[i],posIrest2[i]]
                

print(' \n \n')

### G) 
print('G): What is the optimal split for the JobCategory predictor? \n' )
print('The optimal split for JobCategory will be:', finalTable.Split[finalTable.Gini_Value.idxmin()],'as it has the minimum Gini=', finalTable.Gini_Value[finalTable.Gini_Value.idxmin()])
print(' \n \n')

### H) 
print('H): \n' )
    
inputJobCat = customerData[['JobCategory', 'CreditCard']].replace(np.nan, 'Missing', regex=True)
valuesTableJobCategory = ['Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing']
#it is assigned a number to each category of the JobCategory (0,1,2,3,4,5,6) with ('Agriculture','Crafts','Labor', 'Professional', 'Sales', 'Service', 'Missing')
for i in range(len(valuesTableJobCategory)):
    inputJobCat = inputJobCat.replace(valuesTableJobCategory[i] , i)
inputJobCat.groupby('JobCategory').size()
inputJobCat.groupby('CreditCard').size()


from sklearn import tree
classTree = tree.DecisionTreeClassifier(criterion='gini', random_state=60616)
print(classTree)

D = np.reshape(np.asarray([inputCarOwn['CarOwnership'], inputJobCat['JobCategory']]), (5000, 2))


hmeq_DT = classTree.fit(D, inputCarOwn['CreditCard'])

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(D, inputCarOwn['CreditCard'])))

import graphviz
print('')
dot_data = tree.export_graphviz(hmeq_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['CarOwnership','JobCategory'],
                                class_names = ['American Express', 'Discover', 'MasterCard', 'Others', 'Visa'])

graph = graphviz.Source(dot_data)
graph

graph.render('2predictorstree')


graph.render('2predictorstree')

D = np.reshape(np.asarray(inputCarOwn['CarOwnership']), (5000, 1))


hmeq_DT = classTree.fit(D, inputCarOwn['CreditCard'])

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(D, inputCarOwn['CreditCard'])))

import graphviz
print('')
dot_data = tree.export_graphviz(hmeq_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['CarOwnership'],
                                class_names = ['American Express', 'Discover', 'MasterCard', 'Others', 'Visa'])

graph = graphviz.Source(dot_data)
graph

graph.render('CarPredictortree')


graph.render('CarPredictortree')

D = np.reshape(np.asarray(inputJobCat['JobCategory']), (5000, 1))


hmeq_DT = classTree.fit(D, inputCarOwn['CreditCard'])

print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(D, inputCarOwn['CreditCard'])))

import graphviz
print('')
dot_data = tree.export_graphviz(hmeq_DT,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = ['JobCategory'],
                                class_names = ['American Express', 'Discover', 'MasterCard', 'Others', 'Visa'])

graph = graphviz.Source(dot_data)
graph

graph.render('Jobpredictortree')


graph.render('Jobpredictortree')
