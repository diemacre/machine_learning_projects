#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@Title: Homework 2: QUESTION 1
@author: diego
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 1: Suppose a market basket can possibly contain these seven items: A, B, C, D, E, F, and G.
print('QUESTION 1: \n\n' )
# Load the PANDAS library
import pandas as pd
import itertools

itemSet= {'Item':['A','B','C','D','E','F','G']}
itemSetDF = pd.DataFrame(data=itemSet)

### A) What is the number of possible itemsets?
print('A): \n' )
allPosItem= 0
for i in range(1, 8):
    allPosItem= allPosItem + len(list(itertools.combinations('ABCDEFG',i)))   
print('All the possible itemsets:' , allPosItem, '\n \n')

### B) List all the possible 1-itemsets.
print('B): \n' )
posItem1= list(itertools.combinations('ABCDEFG',1))
print('All the possible 1-itemsets:' ,posItem1, '\n \n')

### C) List all the possible 2-itemsets.
print('C): \n' )
posItem2= list(itertools.combinations('ABCDEFG',2))
print('All the possible 2-itemsets:' ,posItem2, '\n \n')

### D) List all the possible 3-itemsets.
print('D): \n' )
posItem3= list(itertools.combinations('ABCDEFG',3))
print('All the possible 3-itemsets:' ,posItem3, '\n \n')

### E) List all the possible 4-itemsets.
print('E): \n' )
posItem4= list(itertools.combinations('ABCDEFG',4))
print('All the possible 4-itemsets:' ,posItem4, '\n \n')

### F) List all the possible 5-itemsets.
print('F): \n' )
posItem5= list(itertools.combinations('ABCDEFG',5))
print('All the possible 5-itemsets:' ,posItem5, '\n \n')

### G) List all the possible 6-itemsets.
print('G): \n' )
posItem6= list(itertools.combinations('ABCDEFG',6))
print('All the possible 6-itemsets:' ,posItem6, '\n \n')

### H) List all the possible 7-itemsets.
print('H): \n' )
posItem7= list(itertools.combinations('ABCDEFG',7))
print('All the possible 7-itemsets:' ,posItem7, '\n \n \n \n')



