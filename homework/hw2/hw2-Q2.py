#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Title: Homework 2: QUESTION 2
@author: diego
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 2: After you have imported the CSV file, please discover association rules using this dataset.
print('QUESTION 2: \n\n' )
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

groceries = pd.read_csv('Groceries.csv', delimiter=',')

    
### A) How many customers in this market basket data?
print('A): \n' )
nCostumers= groceries.iloc[len(groceries)-1,0]
print('There are: ',nCostumers,'of customers. \n \n')

### B) How many unique items in the market basket across all customers?
print('B): \n' )
nDifItems= len(groceries.Item.unique())
print('There are: ',nDifItems,'of unique items. \n \n')

### C) Create a dataset which contains the number of distinct items in each customer’s market basket. Draw a histogram of the number of unique items.  What are the median, the 25th percentile and the 75th percentile in this histogram?
print('C): \n' )
nItemPerCustomer = groceries.groupby(['Customer'])['Item'].count()
nItemPerCustomer = nItemPerCustomer.sort_values()
print(nItemPerCustomer)
hist = nItemPerCustomer.hist(bins=62)
plt.title("Histogram of frequency of number of items per customer")
plt.xlabel("Number of unique items per costumer")
plt.ylabel("Number of costumers")
plt.show()

print(nItemPerCustomer.describe())
 
### D) Find out the k-itemsets which appeared in the market baskets of at least seventy five (75) customers.  How many itemsets have you found?  Also, what is the highest k value in your itemsets?
print('D): \n' )
itemPerCustomer = groceries.groupby(['Customer'])['Item'].apply(list).values.tolist()
te = TransactionEncoder()
te_ary = te.fit(itemPerCustomer).transform(itemPerCustomer)
itemPerCustomerI = pd.DataFrame(te_ary, columns=te.columns_)
min_sup= 75/len(itemPerCustomer)
frequent_itemsets = apriori(itemPerCustomerI, min_support = min_sup, use_colnames = True)
print('K-itemsets: \n', frequent_itemsets)
print('K-itemsets number: ',len(frequent_itemsets))
print('Highest k number: ', len(frequent_itemsets.itemsets[len(frequent_itemsets)-1]), '\n \n')

### E) Find out the association rules whose Confidence metrics are at least 1%.  How many association rules have you found?  Please be reminded that a rule must have a non-empty antecedent and a non-empty consequent.
print('E): \n' )
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)
print('Association rules: \n ', assoc_rules)
print('Association rules number: ',len(assoc_rules), '\n \n')

### F) Graph the Support metrics on the vertical axis against the Confidence metrics on the horizontal axis for the rules you found in (e).  Please use the Lift metrics to indicate the size of the marker. 
print('F): \n' )
plt.scatter(assoc_rules['confidence'], assoc_rules['support'], s = assoc_rules['lift'])
plt.grid(True)
plt.title("Association rules: confidence vs support")
plt.xlabel("Confidence")
plt.ylabel("Support")
plt.show()

### G) List the rules whose Confidence metrics are at least 60%.  Please include their Support and Lift metrics.
print('G): \n' )
rules0_6=assoc_rules[assoc_rules.confidence >= 0.6]
print('Rules whose Confidence metrics are at least 60%: \n', rules0_6, '\n \n')

### H) What similarities do you find among the consequents that appeared in (g)?
print('H): \n' )
rules0_6C=assoc_rules[assoc_rules.confidence >= 0.6].filter(items=['consequents'])
print('Rules consequents with confidence of 60%: \n', rules0_6C, '\n \n')