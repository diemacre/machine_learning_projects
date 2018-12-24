#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Title: Homework 2: QUESTION 3
@author: diego
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 3: calculate the Elbow value and the Silhouette value. Use the CARS.CSV dataset
print('QUESTION 3: \n\n' )
import numpy as np
import sklearn.cluster as cluster
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score

cars = pd.read_csv('cars.csv', delimiter=',')

### A) List the Elbow values and the Silhouette values for your 1-cluster to 15-cluster solutions.
print('A): \n' )

X = np.array(list(zip(cars['Horsepower'], cars['Weight']))).reshape(cars.shape[0], 2)

# k means determine k
distortions = []
for k in range(1,16):
    kmeanModel = cluster.KMeans(n_clusters=k).fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

# Plot the elbow
K= range(1,16)
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Elbow w')
plt.title('The Elbow Method showing the optimal k')
plt.show()
print(distortions)

silhouette = []
for k in range(2,16):
    clusterer = cluster.KMeans(n_clusters=k)
    cluster_labels = clusterer.fit_predict(X)
    silhouette.append(silhouette_score(X, cluster_labels, metric='euclidean'))

# Plot the silhouette
K= range(2,16)
plt.plot(K, silhouette, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('The silhouette Method showing the optimal k')
plt.show()
print(silhouette)

### B) Based on the Elbow values and the Silhouette values, what do you suggest for the number of clusters?
print('\n \n B): \n')
print('Seeing both previous graphs. At the silhouette graph, for 4 clusters we get the closest value to 1 of the score (0.54147), which means it is most appropriate cluster for the given data. On the other hand, if we have a look on the elbow graph the k=4 is at the “elbow” of the graph. This is also a good indicator that is the proper number of clusters.' )
   
