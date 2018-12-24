#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@Title: Homework 2: QUESTION 3
@author: diego
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 4: calculate the Elbow value and the Silhouette value. Use the CARS.CSV dataset
print('QUESTION 4: \n\n' )
import numpy as np
import sklearn.cluster as cluster
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.neighbors
import math
from numpy import linalg as LA

Spiral = pd.read_csv('Spiral.csv', delimiter=',')


### A) Generate a scatterplot of y (vertical axis) versus x (horizontal axis).  How many clusters will you say by visual inspection?
print('A): \n' )

nObs = Spiral.shape[0]
plt.scatter(Spiral[['x']], Spiral[['y']])
plt.title('Spiral scatterplot')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

### B) Apply the K-mean algorithm directly using your number of clusters (in a). Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?
print('B): \n' )

trainData = Spiral[['x','y']]
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(trainData)

print("Cluster Centroids = \n", kmeans.cluster_centers_)

Spiral['KMeanCluster'] = kmeans.labels_

for i in range(2):
    print("Cluster Label = ", i)
    print(Spiral.loc[Spiral['KMeanCluster'] == i])

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['KMeanCluster']])
plt.title('Spiral 2 clusters')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

### C) Apply the nearest neighbor algorithm using the Euclidean distance.  How many nearest neighbors will you use?
print('C): \n' )

kNNSpec = sklearn.neighbors.NearestNeighbors(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

distObject = sklearn.neighbors.DistanceMetric.get_metric('euclidean')

distances = distObject.pairwise(trainData)

print('We choose 3 neighboors. \n \n')

### D) Generate the sequence plot of the first nine eigenvalues, starting from the smallest eigenvalues.  Based on this graph, do you think your number of nearest neighbors (in a) is appropriate?
print('D): \n' )
    
Adjacency = np.zeros((nObs, nObs))
Degree = np.zeros((nObs, nObs))

for i in range(nObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(nObs):
    sum = 0
    for j in range(nObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
Lmatrix = Degree - Adjacency

evals, evecs = LA.eigh(Lmatrix)

plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.title('First 9 eigenvalues')
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.show()


### E) Apply the K-mean algorithm on your first two eigenvectors that correspond to the first two smallest eigenvalues. Regenerate the scatterplot using the K-mean cluster identifier to control the color scheme?
print('E): \n' )
    
Z = evecs[:,[0,1]]
print(Z[[0]].mean(), Z[[0]].std())
print(Z[[1]].mean(), Z[[1]].std())

plt.scatter(Z[[0]], Z[[1]])
plt.title('2 smallest eigenvalues')
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.show()

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=0).fit(Z)

Spiral['SpectralCluster'] = kmeans_spectral.labels_

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['SpectralCluster']])
plt.title('Spiral using 2 smallest eigenvalues')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()



### F) Comment on your spectral clustering results?
print('F): \n' )
print('Seeing both p')