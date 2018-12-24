#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
@Title: Homework 4: QUESTION 1
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""

## QUESTION 1:
print('QUESTION 1: \n\n' )

import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model


ChicagoDiabetes = pandas.read_csv('ChicagoDiabetes.csv',
                          delimiter=',')

# Feature variables

X= ChicagoDiabetes[['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002',
                    'Crude Rate 2003','Crude Rate 2004','Crude Rate 2005',
                    'Crude Rate 2006','Crude Rate 2007','Crude Rate 2008',
                    'Crude Rate 2009','Crude Rate 2010','Crude Rate 2011',]]

nObs = X.shape[0]
nVar = X.shape[1]

### A) 
print('A): \n' )
pandas.plotting.scatter_matrix(X, figsize=(20,20), c = 'red',
                               diagonal='hist', hist_kwds={'color':['burlywood']})
plt.show()
print(' \n \n')


### B) 
print('B): \n' )
# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)

print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
cumsum_variance = numpy.cumsum(_thisPCA.explained_variance_)

ExplainedVariances = _thisPCA.explained_variance_

print('Explained Variance: \n', _thisPCA.explained_variance_)
print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance)
print('Principal Components: \n', _thisPCA.components_)


### C) 
print('C): \n' )


plt.plot(_thisPCA.explained_variance_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

print(' \n \n')


plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()


### D) 
print('D): \n' )


cumsum_variance = numpy.cumsum(_thisPCA.explained_variance_)
plt.plot(cumsum_variance, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

print(' \n \n')


### F) 
print('F) & G): \n' )


first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)

# Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 15

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20181010).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters-1):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

print(' \n \n')

### H & I) 
print('H) & I): \n' )
   
# Fit the 4 cluster solution'
kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_
X['Cluster ID']= X_transformed['Cluster ID']

X['Community']= ChicagoDiabetes['Community']

q0= X.loc[X['Cluster ID'] == 0]
q1= X.loc[X['Cluster ID'] == 1]
q2= X.loc[X['Cluster ID'] == 2]
q3= X.loc[X['Cluster ID'] == 3]

q0count= q0['Community'].size
q1count= q1['Community'].size
q2count= q2['Community'].size
q3count= q3['Community'].size

print('Number of communities in Cluster 0 = ', q0count, '\n')
print('Communities in Cluster 0 : \n', list(q0['Community']), '\n')
print('Number of communities in Cluster 1 = ', q1count, '\n')
print('Communities in Cluster 1 : \n', list(q1['Community']), '\n')
print('Number of communities in Cluster 2 = ', q2count, '\n')
print('Communities in Cluster 2 : \n', list(q2['Community']), '\n')
print('Number of communities in Cluster 3 = ', q3count, '\n')
print('Communities in Cluster 3 : \n', list(q3['Community']), '\n')

print(' \n \n')


#q0yearCrudeHosp= suma de todos los crude y division entre q0count (hacer la media vamos)
#bucle for e ir hayando la media de cada año y guardando en un array o alg asi

q0yearCrudeHosp=[]
q1yearCrudeHosp=[]
q2yearCrudeHosp=[]
q3yearCrudeHosp=[]

for i in range(12):
    aux = numpy.sum(q0.iloc[:,i])/q0.iloc[:,i].size
    q0yearCrudeHosp = numpy.append(q0yearCrudeHosp,aux)
for i in range(12):
    aux = numpy.sum(q1.iloc[:,i])/q1.iloc[:,i].size
    q1yearCrudeHosp = numpy.append(q1yearCrudeHosp,aux)
for i in range(12):
    aux = numpy.sum(q2.iloc[:,i])/q2.iloc[:,i].size
    q2yearCrudeHosp = numpy.append(q2yearCrudeHosp,aux)
for i in range(12):
    aux = numpy.sum(q3.iloc[:,i])/q3.iloc[:,i].size
    q3yearCrudeHosp = numpy.append(q3yearCrudeHosp,aux)

qRefyearCrudeHosp=[25.4, 25.8, 27.2, 25.4, 26.2, 26.6, 27.4, 28.7, 27.9, 27.5, 26.8, 25.6]
years= ['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']

plt.figure(figsize=(10,10))
plt.plot(years, q0yearCrudeHosp, 'r', label='Cluster 0', marker = 'o') 
plt.plot(years, q1yearCrudeHosp, 'g', label='Cluster 1', marker = 'o') 
plt.plot(years, q2yearCrudeHosp, 'y', label='Cluster 2', marker = 'o')  
plt.plot(years, q3yearCrudeHosp, 'b', label='Cluster 3', marker = 'o') 
plt.plot(years, qRefyearCrudeHosp, label='Reference', marker = 'o')
plt.grid(True)
plt.xlabel('Years')
plt.ylabel('Hospitalization Crude Rate')
plt.legend()
plt.show()


# Draw the first two PC using cluster label as the marker color 
carray = ['red', 'orange', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = X_transformed[X_transformed['Cluster ID'] == i]
    plt.scatter(x = subData[0],
                y = subData[1], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.axis(aspect = 'equal')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis(aspect = 'equal')
plt.legend(title = 'Cluster ID', fontsize = 12, markerscale = 2)
plt.show()






### G) 
print('G): \n' )

print(' \n \n')

### H) 
print('H): \n' )

print(' \n \n')