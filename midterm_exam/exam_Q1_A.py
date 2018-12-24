"""
@author: diego martin crespo
@id: A20432558
@term: Fall 2018
CS-584
"""
import numpy as np
import sklearn.cluster as cluster
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import sklearn.metrics as metrics
from sklearn.metrics import silhouette_score

pothole = pd.read_csv('ChicagoCompletedPotHole.csv', delimiter=',')

inputX = pothole[['N_POTHOLES_FILLED_ON_BLOCK','N_DAYS_FOR_COMPLETION', 'LATITUDE','LONGITUDE']]

inputX['N_POTHOLES_FILLED_ON_BLOCK'] = np.log(inputX['N_POTHOLES_FILLED_ON_BLOCK'])
inputX['N_DAYS_FOR_COMPLETION'] = np.log(1 + inputX['N_DAYS_FOR_COMPLETION'])


trainData= inputX
nData = trainData.shape[0]

# Part (a)
nClusters = np.zeros(15)
Elbow = np.zeros(15)
Silhouette = np.zeros(15)
TotalWCSS = np.zeros(15)
Inertia = np.zeros(15)

for c in range(15):
   KClusters = c + 1
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters,
                           random_state=20181010).fit(trainData)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_

   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(trainData, kmeans.labels_)

   WCSS = np.zeros(KClusters)
   nC = np.zeros(KClusters)

   for i in range(nData):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = trainData.iloc[i, ] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += WCSS[k] / nC[k]
      TotalWCSS[c] += WCSS[k]

   print("Cluster Assignment:", kmeans.labels_)
   for k in range(KClusters):
      print("Cluster ", k)
      print("Centroid = ", kmeans.cluster_centers_[k])
      print("Size = ", nC[k])
      print("Within Sum of Squares = ", WCSS[k])
      print(" ")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(15):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))


plt.plot(range(2,16), Elbow[1:], linewidth=2, marker='o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.title('Elbow Method')
plt.ylabel("Elbow Value")
plt.show()

plt.plot(range(2, 16), Silhouette[1:], linewidth=2, marker='o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.title('Silhouette Method')
plt.ylabel("Silhouette Value")
plt.show()
