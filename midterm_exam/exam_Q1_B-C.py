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

X = np.array(list(zip(np.log(pothole.N_POTHOLES_FILLED_ON_BLOCK), np.log(
    1+pothole.N_DAYS_FOR_COMPLETION), pothole.LATITUDE, pothole.LONGITUDE)))
kmeanModel = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X)
cluster_labels = kmeanModel.labels_
clusterID = pd.DataFrame(cluster_labels, columns=['CLUSTER_ID'])

inputXclustered = pd.concat([inputX, clusterID], axis=1)
print(inputXclustered)

clusterID0 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 0]
clusterID1 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 1]
clusterID2 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 2]
clusterID3 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 3]
#clusterID4 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 4]
#clusterID5 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 5]
#clusterID6 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 6]
#clusterID7 = inputXclustered.loc[inputXclustered['CLUSTER_ID'] == 7]

#Box-plot of variable: 'N_POTHOLES_FILLED_ON_BLOCK'
n_potholes = [clusterID0['N_POTHOLES_FILLED_ON_BLOCK'],
              clusterID1['N_POTHOLES_FILLED_ON_BLOCK'],
              clusterID2['N_POTHOLES_FILLED_ON_BLOCK'],
              clusterID3['N_POTHOLES_FILLED_ON_BLOCK']]
plot_n_holes = plt.boxplot(n_potholes, labels=[0,1,2,3], vert=False)
plt.title('N_POTHOLES_FILLED_ON_BLOCK by CLUSTER_ID')
plt.xlabel('N_POTHOLES_FILLED_ON_BLOCK')
plt.ylabel('CLUSTER_ID')
#plt.yticks(np.arange(0, 5, 0.5))
plt.show()

#Box-plot of variable: 'N_DAYS_FOR_COMPLETION'
n_potholes = [clusterID0['N_DAYS_FOR_COMPLETION'],
              clusterID1['N_DAYS_FOR_COMPLETION'],
              clusterID2['N_DAYS_FOR_COMPLETION'],
              clusterID3['N_DAYS_FOR_COMPLETION']]
plot_n_holes = plt.boxplot(n_potholes, labels=[0, 1, 2, 3], vert=False)
plt.title('N_DAYS_FOR_COMPLETION by CLUSTER_ID')
plt.xlabel('N_DAYS_FOR_COMPLETION')
plt.ylabel('CLUSTER_ID')
#plt.yticks(np.arange(0, 5, 0.5))
plt.show()

#Box-plot of variable: 'LATITUDE'
n_potholes = [clusterID0['LATITUDE'],
              clusterID1['LATITUDE'],
              clusterID2['LATITUDE'],
              clusterID3['LATITUDE']]
plot_n_holes = plt.boxplot(n_potholes, labels=[0, 1, 2, 3], vert=False)
plt.title('LATITUDE by CLUSTER_ID')
plt.xlabel('LATITUDE')
plt.ylabel('CLUSTER_ID')
#plt.xticks(np.arange(41.6, 42.1, 0.02))
plt.show()

#Box-plot of variable: 'LONGITUDE'
n_potholes = [clusterID0['LONGITUDE'],
              clusterID1['LONGITUDE'],
              clusterID2['LONGITUDE'],
              clusterID3['LONGITUDE']]
plot_n_holes = plt.boxplot(n_potholes, labels=[0, 1, 2, 3], vert=False)
plt.title('LONGITUDE by CLUSTER_ID')
plt.xlabel('LONGITUDE')
plt.ylabel('CLUSTER_ID')
#plt.xticks(np.arange(-87.9, -87.5, 0.02))
plt.show()

#Box-plot of of latitude vs longitud
LABEL_COLOR_MAP = {0: 'r',
                   1: 'g',
                   2: 'b',
                   3: 'c',
#                   4: 'm',
#                   5: 'y',
#                   6: 'k',
#                   7: 'w'
                   }

label_color = [LABEL_COLOR_MAP[l] for l in inputXclustered.CLUSTER_ID]
plt.scatter(inputXclustered.LONGITUDE, inputXclustered.LATITUDE, c=label_color, marker='.')
plt.title('LATITUDE vs LONGITUDE')
plt.ylabel('LATITUDE')
plt.xlabel('LONGITUDE')
plt.show()

