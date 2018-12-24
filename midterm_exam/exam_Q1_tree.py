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
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn import metrics
import graphviz
import math

pothole = pd.read_csv('ChicagoCompletedPotHole.csv', delimiter=',')


X = np.array(list(zip(np.log(pothole.N_POTHOLES_FILLED_ON_BLOCK), np.log(
    1+pothole.N_DAYS_FOR_COMPLETION), pothole.LATITUDE, pothole.LONGITUDE)))

kmeanModel = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X)
cluster_labels = kmeanModel.labels_
clusterID = pd.DataFrame(cluster_labels, columns=['CLUSTER_ID'])

X_inputs = pothole[['N_POTHOLES_FILLED_ON_BLOCK',
                    'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE']]
Y_target = cluster_labels
X_name = ['N_POTHOLES_FILLED_ON_BLOCK',
          'N_DAYS_FOR_COMPLETION', 'LATITUDE', 'LONGITUDE']

classTree = tree.DecisionTreeClassifier(
    criterion='entropy', max_depth=2, random_state=20181010)
Cluster_DT = classTree.fit(X_inputs, Y_target)
Y_predict = Cluster_DT.predict(X_inputs)


miss_classi = 1 - metrics.accuracy_score(Y_target, Y_predict)
msse = math.sqrt(metrics.mean_squared_error(Y_target, Y_predict))
print(miss_classi)
print(msse)
print('Accuracy: {:.6f}' .format(classTree.score(X_inputs, Y_target)))

dot_data = tree.export_graphviz(Cluster_DT,
                                out_file=None,
                                impurity=True, filled=True,
                                feature_names=X_name,
                                class_names=['0', '1', '2', '3'])

graph = graphviz.Source(dot_data)
print(graph)

graph.render('Tree Cluster_ID')

