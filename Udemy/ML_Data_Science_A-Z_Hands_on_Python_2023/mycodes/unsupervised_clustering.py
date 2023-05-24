# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:49:43 2023
@author: Dipta
"""

"""""""""""
Clustering

"""""""""""
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
Data_iris = iris.data


"""""""""""
k-mean clustering

Number of clusters = k
Steps
1) Specify 'k'. For optimal value of 'k', use Elbow Point Method.
2) Randomly choose 'k' centroids among the dataset that may or may not 
correspond to any sample data.
3) For each sample point, calculate the distances from each centroid and assign
it to the centroid with the minimum distance, thus forming 'k' clusters.
4) Find the 'actual centroids' of each of the 'k' clusters and reassign all the
sample points according to minimum distance.
5) Keep repeating step (4) until there is no further change in the position
of the centroids.
6) Label the resulting 'k' clusters appropriately.

Evaluation Metrics
Inertia = sum((x_i - xc_j)^2 + (y_i - yc_j)^2),
for all 1<=j<=k, sum over all points (x_i,y_i) within the 'j'-th cluster

Less Inertia, Better clustering

"""""""""""

from sklearn.cluster import KMeans

KMNS = KMeans(n_clusters=3)

KMNS.fit(Data_iris)

Labels = KMNS.predict(Data_iris)

Ctn = KMNS.cluster_centers_

# visulaization of centroids and the sample dataset
# Since 4D visualization is not possible with matplotlib, only the last two
# features are plotted.
plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.scatter(Ctn[:,2],Ctn[:,3], marker= 'o', color = 'red', s=120)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.show()

KMNS.inertia_
# inertia with 3 clusters

"""
inertia vs n_clusters graph
Elbow point: The point in the above graph after which the inertia remains
almost constant. Indicates optimal number of clusters.
"""
K_inertia = []
for i in range(1,10):
    KMNS = KMeans(n_clusters=i, random_state=44)
    KMNS.fit(Data_iris)
    K_inertia.append(KMNS.inertia_)
    
plt.plot(range(1,10),K_inertia, color='green', marker= 'o')
plt.xlabel('number of k')
plt.ylabel('Inertia')
plt.show()
    
"""""""""""
DBSCAN

Density-based Spatial Clustering of Abnormalities with Noise
Used to find out abnormalities/noise/outliers in the dataset.

"""""""""""

from sklearn.cluster import DBSCAN

DBS = DBSCAN(eps = 0.7, min_samples= 4)
"""
eps: float, default=0.5
The maximum distance between two samples for one to be considered as in the 
neighborhood of the other, that is, radius of the neighborhood of the randomly
chosen point as its center.

min_samples: int, default=5
The min number of samples in a neighborhood for a point to be considered as a 
core point. This includes the point itself.
"""

DBS.fit(Data_iris)

Labels = DBS.labels_
"""
-1 indicates outliers.
Smaller epsilon(eps), larger number of outliers.
But if eps is too large, no outlier will be detected.
Larger min_samples, larger number of outliers.
"""

plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.show()

# Poor performance compared to k-Means

"""""""""""
Hierarchical Clustering

"""""""""""
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

HR = linkage(Data_iris, method = 'complete')
# Dnd = dendrogram(HR)
# View changes in the dendrogram with the change in the method name - complete,
# single, average

Labels = fcluster(HR, 3.5, criterion = 'distance') # t=0.8(single),3.5(complete)
"""
fcluster(Z, t, criterion='inconsistent')

Z: ndarray
The hierarchical clustering encoded with the matrix returned by the linkage function.

t: scalar
For criteria 'inconsistent', 'distance' or 'monocrit',
this is the max allowable inter-cluster distance. if 't' is too small, too many
clusters. If 't' is too big, just a single cluster.
For 'maxclust' or 'maxclust_monocrit' criteria,
this would be max number of clusters requested.

criterion: str, optional
The criterion to use in forming flat clusters.

"""

plt.scatter(Data_iris[:,2],Data_iris[:,3], c = Labels)
plt.show()

# Best prediction among all three.


