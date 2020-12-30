# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 21:59:36 2020

@author: Allen
"""

from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
import dataloader

model = KMeans(n_clusters=2, affinity='nearest_neighbors',
                           assign_labels='kmeans')
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels,
            s=50, cmap='viridis');