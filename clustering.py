"""
Created on Sat Oct  3 15:08:19 2020

@author: najmeh
"""


import math
import scipy
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import ipywidgets as widgets

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# our modules
from utils import *
from plot_utils import *

# ignore warnings
#import warnings
warnings.filterwarnings('ignore')
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 120
plt.rcParams['image.cmap'] = 'Spectral'

# create random data
X, y = make_blobs(n_samples=150, centers=3, cluster_std=1.2, random_state=10)

# plot data
plt.scatter(X[:, 0], X[:, 1], edgecolors='k', s=50, c='w')

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
print(labels)


plt.scatter(X[:, 0], X[:, 1], edgecolors='k', s=50, c=labels)

# Normalize X
mu = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mu) / std

# plot kmeans steps
initial_centroids = np.array([[1.1, -0.2], [-2.0, -0.5], [-1.8, -0.1]])
cluster_ids, centroids = plot_kmeans(X, initial_centroids)

plot_kmeans_interactive()


def update_kmeans_plot(K):
    # cluster data
    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(X)
    
    # show clustering result
    plt.scatter(X[:, 0], X[:, 1], edgecolors='k', s=50, c=labels, alpha=0.8)
    plt.title("K = {}, Cost = {:.2f}".format(K, kmeans.inertia_))
    plt.axis('equal')
    

K = widgets.IntSlider(value=2, min=1, max=X.shape[0], step=1, description='K:')
widgets.interact(update_kmeans_plot, K=K);


costs = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=0).fit(X)
    costs.append(km.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), costs, marker='o')
plt.xticks(range(1, 11))
plt.xlabel('Number of clusters')
plt.ylabel('Cost')
plt.show()

# load digits dataset
from sklearn.datasets import load_digits
digits = load_digits()


# cluster digits to 10 clusters
kmeans = KMeans(n_clusters=10)
cluster_ids = kmeans.fit_predict(digits.data)
print(kmeans.cluster_centers_.shape)


# visualize the cluster centers
fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i)
    ax.imshow(kmeans.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
    ax.grid(False)
    ax.axis('off')


# visualize the projected data
from sklearn.manifold import Isomap
X_iso = Isomap(n_neighbors=10).fit_transform(digits.data)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].scatter(X_iso[:, 0], X_iso[:, 1], c=cluster_ids)
ax[1].scatter(X_iso[:, 0], X_iso[:, 1], c=digits.target);


#%%
#-----------------------------
"""
One application
Visualize the projected digits, use the cluster labels as the color. 
"""
# read image
import imageio
img = imageio.imread("data/flower.png")

# compress using k-means
K = 10
img_compressed = compress(img, K=K)

# plot original image
plt.subplot(121)
plt.imshow(img)
plt.title("Original Image")
plt.grid(False)

# plot compressed image
plt.subplot(122)
plt.imshow(img_compressed)
plt.title("Compressed Image (K = {})".format(K))
plt.grid(False)

#%%
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

A = img.reshape((img.shape[0] * img.shape[1], 3))

ax.scatter3D(A[:2000, 0], A[:2000, 1], A[:2000, 2], color=A[:2000]/255.0);




