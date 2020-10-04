"""
               Principal Component Analysis (PCA)

For a complete introduction, please see Dimensionality Reduction

Here we'll explore Principal Component Analysis, which is an extremely useful
linear dimensionality reduction technique.The goal is to reduce the size 
(dimensionality) of a dataset while capturing most of its information.

There are many reason why dimensionality reduction can be useful:

    It can reduce the computational cost when running learning algorithms,
    decrease the storage space, and
    may help with the so-called "curse of dimensionality"
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact

from sklearn.decomposition import PCA

# use seaborn plotting style defaults
import seaborn as sns; sns.set()

from plot_utils import *
plt.rcParams['figure.figsize'] = (10, 8)

# create random data from normal distribution
np.random.seed(1)
X = np.dot(np.random.random(size=(2, 2)), np.random.normal(size=(2, 200))).T

# plot data
c = PCA().fit_transform(X)[:, 0]  # colors
plt.scatter(X[:, 0], X[:, 1], s=50, c=c, cmap='viridis')
plt.title('Original Data')
plt.axis('equal')
plt.show()

#Now, our goal is:

   # to find Principal Axes in the data, and
   # explain how important those axes are in describing the data distribution

pca = PCA(n_components=2).fit(X)

U = pca.components_          # Principal Components (directions)
S = pca.explained_variance_  # importance of ecah direction (variances)

print("1st Principal Component: {} ({:.2f})".format(U[0], S[0]))
print("2nd Principal Component: {} ({:.2f})".format(U[1], S[1]))

#Matrix U is an orthogonal matrix: An orthogonal matrix is a square matrix whose columns and rows are orthogonal unit vectors.

# plot data
plt.scatter(X[:, 0], X[:, 1], s=50, c=c, cmap='viridis', alpha=0.5)

plt.arrow(0, 0, 3 * np.sqrt(S[0]) * U[0, 0], 3 * np.sqrt(S[0]) * U[0, 1], width=.03, head_width=.1, color='k')
plt.arrow(0, 0, 3 * np.sqrt(S[1]) * U[1, 0], 3 * np.sqrt(S[1]) * U[1, 1], width=.03, head_width=.1, color='k')

plt.title("Principal Components")
plt.axis('equal')
plt.show()


pca = PCA(0.95) # keep 95% of variance
X_proj = pca.fit_transform(X)

print(X.shape)
print(X_proj.shape)

#%%
X_approx = pca.inverse_transform(X_proj)

plt.scatter(X[:, 0], X[:, 1], s=50, c=c, cmap='viridis', alpha=0.2)                # plot original data
plt.scatter(X_approx[:, 0], X_approx[:, 1], s=50, c=c, cmap='viridis', alpha=0.9)  # plot projected data
plt.title("Projected Data")
plt.axis('equal')
plt.show()

#%%
"""PCA as data compression"""
#PCA can be used for is a sort of data compression. Using a small n_components allows you to represent a high dimensional point as a sum of just a few principal vectors.

def update_pca_plot(i, n_components):
    pca = PCA(n_components).fit(X)
    im = pca.inverse_transform(pca.transform(X[i:i+1]))
    
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(im.reshape((8, 8)), cmap='binary')
    plt.title('Approximated Data (k={})'.format(n_components))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(X[i].reshape((8, 8)), cmap='binary')
    plt.title('Original Data')
    plt.axis('off')
    plt.show()


idx = widgets.IntSlider(value=20, min=0, max=1796, desc='data')
interact(update_pca_plot, i=idx, n_components=range(1, 65));

def show_all_digit_components(X, index=None):
    index = np.random.choice(X.shape[0]) if index is None else index
    
    fig, axes = plt.subplots(8, 8, figsize=(10, 10))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    for i, ax in enumerate(axes.flat):
        pca = PCA(i + 1).fit(X)
        im = pca.inverse_transform(pca.transform(X[index:index+1]))

        ax.imshow(im.reshape((8, 8)), cmap='binary')
        ax.text(0.95, 0.05, '{0}'.format(i + 1), ha='right', transform=ax.transAxes, color='r')
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show()
    

show_all_digit_components(X, 0)


#%%
"""MNIST Dataset"""

import gzip, pickle

DATA_PATH = 'data/mnist.pkl.gz'

with gzip.open(DATA_PATH, 'rb') as f:
    (X, y), _, _ = pickle.load(f, encoding='latin1')

# As a sanity check, we print out the size of the data.
print('Training data shape:    ', X.shape)
print('Training labels shape:  ', y.shape)


plt.figure(figsize=(12, 6))

pca = PCA().fit(X)  # Notice

plt.bar(range(200),
        pca.explained_variance_ratio_[:200],
        alpha=0.8,
        align='center')

plt.step(range(200),
         np.cumsum(pca.explained_variance_ratio_[:200]),
         where='mid')

plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');


def plot_digits(n_components):
    fig = plt.figure(figsize=(16, 8))
    nside = 10
    
    pca = PCA(n_components).fit(X)
    X_proj = pca.inverse_transform(pca.transform(X[:nside ** 2]))
    X_proj = np.reshape(X_proj, (nside, nside, 28, 28))
    total_var = pca.explained_variance_ratio_.sum()
    
    plt.subplot(121)
    im = np.vstack([np.hstack([X_proj[i, j] for j in range(nside)])
                    for i in range(nside)])
    plt.imshow(im, cmap='binary')
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
    plt.axis('off')
    plt.clim(0, 1)
    
    plt.subplot(122)
    X_org = X[:nside ** 2].reshape((nside, nside, 28, 28))
    im = np.vstack([np.hstack([X_org[i, j] for j in range(nside)]) for i in range(nside)])
    plt.imshow(im, cmap='binary')
    plt.title("Original Data", size=18)
    plt.axis('off')
    

interact(plot_digits, n_components=[10, 20, 30, 40, 50, 100, 150, 200, 784]);
#%%%
"""Face Dataset"""
from sklearn.datasets import fetch_olivetti_faces

faces = fetch_olivetti_faces()
X, y = faces['data'], faces['target']
print(X.shape)
# select 100 faces randomly
X_samples = np.random.permutation(X)[:100]

fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.subplots_adjust(left=0.2, bottom=0.2, right=0.8, top=0.8, hspace=0.01, wspace=0.01)

for i, ax in enumerate(axes.flat):
    ax.imshow(X_samples[i].reshape((64, 64)), cmap='gray')
    ax.set_xticks(())
    ax.set_yticks(())
plt.show()

def update_pca_face_plot(i, n_components):
    pca = PCA(n_components).fit(X)
    im = pca.inverse_transform(pca.transform(X[i:i+1]))
    
    fig = plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(im.reshape((64, 64)), cmap='gray')
    total_var = pca.explained_variance_ratio_.sum()
    plt.title('Approximated Data ({:.2f})'.format(total_var))
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(X[i].reshape((64, 64)), cmap='gray')
    plt.title('Original Data')
    plt.axis('off')
    plt.show()


idx = widgets.IntSlider(value=32, min=0, max=400, desc='data')
widgets.interact(update_pca_face_plot, i=idx, n_components=[10, 50, 100, 150, 200, 300, 400]);



nside=5
X_samples = np.random.permutation(X)[:nside ** 2]


def plot_faces(n_components):
    global X_samples
    
    fig = plt.figure(figsize=(12, 6))    
    pca = PCA(n_components).fit(X)
    X_proj = pca.inverse_transform(pca.transform(X_samples))
    X_proj = np.reshape(X_proj, (nside, nside, 64, 64))
    total_var = pca.explained_variance_ratio_.sum()
    
    
    plt.subplot(121)
    im = np.vstack([np.hstack([X_proj[i, j] for j in range(nside)]) for i in range(nside)])
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
    
    plt.subplot(122)
    X_org = np.reshape(X_samples, (nside, nside, 64, 64))
    im = np.vstack([np.hstack([X_org[i, j] for j in range(nside)]) for i in range(nside)])
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title("Original Faces", size=18)

interact(plot_faces, n_components=[10, 20, 30, 40, 50, 100, 150, 200]);


def plot_faces_components(size):
    n_components = size ** 2
    pca = PCA(n_components).fit(X)
    C = np.reshape(pca.components_[:n_components], (size, size, 64, 64))
    total_var = pca.explained_variance_ratio_.sum()
    
    
    im = np.vstack([np.hstack([C[i, j] for j in range(size)]) for i in range(size)])

    fig = plt.figure(figsize=(10, 10))
    plt.imshow(im, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
    plt.show()


size = widgets.IntSlider(value=5, min=1, max=64, step=1, desc='size')
interact(plot_faces_components, size=size);


def plot_faces_components(size, index):
    n_components = size ** 2
    pca = PCA(n_components).fit(X)
    C = np.reshape(pca.components_[:n_components], (size, size, 64, 64))
    total_var = pca.explained_variance_ratio_.sum()
    
    
    im = np.vstack([np.hstack([C[i, j] for j in range(size)]) for i in range(size)])
    
    x_proj = pca.transform(X[index:index+1])
    x_approx = pca.inverse_transform(x_proj) 
        
    fig = plt.figure(figsize=(18, 6))
    plt.subplot(131)
    plt.imshow(im, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("Principal Components".format(n_components, total_var), size=18)
    
    plt.subplot(132)
    plt.imshow(x_approx.reshape(64, 64), cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("k = {0}, variance = {1:.2f}".format(n_components, total_var), size=18)
        
    plt.subplot(133)
    plt.imshow(X[index].reshape(64, 64), cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title("Original face", size=18)
    plt.show()

size = widgets.IntSlider(value=1, min=1, max=20, step=1, desc='size')
index = widgets.IntSlider(value=32, min=0, max=399, step=1, desc='index')
interact(plot_faces_components, size=size, index=index);