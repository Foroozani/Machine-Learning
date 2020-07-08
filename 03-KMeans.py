"""     K-Means Clustering with Python

K Means Clustering is an unsupervised learning algorithm that tries to cluster
data based on their similarity. Unsupervised learning means that there is no 
outcome to be predicted, and the algorithm just tries to find patterns in the
data. In k means clustering, we have the specify the number of clusters we 
want the data to be grouped into. The algorithm randomly assigns each observation 
to a cluster, and finds the centroid of each cluster. Then, the algorithm
iterates through two steps: Reassign data points to the cluster whose centroid 
is closest. Calculate new centroid of each cluster. These two steps are 
repeated till the within cluster variation cannot be reduced any further. 
The within cluster variation is calculated as the sum of the euclidean 
distance between the data points and their respective cluster centroids
"""

import seaborn as sns 
import matplotlib.pyplot as plt 

# let us create some artificial data for clustering
from sklearn.datasets import make_blobs

data = make_blobs(n_samples=500,n_features=4,centers=6, cluster_std=1.8,random_state=101)
# tuple (n_sample, n_features)
# return X array of shape [n_samples, n_features], y array of shape [n_samples]
# centre is the class, we have n calss of data
data[1]   # is our label, we createted artificially. So we are aware of them
plt.style.use('ggplot')
plt.scatter(data[0][:,0], data[0][:,1], c = data[1])

#%% 
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 6, max_iter = 600)

# Now we fit the model to the features
kmeans.fit(data[0])

kmeans.cluster_centers_
kmeans.labels_    # the label which is predicted and assigned to each data 

# since we created data we can compare the results but with real data we can not do it 
# we can plot and see how it compare with our original 

f, (ax1, ax2) = plt.subplots(1, 2, sharey=True,figsize=(10,6))
ax1.set_title('K Means')
ax1.scatter(data[0][:,0],data[0][:,1],c=kmeans.labels_,cmap='rainbow')
ax2.set_title("Original")
ax2.scatter(data[0][:,0],data[0][:,1],c=data[1],cmap='rainbow')