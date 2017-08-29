
# Unsupervised Learning

Unsupervised learning is where you only have input data (X) and no corresponding output variables.

The goal for unsupervised learning is to model the underlying pattern or distribution in the data in order to learn more about the data.

Common clustering algorithms include:
* __K-means__: Partitions data into k distinct clusters based on distance to the centroid of a cluster
* __Hierarchical clustering__: Builds a multilevel hierarchy of clusters by creating a cluster tree
* __KNN__: Non parametric lazy learning algorithm. When you say a technique is non parametric , it means that it does not make any assumptions on the underlying data distribution. It's lazy because it does not use the training data points to do any generalization.
* __K-mode__: Partition the objects into k groups such that the distance from objects to the assigned cluster modes is minimized.

Let's import the training data, which we saved in first tutorial. [Click here](https://github.com/anantbara/Machine-Learning-Tutorials/blob/master/UG/2.%20Data_Preparation/Data%20Preparation.md) to refer Tutorial-1.


```python
import pickle

# Enter your folder path here where you saved your data as a pickle file in last tutorial
saved_pickle_path = "Data/"

with open(saved_pickle_path+"train_df.pickle", "rb") as f:
    train_df = pickle.load(f)
    
with open(saved_pickle_path+"test_df.pickle", "rb") as f:
    test_df = pickle.load(f)
    
with open(saved_pickle_path+"combine.pickle", "rb") as f:
    combine = pickle.load(f)
    
print(train_df.shape, test_df.shape)
```

    (891, 6) (418, 5)
    

Prepare the data by seperating the input data and label data. Also split the data into train and test data.


```python
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']

# train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_test.shape)
```

    (623, 5) (268,)
    

## Hierarchical clustering

Hierarchical clustering is a general family of clustering algorithms that build nested clusters by merging or splitting them successively. This hierarchy of clusters is represented as a tree (or dendrogram). The root of the tree is the unique cluster that gathers all the samples, the leaves being the clusters with only one sample.

The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together.

The only problem with the technique is that it is able to only handle small number of data-points and is very time consuming. This is because it tries to calculate the distance between all possible combination and then takes one decision to combine two groups/individual data-point. That's why it is not used in industry for development and not covered in this tutorial.

But if you want to learn more about this then [click here](https://pythonprogramming.net/hierarchical-clustering-machine-learning-python-scikit-learn/) to see the practical demonstration of it.

## K-means

The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a criterion known as the inertia or within-cluster sum-of-squares. This algorithm requires the number of clusters to be specified. 


```python
from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np

X_train_np = preprocessing.scale(X_train)
y_train_np = y_train.values
X_test_np = preprocessing.scale(X_test)
y_test_np = y_test.values

#print(X_train)
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train)

correct = 0
for i in range(len(y_test)):
    predict_me = X_test_np[i].reshape(1, -1)
    prediction = kmeans.predict(predict_me)
    if prediction[0] == y_test_np[i]:
        correct += 1

print("Accuracy : ", correct/len(y_test_np))

#score = kmeans.score(X_test, y_test)
#print("Accuracy : ", score)
```

    Accuracy :  0.6007462686567164
    

## K-Modes

k-modes is used for clustering categorical variables. It defines clusters based on the number of matching categories between data points. (This is in contrast to the more well-known k-means algorithm, which clusters numerical data based on Euclidean distance.)


```python
import numpy as np
from kmodes import kmodes

km = kmodes.KModes(n_clusters=2, init='Huang', n_init=5, verbose=1)

km.fit(X_train_np)

correct = 0
for i in range(len(y_test)):
    predict_me = X_test_np[i].reshape(1, -1)
    prediction = km.predict(predict_me)
    if prediction[0] == y_test_np[i]:
        correct += 1

print("\n")
print("Accuracy : ", correct/len(y_test_np))

# Print the cluster centroids
#print(km.cluster_centroids_)
```

    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run 1, iteration: 1/100, moves: 44, cost: 922.0
    Run 1, iteration: 2/100, moves: 0, cost: 922.0
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run 2, iteration: 1/100, moves: 19, cost: 989.0
    Run 2, iteration: 2/100, moves: 0, cost: 989.0
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run 3, iteration: 1/100, moves: 110, cost: 892.0
    Run 3, iteration: 2/100, moves: 8, cost: 892.0
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run 4, iteration: 1/100, moves: 0, cost: 912.0
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run 5, iteration: 1/100, moves: 143, cost: 912.0
    Run 5, iteration: 2/100, moves: 89, cost: 912.0
    Best run was number 3
    
    
    Accuracy :  0.6007462686567164
    

## KNN

KNN falls into Supervised Learning but it's unsupervised version also exist.

Unsupervised KNN is a very simple nonparametric classification algorithm in which you take the kk closest neighbors to a point ("closest" depends on the distance metric you choose) and each neighbor constitutes a "vote" for its label. Then you assign the point the label with the most votes.

In Scikit-Learn package, _NearestNeighbors_ class implements unsupervised nearest neighbors learning. It acts as a uniform interface to three different nearest neighbors algorithms: BallTree, KDTree, and a brute-force algorithm based on routines in sklearn.metrics.pairwise.


```python
from sklearn.neighbors import NearestNeighbors
from matplotlib.colors import ListedColormap
import pylab as pl

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
nbrs.fit(X_train_np)

correct = 0
for i in range(len(y_test)):  
    predict_me = X_test_np[i].reshape(1, -1)
    distances, indices = nbrs.kneighbors(predict_me)
    if y_train_np[indices[0][1]] == y_test_np[i]:
        correct += 1

print("\n")
print("Accuracy : ", correct/len(y_test_np))

#distances, indices = nbrs.kneighbors(X_test_np)
#g = nbrs.kneighbors_graph(X_train_np).toarray()
#print(distances)
#print("Indices : ", indices)
#print(g)
```

    
    
    Accuracy :  0.753731343283582
    

As you can see, KNN gives the best result as compared to kmeans and Kmode. But that doesn't mean that KNN is always the best choice. It totally depends on the data and the problem we are trying to solve. 


```python

```
