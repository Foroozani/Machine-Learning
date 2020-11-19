Machine-Learning, algorithm exapmles
-------------------------------------------------------------
> Regularization and Overfitting Linear regression

*(Overfitting)*:A very common problem in machine learning (ML) when the model is much more complex than it should be (i.e. for using a lot of features) it may perform very well on the training data, but it performs very badly for new (unseen) data. In such situations, the model can not generalize well. Thereare several methods to tackle this problem:

**How to prevent overfitting?**

The possible solutions are:
* To simplify the model by selecting one with fewer parameters, or by reducing the number of features
* Regulalization 
  * L1 Regularization
  * L2 Regularization
* Dropout Regularization 
* Normalizing input
* Early stopping
* Data Augmentation: To gather more training data
* Model Ensembles 
* Cross-validation


*Optimization algorithm*
1. Mini-batch gradient decent (is a _hyperparameter_)
1. Gradient decent with momentum
1. RMSprop
1. Adam optimization algorithm
1. Learning rate decay 
1. Normalizing inputs


![ezgif com-video-to-gif](https://user-images.githubusercontent.com/46888580/99692928-1bca5e00-2a8b-11eb-9ed8-54c553bb46e5.gif)

**What Is Data Normalization, and Why Do We Need It?**

The process of standardizing and reforming data is called “Data Normalization.” It’s a pre-processing step to eliminate data redundancy. Often, data comes in, and you get the same information in different formats. In these cases, you should rescale values to fit into a particular range, achieving better convergence.


**What Is Dropout and Batch Normalization?**

*Dropout* is a technique of dropping out hidden and visible units of a network randomly to prevent overfitting of data (typically dropping 20 percent of the nodes). It doubles the number of iterations needed to converge the network.
*Batch normalization*: is the technique to improve the performance and stability of neural networks by normalizing the inputs in every layer so that they have mean output activation of zero and standard deviation of one.

**What Will Happen If the Learning Rate Is Set Too Low or Too High?**

* When your learning rate is too low, training of the model will progress very slowly as we are making minimal updates to the weights. It will take many updates before reaching the minimum point
* If the learning rate is set too high, this causes undesirable divergent behavior to the loss function due to drastic updates in weights. It may fail to converge (model can give a good output) or even diverge (data is too chaotic for the network to train).


**Hyperparameter tuning**
* We need to tune our hyperparameters to get the best out of them. 
* Hyperparameters are important:

1. Learning rate `alpha`
2. momentum beta
3. monibatch size
4. No. of hidden inputs
5. No. of layers
6. Learning rate decay
7. Regularization lambda 
8. Activation function 
9. Adam `beta1` & `beta2`

It is hard to decide which hyperparemeter is the most important in a problem. TIt depends a lit on your problem. One of the ways to tune is to sample a grid with `N` hyperparameter settings and then try all settings combinations on your problem. One can use coarse to fine sampling.

---   
 >  Support Vector Machines 
   
  The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

  We can also use "GridSearch" method to improve the accuracy.

---

> K-Means algorithm 

K Means Clustering is an unsupervised learning algorithm that tries to cluster data based on their similarity. For this exaple, first we cleare an artificial data 
then we use scikit-learn library to cluster the data.

![K-Means](https://github.com/Foroozani/Machine-Learning1/blob/master/03-kmeans.png)
![example](https://github.com/Foroozani/Machine-Learning1/blob/master/figures/clustering1.png)
---
> Neural-Nets 

Keras Regession: In this example I use TF Regression and try to predict the house prices by using them. As you can guess, there are various methods to suceed this and each method has pros and cons.

The data set can be found at Kaggle: 
[visit the website](https://www.kaggle.com/harlfoxem/housesalesprediction)

---

> Principal Component Analysis (PCA)

Large datasets are increasingly common and are often difficult to interpret. Principal component analysis (PCA) is a technique for reducing the dimensionality of such datasets, increasing interpretability but at the same time minimizing information loss. It does so by creating new uncorrelated variables that successively maximize variance.

![](https://github.com/Foroozani/Machine-Learning1/blob/master/figures/pca2.png)


