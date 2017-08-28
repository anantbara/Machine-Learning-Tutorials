
# Introduction

In last tutorial, we have seen the very first and most important step before actually working on Machine Learning i.e. [Data Preparation](https://github.com/anantbara/Machine-Learning-Tutorials/blob/master/UG/1.%20Data_Preparation/Data%20Preparation.md). Now we are going to explore and experience the power of Machine Learning. Machine learning is the science of getting computers to act without being explicitly programmed. The iterative aspect of machine learning is important because as models are exposed to new data, they are able to independently adapt. They learn from previous computations to produce reliable, repeatable decisions and results. 

## Types of Machine Learning

### 1) Supervised Learning
* In a given example, the input and its corresponding output is also given.
* Denoted by (x,y) where x=input and y=output/label

### 2) Unsupervised Learning
* Output is unknown for each given input
* Input data is clustered/grouped based on their structure/pattern similarity.

### 3) Reinforcement Learning
* Determine what to do based on rewards and punishments.

### 4) Semi-supervised Learning
* Combination of both Supervised and Unsupervised Learning.

Before proceeding to next step, Let us import all the files which we saved in last tutorial.


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
    

As we know trainig data in Supervised machine learning has two components i.e. input and output data. So we have to prepare 2 arrays which holds input data(X) and output data(y) seperately.


```python
X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']
print(X.shape, y.shape)
```

    (891, 5) (891,)
    

It is a good practice to split your training data into 2 parts (7:3) i.e. train data and test data. We use train data to train our model and test data to validate our model.


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(X_train.shape, y_test.shape)
```

    (623, 5) (268,)
    

## Supervised Learning

Supervised learning is where you have input variables (x) and an output variable (Y) and you use an algorithm to learn the mapping function from the input to the output. From it, the supervised learning algorithm seeks to build a model that can make predictions of the response values(output variable) for a new input variable.

Supervised learning problems can be further grouped into regression and classification problems.
* __Classification__: for categorical response values, where the data can be separated into specific “classes”. Ex- “red” or “blue”.
* __Regression__: for continuous-response values. Ex- “dollars” or “weight” because they can have any real values.

Some Popular example of Supervised machine learning algorithm are:
1) Tree
2) Naive Bayes
3) SVM (Support Vector Machine)
4) Logistic Regression

Lets see each one of them by implementing them on our example.

### Tree (Decision Tree)

Decision Trees (DTs) are a non-parametric supervised learning method used for classification and regression. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

For more information about Decision Tree, you can click [here](http://scikit-learn.org/stable/modules/tree.html).




```python
from sklearn.model_selection import cross_val_score
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Accuracy : ",score * 100)
```

    Accuracy :  80.223880597
    

### Naive Bayes

Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem with the “naive” assumption of independence between every pair of features.

For more information about Decision Tree, you can click [here](http://scikit-learn.org/stable/modules/naive_bayes.html).


```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
score = gnb.score(X_test, y_test)
print("Accuracy : ",score * 100)
```

    Accuracy :  77.9850746269
    

### SVM (Support Vector Machine)

Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:
* Effective in high dimensional spaces.
* Still effective in cases where number of dimensions is greater than the number of samples.
* Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
* Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:
* If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.
* SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).


```python
from sklearn import svm   #  SVC, LinearSVC
clf = svm.SVC()
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("Accuracy : ",score * 100)
```

    Accuracy :  81.3432835821
    

### Logistic Regression

Logistic Regression (aka logit, MaxEnt) classifier. It is similar to linear regression, with the only difference being the y data, which should contain integer values indicating the class relative to the observation.

Logistic regression is an estimate of a logit function. Here is how the logit function looks like:
![alt text](Images/logit1.png)


```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#print(logreg.predict(X_test.iloc[[6]]))
score = logreg.score(X_test, y_test)
print("Accuracy : ",score * 100)
```

    Accuracy :  78.7313432836
    

As you noticed that SVM performs better in our test. You can still improve the performance of our model by preparing some usefull features and by also tweaking the hyperparameters of each algorithm.

