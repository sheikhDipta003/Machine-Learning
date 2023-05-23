# -*- coding: utf-8 -*-
"""
Created on Sat May 20 19:39:16 2023
@author: Dipta
"""

"""""""""""
Classification

"""""""""""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris

iris = load_iris()
# load the dataset into 'iris' variable

iris.feature_names
# get 'feature names' of the dataset

Data_iris = iris.data  # put the actual data (150x4) in 'Data_iris' variable

# create a pandas DataFrame
Data_iris = pd.DataFrame(Data_iris, columns = iris.feature_names)

Data_iris['iristype'] = iris.target
# iris.target -> Series indicating the different species of iris flower whose 
# data is recorded in the dataset, Setosa(0), Versicolor(1), Verginica(2)

"""
scatter(x, y, c=None)
c : color, sequence, or sequence of color, optional, default: 'b'
c can be a single color format string, or a sequence of color specifications of
length N, or a sequence of N numbers to be mapped to colors 
"""
plt.scatter(Data_iris.iloc[:,2], Data_iris.iloc[:,3], c = iris.target )
plt.xlabel('Petal Length (cm)', color='red')
plt.ylabel('Petal Width (cm)', color='blue')
plt.show()

x = Data_iris.iloc[:,0:4]
print(x)
y = Data_iris.iloc[:,4]
print(y)


"""""""""""""""
k-NN Classifier
k Nearnest Neighbor

"""""""""""""""

from sklearn.neighbors import KNeighborsClassifier

"""
How to measure 'nearness' to the sample point?

For 'n' sample points,
Minkowski Distance = (sum(|xi-xj|^p + |yi-yj|^p))^(1/p), for all 1<=i,j<=n
if p = 1, Manhattan Distance
if p = 2, Euclidean Distance
if 1 < p < 2, Minkowski Distance
"""
kNN = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 1)
# Classifier implementing the k-nearest neighbors

kNN.fit(x,y)
# Train the classifier with datasets x and y.
# x -> Training data, y -> Target values

x_N = np.array([[5.6,3.4,1.4,0.1]])
kNN.predict(x_N)
# Predict the class labels, i.e. iris-types, for the provided data.
# x_N -> test samples[must be 2D matrix of shape (n_queries, n_features)]
# returns class labels for each data sample

x_N2 = np.array([[7.5,4,5.5,2]])
kNN.predict(x_N2)

"""
How to test our model?

1) 'Randomly' select a fraction, e.g. 10%,20% or 30% of the initial dataset
2) Train the model with rest of the 90%,80% or 70% data
3) Test the model with the data selected in step-1
"""
from sklearn.model_selection import train_test_split
# package for splitting the whole dataset into Train-set and Test-set

"""
random_state: int, RandomState instance or None, default=None
If int, random_state is the seed used by the random number generator; 
if RandomState instance, random_state is the random number generator.

stratify: array-like, default=None
In our case, for example, the test set consists of 20% data from each types of 
iris, not just from one type.
"""
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, 
                                                    train_size = 0.8,
                                                    random_state = 88, 
                                                    shuffle= True,stratify=y)

kNN = KNeighborsClassifier(n_neighbors = 50, metric = 'minkowski', p = 1)

kNN.fit(X_train,y_train)

predicted_types = kNN.predict(X_test)

# calculate accuracy of prediction
from sklearn.metrics import accuracy_score
accuracy_score(y_test,predicted_types)


"""""""""""""""
Decision Tree Classifier

"""""""""""""""

from sklearn.tree import DecisionTreeClassifier

Dt = DecisionTreeClassifier()

Dt.fit(X_train,y_train)

Predicted_types_Dt = Dt.predict(X_test)

accuracy_score(y_test, Predicted_types_Dt)
# higher accuracy than kNN model

"""
k-fold Cross Validation

We go through the entire data set k times. Each time we split, let's say,
20% of the data using different starting point and test our model with that.
In this way, we can actually test our model with the entire dataset, not
just the first 20% as we did before.
"""
from sklearn.model_selection import cross_val_score

Scores_Dt = cross_val_score(Dt, x, y, cv = 10)

"""
Evaluate a score by cross-validation

x: array-like of shape (n_samples, n_features)
The data to fit. Can be for example a list, or an array.

y: array-like of shape (n_samples,) or (n_samples, n_outputs), default=None
The target variable to try to predict

cv: int, cross-validation generator or an iterable, default=None
If int, specify the number of folds in a (Stratified)K-Fold CV
"""

"""""""""""""""
Naive Bayes Classifier

A classifier based on the Bayes' Theorem

"""""""""""""""

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()

NB.fit(X_train,y_train)

Predicted_types_NB = NB.predict(X_test)

accuracy_score(y_test,Predicted_types_NB)

from sklearn.model_selection import cross_val_score
Scores_NB = cross_val_score(NB, x, y, cv = 10)
# equal score to Decision Tree Model

"""""""""""""""
Logistic Regression

A binary classification model, uses logistic function
if 1 >= probability > 0.5 , 'yes' result
if 0 =< probability < 0.5 , 'no' result

"""""""""""""""

from sklearn.datasets import load_breast_cancer

Data_C = load_breast_cancer()

x = Data_C.data
y = Data_C.target

X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.3,
                                                 train_size=0.7, random_state=42)


from sklearn.linear_model import LogisticRegression

Lr = LogisticRegression()

Lr.fit(X_train,y_train)

predicted_classes_Lr = Lr.predict(X_test)

accuracy_score(y_test,predicted_classes_Lr)
# higher than both Naive Bayes and Decision Tree Model

"""""""""""""""
Evaluation Metrics

Confusion Matrix - [TP,FN,FP,TN] - 2x2 - row major
TP - True Positive - actual cancer, predicted cancer
FN - False Negative - actual cancer, predicted 'not cancer'
FP - False Positive - actual 'not cancer', predicted cancer
TN - True Negative - actual 'not cancer', predicted 'not cancer'

Accuracy = (TP+TN)/(TP+FN+FP+TN)
Precision = TP/(TP + FP), or TN/(TN + FN)
Sensitivity/Recall = TP/(TP+FN), or TN/(TN + FP)
F_Score = 2 x precision x recall/(precision + recall)

High precision -> FP low, high recall -> FN low

TPR(True Positive Rate) = TP/(TP + FN)
FPR(False Positive Rate) = FP/(FP + TN)

For a good classifier,
1) We want TPR to be high and FPR to be low, ideally TPR = 1, FPR = 0
2) In ROC Curve (TPR vs FPR), Area Under Curve(AUC) must be high,ideally AUC=1

"""""""""""""""

from sklearn.metrics import confusion_matrix, classification_report

Conf_Mat = confusion_matrix(y_test,predicted_classes_Lr)

Class_rep = classification_report(y_test,predicted_classes_Lr)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

"""
Probability estimates.
The returned estimates for all classes are ordered by the label of classes.

Parameters
X: array-like of shape (n_samples, n_features)
Vector to be scored

Returns
T: array-like of shape (n_samples, n_classes)
The probability of the sample ''for each class'' in the model
"""
y_prob = Lr.predict_proba(X_test)

y_prob = y_prob[:,1]

FPR, TPR, Thresholds = roc_curve(y_test, y_prob)
"""
roc_curve(y_true, y_score)
Compute Receiver operating characteristic (ROC).
Note: this implementation is restricted to the binary classification task.

y_true: ndarray of shape (n_samples,)
True binary labels

y_score: ndarray of shape (n_samples,)
Target scores / probabilty estimates

Returns
fpr: ndarray of shape (>2,)
Increasing false positive rates such that element i is the false positive rate 
of predictions with score >= thresholds[i].

tpr: ndarray of shape (>2,)
Increasing true positive rates such that element i is the true positive rate 
of predictions with score >= thresholds[i].

thresholds: ndarray of shape = (n_thresholds,)
Decreasing thresholds on the decision function used to compute fpr and tpr. 
thresholds[0] represents no instances being predicted and is arbitrarily set 
to max(y_score) + 1.
"""

plt.plot(FPR,TPR)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_prob)


























