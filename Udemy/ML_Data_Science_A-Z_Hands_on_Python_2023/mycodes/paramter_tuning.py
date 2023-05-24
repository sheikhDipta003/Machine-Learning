"""""""""""""""""
SVR Hyper Parameter Tuning

"""""""""""""""""
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

housing = fetch_california_housing()
x = housing.data[0:51]   # The complete dataset is too large
y = housing.target[0:51]

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

parameters = {'kernel':['rbf','linear'],
              'gamma':[1,0.1,0.01]}

grid = GridSearchCV(SVR(),parameters, scoring='neg_mean_squared_error', 
                    refit = True, verbose=2)
"""
GridSearchCV(estimator, param_grid, *, scoring=None, refit=True, verbose=0)

Description
Exhaustive search over specified parameter values for an estimator. The parameters 
of the estimator used to apply these methods are optimized by cross-validated 
grid-search over a parameter grid.

scoring: str
Strategy to evaluate the performance of the cross-validated model on the test set.

refit: bool, str, or callable, default=True
Refit an estimator using the best found parameters on the whole dataset.

verbose: int
Controls the verbosity: the higher, the more detailed messages.
"""

grid.fit(x,y)

best_params = grid.best_params_
print(best_params)

"""""""""""""""""
K-Means Hyper Parameter Tuning

"""""""""""""""""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
iris = load_iris()
Data_iris = iris.data

K_inertia = []

for i in range(1,10):
    KMNS = KMeans(n_clusters = i, random_state=44)
    KMNS.fit(Data_iris)
    K_inertia.append(KMNS.inertia_)


"""""""""""""""""
k-NN Hyper Parameter Tuning

Overfitting
Let's imagine we have trained a model so well that it gives us 100% accuracy
for train-set. But then it will not be able to understand most test cases, 
because those test cases may be very different form the samples in our train-set.
Thus, accuracy of test-set will be very low.

Underfitting
On the other hand, if our model is not trained very well, then also it will have
difficulty understanding the test-set. As a result, the accuracy will be low
for both train-set and test-set.

We need to find optimal values for our parameters so that these two effects
are negligible.

"""""""""""""""""
x = iris.data
y = iris.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.30, 
                                                    train_size=0.70, random_state = 22, 
                                                    shuffle=True, stratify = y)

from sklearn.neighbors import KNeighborsClassifier

kNN_accuracy_test = []
kNN_accuracy_train = []

for k in range(1,50):
    kNN = KNeighborsClassifier(n_neighbors=k, metric= 'minkowski', p=1)
    kNN.fit(X_train,y_train)
    kNN_accuracy_train.append(kNN.score(X_train,y_train))
    kNN_accuracy_test.append(kNN.score(X_test,y_test))
    
plt.plot(np.arange(1,50), kNN_accuracy_train, label = 'train')
plt.plot(np.arange(1,50), kNN_accuracy_test, label = 'test')
plt.xlabel('k')
plt.ylabel('Score')
plt.legend()
plt.show()

# From the graph, when k < 10, we have overfitting; when k > 40, underfitting.
# If we take k=18~22, we see that both the train-set and test-set accuracy is high.
    












