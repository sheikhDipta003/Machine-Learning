# -*- coding: utf-8 -*-
"""
Created on Mon May 22 19:25:24 2023
@author: Dipta
"""

"""""""""""
Regression 

"""""""""""

from sklearn.datasets import fetch_california_housing
import numpy as np

housing = fetch_california_housing()

x = housing.data

y = housing.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size= 0.75,
                                                    random_state=76)

"""
Normalization is not needed for Multiple Linear Regression. We are doing this 
only for demonstration purposes. Moreover, normalizing the dataset does not
have any negative impact on our model.
"""
from sklearn.preprocessing import MinMaxScaler

Sc = MinMaxScaler(feature_range = (0,1))

X_train = Sc.fit_transform(X_train)

X_test = Sc.fit_transform(X_test)

y_train = y_train.reshape(-1,1)  # convert y_train into 2D array
y_train = Sc.fit_transform(y_train) # argument must be 2D array

# y_test is not normalized because we need to check accuracy against it


"""""""""""
MLR 

Multiple Linear Regression
y = a1 x1 + a2 x2 + a3 x3 + ... + an xn + a0
Here, all xi are independent variables.

"""""""""""

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()

Linear_R.fit(X_train,y_train)

Predicted_values_MLR = Linear_R.predict(X_test)

Predicted_values_MLR = Sc.inverse_transform(Predicted_values_MLR)
"""
Undo the scaling of X according to feature_range.

Params
X: array-like of shape (n_samples, n_features)
Input data that will be transformed. It cannot be sparse.

Returns
Xt: ndarray of shape (n_samples, n_features)
Transformed data.
"""

"""""""""""
Evaluation Metrics 

Mean Absolute Error - MAE
Mean Squared Error - MSE
Root Mean Squared Error - RMSE
Mean Absolute Percentage Error - MAPE
R2_score = Coefficient of Determination

For a good regression model, R2_score must be as low as possible.

"""""""""""

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import math

MAE = mean_absolute_error(y_test,Predicted_values_MLR)

MSE = mean_squared_error(y_test,Predicted_values_MLR)

RMSE = math.sqrt(MSE)

R2 = r2_score(y_test,Predicted_values_MLR)

"""
Manually calculating MAPE just to understand the concept better; using built-in
libraries is not actually needed.
"""
def mean_absolute_percentage_error(y_true,y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred)/y_true))*100

MAPE = mean_absolute_percentage_error(y_test, Predicted_values_MLR)

"""""""""""
PLR 

Polynomial Linear Regression
y = a1 x + a2 x^2 + a3 x^3 + ... + an x^n + a0
x is the only independent variable.

"""""""""""
housing = fetch_california_housing()

x = housing.data[:,5]  # selecting a single feature(#rooms) to showcase PLR

y = housing.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.25,
                                                    train_size= 0.75,
                                                    random_state=76)

from sklearn.preprocessing import PolynomialFeatures

Poly_P = PolynomialFeatures(degree=2)

# transforming X_train according to the requirements of quadratic PLR
X_train = X_train.reshape(-1,1)  # convert X_train into 2D array
Poly_X = Poly_P.fit_transform(X_train) # argument must be 2D array
"""
fit_transform(X)
Fit to data, then transform it to appropriate polynomial features.

Generate a new feature matrix consisting of all polynomial combinations of the 
features with degree less than or equal to the specified degree. For example, 
if an input sample is one dimensional and of the form [a], the degree-2 
polynomial features are [1, a, a^2].
"""
# transforming X_test according to the requirements of quadratic PLR
X_test = X_test.reshape(-1,1)  # convert X_train into 2D array
Poly_Xt = Poly_P.fit_transform(X_test)   # argument must be 2D array

from sklearn.linear_model import LinearRegression

Linear_R = LinearRegression()

Poly_L_R = Linear_R.fit(Poly_X,y_train)
Predicted_value_P = Poly_L_R.predict(Poly_Xt)

from sklearn.metrics import r2_score
R2 = r2_score(y_test,Predicted_value_P)


"""""""""""
Random Forest

A random forest is a meta estimator that fits ''a number of classifying decision 
trees'' on various sub-samples of the dataset and uses averaging to improve the 
predictive accuracy and control over-fitting.

"""""""""""

from sklearn.ensemble import RandomForestRegressor

Random_F = RandomForestRegressor(n_estimators = 500, max_depth = 20, random_state=33)
"""
n_estimators: int, default=100
The number of trees in the forest.

max_depth: int, default=None
The maximum depth of the tree. If None, then nodes are expanded until all 
leaves are pure.
"""

Random_F.fit(X_train,y_train)
Predicted_Val_RF = Random_F.predict(X_test)
# X_train,X_test,y_train,y_test are used from the previous calculations for PLR

MAE = mean_absolute_error(y_test,Predicted_Val_RF)

MSE = mean_squared_error(y_test,Predicted_Val_RF)

RMSE = math.sqrt(MSE)

R2 = r2_score(y_test,Predicted_Val_RF)

MAPE = mean_absolute_percentage_error(y_test, Predicted_Val_RF)

"""""""""""
SVR

Support Vector Regression

Uses Kernel Function. These functions take the input samples and transform them
into the specific format the SVR model needs.
Different Kernel Functions- Polynomial, Gaussian, Gaussian Radial Basis Func(RBF),
Sigmoid, Hyperbolic Tangent.
"""""""""""

from sklearn.svm import SVR

Regressor_SVR = SVR(kernel='rbf')

Regressor_SVR.fit(X_train,y_train)
Predicted_values_SVR = Regressor_SVR.predict(X_test)
# X_train,X_test,y_train,y_test are used from the previous calculations for PLR

MAE = mean_absolute_error(y_test,Predicted_values_SVR)

MSE = mean_squared_error(y_test,Predicted_values_SVR)

RMSE = math.sqrt(MSE)

R2 = r2_score(y_test,Predicted_values_SVR)

MAPE = mean_absolute_percentage_error(y_test, Predicted_values_SVR)












