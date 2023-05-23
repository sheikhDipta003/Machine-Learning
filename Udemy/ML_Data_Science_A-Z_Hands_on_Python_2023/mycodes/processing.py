# -*- coding: utf-8 -*-
"""
Created on Thu May 18 22:58:05 2023
@author: Dipta
"""

import pandas as pd
import numpy as np

# read data from csv file as data-frame
Data_Set1 = pd.read_csv('Data_Set.csv')

# identify the 2nd row as header, that is, skip the first two rows
Data_Set2 = pd.read_csv('Data_Set.csv', header = 2)

# df.rename(
#     columns={"oldcolname": "newcolname"},
#     index={"oldindex": "newindex"},
#     )
Data_Set3 = Data_Set2.rename(columns = {'Temperature':'Temp'})

# removes a set of labels from an axis
# df.drop(["labeltodrop"], axis=0)
Data_Set4 = Data_Set3.drop('No. Occupants',axis = 1)

# inplace [bool, default False] If True, do operation inplace and return None
Data_Set3.drop('No. Occupants',axis = 1, inplace = True)


Data_Set5 = Data_Set4.drop(2, axis = 0)

"""
Generate a new DataFrame or Series with the index reset.
Useful when the index needs to be treated as a column, or when the index is 
meaningless and needs to be reset to default.
drop [bool, default False] Just reset the index, without inserting it as a 
column in the new DataFrame.
"""
Data_Set6 = Data_Set5.reset_index(drop = True)

#  a quick statistic summary of your data
Data_Set6.describe()

Data_Set6['E_Heat'].min()

Data_Set6['E_Heat'].replace(-4,21, inplace = True)

# Covariance
# returns a matrix containing covariances between all possible pairs of variables
Data_Set6.cov()

# a visualization tool like matplotlib
import seaborn as sn
sn.heatmap(Data_Set6.corr()) # coefficient of correlation


"""""""""""
Missing Values

"""""""""""

# Print a concise summary of a DataFrame. This method prints information about 
# a DataFrame including the index dtype and columns, non-null values and memory
# usage.
Data_Set6.info()


# Values of the DataFrame are replaced with other values dynamically.
# This differs from updating with .loc or .iloc, which require you to specify 
# a location to update with some value.
Data_Set7 = Data_Set6.replace('!', np.NaN) # did not work
Data_Set7.loc[13,"E_Plug"] = '!' # had to explicitly set the nan value as '!'

Data_Set7.info()

"""""""""""
Accepts a function that takes a Series/DataFrame and returns a Series/
DataFrame. This method passes each column or row of your DataFrame 
one-at-a-time or the entire table at once, depending on the axis keyword 
argument. For columnwise use axis=0, rowwise use axis=1, and for the entire
table at once use axis=None.

"""""""""""
Data_Set7 = Data_Set7.apply(pd.to_numeric) 
# conversion to numeric dtype, e.g. from string

"""
Returns a DataFrame that is a mask of bool values for each element that 
indicates whether that element is an NA value.
NA values, such as None or numpy.NaN, gets mapped to True values. 
Everything else gets mapped to False values.
"""
Data_Set7.isnull()

"""
Remove missing values.
axis [{0 or 'index', 1 or 'columns'}, default 0] Determine if rows or 
columns which contain missing values are removed.
- 0, or 'index' : Drop rows which contain missing values.
- 1, or 'columns' : Drop columns which contain missing value.
"""
Data_Set7.dropna(axis=0, inplace=True)

"""
Fill NA/NaN values using the specified method.
method [{'backfill', 'bfill', 'pad', 'ffill', None}, default None] Method to 
use for filling holes in reindexed Series.
pad / ffill: propagate last valid observation to fill gap
backfill / bfill: use next valid observation to fill gap.
"""
Data_Set8 = Data_Set7.fillna(method = 'ffill')

# Replace missing values, encoded as np.nan, using the mean value of the 
# columns (axis 0) that contain the missing values.
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imp.fit(Data_Set7)
Data_Set9 = imp.transform(Data_Set7)


"""""""""""""""
Outlier Detection

"""""""""""""""
# box-and-whisker plot
Data_Set8.boxplot()

"""
quantile(q=0.5, axis=0, numeric_only=True, interpolation='linear')
q [float or array-like, default 0.5 (50% quantile)] Value between 
0 <= q <= 1, the quantile(s) to compute. 
axis [{0, 1, 'index', 'columns'}, default 0] Equals 0 or 'index' for 
row-wise, 1 or 'columns' for column-wise.
"""
Data_Set8['Time'].quantile(0.25)
Data_Set8['Time'].quantile(0.75)

"""

Q1 = 19.75
Q3 = 32.25
IQR = 32.25 - 19.75 = 12.5

Mild Outlier

Lower Bound = Q1 - 1.5*IQR = 19.75 - 1.5*12.5 = 1
Upper Bound = Q3 + 1.5*IQR = 32.25 + 1.5*12.5 = 51

Extreme Outlier

Upper Bound = Q3 + 3*IQR = 32.25 + 3*12.5 = 69.75

"""

Data_Set8['E_Plug'].replace(120,42,inplace=True)


"""""""""""""""
Concatenation

"""""""""""""""
New_Col = pd.read_csv('Data_New.csv')

# concat(objs, axis=0)
# objs [a sequence or mapping of Series or DataFrame objects]
# axis [{0/'index', 1/'columns'}, default 0] The axis to concatenate along
Data_Set10 = pd.concat([Data_Set8,New_Col], axis = 1)


"""""""""""""""
Dummy Variables

"""""""""""""""

Data_Set10.info()

"""
To convert a categorical variable into a “dummy” or “indicator” DataFrame, 
for example a column in a DataFrame (a Series) which has k distinct values, 
can derive a DataFrame containing k columns of True and False using 
get_dummies().
The goal is to make the Data understandable for machines.
"""
Data_Set11 = pd.get_dummies(Data_Set10['Price'])

Data_Set11.info()

"""""""""""""""
Normalization

Two subpackages for 3 normalization methods:
1) minmax_scale = (x - xmin) / (xmax - xmin) if scale is [0,1]
2) L1 norm - Manhattan Distance
3) L2 norm - Euclidean Distance
"""""""""""""""
from sklearn.preprocessing import minmax_scale, normalize

# First Method: Min Max Scale
Data_Set12 = minmax_scale(Data_Set8, feature_range=(0,1))

Data_Set13 = normalize(Data_Set8, norm = 'l2', axis = 0)
# axis = 0 for normalizing column / axis = 1 is for normalizing each row

"""
normalize() method returns just a 2D array, that is, there is no 'index' or any
column headers; to convert it into a pandas DataFrame, the constructor must
be used.
"""
Data_Set13 = pd.DataFrame(Data_Set13,columns = ['Time','E_Plug','E_Heat',
                                                'Price','Temp'])
 








