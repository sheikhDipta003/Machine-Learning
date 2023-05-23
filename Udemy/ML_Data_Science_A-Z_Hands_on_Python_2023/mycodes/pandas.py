# -*- coding: utf-8 -*-
"""
Created on Tue May 16 19:48:39 2023
@author: Dipta
"""
import numpy as np
import pandas as pd

# Creating a Series (default int index in the absence of 2nd param)
s = pd.Series([1, 3, 5, np.nan, 6, 8], index=pd.date_range("20230101", periods=6))
print(s)

# Creating a DataFrame by passing a NumPy array, with a datetime index and 
# labeled columns
dates = pd.date_range("20230511", periods=6) # list of indices
print(dates)

df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list("ABCD"))
print(df)

# Creating a DataFrame by passing a dictionary of objects that can be converted 
# into a series-like structure.
# Number of indices can be mentioned anywhere, but if it is mentioned multiple
# times, all lengths must be exactly the same.
df2 = pd.DataFrame(
    {
    "A": 1.0,
    "B": pd.Timestamp("20230511"),
    "C": pd.Series(1, index=list(range(4)), dtype="float32"),
    "D": np.array(3, dtype="int32"),
    "E": pd.array(["test", "train", "test", "train"]),
    "F": "foo",
    }
)
print(df2, df2.dtypes, sep='\n')

# view the top and bottom rows of the frame
print(df, df.head(1), df.tail(2), sep='\n')
# Display the index, columns
print(df.index, df.columns, sep='\n')
# to_numpy() gives a NumPy representation of the underlying data
a = df.to_numpy()
b = df2.to_numpy()
print(a, a.dtype)
print(b, b.dtype)

#  a quick statistic summary of your data
print(df.describe())
# transposing the data table
print(df.T)

# Sorting by an axis
print(df)
print(df.sort_index(axis=1, ascending=False))

# Sorting by values
print(df)
print(df.sort_values(by="A"))

## Selection
# Selecting columns
print(df["A"], df[0:3], df["20230511":"20230513"], sep='\n')  # returns a Series

# getting a row using a label
print(df.loc["20230513"])

# Selecting on a multi-axis by label
print(df.loc[:, ["A", "B"]])

# getting a scalar value
print(df.loc["20230511", "A"])

# getting fast access to a scalar (equivalent to the prior method)
print(df.at["20230511", "A"])

## Selection by position
# via the position of the passed integers
print(df.iloc[3])
print(df.iloc[3:5, 2:4])
print(df.iloc[[1, 2, 4], [0, 2]])
# getting a value explicitly
print(df.iloc[1,1], df.iat[1,1], sep='\t')


## Setting values
s1 = pd.Series([1, 2, 3, 4, 5, 6], index=dates)
df["F"] = s1   # creating and setting a new column
df.loc[dates[0], "A"] = 0  # setting a scalar
df.iat[0,1] = 0
df.loc[:, "D"] = np.array([5] * len(df))   #setting an existing column
print(df)

## Writing/Reading Data
df.to_csv("foo.csv")
print(pd.read_csv("foo.csv"))







