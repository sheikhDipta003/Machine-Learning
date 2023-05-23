# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:44:29 2023
@author: Dipta
"""

import numpy as np

a = np.arange(15).reshape(3, 5)
print(a,a.shape,a.ndim,a.dtype,a.size,a.itemsize,type(a),sep='\n')

b = np.array([2,3,4,5])
print(b, b.dtype, sep='\n')

c = np.array([(1.2,-3.4,0.6),(2.1,2.9,1.5),(0.0,0.2,-5.6)])
print(c, c.dtype, sep='\n')

cmp = np.array([1,2,3,4], dtype="complex")
print(cmp)

a = np.zeros((3,5))
print(a)

b = np.ones((3,5), dtype=np.int32)
print(b)

c = np.arange(0, 10, 2.4, dtype=np.float64)
print(c)

d = np.linspace(0, 2, 11)
print(d)

a = np.linspace(0, 2*3.1416, 10)
print(a)

b = np.sin(a)
print(b)

A = np.array([[1,2],[-1,-2]])
B = np.array([[3,4],[-3,-4]])

print(A*B) #element-wise product
print(A @ B) #matrix-multiplication
print(A.dot(B)) #matrix-multiplication (second form)
A *= 3      #element-wise scalar multiplication
print(A)
B += A      #matrix addition
print(B)

rg = np.random.default_rng(1) # create instance of default random number generator
a = rg.random((2, 3))
print(a.sum(), a.min(), a.max(), sep='\t')

b = np.arange(10).reshape(2,5)      #axis0 : column, axis1 : row
print(b, b.sum(axis=0), b.cumsum(axis=1), b.min(axis=0), sep='\n')

# universal functions
# these functions operate elementwise on an array, producing an array as output
A = np.arange(5)
print(A, np.exp(A), np.sqrt(A), np.sort(A), np.min(A), np.max(A), sep='\n')

#slicing, indexing
print(A, A[2:5])
print(A, A[0:6:2])     #from index 0 to 5, every 2nd element
print(A, A[6:0:-1])     #from index 6 to 1, every element
print(A, A[::-1])     #reverse array

# one index per axis. These indices are given in a tuple separated by commas
def f(x, y):
    return 10*x + y
b = np.fromfunction(f, (5, 4), dtype=int)
print(b, b[2,3], b[0:5, 1], b[:,1], b[1:3,:],sep='\n')

#iterating over an array
for row in b:
    print(row)
for elem in b.flat: # flat attribute is an iterator over all the elements
    print(elem)

#changing the shape of an array
a = np.floor(10 * rg.random((3,4)))
print(a, a.shape, sep='\t')
print(a.ravel())        #flattens the array
print(a.reshape(6,2))   
print(a.T)      # returns transposed array
a.resize((2,6)) #modifies the array in-place
print(a)

# Copies and Views
a = np.array([[ 0, 1, 2, 3],
              [ 4, 5, 6, 7],
              [ 8, 9, 10, 11]])
b = a # no new object is created
print(b is a) # a and b are two names for the same ndarray object

#View/Shallow Copy
a = np.arange(12).reshape(3,4)
print(a, a.shape)
c = a.view()    # c is a different ndarray object, but contains a's data
print(c, c.shape, sep='\n')
print(c is a, c.base is a, sep='\t')
# c is a view of the data owned by a
c = c.reshape((2,6))
print(c, a, sep='\n')
c[0, 4] = 1234 # a's data changes
print(a)

# Slicing an array returns a view of it
s = a[:, 1:3]
s[:] = 10 # s[:] is a view of s. Note the difference between s = 10 and s[:] = 10
print(a)

# The copy method makes a complete copy of the array and its data.
d = a.copy() # a new array object with new data is created
print(d is a, d.base is a, sep='\t') # d doesn't share anything with a
d[0, 0] = 9999
print(a)

# Sometimes copy should be called after slicing if the original array is not 
# required anymore. For example, suppose a is a huge intermediate result and the
# final result b only contains a small fraction of a, a deep copy should be made 
# when constructing b with slicing
a = np.arange(int(1e8))
b = a[:100].copy()
del a # the memory of 'a' can be released







