# -*- coding: utf-8 -*-
"""
Created on Tue May 16 23:55:23 2023
@author: Dipta
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.plot([1,2,3,4], [1,4,9,16], 'ro')  # red circles as plot-points
plt.ylabel('some numbers')
plt.axis([0,5,0,20])  # [xmin, xmax, ymin, ymax] ; specifies the viewport
plt.show()

# line attributes
plt.plot([1,2,3,4], [1,4,9,16], 'r', linewidth=1.5)
plt.ylabel('some numbers')
plt.axis([0,5,0,20])  # [xmin, xmax, ymin, ymax] ; specifies the viewport
plt.show()

# evenly sampled time at 200ms intervals
t = np.arange(0., 5., 0.2)
# red dashes, blue squares and green triangles
plt.plot(t, t, 'r--', t, t**2, 'bs', t, t**3, 'g^')
plt.show()

# plot returns a list of Line2D objects
def f(x):
    return 4 * x * x
x = np.arange(0, 2.6, 0.2)
line, = plt.plot(x, f(x), 'g-')
line.set_antialiased(False) # turn off antialising
plt.show()

# use setp() for setting multiple properties of a list of lines
lines = plt.plot(x, x**3, 'r-', x, x+4, 'b--')
plt.setp(lines, label="st-line and cubic function", linewidth=0.6, marker='+')
plt.setp(lines[1], xdata=x, ydata=4*x**2)
plt.show()


# create two subplots in one figure
def f(t):
    return np.exp(-t) * np.cos(2*np.pi*t)
t1 = np.arange(0.0, 5.0, 0.1)
t2 = np.arange(0.0, 5.0, 0.02)

plt.figure(1)  # create the first figure
plt.subplot(2,1,1)  # first subplot of first figure
# numrows, numcols, fignum where fignum ranges from 1 to numrows*numcols.
plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
plt.subplot(2,1,2)      #second subplot of first fig
plt.plot(t2, np.cos(2*np.pi*t2), 'r--')

plt.figure(2) # a second figure, creates a subplot(111) by default
plt.plot(t2, np.sin(2*np.pi*t2)/(2*np.pi*t2), 'r', linewidth=0.7) 
plt.title('sinc function')

plt.figure(1) # make figure 1 current; subplot(212) still current
plt.subplot(2,1,1) # make 'subplot(211) in figure1' current
plt.title('exp and trigon functions') # subplot 211 title
plt.show()

## clear/get current fig -> clf(), gcf()
## clear/get current axes -> cla(), gca()

## Working with text
# The text() command can be used to add text in an arbitrary location, and the 
# xlabel(), ylabel() and title() are used to add text in the indicated 
# locations 

# Fixing random state for reproducibility
np.random.seed(19680801)
mu, sigma = 100, 15
x = mu + sigma * np.random.randn(10000)

# the histogram of the data
# bins: int or sequence or str
#   If bins is an integer, it defines the number of equal-width bins in the range.
#   If bins is a sequence, it defines the bin edges. if bins is: [1, 2, 3, 4]
#   then the first bin is [1, 2) and the second  [2, 3). The last bin, 
#   however, is [3, 4].
n, bins, patches = plt.hist(x, 50, density=True, color='g')
plt.xlabel('Smarts', fontsize=14, color='red')
plt.ylabel('Probability')
plt.title('Histogram of IQ')
plt.text(60, .025, r'$\mu=100,\ \sigma=15$')    #(x,y,text='')
plt.axis([40, 160, 0, 0.03])
plt.grid(True)
plt.show()





















