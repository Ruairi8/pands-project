# https://machinelearninghd.com/iris-dataset-uci-machine-learning-repository-project/
# Comparing petal dimensions in a scatter plot.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.data")
fig = iris[iris[' 5']=='Iris-setosa'].plot(kind='scatter',x=' 3',y=' 4',color='orange', label='Setosa')
iris[iris[' 5']=='Iris-versicolor'].plot(kind='scatter',x=' 3',y=' 4',color='blue', label='versicolor',ax=fig)
iris[iris[' 5']=='Iris-virginica'].plot(kind='scatter',x=' 3',y=' 4',color='green', label='virginica', ax=fig)

fig.set_xlabel("Pepal Length")
fig.set_ylabel("Pepal Width")
fig.set_title("Pepal Length VS Width")
fig = plt.gcf()
fig.set_size_inches(16,8)

x = iris[' 3']
y = iris[' 4']
# Using numpy methods to put a trendline on the scatter plot.
# https://numpy.org/doc/stable/reference/generated/numpy.polyfit.html A least squares polynomial fit. 'deg' is 
# a required parameter - the degree of the fitting polynomial:
z = np.polyfit(x, y, deg=1)
# https://numpy.org/doc/stable/reference/generated/numpy.poly1d.html 1-dimensional polynomial. It had a required 
# parameter, an array, outputs coefficients in order of decreasing power:
p = np.poly1d(z)

plt.plot(x,p(x),"--")
plt.savefig('scatterPetal')
plt.show()