# Outputting a scatter plot of 2 variables Setosa and Versicolor:
# https://machinelearninghd.com/iris-dataset-uci-machine-learning-repository-project/
import pandas as pd
import matplotlib.pyplot as plt
import numpy
iris = pd.read_csv("iris.data")
# '.plot' method allows u to define the type of graph u want using 'kind='. x and y parameters define what columns
# you want to plot from a dataset. Double equal sign asks a question is something something rather than setting
# it equal to something:
fig = iris[iris[' 5']=='Iris-setosa'].plot(kind='scatter',x="1",y=" 2",color='red', label='Setosa')
iris[iris[' 5']=='Iris-versicolor'].plot(kind='scatter',x="1",y=" 2",color='blue', label='versicolor',ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig = plt.gcf()
# 'set_size-inches' allows you to define the width and height of your plot; in format (width, height):
fig.set_size_inches(16,8)
# This creates a file with the scatter plot, the user can define their own name for the file in quotes:
plt.savefig('scatterPlot1')
plt.show()

fig = iris[iris[' 5']=='Iris-versicolor'].plot(kind='scatter',x="1",y=" 2",color='blue', label='Versicolor')
iris[iris[' 5']=='Iris-virginica'].plot(kind='scatter',x="1",y=" 2",color='green', label='Virginica',ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig = plt.gcf()
fig.set_size_inches(16,8)

plt.savefig('scatterPlot2')
plt.show()

fig = iris[iris[' 5']=='Iris-setosa'].plot(kind='scatter',x="1",y=" 2",color='red', label='Setosa')
iris[iris[' 5']=='Iris-virginica'].plot(kind='scatter',x="1",y=" 2",color='green', label='Virginica',ax=fig)
fig.set_xlabel("Sepal Length")
fig.set_ylabel("Sepal Width")
fig.set_title("Sepal Length VS Width")
fig = plt.gcf()
fig.set_size_inches(16,8)

plt.savefig('scatterPlot3')
plt.show()