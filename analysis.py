# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:
from matplotlib.colors import Colormap
import pandas as pd
import numpy as np
irisData = pd.read_csv('iris.data')
# This outputs only data only at certain locations, i.e. at the junction of a given step size:
# print(data.loc[::30])

# 'describe()' method outputs the count, standard deviation, mean, min & max and percentiles of a dataset:
print(irisData.describe())
# Creating a few numpy variables to print some maths statistics:
x = np.mean(irisData["1"])
print("The mean of the sepal length is: {}".format(x))
y = np.std(irisData["1"])
print("The standard deviation of sepal length is: {}".format(y))
z = np.min(irisData["1"])
print("The minimum value for sepal length is: {}".format(z))
x1 = np.max(irisData["1"])
print("The maximum value for sepal length is: {}".format(x1))

# https://stackoverflow.com/questions/64685561/how-to-write-print-statements-to-csv-file-in-python
import csv 
with open("VariableSummaries.txt", "w") as csv:
    print("The mean of the sepal length is: {}".format(x), file=csv)
    print("The standard deviation of sepal length is: {}".format(y), file=csv)
    print("The minimum value of sepal length is: {}".format(z), file=csv)
    print("The maximum value for sepal length is: {}".format(x1), file=csv)

# x.to_csv('VariableSummaries.txt', sep='\t')
# y.to_csv('VariableSummaries.txt', sep='\t')
# z.to_csv('VariableSummaries.txt', sep='\t')
# x1.to_csv('VariableSummaries.txt', sep='\t')


# 'df.values' must be emphasised to create a new dataframe with column names & to avoid missing values:
df = pd.DataFrame(irisData.values, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])

# print(df.head(12)) 
# print(data.loc[:, []])
# df.columns = ["SepalLength", "SepalWidth"]

# Writing the dataframe from to a text file:
# df.to_csv('VariableSummaries.txt', sep='\t')

import seaborn as se
import matplotlib.pyplot as plt
# plt.hist(df1["SepalLength"], bins=8, color='green', alpha=0.5)
# plt.show()
# x = 19
# y = np.random.rand(x)
# z = np.random.rand(x)
#plt.scatter(df1["SepalLength"], df1["SepalWidth"], Colormap["Species"])
#plt.yticks(np.arange(1, 5, 1))
#plt.xticks(np.arange(3, 10, 1))
# plt.show()
# se.scatterplot(x="PetalLength" "SepalLenght", y="PetalWidth" "SepalWidth", hue="Species", data=df)
# plt.show()

# The “loc” functions use the index name of the row to display the particular row of the dataset. 
# The “iloc” functions use the index integer of the row, which gives complete information about the row.
# https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/ 
# Dividing the dataset based on species, in order to look at analyze them separately:
iris_setosa=df.loc[df["Species"]=="Iris-setosa"]
iris_virginica=df.loc[df["Species"]=="Iris-virginica"]
iris_versicolor=df.loc[df["Species"]=="Iris-versicolor"]

# Creating a few numpy variables to print some maths statistics:
x2 = np.mean(iris_setosa)
print("THE mean of the sepal length is: {}".format(x2))
y2 = np.std(iris_virginica)
print("THE standard deviation of sepal length is: {}".format(y2))
z2 = np.min(iris_setosa)
print("THE minimum value for sepal length is: {}".format(z2))
w = np.max(iris_setosa)
print("THE maximum value for sepal length is: {}".format(w))

total = x2 + y2 + z2 + w
total.to_csv("VariableSummaries.txt", sep="\t")

# https://eldoyle.github.io/PythonIntro/08-ReadingandWritingTextFiles/
filename = "VariableSummaries.txt"
file = open(filename, 'w')

file.write("Sepal length mean is: {}".format(x2))
file.write("Sepal length standard deviation is: {}".format(y2))
file.write("Sepal length minimum value is: {}".format(z2))
file.write("Sepal length maximum value is {}".format(w))

file.close() 

