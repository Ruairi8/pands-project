# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:
import pandas as pd
import numpy as np
# Importing the dataset to data frame format using pandas .read_csv():
irisData = pd.read_csv('iris.data')

# 'describe()' method outputs the count, standard deviation, mean, min & max and percentiles of a dataset:
print(irisData.describe())
# Creating a few numpy variables to print some maths statistics:
x = np.mean(irisData["1"])
print("The Mean Of The Sepal Length is: {}".format(x))
y = np.std(irisData["1"])
print("The Standard Deviation Of Sepal Length is: {}".format(y))
z = np.min(irisData["1"])
print("The Minimum Value For Sepal length is: {}".format(z))
x1 = np.max(irisData["1"])
print("The Maximum Value For Sepal Length Is: {}".format(x1))

# 'df.values' must be emphasised to create a new dataframe with column names & to avoid missing values:
df = pd.DataFrame(irisData.values, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])
print(df.head(12)) 


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
y2 = np.std(iris_setosa)
print("THE standard deviation of sepal length is: {}".format(y2))
z2 = np.min(iris_setosa)
print("THE minimum value for sepal length is: {}".format(z2))
w = np.max(iris_setosa)
print("THE maximum value for sepal length is: {}".format(w))


# https://eldoyle.github.io/PythonIntro/08-ReadingandWritingTextFiles/
filename = "VariableSummaries.txt"
file = open(filename, 'w')
# .write() only takes a string as an argument. https://stackoverflow.com/questions/41454921/typeerror-write-argument-must-be-str-not-list
file.write("Mean, Standard Deviation, Minimum & Maximim for Iris-Setosa\n")
# https://www.w3resource.com/pandas/dataframe/dataframe-to_string.php
# .to_string() 
file.write("Mean of Sepal Length for all 3 irises: {}".format(str(x)))
file.write(str(x2.to_string()))

file.write(str(y2.to_string()))

file.write(str(z2.to_string()))

file.write(str(w.to_string()))


