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



# 'df.values' must be emphasised to create a new dataframe with column names & to avoid missing values:
df = pd.DataFrame(irisData.values, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])
# print(df.head(12)) 
# print(data.loc[:, []])
# df.columns = ["SepalLength", "SepalWidth"]

# Writing the dataframe from to a text file:
df.to_csv('VariableSummaries.txt', sep='\t')

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

