# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:

import pandas as pd
data = pd.read_csv('iris.data')
# print(data.loc[::30])
# print(data.describe())


df = pd.DataFrame(data)
# df = pd.DataFrame(data, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])
# print(df)
# 'df.values' must be emphasised to create a new dataframe with column names & to avoid missing values.
df1 = pd.DataFrame(df.values, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])
# print(df1)
# print(df.head(12)) 
# print(data.loc[:, []])
# df.columns = ["SepalLength", "SepalWidth"]

import seaborn as se
import numpy as np
import matplotlib.pyplot as plt
# plt.hist(df1["SepalLength"], bins=8, color='green', alpha=0.5)
# plt.show()
# x = 19
# y = np.random.rand(x)
# z = np.random.rand(x)
plt.scatter(df1["SepalLength"], df1["PetalLength"], color="orange")
plt.yticks(np.arange(0, 5, 1))
plt.xticks(np.arange(3, 11, 1))
plt.show()
# se.scatterplot(x="PetalLength" "SepalLenght", y="PetalWidth" "SepalWidth", hue="Species", data=df)
# plt.show()