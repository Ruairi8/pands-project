# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:

import pandas as pd
data = pd.read_csv('iris.data')
# print(data.loc[::30])
print(data.describe())


df = pd.DataFrame(data, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])
# print(df)
# print(df.head(12)) 
print(data.loc[:, []])
df.columns = ["SepalLength", "SepalWidth"]

import numpy as np
import matplotlib.pyplot as plt
# x = 19
# y = np.random.rand(x)
# z = np.random.rand(x)
plt.scatter(df["SepalLength"], df["SepalWidth"], df["PetalLength"], df["PetalWidth"])
plt.yticks(np.arange(0, 6, 0.5))
plt.xticks(np.arange(0, 6, 0.5))
plt.show()