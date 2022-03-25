# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:

import pandas as pd
data = pd.read_csv('iris.data')
print(data.loc[::30])

df = pd.DataFrame(data, columns=[1, 2, 3, 4, 5])
print(df)
