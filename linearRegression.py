from sklearn.linear_model import LinearRegression
import pandas as pd
# Linear regression on all three categories of Iris.
# https://towardsdatascience.com/linear-regressions-with-scikitlearn-a5d54efe898f
# To solve typeError: fit() missing 1 required positional argument 'y'; LinearRegression required to be in
# braces:
# https://stackoverflow.com/questions/35996970/typeerror-fit-missing-1-required-positional-argument-y

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.data")
x_var = iris["1"]
x_var2 = x_var.values.reshape(-1, 1)
y_var = iris[" 2"]
y_var2 = y_var.values.reshape(-1, 1)
x_training, x_testing, y_training, y_testing = train_test_split(x_var2, y_var2, random_state=0)
