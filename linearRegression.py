from sklearn.linear_model import LinearRegression
import pandas as pd
# Linear regression on all three categories of Iris.
# https://towardsdatascience.com/linear-regressions-with-scikitlearn-a5d54efe898f
# To solve typeError: fit() missing 1 required positional argument 'y'; LinearRegression method required to 
# have empty parenthesis in front of it:
# https://stackoverflow.com/questions/35996970/typeerror-fit-missing-1-required-positional-argument-y

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = pd.read_csv("iris.data")
x_var = iris["1"]
# The '.reshape()' method creates a numpy array as well as an instance of the original array:
x_var2 = x_var.values.reshape(-1, 1)
y_var = iris[" 2"]
y_var2 = y_var.values.reshape(-1, 1)
# 'train_test_split' method to randomly assign data points to training and testing groups; in this case x_training
# and x_testing is from the sepal length column and the y training and testing groups are taken from the sepal
# width column:
x_training, x_testing, y_training, y_testing = train_test_split(x_var2, y_var2, random_state=0)

# Creating a linear regression object setting it equal to a variable, z:
z = LinearRegression()
# '.fit()' will generate parameters for the linear regression object using the training data points:
linearReg = z.fit(x_training, y_training)
# The coefficient are values that multiply the variables. Coefficents show the relationship between dependent
# and independent variables. P-values for coefficients can tell if the relationships observed in training and
# testing sets are also true for the entire dataset:
print("Coefficient: " + str(linearReg.coef_))
# The intercept is where the regression line hits the y axis:
print("Intercept: " + str(linearReg.intercept_))
# R-squared is the percentage of a dependent variable variation.
print("Training R-squared value, Sepal: " + str(linearReg.score(x_training, y_training)))
print("Testing R-squared value, Sepal: " + str(linearReg.score(x_testing, y_testing)))


x_var = iris[" 3"]
x_var2 = x_var.values.reshape(-1, 1)
y_var = iris[" 4"]
y_var2 = y_var.values.reshape(-1, 1)
x_training, x_testing, y_training, y_testing = train_test_split(x_var2, y_var2, random_state=0)
z = LinearRegression()
linearReg = z.fit(x_training, y_training)
print("Coefficient: " + str(linearReg.coef_))
print("Intercept: " + str(linearReg.intercept_))
print("Training R-squared value, Petal: " + str(linearReg.score(x_training, y_training)))
print("Testing R-squared value, Petal: " + str(linearReg.score(x_testing, y_testing)))