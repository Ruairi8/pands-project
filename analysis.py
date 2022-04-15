# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:
import pandas as pd
import numpy as np
from py import std
from yaml import YAMLError
# Importing the dataset to data frame format using pandas .read_csv():
irisData = pd.read_csv('iris.data')

# 'describe()' method outputs the count, standard deviation, mean, min & max and percentiles of a dataset:
print(irisData.describe())
# Creating a few numpy variables to print some maths statistics:
Mean = np.mean(irisData["1"])
print("The Mean Of The Sepal Length is: {}".format(Mean))
Std = np.std(irisData["1"])
print("The Standard Deviation Of Sepal Length is: {}".format(Std))
Min = np.min(irisData["1"])
print("The Minimum Value For Sepal length is: {}".format(Min))
Max = np.max(irisData["1"])
print("The Maximum Value For Sepal Length Is: {}".format(Max))

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
a = np.mean(iris_setosa)
print("THE mean of the sepal length is: {}".format(a))
b = np.std(iris_setosa)
c = np.min(iris_setosa)
d = np.max(iris_setosa)
a1 = np.mean(iris_virginica)
b1 = np.std(iris_virginica)
c1 = np.min(iris_virginica)
d1 = np.max(iris_virginica)
a2 = np.mean(iris_versicolor)
b2 = np.std(iris_versicolor)
c2 = np.min(iris_versicolor)
d2 = np.max(iris_versicolor)



# https://eldoyle.github.io/PythonIntro/08-ReadingandWritingTextFiles/
filename = "VariableSummaries.txt"
file = open(filename, 'w')
# .write() only takes a string as an argument. https://stackoverflow.com/questions/41454921/typeerror-write-argument-must-be-str-not-list
file.write("A summary of the variables - Iris Setosa, Iris Virginica & Iris Versicolor\n")
# https://www.w3resource.com/pandas/dataframe/dataframe-to_string.php
# .to_string() 
file.write("\nMean of Sepal Length for all 3 irises is: {}\n".format(str(Mean)))
file.write("\nStandard deviation of Sepal length for all 3 irises is : {}\n".format(str(Std)))
file.write("\nMinimum value of Sepal Length for all 3 irises is: {}\n".format(str(Min)))
file.write("\nMaximum value of Sepal Length for all 3 irises is: {}\n".format(str(Max)))
file.write("\nIris setosa mathematical stats for lengths and widths:")
file.write("\nMean values:\n")
file.write(str(a.to_string()))
file.write("\n\nStandard deviation values:\n")
file.write(str(b.to_string()))
file.write("\n\nMinimum values:\n")
file.write(str(c.to_string()))
file.write("\n\nMaximum values:\n")
file.write(str(d.to_string()))
file.write("\n\nIris virginica mathematical stats for lengths and widths:\n")
file.write("Mean values:\n")
file.write(str(a1.to_string()))
file.write("\n\nStandard deviation values:\n")
file.write(str(b1.to_string()))
file.write("\n\nMinimum values:\n")
file.write(str(c1.to_string()))
file.write("\n\nMaximum values:\n\n")
file.write(str(d1.to_string()))
file.write("\n\nIris versicolor mathematical stats for lengths and widths:\n")
file.write("Mean values:\n")
file.write(str(a2.to_string()))
file.write("\n\nStandard deviation values:\n")
file.write(str(b2.to_string()))
file.write("\n\nMinimum values:\n")
file.write(str(c2.to_string()))
file.write("\n\nMaximum values:\n")
file.write(str(d2.to_string()))



# https://sebastianraschka.com/Articles/2014_python_lda.html#lda-in-5-steps
# 'scikit-learn' and 'sklearn' refer to the same Python package. It is a Machine Learning package, 
# that allows one to utilize classification, regression, clustering, dimensionality reduction etc. 
# 'scikit-learn' is imported as 'sklearn', to avoid throwing up an error due to the hyphen. https://towardsdatascience.com/scikit-learn-vs-sklearn-6944b9dc1736
# https://deepnote.com/@ndungu/Implementing-KNN-Algorithm-on-the-Iris-Dataset-58Fkk1AMQki-VJOJ3mA_Fg
from sklearn.preprocessing import LabelEncoder
# In Machine Learning techniques such as k-nearest-neighbours, data is split into training and testing groups:
from sklearn.model_selection import train_test_split , KFold

# '.iloc' gets rows & columns at index locations unlike '.loc' which identifies them by index labels.
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
# split the data into train and test sets:
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, shuffle= True, random_state= 0)
# In numpy, '.asarray' method converts an input to an array. Unlike '.array', '.asarray' updates changes made to an array.
x_train= np.asarray(x_train)
y_train= np.asarray(y_train)

x_test= np.asarray(x_test)
y_test= np.asarray(y_test)
print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')

X = df[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values
Y = df['Species'].values


# Experimenting with code from https://sebastianraschka.com/Articles/2014_python_lda.html#lda-in-5-steps
# Creating an instance of labelencoder (part of the sklearn package).
enc = LabelEncoder()
# LabelEncoder.fit maps column strings to numerical values
label_encoder = enc.fit(Y)
# Assigns numerical values in the variable 'Y':
Y = label_encoder.transform(YAMLError) + 1

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[Y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
