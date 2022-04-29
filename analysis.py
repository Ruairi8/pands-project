# Performing analysis on Fisher's Iris dataset.

# Pandas built-in module can be used to output dataframes:
import pandas as pd
import numpy as np
from py import std
from yaml import YAMLError
# Importing the dataset to data frame format using pandas .read_csv():
irisData = pd.read_csv('iris.data')

# 'describe()' method outputs the count, standard deviation, mean, min & max and percentiles of a dataset. Here,
# I am creating a few numpy variables to output mathematical statistics of the data set to a file:
Mean = np.mean(irisData["1"])
Percentile1 = np.percentile(irisData["1"], 25)
Percentile2 = np.percentile(irisData["1"], 75)
Std = np.std(irisData["1"])
Min = np.min(irisData["1"])
Max = np.max(irisData["1"])

Mean2 = np.mean(irisData[" 2"])
Percentile1a = np.percentile(irisData[" 2"], 25)
Percentile2a = np.percentile(irisData[" 2"], 75)
Std2 = np.std(irisData[" 2"])
Min2 = np.min(irisData[" 2"])
Max2 = np.max(irisData[" 2"])


# 'df.values' must be emphasised to create a new dataframe with column names & to avoid missing values:
df = pd.DataFrame(irisData.values, columns=["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"])
print(df["Species"].unique())
Size = df.groupby("Species").size()
print(df.groupby("Species").size())
# The '.head()' method allows the user to pass in an integer argument to show that number of rows from the top:
print(df.head(30)) 


# The “loc” functions use the index name of the row to display the particular row of the dataset. 
# The “iloc” functions use the index integer of the row, which gives complete information about the row.
# https://www.geeksforgeeks.org/python-basics-of-pandas-using-iris-dataset/ 
# Dividing the dataset based on species, in order to look at analyze them separately:
iris_setosa = df.loc[df["Species"]=="Iris-setosa"]
iris_virginica = df.loc[df["Species"]=="Iris-virginica"]
iris_versicolor = df.loc[df["Species"]=="Iris-versicolor"]

# Creating a few numpy variables to print some maths statistics:
a = np.mean(iris_setosa)
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
file.write("\n\nThe number of rows in each Iris: {}\n".format(str(Size)))
file.write("\nMean of Sepal Length for all 3 irises is: {}\n".format(str(Mean)))
file.write("\nThe 25th percential of Sepal Length for all 3 irises is: {}\n".format(Percentile1))
file.write("\nThe 75th percentile of Sepal Length for all 3 irises is: {}\n".format(Percentile2))
file.write("\nStandard deviation of Sepal Length for all 3 irises is : {}\n".format(str(Std)))
file.write("\nMinimum value of Sepal Length for all 3 irises is: {}\n".format(str(Min)))
file.write("\nMaximum value of Sepal Length for all 3 irises is: {}\n".format(str(Max)))
file.write("\nMean of Sepal width for all 3 irises is: {}\n".format(str(Mean2)))
file.write("\nThe 25th percential of Sepal Width for all 3 irises is: {}\n".format(Percentile1a))
file.write("\nThe 75th percentile of Sepal Width for all 3 irises is: {}\n".format(Percentile2a))
file.write("\nStandard deviation of Sepal Width for all 3 irises is : {}\n".format(str(Std2)))
file.write("\nMinimum value of Sepal Width for all 3 irises is: {}\n".format(str(Min2)))
file.write("\nMaximum value of Sepal Width for all 3 irises is: {}\n".format(str(Max2)))
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
# https://docs.python.org/3/library/index.html
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
# In Machine Learning techniques such as k-nearest-neighbours, data is split into training and testing groups:
from sklearn.model_selection import train_test_split , KFold


# '.iloc' gets rows & columns at index locations unlike '.loc' which identifies them by index labels.
x = df.iloc[:, :-1]
y = df.iloc[:, -1]
# split the data into training and testing sets, test_size of 0.2 means 20% of data points used for testing and so the 
# other 80% will be for training:
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, shuffle= True, random_state= 0)
# In numpy, '.asarray' method converts an input to an array. Unlike '.array', '.asarray' updates changes made to an array.
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x_test = np.asarray(x_test)
y_test = np.asarray(y_test)
# f-string literal to output the number of samples in the traning and testing sets using the '.shape()' method:
print(f'training set size: {x_train.shape[0]} samples \ntest set size: {x_test.shape[0]} samples')


# .fit() method calculates the mean and standard deviation:
scaler = Normalizer().fit(x_train) 
# Standardisation involves scaling the dataset so it will have a mean of 0 and a standard deviation of 1:
normalized_x_train = scaler.transform(x_train) 
normalized_x_test = scaler.transform(x_test) 
print('x train before Normalization')
print(x_train[0:5])
# We can see that the normalization process scalably lowered the sizes of the floating point values in the data set:
print('\nx train after Normalization')
print(normalized_x_train[0:5])


# kNN step 1 Euclidean distance. Used to find distance between points, its formula is the square root of 
# (x2-x1) squared + (y2-y1) squared:
def distance_ecu(x_train, x_test_point):
  """
  Input:
    - x_train: corresponding to the training data
    - x_test_point: corresponding to the test point

  Output:
    -distances: The distances between the test point and each point in the training data.

  """
  # Creating an empty array:
  distances = []  
  # Loops through the rows in x_train:
  for row in range(len(x_train)): 
      current_train_point = x_train[row] 
      # Initializing to zero:
      current_distance = 0 
      # Loops through the columns in defined variable 'current_train_point':
      for col in range(len(current_train_point)): 
          current_distance += (current_train_point[col] - x_test_point[col]) ** 2
      # When there is nothing else to add to 'current_distance', the code below is run. 'np.sqrt()' outputs
      # the square root of every element in the array. The output array is also the same shape:
      current_distance = np.sqrt(current_distance)
      # Appending the current_distances to the array 'distances' defined earlier:
      distances.append(current_distance) 

  # Store distances in a dataframe
  distances = pd.DataFrame(data=distances, columns=['dist'])
  return distances

# Step 2 - finding the nearest neighbours:
def nearest_neighbors(distance_point, K):
    """
    Input:
        -distance_point: the distances between the test point and each point in the training data.
        -K             : the number of neighbors

    Output:
        -df_nearest: the nearest K neighbors between the test point and the training data.

    """

# Sort values using the sort_values function. Since version 0.23.0: "Allow specifying index or column level names."
# Here it is choosing index with the label 'dist':
    df_nearest = distance_point.sort_values(by=['dist'], axis=0)

    # Take only the first K neighbors. [:K] outputs everything up to but not including element K in an array.
    # 'df_nearest' dataframe will have the nearest K neighbors between the full training dataset and the test point:
    df_nearest = df_nearest[:K]
    return df_nearest

# Step 3, classify the point based on a majority vote:
def voting(df_nearest, y_train):
    """
    Output: -y_pred: the prediction based on Majority Voting
    """
    # Use the Counter Object to get the labels with K nearest neighbors:
    from collections import Counter
    # Elements stored as dictionary keys and their counts stored as dictionary values:
    counter_vote = Counter(y_train[df_nearest.index])
    # 'most_common' outputs what is found the highest number of times. Slicing is used to get whats at index 0
    # in the dictionary, [0][0] returns the first value contained in the first key:
    y_pred = counter_vote.most_common()[0][0]
    # 'return' returns a result to the caller:
    return y_pred

# KNN Full Algorithm: Putting Everything Together
def KNN_from_scratch(x_train, y_train, x_test, K):

    """
    Input:
    -x_train: the full training dataset
    -y_train: the labels of the training dataset
    -x_test: the full test dataset
    -K: the number of neighbors

    Output:
    -y_pred: the prediction for the whole test set based on Majority Voting.

    """

    y_pred = []

    # Loop over all the test set and perform the three steps kNN steps
    for x_test_point in x_test:
      # Calculating euclidean distance between the training set point, and the testing set point:
      distance_point = distance_ecu(x_train, x_test_point)  
      # Calculating the number of K points:
      df_nearest_point = nearest_neighbors(distance_point, K)
      # Getting the variable with the most data points:
      y_pred_point = voting(df_nearest_point, y_train) 
      y_pred.append(y_pred_point)

    return y_pred  

# Testing the KNN Algorithm on the test dataset. Setting number of neighbours to 3:
K = 3
y_pred_scratch = KNN_from_scratch(normalized_x_train, y_train, normalized_x_test, K)
print(y_pred_scratch)

from sklearn.neighbors import KNeighborsClassifier
# Creating a kNN object with K = 3 as an argument:
knn = KNeighborsClassifier(K)
# Fitting knn on the training set:
knn.fit(normalized_x_train, y_train)
# This does predictions on the normalized testing set:
y_pred_sklearn = knn.predict(normalized_x_test)
print(y_pred_sklearn)
# Numpy method 'array_equal' returns a boolean; true is 2 arrays have the same number of dimensions and element:
print(np.array_equal(y_pred_sklearn, y_pred_scratch))

from sklearn.metrics import accuracy_score
# Comparing the test set values with the predicted values to get the accuracy:
print(f'The accuracy of our implementation is {accuracy_score(y_test, y_pred_scratch)}')
print(f'The accuracy of sklearn implementation is {accuracy_score(y_test, y_pred_sklearn)}')

# Performing Hyperparameter Tuning using K-fold Cross Validation, choosing 4 splits. The k in k Nearest 
# Neighbours is a machine learning hyperparameter. A hyperparameter in a parameter that is already set before 
# starting training on a data set - they can't be learned directly from the training process. 
# Hyperparameter tuning involves tuning parameters as tuples so that a machine learning algorithm such as kNN
# will perform well. https://medium.com/almabetter/cross-validation-and-hyperparameter-tuning-91626c757428:
n_splits = 4 
# 'kFold()' method divides the samples into different groups called 'folds':
kf = KFold(n_splits = n_splits) 

# Keeping track of the accuracy for each K value:
accuracy_k = [] 
# Search for the best value of K:
k_values = list(range(1,140,1)) 

# A 'for loop' to iterate through the K values to standardize the data:
for k in k_values: 
  accuracy_fold = 0
  # The elements here can be named anything but it helps readablilty to give a descriptive name:
  for normalized_x_train_fold_index, normalized_x_valid_fold_index in kf.split(normalized_x_train): 
      normalized_x_train_fold = normalized_x_train[normalized_x_train_fold_index] 
      y_train_fold = y_train[normalized_x_train_fold_index]

      normalized_x_test_fold = normalized_x_train[normalized_x_valid_fold_index]
      y_valid_fold = y_train[normalized_x_valid_fold_index]
      # This function was defined above: 
      y_pred_fold = KNN_from_scratch(normalized_x_train_fold, y_train_fold, normalized_x_test_fold, k)

      accuracy_fold += accuracy_score(y_pred_fold, y_valid_fold) 
  # Diving by the number of splits to get an accuracy and then appending it to the empty array 'accuracy_k':
  accuracy_fold = accuracy_fold / n_splits 
  accuracy_k.append(accuracy_fold)
# Creates a tuple with accuracy corresponding to k value:
print(f'The accuracy for each K value was {list ( zip (accuracy_k, k_values))}') 
# 'np.argmax()' outputs the maximum element of an array:
print(f'Best accuracy was {np.max(accuracy_k)}, which corresponds to a value of K= {k_values[np.argmax(accuracy_k)]}')


# Experimenting with code from https://sebastianraschka.com/Articles/2014_python_lda.html#lda-in-5-steps
X = df[["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]].values
Y = df[['Species']].values
# Creating an instance of labelencoder (part of the sklearn package).
enc = LabelEncoder()
# 'LabelEncoder.fit' maps column strings to numerical values
label_encoder = enc.fit(Y)
# Assigns numerical values in the variable 'Y':
Y = label_encoder.transform(YAMLError) + 1

mean_vectors = []
for cl in range(1,4):
    mean_vectors.append(np.mean(X[Y==cl], axis=0))
    print('Mean Vector class %s: %s\n' %(cl, mean_vectors[cl-1]))
