# https://matplotlib.org/2.0.2/users/pyplot_tutorial.html
# Importing Python library matplotlib.pyplot to create some images or plots; it works somewhat like MATLAB. 
# Pyplot functions can makes many changes to a plot:
import matplotlib.pyplot as plt
import pandas as pd
Iris = pd.read_csv('iris.data')
# https://python.engineering/box-plot-and-histogram-exploration-on-iris-data/
# Setting the width and height in inches using '.rcParams' method:
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
# Setting the dataset columns equal to variables a, b, c and d:
a = Iris["1"]
b = Iris[" 2"]
c = Iris[" 3"]
d = Iris[" 4"]
# Matplotlibs in-built '.hist()' method is used to create a histogram, dividing the plot into 25 sections 
# known as 'bins':
plt.hist(a, bins=25, color='red')
# Creating a title on the plot, along with labels on the axes:
plt.title('Sepal Length Histogram')
plt.xlabel('Sepal_Length_cm')
plt.ylabel('Count')
# Saving the image to a png file:
plt.savefig('histogram1.png')
# '.close()' methods deletes the memory in pyplot of the last image created:
plt.close()
plt.hist(b, bins=25, color='black')
plt.title('Sepal Width Histogram')
plt.xlabel('Sepal_Width_cm')
plt.ylabel('Count')
plt.savefig('histogram2.png')
plt.close()
plt.hist(c, bins=25, color='yellow')
plt.title('Petal Length Histogram')
plt.xlabel('Petal_Lenght_cm')
plt.ylabel('Count')
plt.savefig('histogram3.png')
plt.close()
plt.hist(d, bins=25, color='indigo')
plt.title('Petal Width Histogram')
plt.xlabel('Petal_Width_cm')
plt.ylabel('Count')
plt.savefig('histogram4.png')
plt.close()