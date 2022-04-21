#<h2><em>Analysis of the Iris data set</em></h2>

Author: Ruairi McCool

In this project I will output summaries of the variables involved to a text file and do meaningful analyses using python on the dataset using some of it's built-in libraries. Visual distributions can be created using matplotlib and pandas libraries, for example. I could also divide the data into training and testing groups, as is done in "k nearest neighbours" in machine learning. I will also provide a written summary of the output and the conclusions I have made.
What can be gained by performing analysis and outputting mathematical statistics of the variables in this 
dataset? The answer is that it can be determined which variables and combinations of them can be used to 
distinguish each of the three varieties of iris from one another. It may be necessary to find out how reliable a method of distinguishing one variety of iris from another is, because not every method is one 
hundred percent accurate and there may be a margin of error.

<h4><em>Summary of the data set:</em></h4> 
The Iris flower data set was introduced by Robert Fisher in 1936 in his paper <i>"the use of multiple measurements in taxonomic problems"</i> as an example of linear discriminant analysis. The data set contains fifty samples from each of three species of Iris (Setosa, Virginica and Versicolor). The columns contain the variables sepal length, sepal width, petal length, petal width measured in centimeters and species name. The rows contain the samples - 150 in total. Fisher was able to develop a linear discriminant model to distinguish the species from each other using this information.
# https://www.knowledgehut.com/blog/data-science/linear-discriminant-analysis-for-machine-learning 
Linear Discriminant Analysis (LDA) can be used to reduce the dimensions or variables of a dataset while retaining most of the data. It is also used before preforming Machine Learning techniques to process a dataset and to identify patterns and from that to classify them. Fisher's Discriminant Analysis was a two-class method, later C.R.Mao developed a multi-class method, which is also considered part of LDA. LDA's real-life applications include facial recognition and predictive analysis in marketing. 
Dimensionality Reduction itself involved removing redunant and dependent features by moving the dataset into a lower-dimensional space. A multi-dimensional dataset can be plot in only two or three dimensions using Dimensionality reduction. Logistic Regression is another classification technique although it is not as useful as LDA. Firstly, LR is only worth using for a two class problem, it lacks stability when the classes are well separated or if there is little data to make up certain parameters. In other words, if it becomes unstable it no longer works as intended, and the results cannot be trusted.

'k Nearest Neighbours' (kNN) is a algorithm used to classify data, it can be used on the Iris dataset because the categories are known. Unknown data points are classified by their distances from other data points whose category types are known. If data points are randomly distributed in a dataset, kNN will be of no value and it will be impossible to accurately classify any points in a dataset. The K value in "K Nearest Neighbours" determines how many nearest data points need to be used for classification. Different values of K can be selected by the analyzer to find a suitable value that doesn't create many errors while also keeping the algorithms ability to make accurate predictions. In cases where K > 1, the category is picked which appeared most frequently in the results. If the K value is even and two or more categories showed up the same number of times, it can be decided either not to assign the data point a category or to randomly choose a category. Data must be clustered in advance, to create what is called the 'training data' and the 'testing data'. Low K values are influenced by the effect of outliers which can skew results so it is important to minimize the effect outliers will have on an output.

Summary: In k Nearest Neighbours the accuracy output was 0.96 or 96%, and for different values of K that accuracy only slightly lowered as the K values got higher (up to 29). Despite Versicolor and Virginica subgroups being quite closer together than the Setosa subgroup, this did not affect the ability of the algorithm to give a correct classification for a sample datapoint 96 times out of 100.