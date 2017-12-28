#!/usr/bin/python

print("Reproducing case study of Shvartser posted at Dr. Brownlee's machinelearningmastery.com")

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing as preproc
import numpy

datafile="./diabetes.data"
headers=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataset=pandas.read_csv(datafile, names=headers)

print(" = 3. Summarize the Dataset = ")
# shape
print(" == 3.1 Dimensions of Dataset, shape of data == ")
print(dataset.shape)


# head
print(" == 3.2 Peek at the Data, head -- first 10 items == ")
print(dataset.head(10))

# data types
print(" == 3.a data types for each attributes == ")
print(dataset.dtypes)

# descriptions
print(" == 3.3 Statistical Summary == ")
print(dataset.describe())

# class distribution
print(" == 3.4 class distribution  ==")
print(dataset.groupby('class').size())


print(" = 4. Data Visualization = ")
# box and whisker plots
print(" == 4.1 Univariate Plots: box and whisker plots. why? to determine outliers? = ")
#dataset.plot(kind='box', subplots=True, layout=(2,4), sharex=False, sharey=False)
#plt.show()

# histograms
print(" == 4.1 Univariate Plots: histograms. why? to determine if the distribution is normal-like? == ")
#dataset.hist()
#plt.show()

# scatter plot matrix
print("== 4.2 Multivariate Plots: Multivariate Plots:scatter plot matrix. why? to spot structured relationships between input variables ==")
#scatter_matrix(dataset)
#plt.show()

numpy.set_printoptions(precision=3)
array = numpy.array(dataset.values)


#datasets
## diabetes.arff -- unchanged, original
## standardized.arff
## normalized.arff
## discretize.arff
## missing.arff
## remove_missing.arff:
## replaced_missing.arff:
## square.arff

print("diabetes_attr: unchanged, original attributes")
diabetes_attr = array[:,0:8]
label = array[:,8] #unchanged across preprocessing?
diabetes_df = pandas.DataFrame(diabetes_attr)

print("normalized_attr: range of 0 to 1")
scaler = preproc.MinMaxScaler().fit(diabetes_attr)
normalized_attr = scaler.transform(diabetes_attr)
normalized_df = pandas.DataFrame(normalized_attr)
print(normalized_df.describe())

print("standardized_attr: mean of 0 and stdev of 1")
scaler = preproc.StandardScaler().fit(diabetes_attr)
standardized_attr = scaler.transform(diabetes_attr)
standardized_df = pandas.DataFrame(standardized_attr)
print(standardized_df.describe())




