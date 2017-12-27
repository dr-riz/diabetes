#!/usr/bin/python

print("hw from python")

import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing as preproc
import numpy as np

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

array = np.array(dataset.values)
X = array[:,0:8]
Y = array[:,8]

#X_norm = preproc.normalize(X, norm='l2', axis=0, copy=True, return_norm=True)

scaler = preproc.MinMaxScaler()
print(scaler.fit(X))
preproc.MinMaxScaler(copy=True, feature_range=(0, 1))
X_norm = scaler.transform(X)
attr_norm = pandas.DataFrame(X_norm)
print(attr_norm.describe())


