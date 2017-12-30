#!/usr/bin/python

print("Reproducing case study of Shvartser posted at Dr. Brownlee's machinelearningmastery.com")

# preproc imports
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing as preproc
import numpy

# algo eval imports
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.svm import SVC

# fine tuning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# significance tests
import scipy.stats as stats
import math

# build and save model using Pickle
from random import *
import pickle

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
#array = dataset.values

#datasets
## x diabetes.arff -- unchanged, original
## x normalized.arff
## x standardized.arff
## discretize.arff
## x missing.arff (ambigious)
## x remove_missing.arff:
## x replaced_missing.arff:
## square.arff
## oversampling
## undersampling

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
#scaler = preproc.StandardScaler().fit(diabetes_attr)
#standardized_attr = scaler.transform(diabetes_attr)
standardized_attr = preproc.scale(diabetes_attr)
standardized_df = pandas.DataFrame(standardized_attr)
print(standardized_df.describe())

## missing.arff
print((dataset[['plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']] == 0).sum())

# make a copy of original data set
dataset_cp = dataset.copy(deep=True)

dataset_cp[['plas', 'pres', 'skin', 'test', 'mass']] = dataset_cp[['plas', 'pres', 'skin', 'test', 'mass']].replace(0, numpy.NaN)

# print the first 10 rows of data
print(dataset_cp.head(10))

# count the number of NaN values in each column
print(dataset_cp.isnull().sum())

# summarize the number of rows and columns in the dataset
print(dataset_cp.shape)

# dataset with missing values
dataset_missing = dataset_cp.dropna()

# summarize the number of rows and columns in the dataset
print(dataset_cp.shape)

missing_attr = numpy.array(dataset_missing.values[:,0:8])
missing_label = numpy.array(dataset_missing.values[:,8])


# fill missing values with mean column values
dataset_impute = dataset_cp.fillna(dataset_cp.mean())
# count the number of NaN values in each column
print(dataset_impute.isnull().sum())

impute_attr = numpy.array(dataset_impute.values[:,0:8])
#impute_label = numpy.array(dataset_impute.values[:,8])

print(" = 5. Evaluate Some Algorithms = ")
# Split-out validation dataset
print(" == 5.1 Create a Validation Dataset: Split-out validation dataset == ")

# Test options and evaluation metric
print(" == 5.2 Test Harness: Test options and evaluation metric == ")
seed = 7
scoring = 'accuracy'


# Spot Check Algorithms
print("== 5.3 Build Models: build and evaluate our five models, Spot Check Algorithms ==")
datasets = []
datasets.append(('diabetes_attr', diabetes_attr, label))
datasets.append(('normalized_attr', normalized_attr, label))
datasets.append(('standardized_attr', standardized_attr, label))
datasets.append(('impute_attr', impute_attr, label))
datasets.append(('missing_attr', missing_attr, missing_label))

models = []
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('SVM', SVC()))

print("eval metric: " + scoring)
for dataname, attributes, target in datasets:
	# evaluate each model in turn
	results = []
	names = []
	print("= " + dataname + " = ")
	print("algorithm,mean,std,signficance,p-val")
	for name, model in models:
		kfold = model_selection.KFold(n_splits=10, random_state=seed)
		cv_results = model_selection.cross_val_score(model, attributes, target, cv=kfold, scoring=scoring)
		results.append(cv_results)
		#print("cv_results")
		#print(cv_results)
		names.append(name)
		
		t, prob = stats.ttest_rel(a= cv_results,b= results[0])
		#print("LR vs ", name, t,prob)
		# Below 0.05, significant. Over 0.05, not significant. 
		# http://blog.minitab.com/blog/understanding-statistics/what-can-you-say-when-your-p-value-is-greater-than-005
		statistically_different = (prob < 0.05)
		
		msg = "%s: %f (%f) %s %f" % (name, cv_results.mean(), cv_results.std(), statistically_different, prob)
		print(msg)

	# Compare Algorithms
	print(" == 5.4 Select Best Model, Compare Algorithms == ")
	fig = plt.figure()
	fig.suptitle('Algorithm Comparison for ' + dataname)
	ax = fig.add_subplot(111) # what does 111 mean?
	plt.boxplot(results)
	ax.set_xticklabels(names)
	#plt.show()


test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(diabetes_attr, label, test_size=test_size,
random_state=seed)

alphas = numpy.array([1,0.1,0.01,0.001,0.0001,0])
param_grid = dict(alpha=alphas)
model = Ridge()
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid.fit(X_train, Y_train)
print("grid.best_score=",grid.best_score_)
print("grid.best_estimator_.alpha=",grid.best_estimator_.alpha)

#building model)
model = Ridge(grid.best_estimator_.alpha)
model.fit(X_train, Y_train)
result = model.score(X_test, Y_test) # determine r2 value
print("score on X_test before storing=",result)

#predict
rand_index=randint(0, len(label)-1)
sample = diabetes_attr[rand_index]
actual = label[rand_index]
prediction = model.predict([sample])[0]
print("(actual,prediction) of sample", sample, " at random index=", rand_index, actual, prediction)

# save the model to disk
filename = 'diabetes_py_model.sav' 
pickle.dump(model, open(filename, 'wb'))
# some time later...
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb')) 
result = loaded_model.score(X_test, Y_test)

print("score on X_test after storing=",result)