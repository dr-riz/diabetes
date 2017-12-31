print("Reproducing and expanding case study of Shvartser posted at Dr. Brownlee's machinelearningmastery.com")

# preproc imports
import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import preprocessing as preproc
import numpy
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# algo eval imports
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# fine tuning
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# significance tests
import scipy.stats as stats
import math

# build and save model using Pickle
from random import *
import pickle

# final model
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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

#undersampling
# Separate majority and minority classes
df_majority = dataset[dataset['class']==0]
df_minority = dataset[dataset['class']==1]

print("df_minority['class'].size", df_minority['class'].size)

# Downsample majority class

df_majority_downsampled = resample(df_majority, 
                          replace=False,    # sample without replacement
                          n_samples=df_minority['class'].size,  # match minority class
                          random_state=7) # reproducible results
 
# Combine minority class with downsampled majority class
df_downsampled = pandas.concat([df_majority_downsampled, df_minority])
 
print("undersampled", df_downsampled.groupby('class').size()) 
df_downsampled=df_downsampled.sample(frac=1).reset_index(drop=True)
undersampling_attr = numpy.array(df_downsampled.values[:,0:8])
undersampling_label = numpy.array(df_downsampled.values[:,8])

# oversampling

sm = SMOTE(random_state=7)
x_val = dataset.values[:,0:8]
y_val = dataset.values[:,8]
X_res, y_res = sm.fit_sample(x_val, y_val)

features=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age']
oversampled_df = pandas.DataFrame(X_res)
oversampled_df.columns = features
oversampled_df = oversampled_df.assign(label = numpy.asarray(y_res))
oversampled_df = oversampled_df.sample(frac=1).reset_index(drop=True)

oversampling_attr = oversampled_df.values[:,0:8]
oversampling_label = oversampled_df.values[:,8]

print("oversampled_df", oversampled_df.groupby('label').size()) 


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
datasets.append(('undersampling_attr', undersampling_attr, undersampling_label))
datasets.append(('oversampling_attr', oversampling_attr, oversampling_label))

models = []
models.append(('LR', LogisticRegression(class_weight='balanced')))
models.append(('NB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('DT', DecisionTreeClassifier()))

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

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'penalty': ['l2','l1']}
logr = GridSearchCV(LogisticRegression(class_weight='balanced'), param_grid, scoring='accuracy')
logr.fit(X_train, Y_train)
print("logr.best_score=",logr.best_score_)
print("logr.best_estimator_.C=",logr.best_estimator_.C)
print("logr.best_estimator_.penalty=",logr.best_estimator_.penalty)

#building model
model = LogisticRegression(class_weight='balanced',C=logr.best_estimator_.C, penalty=logr.best_estimator_.penalty)
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


delta0_predictions=loaded_model.predict(X_test)
print("delta0_predictions")
print(accuracy_score(Y_test, delta0_predictions))
print(confusion_matrix(Y_test, delta0_predictions))
print(classification_report(Y_test, delta0_predictions))

delta=0.10
probs=loaded_model.predict_proba(X_test)
print("loaded_model.self.classes_",loaded_model.classes_)
if (probs[0][0] > (probs[0][1] + delta)): 
	pred = 0
print("at random index, (actual,probability, prediction) ", 0, Y_test[0], probs[0], pred)

#predictions=probs[0][0] > (probs[0][1] + delta)
report = [[ins[0], ins[1], 0] if (ins[0] > (ins[1] + delta)) else [ins[0], ins[1], 1]  for ins in probs]

report_df = pandas.DataFrame(report, columns=['neg_prob','pos_prob','pred'])
print(report_df.tail(10))
report_df=report_df.sort_values(by=['pred','pos_prob'])
print(report_df.tail(10))
print(report_df.describe())

predictions = numpy.array(report_df.values)[:,2]
print(predictions)
# IF(N769>(M769+$N$776),"tested_positive","tested_negative")



#print(probs.

print("deltaX",delta)
print(accuracy_score(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
print(classification_report(Y_test, predictions))

positive_prob=numpy.array(report_df.values)[:,1]


from matplotlib.legend_handler import HandlerLine2D

plt.clf()
pred_legend,=plt.plot(predictions, 'r', label="predictions") 
prob_legend,=plt.plot(positive_prob, 'b', label="+ve probability")

plt.legend(handler_map={pred_legend: HandlerLine2D(numpoints=4)})

plt.show()

#print report
