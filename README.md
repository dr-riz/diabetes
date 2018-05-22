# diabetes
Analysing Pima Indians Diabetes dataset with Weka and Python

## Reproducing/Expanding in Weka

### Abstract
Reproducing case study of Shvartser [1] posted at Dr. Brownlee's comprehensive ML learning website [2].

- Reproducing the study and explicitly stating the filters used
- Expanding on the case study by:
	* balancing classes by over and under sampling minority and majority classes, respectively
	* annotating predictions with probabilities
	* exercising control over probability threshold to reduce false negatives at the cost of false positives

### Reproducing
In part 1 of the case study, The case study claims that "Larger values of plas combined with larger values for age, pedi, mass, insu, skin, pres, and preg tends to show greater likelihood of testing positive for diabetes."

I don't see much likelihood from the scatter plot. Both positive and negative data points for the said attributes overlap for the most part.

The datasets are reproduced with the following filters:
1. diabetes.arff: original unchanged dataset.
2. discretize.arff: supervised.attribute.Discretize
3. missing.arff: It is ambiguous which filter was applied to generate this dataset in the case study, so I skip this file. Instead, I provide further treament in (5) and (6).
4. normalized.arff: unsupervised.attribute.Normalize on all attributes. Not applied on the class, which is nominal anyways.
5. remove_missing.arff: Fellow user (credits due) at The UCI ML repository [3,4] observes there are zeros in places where they are biologically impossible, such as the blood pressure. They are likely missing values. I further checked the remaining attributes. Except pregnancy, they either cannot be zero (e.g. mass) or don't have zero (e.g. pedigree). In the former case, I assume zero indicates missing values, and use Dr. Brownlee's post [1] to remove the missing values. This reduces the number of instances from 768 to 392. Needless to say, this purges dataset by half.
6. replaced_missing.arff: Following on comment on (5), I again use Brownlee's post [5] to replace the missing values with the mean of the attribute value. 
7. square.arff: <TBD> 
8. standardized.arff: unsupervised.attribute.Standardize

With these datasets, I have been able to reproduce the Weka Experiment in part 2. I added one more algorithm for establishing a baseline namely, ZeroR. The numbers are highly similar as reported by Shvartser [5]. We have considered both linear, e.g. Logistic Regression (LR), and non-linear, e.g. Random Forest (RF), classifiers. We also see that their evaluation metrics are not statistically different. In such a case, linear model is preferable and I use LR for further analysis.

### Expansion

Now, the case study rightly notices that there is class imbalance: 65% -ve, and 35% +ve. In part 2, I am unsure if the class was balanced in cross validation. In the interest of reproducing the study, I do not balance the classes for the above datasets, namely (1-8). As you may know, class imbalance leads to a majority classifier, and we see this artifact when ZeroR gives us about 65% accuracy. To balance the classes, I generate two additional datasets:

9. oversampling.arff: increases the number of minority class (+ve) instances by a specified percentage. I specify the default of 100%, and that increases the number of +ve instances to be at par with the majority class (-ve) instances. I applied the SMOTE and Randomize filters in that order. The total instance count becomes 1,036. This mildly contaminates the pure data samples with synthethic ones. See Shams youtube for step-by-step method for applying SMOTE and Randomize filters [7]. 
10. undersampling.arff: I use SpreadsubSample and Randomize filters in that order to reduce the number of majority class instances to be at par with minority class instances, namely 268. The total number of instances are now 536. Again, see Shams second video [8] for step-by-step procedure. This method removes valuable instances of the majority class. 

Intuitively, acquiring additional instances is likely to be more effective than undersampling the majority class or just oversampling the minority class.

The accuracy or percent_correct in the Weka Experiment are stated below:

<pre>
Dataset        		    ZeroR |   LR 
diabetes.arff       	65.11 |   77.10 v
oversampling.arff   	51.74 |   75.51 v
undersampling.arff  	49.62 |   73.73 v
</pre>

where v represents the statistical difference.

Note, the accuracy of ZeroR dropped to about 50% as anticipated. The accuracy of LR is still close to our previous best. Determining statistical significance of accuracies for LR across the datasets is still outstanding. 

Next, the Area under ROC in the Weka Experiment are stated below:

<pre>
Dataset        		   ZeroR |   LR 
diabetes.arff       	0.50 |   0.83 v
oversampling.arff   	0.50 |   0.84 v
undersampling.arff  	0.50 |   0.82 v
</pre>

Finally, the F-Measures in the Weka Experiment are stated below:

<pre>
Dataset        		   ZeroR |   LR 
diabetes.arff       	0.79 |   0.83 v
oversampling.arff       0.00^|   0.75 v
undersampling.arff  	0.50 |   0.74 v
</pre>

^Note, F-Measure of ZeroR for oversampling is 0.00. This seems incorrect, and is different (0.446) to the same metric, algorithm and dataset in Weka Explorer.

### Controlling the number of false negatives

It might be nice to see the associated probabilities with a prediction. Incidentally, LR provides associated probability out-of-the-box. The following direction will allow you to store the predictions in csv.

Weka Explorer -> Classify -> More options -> Output predictions Choose -> CSV file

I build LR model with "Use training set" and store the predictions in the pred.csv file. A sample:

<pre>
inst#,actual,predicted,error,neg_prob,pos_prob
1,1:tested_negative,1:tested_negative,,*0.744,0.256
2,1:tested_negative,1:tested_negative,,*0.577,0.423
3,1:tested_negative,1:tested_negative,,*0.829,0.171
4,1:tested_negative,1:tested_negative,,*0.994,0.006
5,1:tested_negative,2:tested_positive,+,0.424,*0.576
</pre>

the * sign indicates the probability of the predicted class. 
the + sign indicate an incorrect prediction.

Whichever class has the highest probability i.e. greater than 0.5 is the predicted class. The confusion matrix for building LR across the whole dataset:

<pre>
=== Confusion Matrix ===
   a   b   <-- classified as
 446  54 |   a = tested_negative
 111 157 |   b = tested_positive
</pre>
 
The false positives (54 above) cause unnecessary worry, and typically follows another test to confirm the result. The false negatives (111 above) are really bad. In this case, LR is predicting that a subject does NOT have a disease where they may have actually got one. Naturally, a physician might want to reduce the number of false negatives at the cost of increasing false positives. I could not adjust the probability threshold from 0.5 to an another value in Weka.

I did a work around by building on pred.csv file. At the bottom of diabetes_proc tab in the diabetes.xlsx file, positive probability and trigger are drawn. The positive probability has been sorted and hence the ascending curve is seen. As the probability crosses 0.5+delta, the trigger happens and the prediction is a positive test. 

I invite you to change the value of delta to see the effects on the chart and the confusion matrix. The value of false negatives reduces from 112 to 84 when the delta value is reduced from zero to -0.10. As a side effect, false positives also increase from 69 to 94 with this delta change.

### Outstanding
- So far, we have prepared and identified which datasets and algorithms seem most suitable. An actual model to generate predictions is pending. I see that as a simple exercise, and is documented in the lesson 14 of the 14-day mini course on Weka [10].
- Determining statistical significance of accuracies for LR across the datasets is still outstanding.

## Reproducing/Expanding in Python

After reproducing and expanding the case study in Weka, I decided to reproduce them in Python. Reasons:
- I have developed understanding of the problem
- Weka case will serve as a baseline for comparison
- I want to get an idea on the amount of effort and flexibility both platforms provide on the same problem

For the warm up, I worked through Dr. Brownlee's "Your First Machine Learning Project in Python Step-By-Step" [11]. I wrote up the code with headings to allow the follower of the output to see what is going on.

In addition to my expanded Weka case study, the Python part expands the case study further by:
- additional algorithms for spot checking
- balancing class labels using LR parameter
- grid search on the hyperparameters
- searching for the cross over point between sensitivity and specificity
- plotting roc curve
- sensitivity and specificity cross over
- (not too exciting) saving and loading the model from disk

### Reproducing & Expansion

Similar to Python Step-By-Step [11], I summarize and visualize the datasets, generating plots where possible. I 

In contrast to creating different files for each datasets, I store the datasets in memory. I rescale the data, both normalization and standardization as suggested in the post [12]. I observe that that the mean and standard deviation are very close to zero and one, respectively, but not exactly. I posted this as a question on the blog [12], and assume that these close-enough values are acceptable in the community and move forward.

I feed the dataset to the algorithms. For each dataset, a comparison plot is generated, where each algorithm performance is drawn using a box plot.

While not too worried about the average accuracy value being different, I notice the Weka and Python are report differences in the statistical significance for the same algorithms. This merits further investigation.

Nonetheless, the algorithms have similar performance. One exception is SVM, which has consistently lower accuracy. The algorithms are instantiated with their default parameters, arguably sufficient for the first run. The algorithms contain both linear (LR and LDA) and nonlinear (KNN, CART, NB etc.) algorithms. In all the cases, I see that LR performs well with high accuracy either LR has the highest accuracy or statistically insignificant to others having higher accuracy. One exception is where RF provides statistically significant better accuracy compared to LR. One possible way forward is to fine tune RF to see if we can further improve the accuracy. This is left as future work.

### Grid Search

I take LR to the next rounds. Next step is to fine tune LR. I perform a grid search [13] for LR over C and penalty parameters. LogisticRegression implementation in scikit takes a class_weight parameter, which when set to 'balanced' automatically adjust weights inversely proportional to class frequencies in the input data. I prefer this over oversampling minority class with SMOTE, as that contaminates the samples mildly. I use test_size = 0.33 for splitting train/test sets for grid search. 

Before the actual grid search, I get a baseline accuracy of LR with default parameters and the train/test datasets, namely about 0.76. To my surprise, optimal C and penalty_model parameters from grid search also give us about the same accuracy i.e. 0.76. It turns out the optimal and default C value is the same i.e. 1. However, there is a difference in the penalty model i.e. default is l2 and the optimal is l1. The different penalty model apparently had no effect on the accuracy.

Next I did a quick store and load of the model, and confirm the accuracy score remains the same. 

With the loaded model, I generate the evaluation metrics on the test set (X_test):
- accuracy_score (about 0.76)^1
- confusion_matrix ('tn, fp, fn, tp:', 121, 41, 21, 71 )
- sensitivity or true positive rate (tpr) (about 0.63)
- specificity or ture negative rate (tnr) (about 0.75)
- classification report 
- roc area under the curve (auc) (about 0.84)

### Controlling number of false negatives -- The sensitivity and specificity trade off

Recall, we want to exercise control over false negatives by adjusting the value of delta i.e. as the positive probability crosses 0.5+delta, the trigger happens and the prediction is a positive test. In general, sensitivity and specificity trade off against each other. Recall, we saw an example in the diabetes.xlsx when we changed the value of delta. The original paper [14] that provides the dataset shows that sensitivity and specificity cross over when their values are about 0.76^1. This may be considered a good balance between tpr and tnr. The default value of zero delta provide us with tpr and tnr of 0.63 and 0.75, respectively. I determine the value of delta where sensitivity and specificity cross by looping through a set of delta values and generating respective tpr and tnr values. Note, I am superimposing the delta value on the underlying LR model and probabilities, and then generating predictions and other metrics on the superimposed perspective. The underlying LR model remains unchanged. I looked into pushing the delta value into the underlying but decided against it due to the reasoning by lejlot at stackoverflow [15]. 

I draw a plot show the changing values of sensitivity and specificity with each delta value. From the plot, it can be seen that sensitivity and specificity cross when delta is about 0.026. The rate value of sensitivity and specificity is about 0.76. We have reproduced the result of the paper by LR. In the shell output, we can also see that false negative (fn) increase when the value of delta increases. 

- ('deltaX,sensitivity_tpr,specificity_tnr:', -0.02, 0.7717391304347826, 0.7407407407407407)
- ('confusion_matrix: tn, fp, fn, tp:', 120, 42, *21*, 71)
...
- ('deltaX,sensitivity_tpr,specificity_tnr:', 0.06, 0.7391304347826086, 0.7654320987654321)
- ('confusion_matrix: tn, fp, fn, tp:', 124, 38, *24*, 68)
...
- ('deltaX,sensitivity_tpr,specificity_tnr:', 0.12, 0.717391304347826, 0.7777777777777778)
- ('confusion_matrix: tn, fp, fn, tp:', 126, 36, *26*, 66)

Let delta = -0.10
The evaluation metrics on the test set (X_test):
- accuracy_score (about 0.74)
- confusion_matrix ('tn, fp, fn, tp:', 111, 51, *16*, 76) <= reduced from 21 when delta=0
- sensitivity or true positive rate (tpr) (about 0.83) <= increased
- specificity or ture negative rate (tnr) (about 0.67) <= decreased
- classification report

Finally, we can also plot the probability values for sorted positive tests and their respective triggers, and visualize them graphically. 

### Future work
- Investigate why Weka and Python report differences in the statistical significance with their default parameters.
- Explore if fine tuning RF further improve the accuracy of predictions

## Reproducing in Jupyter and Databrick notebooks
After reproducing and expanding the case study in Python, I first reproduced it in Jupyter notebook (diabetes.ipynb). The motivation is to get hands-on with Jupyter notebook. In hindsight, it might be natural to explore and develop a data science case study with the notebook first and then transform it into a script. This is because the notebook allows more interactive to and fro modications and micro executions.

Once the Jupyter notebook is complete, I move on to Databricks platform [16]. Databricks aims to help users with cloud-based big data processing using Spark. Also, It has a notebook similar in principle to Jupyter. The community edition offers a single node cluster and associated workspace to create and execute multiple notebooks. The cluster springs to life within minutes, and I use Databricks Runtime Version 4.1(includes Apache Spark 2.3.0, Scala 2.11). I see the Databricks notebook over spark cluster as a jumping board to perform data science at scale. So, I transform diabetes.py into a Databricks notebook [17]. Naturally, I need to upload the data (diabetes.data) to the cluster. It turns out to be mostly a copy paste job except with the following (annoying) differences:
1. With present version of sklearn (0.18.1) in Databricks cluster, the latest version of SMOTE package does not work. I installed an older compatible version.
2. The plot.show() needs to be followed by display() for the plot to appear.

After you sign up to Databrciks (say community edition) and setup a cluster, you may import my notebook with a click of a button [17]. Isn't that nice? :-) The only caveat is notebook available for 6 motnhs after publishing. So, I also export it as diabetes_databricks.ipynb so that it remains eternal. It is possible to load Jupyter notebook in Databricks [18], though I haven't tried myself. I need to see how can I use pyspark and scala/spark, perhaps in a different sitting. 

## Conclusions
Personally, I feel drawing a visual grid of too many attributes is overwhelming, especially for scatter plot between attributes. I found a correlation matrix much more helpful in quantifying relationships. In this tutorial, we didn't address any outliers or performed any feature engineering. One reason that jumps out is that it is highly curated dataset. 

My thoughts are Weka Explorer and Experimenter are excellent tools to get quick and dirty analysis with algorithms and datasets without writing code. For run time changes and further fine tuning, coding seems necessary.

## References
<pre> 
[1] https://machinelearningmastery.com/case-study-predicting-the-onset-of-diabetes-within-five-years-part-1-of-3/
[2] https://machinelearningmastery.com/start-here/
[3] https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
[4] Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
[5] https://machinelearningmastery.com/how-to-handle-missing-values-in-machine-learning-data-with-weka/
[6] https://machinelearningmastery.leadpages.co/leadbox/144d0b573f72a2%3A164f8be4f346dc/5675267779461120/
[7] https://www.youtube.com/watch?v=w14ha2Fmg6U&t=242s
[8] https://www.youtube.com/watch?v=ocOlm73HeNs
[9] https://social.msdn.microsoft.com/Forums/azure/en-US/71d0efe7-4de0-4434-a4a6-5c27af876c1b/smote-consequences-a-question-and-an-alternative?forum=MachineLearning
[10] https://machinelearningmastery.com/applied-machine-learning-weka-mini-course/
[11] https://machinelearningmastery.com/machine-learning-in-python-step-by-step/
[12] https://machinelearningmastery.com/rescaling-data-for-machine-learning-in-python-with-scikit-learn/#comment-425333
[13] https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/
[14] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2245318/pdf/procascamc00018-0276.pdf
[15] https://stackoverflow.com/questions/19984957/scikit-predict-default-threshold
[16] https://databricks.com/try-databricks
[17] https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6710283852221561/2246037980686797/3061735664085706/latest.html
[18] https://vincentlauzon.com/2018/02/27/import-notebooks-in-databricks/
</pre>