# diabetes
Analysing Pima Indians Diabetes Data Set with Weka

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

The data sets are reproduced with the following filters:
1. diabetes.arff: original unchanged data set.
2. discretize.arff: supervised.attribute.Discretize
3. missing.arff: It is ambiguous which filter was applied to generate this data set in the case study, so I skip this file. Instead, I provide further treament in (5) and (6).
4. normalized.arff: unsupervised.attribute.Normalize on all attributes. Not applied on the class, which is nominal anyways.
5. remove_missing.arff: Fellow user (credits due) at The UCI ML repository [3,4] observes there are zeros in places where they are biologically impossible, such as the blood pressure. They are likely missing values. I further checked the remaining attributes. Except pregnancy, they either cannot be zero (e.g. mass) or don't have zero (e.g. pedigree). In the former case, I assume zero indicates missing values, and use Dr. Brownlee's post [5] to remove the missing values. This reduces the number of instances from 768 to 392. Needless to say, this purges data set by half.
6. replaced_missing.arff: Following on comment on (5), I again use Brownlee's post [5] to replace the missing values with the mean of the attribute value. 
7. square.arff: <TBD> 
8. standardized.arff: unsupervised.attribute.Standardize

With these data sets, I have been able to reproduce the Weka Experiment in part 2. I added one more algorithm for establishing a baseline namely, ZeroR. The numbers are highly similar as reported by Shvartser [5]. We have considered both linear, e.g. Logistic Regression (LR), and non-linear, e.g. Random Forest (RF), classifiers. We also see that their evaluation metrics are not statistically different. In such a case, linear model is preferable and I use LR for further analysis.

### Expansion

Now, the case study rightly notices that there is class imbalance: 65% -ve, and 35% +ve. In part 2, I am unsure if the class was balanced in cross validation. In the interest of reproducing the study, I do not balance the classes for the above data sets, namely (1-8). As you may know, class imbalance leads to a majority classifier, and we see this artifact when ZeroR gives us about 65% accuracy. To balance the classes, I generate two data sets:

9. oversampling.arff: increases the number of minority class (+ve) instances by a specified percentage. I specify the default of 100%, and that increases the number of +ve instances to be at par with the majority class (-ve) instances. I applied the SMOTE and Randomize filters in that order. The total instance count becomes 1,036. This mildly contaminates the pure data samples with synthethic ones. See Shams youtube for step-by-step method for applying SMOTE and Randomize fileters [7]. 
10. undersampling.arff: I use SpreadsubSample and Randomize fileters in that order to reduce the number of majority class instances to be at par with minority class instances, namely 268. The total number of instances are now 536. Again, see Shams second video [8] for step-by-step procedure. This method removes valuable instances of the majority class. 

Intuitively, acquiring additional instances is likely to be more than undersampling the majority class or just oversampling the minority class.

The accuracy or percent_correct in the Weka Experiment are stated below:

<pre>
Dataset        		    ZeroR |   LR 
diabetes.arff       	65.11 |   77.10 v
oversampling.arff   	51.74 |   75.51 v
undersampling.arff  	49.62 |   73.73 v
</pre>

where v represents the statistical difference.

Note, the accuracy of ZeroR dropped to about 50% as anticipated. The accuracy of LR is still close to our previous best. Determining statistical significance of accuracies for LR across the data sets is still outstanding. 

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
oversampling.arff   	0.00 |   0.75 v
undersampling.arff  	0.50 |   0.74 v
</pre>

Note, F-Measure of ZeroR for oversampling is 0.00. This seems incorrect, and is different (0.446) to the same metric, algorithm and data set in Weka Explorer.

### Controlling the number of false negatives

It might be nice to see the associated probabilities with a prediction. Incidentally, LR provides associated probability out-of-the-box. The following direction will allow you to store the predictions in csv.

Weka Explorer -> Classify -> More options -> Output predictions Choose -> CSV file

I build LR model with "Use training set" and store the predictions in the pred.csv file. A sample:

inst#,actual,predicted,error,neg_prob,pos_prob
1,1:tested_negative,1:tested_negative,,*0.744,0.256
2,1:tested_negative,1:tested_negative,,*0.577,0.423
3,1:tested_negative,1:tested_negative,,*0.829,0.171
4,1:tested_negative,1:tested_negative,,*0.994,0.006
5,1:tested_negative,2:tested_positive,+,0.424,*0.576

the * sign indicates the probability of the predicted class. 
the + sign indicate an incorrect prediction.

Whichever class has the highest probability i.e. greater than 0.5 is the predicted class. The confusion matrix for building LR across the whole data set:

=== Confusion Matrix ===
   a   b   <-- classified as
 446  54 |   a = tested_negative
 111 157 |   b = tested_positive
 
The false positives (54 above) cause unnecessary worry, and typically follows another test to confirm the result. The false negatives (111 above) are really bad. In this case, LR is predicting that a subject does NOT have a disease where they may have actually got one. Naturally, a physician might want to reduce the number of false negatives at the cost of increasing false positives. I could not adjust the probability threshold from 0.5 to an another value in Weka.

I did a work around by building on pred.csv file. At the bottom of diabetes_proc tab in the diabetes.xlsx file, positive probability and trigger are drawn. The positive probability has been sorted and hence the ascending curve is seen. As the probability crosses 0.5+delta, the trigger happens and the prediction is a positive test. 

I invite you to change the value of delta to see the effects on the chart and the confusion matrix. The value of false negatives reduces from 112 to 84 when the delta value is reduced from zero to -0.10. As a side effect, false positives also increase from 69 to 94 with this delta change.

Conclusions
My thoughts are Weka Explorer and Experimenter are excellent tools to get quick and dirty analysis with algorithms and data sets without writing code. For run time changes and further fine tuning, coding seems necessary.

### Outstanding
- So far, we have prepared and identified which datasets and algorithms seem most suitable. An actual model to generate predictions is pending. I see that as a simple exercise, and is documented in the lesson 14 of the 14-day mini course on Weka [10].

### Future Work
- Reproduce the above work in Python


### References
 
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
