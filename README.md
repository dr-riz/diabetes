# diabetes
Analysing Pima Indians Diabetes Data Set with Weka
 
Reproducing case study of Shvartser [1] posted at Brownlee's comprehensive ML learning website [2].

In part 1, The case study claims that "Larger values of plas combined with larger values for age, pedi, mass, insu, skin, pres, and preg tends to show greater likelihood of testing positive for diabetes."

I don't see much likelihood from the scatter plot. Both positive and negative data points for the said attributes overlap for the most part.

The data sets are reproduced with the following filters:
(1) diabetes.arff: original unchanged data set.
(2) diabetes_discretize.arff: supervised.attribute.Discretize
(3) diabetes_missing.arff: It is ambiguous which filter was applied to generate this data set in the case study, so I skip this file. Instead, I provide further treament in (5) and (6).
(4) diabetes_normalized.arff: unsupervised.attribute.Normalize on all attributes. Not applied on the class, which is nominal anyways.
(5) diabetes_remove_missing.arff: Fellow user (credits due) at The UCI ML repository [3,4] observes there are zeros in places where they are biologically impossible, such as the blood pressure. They are likely missing values. I further checked the remaining attributes. They either cannot be zero (e.g. mass) or don't have zero (e.g. pedigree), except pregnancy. In the former case, I assume zero indicates missing values, and use Brownlee's post [5] to remove the missing values. This reduces the number of instances from 768 to 392. Needless to say, this purges data set by half.
(6) diabetes_replaced_missing.arff: Following on comment on (5), I again use Brownlee's post [5] to replace the missing values with the mean of the attribute value. 
(7) diabetes_square.arff: <TBD> 
(8) diabetes_standardized.arff: unsupervised.attribute.Standardize

With these data sets, I have been able to reproduce the Weka Experiment in part 2. I added one more algorithm for establishing a baseline namely, zeroR. The numbers are highly similar as reported. We have considered both linear, e.g. Logistic Regression (LR), and non-linear, e.g. Random Forest (RF), classifiers. We also see that their evaluation metrics are not statistically different. In such a case, linear model is preferable and I use LR for further analysis.

Now, the case study rightly notices that there is class imbalance: 65% -ve, and 35% +ve. In part 2, I am unsure if the class was balanced in cross validation. In the interest of reproducing the study, I do not balance the classes for the above data sets, namely (1-8). As you may know, class imbalance leads to a majority classifier, and we see this artifact when ZeroR gives us about 65% accuracy. To balance the classes, I generate two data sets:

(9) diabetes_oversampling.arff: increases the number of minority class (+ve) instances to be at par with the majority class (-ve) instances, namely 500. I applied the SMOTE and Randomize filters in that order. See Shams youtube for step-by-step method [7].   
(10) diabetes_undersampling.arff: I use SpreadsubSample and Randomize fileters in that order to reduce the number of majority class instances to be at par with minority class instances, namely 268. The total number of instances are now






Nonetheless, I generate

In part 2, there are 3 missing datasets: (a) _missing, (b) _remove_missing and (c) _replaced_missing. how did you tread (a) and (c)?



I assume z

or ca, and we assume so. 

(6) 

 
[1] https://machinelearningmastery.com/case-study-predicting-the-onset-of-diabetes-within-five-years-part-1-of-3/
[2] https://machinelearningmastery.com/start-here/
[3] https://archive.ics.uci.edu/ml/datasets/pima+indians+diabetes
[4] Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.
[5] https://machinelearningmastery.com/how-to-handle-missing-values-in-machine-learning-data-with-weka/
[6] https://machinelearningmastery.leadpages.co/leadbox/144d0b573f72a2%3A164f8be4f346dc/5675267779461120/
[7] https://www.youtube.com/watch?v=w14ha2Fmg6U&t=242s
[8] https://www.youtube.com/watch?v=ocOlm73HeNs
