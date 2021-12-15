# Credit_Risk_Analysis

## Overview
Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. Therefore, different techniques to train and evaluate models with unbalanced classes. Use imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, oversample the data using the RandomOverSampler and SMOTE algorithms, and undersample the data using the ClusterCentroids algorithm. Then, a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. Next, compare two new machine learning models that reduce bias, 'BalancedRandomForestClassifier' and 'EasyEnsembleClassifier', to predict credit risk. Evaluate the performance of these models and provide a written recommendation on whether they should be used to predict credit risk.

## Results
Credit card data was cleaned prior to implementing machine learning techniques.  Null columns and rows were dropped, interest rates were converted to numerical values, and target (y-axis) columns were converted to low_risk and high_risk based on their values.

Once the data was cleaned, it was split into training and testing categories, which resulted in four sets of data:
- X_train
- X_test
- y_train
- y_test

A random_state of 1 was used across all models to ensure reproducible results.  The balance of low_risk and high_risk is unbalanced, but this was expected as credit risk is an inherently unbalanced classification problem, since good loans easily outnumber risky loans.

![01-dataset](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/01-Balance%20of%20Dataset.png?raw=true)

### Oversampling Algorithms
#### Naive Random Oversampling
In this model, instances of the high_risk class were oversampled, which is where data from the high_risk data set is randomly selected and added to the training set until the high_risk and low_risk classes were balanced.

Unbalanced                |  Balanced
:------------------------:|:-------------------------:
![02-unbalanced](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/02-Unbalanced.png?raw=true)|![03-balanced](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/03-Balanced.png?raw=true)

Once the datasets were balanced, the model trained the data, the algorithm analyzed the data and attempts to learn patterns in the data.

Naive random oversampling on this data gave the following scores: Balanced Accuracy: 0.6287

![04-imbalanced_class_report](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/04-imbalanced_class_report.png?raw=true)

A balanced accuracy score of 0.6287 means that 37.1% of classes are incorrect and 62.9% are correct.

An average precision score of 0.99 means that this model quantified the number of positive class predictions that actually belong to the positive class 99% of the time.

An average recall score of 0.64 means that this model quantified the number of positive class predictions made out of all positive examples 64% of the time.


#### SMOTE Oversampling
The synthetic minority oversampling technique (SMOTE) is another oversampling approach to deal with unbalanced datasets. In SMOTE, like random oversampling, the size of the minority (high_risk) is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.

Once the data was balanced and trained, SMOTE oversampling gave the following scores:  Balanced Accuracy: 0.612

![05-smote_class_report](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/05-smote_class_report.png?raw=true)

The balanced accuracy score for this model means that 61.2% of classes are correct and 38.8% are incorrect.

An average precision score of 0.99 means that this model predicted positive class predictions 99% of the time.

An average recall score of 0.68 means that 68% of class predictions made out of all positive examples in the dataset were correct and 32% were incorrect.

Comparing the performance of the naive random oversampling and SMOTE oversampling models, they appeared to perform about the same.  It's important to note that although SMOTE reduces the risk of oversampling, it does not always outperform random oversampling.  Another deficiency of SMOTE is its vulnerability to outliers. We said earlier that a minority class instance is selected, and new values are generated based on its distance from its neighbors. If the neighbors are extreme outliers, the new values will reflect this. Finally, keep in mind that sampling techniques cannot overcome the deficiencies of the original dataset!


### Undersampling Algorithm
#### ClusterCentroids

Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.

Once the data were balanced and trained, ClusterCentroids undersampling gave the following scores: Balanced Accuracy: 0.513

![06-ccluster_class_report](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/06-ccluster_class_report.png?raw=true)

The balanced accuracy score for this model was 0.513, which means that 48.7% of classes are incorrect and 51.3% are correct.

An average precision score of 0.99 means the ClusterCentroid algorithm predicted positive class predictions 99% of the time on the dataset.

An average recall score of 0.46 means that 46% of class predictions made from all positive examples in the dataset were correct, whereas 54% were incorrect.  These results are worse than those from random undersampling! This underscores an important point: While resampling can attempt to address imbalance, it does not guarantee better results.


### Combination Sampling
#### SMOTEENN
As previously discussed, a downside of oversampling with SMOTE is its reliance on the immediate neighbors of a data point. Because the algorithm doesn't see the overall distribution of data, the new data points it creates can be heavily influenced by outliers. This can lead to noisy data. With downsampling, the downsides are that it involves loss of data and is not an option when the dataset is small. One way to deal with these challenges is to use a sampling strategy that is a combination of oversampling and undersampling.

SMOTEENN combines the SMOTE and Edited Nearest Neighbors (ENN) algorithms. SMOTEENN is a two-step process:

	1. Oversample the minority class with SMOTE.
	2. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.

Once the data were balanced and trained, the SMOTEEN algorithm gave the following scores: Balanced Accuracy: 0.622

![07-smoteenn_class_report](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/07-smoteenn_class_report.png?raw=true)

SMOTEENN's balanced accuracy score was 0.622, which means 62.2% of class predictions were correct and 37.8% were incorrect.

An average precision score of 0.99 means the SMOTEENN algorithm predicted positive class predictions 99% of the time on this dataset.

An average recall score of 0.57 means that 57% of class predictions made out of all positive examples in the dataset were correct, whereas 43% were incorrect.


### Ensemble Learners
#### Balanced Random Forest Classifier
These simple trees are weak learners because they are created by randomly sampling the data and creating a decision tree for only that small portion of data. And since they are trained on a small piece of the original data, they are only slightly better than a random guess. However, many slightly better than average small decision trees can be combined to create a strong learner, which has much better decision-making power.

Random forest algorithms are beneficial because they:

	* Are robust against overfitting as all of those weak learners are trained on different pieces of the data.
	* Can be used to rank the importance of input variables in a natural way.
	 *Can handle thousands of input variables without variable deletion.
	* Are robust to outliers and nonlinear data.
	* Run efficiently on large datasets.

Once the data were balanced and trained, the balanced random forest algorithm gave the following scores: Balanced Accuracy: 0.788

![08-randomf_class_report](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/08-randomf_class_report.png?raw=true)

This algorithm's balanced accuracy score is 0.788, which means nearly 79% of class predictions were correct and 21% were incorrect.

Balanced Random Forest's average precision score of 0.99 means that this algorithm predicted positive class predictions 99% of the time on this dataset.

An average recall score of 0.91 means that 91% of class predictions made out of all positive examples in this dataset were correct, whereas 9% were incorrect.


#### Easy Ensemble AdaBoost Classifier
The idea behind Adaptive Boosting, called AdaBoost, is easy to understand. In AdaBoost, a model is trained then evaluated. After evaluating the errors of the first model, another model is trained. This time, however, the model gives extra weight to the errors from the previous model. The purpose of this weighting is to minimize similar errors in subsequent models. Then, the errors from the second model are given extra weight for the third model. This process is repeated until the error rate is minimized.

Once the data were balanced and trained, the Easy Ensemble AdaBoost Classifier algorithm gave the following scores: Balanced Accuracy: 0.925

![09-easy_ensamble_class_report](https://github.com/ASCHEET/Credit_Risk_Analysis/blob/main/Resources/09-easy_ensamble_class_report.png?raw=true)

Easy Ensemble Classifier's accuracy score of 0.925 means that its predictions were correct 92.5% of the time and 7.5% were incorrect.

This algorithm's precision score of 0.99 means that it predicted positive class predictions 99% of the time on this dataset.

The average recall score of 0.94 means that 94% of class predictions made from all positive examples in this dataset were correct.  


## Summary
The oversampling, undersampling, and combination sampling algorithms' performance were relatively the same. Balanced Random Forest Classifier had a higher balanced accuracy score than the previous algorithms tested, but it was not good enough for predicting credit risk.

Out of the six supervised machine learning algorithms tested, Easy Ensemble Classifier performed the best overall.  It had a balanced accuracy score, along with high precision and recall scores.  It also had a high specificity score, which means this algorithm correctly determined actual negatives 92.5% of the time, and a high F1 score.  This means the harmonic mean of precision and recall were 0.97 out of 1.0.
