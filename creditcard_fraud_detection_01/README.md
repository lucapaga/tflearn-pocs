# Credit Card Transactions' Fraud Detector

## Credits

This "problem" comes from https://www.kaggle.com/dalpozz/creditcardfraud

## Description of the DATA-SET

The datasets contains transactions made by credit cards in September 2013 by european cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, ... V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-senstive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

The dataset has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group (http://mlg.ulb.ac.be) of ULB (Université Libre de Bruxelles) on big data mining and fraud detection. More details on current and past projects on related topics are available on http://mlg.ulb.ac.be/BruFence and http://mlg.ulb.ac.be/ARTML

Please cite: Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. Calibrating Probability with Undersampling for Unbalanced Classification. In Symposium on Computational Intelligence and Data Mining (CIDM), IEEE, 2015

## Test Runs

### MODEL AND TRAINING:

Nr. | Activation/H | Activation/O | Loss | Optimizer | Metric | Rate | Nr. of Epochs | Loss | Metric | VAL Loss | VAL Metric
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
1  | TANH    | SOFTMAX | 'softmax_categorical_crossentropy'   | ADAM | 'accuracy' | 0.005 | 20 | 0.82385    | 0.9952 | 0.75975    | 0.9960
2  | SIGMOID | SOFTMAX | 'softmax_categorical_crossentropy'   | ADAM | 'accuracy' | 0.005 | 20 | 0.77549    | 0.9913 | 0.77621    | 0.9899
3  | SIGMOID | SOFTMAX | CUSTOM (called x-entropy as well...) | ADAM | 'accuracy' | 0.005 | 20 | 2229.34668 | 0.9992 | 1486.91815 | 0.9992
4  | TANH    | SOFTMAX | CUSTOM (called x-entropy as well...) | ADAM | 'accuracy' | 0.005 | 20 | 1597.01648 | 0.9993 | 2822.24431 | 0.9994
5  | RELU    | SOFTMAX | CUSTOM (called x-entropy as well...) | ADAM | 'accuracy' | 0.005 | 20 | NaN        | 0.0017 | NaN        | 0.0017
6  | RELU    | SOFTMAX | 'softmax_categorical_crossentropy'   | ADAM | 'accuracy' | 0.005 | 20 | 0.67621    | 0.9931 | 0.74155    | 0.9943
7  | RELU    | SOFTMAX | 'softmax_categorical_crossentropy'   | ADAM | 'accuracy' | 0.005 | 20 | 0.31490    | 0.9984 | 0.31498    | 0.9983
8  | RELU    | SOFTMAX | 'softmax_categorical_crossentropy'   | ADAM | 'accuracy' | 0.005 | 20 | 0.31532    | 0.9979 | 0.31498    | 0.9983


Notes:
 - Test nr. 7 is a 6 without the "imbalance fixing"
 - Test nr. 8 is a 8 without the dropout layer


### RESULTS:

Test Run | Fraud/Normal | Fraud Probability | Normal Probability | Match?
--- | --- | --- | --- | ---
1  | FRAUD  | 4.207121762078714 e-08  | 1.0                | NO
1  | NORMAL | 4.207121762078714 e-08  | 1.0                | YES
2  | FRAUD  | 2.3533789317298215 e-07 | 0.9999997615814209 | NO
2  | NORMAL | 2.3533789317298215 e-07 | 0.9999997615814209 | YES
3  | FRAUD  | 7.377072324743494 e-05  | 0.9999262094497681 | NO
3  | NORMAL | 7.377072324743494 e-05  | 0.9999262094497681 | YES
4  | FRAUD  | 1.223582057718886 e-05  | 0.9999877214431763 | NO
4  | NORMAL | 1.223582057718886 e-05  | 0.9999877214431763 | YES
5  | FRAUD  | NaN                     | NaN                | NO
5  | NORMAL | NaN                     | NaN                | YES
6  | FRAUD  | 0                       | 1                  | NO
6  | NORMAL | 0                       | 1                  | YES
7  | FRAUD  | 0                       | 1                  | NO
7  | NORMAL | 0                       | 1                  | YES
8  | FRAUD  | 0                       | 1                  | NO
8  | NORMAL | 0                       | 1                  | YES
