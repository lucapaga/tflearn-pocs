import numpy as np
import tflearn
import pandas as pd
import matplotlib.pyplot as plt

#from tensorflow.examples.tutorials.mnist import input_data

# THE DNN MODEL!
from dnn_architecture import create_net_architecture

# ------------------------------------------------------------------------------
# ---------[ NETWORK: TRAINING ]------------------------------------------------
#
def train_dnn(model, data, labels, tData, tLabels, num_epochs=1, a_batch_size=128, verbose=True):
    # TRAIN NETWORK WITH LABELED DATA
    model.fit(  data, labels,                   # We use the whole (full) data set
                validation_set=(tData, tLabels),
                n_epoch=num_epochs,             # the network will see all data 10 (n_epoch) times
                batch_size=a_batch_size,        # the above-mentioned batch-size that lets us save memory
                show_metric=verbose         )   # kind-a log for each step of each epoch

    return model                                # TRAINED MODEL, READY TO PROOF
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ TRAINING: LABELED DATA ]-------------------------------------------
#
def provide_training_labeled_data():
    # DOWNLOAD/PROVIDE LABELED DATA TO USE FOR TRAINING
    print('[DEBUG] Loading CSV ...')
    df = pd.read_csv("../creditcard_data/creditcard.csv")

    ''' OLD APPROACH
    print('[DEBUG] Using a 95% of the full set as TRAINING SET')
    dfTRAIN = df.sample(frac=.80)
    print('[DEBUG] Using a 10% of the full set as TEST SET (thus overlapping)')
    dfTEST  = df.sample(frac=.01)

    print('[DEBUG] Dividing into FEATURES and LABELS')
    data = dfTRAIN.iloc[:, 0:len(dfTRAIN.columns) - 1]
    labels = dfTRAIN.iloc[:, len(dfTRAIN.columns) - 1:len(dfTRAIN.columns)]
    tData = dfTEST.iloc[:, 0:len(dfTEST.columns) - 1]
    tLabels = dfTEST.iloc[:, len(dfTEST.columns) - 1:len(dfTEST.columns)]
    '''

    # Adding boolean column that marks if NORMAL
    df.loc[df.Class == 0, 'Normal'] = 1
    df.loc[df.Class == 1, 'Normal'] = 0
    # Renaming "Class" column to make it a boolean column that marks if FRAUD
    df = df.rename(columns={'Class': 'Fraud'})
    # The two data-sets
    Fraud = df[df.Fraud == 1]
    Normal = df[df.Normal == 1]
    # Get an 80% of whole data-set for TRAINING
    X_train = Fraud.sample(frac=0.8)
    count_Frauds = len(X_train)
    X_train = pd.concat([X_train, Normal.sample(frac = 0.8)], axis = 0)
    # Get the remainder as TEST set
    X_test = df.loc[~df.index.isin(X_train.index)]
    # Get labels, first the "Fraud" column ...
    y_train = X_train.Fraud
    y_test = X_test.Fraud
    # ... than the "Normal" column
    y_train = pd.concat([y_train, X_train.Normal], axis=1)
    y_test = pd.concat([y_test, X_test.Normal], axis=1)
    # Exclude label data
    X_train = X_train.drop(['Fraud','Normal'], axis = 1)
    X_test = X_test.drop(['Fraud','Normal'], axis = 1)

    '''
    Due to the imbalance in the data, ratio will act as an equal weighting system for our model.
    By dividing the number of transactions by those that are fraudulent, ratio will equal the value that when multiplied
    by the number of fraudulent transactions will equal the number of normal transaction.
    Simply put: # of fraud * ratio = # of normal
    '''
    #ratio = len(X_train)/count_Frauds
    #y_train.Fraud *= ratio
    #y_test.Fraud *= ratio

    #Transform each feature in features so that it has a mean of 0 and standard deviation of 1;
    #this helps with training the neural network.
    features = X_train.columns.values
    for feature in features:
        mean, std = df[feature].mean(), df[feature].std()
        X_train.loc[:, feature] = (X_train[feature] - mean) / std
        X_test.loc[:, feature] = (X_test[feature] - mean) / std

    data    = X_train
    labels  = y_train
    tData   = X_test
    tLabels = y_test

    print('[DEBUG] TRAIN |        Volume (SAMPLES): ', str(len(data.values)))           # 30 inputs
    print('[DEBUG] TRAIN |        Volume (FRAUDIZ): ', str(len(data[df.Fraud == 1].values)))           # 30 inputs
    print('[DEBUG] TRAIN |        Volume (NON-FRZ): ', str(len(data[df.Normal == 1].values)))           # 30 inputs
    print('[DEBUG] TRAIN |  DATA Columns (' , str(len(data.columns)), '): ', data.columns.values)           # 30 inputs
    print('[DEBUG] TRAIN | LABEL Columns (' , str(len(labels.columns)), '): ', labels.columns.values)       # 1  output (binary classifier)
    print('[DEBUG]  TEST |        Volume (SAMPLES): ', str(len(tData.values)))           # 30 inputs
    print('[DEBUG] TRAIN |        Volume (FRAUDIZ): ', str(len(tData[df.Fraud == 1].values)))           # 30 inputs
    print('[DEBUG] TRAIN |        Volume (NON-FRZ): ', str(len(tData[df.Normal == 1].values)))           # 30 inputs
    print('[DEBUG]  TEST |  DATA Columns (' , str(len(tData.columns)), '): ', tData.columns.values)           # 30 inputs
    print('[DEBUG]  TEST | LABEL Columns (' , str(len(tLabels.columns)), '): ', tLabels.columns.values)       # 1  output (binary classifier)

    return data.as_matrix(), labels.as_matrix(), tData.as_matrix(), tLabels.as_matrix()
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ THE PROGRAM! ]-----------------------------------------------------
data, labels, tData, tLabels = provide_training_labeled_data()
dnn_model = create_net_architecture()
dnn_model = train_dnn(dnn_model, data, labels, tData, tLabels, a_batch_size=2048, num_epochs=5)
dnn_model.save("sessions/model_20170426_1530.tfl")
# ------------------------------------------------------------------------------
