import numpy as np
import tflearn

from tensorflow.examples.tutorials.mnist import input_data

# THE DNN MODEL!
from dnn_architecture import create_net_architecture

# ------------------------------------------------------------------------------
# ---------[ NETWORK: TRAINING ]------------------------------------------------
#
def train_dnn(model, data, labels, num_epochs=1, a_batch_size=128, verbose=True):
    # TRAIN NETWORK WITH LABELED DATA
    model.fit(  data, labels,                   # We use the whole (full) data set
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
    mnist = input_data.read_data_sets("../myMNIST/data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return trX, trY
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ THE PROGRAM! ]-----------------------------------------------------
data, labels = provide_training_labeled_data()
dnn_model = create_net_architecture()
dnn_model = train_dnn(dnn_model, data, labels, num_epochs=10)
dnn_model.save("sessions/model_20170421_1715.tfl")
# ------------------------------------------------------------------------------
