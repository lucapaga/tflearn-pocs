import numpy as np
import tflearn

from tensorflow.examples.tutorials.mnist import input_data

# WHEN IT COMES TO IMAGE PROCESSING
import matplotlib.image as mpimg


# ------------------------------------------------------------------------------
# ---------[ NETWORK: ARCHITECTURE ]--------------------------------------------
#
input_layer_neurons=784                 # INPUT NEURONS
p_keep_input=0.8                        # DROPOUT PARAMETERS FOR INPUT
first_h_layer_neurons=1568              # 1st HIDDEN LAYER NEURONS
second_h_layer_neurons=1568             # 2nd HIDDEN LAYER NEURONS
p_keep_hidden=0.5                       # DROPOUT PARAMETERS FOR HIDDEN LAYERS
output_layer_neurons=10                 # OUTPUT NEURONS
training_speed=0.001                    # TRAINING SPEED
weight_init_gauss_std_dev_value=0.01    # GAUSSIAN DISTRIBUTION used to INIT WEIGHTS
#
def create_net_architecture():
    net = tflearn.input_data(shape=[None, input_layer_neurons]) # nr. X INPUTS      (28x28 PIXELS = 784 features)
                                                                # 'None' stands for an unknown dimension, so we can change the total number of samples that are processed in a batch
                                                                # we will process samples 'per batch' to save memory
                                                                # --
    net = tflearn.layers.core.dropout(net, p_keep_input)
    net = tflearn.fully_connected(net, first_h_layer_neurons,   # 1st HIDDEN LAYER
                                    activation='relu')          #                 , activation='ReLU'
    net = tflearn.layers.core.dropout(net, p_keep_hidden)
    net = tflearn.fully_connected(net, second_h_layer_neurons,  # 2nd HIDDEN LAYER
                                    activation='relu')          #                 , activation='ReLU'
    net = tflearn.layers.core.dropout(net, p_keep_hidden)
    net = tflearn.fully_connected(net, output_layer_neurons,    # OUTPUT LAYER
                                  activation='linear')          #                 , activation='linear'
                                                                # --
    #
    #
    net = tflearn.regression(net,                               # RMSProp OPTIMIZER
                             optimizer=tflearn.optimizers.RMSProp(
                                    learning_rate=training_speed))
    #
    # FROM NETWORK TO MODEL
    model = tflearn.DNN(net)

    return model
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ NETWORK: TRAINING ]------------------------------------------------
#
def train_dnn(model, data, labels, num_epochs=1, a_batch_size=128, verbose=True):
    # TRAIN NETWORK WITH LABELED DATA
    data, labels = provide_training_labeled_data()
    model.fit(  data, labels,               # We use the whole (full) data set
                n_epoch=num_epochs,         # the network will see all data 10 (n_epoch) times
                batch_size=a_batch_size,    # the above-mentioned batch-size that lets us save memory
                show_metric=verbose     )   # kind-a log for each step of each epoch

    return model                            # TRAINED MODEL, READY TO PROOF
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ TRAINING: LABELED DATA ]-------------------------------------------
#
def provide_training_labeled_data():
    # DOWNLOAD/PROVIDE LABELED DATA TO USE FOR TRAINING
    mnist = input_data.read_data_sets("data/", one_hot=True)
    trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    return trX, trY
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ THE PROGRAM! ]-----------------------------------------------------
data, labels = provide_training_labeled_data()
dnn_model = create_net_architecture()
dnn_model = train_dnn(dnn_model, data, labels, num_epochs=10)
dnn_model.save("sessions/model_20170421_1630.tfl")
# ------------------------------------------------------------------------------
