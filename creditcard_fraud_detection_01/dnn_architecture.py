import numpy as np
import tflearn
import tensorflow as tf


# ------------------------------------------------------------------------------
# ---------[ NETWORK: ARCHITECTURE ]--------------------------------------------
#
def custom_loss_function(y_pred, y_true):
    return - tf.reduce_sum(y_true * tf.log(y_pred))

def custom_metric_function(y_pred, y_true, x):
    print('y_pred=', y_pred)
    print('y_true=', y_true)
    print('     x=', x)
    correct_prediction = tf.equal(tf.argmax(y_pred,1), tf.argmax(y_true,1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ NETWORK: ARCHITECTURE ]--------------------------------------------
#
input_layer_neurons=30                 # INPUT NEURONS
first_h_layer_neurons=15               # 1st HIDDEN LAYER NEURONS
second_h_layer_neurons=45              # 2nd HIDDEN LAYER NEURONS
third_h_layer_neurons=60              # 2nd HIDDEN LAYER NEURONS
p_keep_hidden=0.9                       # DROPOUT PARAMETERS FOR HIDDEN LAYERS
output_layer_neurons=2                 # OUTPUT NEURONS
training_speed=0.005                    # TRAINING SPEED
#weight_init_gauss_std_dev_value=0.01    # GAUSSIAN DISTRIBUTION used to INIT WEIGHTS
#
def create_net_architecture():
    input_layer = tflearn.input_data(
                        shape=[None,    input_layer_neurons],
                        name='input'
                    )
    dense1      = tflearn.fully_connected(
                        input_layer,
                        first_h_layer_neurons,
                        activation='sigmoid',
                        weights_init=tflearn.initializations.truncated_normal(stddev=0.15),
                        name='dense1'
                    )
    dense2      = tflearn.fully_connected(
                        dense1,
                        second_h_layer_neurons,
                        activation='sigmoid',
                        weights_init=tflearn.initializations.truncated_normal(stddev=0.15),
                        name='dense2'
                    )
    dense3      = tflearn.fully_connected(
                        dense2,
                        third_h_layer_neurons,
                        activation='sigmoid',
                        weights_init=tflearn.initializations.truncated_normal(stddev=0.15),
                        name='dense3'
                    )
    pre_smax    = tflearn.dropout(
                        dense3,
                        p_keep_hidden,
                        name='dense3_dropout'
                    )
    softmax     = tflearn.fully_connected(
                        pre_smax,
                        output_layer_neurons,
                        activation='softmax',
                        weights_init=tflearn.initializations.truncated_normal(stddev=0.15),
                        name='softmax_final'
                    )
    #
    # loss='softmax_categorical_crossentropy',
    # metric=custom_metric_function,
    regression  = tflearn.regression(
                        softmax,
                        loss=custom_loss_function,
                        optimizer='adam',
                        metric='accuracy',
                        learning_rate=training_speed
                    )
    #
    # FROM NETWORK TO MODEL
    model = tflearn.DNN(regression)

    return model
# ------------------------------------------------------------------------------
