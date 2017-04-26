import numpy as np
import tflearn

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
    input_layer = tflearn.input_data(       shape=[None,    input_layer_neurons],                         name='input'            )
    dense1      = tflearn.fully_connected(  input_layer,    first_h_layer_neurons,  activation='sigmoid', name='dense1',
                                                            weights_init=tflearn.initializations.truncated_normal(stddev=0.15)    )
    dense2      = tflearn.fully_connected(  dense1,         second_h_layer_neurons, activation='sigmoid', name='dense2',
                                                            weights_init=tflearn.initializations.truncated_normal(stddev=0.15)    )
    dense3      = tflearn.fully_connected(  dense2,         third_h_layer_neurons,  activation='sigmoid', name='dense3',
                                                            weights_init=tflearn.initializations.truncated_normal(stddev=0.15)    )
    pre_smax    = tflearn.dropout(          dense3,         p_keep_hidden,                                name='dense3_dropout'   )
    softmax     = tflearn.fully_connected(  pre_smax,       output_layer_neurons,   activation='softmax',
                                                            weights_init=tflearn.initializations.truncated_normal(stddev=0.15)    )
    #
    regression  = tflearn.regression(       softmax,
                                            optimizer='adam',
                                            learning_rate=training_speed,
                                            loss='categorical_crossentropy'    )
    #
    # FROM NETWORK TO MODEL
    model = tflearn.DNN(regression)

    return model
# ------------------------------------------------------------------------------
