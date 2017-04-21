import numpy as np
import tflearn

# ------------------------------------------------------------------------------
# ---------[ NETWORK: ARCHITECTURE ]--------------------------------------------
#
input_layer_neurons=784                 # INPUT NEURONS
#p_keep_input=0.8                        # DROPOUT PARAMETERS FOR INPUT
first_h_layer_neurons=128               # 1st HIDDEN LAYER NEURONS
second_h_layer_neurons=256              # 2nd HIDDEN LAYER NEURONS
#p_keep_hidden=0.5                       # DROPOUT PARAMETERS FOR HIDDEN LAYERS
output_layer_neurons=10                 # OUTPUT NEURONS
training_speed=0.001                    # TRAINING SPEED
#weight_init_gauss_std_dev_value=0.01    # GAUSSIAN DISTRIBUTION used to INIT WEIGHTS
#
def create_net_architecture():
    input_layer = tflearn.input_data(       shape=[None,    input_layer_neurons],   name='input'            )
    dense1      = tflearn.fully_connected(  input_layer,    first_h_layer_neurons,  name='dense1'           )
    dense2      = tflearn.fully_connected(  dense1,         second_h_layer_neurons, name='dense2'           )
    softmax     = tflearn.fully_connected(  dense2,         output_layer_neurons,   activation='softmax'    )
    #
    regression  = tflearn.regression(       softmax,
                                            optimizer='adam',
                                            learning_rate=training_speed,
                                            loss='categorical_crossentropy'                                 )
    #
    # FROM NETWORK TO MODEL
    model = tflearn.DNN(regression)

    return model
# ------------------------------------------------------------------------------
