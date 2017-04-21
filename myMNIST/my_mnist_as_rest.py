import numpy as np
import tflearn

# REST SERVER
from flask import Flask, jsonify, abort, request, make_response, url_for

# CREATE VECTORS FROM CSV
from tflearn.data_utils import load_csv

# WHEN IT COMES TO IMAGE PROCESSING
import matplotlib.image as mpimg

from tensorflow.examples.tutorials.mnist import input_data


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
# ---------[ NETWORK: PREDICTION ]----------------------------------------------
#
def use_dnn(input_data):
    model = dnn_model
    pred = model.predict(input_data)
    return pred
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ---------[ REST API: CONFIGURATION ]------------------------------------------
context_path='/tflrn/mymnist/dnn/api/v1.0'

app = Flask(__name__, static_url_path = "")

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ REST API: SUPPORT ]------------------------------------------------
#
def load_image(image_file_name):
    img = mpimg.imread('../MY_data/' + image_file_name)
    img = img[:,:,0]        # slicing: picking only one channel of RGB (black'n'white!)
    img = img.flatten('C')  # from matrix to vector
    return img
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ REST API: ROUTES ]-------------------------------------------------
#
@app.route(context_path + '/divine/<string:image_name>', methods = ['GET'])
#@auth.login_required
def guess_number(image_name):
    print('[' + context_path + '/divine]  -----------------------------------------------------')
    print('[' + context_path + '/divine]          IMAGE NAME: ' + image_name                    )
    print('[' + context_path + '/divine]   ...... EVALUATING ..................................')

    input_data = load_image(image_name)
    print('[' + context_path + '/divine]              IMAGE DATA: ' , input_data                )

    guessed_vector = use_dnn([input_data])
    print('[' + context_path + '/divine]              ELABORATED: ' , guessed_vector            )

    guessed_valz = np.argmax(guessed_vector)
    print('[' + context_path + '/divine]    REQUESTED IMAGE IS A: ' , guessed_valz              )

    return jsonify( { 'p2s': str(guessed_valz) } )
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ---------[ REST API: SERVER ]-------------------------------------------------
#
if __name__ == '__main__':
#    data, labels = provide_training_labeled_data()
    dnn_model = create_net_architecture()
#    train_dnn(dnn_model, data, labels)
    dnn_model.load('sessions/model_20170421_1630.tfl')
    app.run(debug = True)
# ------------------------------------------------------------------------------
