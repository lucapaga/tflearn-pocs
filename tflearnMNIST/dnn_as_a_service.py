import numpy as np
import tflearn

# REST SERVER
from flask import Flask, jsonify, abort, request, make_response, url_for

# WHEN IT COMES TO IMAGE PROCESSING
import matplotlib.image as mpimg

# THE DNN MODEL!
from dnn_architecture import create_net_architecture

# INIT
#dnn_model = None

# ------------------------------------------------------------------------------
# ---------[ NETWORK: PREDICTION ]----------------------------------------------
#
def use_dnn(input_data):
    model = dnn_model
    pred  = model.predict(input_data)
    return pred
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ---------[ REST API: CONFIGURATION ]------------------------------------------
context_path='/tflrn/tflmnist/dnn/api/v1.0'

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
    dnn_model = create_net_architecture()
    dnn_model.load('sessions/bl20170421/model_20170421_1715.tfl')
    app.run(debug = True)
# ------------------------------------------------------------------------------
