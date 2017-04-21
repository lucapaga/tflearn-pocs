import numpy as np
import tflearn

# REST SERVER
from flask import Flask, jsonify, abort, request, make_response, url_for

# CREATE VECTORS FROM CSV
from tflearn.data_utils import load_csv


# ------------------------------------------------------------------------------
# ---------[ NETWORK: ARCHITECTURE ]--------------------------------------------
#
def create_net_architecture():
    net = tflearn.input_data(shape=[None, 6])              # nr. 6 INPUTS      (6 COLUMNS = 6 features)
                                                           # 'None' stands for an unknown dimension, so we can change the total number of samples that are processed in a batch
                                                           # we will process samples 'per batch' to save memory
                                                           # --
    net = tflearn.fully_connected(net, 320)                # 1st HIDDEN LAYER, activation='linear'  (intially: 32 NEURONS)
    net = tflearn.fully_connected(net,  64)                # 2st HIDDEN LAYER, activation='linear'  (intially: 32 NEURONS)
    net = tflearn.fully_connected(net,   2,                # OUTPUT LAYER                           (intially:  2 NEURONS)
                                  activation='softmax')    #                 , activation='softmax'
                                                           # --
    #
    #
    net = tflearn.regression(net)                          # Estimator Layer (so called in TFLearn)
                                                           # 'Applies a regression (linear or logistic) to the provided input.'
                                                           #   - optimizer='adam'                   [Adaptive Moment Estimation]
                                                           #   - loss='categorical_crossentropy'    [LOSS FUNCTION]
                                                           #   - learning_rate=0.001
                                                           #   - batch_size=64
    return net
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ---------[ NETWORK: TRAINING ]------------------------------------------------
#
dnn_model = None
#
def train_dnn(data, labels):
    net = create_net_architecture()
    model = tflearn.DNN(net)

    # TRAIN NETWORK WITH LABELED DATA
    data, labels = provide_training_labeled_data()
    model.fit(  data, labels,           # We use the whole (full) data set
                n_epoch=10,             # the network will see all data 10 (n_epoch) times
                batch_size=16,          # the above-mentioned batch-size that lets us save memory
                show_metric=True    )   # kind-a log for each step of each epoch

    return model                        # TRAINED MODEL, READY TO PROOF
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ NETWORK: PREDICTION ]----------------------------------------------
#
def use_dnn(pclass, name, sex, age, sibsp, parch, ticket, fare):
    to_ignore = [1, 6]
    model = dnn_model

    #          ----------------------------------------------------------------------------------------
    #          | PCLASS | NAME                  | SEX      | AGE | SIBSP | PARCH | TICKET | FARE      |
    dicaprio = [ 3,       'Jack Dawson',          'male',    19,   0,      0,      'N/A',     5.0000  ]     # STATIC SAMPLES: BENCHMARK
    winslet =  [ 1,       'Rose DeWitt Bukater',  'female',  17,   1,      2,      'N/A',   100.0000  ]     # STATIC SAMPLES: BENCHMARK
    paga =     [ 3,       'Luca Paganelli',       'male',    38,   0,      1,      'N/A',     8.0000  ]     # STATIC SAMPLES: BENCHMARK
    claire =   [ 3,       'Chiara Salomoni',      'female',  34,   0,      1,      'N/A',     8.0000  ]     # STATIC SAMPLES: BENCHMARK
    # -------------------------------------------------------------------------------------------------
    someone =  [ pclass,  name,                   sex,       age,  sibsp,  parch,  ticket,      fare  ]     # ACTUAL REQUEST

    # DATA PREPROCESSING
    dicaprio, winslet, paga, claire, someone = preprocess([dicaprio, winslet, paga, claire, someone], to_ignore)

    # NETWORK ELABORATION
    pred = model.predict([dicaprio, winslet, paga, claire, someone])

    print("DiCaprio Surviving Rate:", pred[0][1])
    print("Winslet  Surviving Rate:", pred[1][1])
    print("Paga     Surviving Rate:", pred[2][1])
    print("Claire   Surviving Rate:", pred[3][1])

    # RESULT
    print("API Req. Surviving Rate:", pred[4][1])

    return pred[4][1]
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ TRAINING: LABELED DATA ]-------------------------------------------
#
def provide_training_labeled_data():
    # DOWNLOAD/PROVIDE
    #titanic.download_dataset('../tutorial/data/titanic_dataset.csv')    # where to store the CSV?

    # Load CSV file, indicate that the first column represents labels
    data, labels = load_csv('../tutorial/data/titanic_dataset.csv',     # the input file
                            target_column=0,                # our 'LABELS' are located in the first column (id: 0)
                            categorical_labels=True,        # ?
                            n_classes=2                 )   # 'survived' or 'not' (LABEL values)

    # we make the assumption that 'name' field will not be very useful in our task,
    # because we estimate that a passenger name and his chance of surviving are not correlated.
    # With such thinking, we discard 'name' (2, 3rd) and 'ticket' (7, 8th) fields.
    # As during load the target_column is "removed" than 'name' is at '1' and 'ticket' at '6'.
    to_ignore=[1, 6]

    # Preprocess data
    data = preprocess(data, to_ignore)

    return data, labels
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ DATA PREPROCESSING ]-----------------------------------------------
#
def preprocess(data, columns_to_ignore):
    # Sort by descending id and delete columns (??)
    for id in sorted(columns_to_ignore, reverse=True):
        [r.pop(id) for r in data]
    for i in range(len(data)):
      # We need to CONVERT all our data to NUMERICAL VALUES,
      # because a neural network model can only perform operations over numbers
      # --
      # In this simple case, we will just assign '0' to males and '1' to females.
      data[i][1] = 1. if data[i][1] == 'female' else 0.
    return np.array(data, dtype=np.float32)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ REST API: CONFIGURATION ]------------------------------------------
context_path='/tflrn/tutorial/dnn/api/v1.0'

app = Flask(__name__, static_url_path = "")

@app.errorhandler(400)
def not_found(error):
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify( { 'error': 'Not found' } ), 404)

# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ REST API: ROUTES ]-------------------------------------------------
#
@app.route(context_path + '/survives/<int:pclass>/<string:name>/<string:sex>/<int:age>/<int:sibsp>/<int:parch>/<string:ticket>/<float:fare>', methods = ['GET'])
#@auth.login_required
def guess_number(pclass, name, sex, age, sibsp, parch, ticket, fare):
    print('[' + context_path + '/survives] --------------------------------------')
    print('[' + context_path + '/survives]                     PRICE CLASS: ' + str(pclass) )
    print('[' + context_path + '/survives]                            NAME: ' + name        )
    print('[' + context_path + '/survives]                             SEX: ' + sex         )
    print('[' + context_path + '/survives]                             AGE: ' + str(age)    )
    print('[' + context_path + '/survives]  NR. OF SIBLINGS/SPOUSES ABOARD: ' + str(sibsp)  )
    print('[' + context_path + '/survives]  NR. OF PARENTS/CHILDREN ABOARD: ' + str(parch)  )
    print('[' + context_path + '/survives]                      TICKET NR.: ' + ticket      )
    print('[' + context_path + '/survives]                            FARE: ' + str(fare)   )
    print('[' + context_path + '/survives]   ...... EVALUATING .................................')
    guessed_val = use_dnn(pclass, name, sex, age, sibsp, parch, ticket, fare);
    print('[' + context_path + '/survives]          PROBABILITY TO SURVIVE: ' + str(guessed_val))
    print('[' + context_path + '/survives] ---<END>---------------------------------------------')
    return jsonify( { 'p2s': str(guessed_val) } )
# ------------------------------------------------------------------------------



# ------------------------------------------------------------------------------
# ---------[ REST API: SERVER ]-------------------------------------------------
#
if __name__ == '__main__':
    data, labels = provide_training_labeled_data()
    dnn_model = train_dnn(data, labels)
    app.run(debug = True)
# ------------------------------------------------------------------------------
