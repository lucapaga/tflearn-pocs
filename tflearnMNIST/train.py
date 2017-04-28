import numpy as np
import tflearn

from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.image as mpimg

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
                validation_set=0.1,             # 10% of the overall set is used for validation and not for training
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


#----- JUST TO CHECK ON MY HANDWRITTEN DIGITS! ---------------------------------
def load_image(image_file_name):
    img = mpimg.imread('../MY_data/' + image_file_name)
    img = img[:,:,0]        # slicing: picking only one channel of RGB (black'n'white!)
    img = img.flatten('C')  # from matrix to vector
    return img

def use_dnn(input_data):
    model = dnn_model
    pred  = model.predict(input_data)
    return pred
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# ---------[ THE PROGRAM! ]-----------------------------------------------------
dnn_model = create_net_architecture()
session_directory = "bl201704281722"
do_train = True
if do_train:
    data, labels = provide_training_labeled_data()
    dnn_model = train_dnn(dnn_model, data, labels, num_epochs=20, a_batch_size=1024)
    dnn_model.save("sessions/" + session_directory + "/model_20170428_1615.tfl")
else:
    dnn_model.load("sessions/" + session_directory + "/model_20170428_1615.tfl")


#----- JUST TO CHECK ON MY HANDWRITTEN DIGITS! ---------------------------------
for image_name in [ 'aDigit_UNO.png', 'aDigit_UNOw.png', 'aDigit_DUE.png', 'aDigit_TRE.png', 'aDigit_QUATTRO.png', 'aDigit_CINQUE.png', 'aDigit_SEI.png', 'aDigit_SETTE.png', 'aDigit_OTTO.png', 'aDigit_NOVE.png' ]:
    input_data = load_image(image_name)
    guessed_vector = use_dnn([input_data])
    guessed_valz = np.argmax(guessed_vector)
    print('----------------------------------------------------------------------------------------')
    print('----------------------------------------------------------------------------------------')
    print(' VERIFICATION  ->  IMAGE="' + image_name + '"  |  PREDICTION="' + str(guessed_valz) + '"')
    print('----------------------------------------------------------------------------------------')
    print(' ARRAY: ', guessed_vector)

# ------------------------------------------------------------------------------
