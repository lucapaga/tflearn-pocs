import numpy as np
import tflearn

# Download the Titanic dataset
from tflearn.datasets import titanic
titanic.download_dataset('data/titanic_dataset.csv')    # where to store the CSV?

# --------------------[ CSV SCHEMA ]--------------------------------------------
#  (0) survived
#  (1) pclass
#  (2) name             -> This data is useless
#  (3) sex
#  (4) age
#  (5) sibsp            - Number of Siblings/Spouses Aboard
#  (6) parch            - Number of Parents/Children Aboard
#  (7) ticket           -> This data is useless
#  (8) fare
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ DATA LOAD ]--------------------------------------------------------
# Load CSV file, indicate that the first column represents labels
from tflearn.data_utils import load_csv
data, labels = load_csv('data/titanic_dataset.csv',     # the input file
                        target_column=0,                # our 'LABELS' are located in the first column (id: 0)
                        categorical_labels=True,        # ?
                        n_classes=2                 )   # 'survived' or 'not' (LABEL values)

# ------------------------------------------------------------------------------
# ---------[ DATA PREPROCESSING ]-----------------------------------------------
#
# Preprocessing function
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

# we make the assumption that 'name' field will not be very useful in our task,
# because we estimate that a passenger name and his chance of surviving are not correlated.
# With such thinking, we discard 'name' (2, 3rd) and 'ticket' (7, 8th) fields.
# As during load the target_column is "removed" than 'name' is at '1' and 'ticket' at '6'.
to_ignore=[1, 6]

# Preprocess data
data = preprocess(data, to_ignore)
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ NETWORK ARCHITECTURE ]---------------------------------------------
#
net = tflearn.input_data(shape=[None, 6])              # nr. 6 INPUTS      (6 COLUMNS = 6 features)
                                                       # 'None' stands for an unknown dimension, so we can change the total number of samples that are processed in a batch
                                                       # we will process samples 'per batch' to save memory
                                                       # --
net = tflearn.fully_connected(net, 320)                # 1st HIDDEN LAYER  (intially: 32 NEURONS)
net = tflearn.fully_connected(net, 64)                 # 2st HIDDEN LAYER  (intially: 32 NEURONS)
net = tflearn.fully_connected(  net, 2,                # OUTPUT LAYER      (2 NEURONS)
                                activation='softmax')  #  'ACTIVATION FUNCTION' ONLY ON OUTPUT NEURONS (?)
                                                            # --
# ------------------------------------------------------------------------------
net = tflearn.regression(net)                               # Don't Know...
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ TRAINING ]---------------------------------------------------------
#
model = tflearn.DNN(net)                               # The model ('regression'?)
                                                       # --
# Start training (apply GRADIENT DESCENT algorithm)    # 'GRADIENT DESCENT' == 'regression' ??
model.fit(  data, labels,           # We use the whole (full) data set
            n_epoch=10,             # the network will see all data 10 (n_epoch) times
            batch_size=16,          # the above-mentioned batch-size that lets us save memory
            show_metric=True    )   # kind-a log for each step of each epoch
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# ---------[ PROOFING ]---------------------------------------------------------
#
#          ----------------------------------------------------------------------------------------
#          | PCLASS | NAME                  | SEX      | AGE | SIBSP | PARCH | TICKET | FARE      |
dicaprio = [ 3,       'Jack Dawson',          'male',    19,   0,      0,      'N/A',     5.0000  ]
winslet =  [ 1,       'Rose DeWitt Bukater',  'female',  17,   1,      2,      'N/A',   100.0000  ]
paga =     [ 3,       'Luca Paganelli',       'male',    38,   0,      1,      'N/A',     8.0000  ]
claire =   [ 3,       'Chiara Salomoni',      'female',  34,   0,      1,      'N/A',     8.0000  ]
# -- Preprocess data
dicaprio, winslet, paga, claire = preprocess([dicaprio, winslet, paga, claire], to_ignore)
# -- Predict surviving chances (class 1 results)
pred = model.predict([dicaprio, winslet, paga, claire])
print("DiCaprio Surviving Rate:", pred[0][1])
print("Winslet  Surviving Rate:", pred[1][1])
print("Paga     Surviving Rate:", pred[2][1])
print("Claire   Surviving Rate:", pred[3][1])
