import sys
from MARC.core.logger import Logger

logger = Logger()

if len(sys.argv) < 3:
    print('Please run as:')
    print('\tpython test.py', 'TRAIN_FILE', 'TEST_FILE', 'EMBEDDING_SIZE')
    exit()

TRAIN_FILE = sys.argv[1]
TEST_FILE = sys.argv[2]
EMBEDDER_SIZE = int(sys.argv[3])

print('====================================', 'PARAMS',
      '====================================')
print('TRAIN_FILE =', TRAIN_FILE)
print('TEST_FILE =', TEST_FILE)
print('EMBEDDER_SIZE =', EMBEDDER_SIZE)

###############################################################################
#   LOAD DATA
###############################################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from MARC.core.utils.geohash import bin_geohash


def get_trajectories(train_file, test_file, tid_col='tid',
                     label_col='label', geo_precision=8, drop=[]):
    file_str = "'" + train_file + "' and '" + test_file + "'"
    logger.log(Logger.INFO, "Loading data from file(s) " + file_str + "... ")
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)
    df = df_train.copy().append(df_test)
    tids_train = df_train[tid_col].unique()

    keys = list(df.keys())
    vocab_size = {}
    keys.remove(tid_col)

    for col in drop:
        if col in keys:
            keys.remove(col)
            logger.log(Logger.WARNING, "Column '" + col + "' dropped " +
                       "from input file!")
        else:
            logger.log(Logger.WARNING, "Column '" + col + "' cannot be " +
                       "dropped because it was not found!")

    num_classes = len(set(df[label_col]))
    count_attr = 0
    lat_lon = False

    if 'lat' in keys and 'lon' in keys:
        keys.remove('lat')
        keys.remove('lon')
        lat_lon = True
        count_attr += geo_precision * 5
        logger.log(Logger.INFO, "Attribute Lat/Lon: " +
                   str(geo_precision * 5) + "-bits value")

    for attr in keys:
        df[attr] = LabelEncoder().fit_transform(df[attr])
        vocab_size[attr] = max(df[attr]) + 1

        if attr != label_col:
            values = len(set(df[attr]))
            count_attr += values
            logger.log(Logger.INFO, "Attribute '" + attr + "': " +
                       str(values) + " unique values")

    logger.log(Logger.INFO, "Total of attribute/value pairs: " +
               str(count_attr))
    keys.remove(label_col)

    x = [[] for key in keys]
    y = []
    idx_train = []
    idx_test = []
    max_length = 0
    trajs = len(set(df[tid_col]))

    if lat_lon:
        x.append([])

    for idx, tid in enumerate(set(df[tid_col])):
        logger.log_dyn(Logger.INFO, "Processing trajectory " + str(idx + 1) +
                       "/" + str(trajs) + ". ")
        traj = df.loc[df[tid_col].isin([tid])]
        features = np.transpose(traj.loc[:, keys].values)

        for i in range(0, len(features)):
            x[i].append(features[i])

        if lat_lon:
            loc_list = []
            for i in range(0, len(traj)):
                lat = traj['lat'].values[i]
                lon = traj['lon'].values[i]
                loc_list.append(bin_geohash(lat, lon, geo_precision))
            x[-1].append(loc_list)

        label = traj[label_col].iloc[0]
        y.append(label)

        if tid in tids_train:
            idx_train.append(idx)
        else:
            idx_test.append(idx)

        if traj.shape[0] > max_length:
            max_length = traj.shape[0]

    if lat_lon:
        keys.append('lat_lon')
        vocab_size['lat_lon'] = geo_precision * 5

    one_hot_y = OneHotEncoder().fit(df.loc[:, [label_col]])

    x = [np.asarray(f) for f in x]
    y = one_hot_y.transform(pd.DataFrame(y)).toarray()
    logger.log(Logger.INFO, "Loading data from files " + file_str + "... DONE!")
    
    x_train = np.asarray([f[idx_train] for f in x])
    y_train = y[idx_train]
    x_test = np.asarray([f[idx_test] for f in x])
    y_test = y[idx_test]

    logger.log(Logger.INFO, 'Trajectories:  ' + str(trajs))
    logger.log(Logger.INFO, 'Labels:        ' + str(len(y[0])))
    logger.log(Logger.INFO, 'Train size:    ' + str(len(x_train[0]) / trajs))
    logger.log(Logger.INFO, 'Test size:     ' + str(len(x_test[0]) / trajs))
    logger.log(Logger.INFO, 'x_train shape: ' + str(x_train.shape))
    logger.log(Logger.INFO, 'y_train shape: ' + str(y_train.shape))
    logger.log(Logger.INFO, 'x_test shape:  ' + str(x_test.shape))
    logger.log(Logger.INFO, 'y_test shape:  ' + str(y_test.shape))

    return (keys, vocab_size, num_classes, max_length,
            x_train, x_test,
            y_train, y_test)


(keys, vocab_size,
 num_classes,
 max_length,
 x_train, x_test,
 y_train, y_test) = get_trajectories(train_file=TRAIN_FILE,
                                     test_file=TEST_FILE,
                                     tid_col='tid',
                                     label_col='label')


###############################################################################
#   PREPARING CLASSIFIER DATA
###############################################################################
from keras.preprocessing.sequence import pad_sequences


cls_x_train = [pad_sequences(f, max_length, padding='pre') for f in x_train]
cls_x_test = [pad_sequences(f, max_length, padding='pre') for f in x_test]
cls_y_train = y_train
cls_y_test = y_test


###############################################################################
#   CLASSIFIER
###############################################################################
from keras.models import Model
from keras.layers import Dense, LSTM, GRU, Dropout
from keras.initializers import he_uniform
from keras.regularizers import l1
from keras.optimizers import Adam
from keras.layers import Input, Add, Average, Concatenate, Embedding
from keras.callbacks import EarlyStopping
from MARC.core.utils.metrics import compute_acc_acc5_f1_prec_rec


CLASS_DROPOUT = 0.5
CLASS_HIDDEN_UNITS = 50
CLASS_LRATE = 0.001
CLASS_BATCH_SIZE = 64
CLASS_EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 30
BASELINE_METRIC = 'acc'
BASELINE_VALUE = 0.5


print('=====================================', 'OURS',
      '=====================================')

print('CLASS_DROPOUT =', CLASS_DROPOUT)
print('CLASS_HIDDEN_UNITS =', CLASS_HIDDEN_UNITS)
print('CLASS_LRATE =', CLASS_LRATE)
print('CLASS_BATCH_SIZE =', CLASS_BATCH_SIZE)
print('CLASS_EPOCHS =', CLASS_EPOCHS)
print('EARLY_STOPPING_PATIENCE =', EARLY_STOPPING_PATIENCE)
print('BASELINE_METRIC =', BASELINE_METRIC)
print('BASELINE_VALUE =', BASELINE_VALUE, '\n')


class EpochLogger(EarlyStopping):

    def __init__(self, metric='val_acc', baseline=0):
        super(EpochLogger, self).__init__(monitor='val_acc',
                                          mode='max',
                                          patience=EARLY_STOPPING_PATIENCE)
        self._metric = metric
        self._baseline = baseline
        self._baseline_met = False

    def on_epoch_begin(self, epoch, logs={}):
        print("===== Training Epoch %d =====" % (epoch + 1))

        if self._baseline_met:
            super(EpochLogger, self).on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs={}):
        pred_y_train = np.array(self.model.predict(cls_x_train))
        (train_acc,
         train_acc5,
         train_f1_macro,
         train_prec_macro,
         train_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_train,
                                                         pred_y_train,
                                                         print_metrics=True,
                                                         print_pfx='TRAIN')

        pred_y_test = np.array(self.model.predict(cls_x_test))
        (test_acc,
         test_acc5,
         test_f1_macro,
         test_prec_macro,
         test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                        pred_y_test,
                                                        print_metrics=True,
                                                        print_pfx='TEST')
        print(test_acc)
        print(test_acc5)
        print(test_f1_macro)
        print(test_prec_macro)
        print(test_rec_macro)

    def on_train_begin(self, logs=None):
        super(EpochLogger, self).on_train_begin(logs)

    def on_train_end(self, logs=None):
        if self._baseline_met:
            super(EpochLogger, self).on_train_end(logs)

inputs = []
embeddings = []

for idx, key in enumerate(keys):
    if key == 'lat_lon':
        i = Input(shape=(max_length, vocab_size[key]),
                  name='input_' + key)
        e = Dense(units=EMBEDDER_SIZE,
                  kernel_initializer=he_uniform(seed=1),
                  name='emb_' + key)(i)
    else:
        i = Input(shape=(max_length,),
                  name='input_' + key)
        e = Embedding(vocab_size[key],
                      EMBEDDER_SIZE,
                      input_length=max_length,
                      name='emb_' + key)(i)
    inputs.append(i)
    embeddings.append(e)

if len(embeddings) == 1:
    hidden_input = embeddings[0]
else:
    hidden_input = Concatenate(axis=2)(embeddings)

hidden_dropout = Dropout(CLASS_DROPOUT)(hidden_input)

rnn_cell = LSTM(units=CLASS_HIDDEN_UNITS, recurrent_regularizer=l1(0.02))(hidden_dropout)

rnn_dropout = Dropout(CLASS_DROPOUT)(rnn_cell)

softmax = Dense(units=num_classes,
                kernel_initializer=he_uniform(),
                activation='softmax')(rnn_dropout)

classifier = Model(inputs=inputs, outputs=softmax)
opt = Adam(lr=CLASS_LRATE)

classifier.compile(optimizer=opt,
                   loss='categorical_crossentropy',
                   metrics=['acc', 'top_k_categorical_accuracy'])

from keras.models import load_model
from keras.models import model_from_json

# load json and create model
json_file = open('MARC/MARC.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

classifier2 = model_from_json(loaded_model_json)
classifier2.load_weights('MARC/MARC_Weight.h5')

pred_y_test = np.array(classifier2.predict(cls_x_test))
(test_acc,
 test_acc5,
 test_f1_macro,
 test_prec_macro,
 test_rec_macro) = compute_acc_acc5_f1_prec_rec(cls_y_test,
                                                pred_y_test,
                                                print_metrics=True,
                                                print_pfx='TEST')
print(test_acc)
print(test_acc5)
print(test_f1_macro)
print(test_prec_macro)
print(test_rec_macro)