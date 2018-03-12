# -*- coding: utf-8 -*-

import numpy as np
import os
from keras.models import Sequential, load_model
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten

def load_data():
    file_dir = 'matrixized-data\\'
    X_train_file = file_dir + 'features_train.npy'
    y_train_file = file_dir + 'y_train.npy'
    X_valid_file = file_dir + 'features_valid.npy'
    y_valid_file = file_dir + 'y_valid.npy'
    X_test_file = file_dir + 'features_test.npy'
    y_test_file = file_dir + 'y_test.npy'

    X_train = np.load(X_train_file)
    y_train = np.load(y_train_file)
    X_valid = np.load(X_valid_file)
    y_valid = np.load(y_valid_file)
    X_test = np.load(X_train_file)
    y_test = np.load(y_train_file)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def vgg16_regression_model(num_classes):
    """
    VGG 16 Regression Model for Keras
    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color 
      num_classes - number of categories for our classification/regression task
    """
    model = Sequential()

    # Add Fully Connected Layer
    model.add(Dense(4096, activation='relu', input_shape=(25088,)))
    #model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(num_classes)) # For Imagenet 1000 item classification

    # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model

# Load our model
model = load_model('imagenet_models\\split_model_TOP.h5')

(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data()

# Make predictions
batch_size = 256

predictions = model.predict(X_train, batch_size=batch_size, verbose=1)
predictions.reshape(-1, 5, 2)
print(predictions[:10])

try:
    os.makedirs('train_prediction_txts')
except FileExistsError:
    pass

# txt prediction files are wrong type, check data preprocessing to see
for i, p in enumerate(predictions):
    np.savetxt('train_prediction_txts\\%s.prediction.txt' % str(i + 1).zfill(5), predictions[i].reshape(2,5).transpose())

scores = model.evaluate(X_test, y_test, batch_size=batch_size)
print('mse=%f, mae=%f, mape=%f' % (scores[0],scores[1],scores[2]))

#tf_session = K.get_session()
#preds = model.predict(X_test, batch_size=batch_size)
#mse = metrics.mean_squared_error(y_test, preds)
#mae = metrics.mean_absolute_error(y_test, preds)
#mape = metrics.mean_absolute_percentage_error(y_test, preds)
#print 'mse=%f, mae=%f, mape=%f' % (mse.eval(session=tf_session), mae.eval(session=tf_session), mape.eval(session=tf_session))
