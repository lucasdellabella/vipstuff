import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras.applications.vgg16 import preprocess_input, decode_predictions
from keras.backend import set_session
import tensorflow as tf

config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

def load_data():
    file_dir = 'matrixized-data/'
    X_train_file = file_dir + 'X_train.npy'
    y_train_file = file_dir + 'y_train.npy'
    X_valid_file = file_dir + 'X_valid.npy'
    y_valid_file = file_dir + 'y_valid.npy'
    X_test_file = file_dir + 'X_test.npy'
    y_test_file = file_dir + 'y_test.npy'

    X_train = np.load(X_train_file).astype(np.float16)
    X_valid = np.load(X_valid_file).astype(np.float16)
    X_test = np.load(X_test_file).astype(np.float16)

    # A: Center samples around 0
    #VGG_TRAIN_MEANS = np.array([104, 117, 124], dtype='uint8')
    #X_train -= VGG_TRAIN_MEANS
    #X_valid -= VGG_TRAIN_MEANS
    #X_test -= VGG_TRAIN_MEANS

    # B: Center samples around 0
    df = 'channels_last'
    mode = 'tf'
    X_train = preprocess_input(X_train, data_format=df, mode=mode)
    X_valid = preprocess_input(X_valid, data_format=df, mode=mode)
    X_test = preprocess_input(X_test, data_format=df, mode=mode)

    y_train = np.load(y_train_file)
    y_valid = np.load(y_valid_file)
    y_test = np.load(y_test_file)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def vgg16_regression_model(img_rows, img_cols, channel=1, num_classes=None):
    """VGG 16 Regression Model for Keras

    Model Schema is based on
    https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

    ImageNet Pretrained Weights
    https://drive.google.com/file/d/0Bz7KyqmuGsilT0J5dmRCM0ROVHc/view?usp=sharing

    Parameters:
      img_rows, img_cols - resolution of inputs
      channel - 1 for grayscale, 3 for color
      num_classes - number of categories for our classification/regression task
    """
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(img_rows, img_cols, channel)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # Add Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax')) # For Imagenet 1000 item classification

    # Loads ImageNet pre-trained data
    model.load_weights('imagenet_models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')

    # Truncate and replace softmax layer with a normal Dense layer without softmax activation
    # to allow for regressional training
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    model.add(Dense(num_classes))

    for i in range(5):
        model.layers.pop()

    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []

    # Uncomment below to set the dense layers and last conv block to be fine tunable
    #for layer in model.layers[:-13]:
        #layer.trainable = False

    model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model

if __name__ == "__main__":
    ###### MAIN CODE ######
    img_rows, img_cols = 224, 224 # Resolution of images
    num_channels = 3
    num_classes = 10
    batch_size = 16
    epochs = 25

    # Load our model, with first 4 conv blocks frozen
    model = vgg16_regression_model(img_rows, img_cols, num_channels, num_classes)

    print(model.summary())

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data()

    VGG_TRAIN_MEANS = np.array([104, 117, 124], dtype='uint8')

    X_train -= VGG_TRAIN_MEANS
    X_valid -= VGG_TRAIN_MEANS
    X_test -= VGG_TRAIN_MEANS

    # Start Fine-tuning
    #model.fit(X_train, y_train,
              #batch_size=batch_size,
              #epochs=epochs,
              #shuffle=True,
              #verbose=1,
              #validation_data=(X_valid, y_valid),
              #)

    #model.save('imagenet_models/trained_regression_model_25epoch.h5')

    # Make predictions
    predictions_train = model.predict(X_train, batch_size=batch_size, verbose=1)
    predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
    predictions_test = model.predict(X_test, batch_size=batch_size, verbose=1)

    np.save('matrixized-data\\features_train.npy', predictions_train)
    np.save('matrixized-data\\features_valid.npy', predictions_valid)
    np.save('matrixized-data\\features_test.npy', predictions_test)
