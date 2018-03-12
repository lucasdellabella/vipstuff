import numpy as np
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Dropout, Flatten
from keras.layers.normalization import BatchNormalization

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
    X_test = np.load(X_test_file)
    y_test = np.load(y_test_file)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

def vgg16_regression_model(label_dimension):
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
    '''
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

    model.add(Flatten())
    '''

    # Add Fully Connected Layer
    model.add(Dense(4096, activation='relu', input_shape=(25088,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(label_dimension)) # For Imagenet 1000 item classification

    # sgd = SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model

###### MAIN CODE ###### 
label_dimension = 10 
batch_size = 512 
epochs = 200

# Load our model, with first 4 conv blocks frozen

(X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_data()

print(X_train.shape)

model = vgg16_regression_model(label_dimension)

print(model.summary())

# Start Fine-tuning
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, y_valid),
          )

model.save('imagenet_models\\split_model_TOP.h5')

# Make predictions
predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)

print(predictions_valid)
