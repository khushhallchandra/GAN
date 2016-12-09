from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import SGD
from keras.layers.normalization import BatchNormalization

def generative(nb_filters=1024, kernel_size, s, input_shape):
    model = Sequential()
    
    model.add(Dense(output_dim=nb_filters*s*s, input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Reshape([s, s, nb_filters]))

    # conv1
    model.add(Convolution2D(nb_filters/2, kernel_size[0], kernel_size[1], border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # conv2
    model.add(Convolution2D(nb_filters/4, kernel_size[0], kernel_size[1], border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # conv3
    model.add(Convolution2D(nb_filters/8, kernel_size[0], kernel_size[1], border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # conv4
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1], border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    return model
