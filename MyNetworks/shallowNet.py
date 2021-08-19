## INPUT => CONV => RELU => FC
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense
from keras.layers.core import  Flatten #Flattens multi dimensional to 1D
from keras import backend as K

class ShallowNet:
    
    @staticmethod
    def build(width, height, depth, classes) -> Sequential:
        model = Sequential()
        input_shape = (height,width, depth)

        if K.image_data_format() == 'channels_first':
            input_shape= (depth, height, width)

        model.add(Conv2D(32, (3,3), padding='same',
                input_shape= input_shape))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model

        