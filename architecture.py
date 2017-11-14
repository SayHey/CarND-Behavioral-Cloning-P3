# Define model architecture

from keras.models import Sequential
from keras.layers import Flatten, Dense

# Nvidia's "End to End Learning for Self-Driving Cars" architecture

def model_architecture(input_shape):

    model = Sequential()    
    
    # Normalization layer
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=input_shape))
    
    # Convolutional layers
    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(ELU())

    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", init='he_normal'))
    model.add(ELU())
    
    # Flatten layer
    model.add(Flatten())
    
    # Fully connected layers
    model.add(Dense(1164, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.2))

    model.add(Dense(100, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(50, init='he_normal'))
    model.add(ELU())
    model.add(Dropout(0.2))
    
    model.add(Dense(10, init='he_normal'))
    model.add(ELU())
    
    # Output layer
    model.add(Dense(1, init='he_normal'))
    
    return model

