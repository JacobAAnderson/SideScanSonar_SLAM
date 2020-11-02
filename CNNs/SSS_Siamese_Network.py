
import os
import PIL

import numpy as np

from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.models import Model
from keras.layers.core import Lambda, Flatten, Dense
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D


def loadimgs(path):

    x = []
    imageNames = []

    # read all the images in the current category
    for filename in os.listdir(path):

        image_path = os.path.join(path, filename)

        image = Image.open(image_path)
        image = image.resize((105, 105), PIL.Image.ANTIALIAS)

        try:
            x.append(image)
            imageNames.append(filename)

        # edge case  - last one
        except ValueError as e:
            print(e)
            print("error - image:", image_path)

    return x, imageNames


def initialize_weights(shape, name=None):
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None):
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):

    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)

    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10, 10), activation='relu', input_shape=input_shape, kernel_initializer=initialize_weights, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7, 7), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (4, 4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (4, 4), activation='relu', kernel_initializer=initialize_weights, bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid', kernel_regularizer=l2(1e-3), kernel_initializer=initialize_weights, bias_initializer=initialize_bias))

    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)

    # Add a customized layer to compute the absolute difference between the encodings
    l1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_layer([encoded_l, encoded_r])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', bias_initializer=initialize_bias)(l1_distance)

    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input, right_input], outputs=prediction)

    # return the model
    return siamese_net
