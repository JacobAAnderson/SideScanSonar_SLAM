
import os
import random

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm_notebook

from skimage.transform import resize

from keras.models import Model
from keras.layers import BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.preprocessing.image import img_to_array, load_img


# U-NET_________________________________________________________________________________________________________________
def get_unet(input_img, n_filters=16, dropout=0.5, batchnorm=True):

    # ___ contracting path _______________________________________________
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout*0.5)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)
    p4 = Dropout(dropout)(p4)

    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)

    # ___ expansive path __________________________________________________
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    model = Model(inputs=[input_img], outputs=[outputs])

    return model


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):

    # first layer
    x = Conv2D(filters=n_filters,
               kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal",
               padding="same")(input_tensor)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # second layer
    x = Conv2D(filters=n_filters,
               kernel_size=(kernel_size, kernel_size),
               kernel_initializer="he_normal",
               padding="same")(x)

    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x


# ___ Data Loader -_____________________________________________________________________________________________________
def get_data(path, im_height, im_width, train=True):
    ids = next(os.walk(path + "images"))[2]
    x = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    names = []

    if train:
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)

    print('\n\nGetting and resizing images ... \n')
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        # Load images

        names.append(id_)

        img = load_img(path + '/images/' + id_, grayscale=True)
        x_img = img_to_array(img)
        x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)

        x[n, ..., 0] = x_img.squeeze() / 255

        # Load masks
        if train:
            mask = img_to_array(load_img(path + '/masks/' + id_, grayscale=True))
            mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

            y[n] = mask / 255

    print('Data Loading Done!\n')
    print("%r Images Loaded\n\n" % len(x))

    if train:
        return x, y, names
    else:
        return x, names


# ___ Plot Training Results ____________________________________________________________________________________________
def plot_results(results):

    plt.figure(figsize=(8, 8))
    plt.title("Learning curve")
    plt.plot(results.history["loss"], label="loss")
    plt.plot(results.history["val_loss"], label="val_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.ylabel("log_loss")
    plt.legend()
    plt.show()


# ____ Plot Data _______________________________________________________________________________________________________
def plot_sample(x, y, preds, binary_preds, ix=None):

    if ix is None:
        ix = random.randint(0, len(x))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(x[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Sonar Image')
    ax[0].grid(False)

#    ax[1].imshow(y[ix].squeeze())
#    ax[1].set_title('Labeled Features')
#    ax[1].grid(False)

    ax[1].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Feature Prediction')
    ax[1].grid(False)

    ax[2].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    if has_mask:
        ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Feature Prediction binary')
    ax[2].grid(False)

    plt.show()


# ____ Plot Input Data _________________________________________________________________________________________________
def plot_input(x, y, ix=None):

    if ix is None:
        ix = random.randint(0, len(x))

    has_mask = y[ix].max() > 0

    plt.figure(figsize=(8, 8))
    plt.title('Sonar Image')
    plt.imshow(x[ix, ..., 0], cmap='gray')

    if has_mask:
        plt.contour(y[ix].squeeze(), colors='r', levels=[0.5])

    plt.grid(False)

    plt.show()


# ____ Plot Output Data _________________________________________________________________________________________________
def plot_output(x, y, ix=None):

    if ix is None:
        ix = random.randint(0, len(x))

    has_mask = y[ix].max() > 0

    plt.figure(figsize=(8, 8))
    plt.title('Sonar Image')
    plt.imshow(x[ix, ..., 0], cmap='plasma')

    if has_mask:
        plt.contour(y[ix].squeeze(), colors='k', levels=[0.5])

    plt.grid(False)

    plt.show()


# ____ Plot Prediction _________________________________________________________________________________________________
def plot_prediction(x, preds, binary_preds, ix=None):

    if ix is None:
        ix = random.randint(0, len(x))

    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(x[ix, ..., 0], cmap='gray')

    ax[1].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    ax[1].set_title('Feature Prediction')
    ax[1].grid(False)

    ax[2].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    ax[2].set_title('Feature Prediction binary')
    ax[2].grid(False)

    plt.show()
