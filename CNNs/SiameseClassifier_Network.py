
import os
import PIL

import numpy as np
import numpy.random as rng
import matplotlib.pyplot as plt

from PIL import Image
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.models import Model
from sklearn.utils import shuffle
from keras.layers.core import Lambda, Flatten, Dense
from keras.regularizers import l2
from keras.layers.pooling import MaxPooling2D


def loadimgs(path, n = 0):
    '''
    path => Path of train directory or test directory
    '''

    X=[]
    y = []
    cat_dict = {}
    lang_dict = {}
    curr_y = n

    # we load every alphabet separately so we can isolate them later
    for alphabet in os.listdir(path):
        print("loading Data: " + alphabet)
        lang_dict[alphabet] = [curr_y, None]
        alphabet_path = os.path.join(path, alphabet)

        # every letter/category has it's own column in the array, so  load separately
        for letter in os.listdir(alphabet_path):
            cat_dict[curr_y] = (alphabet, letter)
            category_images = []
            letter_path = os.path.join(alphabet_path, letter)

            # read all the images in the current category
            for filename in os.listdir(letter_path):
                image_path = os.path.join(letter_path, filename)

#                image = imread(image_path)
                image = Image.open(image_path)
                image = image.resize((105, 105), PIL.Image.ANTIALIAS)

                category_images.append(image)
                y.append(curr_y)

            try:
                X.append(np.stack(category_images))
            # edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)

            curr_y += 1
            lang_dict[alphabet][1] = curr_y - 1

    y = np.vstack(y)
    X = np.stack(X)

    return X, y, lang_dict


def initialize_weights(shape, name=None):
    '''
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer weights with mean as 0.0 and standard deviation of 0.01
    '''
    return np.random.normal(loc=0.0, scale=1e-2, size=shape)


def initialize_bias(shape, name=None):
    '''
        The paper, http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
        suggests to initialize CNN layer bias with mean as 0.5 and standard deviation of 0.01
    '''
    return np.random.normal(loc=0.5, scale=1e-2, size=shape)


def get_siamese_model(input_shape):
    '''
        Model architecture based on the one provided in: http://www.cs.utoronto.ca/~gkoch/files/msc-thesis.pdf
    '''

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


def get_batch(batch_size, Xtrain, Xval, train_classes, val_classes, s="train"):
    '''
    Create batch of n pairs, half same class, half different class
    '''
    
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes

    n_classes, n_examples, w, h = X.shape

    # randomly sample several classes to use in the batch
    categories = rng.choice(n_classes, size=(batch_size,), replace=False)

    # initialize 2 empty arrays for the input image batch
    pairs = [np.zeros((batch_size, h, w, 1)) for i in range(2)]

    # initialize vector for the targets
    targets = np.zeros((batch_size,))

    # make one half of it '1's, so 2nd half of batch has same class
    targets[batch_size//2:] = 1
    for i in range(batch_size):
        category = categories[i]
        idx_1 = rng.randint(0, n_examples)
        pairs[0][i, :, :, :] = X[category, idx_1].reshape(w, h, 1)
        idx_2 = rng.randint(0, n_examples)

        # pick images of same class for 1st half, different for 2nd
        if i >= batch_size // 2:
            category_2 = category
        else:
            # add a random number to the category modulo n classes to ensure 2nd image has a different category
            category_2 = (category + rng.randint(1,n_classes)) % n_classes

        pairs[1][i, :, :, :] = X[category_2, idx_2].reshape(w, h, 1)

    return pairs, targets


def generate(batch_size, s="train"):
    '''
    a generator for batches, so model.fit_generator can be used. 
    '''
    
    while True:
        pairs, targets = get_batch(batch_size, s)
        yield (pairs, targets)


def make_oneshot_task(N, Xtrain, Xval, train_classes, val_classes, s="val", language=None):
    """Create pairs of test image, support set for testing N way one-shot learning. """
    if s == 'train':
        X = Xtrain
        categories = train_classes
    else:
        X = Xval
        categories = val_classes

    n_classes, n_examples, w, h = X.shape
    indices = rng.randint(0, n_examples, size=(N,))

    if language is not None:                                                            # if language is specified, select characters for that language
        low, high = categories[language]
        if N > high - low:
            raise ValueError("This language ({}) has less than {} letters".format(language, N))
        categories = rng.choice(range(low, high), size=(N,), replace=False)

    else: # if no language specified just pick a bunch of random letters
        categories = rng.choice(range(n_classes), size=(N,), replace=False)

    true_category = categories[0]
    ex1, ex2 = rng.choice(n_examples, replace=False, size=(2,))
    test_image = np.asarray([X[true_category, ex1, :, :]]*N).reshape(N, w, h, 1)
    support_set = X[categories, indices, :, :]
    support_set[0, :, :] = X[true_category, ex2]
    support_set = support_set.reshape(N, w, h, 1)
    targets = np.zeros((N,))
    targets[0] = 1
    targets, test_image, support_set = shuffle(targets, test_image, support_set)
    pairs = [test_image, support_set]

    return pairs, targets


def test_oneshot(model, Xtrain, Xval, train_classes, val_classes, N, k, s="val", verbose=0):
    """Test average N way oneshot learning accuracy of a siamese neural net over k one-shot tasks"""
    n_correct = 0
    if verbose:
        print("Evaluating model on {} random {} way one-shot learning tasks ... \n".format(k, N))
    for i in range(k):
        inputs, targets = make_oneshot_task(N, Xtrain, Xval, train_classes, val_classes, s)
        probs = model.predict(inputs)
        if np.argmax(probs) == np.argmax(targets):
            n_correct += 1
    percent_correct = (100.0 * n_correct / k)
    if verbose:
        print("Got an average of {}% {} way one-shot learning accuracy \n".format(percent_correct, N))
    return percent_correct


# ____ Plotting ________________________________________________________________________________________________________
def concat_images(X):
    '''Concatenates a bunch of images into a big matrix for plotting purposes.'''
    nc, h, w, _ = X.shape
    X = X.reshape(nc, h, w)
    n = np.ceil(np.sqrt(nc)).astype("int8")
    img = np.zeros((n*w, n*h))
    x = 0
    y = 0
    for example in range(nc):
        img[x*w:(x+1)*w, y*h:(y+1)*h] = X[example]
        y += 1
        if y >= n:
            y = 0
            x += 1
    return img


def plot_oneshot_task(pairs):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.matshow(pairs[0][0].reshape(105, 105), cmap='gray')
    img = concat_images(pairs[1])
    ax1.get_yaxis().set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax2.matshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# ______________________________________________________________________________________________________________________
# _____ Testing ________________________________________________________________________________________________________
# ______________________________________________________________________________________________________________________

def nearest_neighbour_correct(pairs, targets):
    '''returns 1 if nearest neighbour gets the correct answer for a one-shot task
        given by (pairs, targets)'''
    l2_distances = np.zeros_like(targets)
    for i in range(len(targets)):
        l2_distances[i] = np.sum(np.sqrt(pairs[0][i]**2 - pairs[1][i]**2))
    if np.argmin(l2_distances) == np.argmax(targets):
        return 1
    return 0


def test_nn_accuracy(N_ways, Xtrain, Xval, train_classes, val_classes, n_trials):
    '''Returns accuracy of NN approach '''
    print("Evaluating nearest neighbour on {} unique {} way one-shot learning tasks ...".format(n_trials, N_ways))

    n_right = 0

    for i in range(n_trials):
        pairs, targets = make_oneshot_task(N_ways, Xtrain, Xval, train_classes, val_classes, "val")
        correct = nearest_neighbour_correct(pairs, targets)
        n_right += correct
    return 100.0 * n_right / n_trials

