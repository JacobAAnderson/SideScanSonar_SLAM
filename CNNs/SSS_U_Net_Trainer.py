
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras.layers import Input
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from SSS_U_Net_Network import get_unet, get_data, plot_sample, plot_results

plt.style.use("ggplot")


# ___ Set some parameters _______________________________________________________________________

# ____ Seismic data from example ____
# im_width = 128
# im_height = 128
# border = 5
# seed = 2018
# bs = 32
# training_epochs = 100

# path_train = '/home/jake/PycharmProjects/SSS_Keras/tgs-salt-identification-challenge/train/'
# path_test = '/home/jake/PycharmProjects/SSS_Keras/tgs-salt-identification-challenge/test/'
# model_name = 'model-tgs-salt.h5'

# _______________________________________________________________________________________________
# ____ Side Scan Sonar Data ____
im_width = 256          # Image Width
im_height = 256         # Image Height
border = 5
seed = 2018
bs = 10                 # Batch Size
training_epochs = 100   # Number of Training Epochs
thresHold = 0.5         # Prediction Thresholds


path_train = '/home/jake/Data_Processed/SideScanSonarImages/U-Net_Features/train/'
path_test = '/home/jake/Data_Processed/SideScanSonarImages/U-Net_Features/test/'
model_name = 'SSS_U_Net_FeatureExtraction_Model_Barbados_CatalinaIS_LM2_SWA_Cam.h5'


# ___ Load Data _________________________________________________________________________________
X, y, names = get_data(path_train, im_height, im_width, train=True)                                                         # Get and resize train images and masks

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.01, random_state=2018)  # Split train and valid


# ___ Check if training data looks all right _________________________________
# import random
# ix = random.randint(0, len(X_train))
# has_mask = y_train[ix].max() > 0

# fig, ax = plt.subplots(1, 2, figsize=(20, 10))

# ax[0].imshow(X_train[ix, ..., 0], cmap='gray', interpolation='bilinear')

# if has_mask:
#    ax[0].contour(y_train[ix].squeeze(), colors='k', levels=[0.5])

# ax[0].set_title('Sonar')
# ax[1].imshow(y_train[ix].squeeze(), interpolation='bilinear', cmap='gray')
# ax[1].set_title('Features')

# plt.show()


# ___ Data Augmentation __________________________________________________________________________________________
data_gen_args = dict(horizontal_flip=True,
                     vertical_flip=True,
                     rotation_range=180,
                     width_shift_range=0.2,
                     height_shift_range=0.2,
                     zoom_range=[-2, 2])
#                     brightness_range=[-1, 1],

image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

image_generator = image_datagen.flow(X_train, seed=seed, batch_size=bs, shuffle=True)
mask_generator = mask_datagen.flow(y_train, seed=seed, batch_size=bs, shuffle=True)


train_generator = zip(image_generator, mask_generator)      # Just zip the two generators to get a generator that provides augmented images and masks at the same time


# ___ U-NET______________________________________________________________________________________________________________
input_img = Input((im_height, im_width, 1), name='img')

model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

model.summary()

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
    ModelCheckpoint(model_name, verbose=1, save_best_only=True, save_weights_only=True)
]


# ____ Run U-Net without Data Augmentation ____
# results = model.fit(X_train, y_train, batch_size=32, epochs=1, callbacks=callbacks,
#                     validation_data=(X_valid, y_valid))


# ____ Run U-Net with Data Augmentation ____
results = model.fit_generator(train_generator, steps_per_epoch=(len(X_train) // bs), epochs=training_epochs, callbacks=callbacks, validation_data=(X_valid, y_valid))

plot_results(results)                                                   # Plot learning curve


# ___ Test The Network ______________________________________________________________________
print('\n\n\nTesting Model\n_____________________________________________________________________________\n')

model.load_weights(model_name)                                          # Load best model

model.evaluate(X_valid, y_valid, verbose=1)                             # Evaluate on validation set (this must be equals to the best log_loss)

preds_train = model.predict(X_train, verbose=1)                         # Predict on train, val and test
preds_val = model.predict(X_valid, verbose=1)

preds_train_t = (preds_train > thresHold).astype(np.uint8)              # Threshold predictions
preds_val_t = (preds_val > thresHold).astype(np.uint8)


# ___ Plot Results ___________________________________________________________________________

# plot_sample(X_train, y_train, preds_train, preds_train_t, ix=14)
print("\nPredictions On Training Data\n")
plot_sample(X_train, y_train, preds_train, preds_train_t)               # Check if training data looks all right


# plot_sample(X_valid, y_valid, preds_val, preds_val_t, ix=19)
print("\nPredictions On Validation Data\n")
plot_sample(X_valid, y_valid, preds_val, preds_val_t)                   # Check if valid data looks all right


# ______________________________________________________________________________________________________________
#  Use Network on New Data
# ______________________________________________________________________________________________________________

X_new, y_new = get_data(path_test, im_height, im_width, train=True)     # Get and resize train images and masks

preds_new = model.predict(X_new, verbose=1)                             # Predict on train, val and test
preds_new_t = (preds_new > thresHold ).astype(np.uint8)                 # Threshold predictions

plot_sample(X_new, y_new, preds_new, preds_new_t)                       # Check if training data looks all right

print("\n\n\nScript Done!!")
