
import pickle
import os

import matplotlib.pyplot as plt
import numpy as np

from keras.optimizers import Adam
from SiameseClassifier_Network import get_siamese_model, test_oneshot, make_oneshot_task, test_nn_accuracy, plot_oneshot_task


# __ Example Data Set __________________________________________________________________________________________________
# train_folder = "/home/jake/PycharmProjects/SSS_Keras/CharactureData/images_background/"
# val_folder = '/home/jake/PycharmProjects/SSS_Keras/CharactureData/images_evaluation/'
# save_path = '/home/jake/PycharmProjects/SSS_Keras/CharactureData/data/'
# model_path = '/home/jake/PycharmProjects/SSS_Keras/CharactureData/weights/'
# model_name = 'siamese_weights.h5'
# ways = np.arange(1, 20, 2)

# __ SSS Data Set ______________________________________________________________________________________________________
train_folder = "/home/jake/Data_Processed/SideScanSonarImages/Siamese_Matched_Images/train/"
val_folder = '/home/jake/Data_Processed/SideScanSonarImages/Siamese_Matched_Images/test/'
save_path = '/home/jake/PycharmProjects/SSS_Keras/Data_SSS/data/'
model_path = '/home/jake/PycharmProjects/SSS_Keras/Data_SSS/models/'
model_name = 'SSS_Siamese_weights.h5'
ways = np.arange(1, 8, 2)

dashed_line = "-----------------------------------------------------------------------------------------------"


# __ Get preloaded data from a pickle __________________________________________________________________________________
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)


# ____ Test Network ____________________________________________________________________________________________________
optimizer = Adam(lr=0.00006)

model = get_siamese_model((105, 105, 1))
model.compile(loss="binary_crossentropy", optimizer=optimizer)
model.load_weights(os.path.join(model_path, model_name))


resume = False
trials = 50

val_accs, train_accs,nn_accs = [], [], []


for N in ways:
    print("Test Validation Data:")
    val_accs.append(test_oneshot(model, Xtrain, Xval, train_classes, val_classes, N, trials, "val", verbose=True))

    print("Test Training Data:")
    train_accs.append(test_oneshot(model, Xtrain, Xval, train_classes, val_classes, N, trials, "train", verbose=True))

    nn_acc = test_nn_accuracy(N, Xtrain, Xval, train_classes, val_classes, trials)
    nn_accs.append(nn_acc)
    print("NN Accuracy = ", nn_acc)
    print("\n{}\n".format(dashed_line))


with open(os.path.join(save_path, "accuracies.pickle"), "wb") as f:
    pickle.dump((val_accs, train_accs, nn_accs), f)


with open(os.path.join(save_path, "accuracies.pickle"), "rb") as f:
    (val_accs, train_accs, nn_accs) = pickle.load(f)


# ____ Plotting ________________________________________________________________________________________________________

# __ Plot test results __
fig, ax = plt.subplots(1)
ax.plot(ways, val_accs, "m", label="Siamese(val set)")
ax.plot(ways, train_accs, "y", label="Siamese(train set)")
plt.plot(ways, nn_accs, label="Nearest neighbour")

ax.plot(ways, 100.0/ways, "g", label="Random guessing")
plt.xlabel("Number of possible classes in one-shot tasks")
plt.ylabel("% Accuracy")
plt.title("Omiglot One-Shot Learning Performance of a Siamese Network")
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.show()


# __ Plotting Demo __
print("Plotting Demos")

# pairs, targets = make_oneshot_task(16, Xtrain, Xval, train_classes, val_classes, "train", "Sanskrit")
pairs, targets = make_oneshot_task(4, Xtrain, Xval, train_classes, val_classes, "train")
plot_oneshot_task(pairs)

# inputs, targets = make_oneshot_task(20, Xtrain, Xval, train_classes, val_classes, "val", 'Oriya')
inputs, targets = make_oneshot_task(5, Xtrain, Xval, train_classes, val_classes, "val")
plot_oneshot_task(inputs)


print("\n\nDone")
