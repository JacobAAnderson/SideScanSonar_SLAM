
import os
import time
import pickle

from keras.optimizers import Adam
from SiameseClassifier_Network import loadimgs, get_siamese_model, get_batch, test_oneshot

# __ Example Data Set __________________________________________________________________________________________________
# train_folder = "/home/jake/PycharmProjects/SSS_Keras/CharactureData/images_background/"
# val_folder = '/home/jake/PycharmProjects/SSS_Keras/CharactureData/images_evaluation/'
# save_path = '/home/jake/PycharmProjects/SSS_Keras/CharactureData/data/'
# model_path = '/home/jake/PycharmProjects/SSS_Keras/CharactureData/weights/'
# model_name = 'siamese_weights.h5'

# -- Hyper parameters ----
# evaluate_every = 200    # interval for evaluating on one-shot tasks
# batch_size = 32
# n_iter = 20000          # No. of training iterations
# N_way = 20              # how many classes for testing one-shot tasks
# n_val = 250             # how many one-shot tasks to validate on
# best = -1


# __ SSS Data Set ______________________________________________________________________________________________________
train_folder = "/home/jake/Data_Processed/SideScanSonarImages/Siamese_Matched_Images/train/"
val_folder = '/home/jake/Data_Processed/SideScanSonarImages/Siamese_Matched_Images/test/'
save_path = '/home/jake/PycharmProjects/SSS_Keras/Data_SSS/data/'
model_path = '/home/jake/PycharmProjects/SSS_Keras/Data_SSS/models/'
model_name = 'SSS_Siamese_weights.h5'

# -- Hyper parameters ---
evaluate_every = 5    # interval for evaluating on one-shot tasks
batch_size = 6
n_iter = 200            # No. of training iterations
N_way = 4             # how many classes for testing one-shot tasks
n_val = 5             # how many one-shot tasks to validate on
best = -1


# ____ Load Data _______________________________________________________________________________________________________
# -- Get New Data ---
# X, y, c = loadimgs(train_folder)

# with open(os.path.join(save_path, "train.pickle"), "wb") as f:
#     pickle.dump((X, c), f)

# Xval, yval, cval = loadimgs(val_folder)

# with open(os.path.join(save_path, "val.pickle"), "wb") as f:
#     pickle.dump((Xval, cval), f)


# -- Get preloaded data from a pickle --
with open(os.path.join(save_path, "train.pickle"), "rb") as f:
    (Xtrain, train_classes) = pickle.load(f)

with open(os.path.join(save_path, "val.pickle"), "rb") as f:
    (Xval, val_classes) = pickle.load(f)


# ____ Train Model _____________________________________________________________________________________________________
dashed_line = "-----------------------------------------------------------------------------------------------"

print("\nValidation alphabets:", end="\n\n")
print(list(val_classes.keys()))
print("\nTraining alphabets: \n")
print(list(train_classes.keys()))

optimizer = Adam(lr=0.00006)

model = get_siamese_model((105, 105, 1))
model.compile(loss="binary_crossentropy", optimizer=optimizer)
model.summary()

print("\n\nStarting training process!\n")

t_start = time.time()

for i in range(1, n_iter+1):
    (inputs, targets) = get_batch(batch_size, Xtrain, Xval, train_classes, val_classes)
    loss = model.train_on_batch(inputs, targets)

    if i % evaluate_every == 0:

        print("\n{}\n".format(dashed_line))
        print("Epoch {} of {}".format(i, n_iter))
        print("Time for {0} iterations: {1} minutes".format(i, (time.time()-t_start)/60.0))
        print("Train Loss: {0}".format(loss))

        val_acc = test_oneshot(model, Xtrain, Xval, train_classes, val_classes, N_way, n_val, verbose=True)

        if val_acc >= best:
            print("Current best: {0}, previous best: {1}".format(val_acc, best))
            print("Saving weights to {}{}".format(model_path, model_name))
            model.save_weights(os.path.join(model_path, model_name))
            best = val_acc


print("\n\nDone!")
