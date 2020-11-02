
import os

import numpy as np
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from SSS_Siamese_Network import loadimgs, get_siamese_model


# __ SSS Data Set ______________________________________________________________________________________________________
test_folder = "/home/jake/Data_Processed/SSS_images_for_siameseNet/images"
output_folder = "/home/jake/Data_Processed/SSS_images_for_siameseNet/"
model_path = '/home/jake/PycharmProjects/SSS_Keras/Data_SSS/models/'
model_name = 'SSS_Siamese_weights2.h5'

dashed_line = "-----------------------------------------------------------------------------------------------"


# ____ Load Data _______________________________________________________________________________________________________
# -- Get New Data ---
testImgs, imageNames = loadimgs(test_folder)

# ____ Test Network ____________________________________________________________________________________________________
optimizer = Adam(lr=0.00006)

model = get_siamese_model((105, 105, 1))
model.compile(loss="binary_crossentropy", optimizer=optimizer)
model.load_weights(os.path.join(model_path, model_name))


n = len(testImgs)

with open(output_folder + 'sss_siamese_predictions.txt', 'w') as f:

    for ii in range(0, n):

        print(">> Iteration: {} of {}".format(ii, n), end='\r', flush=True)

        for jj in range(ii+1, n):

            test_image = np.asarray(testImgs[ii]).reshape(1, 105, 105, 1)
            target_image = np.asarray(testImgs[jj]).reshape(1, 105, 105, 1)

            eval_pair = [test_image, target_image]

            prob = model.predict(eval_pair)

            f.write("%s, %s, %f\n" % (imageNames[ii], imageNames[jj], prob))

#            print("Probability: {}".format(prob))

#           if prob > 0.8:

#                fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

#                ax1.matshow(test_image.squeeze(), cmap='gray', label="Images: {}".format(ii))
#                ax1.get_yaxis().set_visible(False)
#                ax1.get_xaxis().set_visible(False)

#                ax2.matshow(target_image.squeeze(), cmap='gray', label="Image: {}".format(jj))

#                plt.title("Probability: {}".format(prob))
#                plt.xticks([])
#                plt.yticks([])
#                plt.show()


print("\n\nDone")
