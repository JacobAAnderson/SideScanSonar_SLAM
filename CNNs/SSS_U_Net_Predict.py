
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.layers import Input
from keras.optimizers import Adam

from SSS_U_Net_Network import get_unet, get_data, plot_sample, plot_input, plot_output, plot_prediction

plt.style.use("ggplot")


# ____ Side Scan Sonar Data ____
im_width = 256          # Image Width
im_height = 256         # Image Height
border = 5
seed = 2018
bs = 10                 # Batch Size
training_epochs = 100   # Number of Training Epochs
thresHold = 0.20         # Prediction Thresholds


# path_train = '/home/jake/Data_Processed/SideScanSonarImages/U-Net_Features/train/'
path_test = '/home/jake/Data_Processed/SideScanSonarImages/U-Net_Features/test/'
# path_test = '/home/jake/Data_Processed/SSS_images_for_UNet/'
outputz_path = '/home/jake/Data_Processed/SSS_images_for_UNet/predictions/'
model_name = 'SSS_U_Net_FeatureExtraction_Model_Barbados_CatalinaIS_LM2_SWA_Cam.h5'


# ___ Load Data _________________________________________________________________________________
x, names = get_data(path_test, im_height, im_width, train=False )                          # Get and resize train images and masks



#with open(outputz_path +'imageNames.txt', 'w') as f:
#    for name in names:
#        f.write("%s\n" % name)

# ___ U-NET______________________________________________________________________________________________________________
input_img = Input((im_height, im_width, 1), name='img')

model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

model.load_weights(model_name)                                                      # Load best model

preds = model.predict(x, verbose=1)                                                 # Predict on train, val and test
preds_t = (preds > thresHold).astype(np.uint8)                                      # Threshold predictions


for ii in range(len(x)):

    """

    im = Image.fromarray(preds[ii].squeeze()*255)
    im = im.convert("L")
    im.save(outputz_path + "prediction_" + names[ii])

    im = Image.fromarray(preds_t[ii].squeeze()*255)
    im.save(outputz_path + "binary_"  + names[ii])
    """

    plot_prediction(x, preds, preds_t, ii)

#    plot_sample(x, y, preds, preds_t, ii)                                           # Check if training data looks all right
#    plot_input(x, y, ii)
#    plot_output(preds, y, ii)

print("\n\n\nScript Done!!")
