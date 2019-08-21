#!/usr/bin/env python


# Every python controller needs these lines
import rospy
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input
from keras.optimizers import Adam
from u_net import get_unet

from sensor_msgs.msg import Image
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import UInt8MultiArray

model_name = 'SSS_U_Net_FeatureExtraction_Model_Barbados_CatalinaIS_LM1_LM2_SWE.h5'
model_path = '/home/jake/catkin_ws/src/sidescansoanr_slam/models/'

im_width = 256           # Image Width
im_height = 256          # Image Height
thresHold = 0.20         # Prediction Thresholds

def sssFeaturePredict_callback(msg, model):

    im = np.frombuffer(msg.data, dtype = np.uint8).reshape(1,im_width, im_height, -1)
    im = im.astype(np.float32)

    preds = model.predict(im, verbose=1)                                                 # Predict on train, val and test

    preds_t = (preds > thresHold) * 255                                      # Threshold predictions

    y = preds_t[0,...,0]

    '''
    plt.figure()
    plt.imshow(y, cmap='gray')
    plt.show()
    '''
    
    y = y.reshape(im_width * im_height, -1)

    output = Image()
    output.header.frame_id = msg.header.frame_id
    output.data = tuple( y.astype(np.uint8))

    pub.publish(output)





if __name__ == '__main__':

    input_img = Input((im_height, im_width, 1), name='img')

    uNet_model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)
    uNet_model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    uNet_model.load_weights(model_path + model_name)                                                      # Load best model
    uNet_model._make_predict_function()

    rospy.init_node('sss_featurePredictor')
    sub = rospy.Subscriber('/rawSSSim', numpy_msg(Image), sssFeaturePredict_callback, uNet_model)
    pub = rospy.Publisher('/sssFeatures', Image, queue_size=1)

    print("\n\n\nSSS Feature Predictor Set Up!!\n\n\n")

    rospy.spin()
