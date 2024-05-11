import os
from PIL import Image
import core as cc
import tensorflow as tf
# tf.get_logger().setLevel('ERROR')
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras

PATIENCE = 7
CAPTCHA_TYPE = cc.CaptchaType.NH_WEB_MAIL
WEIGHT_ONLY = False

train_img_path_list = cc.get_image_files(CAPTCHA_TYPE, train=True)
img = Image.open(train_img_path_list[0])
img_width = img.width
img_height = img.height

CM = cc.CreateModel(train_img_path_list, img_width, img_height)
model = CM.train_model(epochs=100, earlystopping=True, early_stopping_patience=PATIENCE)
weights_path = cc.get_weights_path(CAPTCHA_TYPE, WEIGHT_ONLY)
model.save_weights(weights_path)
