import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['AUTOGRAPH_VERBOSITY'] = '0'
import sys
import glob
import time
import CaptchaCracker as cc
from io import StringIO 

# import tensorflow as tf
import logging

class NullIO(StringIO):
    def write(self, txt):
        pass


# tf.autograph.set_verbosity(0)
# tf.get_logger().setLevel('ERROR')
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# tf.get_logger().setLevel(logging.ERROR)
# tf.autograph.set_verbosity(1)


old_stdout = sys.stdout # backup current stdout
sys.stdout = NullIO()
# sys.stderr = NullIO()

# Target image data size
img_width = 121
img_height = 41
# Target image label length
max_length = 6
# Target image label component
characters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}

# Model weight file path
weights_path = "model/weights.h5"
# Creating a model application instance
AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)

# Target image path
# target_img_path = "../data/target.png"
target_img_path_list = glob.glob("../images/target/*.png")

matched = 0

start = time.time()

# Predicted value
for target_img_path in target_img_path_list:
    pred = AM.predict(target_img_path)
    ori = target_img_path.split(os.path.sep)[-1].split(".")[0]
    msg = ""
    if(ori == pred):
        matched += 1
    else:
        msg = " Not matched!"
    print("ori : ", ori, "pred : ", pred, msg)


end = time.time()
sys.stdout = old_stdout

print("Matched:", matched, ", Tottal : ", len(target_img_path_list))
print("pred time : ", end - start, "sec")