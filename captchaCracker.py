import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import core as cc
from PIL import Image
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import time
start = time.time()

CAPTCHA_TYPE = cc.CaptchaType.SUPREME_COURT
WEIGHT_ONLY = True
ARGV = sys.argv
NULL_OUT = open(os.devnull, 'w')
ORI_OUT = sys.stdout
BASE_DIR = cc.get_base_dir()

def main(captchaType:cc.CaptchaType, imagePath:str):

    pred = ""
    baseDir = BASE_DIR

    try:
        img = Image.open(os.path.join(baseDir, imagePath))
        img_width = img.width
        img_height = img.height
        weights_path = cc.get_weights_path(captchaType, WEIGHT_ONLY)
        train_img_path_list = cc.get_image_files(captchaType, train=True)
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in train_img_path_list]
        max_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        prediction_model = cc.load_model(weights_path, img_width, img_height, max_length, characters)
        pred = cc.predict(prediction_model, imagePath, img_width, img_height, max_length, characters)
    except Exception as e:
        sys.stdout = ORI_OUT
        print("Error:", e)

    return pred

if len(ARGV) < 3:
    print("Usage: " + os.path.basename(ARGV[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    sys.stdout = NULL_OUT
    CAPTCHA_TYPE = cc.CaptchaType(ARGV[1])
    imagePath = ARGV[2]
    pred = main(CAPTCHA_TYPE, imagePath)
    sys.stdout = ORI_OUT
    print(pred)
    end = time.time()
    # print("time : ", end - start, "sec")
    sys.exit(0)

else:
    print("module imported")
