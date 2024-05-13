import os
import glob
import time
from PIL import Image
from enum import Enum
import core as cc

class CaptchaType(Enum):
    SUPREME_COURT = "supreme_court"
    GOV24 = "gov24"
    NH_WEB_MAIL = "nh_web_mail" 

class Quieter():
    import os
    import sys
    import tensorflow as tf
    import absl.logging
    
    def __init__(self):
        import sys
        self.NULL_OUT = open(os.devnull, 'w')
        self.STD_OUT = sys.stdout

    def start(self):
        import os
        import sys
        import tensorflow as tf
        import absl.logging

        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        tf.get_logger().setLevel('ERROR')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        absl.logging.set_verbosity(absl.logging.ERROR)
        sys.stdout = self.NULL_OUT

    def stop(self):
        import os
        import sys
        import tensorflow as tf
        import absl.logging
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
        tf.get_logger().setLevel('INFO')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        absl.logging.set_verbosity(absl.logging.INFO)
        sys.stdout = self.STD_OUT

def get_base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def get_image_files(cap_type:CaptchaType, train=True):
    baseDir = get_base_dir()
    imgDir = os.path.join(baseDir, "images", cap_type.value, "train" if train else "pred")
    return glob.glob(imgDir + os.sep + "*.png")

def get_image_info(img_path_list:list):
    img_path = img_path_list[-1]
    img = Image.open(img_path)
    img_width = img.width
    img_height = img.height
    return img_width, img_height

def get_train_info(train_img_path_list):
    labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in train_img_path_list]
    max_length = max([len(label) for label in labels])
    characters = sorted(set(char for label in labels for char in label))
    return max_length, characters

def get_weights_path(cap_type:CaptchaType, weightsOnly=True):
    baseDir = get_base_dir()
    weightsDir = os.path.join(baseDir, "model", cap_type.value)
    if weightsOnly:
        weightsDir = weightsDir + ".weights.h5"

    return weightsDir

def model_train(captcha_type:CaptchaType, patience:int=7):
    train_img_path_list = get_image_files(captcha_type, train=True)
    img_width, img_height = get_image_info(train_img_path_list)
    CM = cc.CreateModel(train_img_path_list, img_width, img_height)
    model = CM.train_model(epochs=100, earlystopping=True, early_stopping_patience=patience)
    weights_path = get_weights_path(captcha_type, True)
    model.save_weights(weights_path)
    weights_path = get_weights_path(captcha_type, False)
    model.save(weights_path)

def model_predict(captcha_type:CaptchaType, weight_only=True):
    start = time.time()
    matched = 0

    pred_img_path_list = get_image_files(captcha_type, train=False)
    train_img_path_list = get_image_files(captcha_type, train=True)
    img_width, img_height = get_image_info(train_img_path_list)
    max_length, characters = get_train_info(train_img_path_list)
    weights_path = get_weights_path(captcha_type, weight_only)
    model = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)

    for pred_img_path in pred_img_path_list:
        pred = model.predict(pred_img_path)
        ori = pred_img_path.split(os.path.sep)[-1].split(".")[0]
        msg = ""
        if(ori == pred):
            matched += 1
        else:
            msg = " Not matched!"
        print("ori : ", ori, "pred : ", pred, msg)

    end = time.time()

    print("Matched:", matched, ", Tottal : ", len(pred_img_path_list))
    print("pred time : ", end - start, "sec")