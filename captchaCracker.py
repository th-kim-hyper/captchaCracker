# -*- coding:utf-8 -*-

import os
import sys
import time
from hyper import CaptchaType, Hyper, ApplyModel

start = time.time()
ARGV = sys.argv
CAPTCHA_TYPE = CaptchaType.SUPREME_COURT
WEIGHT_ONLY = True
hyper = Hyper()
pred = ""

def main(captchaType:CaptchaType, weight_only, imagePath:str):


    try:
        train_img_path_list = hyper.get_image_files(captchaType, train=True)
        img_width, img_height = hyper.get_image_info(train_img_path_list)
        max_length, characters = hyper.get_train_info(train_img_path_list)
        weights_path = hyper.get_weights_path(captchaType, weight_only)
        model = ApplyModel(weights_path, img_width, img_height, max_length, characters)
        pred = model.predict(imagePath)
    except Exception as e:
        hyper.quiet(False)
        print("Error:", e)

    return pred

if len(ARGV) < 3:
    print("Usage: " + os.path.basename(ARGV[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    hyper.quiet(True)
    CAPTCHA_TYPE = CaptchaType(ARGV[1])
    imagePath = ARGV[2]
    pred = main(CAPTCHA_TYPE, WEIGHT_ONLY, imagePath)
    hyper.quiet(False)
    print(pred)
    # end = time.time()
    # print("time : ", end - start, "sec")
    # print(pred, CAPTCHA_TYPE, WEIGHT_ONLY, imagePath)
    sys.exit(0)

else:
    print("module imported")
