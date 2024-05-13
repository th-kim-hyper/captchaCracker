import sys
import os
import time
import hyper
import core as cc

start = time.time()
ARGV = sys.argv
CAPTCHA_TYPE = hyper.CaptchaType.SUPREME_COURT
WEIGHT_ONLY = True
quiter = hyper.Quieter();

def main(captchaType:hyper.CaptchaType, weight_only, imagePath:str):

    pred = ""

    try:
        train_img_path_list = hyper.get_image_files(captchaType, train=True)
        img_width, img_height = hyper.get_image_info(train_img_path_list)
        max_length, characters = hyper.get_train_info(train_img_path_list)
        weights_path = hyper.get_weights_path(captchaType, weight_only)
        model = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)
        pred = model.predict(imagePath)
    except Exception as e:
        quiter.stop()
        print("Error:", e)

    return pred

if len(ARGV) < 3:
    print("Usage: " + os.path.basename(ARGV[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    quiter.start()
    CAPTCHA_TYPE = hyper.CaptchaType(ARGV[1])
    imagePath = ARGV[2]
    pred = main(CAPTCHA_TYPE, WEIGHT_ONLY, imagePath)
    quiter.stop()
    print(pred)
    # end = time.time()
    # print("time : ", end - start, "sec")
    sys.exit(0)

else:
    print("module imported")
