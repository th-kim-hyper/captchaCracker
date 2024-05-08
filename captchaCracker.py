import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import core as cc
from PIL import Image
import glob

def main(captchaType:cc.CaptchaType, imagePath:str)->str:

    import logging
    logging.getLogger('tensorflow').disabled = True

    pred = ""
    baseDir = os.path.dirname(__file__)

    try:
        img = Image.open(os.path.join(baseDir, imagePath))
        img_width = img.width
        img_height = img.height

        weights_path = os.path.join(baseDir, "model", captchaType.value + ".weights.h5")
        train_img_dir = os.path.join(baseDir, "images", captchaType.value, "train")
        train_img_path_list = glob.glob(train_img_dir + os.sep + "*.png")
        labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in train_img_path_list]
        max_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))

        AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)
        pred = AM.predict(imagePath)
        print(pred)
    except Exception as e:
        print("Error:", e)

    # return pred

captchaType = cc.CaptchaType.SUPREME_COURT
# MODEL_PATH = "model/supremecourt.weights.h5"

argv = sys.argv

if len(argv) < 3:
    print("Usage: " + os.path.basename(argv[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    captchaType = cc.CaptchaType(argv[1])
    imagePath = argv[2]
    # imagePath = os.path.join(os.path.dirname(__file__), argv[2])
    main(captchaType, imagePath)
else:
    print("module imported")
