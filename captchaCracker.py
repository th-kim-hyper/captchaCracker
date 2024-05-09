import sys
import os
import core as cc
from PIL import Image
import glob

f = open(os.devnull, 'w')
s = sys.stdout
sys.stdout = f

def main(captchaType:cc.CaptchaType, imagePath:str)->str:

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
        sys.stdout = s
        print(pred)
    except Exception as e:
        print("Error:", e)

    return pred

captchaType = cc.CaptchaType.SUPREME_COURT

argv = sys.argv

if len(argv) < 3:
    sys.stdout = s
    print("Usage: " + os.path.basename(argv[0]) + " supreme_court|gov24|nh_web_mail IMAGE_FILE")
    sys.exit(-1)

if("__main__" == __name__):
    captchaType = cc.CaptchaType(argv[1])
    imagePath = argv[2]
    main(captchaType, imagePath)
else:
    sys.stdout = s
    print("module imported")
