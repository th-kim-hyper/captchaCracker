import os
import glob
import time
from PIL import Image
import core as cc

captchaType = cc.CaptchaType.SUPREME_COURT
pred_img_path_list = glob.glob("images/"+ captchaType.value + "/pred/*.png")
train_img_path_list = glob.glob("images/"+ captchaType.value + "/train/*.png")

# Target image data size
img = Image.open(pred_img_path_list[0])
img_width = img.width
img_height = img.height

labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in train_img_path_list]
max_length = max([len(label) for label in labels])
characters = sorted(set(char for label in labels for char in label))

weights_path = "model/"+ captchaType.value + ".weights.h5"

AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)

matched = 0

start = time.time()

# Predicted value
for pred_img_path in pred_img_path_list:
    pred = AM.predict(pred_img_path)
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