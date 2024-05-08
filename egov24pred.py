import os
import glob
import time
from PIL import Image
import core as cc
# from io import StringIO 

captchaType = cc.CaptchaType.GOV24

# Target image path
# target_img_path = "../data/target.png"
target_img_path_list = glob.glob("../images/gov24/target/*.png")

# Target image data size
img = Image.open(target_img_path_list[0])
img_width = img.width
img_height = img.height
# Target image label length
# Target image label component
characters = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
max_length = 6

labels = [img.split(os.path.sep)[-1].split(".png")[0] for img in target_img_path_list]
max_length = max([len(label) for label in labels])
characters = sorted(set(char for label in labels for char in label))

# Model weight file path
weights_path = "model/gov24.weights.h5"
# weights_path = "model/gov24"
# Creating a model application instance
AM = cc.ApplyModel(weights_path, img_width, img_height, max_length, characters)

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
    
    # if(len(pred) != 6):
    #     print("Error: ", pred)
    #     continue
    # copyfile(target_img_path, "../images/gov24/" + pred + ".png")
    # os.rename(target_img_path, "../images/gov24/" + pred + ".png")


end = time.time()

print("Matched:", matched, ", Tottal : ", len(target_img_path_list))
print("pred time : ", end - start, "sec")