from PIL import Image
import glob
import CaptchaCracker as cc

# Training image data path
train_img_path_list = glob.glob("../images/train/*.png")

# Training image data size
img_width = 121
img_height = 41

img = Image.open(train_img_path_list[0])
img_width = img.width
img_height = img.height

# Creating an instance that creates a model
CM = cc.CreateModel(train_img_path_list, img_width, img_height)

# Performing model training
model = CM.train_model(epochs=25)

# Saving the weights learned by the model to a file
model.save_weights("model/weights.h5")
# model.save("model/model.tf", save_format="tf")
# model.save_weights("model/weights.tf", save_format="tf")