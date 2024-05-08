from PIL import Image
import glob
import core as cc

captchaType = cc.CaptchaType.GOV24

# Training image data path
train_img_path_list = glob.glob("images/"+ captchaType.value + "/train/*.png")

# Training image data size
img = Image.open(train_img_path_list[0])
img_width = img.width
img_height = img.height

# Creating an instance that creates a model
CM = cc.CreateModel(train_img_path_list, img_width, img_height)

# Performing model training
model = CM.train_model(epochs=60, earlystopping=True)

# Saving the weights learned by the model to a file
weights_path = "model/"+ captchaType.value + ".weights.h5"
model.save_weights(weights_path)

CM.predict(model, train_img_path_list)