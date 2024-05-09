from PIL import Image
import glob
import core as cc

captchaType = cc.CaptchaType.NH_WEB_MAIL

# Training image data path
train_img_path_list = glob.glob("images/"+ captchaType.value + "/train/*.png")

# Training image data size
img = Image.open(train_img_path_list[0])
img_width = img.width
img_height = img.height

# Creating an instance that creates a model
CM = cc.CreateModel(train_img_path_list, img_width, img_height)

# Performing model training
model = CM.train_model(epochs=100, earlystopping=True, early_stopping_patience=6)

# Saving the weights learned by the model to a file
weights_path = "model/"+ captchaType.value + ".weights.h5"
# model.save_weights(weights_path)
model.save(weights_path)
# model.save("model/"+ captchaType.value)
