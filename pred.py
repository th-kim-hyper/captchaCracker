from cc.Core import get_captcha_type_list, CaptchaType, TrainData, Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

captcha_type_list:list[CaptchaType] = get_captcha_type_list()
train_data:TrainData = captcha_type_list['supreme_court'].data
model = Model(train_data=train_data)
model.validate_model()

print("Done!")
