import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)

from cc.Core import CaptchaType, TrainData, Model, get_captcha_type_list

captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list['supreme_court'].data

model = Model(train_data=train_data)
model.validate_model()

print("Done!")
