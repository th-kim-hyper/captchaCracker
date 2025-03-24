import os, warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)

from cc.Core import CaptchaType, TrainInfo, Model, get_captcha_type_list

captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list['supreme_court'].train_data

model = Model(train_data=train_data)
model.validate_model()

print("Done!")

# i = TrainInfo('default', 0)

