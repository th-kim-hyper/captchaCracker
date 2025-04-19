import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from cc.Core import Model, get_captcha_type_list

captcha_id = 'kshop'
captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list[captcha_id].train_data
train_data.threshold = 60
model = Model(train_data=train_data, weights_only=False)
model.validate_model()

print("Done!")

# i = TrainInfo('default', 0)
