from cc.Core import Model, get_captcha_type_list, CaptchaType, TrainData
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

captcha_type_list = get_captcha_type_list()
train_data = captcha_type_list['default'].data
epochs = 100
batch_size = 32
earlystopping = True
early_stopping_patience = 10
save_weights = True
save_model = False

model = Model(train_data=train_data, save_model=save_model, save_weights=save_weights) 
model.train_model(
    epochs=epochs,
    batch_size=batch_size,
    earlystopping=earlystopping,
    early_stopping_patience=early_stopping_patience,
    save_weights=save_weights,
    save_model=save_model)

print("Done!")
