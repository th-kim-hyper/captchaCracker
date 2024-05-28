from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.NH_WEB_MAIL
PATIENCE = 8

# Hyper(CAPTCHA_TYPE, quiet_out=True).model_train(epochs=1, patience=PATIENCE, save_model=False)
Hyper(CAPTCHA_TYPE).train_model(early_stopping_patience=PATIENCE)