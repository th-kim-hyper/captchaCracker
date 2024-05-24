from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.SUPREME_COURT
PATIENCE = 7

Hyper(CAPTCHA_TYPE).model_train(patience=PATIENCE, save_model=False)
