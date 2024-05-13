import hyper

CAPTCHA_TYPE = hyper.CaptchaType.NH_WEB_MAIL
WEIGHT_ONLY = False

hyper.model_predict(CAPTCHA_TYPE, WEIGHT_ONLY)
