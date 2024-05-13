import hyper

CAPTCHA_TYPE = hyper.CaptchaType.NH_WEB_MAIL
PATIENCE = 7

hyper.model_train(CAPTCHA_TYPE, PATIENCE)
