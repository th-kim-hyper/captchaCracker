from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType(1,2,3,None).NH_WEB_MAIL
WEIGHT_ONLY = True

Hyper(CAPTCHA_TYPE, WEIGHT_ONLY).validate_model()
# hyper.quiet(True)
# hyper.model_validate()
# hyper.quiet(False)
