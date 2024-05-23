from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.NH_WEB_MAIL
WEIGHT_ONLY = False

hyper = Hyper(CAPTCHA_TYPE, WEIGHT_ONLY)
# hyper.quiet(True)
hyper.model_validate()
# hyper.quiet(False)
