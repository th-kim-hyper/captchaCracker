from hyper import CaptchaType, Hyper

CAPTCHA_TYPE = CaptchaType.SUPREME_COURT
WEIGHT_ONLY = True

hyper = Hyper(CAPTCHA_TYPE, WEIGHT_ONLY)
# hyper.quiet(True)
hyper.model_validate()
# hyper.quiet(False)
