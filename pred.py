from hyper import CaptchaType, Hyper

captcha_type = CaptchaType.GOV24
weights_only = True

Hyper(captcha_type, weights_only).validate_model()
