from hyper import CaptchaType, Hyper

captcha_type = CaptchaType.NH_WEB_MAIL
weights_only = True

Hyper(captcha_type, weights_only).validate_model()
