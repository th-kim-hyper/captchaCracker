import os, base64, requests, glob, shutil
from datetime import datetime
from time import sleep
from playwright.sync_api import sync_playwright

# def image_save():
#     btn = iframe_element.wait_for_selector('input.w2trigger.btn_fcm.ty1')
#     btn.click()
#     sleep(1)

#     img = iframe_element.wait_for_selector('img.w2image.mr10.pb5')
#     timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
#     scrap_path = r'C:\python\captchaCracker\images\supreme_court\scrap'
#     image_path = os.path.join(scrap_path, f'{timestamp}.png')
#     img.screenshot(path=image_path)
#     # data_url = img2DataUrl(img)
    
#     # save_base64_to_png(data_url.replace('data:image/png;base64,', ''), image_path)
#     return image_path

# def save_base64_to_png(base64_str, file_path):
#     img_data = base64.b64decode(base64_str)
#     with open(file_path, 'wb') as f:
#         f.write(img_data)

# def img2DataUrl(image_path):
#     with open(image_path, 'rb') as img_file:
#         img_data = img_file.read()
#     return 'data:image/png;base64,' + base64.b64encode(img_data).decode('utf-8')

def predictCaptcha(captcha_id, captcha_image_path):
    url = 'https://dev.hyperinfo.co.kr/captcha/api/predict' if 'https' in captcha_image_path else 'http://dev.hyperinfo.co.kr:12004/api/predict'
    data_url = img2DataUrl(captcha_image_path)
    
    form_data = {
        'captcha_id': captcha_id,
        'captcha_data_url': data_url
    }
    
    response = requests.post(url, data=form_data, headers={
        'Host': 'dev.hyperinfo.co.kr',
        'Content-Type': 'application/x-www-form-urlencoded'
    })
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print('Error:', response.status_code, response.text)
        return None

def scrap_supreme_court(count):

    url = "https://www.scourt.go.kr/portal/information/events/search/search.jsp"

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(url)
        iframe = page.frames[1]
        result = []

        for i in range(count):
            btn = iframe.wait_for_selector('input.w2trigger.btn_fcm.ty1')
            btn.click()
            sleep(1)
            img = iframe.wait_for_selector('img.w2image.mr10.pb5')
            timestamp = datetime.now().strftime(r'%Y%m%d_%H%M%S_%f')[:-3]
            scrap_path = os.path.join('images', 'supreme_court', 'scrap')
            image_path = os.path.join(scrap_path, f'{timestamp}.png')
            img.screenshot(path=image_path)
            result.append(image_path)
            print(f'{i+1}/{count} Scraped: {image_path}')
    return result

def scrap(captcha_id, count):

    if captcha_id == 'supreme_court':
        scrap_supreme_court(count)

scrap('supreme_court', 10)
