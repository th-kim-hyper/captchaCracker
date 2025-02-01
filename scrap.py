import os, base64, requests, glob, shutil
from datetime import datetime
from time import sleep
from playwright.sync_api import sync_playwright

def image_save():
    btn = iframe_element.wait_for_selector('input.w2trigger.btn_fcm.ty1')
    btn.click()
    sleep(1)

    img = iframe_element.wait_for_selector('img.w2image.mr10.pb5')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
    scrap_path = r'C:\python\captchaCracker\images\supreme_court\scrap'
    image_path = os.path.join(scrap_path, f'{timestamp}.png')
    img.screenshot(path=image_path)
    # data_url = img2DataUrl(img)
    
    # save_base64_to_png(data_url.replace('data:image/png;base64,', ''), image_path)
    return image_path

def save_base64_to_png(base64_str, file_path):
    img_data = base64.b64decode(base64_str)
    with open(file_path, 'wb') as f:
        f.write(img_data)

# def img2DataUrl(img):
#     script = '''
#     img => {
#         const canvas = document.createElement('canvas');
#         const ctx = canvas.getContext('2d');
#         canvas.width = img.width;
#         canvas.height = img.height;
#         // 흰색 배경 그리기
#         ctx.fillStyle = '#FFFFFF';
#         ctx.fillRect(0, 0, canvas.width, canvas.height);
#         // 이미지 그리기
#         ctx.drawImage(img, 0, 0);
#         return canvas.toDataURL('image/png');
#     }
#     '''
#     return img.evaluate(script)

def img2DataUrl(image_path):
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
    return 'data:image/png;base64,' + base64.b64encode(img_data).decode('utf-8')


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

# file_list = glob.glob('images/*.png')
# for file in file_list:
#     if '[UNK]' not in file:
#         shutil.move(file, os.path.join("images", 'train', os.path.basename(file)))
    # else:
    #     shutil.move(file, os.path.join("images", 'train', f'{pred}.png'))


# for file in file_list:
#     pred = predictCaptcha('supreme_court', file)
#     print(f'origin = {file},    pred = {pred}')
#     shutil.move(file, os.path.join('images', f'{pred}.png'))

url = "https://www.scourt.go.kr/portal/information/events/search/search.jsp"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page()
    page.goto(url)

    iframe_element = page.frames[1]

    for i in range(2):
        image_save()
        print(f'{i+1}번째 이미지 저장 완료')  

    browser.close()