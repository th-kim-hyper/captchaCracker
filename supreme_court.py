import os, shutil, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from PIL import Image
from cc.Core import Model, get_captcha_type_list

NULL_OUT = open(os.devnull, "w")
STD_OUT = sys.stdout

def supremeCourtPreprocess(image_path):
    w = 120
    h = 40

    with Image.open(image_path) as img:
        img_mode = img.mode
        
        if img_mode in ('RGBA', 'LA'):
            background = Image.new(img_mode[:-1], img.size, (255, 255, 255))
            background.paste(img, (0,0), img.split()[-1])
            background.convert('RGB')
            background.save(image_path)

        if img.size[0] > w and img.size[1] > h:
            with Image.new('RGB', (w, h), (255, 255, 255)) as bg:
                cropped_img = img.crop((1, 1, w+1, h+1))
                bg.paste(cropped_img, (0, 0))
                bg.save(image_path, format='PNG')


if("__main__" == __name__):
    
    argv = sys.argv
    exec = os.path.basename(argv[0])
    
    if len(argv) < 2:
        print('사용법 : ' + exec + ' <이미지파일경로>')
        print('<이미지파일경로>는 인식할 이미지 파일의 경로를 입력합니다. 예) "C:\\temp\\download.png"')
        sys.exit(-1)    

    sys.stdout = NULL_OUT
    # captcha_type = argv[1]
    captcha_type = "supreme_court"
    image_path = argv[1]
    config = {'data_base_dir': './images', 'model_base_dir': './model'}
    data_base_dir = config['data_base_dir']
    model_base_dir = config['model_base_dir']
    captcha_type_list = get_captcha_type_list() # get_train_data_list()
    train_data = captcha_type_list[captcha_type].data
    weights_only = False
    
    model = Model(train_data=train_data, weights_only=False, verbose=0)
    temp_dir = os.path.abspath("./temp")

    if os.path.exists(temp_dir) == False:
        os.makedirs(temp_dir)

    temp_image_path=os.path.join("temp", f"{time.time():12.0f}.png")
    supremeCourtPreprocess(image_path)
    shutil.copy(image_path, temp_image_path)
    pred, accuracy = model.predict(temp_image_path)
    sys.stdout = STD_OUT
    print(pred, end='')

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    sys.exit(0)
