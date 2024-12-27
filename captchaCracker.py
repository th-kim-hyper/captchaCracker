import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

import sys, time
from PIL import Image
from cc.Utils import get_train_data_list
from cc.Core import Model

NULL_OUT = open(os.devnull, "w")
STD_OUT = sys.stdout

if("__main__" == __name__):
    
    argv = sys.argv
    exec = os.path.basename(argv[0])
    
    if len(argv) > 3:
        print('사용법 : ' + exec + ' <캡챠유형> <이미지파일경로>')
        print('해당 캡챠 유형의 이미지를 인식한 결과를 반환합니다.')
        print('<캡챠유형>은 미리제공되는 3가지 기본 유형이 있습니다. "supreme_court", "gov24", "wetax"')
        print('<이미지파일경로>는 인식할 이미지 파일의 경로를 입력합니다. 예) "C:\\temp\\download.png"')
        sys.exit(-1)    

    sys.stdout = NULL_OUT
    captcha_type = argv[1]
    image_path = argv[2]
    config = {'data_base_dir': './images', 'model_base_dir': './model'}
    data_base_dir = config['data_base_dir']
    model_base_dir = config['model_base_dir']
    train_data_list = get_train_data_list()
    train_data = train_data_list[captcha_type]
    weights_only = False
    
    model = Model(train_data=train_data, weights_only=False, verbose=0)
    temp_dir = os.path.abspath("./temp")

    if os.path.exists(temp_dir) == False:
        os.makedirs(temp_dir)

    temp_image_path=os.path.join("temp", f"{time.time():12.0f}.png")
    
    with Image.open(image_path) as image:

        if image.mode in ('RGBA', 'LA'):
            background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image, image.split()[-1]) # omit transparency
            image = background

        image.save(temp_image_path)
        
    pred = model.predict(temp_image_path)
    
    # model.quiet(False)
    sys.stdout.write(pred)
    # model.quiet(True)

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    sys.exit(0)
