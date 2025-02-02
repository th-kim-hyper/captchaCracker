import os, shutil, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from PIL import Image
from cc.Core import Model, TrainData

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
    base_dir = os.path.abspath(os.path.dirname(__file__))
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        base_dir = meipass
    images_dir = os.path.join(base_dir, "images")
    model_dir = os.path.join(base_dir, "model")
    
    if len(argv) < 2:
        print('사용법 : ' + exec + ' <이미지파일경로>')
        print('<이미지파일경로>는 인식할 이미지 파일의 경로를 입력합니다. 예) "C:\\temp\\download.png"')
        sys.exit(-1)    

    captcha_type = "supreme_court"
    image_path = argv[1]

    if os.path.exists(image_path) == False:
        print('파일을 찾을 수 없습니다. : ' + image_path)
        sys.exit(-1)

    sys.stdout = NULL_OUT
    train_data = TrainData(
        id=captcha_type, image_dir=images_dir, model_dir=model_dir,
        image_width=120, image_height=40, label_length=6, init=False)
    weights_only = False
    model = Model(train_data=train_data, weights_only=False, verbose=0)
    temp_dir = os.path.abspath("./temp")

    if os.path.exists(temp_dir) == False:
        os.makedirs(temp_dir)

    temp_image_path=os.path.join("temp", f"{time.time():12.0f}.png".replace(" ", ""))
    supremeCourtPreprocess(image_path)
    shutil.copy(image_path, temp_image_path)
    pred, accuracy = model.predict(temp_image_path)
    sys.stdout = STD_OUT
    print(pred, end='')

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    sys.exit(0)
