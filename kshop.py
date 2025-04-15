import os, shutil, sys, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from PIL import Image
from cc.Core import Model, TrainInfo

NULL_OUT = open(os.devnull, "w")
STD_OUT = sys.stdout
CURRUNT_DIR = os.path.abspath(os.path.dirname(__file__))
ARGV = sys.argv
EXEC = os.path.basename(ARGV[0])
CAPTCHA_TYPE = "kshop"
REV = 0

if("__main__" == __name__):
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        CURRUNT_DIR = os.path.abspath(meipass)

    if len(ARGV) < 2:
        print('사용법 : ' + EXEC + ' <이미지파일경로>')
        print('<이미지파일경로>는 인식할 이미지 파일의 경로를 입력합니다. 예) "C:\\temp\\download.png"')
        sys.exit(-1)

    image_path = os.path.abspath(ARGV[1])

    if os.path.exists(image_path) == False:
        sys.stderr.write('파일을 찾을 수 없습니다. : ' + image_path + '\n')
        sys.exit(-1)

    base_dir = os.path.join(CURRUNT_DIR, "captcha_data")
    images_dir = os.path.join(base_dir, CAPTCHA_TYPE, str(REV), "images")
    model_dir = os.path.join(base_dir, CAPTCHA_TYPE, str(REV), "model")
    image_width = 263
    image_height = 54
    label_length = 6
    characters = sorted(list("0123456789"))
    weights_only = False
    verbose = 0

    if os.path.exists(model_dir) == False:
        sys.stderr.write('디렉토리를 찾을 수 없습니다. : ' + model_dir + '\n')
        sys.exit(-1)

    train_data = TrainInfo(
        id=CAPTCHA_TYPE, rev=REV, base_dir=base_dir,
        image_width=image_width, image_height=image_height,
        label_length=label_length, characters=characters, init=False)
    model = Model(train_data=train_data, weights_only=weights_only, verbose=verbose)
    temp_dir = os.path.abspath("./temp")

    if os.path.exists(temp_dir) == False:
        os.makedirs(temp_dir)

    temp_image_path=os.path.join("temp", f"{time.time():12.0f}.png".replace(" ", ""))
    shutil.copy(image_path, temp_image_path)
    pred, accuracy = model.predict(temp_image_path)
    print(pred, end='')

    if os.path.exists(temp_image_path):
        os.remove(temp_image_path)

    sys.exit(0)
