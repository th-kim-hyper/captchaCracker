import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import logging
# logging.getLogger("tensorflow").setLevel(logging.ERROR)
# import tensorflow as tf
# tf.autograph.set_verbosity(0)

def main(ImageFile:str)->str:

    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    IMAGE_FILE = ImageFile
    IMAGE_WIDTH = 121
    IMAGE_HEIGHT = 41
    
    CHAR_LEN = 6
    CHARACTERS = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}
    MODEL_PATH = "model/weights.h5"

    pred = ""

    try:
        import logging
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        import tensorflow as tf
        tf.autograph.set_verbosity(0)
        import CaptchaCracker as cc
        from PIL import Image

        img = Image.open(IMAGE_FILE)
        IMAGE_WIDTH = img.width
        IMAGE_HEIGHT = img.height

        AM = cc.ApplyModel(MODEL_PATH, IMAGE_WIDTH, IMAGE_HEIGHT, CHAR_LEN, CHARACTERS)
        pred = AM.predict(IMAGE_FILE)
        print(pred)
    except Exception as e:
        print("Error:", e)

    return pred

if("__main__" == __name__):
    argv = sys.argv

    if len(argv) < 2:
        print("Usage: python " + os.path.basename(argv[0]) + " IMAGE_FILE")
        sys.exit(-1)

    main(sys.argv[1])
else:
    print("module imported")