from cgi import FieldStorage
import sys, time, os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
print(BASE_DIR)

import util
from flask import Flask, Request, Response, render_template, flash, request, redirect, url_for, send_file, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import FileStorage
from PIL import Image
from pprint import pprint as pp
from core import Model
# from hyper import CaptchaType, Hyper

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "model")
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
DEFAULT_MODEL_TYPE = 'supreme_court'

app = Flask(__name__)
app.config['BASE_DIR'] = BASE_DIR
app.config['IMAGE_DIR'] = IMAGE_DIR 
app.config['MODEL_DIR'] = MODEL_DIR
app.config['UPLOAD_DIR'] = UPLOAD_DIR

train_data_list = util.get_train_data_list()
# train_data = train_data_list['wetax']

models = {}

for key in train_data_list.keys():
    train_data = train_data_list[key]
    model = Model(train_data=train_data, weights_only=False, quiet_out=True)
    models.update({key: model})

# pp(models.keys())

# models[CaptchaType.SUPREME_COURT.value] = Hyper(captcha_type=CaptchaType.SUPREME_COURT, weights_only=True, quiet_out=False)
# models[CaptchaType.GOV24.value] = Hyper(captcha_type=CaptchaType.GOV24, weights_only=True, quiet_out=False)

def predict(captcha_type, file:FileStorage, image_file_path=None):
    
    original_file_name = file.filename
    start_time = time.time()
    predict_model = models[captcha_type]
    # upload_file_name = f"{int(time.time())}.png"
    upload_file_name = original_file_name
    upload_dir = os.path.join(app.config['UPLOAD_DIR'], captcha_type)
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)
        
    save_path = os.path.join(upload_dir, upload_file_name)

    with Image.open(file.stream) as image:
        image = Image.open(file.stream)
        
        if(image.mode in ('RGBA', 'LA')):
            background = Image.new(image.mode[:-1], image.size, (255, 255, 255))
            background.paste(image, image.split()[-1])
            image = background
            
        image = image.convert(image.mode[:-1])
        image.save(save_path)  

    model:Model = models[captcha_type]
    predict_model = model.load_prediction_model()
    pred = predict_model.predict(save_path)
    # file_name = save_path.split(os.sep)[-1]
    
    p_time = time.time() - start_time
    
    result = {}
    result['model_type'] = captcha_type
    result['save_path'] = save_path
    result['file_name'] = original_file_name
    result['pred'] = pred
    result['p_time'] = p_time
    
    return result

# def setBGColor(image, fill_color = (255,255,255)):
#     color_bg = Image.new("RGBA", image.size, fill_color)
#     color_bg.paste(image, (0, 0), image)
#     color_bg = color_bg.convert("L")
#     image.close()
#     return color_bg

@app.route('/', methods=['GET', 'POST'])
def index(name=None):

    result = {}

    if request.method == 'POST':
        file = request.files['captchaFile']
        # file = request.files[request.files.keys()[0]]
        
        if(file != None):
            model_type = request.form['modelType'] if request.form['modelType'] is None else DEFAULT_MODEL_TYPE
            result = predict(model_type, file)

    return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def predictApi(name=None):

    result = {}

    if request.method == 'POST':
        file = request.files['captchaFile']
        # file = request.files[request.files.keys()[0]]
        
        if(file != None):
            model_type = request.form['modelType'] if request.form['modelType'] is None else DEFAULT_MODEL_TYPE
            result = predict(model_type, file)
        
    return jsonify(result)

@app.route('/images', methods=['GET'])
def images(name=None):
    model_type = request.args.get('t')
    file_name = request.args.get('f')
    filepath = os.path.join(UPLOAD_DIR, model_type, file_name)
    return send_file(filepath, mimetype='image/png')

if __name__ == '__main__':
    app.debug = True
    
    app.run("0.0.0.0")