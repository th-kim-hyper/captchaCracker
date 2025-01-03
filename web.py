# from cgi import FieldStorage
import sys, time, os
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
from flask import Flask, render_template, request, jsonify, send_file
# from werkzeug.datastructures import FileStorage
# from PIL import Image
from cc.Core import get_captcha_type_list, CaptchaType, TrainData, Model, setBG, convert_transparent_to_white
import base64
from flask_cors import CORS
from PIL import Image

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "model")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
UPLOAD_DIR = os.path.join(BASE_DIR, "upload")
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
models = {}

app = Flask(__name__)
CORS(app)
app.config['BASE_DIR'] = BASE_DIR
app.config['IMAGE_DIR'] = IMAGE_DIR 
app.config['MODEL_DIR'] = MODEL_DIR
app.config['UPLOAD_DIR'] = UPLOAD_DIR
app.config['MODELS'] = models

captcha_type_list = get_captcha_type_list(image_dir=IMAGE_DIR, model_dir=MODEL_DIR)
captcha_list = captcha_type_list.values()

for key in captcha_type_list.keys():
    captcha_type:CaptchaType = captcha_type_list[key]
    train_data:TrainData = captcha_type.data
    model = Model(train_data=train_data)
    model.load_prediction_model()
    models.update({key: model})

def predict(captcha_id, captcha_file):
    start_time = time.time()
    model:Model = models[captcha_id]
    pred, confidence = model.predict(captcha_file)
    p_time = time.time() - start_time
    
    result = {}
    result['captcha_id'] = captcha_id
    result['captcha_file'] = captcha_file
    result['pred'] = pred
    result['confidence'] = float(confidence)
    result['p_time'] = p_time

    return result

@app.route('/', methods=['GET', 'POST'])
def index(name=None):

    captcha_file = None
    captcha_id = None
    captcha_text = None
    result = {}

    if request.method == 'POST' and 'captchaFile' in request.files:
        captcha_file = request.files['captchaFile']
        captcha_id = request.form['modelType']
        captcha_text = request.form['captchaText']
        result['captcha_text'] = captcha_text
        result['captcha_id'] = captcha_id
        
        if(captcha_file != None):
            result = predict(captcha_id, captcha_file)

    return render_template('index.html', captcha_list=captcha_list, result=result)

@app.route('/predictPost', methods=['POST'])
def predictPost(name=None):
    captcha_file = None
    captcha_id = None
    result = {}

    if request.method == 'POST' and 'captcha_file' in request.files:
        captcha_file = request.files['captcha_file']
        captcha_id = request.form['captcha_id']
        result['captcha_id'] = captcha_id
        
        if(captcha_file != None):
            file_name = captcha_file.filename
            upload_dir = os.path.join(UPLOAD_DIR, captcha_id)
            image_path = os.path.join(upload_dir, file_name)

            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            captcha_file.save(image_path)
            result = predict(captcha_id, image_path)
    
    return jsonify(result)
    # return render_template('index.html', result=result)

@app.route('/api/predict', methods=['POST'])
def predictApi(name=None):
    captcha_file = None
    captcha_id = None
    result = {}

    if request.method == 'POST' and 'captcha_data_url' in request.form:
        captcha_id = request.form['captcha_id']
        captcha_data_url = request.form['captcha_data_url']
        timestamp = int(time.time())
        file_name = f"captcha_{timestamp}.png"
        tmp_path = os.path.join(TEMP_DIR, file_name)

        # Decode the base64 data URL
        header, encoded = captcha_data_url.split(",", 1)
        captcha_data = base64.b64decode(encoded)

        # Convert the data to an image
        # captcha_file = BytesIO(captcha_data)
        if not os.path.exists(TEMP_DIR):
            os.makedirs(TEMP_DIR)

        with open(tmp_path, "wb") as f:
            f.write(captcha_data)

        setBG(tmp_path)
        # convert_transparent_to_white(tmp_path, tmp_path)            
        captcha_image = Image.open(tmp_path)
        # Crop 1 pixel border from the image
        width, height = captcha_image.size
        captcha_file = captcha_image.crop((1, 1, width - 1, height - 1))
        result['captcha_id'] = captcha_id
        
        if(captcha_file != None):
            upload_dir = os.path.join(UPLOAD_DIR, captcha_id)
            image_path = os.path.join(upload_dir, file_name)

            if not os.path.exists(upload_dir):
                os.makedirs(upload_dir)

            captcha_file.save(image_path)
            result = predict(captcha_id, image_path)
    
    return jsonify(result)

@app.route('/images', methods=['GET'])
def images(name=None):
    captcha_id = request.args.get('t')
    captcha_file = request.args.get('f')
    filepath = os.path.join(UPLOAD_DIR, captcha_id, captcha_file)
    return send_file(filepath, mimetype='image/png')

if __name__ == '__main__':
    app.debug = False    
    app.run("0.0.0.0")