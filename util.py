import os, glob, yaml
from PIL import Image
from dataclasses import dataclass

digits = ['0','1','2','3','4','5','6','7','8','9']
lowercase = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
uppercase = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
alphanumeric = digits + lowercase + uppercase

def load_config(cfg_file):
    with open(cfg_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def get_train_data_list():
    default = TrainData('DEFAULT', 'default', '기본 학습 데이터')
    supreme_court = TrainData('SUPREME_COURT', 'supreme_court', '대법원 학습 데이터')
    gov24 = TrainData('GOV24', 'gov24', '대한민국 정부 24 학습 데이터')
    wetax = TrainData('WETAX', 'wetax', '지방세 납부/조회 학습 데이터')
    
    return {
        'default':default, 'supreme_court':supreme_court, 'gov24':gov24, 'wetax':wetax
    }

def setBG(image_path, color=(255,255,255)):
        img = Image.open(image_path)
        fill_color = color
        img = img.convert("RGBA")
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, fill_color)
            background.paste(img, img.split()[-1]) # omit transparency
            img = background
        image_path = "./temp_white_bg.png"
        img.save(image_path)
        return image_path

@dataclass
class TrainData():
    id: str = 'SUPREME_COURT'
    name: str = 'supreme_court'
    description: str = '대법원 학습 데이터'
    data_base_dir: str = "./images"
    model_base_dir: str = "./model"
    image_width: int = 0
    image_height: int = 0
    label_length: int = 0
    characters: list = None
    train_data_list: list = None
    pred_data_list: list = None
    labels: list = None 

    def __post_init__(self):
        (   self.image_width, 
            self.image_height, 
            self.label_length, 
            self.characters, 
            self.train_data_list,
            self.pred_data_list,
            self.labels ) = self.get_train_info()

    def get_data_files(self, train=True):
        data_dir = os.path.join(self.data_base_dir, self.name, "train" if train else "pred")
        return glob.glob(data_dir + os.sep + "*.png")

    def get_train_info(self):
        train_data_list = self.get_data_files(train=True)
        pred_data_list = self.get_data_files(train=False)
        image = Image.open(train_data_list[0])
        image_width, image_height = image.size
        labels = [os.path.basename(data_path).split(".")[0] for data_path in train_data_list]
        label_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        return image_width, image_height, label_length, characters, train_data_list, pred_data_list, labels    

    def get_model_path(self, weights_only=False):
        weights_path = os.path.join(self.model_base_dir, self.name)
        
        if os.path.exists(weights_path) == False:
            os.makedirs(weights_path)
        
        if weights_only:
            weights_path = os.path.join(weights_path, "weights.h5")

        return weights_path

# @dataclass
# class Captcha():
#     id: str = 'SUPREME_COURT'
#     name: str = 'supreme_court'
#     description: str = '대법원 캡챠'
#     train_data:TrainData = None

def load_config(cfg_file):
    with open(cfg_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def get_train_data_list():
    default = TrainData('DEFAULT', 'default', '기본 학습 데이터')
    supreme_court = TrainData('SUPREME_COURT', 'supreme_court', '대법원 학습 데이터')
    gov24 = TrainData('GOV24', 'gov24', '대한민국 정부 24 학습 데이터')
    wetax = TrainData('WETAX', 'wetax', '지방세 납부/조회 학습 데이터')
    return {'default':default, 'supreme_court':supreme_court, 'gov24':gov24, 'wetax':wetax}

def setBG(image_path, color=(255,255,255)):
        img = Image.open(image_path)
        fill_color = color
        img = img.convert("RGBA")
        if img.mode in ('RGBA', 'LA'):
            background = Image.new(img.mode[:-1], img.size, fill_color)
            background.paste(img, img.split()[-1]) # omit transparency
            img = background
        image_path = "./temp_white_bg.png"
        img.save(image_path)
        return image_path
