from dataclasses import dataclass
import os
import glob
from PIL import Image

@dataclass
class TrainData:
    id: str = "SUPREME_COURT"
    name: str = "supreme_court"
    description: str = "대법원 학습 데이터"
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
        (
            self.image_width,
            self.image_height,
            self.label_length,
            self.characters,
            self.train_data_list,
            self.pred_data_list,
            self.labels,
        ) = self.get_train_info()

    def get_data_files(self, train=True):
        data_dir = os.path.join(
            self.data_base_dir, self.name, "train" if train else "pred"
        )
        return glob.glob(data_dir + os.sep + "*.png")

    def get_train_info(self):
        train_data_list = self.get_data_files(train=True)
        pred_data_list = self.get_data_files(train=False)
        image = Image.open(train_data_list[0])
        image_width, image_height = image.size
        labels = [
            os.path.basename(data_path).split(".")[0] for data_path in train_data_list
        ]
        label_length = max([len(label) for label in labels])
        characters = sorted(set(char for label in labels for char in label))
        return (
            image_width,
            image_height,
            label_length,
            characters,
            train_data_list,
            pred_data_list,
            labels,
        )

    def get_model_path(self, weights_only=False):
        weights_path = os.path.join(self.model_base_dir, self.name)

        if os.path.exists(weights_path) == False:
            os.makedirs(weights_path)

        if weights_only:
            weights_path = os.path.join(weights_path, "weights.h5")

        return weights_path
