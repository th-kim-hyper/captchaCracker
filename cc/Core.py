import os, glob, time
from PIL import Image
from dataclasses import dataclass
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, callbacks, backend
from typing import Final

DIGITS: Final = "0123456789"
LOWER_CASE: Final = "abcdefghijklmnopqrstuvwxyz"
UPPER_CASE: Final = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ALPHABET: Final = LOWER_CASE + UPPER_CASE
ALPHA_NUMERIC: Final = DIGITS + ALPHABET

def get_captcha_type_list(image_dir: str = "./images", model_dir: str = "./model"):
    default = CaptchaType(id="default", name="기본값", desc="기본 학습 데이터", image_dir=image_dir, model_dir=model_dir)
    supreme_court = CaptchaType(id="supreme_court", name="대법원", desc="대법원 학습 데이터", image_dir=image_dir, model_dir=model_dir)
    gov24 = CaptchaType(id="gov24", name="gov24", desc="대한민국 정부 24 학습 데이터", image_dir=image_dir, model_dir=model_dir)
    wetax = CaptchaType(id="wetax", name="wetax", desc="지방세 납부/조회 학습 데이터", image_dir=image_dir, model_dir=model_dir)

    return {
        "default": default,
        "supreme_court": supreme_court,
        "gov24": gov24,
        "wetax": wetax,
    }

def setBG(image_path, color=(255,255,255)):
    with Image.open(image_path) as img:
        fill_color = color
        # img = img.convert(img.mode)
        img_mode = img.mode
        if img_mode in ('RGBA', 'LA'):
            background = Image.new(img_mode[:-1], img.size, fill_color)
            background.paste(img, img.split()[-1]) # omit transparency
            background.save(image_path)

def convert_transparent_to_white(image_path: str, output_path: str):
    with Image.open(image_path) as img:
        if img.mode in ('RGBA', 'LA'):
            alpha = img.convert(img.mode).split()[-1]
            bg = Image.new(img.mode, img.size, (255, 255, 255) + (255,))
            bg.paste(img, mask=alpha)
            img = bg.convert(alpha)
        img.save(output_path)

@dataclass
class TrainData:
    id: str = "default"
    image_dir: str = "./images"
    model_dir: str = "./model"
    image_width: int = 0
    image_height: int = 0
    label_length: int = 0
    characters = list(DIGITS)
    init:bool = True

    def __post_init__(self):
        if self.init == True:
            (
                self.image_width,
                self.image_height,
                self.label_length,
                self.characters
            ) = self.get_train_info()

    def get_train_info(self):
        train_data_list = self.get_data_files(train=True)
        
        with Image.open(train_data_list[0]) as image:
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
        ) 

    def get_data_files(self, train=True):
        data_dir = os.path.join(
            self.image_dir, self.id, "train" if train else "pred"
        )
        return glob.glob(data_dir + os.sep + "*.png")

    def get_labels(self, train=True):
        return [
            os.path.basename(data_path).split(".")[0]
            for data_path in self.get_data_files(train)
        ]

    def get_model_path(self, weights_only=False):
        weights_path = os.path.join(self.model_dir, self.id)

        if os.path.exists(weights_path) == False:
            os.makedirs(weights_path)

        if weights_only:
            weights_path = os.path.join(weights_path, "weights.h5")

        return weights_path

@dataclass
class CaptchaType:
    id: str
    name: str
    desc: str
    image_dir: str
    model_dir: str
    data: TrainData = None

    def __post_init__(self):
        self.data = TrainData(id=self.id, image_dir=self.image_dir, model_dir=self.model_dir)

class CTCLayer(layers.Layer):

    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred

class Model:

    def __init__(self, train_data: TrainData, weights_only = True, save_model=False, save_weights=True, verbose=1):
        self.train_data = train_data
        self.weights_only = weights_only
        self.save_model = save_model
        self.save_weights = save_weights
        self.char_to_num = layers.StringLookup(
            vocabulary=train_data.characters, num_oov_indices=0, mask_token=None
        )
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.predict_model = None
        self.verbose = verbose

    def split_dataset(self, batch_size=32, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        images = np.array(self.train_data.get_data_files(train=True))
        labels = np.array(self.train_data.get_labels(train=True))
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = (
            images[indices[:train_samples]],
            labels[indices[:train_samples]],
        )
        x_valid, y_valid = (
            images[indices[train_samples:]],
            labels[indices[train_samples:]],
        )

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample,
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        return train_dataset, validation_dataset

    def encode_single_sample(self, image_path, label):
        image_width = self.train_data.image_width
        image_height = self.train_data.image_height
        image = tf.io.read_file(image_path)
        image = tf.io.decode_png(image, channels=1)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, [image_height, image_width])
        image = tf.transpose(image, perm=[1, 0, 2])
        label = self.char_to_num(
            tf.strings.unicode_split(label, input_encoding="UTF-8")
        )
        return {"image": image, "label": label}

    def build_model(self):
        # Inputs to the model
        input_img = layers.Input(
            shape=(self.train_data.image_width, self.train_data.image_height, 1),
            name="image",
            dtype="float32",
        )
        labels = layers.Input(name="label", shape=(None,), dtype="float32")

        # First conv block
        x = layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv1",
        )(input_img)
        x = layers.MaxPooling2D((2, 2), name="pool1")(x)

        # Second conv block
        x = layers.Conv2D(
            64,
            (3, 3),
            activation="relu",
            kernel_initializer="he_normal",
            padding="same",
            name="Conv2",
        )(x)
        x = layers.MaxPooling2D((2, 2), name="pool2")(x)

        # We have used two max pool with pool size and strides 2.
        # Hence, downsampled feature maps are 4x smaller. The number of
        # filters in the last layer is 64. Reshape accordingly before
        # passing the output to the RNN part of the model
        new_shape = (
            (self.train_data.image_width // 4),
            (self.train_data.image_height // 4) * 64,
        )
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(
            x
        )
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(
            x
        )

        # Output layer
        x = layers.Dense(
            len(list(self.train_data.characters)) + 1,
            activation="softmax",
            name="dense2",
        )(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        
        opts_props = dir(optimizers)
        
        if 'Adam' in opts_props:
            opt = optimizers.Adam()
        else:
            opt = optimizers.adam_v2.Adam()

        # Compile the model and return
        model.compile(optimizer=opt)
        return model

    def train_model(
        self,
        epochs=100,
        batch_size=32,
        earlystopping=True,
        early_stopping_patience: int = 8,
        save_weights: bool = True,
        save_model: bool = True,
    ):

        train_dataset, validation_dataset = self.split_dataset(
            batch_size=batch_size, train_size=0.9, shuffle=True
        )
        model = self.build_model()

        if earlystopping == True:
            early_stopping = callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
            )
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=[early_stopping],
                verbose=1,
            )
        else:
            # Train the model
            history = model.fit(
                train_dataset, validation_data=validation_dataset, epochs=epochs, verbose=self.verbose,
            )

        if save_weights:
            weights_path = self.train_data.get_model_path(True)
            model.save_weights(weights_path)

        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)

        if save_model:
            model_path = self.train_data.get_model_path(False)
            print("model_path : ", model_path)
            model.save(model_path)

    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, : self.train_data.label_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = (
                tf.strings.reduce_join(self.num_to_char(res + 1))
                .numpy()
                .decode("utf-8")
            )
            output_text.append(res)
        return output_text

    def load_prediction_model(self):

        if self.weights_only:
            model = self.build_model()
            weights_path = self.train_data.get_model_path(weights_only=True)
            model.load_weights(weights_path)
        else:
            weights_path = self.train_data.get_model_path(weights_only=False)
            model = models.load_model(weights_path)

        self.predict_model = models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )

        return self.predict_model

    def predict(self, image_path: str):
        image_width = self.train_data.image_width
        image_height = self.train_data.image_height
        target_img = self.encode_single_sample(image_path, "")["image"]
        target_img = tf.reshape(target_img, shape=[1, image_width, image_height, 1])

        if self.predict_model is None:
            self.load_prediction_model()
    
        pred_val = self.predict_model.predict(target_img, verbose=self.verbose)
        pred = self.decode_batch_predictions(pred_val)[0]

        confidence = np.max(pred_val, axis=-1).mean()

        return pred, confidence

    def validate_model(self):
        start = time.time()
        matched = 0
        pred_img_path_list = self.train_data.get_data_files(train=False)

        for pred_img_path in pred_img_path_list:
            self.verbose = 0
            pred, confidence = self.predict(pred_img_path)
            ori = os.path.basename(pred_img_path).split(".")[0]
            msg = ""
            if ori == pred:
                matched += 1
            else:
                msg = " Not matched!"
            print("ori : ", ori, "pred : ", pred, "confidence : ", confidence, msg)

        end = time.time()
        print(
            "Matched:",
            matched,
            ", Tottal : ",
            len(pred_img_path_list),
            ", Accuracy : ",
            matched / len(pred_img_path_list) * 100,
            "%",
        )
        print("pred time : ", end - start, "sec")
