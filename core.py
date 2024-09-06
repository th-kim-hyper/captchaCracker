import sys, os, time, absl.logging, logging
import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers, callbacks, backend
from util import TrainData

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
    
    def __init__(self, train_data:TrainData, weights_only=True, quiet_out=True):
        self.NULL_OUT = open(os.devnull, 'w')
        self.STD_OUT = sys.stdout
        self.quiet(quiet_out)
        self.captcha_data = train_data
        self.weights_only = weights_only
        self.quiet_out = quiet_out
        self.train_data = train_data
        # Mapping characters to integers
        self.char_to_num = layers.StringLookup(
            vocabulary=train_data.characters, num_oov_indices=0, mask_token=None
        )
        # Mapping integers back to original characters
        self.num_to_char = layers.StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )
        self.predict_model = None

    def quiet(self, value:bool):
        
        if value:
            logging.getLogger("tensorflow").setLevel(logging.ERROR)
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            tf.get_logger().setLevel('ERROR')
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
            absl.logging.set_verbosity(absl.logging.ERROR)
            # sys.stdout = self.NULL_OUT
        else:
            logging.getLogger("tensorflow").setLevel(logging.INFO)
            os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
            tf.get_logger().setLevel('INFO')
            tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
            absl.logging.set_verbosity(absl.logging.INFO)
            # sys.stdout = self.STD_OUT

    def split_dataset(self, batch_size=32, train_size=0.9, shuffle=True):
        # 1. Get the total size of the dataset
        images = np.array(self.train_data.train_data_list)
        labels = np.array(self.train_data.labels)
        size = len(images)
        # 2. Make an indices array and shuffle it, if required
        indices = np.arange(size)
        if shuffle:
            np.random.shuffle(indices)
        # 3. Get the size of training samples
        train_samples = int(size * train_size)
        # 4. Split data into training and validation sets
        x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
        x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = (
            train_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            .batch(batch_size)
            .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        )

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
        validation_dataset = (
            validation_dataset.map(
                self.encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
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
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        return {"image": image, "label": label}

    def build_model(self):
        # Inputs to the model
        input_img = layers.Input(
            shape=(self.captcha_data.image_width, self.captcha_data.image_height, 1), name="image", dtype="float32"
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
        new_shape = ((self.captcha_data.image_width // 4), (self.captcha_data.image_height // 4) * 64)
        x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
        x = layers.Dense(64, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)

        # RNNs
        x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

        # Output layer
        x = layers.Dense(len(list(self.captcha_data.characters)) + 1, activation="softmax", name="dense2")(x)

        # Add CTC layer for calculating CTC loss at each step
        output = CTCLayer(name="ctc_loss")(labels, x)

        # Define the model
        model = models.Model(
            inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
        )
        # Optimizer
        opt = optimizers.adam_v2.Adam()
        # Compile the model and return
        model.compile(optimizer=opt)
        return model
   
    def train_model(self, 
                    epochs=100, 
                    batch_size=32, 
                    earlystopping=True, 
                    early_stopping_patience:int=8, 
                    save_weights:bool=True,
                    save_model:bool=True):
        
        train_dataset, validation_dataset = self.split_dataset(batch_size=batch_size, train_size=0.9, shuffle=True)
        model = self.build_model()
                
        if earlystopping == True:
            early_stopping = callbacks.EarlyStopping(
                monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
            )
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs,
                callbacks=[early_stopping],
            )
        else:
            # Train the model
            history = model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=epochs
            )

        if save_model:
            model_path = self.train_data.get_model_path(False)
            model.save(model_path)
        
        if save_weights:
            weights_path = self.train_data.get_model_path(True)
            model.save_weights(weights_path)
   
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search
        results = backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
            :, :self.train_data.label_length
        ]
        # Iterate over the results and get back the text
        output_text = []
        for res in results:
            res = tf.strings.reduce_join(self.num_to_char(res+1)).numpy().decode("utf-8")
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

        prediction_model = models.Model(
            model.get_layer(name="image").input, model.get_layer(name="dense2").output
        )

        return prediction_model

    def predict(self, image_path:str, prediction_model=None):
        image_width = self.train_data.image_width
        image_height = self.train_data.image_height
        target_img = self.encode_single_sample(image_path, "")['image']
        target_img = tf.reshape(target_img, shape=[1,image_width,image_height,1])

        if prediction_model is None:
            prediction_model = self.load_prediction_model() if self.predict_model is None else self.predict_model
        
        self.predict_model = prediction_model
        pred_val = self.predict_model.predict(target_img)
        pred = self.decode_batch_predictions(pred_val)[0]

        return pred

    def validate_model(self):
        start = time.time()
        matched = 0
        prediction_model = self.load_prediction_model()
        pred_img_path_list = self.train_data.pred_data_list

        for pred_img_path in pred_img_path_list:
            pred = self.predict(pred_img_path, prediction_model=prediction_model)
            ori = os.path.basename(pred_img_path).split(".")[0]
            msg = ""
            if(ori == pred):
                matched += 1
            else:
                msg = " Not matched!"
            print("ori : ", ori, "pred : ", pred, msg)

        end = time.time()
        print("Matched:", matched, ", Tottal : ", len(pred_img_path_list), ", Accuracy : ", matched/len(pred_img_path_list) * 100, "%")
        print("pred time : ", end - start, "sec")
