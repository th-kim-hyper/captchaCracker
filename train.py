from util import load_config, get_train_data_list
from core import Model

# config = load_config('config.yaml')
# data_base_dir = config['data_base_dir']
# model_base_dir = config['model_base_dir']
train_data_list = get_train_data_list()
train_data = train_data_list['wetax']
epochs = 100
batch_size = 32
earlystopping = True
early_stopping_patience = 10
save_weights = True
save_model = True

model = Model(train_data=train_data, quiet_out=False)
model.train_model(
    epochs=epochs,
    batch_size=batch_size,
    earlystopping=earlystopping,
    early_stopping_patience=early_stopping_patience,
    save_weights=save_weights,
    save_model=save_model)

print("Done!")
