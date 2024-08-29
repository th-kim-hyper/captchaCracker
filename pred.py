from util import get_train_data_list
from core import Model

train_data_list = get_train_data_list()
train_data = train_data_list['default']

model = Model(train_data=train_data, quiet_out=False)
model.validate_model()

print("Done!")
