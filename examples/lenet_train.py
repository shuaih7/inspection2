import os
import tensorflow as tf
from inspection2.models.lenet import LeNet_5
from inspection2.data.params import DataParamCls

model = LeNet_5()
batch_size = 128
epochs = 10
is_shuffle = True

model.log_dir = r"C:\projects\inspection2\lenet\log"
model.model_dir = r"C:\projects\inspection2\lenet\model"
data_dir = r"C:\projects\inspection2\dataset\mnist"

data_param = DataParamCls()
data_param.x_train_path = data_dir
data_param.y_train_path = data_dir
data_param.x_valid_path = data_dir
data_param.y_valid_path = data_dir
data_param.num_classes  = 10

model.load_data(data_param)
model.train(batch_size=batch_size, epochs=epochs, shuffle=is_shuffle)

