import os
import tensorflow as tf
from inspection2.model.lenet import LeNet_5

model = LeNet_5()
model.batch_size = 128
model.epochs = 10
model.is_shuffle = True

model.log_dir = r"C:\projects\inspection2\lenet\log"
model.model_dir = r"C:\projects\inspection2\lenet\model"
data_dir = r"C:\projects\inspection2\dataset\mnist"

model.load_data(data_dir, data_dir, data_dir, data_dir, num_classes=10)
model.train()

