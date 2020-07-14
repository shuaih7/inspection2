import os
import tensorflow as tf
from inspection2.model.lenet import LeNet_5

model = LeNet_5()
model.batch_size = 128
model.epochs = 10
model.is_shuffle = True

model.log_dir = r"E:\Deep_Learning\inspection2\lenet_5\log"
model.model_dir = r"E:\Deep_Learning\inspection2\lenet_5\model"

model.train()

