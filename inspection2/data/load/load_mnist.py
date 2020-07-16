# -*- coding: utf-8 -*-

# Header ...

import os, struct, math
import numpy as np
from inspection2.backend.utils import to_categorical as one_hot


def load_mnist(data_param, train_kind="train", valid_kind="t10k"):
    x_train_path, y_train_path = data_param.x_train_path, data_param.y_train_path
    x_valid_path, y_valid_path = data_param.x_valid_path, data_param.y_valid_path
    
    x_train, y_train = load_single_mnist(x_train_path, y_train_path, kind=train_kind, num_classes=data_param.num_classes)
    x_valid, y_valid = load_single_mnist(x_valid_path, y_valid_path, kind=valid_kind, num_classes=data_param.num_classes)
    
    num, params = x_train.shape[:2]
    size = int(math.sqrt(params))
    if x_train is not None: x_train = np.array(x_train.reshape((num, size, size, 1)), dtype=np.float32)
    if y_train is not None: y_train = one_hot(y_train, num_classes=data_param.num_classes)
    
    num, params = x_valid.shape[:2]
    if x_valid is not None: x_valid = np.array(x_valid.reshape((num, size, size, 1)), dtype=np.float32)
    if y_valid is not None: y_valid = one_hot(y_valid, num_classes=data_param.num_classes)
    
    return x_train, y_train, x_valid, y_valid
    
    
def load_single_mnist(image_path, label_path, kind='train', num_classes=10):    
    labels_path = os.path.join(image_path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(label_path,'%s-images.idx3-ubyte'% kind)
    
    if os.path.isfile(labels_path):
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            labels = np.fromfile(lbpath,dtype=np.uint8)
    else: labels = None
        
    if os.path.isfile(labels_path):
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    else: images = None
           
    return images, labels