# -*- coding: utf-8 -*-

# Header ...

import numpy as np
import struct, os, math
from tensorflow.keras.utils import to_categorical as one_hot


def load_mnist(x_train_path, y_train_path, x_valid_path, y_valid_path, train_kind="train", valid_kind="t10k", num_classes=10):    
    #x_train_path, y_train_path, x_valid_path, y_valid_path = paths
    x_train_file = os.path.join(x_train_path, '%s-images.idx3-ubyte'% train_kind)
    y_train_file = os.path.join(y_train_path, '%s-labels.idx1-ubyte'% train_kind)
    x_valid_file = os.path.join(x_valid_path, '%s-images.idx3-ubyte'% valid_kind)
    y_valid_file = os.path.join(y_valid_path, '%s-labels.idx1-ubyte'% valid_kind)
    
    if os.path.isfile(y_train_file):
        with open(y_train_file, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            y_train = np.fromfile(lbpath,dtype=np.uint8)
            
            y_train = one_hot(y_train, num_classes=num_classes)
    else: y_train = None
    
    if os.path.isfile(x_train_file):
        with open(x_train_file, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            x_train = np.fromfile(imgpath,dtype=np.uint8).reshape(len(y_train), 784)
            
            num, params = x_train.shape[:2]
            size = int(math.sqrt(params))
            x_train = np.array(x_train.reshape((num, size, size, 1)), dtype=np.float32)
    else: x_train = None

    if os.path.isfile(y_valid_file):
        with open(y_valid_file, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            y_valid = np.fromfile(lbpath,dtype=np.uint8)
            
            y_valid = one_hot(y_valid, num_classes=num_classes)
    else: y_valid = None
    
    if os.path.isfile(x_valid_file):        
        with open(x_valid_file, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            x_valid = np.fromfile(imgpath,dtype=np.uint8).reshape(len(y_valid), 784)
            
            num, params = x_valid.shape[:2]
            size = int(math.sqrt(params))
            x_valid = np.array(x_valid.reshape((num, size, size, 1)), dtype=np.float32)
    else: x_valid = None
    
    return x_train, y_train, x_valid, y_valid
        
    
