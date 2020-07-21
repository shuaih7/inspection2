# -*- coding: utf-8 -*-

# Header ...

import numpy as np
import os, struct, math
from inspection2.utils import one_hot
from inspection2.data.preprocess import norm_pixel
from inspection2.load.dataset.base import LoadDataset


class LoadMnist(LoadDataset):
    def __init__(self, data_param, logger=None):
        super(LoadMnist, self).__init__(data_param=data_param, logger=logger)
        
    def config_init(self, init_func=None):
        if init_func is not None: self.init_func = init_func
        elif self.init_func is None: self.init_func = read_mnist_from_file
        
    def config_maps(self, map_func=[], map_ext_args=[]):
        if len(map_func): 
            self.map_func = map_func
            self.map_ext_args = map_ext_args
        elif len(self.map_func) == 0:
            preprocess_func    = norm_pixel
            preprocess_ext_arg = []
            
            self.map_func     = [preprocess_func]
            self.map_ext_args = [preprocess_ext_arg]
      
        if len(map_func) != len(map_ext_args):
            self.logger.error("The lengths of map functions and the extra argument not matching.")
            

def read_mnist_from_file(data_param, mode="train", train_kind="train", valid_kind="t10k"):
    if mode == "train":   
        kind     = train_kind
        x_path   = data_param.x_train_path
        y_dtype  = data_param.y_train_path
    elif mode == "valid": 
        kind     = valid_kind
        x_path   = data_param.x_valid_path
        y_dtype  = data_param.y_valid_path
    else: raise ValueError("Invalid input mode.")
    
    # Load mnist data from file
    labels_path = os.path.join(x_path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(x_path,'%s-images.idx3-ubyte'% kind)
    
    if os.path.isfile(labels_path):
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            labels = np.fromfile(lbpath,dtype=np.uint8)
    else: labels = None
        
    if os.path.isfile(images_path):
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    else: images = None
    
    # Reshape the loaded numpy arrays
    num, params = images.shape[:2]
    size = int(math.sqrt(params))
    images = np.array(images.reshape((num, size, size, 1)), dtype=np.float32)
    labels = one_hot(labels, num_classes=data_param.num_classes)
           
    return images, labels

    