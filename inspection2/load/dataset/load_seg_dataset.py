# -*- coding: utf-8 -*-

# Header ...

import os, cv2
import numpy as np
from .base import LoadDataset
from inspection2.data.preprocess import norm_pixel
from inspection2.utils import read_lines_from_txt, np_dtype, be_dtype


class LoadSegDataset(LoadDataset):
    def __init__(self, data_param, logger=None):
        super(LoadSegDataset, self).__init__(data_param=data_param, logger=logger)
        
    def config_init(self.init_func=None):
        if init_func is not None: self.init_func = init_func
        elif self.init_func is None: self.init_func = read_list_from_txt
        
    def config_maps(self, map_func=[], map_ext_args=[]):
        if len(map_func): 
            self.map_func = map_func
            self.map_ext_args = map_ext_args
        elif len(self.map_func) == 0:    
            load_func          = read_seg_data
            load_ext_arg       = []
            resize_func        = resize_seg_data
            resize_ext_arg     = [self.data_param.image_size, self.data_param.label_size]
            preprocess_func    = norm_pixel
            preprocess_ext_arg = ["x"]
            
            self.map_func     = [load_func, resize_func, preprocess_func]
            self.map_ext_args = [load_ext_arg, resize_ext_arg, preprocess_ext_arg]
      
        if len(map_func) != len(map_ext_args):
            self.logger.error("The lengths of map functions and the extra argument not matching.")

"""
def load_seg_dataset(data_param, load_func=read_seg_data, preprocess_func=no_preprocess, resize_func=resize_seg_data):
    train_ds, valid_ds = None, None
    
    if data_param.train_txt_file is not None: train_ds = load_single_seg_dataset(data_param, load_func=read_seg_data, preprocess_func=preprocess_func,
                                                                                 resize_func=resize_seg_data, mode="train")
    if data_param.valid_txt_file is not None: valid_ds = load_single_seg_dataset(data_param, load_func=read_seg_data, preprocess_func=preprocess_func,
                                                                                 resize_func=resize_seg_data, mode="valid")
    return train_ds, valid_ds
    

def load_single_seg_dataset(data_param, load_func=read_seg_data, preprocess_func=no_preprocess, resize_func=resize_seg_data, mode="train"):
    if mode == "train":
        txt_file = data_param.train_txt_file
        x_path   = data_param.x_train_path
        y_path   = data_param.y_train_path
    elif mode == "valid":
        txt_file = data_param.valid_txt_file
        x_path   = data_param.x_valid_path
        y_path   = data_param.y_valid_path
    else: raise ValueError("The mode is not valid.")
                            
    x_dtype   = data_param.image_dtype
    y_dtype   = data_param.label_dtype
    item_list = read_lines_from_txt(txt_file)
    x_list    = [os.path.join(x_path, l + x_suffix) for l in item_list]
    y_list    = [os.path.join(y_path, l + y_suffix) for l in item_list]
    
    # Creating the dataset API for data pipelining
    dataset = Dataset.from_tensor_slice((x_list, y_list))
    
    # Pipelining the loading processes 
    dataset = dataset.map(lambda x_item, y_item: tuple(py_function(load_func, [x_item, y_item, np_dtype(x_dtype), np_dtype(y_dtype)], 
                                                       Tout = [be_dtype(x_dtype), be_dtype(y_dtype)])), 
                          num_parallel_calls=data_param.num_parallel_calls)
    
    # Pipelining the preprocesisng processes
    dataset = dataset.map(lambda x, y: tuple(py_function(preprocess_func, [x_item, y_item, np_dtype(x_dtype), np_dtype(y_dtype)], 
                                                       Tout = [be_dtype(x_dtype), be_dtype(y_dtype)])), 
                          num_parallel_calls=data_param.num_parallel_calls)
    
    # Resize the input images - width and height dimension using opencv explicitly
    # Please make sure all of the images have the same channel number before feeding
    # Only explicit resize mode is supported in this time
    dataset = dataset.map(lambda x, y: tuple(py_function(resize_func, [x, y, data_param.image_size, data_param.label_size],
                                             Tout = [be_dtype(x_type), be_dtype(y_type)])),
                          num_parallel_calls=data_param.num_parallel_calls)

    return dataset
"""    

def read_list_from_txt(data_param, mode="train"):
    if mode == "train":   
        x_path   = data_param.x_train_path
        y_dtype  = data_param.y_train_path
        txt_file = data_param.train_txt_file
    elif mode == "valid": 
        x_path   = data_param.x_valid_path
        y_dtype  = data_param.y_valid_path
        txt_file = data_param.valid_txt_file
    else: raise ValueError("Invalid input mode.")
    
    item_list = read_lines_from_txt(txt_file)
    x_list    = [os.path.join(x_path, l + x_suffix) for l in item_list]
    y_list    = [os.path.join(y_path, l + y_suffix) for l in item_list]
    
    return x_list, y_list
    

def read_seg_data(x_item, y_item):
    x_data = cv2.imread(x_item, -1)
    y_data = cv2.imread(y_item, -1)
    
    if x_data.shape[-1] not in [1,3,4]: x_data = np.expand_dims(x_data, axis=-1)
    if y_data.shape[-1] not in [1,3,4]: y_data = np.expand_dims(y_data, axis=-1)
    
    #x_data = x_data.astype(x_type)
    #y_data = y_data.astype(y_type)
    
    return x_data, y_data
    
    
def resize_seg_data(x, y, x_size, y_size):
    xh, xw = x.shape[:2]  # Original image height 
    yh, yw = y.shape[:2]  # Original label width
    x_h, x_w, _ = x_size  # Target image height
    y_h, y_w, _ = y_size  # Target label width
    
    x_resize, y_resize = x, y # Initialize
    
    if x_h or x_w: x_resize = cv2.resize(x, (max(xw, x_w), max(xh, x_h)), interpolation = cv2.INTER_LINEAR)
    if y_h or y_w: y_resize = cv2.resize(y, (max(yw, y_w), max(yh, y_h)), interpolation = cv2.INTER_NEAREST)
    
    return x_resize, y_resize   
