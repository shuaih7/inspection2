# -*- coding: utf-8 -*-

# Header ...

import numpy as np
from inspection2.backend import py_function
from inspection2.backend.data import Dataset
from inspection2.utils import read_list_from_txt, np_dtype, be_dtype


def load_seg_dataset(data_param, load_x_func, load_y_func):
    pass
    
    

def load_single_seg_dataset(data_param, mode="train"):
    if mode == "train":
        txt_file = data_param.train_txt_file
        x_path   = data_param.x_train_path
        y_path   = data_param.y_train_path
    elif mode == "valid":
        txt_file = data_param.valid_txt_file
        x_path = data_param.x_valid_path
        y_path = data_param.y_valid_path
    else: raise ValueError("The mode is not valid.")
                            
    x_dtype   = data_param.image_dtype
    y_dtype   = data_param.label_dtype
    item_list = read_list_from_txt(txt_file)
    x_list    = [os.path.join(x_path, l + x_suffix) for l in item_list]
    y_list    = [os.path.join(y_path, l + y_suffix) for l in item_list]
    
    dataset = Dataset.from_tensor_slice((x_list, y_list))
    dataset = dataset.map(lambda x_item, y_item: tuple(py_function(read_seg_data, [x_item, y_item, np_dtype(x_dtype), np_dtype(y_dtype)], 
                                                       Tout = [be_dtype(x_dtype), be_dtype(y_dtype)])), 
                          num_parallel_calls=data_param.num_parallel_calls)
    
    if data_param.image_h is not None and data_param.image_w is not None:
        if explicit_resize:
            dataset = dataset.map(lambda x_item, y_item: tuple(py_function(resize_seg_data, [])))
        
        
    
    return dataset
    

def read_seg_data(x_item, y_item, x_type=np.float32, y_type=np.uint8):
    x_data = cv2.imread(x_item, -1)
    y_data = cv2.imread(y_item, -1)
    
    x_data = x_data.astype(x_type)
    y_data = y_data.astype(y_type)
    
    return x_data, y_data
    
    
def resize_seg_data()
        
    
