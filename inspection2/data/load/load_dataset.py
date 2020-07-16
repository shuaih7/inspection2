# -*- coding: utf-8 -*-

# Header ...

import numpy as np
from inspection2.backend import py_function
from inspection2.backend.data import Dataset
from inspection2.utils import read_list_from_txt


def load_seg_dataset(data_param, load_x_func, load_y_func):
    pass
    
    

def load_single_seg_dataset(txt_file, x_path, x_suffix, y_path, y_suffix, img_size=None, num_parallel_calls=4):
    item_list = read_list_from_txt(txt_file)
    x_list    = [os.path.join(x_path, l + x_suffix) for l in item_list]
    y_list    = [os.path.join(y_path, l + y_suffix) for l in item_list]
    
    dataset = Dataset.from_tensor_slice((x_list, y_list))
    dataset = dataset.map(lambda x_item, y_item: tuple(py_function(read_seg_data, [x_item, y_item], [np.float32, np.uint8])), 
                          num_parallel_calls=num_parallel_calls)
    
    if img_size is not None: 
        
    
    return dataset
    

def read_seg_data(x_item, y_item, x_type=np.float32, y_type=np.uint8):
    x_data = cv2.imread(x_item, -1)
    y_data = cv2.imread(y_item, -1)
    
    x_data = x_data.astype(np.float32)
    y_data = y_data.astype(y_type)
    
    return x_data, y_data
        
    
