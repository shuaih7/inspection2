# -*- coding: utf-8 -*-

# Header ...

from inspection2.backend import py_function
from inspection2.backend.data import Dataset

def load_dataset(train_list, valid_list, x_suffix, y_suffix, 
                 load_x_func, load_y_func, num_parallel_calls=4):
    pass
    

def load_single_dataset(file_list, num_parallel_calls=4):
    dataset = Dataset.from_tensor_slice((x_path, y_path))
    dataset = dataset.map(lambda x_list, y_list: tuple(py_function(fetch_list_from_file, file_list)), 
                num_parallel_calls=num_parallel_calls)
    


        
    
