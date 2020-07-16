# -*- coding: utf-8 -*-

# Header ...

import os
#from inspection2.utils import read_list_from_txt


class DataParam(object):

    def __init__(self):
    
        # General parameters
        self._x_train_path = None
        self._y_train_path = None
        self._x_valid_path = None
        self._y_valid_path = None
        
        self.x_suffix = ""
        self.y_suffix = ""
        
        # Dataset Parameters
        self._train_txt_file     = None
        self._valid_txt_file     = None
        self._shuffle_size       = 1000
        self._num_parallel_calls = 4
        
        @property
        def x_train_path(self):
            if not os.path.exists(self._x_train_path):
                raise ValueError("The path for x training files does not exist.")
            return self._x_train_path
                
        @x_train_path.setter
        def x_train_path(self, path):
            self._x_train_path = path
            
        @property
        def y_train_path(self):
            if not os.path.exists(self._y_train_path):
                raise ValueError("The path for y training files does not exist.")
            return self._y_train_path
                
        @y_train_path.setter
        def y_train_path(self, path):
            self._y_train_path = path
            
        @property
        def x_valid_path(self):
            if not os.path.exists(self._x_valid_path):
                raise ValueError("The path for x validation files does not exist.")
            return self._x_valid_path
                
        @x_valid_path.setter
        def x_valid_path(self, path):
            self._x_valid_path = path
            
        @property
        def y_valid_path(self):
            if not os.path.exists(self._y_valid_path):
                raise ValueError("The path for y validation files does not exist.")
            return self._y_valid_path
                
        @y_valid_path.setter
        def y_valid_path(self, path):
            self._y_valid_path = path
            
        @property
        def train_txt_file(self):
            if not os.path.isfile(self._train_txt_file):
                raise ValueError("The path for training text files does not exist.")
            return self._train_txt_file
                
        @train_txt_file.setter
        def train_txt_file(self, file):
            self._train_txt_file = file
            
        @property
        def valid_txt_file(self):
            if not os.path.isfile(self._valid_txt_file):
                raise ValueError("The path for validation text files does not exist.")
            return self._valid_txt_file
                
        @valid_txt_file.setter
        def valid_txt_file(self, file):
            self._valid_txt_file = file
            
        """
        @property
        def train_list(self):
            if self._train_txt_file is not None:
                item_list = read_list_from_txt(self._train_txt_file)
                return item_list
                
        @property
        def valid_list(self):
            if self._valid_txt_file is not None:
                item_list = read_list_from_txt(self._valid_txt_file)
                return item_list
        """
            
        @property
        def shuffle_size(self):
            if self._shuffle_size < 1: 
                print("WARNING: The minimum shuffle size should not be smaller than 1.")
                self._shuffle_size = 1
            return self._shuffle_size
                
        @shuffle_size.setter
        def shuffle_size(self, size):
            self._shuffle_size = size
            
        @property
        def num_parallel_calls(self):
            if self._num_parallel_calls < 1: 
                print("WARNING: The minimum number of parallel calls should not be smaller than 1.")
                self._num_parallel_calls = 1
            return self._num_parallel_calls
                
        @num_parallel_calls.setter
        def num_parallel_calls(self, num):
            self._num_parallel_calls = num
           
        
        
        
        
    
