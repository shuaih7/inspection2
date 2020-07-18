# -*- coding: utf-8 -*-

# Header ...

import os, logging
import numpy as np
from abc import ABC, abstractmethod
from inspection2.backend import py_function
from inspection2.backend.data import Dataset
from inspection2.utils import np_dtype, be_dtype


class LoadDataset(ABC):
    def __init__(self, data_param, logger=None):
        self.data_param = data_param
        self.logger     = logger
        if logger is None: self.logger = logging.getLogger(__name__)
        
        self.is_built      = False
        self.init_func     = None
        self.map_func      = []
        self.map_ext_args  = []
        
    @abstractmethod
    def config_init(self, init_func=None):
        if init_func is not None: self.init_func = init_func
        
    def config_maps(self, map_func=[], map_ext_args=[]):
        if len(map_func): 
            self.map_func = map_func
            self.map_ext_args = map_ext_args
        if len(map_func) != len(map_ext_args):
            self.logger.error("The lengths of map functions and the extra argument not matching.")
        
    def add_maps(self, map_func, map_ext_args):
        self.map_func.append(map_func)
        self.map_ext_args.append(map_ext_args)
        
    def build(self):
        self.config_init()
        self.logger.info("Successfully construct the dataset initialization function.")
        
        self.config_maps()
        self.logger.info("Successfully construct the dataset mapping function(s).")
        
        self.is_built = True
        
    def create_dataset(self, **kwargs):
        train_ds = self.create_train_dataset(**kwargs)
        valid_ds = self.create_valid_dataset(**kwargs)
        return train_ds, valid_ds

    def create_train_dataset(self, **kwargs):
        if not self.is_built: self.build()
        train_ds, x_dtype, y_dtype = None, self.data_param.x_dtype, self.data_param.y_dtype
        
        try: 
            x, y = self.init_func(self.data_param, mode="train", **kwargs)
            
            # Creating the dataset API for data pipelining
            train_ds = Dataset.from_tensor_slice((x, y))
            
            # Pipelining the loading processes 
            for func, ext_arg in zip(self.map_func, self.map_ext_args)
                train_ds = dataset.map(lambda x, y: tuple(py_function(func, [x, y]+ext_arg, Tout = [be_dtype(x_dtype), be_dtype(y_dtype)])), 
                                      num_parallel_calls=data_param.num_parallel_calls)
                                      
        except Exception as expt: logger.warning(expt)

        return train_ds
        
    def create_valid_dataset(self, **kwargs):
        if not self.is_built: self.build()
        valid_ds, x_dtype, y_dtype = None, self.data_param.x_dtype, self.data_param.y_dtype
        
        try:
            x, y = self.init_func(self.data_param, mode="valid", **kwargs)
            
            # Creating the dataset API for data pipelining
            valid_ds = Dataset.from_tensor_slice((x, y))
            
            # Pipelining the loading processes 
            for func, ext_arg in zip(self.map_func, self.map_ext_args)
                valid_ds = dataset.map(lambda x, y: tuple(py_function(func, [x, y]+ext_arg, Tout = [be_dtype(x_dtype), be_dtype(y_dtype)])), 
                                      num_parallel_calls=data_param.num_parallel_calls)
                                      
        except Exception as expt: logger.warning(expt)

        return valid_ds
        
    
