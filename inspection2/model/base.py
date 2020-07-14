# -*- coding: utf-8 -*-

# Header ...

import os
import tensorflow as tf
from abc import ABC, abstractmethod
from inspection2.utils.logger import Logger


class Base(ABC):
    def __init__(self):
        # Basic parameters:
        self.name      = "project"
        self.log_dir   = os.getcwd()
        self.model_dir = os.getcwd()
  
        # Model configurations
        self.logger     = None
        self.net        = None
        self.optimizer  = None
        self.loss       = None
        self.metrics    = None
        self.callbacks  = None
        
        self.load_func       = None
        self.preprocess_func = None
        self.packaging_func  = None
        
        # Model default parameters
        self.batch_size = 32
        self.is_shuffle = True
        self.epochs     = 100
        self.is_built   = False
        
        self.x_train  = None
        self.y_train  = None
        self.x_valid  = None
        self.y_valid  = None
        self.input_shape = None
        
    def config_logger(self, logger=None):
        self.logger = Logger(logger, name=self.name, log_dir=self.log_dir)
        
    def config_gpu(self, gpu_config):
        pass
        
    @abstractmethod
    def load_data(self, *args, **kwargs):
        if self.load_func is not None: 
            x_train, y_train, x_valid, y_valid = self.load_func(*args, **kwargs)
            
            if self.preprocess_func is not None: 
                self.x_train, self.y_train, self.x_valid, self.y_valid = self.preprocess_func(x_train, y_train, x_valid, y_valid)
            else:
                self.x_train, self.y_train, self.x_valid, self.y_valid = x_train, y_train, x_valid, y_valid
                
        if self.input_shape is None and self.x_train is not None: self.input_shape = x_train.shape[1:]
        
    @abstractmethod
    def config_load(self, load_func=None):
        if load_func is not None: self.load_func = load_func
        
    # Pre-processing function will be nested in load_data()
    def config_preprocess(self, preprocess_func=None):
        if preprocess_func is not None: self.preprocess_func = preprocess_func
        
    @abstractmethod
    def config_net(self, net=None) -> tf.keras.models.Model:
        if net is not None: self.net = net
        
    @abstractmethod
    def config_optimizer(self, optmizer=None) -> tf.keras.optimizers:
        if optimizer is not None: self.optimizer = optimizer
        
    @abstractmethod
    def config_loss(self, loss=None) -> tf.keras.losses:
        if loss is not None: self.loss = loss
        
    @abstractmethod
    def config_metrics(self, metrics=None) -> tf.keras.metrics:
        if metrics is not None: self.metrics = metrics
        
    @abstractmethod
    def config_callbacks(self, callbacks=None) -> tf.keras.callbacks:
        if callbacks is not None: self.callbacks = callbacks
    
    # Packing function will be nested in predict()
    def config_packaging(self, packaging_func=None):
        if packaging_func is not None: self.packaging_func = packaging_func
        
    def build(self):
        try:
            self.config_logger()
            self.logger.info("Start building up {0}".format(self.name) + "...")
            
            self.config_load()
            self.logger.info("Successfully configured load function.")
            
            self.config_net()
            self.logger.info("Successfully construct the model.")
            
            self.config_optimizer()
            self.logger.info("Successfully configured optimizer.")
            
            self.config_loss()
            self.logger.info("Successfully configured loss function.")
            
            self.config_metrics()
            self.logger.info("Successfully configured evaluation metrics.")
            
            self.config_callbacks()
            self.logger.info("Successfully configured callback functions.")
            
        except Exception as expt:
            self.logger.error(expt)
            
        self.is_built = True
        
    def train(self):
        if not self.is_built: self.build()
        
        x_train, y_train = self.x_train, self.y_train
        x_valid, y_valid = self.x_valid, self.y_valid
        
        if x_train is None or y_train is None: 
            self.logger.error("The training data or label should not be None.", error_type=ValueError)
        elif x_valid is None or y_valid is None: 
            validation_data = None
            self.logger.warning("Warning: No validation set is feeded.")
        else: validation_data = (x_valid, y_valid)
        
        model = self.net
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.fit(x=x_train, y=y_train, validation_data=validation_data, batch_size=self.batch_size, epochs=self.epochs, 
                  verbose=1, shuffle=self.is_shuffle, callbacks=self.callbacks)
                  
    def predict(self):
        pass