# -*- coding: utf-8 -*-

# Header ...

import os
from abc import ABC, abstractmethod
from inspection2.utils.logger import Logger
from inspection2.backend.data import AUTOTUNE


class Base(ABC):
    def __init__(self, input_shape=None, name="project", model_dir=None, log_dir=None):
        # Basic parameters:
        self.input_shape = input_shape
        self.name        = name
        self.model_dir   = model_dir
        self.log_dir     = log_dir
        self.logger      = Logger(logger=None, name=self.name, log_dir=self.log_dir)
        self.check_mdoel_params()
  
        # Model configurations
        self.net        = None
        self.optimizer  = None
        self.loss       = None
        self.metrics    = None
        self.callbacks  = None
        
        # Model fucntions initialization
        self.packaging_func  = None
        
        # Model default parameters
        self.batch_size = 32
        self.is_shuffle = True
        self.epochs     = 100
        self.verbose    = 1
        self.is_built   = False
        
        # Model dataset initialization
        self.x_train     = None
        self.y_train     = None
        self.x_valid     = None
        self.y_valid     = None
        
        self.train_ds    = None
        self.valid_ds    = None
        
    def check_mdoel_params(self):
        if self.name is None or self.name == "": self.name = "project"
        if not os.path.exists(self.model_dir):
            self.model_dir = os.path.join(os.getcwd(), "model")
            if not os.path.exists(self.model_dir): os.mkdir(self.model_dir)
            self.logger.warning("The specified model dir does not exist, redirecting to {0}".format(self.model_dir))
        if not os.path.exists(self.log_dir):
            self.log_dir = os.path.join(os.getcwd(), "log")
            if not os.path.exists(self.log_dir): os.mkdir(self.log_dir)
            self.logger.warning("The specified log dir does not exist, redirecting to {0}".format(self.log_dir))

    def config_gpu(self, gpu_config):
        pass
        
    @abstractmethod
    def load_data(self, data_param, **kwargs):
        pass
        
    @abstractmethod
    def config_net(self, net=None):
        if net is not None: self.net = net
        
    @abstractmethod
    def config_optimizer(self, optmizer=None):
        if optimizer is not None: self.optimizer = optimizer
        
    @abstractmethod
    def config_loss(self, loss=None):
        if loss is not None: self.loss = loss
        
    @abstractmethod
    def config_metrics(self, metrics=None):
        if metrics is not None: self.metrics = metrics
        
    @abstractmethod
    def config_callbacks(self, callbacks=None):
        if callbacks is not None: self.callbacks = callbacks
    
    # Packing function will be nested in predict()
    def config_packaging(self, packaging_func=None):
        if packaging_func is not None: self.packaging_func = packaging_func
        
    def build(self):
        try:
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
        
    def train(self, **kwargs):
        if self.x_train is not None:    self.train_local(**kwargs)
        elif self.train_ds is not None: self.train_dataset(**kwargs)
        else: self.logger.error("Please specify the input data fro training.")
        
    def train_local(self, batch_size=32, epochs=10, verbose=1, shuffle=True):
        if not self.is_built: self.build()
        
        x_train, y_train = self.x_train, self.y_train
        x_valid, y_valid = self.x_valid, self.y_valid
        
        if x_train is None or y_train is None: 
            self.logger.error("The training data or label should not be None.")
        elif x_valid is None or y_valid is None: 
            validation_data = None
            self.logger.warning("Warning: No validation set is feeded.")
        else: validation_data = (x_valid, y_valid)
        
        model = self.net
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.fit(x=x_train, y=y_train, validation_data=validation_data, batch_size=batch_size, epochs=epochs, 
                  verbose=verbose, shuffle=shuffle, callbacks=self.callbacks)
                  
    def train_dataset(self, batch_size=32, epochs=10, verbose=1, shuffle=True):
        if not self.is_built: self.build()
        
        if self.train_ds is None: self.logger.error("The training dataset is not specified.")
        else: self.train_ds.shuffle(self.data_param.shuffle_size).repeat(1).batch(batch_size).prefetch(AUTOTUNE)
        
        if self.valid_ds is not None: self.valid_ds.repeat(1).batch(batch_size).prefetch(AUTOTUNE)
        
        model, train_ds, valid_ds = self.net, self.train_ds, self.valid_ds
        model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        model.fit(x=train_ds, validation_data=valid_ds, batch_size=batch_size, epochs=epochs, 
                  verbose=verbose, shuffle=shuffle, callbacks=self.callbacks)
                  
    def train_database(self, batch_size=32, epochs=10, verbose=1, shuffle=True):
        pass
                  
    def predict(self):
        pass