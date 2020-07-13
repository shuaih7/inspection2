# -*- coding: utf-8 -*-

# Header ...

import os
from abc import ABC, abstractmethod
import tensorflow as tf


class Base(ABC):
    def __init__(self):
        # Basic parameters:
        self.name      = "project"
        self.log_dir   = os.getcwd()
        self.model_dir = os.getcwd()
        self.mode      = "train_val"
        
        # Model parameters
        self.batch_size = 32
        self.is_shuffle = True
        self.epochs     = 100
        
        # Model configurations
        self.net        = None
        self.optimizer  = None
        self.loss       = None
        self.metrics    = None
        self.callbacks  = None
        
        # Model status and dataset
        self.is_built = False
        self.x_train  = None
        self.y_train  = None
        self.x_valid  = None
        self.y_valid  = None
        
    @abstractmethod
    def load_data(self, load_func=None, *args, **kwargs) -> function:
        pass   
        
    @abstractmethod
    def config_net(self) -> tf.keras.models.Model:
        pass
        
    @abstractmethod
    def config_optimizer(self) -> tf.keras.optimizers:
        pass
        
    @abstractmethod
    def config_loss(self) -> tf.keras.losses:
        pass
        
    @abstractmethod
    def config_metrics(self) -> tf.keras.metrics:
        pass
        
    @abstractmethod
    def config_callbacks(self) -> tf.keras.callbacks:
        pass
        
    def build(self):
        self.config_net()
        self.config_optimizer()
        self.config_loss()
        self.config_metrics()
        self.config_callbacks()
        self.is_built = True