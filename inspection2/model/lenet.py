# -*- coding: utf-8 -*-

# Header ...

import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from inspection2.nets import lenet_5
from inspection2.model.base import Base
from inspection2.data.load import load_mnist
from inspection2.data.preprocess import normalize_pixel


class LeNet_5(Base):
    def __init__(self):
        super(LeNet_5, self).__init__()
        
    def load_data(self, *args, **kwargs):
        if self.load_func is None: self.config_load()
        x_train, y_train, x_valid, y_valid = self.load_func(*args, **kwargs)
        
        self.config_preprocess()
        if self.preprocess_func is not None:
            self.x_train, self.y_train, self.x_valid, self.y_valid = self.preprocess_func(x_train, y_train, x_valid, y_valid, mode="train")
        else:
            self.x_train, self.y_train, self.x_valid, self.y_valid = x_train, y_train, x_valid, y_valid
            
        if self.input_shape is None and self.x_train is not None: self.input_shape = x_train.shape[1:]
        
    def config_load(self, load_func=None):
        if load_func is not None: self.load_func = load_func
        elif self.load_func is None: self.load_func = load_mnist
            
    # Pre-processing function will be nested in load_data()
    def config_preprocess(self, preprocess_func=None):
        if preprocess_func is not None: self.preprocess_func = preprocess_func
        elif self.preprocess_func is None: self.preprocess_func = normalize_pixel
        
    def config_net(self, net=None):
        if net is not None: self.net = net
        elif self.net is None: self.net = lenet_5(input_shape=self.input_shape, use_batch_norm=True, batch_trainable=True)
        self.net.summary()
        
    def config_optimizer(self, optimizer=None):
        if optimizer is not None: self.optimizer = optimizer
        elif self.optimizer is None: self.optimizer = tf.keras.optimizers.Adam(lr=0.01)  
        
    def config_loss(self, loss=None):
        if loss is not None: self.loss = loss
        elif self.loss is None: self.loss = "binary_crossentropy"
        
    def config_metrics(self, metrics=None):
        if metrics is not None: self.metrics = metrics
        elif self.metrics is None: self.metrics = ["accuracy"]
        
    def config_callbacks(self, callbacks=None):
        if callbacks is not None: self.callbacks = callbacks
        elif self.callbacks is None: 
            callback_tensorboard = TensorBoard(log_dir=self.log_dir, histogram_freq=1)
            callback_model = ModelCheckpoint(os.path.join(self.model_dir, self.name), monitor="val_loss", verbose=1, save_best_only=False, mode="min")
            self.callbacks = [callback_tensorboard, callback_model]

        
        
        
        
        

