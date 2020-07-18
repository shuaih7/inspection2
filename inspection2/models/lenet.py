# -*- coding: utf-8 -*-

# Header ...

import os
from inspection2.nets import lenet_5
from inspection2.models import Base
from inspection2.load import LoadMnist
from inspection2.backend.optimizers import Adam
from inspection2.backend.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard


class LeNet_5(Base):
    def __init__(self, input_shape, name="project", model_dir=None, log_dir=None):
        super(LeNet_5, self).__init__(input_shape=input_shape, name=name, model_dir=model_dir, log_dir=log_dir)
        
    def load_data(self, data_param, **kwargs):
        ds = LoadMnist(data_param, logger=self.logger)
        train_ds, 
        
    def config_net(self, net=None):
        if net is not None: self.net = net
        elif self.net is None: self.net = lenet_5(input_shape=self.input_shape, use_batch_norm=True, batch_trainable=True)
        self.net.summary()
        
    def config_optimizer(self, optimizer=None):
        if optimizer is not None: self.optimizer = optimizer
        elif self.optimizer is None: self.optimizer = Adam(lr=0.01)  
        
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

        
        
        
        
        

