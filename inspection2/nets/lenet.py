# -*- coding: utf-8 -*-

# Header ...

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, BatchNormalization, MaxPooling2D, Activation, Flatten


def lenet_5(input_shape, 
            num_classes=10,
            use_batch_norm=False,
            batch_trainable=True,
            classifier_activation="softmax"):
    """ Create a LeNet-5
    
    Parameters
    ----------
    input_shape: A tuple of size [batch, height_in, width_in, channels].
    
    Returns
    ----------
    net: 
    
    """
    net_input = Input(input_shape)
    
    conv1 = Conv2D(6, kernel_size=3, strides=1, name="conv1", padding="valid")(net_input)
    if use_batch_norm: conv1 = BatchNormalization(name="bn1", trainable=batch_trainable)(conv1)
    conv1 = MaxPooling2D(pool_size=2, strides=2, name="maxpool1")(conv1)
    conv1 = Activation("relu")(conv1)
    
    conv2 = Conv2D(16, kernel_size=3, strides=1, name="conv2", padding="valid")(conv1) 
    if use_batch_norm: conv2 = BatchNormalization(name="bn2", trainable=batch_trainable)(conv2)
    conv2 = MaxPooling2D(pool_size=2, strides=2, name="maxpool2")(conv2)
    conv2 = Activation("relu")(conv2)
    fc0   = Flatten()(conv2)
    
    fc1   = Dense(120, name="fc1")(fc0)
    if use_batch_norm: fc1 = BatchNormalization(name="bn3", trainable=batch_trainable)(fc1)
    fc1   = Activation("relu")(fc1)
    
    fc2   = Dense(84, name="fc2")(fc1)
    if use_batch_norm: fc2 = BatchNormalization(name="bn4", trainable=batch_trainable)(fc2)
    fc2   = Activation("relu")(fc2)
    
    fc3   = Dense(num_classes, name="fc3")(fc2)
    if use_batch_norm: fc3 = BatchNormalization(name="bn5", trainable=batch_trainable)(fc3)
    fc3   = Activation("relu")(fc3)
    
    if classifier_activation is not None: pred_score = Activation(classifier_activation)(fc3)
    else: pred_score = fc3
    
    net = Model(net_input, pred_score, name="lenet_5")
                          
    return net
    
    