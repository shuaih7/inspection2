# -*- coding: utf-8 -*-

# Header ...

#import os
from tensorflow.keras import layers
#import Input, InputLayer
#from tensorflow.keras.layers import Conv2d, Dense, BatchNormalization, MaxPooling2D, Activation, Flatten


def Input(shape=None, batch_size=None, name=None, dtype=None, sparse=False, tensor=None, ragged=False, **kwargs):
    return layers.Input(shape=shape, batch_size=batch_size, name=name, dtype=dtype, sparse=sparse, tensor=tensor, ragged=ragged, **kwargs)
    
    
def InputLayer(input_shape=None, batch_size=None, dtype=None, input_tensor=None, sparse=False, name=None, ragged=False, **kwargs):
    # Note: It is generally recommend to use the functional layer API via Input, (which creates an InputLayer) without directly using InputLayer.
    return layers.InputLayer(input_shape=input_shape, batch_size=batch_size, dtype=dtype, input_tensor=input_tensor, sparse=sparse, name=name, ragged=ragged, **kwargs)
    
    
def Conv2d(
    filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
    dilation_rate=(1, 1), activation=None, use_bias=True,
    kernel_initializer='glorot_uniform', bias_initializer='zeros',
    kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
    kernel_constraint=None, bias_constraint=None, use_batch_norm=False, batch_trainable=True, groups=0, **kwargs
):
    # Note: Compare to the original tf.keras.Conv2d, the following three parameters are added:
    #       1. use_batch_norm:  Boolean, if True add a batch normalization layer after before the activation layer,
    #       2. batch_trainable: Boolean, if True the variables will be marked as trainable,
    #       3. groups:          An int, add a group normalization layer after before the activation layer.
    
    if not use_batch_norm: act_func = activation
    else: act_func = None
    
    cur_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, data_format=data_format,
                              dilation_rate=dilation_rate, activation=act_func, use_bias=use_bias,
                              kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                              kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer,
                              kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
    
    if not use_batch_norm: return cur_layer
    
    cur_layer = layers.BatchNormalization(trainable=batch_trainable)(cur_layer)
    return layers.Activation(activation)(cur_layer)
    
    

