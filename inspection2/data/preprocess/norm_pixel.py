# -*- coding: utf-8 -*-

# Header ...

import os
import numpy as np


def norm_pixel(x_train=None, y_train=None, x_valid=None, y_valid=None, mode="x"):
    if ("x" in mode) and (x_train is not None) and (x_valid is not None):
        x_train, x_valid = x_train.astype(np.float32) / 255.0, x_valid.astype(np.float32) / 255.0
    if ("y" in mode) and (x_valid is not None) and (y_valid is not None):
        y_train, y_valid = y_train.astype(np.float32) / 255.0, y_valid.astype(np.float32) / 255.0
        
    return x_train, y_train, x_valid, y_valid