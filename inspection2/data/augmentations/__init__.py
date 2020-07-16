# -*- coding: utf-8 -*-

# Header ...




def no_augment(x_train, y_train, x_valid=None, y_valid=None):
    if x_valid is None or y_valid is None: return x_train, y_train
    else: return x_train, y_train, x_valid, y_valid