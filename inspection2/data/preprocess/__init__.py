# -*- coding: utf-8 -*-

# Header ...

from .norm_pixel import norm_pixel


def no_preprocess(x_train, y_train, x_valid, y_valid):
    return x_train, y_train, x_valid, y_valid