# -*- coding: utf-8 -*-

# Header ...

import os
import numpy as np


def one_hot(array, num_classes=10, dtype=np.float32):
    if num_classes <= 0: num_classes = array.max()
    
    return np.eye(num_classes)[array]
