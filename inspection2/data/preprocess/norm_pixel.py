# -*- coding: utf-8 -*-

# Header ...

import os
import numpy as np


def norm_pixel(x, y, mode="x"):
    if ("x" in mode): x = x.astype(np.float32) / 255.0
    if ("y" in mode): y = y.astype(np.float32) / 255.0
        
    return x, y