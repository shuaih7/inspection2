# -*- coding: utf-8 -*-

# Header ...

import os
import numpy as np
from inspection2.utils import be_dtype
from inspection2.backend import cast as be_cast


def norm_pixel(x, y):
    return be_cast(x, dtype=be_dtype("float32")) / 255.0, y
    

def norm_both_pixel(x, y):
    return be_cast(x, dtype=be_dtype("float32")) / 255.0, be_cast(y, dtype=be_dtype("float32")) / 255.0