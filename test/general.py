# -*- coding: utf-8 -*-

# This is the general test script ...

import os, cv2
import numpy as np

def norm_pixel(x, y, mode="x"):
    if ("x" in mode): x = x.astype(np.float32) / 255.0
    if ("y" in mode): y = y.astype(np.float32) / 255.0
        
    return x, y

a = np.ones((20,28,28,1), dtype=np.float32)
b = a.copy()

c, d = norm_pixel(a, b)
print(c.shape)
print(d.shape)