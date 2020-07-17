# -*- coding: utf-8 -*-

# Header ...

import os
import numpy as np
from inspection2.backend import dtypes as be


def np_dtype(string):
    if string   = "int16":   return np.int16
    elif string = "int32":   return np.int32
    elif string = "int64":   return np.int64
    elif string = "uint8":   return np.uint8
    elif string = "uint16":  return np.uint16
    elif string = "uint32":  return np.uint32
    elif string = "uint64":  return np.uint64
    elif string = "float16": return np.float16
    elif string = "float32": return np.float32
    elif string = "float64": return np.float64
    else: raise ValueError("The data type is not supported for casting.")
    
   
def be_dtype(string):
    if string   = "int16":   return be.int16
    elif string = "int32":   return be.int32
    elif string = "int64":   return be.int64
    elif string = "uint8":   return be.uint8
    elif string = "uint16":  return be.uint16
    elif string = "uint32":  return be.uint32
    elif string = "uint64":  return be.uint64
    elif string = "float16": return be.float16
    elif string = "float32": return be.float32
    elif string = "float64": return be.float64
    else: raise ValueError("The data type is not supported for casting.")