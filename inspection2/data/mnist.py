# -*- coding: utf-8 -*-

# Header ...

import numpy as np
import struct, os, math
from tensorflow.keras.utils import to_categorical as one_hot


def load_mnist(path, kind='train', num_classes=10):    
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = np.fromfile(lbpath,dtype=np.uint8)
        
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
        
    # Pre-processing
    num, params = images.shape[:2]
    size = int(math.sqrt(params))
    images = np.array(images.reshape((num, size, size, 1)), dtype=np.float32)
    labels = one_hot(labels, num_classes=num_classes)
        
    return images, labels