# -*- coding: utf-8 -*-

# This is the general test script ...

import os, cv2
import numpy as np
from inspection2.data.mnist import load_mnist

path = r"E:\Deep_Learning\Dataset"

images, labels = load_mnist(path, kind="t10k")

print(type(images))
print(images.shape)
print(labels.shape)
print(labels.max())

