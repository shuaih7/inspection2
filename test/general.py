# -*- coding: utf-8 -*-

# This is the general test script ...

import os, cv2
import numpy as np

a = np.ones((50,50,1), dtype=np.float32) * 10
a += 0.1

b = cv2.resize(a, (30,30), interpolation = cv2.INTER_LINEAR)

print(b.shape)
print(type(b[0,0]))
print(b)

