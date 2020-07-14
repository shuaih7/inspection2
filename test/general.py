# -*- coding: utf-8 -*-

# This is the general test script ...

import os
from package import foo, foo1

foo1()

class lenet(object):
    def __init__(self):
        pass
        
class base(object):
    def __init__(self):
        self.net = lenet
        print("The name is {0}".format(self.net.__name__))
        
bse = base()