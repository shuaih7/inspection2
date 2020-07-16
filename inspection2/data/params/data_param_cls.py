# -*- coding: utf-8 -*-

# Header ...

from .data_param import DataParam


class DataParamCls(DataParam):

    def __init__(self):
        super(DataParamCls, self).__init__()
        
        # Classification parameters
        self._num_classes = 10
        
        @property
        def num_classes(self):
            if self._num_classes < 1: 
                print("WARNING: The minimum number of classes should not be smaller than 1.")
                self._num_classes = 1
            return self._num_classes
                
        @num_classes.setter
        def num_classes(self, num):
            self._num_classes = num