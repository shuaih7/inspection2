# -*- coding: utf-8 -*-

# Header ...

from .data_param import DataParam


class DataParamCls(DataParam):

    def __init__(self):
        super(DataParamCls, self).__init__()
        
        # Classification parameters
        self.image_h = None           # Modify if you would like to reshape the image height
        self.image_w = None           # Modify if you would like to reshape the image width
        self.image_c = None           # Modify if you would like to reshape the image channel
        self.image_dtype = "float32"  # Data type casting to after images loaded
        
        self.explicit_resize = False  # Explicitly resize the image using cv methods
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
            
        @property
        def image_size(self):
            return (self.image_h, self.image_w, self.image_c)