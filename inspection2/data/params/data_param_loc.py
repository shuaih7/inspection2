# -*- coding: utf-8 -*-

# Header ...

from .data_param import DataParam


class DataParamLoc(DataParam):

    def __init__(self):
        super(DataParamLoc, self).__init__()
        
        # Classification parameters
        self._image_h = 0             # Modify if you would like to reshape the image height
        self._image_w = 0             # Modify if you would like to reshape the image width
        self._image_c = 0             # Modify if you would like to reshape the image channel
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
        def image_h(self):
            return self._image_h
            
        @image_h.setter
        def image_h(self, h):
            if h <= 4: raise ValueError("The image heihgt should be larger than 4.")
            self._image_h = h
            
        @property
        def image_w(self):
            return self._image_w
            
        @image_w.setter
        def image_w(self, w):
            if w <= 4: raise ValueError("The image width should be larger than 4.")
            self._image_w = w
            
        @property
        def image_c(self):
            return self._image_c
            
        @image_c.setter
        def image_c(self, c):
            if c not in [1,3,4]: raise ValueError("The image channel should have the value 1, 3, or 4.")
            self._image_c = c

        @property
        def image_size(self):
            return (self.image_h, self.image_w, self.image_c)