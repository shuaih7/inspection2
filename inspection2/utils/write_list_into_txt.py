# -*- coding: utf-8 -*-

# Header ...

import os
import glob as gb
from sklearn.utils import shuffle


def write_list_into_txt(file_path=None, file_suffix=".jpg", save_path=None, save_name="List", is_shuffle=True):
    if file_path is None or save_path is None: return
    file_list = gb.glob(file_path + r"/*"+file_suffix)
    if is_shuffle: file_list = shuffle(file_list)
    
    txt_name = os.path.join(save_path, save_name+".txt")
    with open(txt_name, "w") as f:
        for fname in file_list:
            _, temp_name = os.path.split(fname)
            filename, _ = os.path.splitext(temp_name)
            f.write(filename)
            f.write("\n")
        f.close()
            