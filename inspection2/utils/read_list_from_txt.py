# -*- coding: utf-8 -*-

# Header ...

import os
import glob as gb


def read_list_from_txt(txt_file):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    item_list = [l.replace("\n", "") for l in lines if l is not ""]
    return item_list