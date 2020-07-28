# -*- coding: utf-8 -*-

# This is the general test script ...

import os, cv2
import numpy as np
import tensorflow as tf

def add_sub(x, y):
    '''
    在此函数中使用纯python编程方式
    '''
    x_ = x.numpy()
    y_ = y.numpy()
    result1 = x_ + y_ - (x_ - y_)
    result2 = x_ + y_ + (x_ - y_)  
    # 返回的就是普通的python对象，但是它会自动转化成tensor来作为最终的结果，是自动完成的
    return result1,result2
 
x = tf.Variable(initial_value = [10,20,30,40,50,60,70,80])
y = tf.Variable(initial_value = [100,200,300,400,500,600,700,800])
 
y1,y2 = tf.py_function(func=add_sub, inp=[x, y], Tout=[tf.int32,tf.int32])
print(y1)  
print(y2)  

# success!

dataset = tf.data.Dataset.from_tensor_slices((x, y))
dataset = dataset.map(lambda x, y: tf.py_function(func=add_sub, inp=[x, y], Tout=[tf.float32,tf.float32]), num_parallel_calls=4)
dataset = dataset.repeat(1).shuffle(buffer_size=4).batch(4)
                                      
print("Done")
