# -*- coding: utf-8 -*-

# This is the general test script ...


import os, struct, math, sys, cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras import *
from tensorflow.keras import Input
from inspection2.data.params import DataParamCls
from inspection2.utils import one_hot
from inspection2.nets import lenet_5
from inspection2.models import Base
from inspection2.load import LoadMnist
from inspection2.backend.optimizers import Adam
from inspection2.backend.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

def norm_pixel(x, y):
    return tf.cast(x, dtype=tf.float32) / 255.0, y

def lenet_5(input_shape, 
            num_classes=10,
            use_batch_norm=False,
            batch_trainable=True,
            classifier_activation="softmax"):
    """ Create a LeNet-5
    
    Parameters
    ----------
    input_shape: A tuple of size [batch, height_in, width_in, channels].
    
    Returns
    ----------
    net: 
    
    """
    net_input = Input(input_shape)
    
    conv1 = Conv2D(6, kernel_size=3, strides=1, name="conv1", padding="valid", data_format="channels_last")(net_input)
    if use_batch_norm: conv1 = BatchNormalization(name="bn1", trainable=batch_trainable)(conv1)
    conv1 = MaxPooling2D(pool_size=2, strides=2, name="maxpool1")(conv1)
    conv1 = Activation("relu")(conv1)
    
    conv2 = Conv2D(16, kernel_size=3, strides=1, name="conv2", padding="valid", data_format="channels_last")(conv1) 
    if use_batch_norm: conv2 = BatchNormalization(name="bn2", trainable=batch_trainable)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = MaxPooling2D(pool_size=2, strides=2, name="maxpool2")(conv2)
    fc0   = Flatten()(conv2)
    
    fc1   = Dense(120, name="fc1")(fc0)
    if use_batch_norm: fc1 = BatchNormalization(name="bn3", trainable=batch_trainable)(fc1)
    fc1   = Activation("relu")(fc1)
    
    fc2   = Dense(84, name="fc2")(fc1)
    if use_batch_norm: fc2 = BatchNormalization(name="bn4", trainable=batch_trainable)(fc2)
    fc2   = Activation("relu")(fc2)
    
    fc3   = Dense(num_classes, name="fc3")(fc2)
    if use_batch_norm: fc3 = BatchNormalization(name="bn5", trainable=batch_trainable)(fc3)
    fc3   = Activation("relu")(fc3)
    
    if classifier_activation is not None: pred_score = Activation(classifier_activation)(fc3)
    else: pred_score = fc3
    
    net = Model(net_input, pred_score, name="lenet_5")
                          
    return net
            
            
def read_mnist_from_file(data_param, mode="train", train_kind="train", valid_kind="t10k"):
    if mode == "train":   
        kind     = train_kind
        x_path   = data_param.x_train_path
        y_dtype  = data_param.y_train_path
    elif mode == "valid": 
        kind     = valid_kind
        x_path   = data_param.x_valid_path
        y_dtype  = data_param.y_valid_path
    else: raise ValueError("Invalid input mode.")
    
    # Load mnist data from file
    labels_path = os.path.join(x_path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(x_path,'%s-images.idx3-ubyte'% kind)
    
    if os.path.isfile(labels_path):
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',lbpath.read(8))
            labels = np.fromfile(lbpath,dtype=np.uint8)
    else: labels = None
        
    if os.path.isfile(images_path):
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
            images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
    else: images = None
    
    # Reshape the loaded numpy arrays
    num, params = images.shape[:2]
    size = int(math.sqrt(params))
    images = np.array(images.reshape((num, size, size, 1)), dtype=np.float32)
    labels = one_hot(labels, num_classes=data_param.num_classes)
           
    return images, labels           
            
if __name__ == "__main__":
    log_dir = r"C:\projects\inspection2\lenet\log"
    model_dir = r"C:\projects\inspection2\lenet\model"
    data_dir  = r"C:\projects\inspection2\dataset\mnist"
    #data_dir = r"E:\Deep_Learning\inspection2\lenet_5\dataset"

    model = lenet_5(input_shape=(28,28,1))

    data_param = DataParamCls()
    data_param.x_train_path = data_dir
    data_param.y_train_path = data_dir
    data_param.x_valid_path = data_dir
    data_param.y_valid_path = data_dir
    data_param.shuffle_size = 1000
    data_param.num_parallel_calls = 4
    data_param.num_classes  = 10
    
    images, labels = read_mnist_from_file(data_param, mode="train")
    #images = np.zeros((60000,28,28,1), dtype=np.float32)
    #labels = np.zeros((60000,10), dtype=np.float32)
    
    # Checking ...
    print(images.shape)
    print(labels.shape)
    
    train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    
    def internal_generator():
        for image, label in zip(images, labels):
            yield image, label
         
    train_ds = tf.data.Dataset.from_generator(internal_generator, (tf.float32, tf.float32), (tf.TensorShape([28,28,1]), tf.TensorShape([10])))
    train_ds = train_ds.map(lambda x, y: tf.py_function(func=norm_pixel, inp=[x, y], Tout = [tf.float32, tf.float32]), num_parallel_calls=4)
                                      
    train_ds = train_ds.repeat(5).shuffle(buffer_size=1024).batch(64)
    
    num = 0
    for item in train_ds: num += 1
    print(num)
    sys.exit()
    
    def generator():
        for inp, out in train_ds:
            yield inp, out
            
    """
    index = 0
    for tr, val in train_ds:
        if index > 2: break
        print("\n\n")
        tr_np = tr.numpy()
        cv2.imshow("char", tr_np[0,:,:,0])
        cv2.waitKey(0)
        print(tr.get_shape().as_list())
        #\val_np = val.numpy()
        print(val.get_shape().as_list())
        index += 1
    sys.exit()
    """
    model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])
    #model.fit(train_ds, verbose=1)
    model.fit_generator(generator(), steps_per_epoch=938, verbose=1, epochs=5)
    