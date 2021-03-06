# Configuring DNN models here.
# Adapted from https://github.com/zhixuhao/unet/blob/master/model.py

import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.layers import concatenate
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

# Modified UNet so that it works for 1D data.
def Unet1D(input_size, k, loss, pretrained_weights = None, learning_rate = 1e-4):
    inputs = Input(shape=input_size)
    conv1 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)
    conv2 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)
    conv3 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
    conv4 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)

    conv5 = Conv2D(1024, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, (2,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, (2, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, (2, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, (2, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss, metrics = ['accuracy'])
    
    print(model.summary())

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model

# Using dilated convolution here.
def DilatedUnet1D(input_size, k, loss, pretrained_weights = None, atrous_rate=(2, 1), learning_rate = 1e-4):
    inputs = Input(shape=input_size)
    conv1 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)
    conv2 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)
    conv3 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
    conv4 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)

    conv5 = Conv2D(1024, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, (2,1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(UpSampling2D(size = atrous_rate)(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, (2, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(UpSampling2D(size = atrous_rate)(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, (2, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(UpSampling2D(size = atrous_rate)(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, (2, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(UpSampling2D(size = atrous_rate)(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, (3, 1), activation = 'relu', padding = 'same', dilation_rate=atrous_rate, kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss, metrics = ['accuracy'])
    
    print(model.summary())

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model
    
def Unet1DNoPooling(input_size, k, loss, pretrained_weights = None, learning_rate = 1e-4):
    inputs = Input(shape=input_size)
    conv1 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 1))(conv1)
    conv2 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 1))(conv2)
    conv3 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 1))(conv3)
    conv4 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 1))(drop4)

    conv5 = Conv2D(1024, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, (2,1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, (2, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, (2, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, (2, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,1))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, (3, 1), activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = learning_rate), loss = loss, metrics = ['accuracy'])
    
    print(model.summary())

    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model