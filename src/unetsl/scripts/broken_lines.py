#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
import keras.engine as engine
import keras.layers as layers
import keras.optimizers as optimizers
import unetsl
import unetsl.cli_interface
import unetsl.model
import unetsl.data


import random
import sys
import numpy
from keras.models import load_model
from skimage.external.tifffile import TiffWriter
from keras import backend as K
import tensorflow

from matplotlib import pyplot

from keras.utils import plot_model
import random
import math, json

def tupleFromShape(shape):
    s = []
    for dim in shape:
        if dim.value:
            s.append(dim.value)
    return tuple(s)

def getDataGenerator(solid, broken, batch_size=128):
    """
       
    """
    zed = [([chunk], [1, 0]) for chunk in solid] + [([chunk], [0, 1]) for chunk in broken]
    random.shuffle(zed)
    
    xbatch = []
    ybatch = []
    while True:
        for i in range(len(zed)):
            x,y = zed[i]
            xbatch.append(x)
            ybatch.append(y)
            
            if len(xbatch)==batch_size:
                yield numpy.array(xbatch),numpy.array(ybatch)
                xbatch = []
                ybatch = []


    
def create():
    config = {
            "output_model" : sys.argv[2],
            "input_shape" : (1, 32, 32),
            "kernel_shape": json.loads(sys.argv[3]),
            "convolutions": json.loads(sys.argv[4]),
            "n_filters": json.loads(sys.argv[5]),
            "pooling": json.loads(sys.argv[6])
            }
    unetsl.cli_interface.configure(config)
    with keras.backend.get_session() as sess:
        createModel(config)
    


def createModel(config):
    """
        Create a simple CNN model for tensorflow that will detect broken lines.
    """
    keras.backend.set_image_data_format("channels_first")
    
    kernel_shape = config["kernel_shape"]
    input_shape = config["input_shape"]
    n_filters = config["n_filters"]
    convolutions = config["convolutions"]
    
    input_layer = engine.Input(input_shape)
    active_1 = keras.layers.Activation("relu")(input_layer)
    last = active_1
    c_filters = n_filters
    
    for j in range(convolutions):
        last = layers.Conv2D(c_filters, kernel_shape, padding='valid')(last)
        last = keras.layers.Activation("relu")(last)
        c_filters = c_filters*2        
    if config["pooling"]:    
        last = layers.MaxPool2D(tuple(config["pooling"]))(last)
    
    s = tupleFromShape(last.shape)
    p = 1
    for d in s:
        p *= d
    
    shaped_1 = layers.Reshape((p,))(last)
    dense_1 = layers.Dense(2)(shaped_1)

    out = layers.Activation("softmax")(dense_1)
    model = engine.Model(input_layer, out)
    model.save(config["output_model"])
    return model


def train():
    config = {
            "input_model":sys.argv[2],
            "output_model": sys.argv[2],
            "input_broken_image": "broken.tif",
            "input_full_image": "full.tif",
            }
    
    gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.25)
    with tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpu_options)) as sess:
        trainModel(config)
    
def trainModel(config):
    
    model= unetsl.model.loadModel(config["input_model"])
    full, _ = unetsl.data.loadImage(config["input_full_image"])
    broken, _ = unetsl.data.loadImage(config["input_broken_image"])
    
    full = full[0]
    broken = broken[0]
    
    n = len(full) - 1024
    n2 = len(broken) - 1024
    
    random.shuffle(full)
    random.shuffle(broken)
    
    optimizer = keras.optimizers.Adam(0.00001)
    #optimizer = keras.optimizers.SGD(0.00001)
    loss_function = keras.losses.mean_squared_error
    loss_function = keras.losses.categorical_crossentropy
    #loss_function = unetsl.model.sorensenDiceCoefLoss
    logger = unetsl.model.LightLog(config["output_model"], model)
    unetsl.model.recompileModel(
            model, 
            optimizer, 
            loss_function=loss_function)
    print("to here")
    batch_size = 256
    steps = n//batch_size + n2//batch_size
    optimizer = model.optimizer
    model.fit_generator(generator=getDataGenerator(full[:n], broken[:n2], batch_size),
                    steps_per_epoch=steps,
                    epochs=200,
                    validation_data=getDataGenerator(full[n:], broken[n2:], batch_size),
                    validation_steps=2048//batch_size,
                    callbacks=[logger]
                        )
    model.save(config["output_model"])
    
def predict():
    config = {
            "input_model":sys.argv[2],
            "broken_image": sys.argv[2].replace(".h5", "-broken-prediction.tif"),
            "full_image": sys.argv[2].replace(".h5", "-full-prediction.tif"),
            "image_to_predict":sys.argv[3]
            }
    findBrokenLines(config)
    
def findBrokenLines(config):
    
    img, _ = unetsl.data.loadImage(config["image_to_predict"])
    print(img.shape)
    skeletons  = img[0]
    print(skeletons.shape)
    ns = [2] + list(img.shape[1:])
    
    model= unetsl.model.loadModel(config["input_model"] )
    w = img.shape[-1]
    h = img.shape[-2]
    input_shape = unetsl.model.getInputShape(model)    
    sy = input_shape[-2]
    sx = input_shape[-1]
    horizontal_boxes = w//input_shape[-1]
    vertical_boxes = h//input_shape[-2]
    
    broken_boxes = []
    full_boxes = []
    for i in range(len(skeletons)):
        for j in range(vertical_boxes):
            y_low = j*sy
            
            for k in range(horizontal_boxes):
                x_low = k*sx
                patch = numpy.array([skeletons[
                        i: i+1, 
                        y_low : y_low + sy,
                        x_low : x_low + sx
                    ]])
                y = model.predict(patch)
                
                if( y[0,0]>y[0,1]):
                    full_boxes.append(patch)
                else:
                    broken_boxes.append(patch)
                    
    if len(full_boxes)>0:
        unetsl.data.saveImage(config["full_image"], numpy.array(full_boxes))
        
    if len(broken_boxes)>0:
        unetsl.data.saveImage(config["broken_image"], numpy.array(broken_boxes))
        
def show():
    config = {
            "input_model":sys.argv[2],
            "input_image": sys.argv[3],
            }
    #if unetsl.cli_interface.configure(config):
    showPrediction(config)
    
def showPrediction(config):
    images, _ = unetsl.data.loadImage(config["input_image"])
    model = unetsl.model.loadModel(config["input_model"])
    in_shape = unetsl.model.getInputShape(model)
    
    frame = images[0].shape
    stride = in_shape
    patch_shape = in_shape
    print(images.shape)
    sub_model = keras.models.Model(model.layers[0].input, model.layers[5].output)
    out_shape = unetsl.model.getOutputShape(sub_model)
    cx = int(math.sqrt(out_shape[0]))
    cy = out_shape[0]//cx
    
    for img in images[0]:
        out = numpy.zeros((img.shape[-2], img.shape[-1]))
        
        out2 = numpy.zeros(((img.shape[-2]*cy, img.shape[-1]*cx)))
        show = False
        for i in range(0,frame[1], stride[1]):
            for j in range(0, frame[2], stride[2]):
                patch = img[i:i+patch_shape[-2], j:j+patch_shape[-1]]
                pred = model.predict( numpy.array([[ patch ]]) )
                if pred[0,0]>pred[0,1]:
                    show =True
                conv = sub_model.predict(numpy.array([[patch]]))
                
                out[i:i+patch_shape[-2], j:j+patch_shape[-1]] = (pred[0,0] - pred[0,1])*patch 
                for k in range(cx):
                    for l in range(cy):
                        oy = l*img.shape[-2] + i
                        ox = k*img.shape[-1] + j
                        
                        out2[oy:oy+out_shape[-2], ox: ox + out_shape[-1]] = conv[0,k + l*cx]
        
        if show:
            pyplot.figure(0)
            pyplot.imshow(out2)
            pyplot.figure(1)
            pyplot.imshow(out)
            pyplot.show()
        
    
    
    
    
    
if __name__=="__main__":
    
    
    actions = {
            'c': create,
            't': train,
            'p': predict,
            's': show
            }
            
    actions[sys.argv[1]]()
    
