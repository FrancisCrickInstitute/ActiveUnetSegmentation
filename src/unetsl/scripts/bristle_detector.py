#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Take in 32x32 size squares 
"""

from keras.models import Sequential
import keras.layers as layers
import keras.optimizers as optimizers
from keras.utils import Sequence
import keras
import unetsl.data
import unetsl.model
import unetsl.cli_interface
import os,sys
import numpy
import random

class BSeq(Sequence):
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.n = len(data)
        self.batches = self.n//self.batch_size
        if self.batches*self.batch_size < self.n:
            self.batches += 1
        
    def __len__(self):
        return self.batches
    
    def __getitem__(self, index):
        low = index*self.batch_size
        vals = [self.data[(i+low)%self.n] for i in range(self.batch_size) ]
        x = numpy.array([[v[0]] for v in vals])
        y = numpy.array([v[1] for v in vals])
        
        return x,y    
                    

def createModel(config):
    
    lays = [
            layers.Conv2D(8, (3,3),input_shape=(1,32, 32), data_format="channels_first", padding="valid"),
            layers.Activation("relu"),
            layers.Conv2D(16, (5, 5), data_format="channels_first", padding="valid"),
            layers.Dropout(0.2),
            layers.MaxPooling2D((2, 2), padding="valid", data_format="channels_first"),
            layers.Activation("relu"),
            layers.Conv2D(32, (5, 5), data_format="channels_first", padding="valid"),
            layers.Activation("relu"),
            layers.Reshape((32*9*9,)),
            layers.Dense(2)
            ]
    model = Sequential()
    for l in lays:
        model.add(l)
        print(model.output.shape)
    return model

def getData(bristles, non_bristles):
    brls, _ = unetsl.data.loadImage(bristles)
    nbrls, _ = unetsl.data.loadImage(non_bristles)
    z1 = [(sli, numpy.array([1,0])) for sli in brls[0]]
    z2 = [(sli, numpy.array([0,1])) for sli in nbrls[0]]
    total = z1 + z2
    random.shuffle(total)
    f = int(0.8*len(total))
    return total[:f], total[f:]
    
    
def gen(data, batch_size):
    batch = []
    for slice in data:
        batch += slice
        if len(batch)<batch_size:
            continue
        else:
            yield batch
            batch = []
    if len(batch)>0:
        missing = batch_size - len(batch)
        for i in range(missing):
            batch += data[i]
    yield batch

        
         
def trainModel(config):
    model_file = config["model file"]
    if not os.path.exists(model_file):
        model = createModel(config)
    else:
        model = unetsl.model.loadModel(model_file)
    
    model.compile(optimizers.Adam(0.000001), keras.losses.mean_absolute_error , metrics = ['binary_accuracy', 'accuracy'])
    training, validation = getData(config["bristles file"], config["non-bristles file"])
    
    logger = unetsl.model.LightLog(model_file, model)
    batch_size = 16

    tSeq = BSeq(training, batch_size)
    vSeq = BSeq(validation, batch_size)
    
    model.fit_generator(generator=tSeq,
                        epochs=10000,
                        validation_data=vSeq,
                        callbacks=[logger])
    
    
    
    pass
    
def getDefaultConfig():
    return {
            "model file": "bristle.h5",
            "bristles file" : "bristles.tif",
            "non-bristles file" : "non-bristles.tif"
            }
if __name__=="__main__":
    print("usage: bristle_detector.py config.json")
    config = getDefaultConfig();
    if os.path.exists(sys.argv[1]):
        unetsl.model.loadConfig(config, sys.argv[1])
    if unetsl.cli_interface.configure(config):
        unetsl.model.saveConfig(config, sys.argv[1])
        trainModel(config)
        
