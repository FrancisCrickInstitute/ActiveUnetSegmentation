#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import unetsl, unetsl.model, unetsl.data, unetsl.config
from unetsl.data import VolumeViewer
from matplotlib import pyplot
from matplotlib.widgets import Slider

import sys
import numpy
import time
import math

import click


    
def getDataGenerator(source, n_labels, patch_size, stride):
    if stride is None:
        stride = (patch_size[0]//3, patch_size[1]//3, patch_size[2]//3, patch_size[3]//3)
    
    source.updateGeometry(n_labels, patch_size, stride)
    source.generateIndexes();
    return source.size(), source.getDataGenerator()

def get_dims(n_chan):
    mx = int(math.sqrt(n_chan))
    
    factors = []
    for i in range(1, mx+1):
        if n_chan%i==0:
            factors.append((n_chan/i, i))
    factors.sort()
    return factors[-1]

class PredictionGenerator:
    def __init__(self, model, layer_index):
        import keras.models
        inputs = model.inputs
        output = model.layers[layer_index].output
        self.model = keras.models.Model(inputs, output)
    def predict(self, chunk):
        p = self.model.predict(chunk)
        print(p.shape)
        return numpy.sum(p, axis=1, keepdims=True)

@click.command()
@click.option("-c", "config_file", type=click.Path(exists=True), required=True)
@click.option("-l", "layer_number", type=click.INT, default=-1)
def displayLayerOutput(config_file, layer_number):
    config = unetsl.config.getDefaultConfigurationTool()
    config.load(config_file)
    model_name = str(config_file).replace(".json", ".h5");
    model = unetsl.model.loadModel(model_name)
    source_configs = config[unetsl.DATA_SOURCES]
    if layer_number==-1:
        for i, j in enumerate(model.layers):
            print("%d \t %s"%(i, j.name))
        selection = input("enter layer: ")
        layer_number = int(selection)
    model = PredictionGenerator(model, layer_number)
    
    for source_config in source_configs:
        displayModelOutput(model, [ source_config ])
        
def displayModelOutput(model, source_configs, stride=None, normalize_samples = False):
    
    sources = unetsl.data.getDataSources(source_configs, normalize_samples)
    input_shape = unetsl.model.getInputShape(model.model)
    output_shape = unetsl.model.getOutputShape(model.model)
    if stride is None:
        stride = input_shape
    #pyplot.ion()
    volumes = []
    DO_ALL = False
    for source in sources:
        print(source)
                
        n, dataGenerator = getDataGenerator(source, output_shape[0], input_shape, stride)
                
        print("n: ", n, " steps" )
        for i in range(n):
            
            vals = dataGenerator.__next__()
            x = vals[0]
            y = model.predict(x)
            if isinstance(y, dict):
                keys = list(y)
                keys.sort()
                y = [y[f] for f in keys]
                
            elif not isinstance(y, list):
                y = [y]
            
            print("y-type", type(y[0]))
            print("output len: ", len(y))
            print("x-shape: ", x.shape, "y-shape", [s.shape for s in y])            
            
            fig = 0
            if len(volumes)==0:
                volumes.append(VolumeViewer(fig, x[0]))
                for ys in y:
                    fig += 1
                    volumes.append(VolumeViewer(fig, ys[0], limits=(0,1)))
            
            else:
                datas = [x] + y
                for i, data in enumerate(datas):
                    volumes[i].setData(data[0])
            if not DO_ALL:
                check = input("return to continue, n for next, a for all: ")
                if check=="n" :
                    break
                if check=="a":
                    DO_ALL=True
        print("next source")

def main():
    showDataSources()
    
    
@click.command()
@click.option("-c", "config_file", type=click.Path(exists=True), required=True)
def showDataSources(config_file):
    print("usage: inspect_data_sources -c model_cfg.json")
    #running
    config = unetsl.config.getDefaultConfigurationTool()
    config.load(config_file)
    model_name = str(config_file).replace(".json", ".h5")
    model = unetsl.model.loadModel(model_name)
    source_configs = config[unetsl.DATA_SOURCES]
    
    input_shape = unetsl.model.getInputShape(model)
    
    output_shape = unetsl.model.getOutputShape(model)
    print("output shape: ",output_shape)
    n_labels = output_shape[0]
    
    patch_size = input_shape
    stride = config[unetsl.STRIDE]
    
    sources = unetsl.data.getDataSources(source_configs, config[unetsl.NORMALIZE_SAMPLES])
    pyplot.ion()
    volumes = []
    DO_ALL = False
    for source in sources:
        print(source)
        print("patch_size", patch_size, "stride", stride)
        
        n, dataGenerator = getDataGenerator(source, n_labels, patch_size, stride)
        print("n: ", n, " steps" )
        for i in range(n):
            
            vals = dataGenerator.__next__()
            if len(vals)==3:
                x,y,w = vals
                weight = numpy.sum(w)
                if weight==0:
                    continue
                else:
                    pass
            else:
                x,y = vals
            
            if isinstance(y, dict):
                keys = list(y)
                keys.sort()
                y = [y[f] for f in keys]
                
            elif not isinstance(y, list):
                y = [y]
            
            print("y-type", type(y[0]))
            print("output len: ", len(y))
            print("x-shape: ", x.shape, "y-shape", [s.shape for s in y])            
            
            fig = 0
            if len(volumes)==0:
                volumes.append(VolumeViewer(fig, x[0]))
                for ys in y:
                    fig += 1
                    volumes.append(VolumeViewer(fig, ys[0], limits=(0,1)))
            
            else:
                datas = [x] + y
                for i, data in enumerate(datas):
                    volumes[i].setData(data[0])
            if not DO_ALL:
                check = input("return to continue, n for next, a for all: ")
                if check=="n" :
                    break
                if check=="a":
                    DO_ALL=True
        print("next source")
    
    
if __name__=="__main__":
    #displayLayerOutput()
    main()
