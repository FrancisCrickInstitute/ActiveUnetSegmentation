#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from train_3d_time_series import TimeSeriesDataGenerator, get3DTimeSeriesDefaults

import unetsl.data
import unetsl.model
import unetsl.cli_interface

import sys
import pathlib
import re
import numpy
from matplotlib import pyplot


def displayGeneratedData(config):
    grab = re.compile("\\d+")
    
    input_folder = [ str(f) for f in pathlib.Path(config["input_folder"][0]).iterdir() if f.match("*tif")]

    n_inp = [(int(grab.findall(n)[0]), n) for n in input_folder]
    n_inp.sort()
    skeleton_folder = [ str(f) for f in pathlib.Path(config["skeleton_folder"][0]).iterdir() if f.match("*tif")]
    
    n_skel = [(int(grab.findall(n)[0]), n) for n in skeleton_folder]
    n_skel.sort()
    
    files = [pair for pair in zip([a[1] for a in n_inp], [b[1] for b in n_skel])]
    
    
    model = unetsl.model.loadModel(config["model file"])
    patch_size = unetsl.model.getInputShape(model)
    out_patch_size = unetsl.model.getOutputShape(model)
    print("input shape: ", patch_size)
    print("out_patch_size: ", out_patch_size)
    channels = patch_size[0]
    batch = config[unetsl.model.BATCH_SIZE]
    
    n = len(files)
    
    
    train_gen = TimeSeriesDataGenerator(files[:n], patch_size, out_patch_size, channels, batch, config["crop"], config["stride"]).getGenerator()
    
    while True:
        x, y = train_gen.__next__()
        #ypred = pred = model.predict(x)
        #print(ypred.shape)
        for i in range(len(x)):
            #batch
            xstack = x[i]
            ystack = y[i]
            x_channels = xstack.shape[0]
            y_channels = ystack.shape[0]
            rows = y_channels
            if x_channels>y_channels:
                rows = x_channels
            print("xss: ", xstack.shape)
            for k in range(xstack.shape[1]):
                fig = pyplot.figure(i)
                for j in range(x_channels):
                    fig.add_subplot(rows, 3, 1 + 3*j)
                    #pyplot.imshow(numpy.sum(xstack[j], axis=0))
                    pyplot.imshow(xstack[j, k])
                for j in range(y_channels):
                    fig.add_subplot(rows, 3, 2 + 3*j)
                    #pyplot.imshow(numpy.sum(ystack[j], axis=0))
                    pyplot.imshow(ystack[j,k])
                    
                #for j in range(y_channels):
                #    fig.add_subplot(rows, 3, 3+ 3*j)
                #    pyplot.imshow(numpy.sum(ypred[i, j], axis=0))
            
                pyplot.show()
            
            
    
    

if __name__=="__main__":
    print("usage: test-3d-time-series.py -c config.json");
    config = unetsl.model.getDefaultTrainingConfig()
    mc = unetsl.model.getDefaultConfig()
    config.update(mc)
    get3DTimeSeriesDefaults(config)
    
    if len(sys.argv)>2:
            if sys.argv[1] == "-c":
                config_path = pathlib.Path(sys.argv[2])
                if config_path.exists():
                    unetsl.model.loadConfig( config, config_path )
            else:
                print("defaults will be used")                
    else:
        sys.exit(0)
    
    if unetsl.cli_interface.configure(config):
        if not pathlib.Path(config["model file"]).exists():
            createFlattenedModel(config)
        #unetsl.model.saveConfig(config, config_path)
        displayGeneratedData( config )
