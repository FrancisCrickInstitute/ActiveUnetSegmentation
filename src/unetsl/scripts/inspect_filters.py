#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib import pyplot
import math
import unetsl.model
import numpy
import sys

def plotConvolutionalLayerFilters(layer, frames=[0]):
    weights = layer.get_weights()
    filters = weights[0]
    h = filters.shape[0]
    w = filters.shape[1]
    n = filters.shape[3]
    box_h = int(math.sqrt(n))
    box_w = n//box_h
    
    
    out = numpy.zeros((box_h*h, box_w*w))
    for i in range(box_h):
        for j in range(box_w):
            x = j*w
            y = i*h
            for frame in frames:
                out[y:y+h, x:x+w] += filters[:, :, frame, i*box_w + j]
    
    pyplot.imshow(out)
    pyplot.show()
    
def canDisplayLayer(layer):
    w = layer.get_weights()
    if len(w)>0:
        w1 = w[0]
        if len(w1.shape)>=4:
            return True
    return False

if __name__=="__main__":
    model = unetsl.model.loadModel(sys.argv[1])
    print(len(model.layers), "layers in model")
    for layer in model.layers:
        if canDisplayLayer(layer):
            frames = layer.weights[0].shape[2]
            for frame in range(frames):
                plotConvolutionalLayerFilters(layer, [frame])
    