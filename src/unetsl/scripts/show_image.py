#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import unetsl.cli_interface
from unetsl.cli_interface import getFilePrompt
import unetsl.model
import unetsl.data

import keras

from matplotlib import pyplot
import numpy
import pathlib

import sys

        

def main(args=None):
    print("usage: show_image model.h5 image.tif")
    print("displays the supplied image and labels to test that channels are being loaded properly");
    if args is None:
        print("enter model file name: ")
        model_file = getFilePrompt("first: " )
        print(model_file)
        input_file = getFilePrompt("input image filename: ")
        print(input_file)
    else:
        model_file = args[1]
        input_file = args[2]
        label_file = args[3]
    
    model = unetsl.model.loadModel(model_file);
    input_shape = unetsl.model.getInputShape(model)
    output_shape = unetsl.model.getOutputShape(model)
    
    image, tags = unetsl.data.loadImage(input_file)
    labels, tags = unetsl.data.loadImage(label_file)
    print(image.shape, labels.shape)
    image = unetsl.data.splitIntoChannels(input_shape, image)
    z = image.shape[1]
    n_labels = output_shape[0]
    labels = unetsl.data.getMultiClassLabels(labels[0], n_labels)

    for i in range(z):
        
        
        
        fig = pyplot.figure(i)
        vertical_panels = 0
        if input_shape[0]>output_shape[0]:
            vertical_panels = input_shape[0]
        else:
            vertical_panels = output_shape[0]
            
        for c in range(input_shape[0]):
            fig.add_subplot(vertical_panels, 2, c*2+1)
            pyplot.imshow(image[c, i])
            
        for l in range(n_labels):
            fig.add_subplot(vertical_panels, 2, l*2+2)
            print(labels.shape)
            pyplot.imshow(labels[l,i%labels.shape[1]])
        
        pyplot.show()            
            

if __name__=="__main__":
    args = None
    if(len(sys.argv)>1):
        args = sys.argv
    main(args)
