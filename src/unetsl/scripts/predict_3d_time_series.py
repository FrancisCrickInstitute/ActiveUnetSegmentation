#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    For predicting a time series of 3d stacks that are stored in different files.

"""
import unetsl.cli_interface
import unetsl.model
import unetsl.data

import sys
import numpy
import re

def predictImage(img_stack, model, cutoff=0.001, stride=None):
    # c, t, z, y, x
    img_shape = img_stack.shape
    
    output_shape = unetsl.model.getOutputShape(model)
    patch_shape = unetsl.model.getInputShape(model)
    
    print("out ", output_shape, " patch ", patch_shape)
    
    print("img ", img_stack.shape)
    #if an axis has been condensed.
    in_z = img_shape[2]
    z_patches = in_z//patch_shape[1]
    if patch_shape[1]*z_patches < in_z:
        z_patches += 1
    
    out_z = img_shape[2] * output_shape[1]//patch_shape[1]
    
    print("out_z", out_z, "os", output_shape)
        
    dx = patch_shape[-1]
    dy = patch_shape[-2]
    dz = patch_shape[-3]
    dc = patch_shape[-4] #time steps are channels
    
    labels = output_shape[0]
    
    output= numpy.zeros((1, out_z, img_shape[-2], img_shape[-1]), dtype="uint8")
    
    for kd in range(z_patches):
        k = kd*patch_shape[1]
        ok = kd*output_shape[1]
        
        if k + dz > img_shape[-3]:
            k = img_shape[-3] - dz
        
        if ok + output_shape[1] > out_z:
            ok = out_z - output_shape[1]
        
        for j in range(0, img_shape[-2], dy):
            if j + dy > img_shape[-2]:
                j = img_shape[-2] - dy
            
            for i in range(0, img_shape[-1], dx):
                if i + dx > img_shape[-1]:
                    i = img_shape[-1] - dx
                chunk = img_stack[:, 0:dc, k: k + dz, j:j+dy, i:i+dx]
                prediction = model.predict(chunk)
                result= numpy.zeros(prediction[0,0].shape)
                for c in range(labels):
                    result += (prediction[0, c] > cutoff )*(1 << c)
                output[0, ok:ok + output_shape[1], j:j+dy, i:i+dx] = result
    return output

def getNumericRanking(name):
    p = re.compile("\\d+")
    
    v = p.findall(name)
    
    print(v)
    return [int(i) for i in v]
    
def main(config):
    model = unetsl.model.loadModel(config[unetsl.model.MODEL_FILE])
    in_files = config["input files"]
    
    matched = [(getNumericRanking(f), f) for f in in_files]
    matched.sort()
    in_files = [f[1] for f in matched]
    
    input_shape = unetsl.model.getInputShape(model)
    output_shape = unetsl.model.getOutputShape(model)
    
    print(input_shape);
    print(output_shape);
    
    time_points = input_shape[0]
    
    stack = []
    predicted_stack = []
    
    for time in range(len(in_files)):
        stack = []
        for i in range(input_shape[0]):
            dex = time + i - 1
            if dex<0:
                dex = 0
            if dex>=len(in_files):
                dex = len(in_files)-1
                
            img, _ = unetsl.data.loadImage(in_files[dex])
            stack.append(img[0])
        
        prediction = predictImage(numpy.array([stack]), model)
        unetsl.data.saveImage(config["output file"].replace(".tif", "-%03d.tif"%time), prediction)
    
    
    
    
if __name__=="__main__":
    config = {
            unetsl.model.MODEL_FILE: sys.argv[1],
            "input files": sys.argv[2:],
            "output file": "time-volume-testing.tif"            
            }
    if unetsl.cli_interface.configure(config):
        main(config)
