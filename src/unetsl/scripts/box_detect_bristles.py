#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unetsl.model
import unetsl.data
import skimage.draw

import sys
import numpy

if __name__=="__main__":
    print("usage: box_detect_bristles.py model.h5 input.tif output.tif")
    img, _ = unetsl.data.loadImage(sys.argv[2])
    model = unetsl.model.loadModel(sys.argv[1])
    
    patch_size = unetsl.model.getInputShape(model)
    ch1 = img[0]
    
    d, h, w = ch1.shape
    
    pw = patch_size[-1]
    ph = patch_size[-2]
    
    labels = numpy.zeros(ch1.shape, dtype=ch1.dtype)
    
    for i in range(d):
        for j in range(0, h, ph//2):
            for k in range(0,w, pw//2):
                
                x=k
                if x+pw > w:
                    x = w-pw
                y = j
                if y+ph > h:
                    y = h - ph
                    
                
                patch = numpy.array([ch1[i:i+1, y:y+ph, x:x+pw]])
                cfy = model.predict(patch)
                if cfy[0,0]>cfy[0,1]:
                    ys, xs = skimage.draw.polygon_perimeter([y, y, y+ph, y+ph], [x, x+pw, x+pw, x], (h, w))
                    labels[i, ys, xs] = 255
    unetsl.data.saveImage(sys.argv[3], numpy.array([ch1, labels], dtype="uint8"))

    