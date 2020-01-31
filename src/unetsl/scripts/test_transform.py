#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import unetsl.data
import sys
import skimage.transform as transform

import math, numpy

from matplotlib import pyplot
def getCropStride(stride, angle):
    """
        Gets the stride required for the corresponding rotation angle.
    """
    new_stride = [int(s) for s in stride]
    cos = math.cos(angle)
    sin = math.sin(angle)
    
    if angle>=0 and angle<math.pi/2:
        new_stride[-1] = cos*stride[-1] + sin*stride[-2]
        new_stride[-2] = sin*stride[-1] + cos*stride[-2]
    elif angle>=math.pi/2 and angle<math.pi:
        new_stride[-1] = -cos*stride[-1] + sin*stride[-2]
        new_stride[-2] = sin*stride[-1] - cos*stride[-2]
    elif angle>=math.pi and angle<3*math.pi/2:
        new_stride[-1] = -cos*stride[-1] - sin*stride[-2]
        new_stride[-2] = - sin*stride[-1] - cos*stride[-2]
    elif angle>=1.5*math.pi:
        new_stride[-1] = + cos*stride[-1] - sin*stride[-2]
        new_stride[-2] = - sin*stride[-1] + cos*stride[-2]

    new_stride = [int(s) for s in new_stride]
    return new_stride

    
if __name__=="__main__":
    print("testings transforms")
    
    stride = (1, 8, 64, 64)
    img, _tags = unetsl.data.loadImage(sys.argv[1])
    print(img.shape)
    new_stride = getCropStride(stride, math.pi/6)
    print(new_stride)
    patch = img[0:new_stride[0],
                14:14 + new_stride[1],
                150: 150+new_stride[2], 
                150: 150 + new_stride[2]
                ]
    fig = pyplot.figure(0)
    pyplot.imshow(patch[0,4])
    fig1 = pyplot.figure(1)
    trans = []
    angle = 30
    for sl in patch[0]:
        trans.append(transform.rotate(sl, angle))
        
    trans = numpy.array([trans])
    print(trans.shape)
    
    ox = (new_stride[-1] - stride[-1])//2
    oy = (new_stride[-2] - stride[-2])//2
    trans = trans[0:stride[0], 0:stride[1], oy: oy + stride[2], ox : ox + stride[3]]
    
    pyplot.imshow(trans[0,4])
    
    
    
    pyplot.show()
    
    