# -*- coding: utf-8 -*-

import sys
import unetsl.data
import numpy
import skimage.measure
import skimage.morphology
import skimage.feature

from matplotlib import pyplot

if __name__=="__main__":
    img, tags = unetsl.data.loadImage(sys.argv[1])
    frame = img[0,0]
    thresh = (frame>4)*1
    
    local_maxi = skimage.feature.peak_local_max(frame, indices=False, footprint=numpy.ones((5, 5)), labels=thresh)
    
    lbled = skimage.measure.label(thresh, background=0, connectivity=1)
    print(lbled.shape, lbled.dtype)
    ws = skimage.morphology.watershed(-thresh, lbled)
    s = skimage.morphology.skeletonize(skimage.filters.sobel(ws)!=0)*1
    pyplot.figure(0)
    pyplot.imshow(ws)
    pyplot.figure(1)
    pyplot.imshow(s)
    pyplot.figure(2)
    pyplot.imshow(lbled)
    pyplot.show()
    