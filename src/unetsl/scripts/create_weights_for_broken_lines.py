#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import sys
import unetsl, unetsl.data, unetsl.model, unetsl.predict
import numpy
import skimage
import scipy.ndimage.filters as filters



point_kernel = numpy.array( 
                        [
                            [ 1,  1,  1],
                            [ 1, 10,  1],
                            [ 1,  1,  1]
                        ] )

spread_kernel = numpy.array([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
        ])

endpoint_kernel = numpy.array( 
                        [
                            [ 1,  1,  1],
                            [ 1,  0,  1],
                            [ 1,  1,  1]
                        ] )

def getArgument(option):
    if option in sys.argv:
        return sys.argv[sys.argv.index(option) + 1]
    else:
        return None
    
def main():
    """
        This will go through a predict the skeleton, then go through the 
        predicted skeleton, 
        
    """
    
    print("usage: \ncreate_weights_for_broken_lines.py -m model.h5* -f image.tif -o out.tif")
    print("or: create_weights_for_broken_lines.py -p prediction.tif -o out.tif")
    if '-o' not in sys.argv:
        print("An output image must be specified using -o")
        sys.exit(-1);
    out_filename = getArgument("-o")
    
    if '-p' in sys.argv:
        pred, tags = unetsl.data.loadImage(getArgument('-p'))
        
    elif '-m' in sys.argv:
        model = unetsl.model.loadModel(getArgument('-p'))
        
        if '-f' not in sys.argv:
            print("when using -m, -f must also be specified")
            sys.exit(-1)
        img, tags = unetsl.data.loadImage(getArgument('-f'))
        pred = unetsl.predict.predictImage(model, img)
    else:
        print("must specify either -m to specify a model to create prediction, or -p to specify a prediction to use.")
        sys.exit(-1)
        
    weights = []
    for label in pred:
        for slc in label:
            skeleton = (slc>0)*1.0
            skeleton = skimage.morphology.skeletonize(skeleton)*1.0
            broken = filters.convolve( skeleton, point_kernel, mode='constant', cval=1.0)
            fillers = 20*(filters.convolve(broken, endpoint_kernel, mode='constant', cval=0.0)==2)
            
            #broken = (broken==11)*5
            broken = fillers + numpy.array(skeleton, dtype="uint8")*2
            broken = filters.convolve( broken, spread_kernel, mode='constant', cval=0)
            
            weights.append(broken)
    unetsl.data.saveImage(out_filename, numpy.array(weights, dtype='uint8'), tags)

if __name__=="__main__":
    main()
    
