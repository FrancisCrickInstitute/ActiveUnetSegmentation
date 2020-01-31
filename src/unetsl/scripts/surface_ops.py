# -*- coding: utf-8 -*-

#!/usr/bin/env python3
import tensorflow
import numpy
import unetsl.data
import sys

def getGaussianKernel(sigma, N=128):
    xi = numpy.repeat(   
            numpy.arange(N).reshape( (1, N) ), 
            axis=0, 
            repeats = N
        )
    yi = numpy.repeat( numpy.arange(N).reshape((N, 1)), axis = 1, repeats=N)
    
    kernel = numpy.exp( 
            ( (xi - N//2)**2 + (yi - N//2)**2 ) 
                      / ( - 2 * sigma**2)  ).reshape((1, 128, 128, 1, 1))
    return kernel

def main(image_generator, result_manager):
    #inp = tensorflow.constant(img, dtype="float32")
    inp = tensorflow.placeholder(dtype="float32", name="image_to_blur")
    
    kernel = getGaussianKernel( 40 )
    #blurred = scipy.ndimage.convolve(inp, kernel)
    
    blurred = tensorflow.nn.conv3d(
            inp, 
            filter=kernel, 
            strides=(1, 1, 1, 1, 1), 
            padding="SAME",
            data_format="NCDHW"
            )
    mxed = tensorflow.argmax( blurred, axis=2, output_type=tensorflow.dtypes.int32 )
    
    height_kernel = getGaussianKernel( 10 )
    
    blurred_mxed = tensorflow.nn.conv2d( 
            tensorflow.cast(mxed, dtype = "float32"), 
            height_kernel[0], 
            strides=(1, 1, 1, 1), 
            padding="SAME", 
            data_format="NCHW" 
            )
    with tensorflow.Session() as sess:
        for name, img in image_generator:
            res = blurred_mxed.eval({inp.name: img})
            result_manager(name, res)
            
    return res
    
def shapeImage(img):
    return img.reshape((1, *img.shape))
    

class HeightMapStacker:
    def __init__(self):
        self.tags = {}
        self.output = []
    def imageGenerator(self, images):
        stack = []
        for image in images:
            img, tags = unetsl.data.loadImage(image)
            self.tags.update(tags)
            stack.append(img)
            if len(stack)==1:
                yield "stacked", numpy.array(stack)
                stack.clear()
        if len(stack)>0:
            yield "stacked", numpy.array(stack)
    
    def accumulate(self, image_name, image):
        for slc in image:
            self.output.append([ slc ])
            
if __name__=="__main__":
    hms = HeightMapStacker()
    
    out = main( hms.imageGenerator(sys.argv[1:]), hms.accumulate )
    
    
    unetsl.data.saveImage( "test.tif", numpy.array(hms.output, dtype="float32"), hms.tags)