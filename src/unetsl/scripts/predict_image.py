#!/usr/bin/env python3

import numpy
from tifffile import TiffWriter
import sys

import unetsl.cli_interface
import unetsl.model
import unetsl.data
import unetsl.predict
import unetsl.management

import unetsl
import pathlib

import json
import click

def getFileName(str_path):
    return pathlib.PurePath(str_path).name

    

@click.command()
@click.argument("model", type=click.Path(exists=True))
@click.argument("image", type=click.Path(exists=True))
@click.argument("prediction", required=False, default = None, type=click.Path())
@click.option("-D", "extended_options", multiple=True)
@click.option("-b", "--batch", is_flag=True)
@click.option("--gpus", envvar="GPUS", default=1, type=int)
@click.option("-m", "multi_channel", is_flag=True)
def main(model, image, prediction, batch, extended_options, gpus, multi_channel):
    print("usage: \n predict_image.py model.h5 image.tif output.tif")
    extended = unetsl.config.parseExtendedOptions(extended_options)
    output_index = extended.get(unetsl.predict.OUTPUT_INDEX, -1)
    if prediction is None:
        
        prediction = unetsl.management.getOutputName(
                getFileName(model), 
                getFileName(image),
                output_index
                )
    
    if multi_channel:
        predictor = unetsl.predict.predictMultiChannelImage
        default_reduction = []
        default_shaper = []
        output_index = [-1]
    else:
        predictor = unetsl.predict.predictImage
        default_reduction = 1
        default_shaper = "upsample"
        
    config = {
            unetsl.predict.MODEL_KEY : model,
            unetsl.predict.IMG_KEY : image,
            unetsl.predict.OUT_KEY : prediction,
            unetsl.predict.REDUCTION_TYPE : default_reduction,
            unetsl.predict.DEBUG : False,
            unetsl.predict.OUTPUT_INDEX: output_index,
            unetsl.NORMALIZE_SAMPLES: False,
            unetsl.BATCH_SIZE : 16,
            unetsl.predict.LAYER_SHAPER : default_shaper
            }
        

    config.update(extended)
    
    if batch:
        print("batch mode")
    elif not unetsl.cli_interface.configure(config):
        #cancelled
        return 0
    
    
    
    model = unetsl.model.loadModel(config[unetsl.predict.MODEL_KEY])
    if gpus>1:
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=gpus)
        
    image, tags = unetsl.data.loadImage(config[unetsl.predict.IMG_KEY])
    output_index = config[unetsl.predict.OUTPUT_INDEX]
    sample_normalize = config[unetsl.NORMALIZE_SAMPLES]
    
    if multi_channel:
        predictor = unetsl.predict.predictMultiChannelImage
    else:
        predictor = unetsl.predict.predictImage
    
    out, debug = predictor( 
            model, image, 
            reduction_type=config[unetsl.predict.REDUCTION_TYPE], 
            debug=config[unetsl.predict.DEBUG],  
            output_index=output_index, 
            sample_normalize=sample_normalize, 
            batch_size=config[unetsl.BATCH_SIZE], 
            GPUS=gpus, shaper=unetsl.predict.getShaper(config[unetsl.predict.LAYER_SHAPER]));
    unetsl.data.saveImage(config[unetsl.predict.OUT_KEY], out, tags)

    if config[unetsl.predict.DEBUG]:
        print(debug.shape)
        unetsl.data.saveImage("debug.tif", numpy.array([debug], dtype="float32"), tags)
        
                
if __name__=="__main__":
    main()
