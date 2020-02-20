#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import unetsl.cerberus
import unetsl.model

import unetsl.masker

import numpy

import click

def guessShaper(key):
    """
        REDUNDANT see unetsl.cerberus.__main__
        
    """
    kl = key.lower()
    if "crop" in kl:
        return "crop"
    
    return "upsample"

def getCerberusPredictor(model, batch=True, gpus=2):
    config = {
            unetsl.predict.DEBUG : False,
            unetsl.NORMALIZE_SAMPLES: False,
            unetsl.BATCH_SIZE : 4
        }
    output_map = unetsl.model.getOutputMap(model)
        
    rtm = {}
    sm = {}
    
    for key in output_map:
        rtm[key] = unetsl.predict.guessReduction(output_map[key])
        sm[key] = guessShaper(key)
    
    
    tune_config = {
        unetsl.predict.REDUCTION_TYPE : rtm,
        unetsl.predict.LAYER_SHAPER : sm
    }

    if batch:
        pass
    elif not unetsl.cli_interface.configure(tune_config):
        #cancelled
        return 0
    
    if gpus>1:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=gpus)
        
    
    predictor = unetsl.predict.MultiChannelPredictor(model, None)
    
    rtm = tune_config[unetsl.predict.REDUCTION_TYPE]
    predictor.reduction_types = tuple( rtm[key] for key in rtm )
    
    lsm = tune_config[unetsl.predict.LAYER_SHAPER]
    
    predictor.layer_shapers = tuple( unetsl.predict.getShaper( lsm[key] ) for key in lsm)
    predictor.batch_size = config[unetsl.BATCH_SIZE]
    predictor.debug = config[unetsl.predict.DEBUG]
    predictor.sample_normalize = config[unetsl.NORMALIZE_SAMPLES] 
    predictor.batch_size = config[unetsl.BATCH_SIZE]
    predictor.GPUS = gpus
    
    return predictor

import time
    

@click.command()
@click.argument("masker_file")
@click.argument("cerberus_file")
@click.argument("img_file")
def predict(masker_file, cerberus_file, img_file):
    masker_file = pathlib.Path(masker_file)
    img_file = pathlib.Path(img_file)
    cerberus_file = pathlib.Path(cerberus_file)
    
    mm = unetsl.masker.MaskerModel( (1, 384, 384, 384) )
    mm.loadModel(masker_file)
        
    img_stack, tags = unetsl.data.loadImage(img_file)
    
    
    out_name = "pred-%s-%s-%s"%(
            masker_file.name.replace(".h5", ""), 
            cerberus_file.name.replace(".h5", ""),
            img_file.name
        )
    stacks = []
    #for i in range(len(img_stack)):
    cerb_predictor = getCerberusPredictor(unetsl.model.loadModel(cerberus_file))
    for i in range(img_stack.shape[0]):
        print(time.time(), i, "/", img_stack.shape[0])
        pred = mm.predictImages(img_stack[i:i+1])
        pred2, debug = cerb_predictor.predictImage(pred)
        stacks.append(pred2)
    prediction = numpy.concatenate(stacks, axis=0)
    unetsl.data.saveImage( out_name, prediction, tags)

    

if __name__=="__main__":
    
    predict()    
    