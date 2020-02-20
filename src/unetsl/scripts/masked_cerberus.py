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

def getCerberusPredictor(model, batch_size=1, gpus=1):
    batch=True #no config.
    
    config = {
            unetsl.predict.DEBUG : False,
            unetsl.NORMALIZE_SAMPLES: False,
            unetsl.BATCH_SIZE : batch_size
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
@click.option("--gpus", default=1, envvar="GPUS")
@click.option("--batch_size", default=2, envvar="BATCH_SIZE")
@click.option("--chunk_size", default=1, envvar="CHUNK_SIZE")
def predict(masker_file, cerberus_file, img_file, gpus, batch_size, chunk_size):
    masker_file = pathlib.Path(masker_file)
    img_file = pathlib.Path(img_file)
    cerberus_file = pathlib.Path(cerberus_file)
    
    mm = unetsl.masker.MaskerModel( (1, 384, 384, 384) )
    mm.loadModel(masker_file)
        
    img_stack, tags = unetsl.data.loadImage(img_file)
    
    
    
    stacks = []
    #for i in range(len(img_stack)):
    cerb_predictor = getCerberusPredictor(unetsl.model.loadModel(cerberus_file), batch_size=batch_size, gpus=gpus)
    
    start = time.time()
    print(start, gpus, chunk_size, batch_size)
    for i in range(0, img_stack.shape[0], chunk_size):
        chunk = img_stack[i:i+chunk_size]
        dots = "".join(["."]*chunk.shape[0])
        print(dots, end=" ")
        pred = mm.predictImages(img_stack[i:i+chunk_size])
        pred2, debug = cerb_predictor.predictImage(pred)
        stacks.append(pred2)
    with open("log.txt", 'a', encoding="UTF8") as fi:
        fi.write("%s %s %s %s\n"%(gpus, chunk_size, batch_size, time.time()-start))
    print(time.time() - start)
    
    for i in range(3):
        prediction = numpy.concatenate([ slc[:, i:i+1] for slc in stacks], axis=0)
        out_name = "pred-c%d-%s-%s-%s"%(
                i,
                masker_file.name.replace(".h5", ""), 
                cerberus_file.name.replace(".h5", ""),
                img_file.name
            )
        unetsl.data.saveImage( out_name, prediction, tags)
    
    

if __name__=="__main__":
    
    predict()    
    