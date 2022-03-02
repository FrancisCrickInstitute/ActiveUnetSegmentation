# -*- coding: utf-8 -*-
import unetsl
import unetsl.config

import pathlib
import json



class HeadConstants:
    name = "name"
    n_labels = "n_labels"
    bits = "bits"
    activation = "activation"
    depth = "depth"
    resampler = "resampler"
    offset = "offset"

def getDefaultCerberusConfig():
    unet = unetsl.config.getDefaultModelConfig()
    
    unet.pop(unetsl.N_LABELS, None)
    unet.pop(unetsl.ACTIVATION, None)
    
    dflt = {"unet" : unet,
            "heads": [],
            unetsl.DATA_SOURCES : []
            }
    default_heads = [
            ("distance", 1, 6, 2, "relu", 0, unetsl.data.LINEAR_LABELS, "min pool"),
            ("membrane-scale", 2, 2, 0, "sigmoid", 1, unetsl.data.MULTICLASS_LABELS, "max pool"),
            ("membrane-crop", 2, 2, 0, "sigmoid", 0, unetsl.data.MULTICLASS_LABELS, "crop") 
            ]
    
    for nm, n_labels, bits, offset, activation, depth, labeller_name, resampler in default_heads:
        hd = getDefaultHeadConfig()
        hd[HeadConstants.name] = nm
        hd[HeadConstants.n_labels] = n_labels
        hd[HeadConstants.bits] = bits
        hd[HeadConstants.offset] = offset
        hd[HeadConstants.activation] = activation
        hd[HeadConstants.depth] = depth
        hd[unetsl.data.LABELLER] = labeller_name
        hd[HeadConstants.resampler] = resampler
        dflt["heads"].append(hd)
    return dflt

def getDefaultCerberusPredictionConfig():
    return {
            }

def saveConfig(cfg, pth):
    with open(pth, 'w', encoding="utf8") as f:
        json.dump(cfg, f, indent="  ")
    

def guessDefaultLossFunction(name):
    """
      For setting up a default model that "works" without changing any parameters 
      during create/attach/train.
      
      'distance' is a head with a distance transform output, this presumes
      linear labels, and the logMse. This seems a bit more robust than the mse
      and it prevents the distance transform from dominating the loss optimization.
      
      'membrane' or 'skeleton' would be a sparse labelling where the dice
      coefficient has shown to work well.
      
    """
    if 'distance' in name:
        return "unetsl.model.logMse"
    elif 'membrane' in name or 'skeleton' in name:
        return "unetsl.model.sorensenDiceCoefLoss"
    
    return "keras.losses.mean_squared_error"

def getTrainingConfig(config):
    """
        Checks the provided config for traning config, if it isn't present, 
        a defulat version will be loaded.
        
    """
    if config is None:
        config = getDefaultCerberusConfig()
    
    if "training" not in config:
        training_config = dict()
        training_config.update( unetsl.config.getDefaultTrainingConfig() )
        
        training_config[unetsl.LOSS_FUNCTION] = {}
        training_config["loss weights"] = {}
        config["training"] = training_config
    else:
        training_config = config["training"]
        
    loss_fns = training_config[unetsl.LOSS_FUNCTION]
    loss_wts = training_config["loss weights"]
    
    for head in config["heads"]:
        if head["name"] not in loss_fns:
            loss_fns[ head["name"] ] = guessDefaultLossFunction(head["name"])
        if head["name"] not in loss_wts:
            loss_wts[ head["name"] ] = 1
    
    return training_config

def getDefaultHeadConfig():
    cfg = {
            HeadConstants.name : None,
            HeadConstants.n_labels : -1,
            HeadConstants.bits : -1,
            HeadConstants.offset: -1,
            HeadConstants.activation : "sigmoid",
            HeadConstants.depth : 0, 
            HeadConstants.resampler : "max pool"
            }
    return cfg

def loadConfig(config_path):
    pth = pathlib.Path(config_path)
    if pth.exists():
        cfg = json.load(open(pth, 'r'))
    else:
        cfg = getDefaultCerberusConfig()
    return cfg
