# -*- coding: utf-8 -*-

#Model config.

#Training config

import json
import pathlib
    

INPUT_SHAPE = "input shape"
KERNEL_SHAPE = "kernel shape"
POOLING_SHAPE = "pooling"



LEARNING_RATE = "initial learning rate"
SEGMENTATIONS = "segmentations"
OPTIMIZER = "optimizer"
LOSS_FUNCTION = "loss function"
STRIDE = "stride"
BATCH_SIZE = "batch size"
ACTIVATION = "activation"
MULTI_GPU = "multi-gpu"
CROP = "crop"
DATA_SOURCES = "data sources"
N_LABELS = "n labels"
DEPTH = "depth"
N_FILTERS = "n filters" #TODO sitch key name
ACTIVATION = "activation"
CONDENSER = "condenser"
VALIDATION_FRACTION = "validation fraction"
EPOCHS = "epochs"
SPATIAL_DROPOUT_RATE = "spatial dropout rate"
SAMPLES_TO_PREDICT = "samples to predict"
NORMALIZE_SAMPLES = "normalize samples"
#data source

#[[ '*', '*', '*', '*'], [ '*', '*', '*', '*' ]], 

def parseExtendedOptions(extended_options):
    result = {}
    for line in extended_options:
        key, s_value = line.split("=")
        try:
            value = json.loads(s_value)
            result[key]=value
        except Exception as exc:
            print("could not parse %s and %s"%(key, s_value))
            raise Exception(exc)
            
    return result

def getDefaultTrainingConfig():
    dc = {
        LEARNING_RATE:0.000001,
        OPTIMIZER: "keras.optimizers.Adam",
        LOSS_FUNCTION:  "unetsl.model.sorensenDiceCoefLoss",
        STRIDE: (0, 3, 32, 32), 
        BATCH_SIZE: 1,
        MULTI_GPU: False,
        SAMPLES_TO_PREDICT: [],
        NORMALIZE_SAMPLES: False,
        VALIDATION_FRACTION: 0.03125,
        EPOCHS : 2000
    }
    return dc

def getDefaultModelConfig():
    parameters = {
            INPUT_SHAPE : (1,3, 64, 64),
            KERNEL_SHAPE : (3, 3, 3),
            POOLING_SHAPE : (1, 2, 2),
            N_LABELS : 1,
            DEPTH : 3,
            N_FILTERS : 32, 
            ACTIVATION : "sigmoid",
            SPATIAL_DROPOUT_RATE : 0.0625
        }
    return parameters

class ConfigurationTool:
    model_keys = set(k for k in getDefaultModelConfig())
    training_keys = set(k for k in getDefaultTrainingConfig())
    
    def __init__(self):
        self.data = dict()
        
    def __getitem__(self, key):
        if key not in ConfigurationTool.model_keys and key not in ConfigurationTool.training_keys:
            print("warning: unclassified key %s in config file"%key)
        return self.data[key]
    def __setitem__(self, key, value):
        if key not in ConfigurationTool.model_keys and key not in ConfigurationTool.training_keys:
            print("warning: unclassified key %s in config file"%key)
        self.data[key] = value
    def get(self, key, opt=None):
        if key in self.data:
            return self.data[key]
        if opt is not None:
            return opt
        raise KeyError(key)
        
    def getTrainingKeys(self):
        pass
    def getModelKeys(self):
        pass
    def update(self, conf):
        for key in conf:
            self[key] = conf[key]
    def items(self):
        return self.data.items()
    def load(self, config_path):
        if type(config_path) is str:
            config_path = pathlib.Path(config_path)
        with config_path.open("r") as c:
            self.update(json.load(c))
    def save(self, config_path):
        if type(config_path) is str:
            config_path = pathlib.Path(config_path)
        with config_path.open("w") as out:
            json.dump(self.data, out, indent="    ")
    def keys(self):
        return self.data.keys()
    def updateWithExtendedOptions(self, extended_options):
        self.update(parseExtendedOptions(extended_options))

def getDefaultConfigurationTool():
    tool = ConfigurationTool()
    tool.update(getDefaultModelConfig())
    tool.update(getDefaultTrainingConfig())
    return tool
