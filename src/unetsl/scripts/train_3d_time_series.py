#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unetsl.cli_interface
import unetsl.model
import unetsl.data

import sys
import pathlib
import re
import numpy
import keras
import random

def debugLog(message):
    with open("debug-log.txt", 'a') as f:
        f.write(message)
                
def createFlattenedModel(config):
    input_shape = config["input shape"]
    model = unetsl.model.createUnet3dModel(
            input_shape, 
            pool_size=config["pooling"],
            n_labels = config["nlabels"],
            kernel_shape = config["kernel shape"], 
            depth=config["depth"],
            n_filters=config["n_filters"],
            activation_name=config["activation"]
        )
    condenser = config["condenser"]
    n_labels = config["nlabels"]
    print("condenser: ", condenser)
    final_convolution = keras.layers.Conv3D(n_labels, condenser, padding="valid")(model.layers[-3].output)
    act = keras.layers.Activation(config["activation"])(final_convolution)
    
    model = keras.models.Model(inputs = model.layers[0].input, outputs=act)
    model.save(config["model file"]);
    
    
def trainModel(config):
    grab = re.compile("t(\\d+).*tif")
    
    input_folder = [ str(f) for f in pathlib.Path(config["input_folder"][0]).iterdir() if f.match("*tif")]
    n_inp = [(int(grab.findall(n)[0]), n) for n in input_folder]
    n_inp.sort()
    skeleton_folder = [ str(f) for f in pathlib.Path(config["skeleton_folder"][0]).iterdir() if f.match("*tif")]
    
    n_skel = [(int(grab.findall(n)[0]), n) for n in skeleton_folder]
    n_skel.sort()
    
    files = [pair for pair in zip([a[1] for a in n_inp], [b[1] for b in n_skel])]
    
    
    model = unetsl.model.loadModel(config["model file"])
    patch_size = unetsl.model.getInputShape(model)
    out_patch_size = unetsl.model.getOutputShape(model)
    print("input shape: ", patch_size)
    print("out_patch_size: ", out_patch_size)
    channels = patch_size[0]
    batch = config[unetsl.model.BATCH_SIZE]
    #random.shuffle(files) cannot shuffle 
    
    n = len(files)*6//8
    
    n_v = len(files) - n
    print(config)
    train_gen = unetsl.data.TimeSeriesDataGenerator(files[:n], patch_size, out_patch_size, channels, batch, config["crop"], config["stride"])
    valid_gen = unetsl.data.TimeSeriesDataGenerator(files[n:], patch_size, out_patch_size, channels, batch, config["crop"], config["stride"])
    
    
    lr = config[unetsl.model.LEARNING_RATE]
    loss_fun = unetsl.model.getLossFunction(config[unetsl.model.LOSS_FUNCTION])
    optimizer = unetsl.model.getOptimizer(config[unetsl.model.OPTIMIZER], lr)
    
    logger = unetsl.model.LightLog(config["output file"], model)
    
    tc = str(logger.file).replace(".txt", ".json")
    unetsl.model.saveConfig(config, pathlib.Path(tc))
    
    
    
    if config["multi-gpu"]:
        from keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=2)
    
    unetsl.model.recompileModel(
            model, 
            optimizer, 
            loss_function=loss_fun)
    print(n, " files for training, ", len(files)-n, "files for validation")
    model.fit_generator(generator=train_gen.getGenerator(),
                        steps_per_epoch=train_gen.getCount(),
                        epochs=20000,
                        validation_data=valid_gen.getGenerator(),
                        validation_steps=valid_gen.getCount(),
                        callbacks=[logger]
                        )

    
def get3DTimeSeriesDefaults(config):
    config["input_folder"] = []
    config["skeleton_folder"] = []
    config["multi_gpu"] = False
    config["condenser"] = [1, 1, 1]
    
def main():
    print("train_3d_time_series -c config.json data_dir seg_dir")
    
    config = unetsl.model.getDefaultTrainingConfig()
    mc = unetsl.model.getDefaultConfig()
    config.update(mc)
    get3DTimeSeriesDefaults(config)
    
    if len(sys.argv)>2:
            if sys.argv[1] == "-c":
                config_path = pathlib.Path(sys.argv[2])
                if config_path.exists():
                    unetsl.model.loadConfig( config, config_path )
            else:
                print("a config file must be specified. If it doesn't exist,", 
                       "it will be created")                
    else:
        sys.exit(0)
    
    if unetsl.cli_interface.configure(config):
        if not pathlib.Path(config["model file"]).exists():
            createFlattenedModel(config)
        unetsl.model.saveConfig(config, config_path)
        trainModel( config )
    
    
    
    
if __name__=="__main__":
    main()
