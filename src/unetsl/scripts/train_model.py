#!/usr/bin/env python3

import sys, pathlib, os


import unetsl.cli_interface
import unetsl.model
import unetsl.data
import unetsl.config

import math
import time

import tensorflow as tf
import click


def trainModel(config, n_gpus, base_name):
    input_file = base_name
    model_output_file = base_name
    if pathlib.Path(input_file).exists():
        model = unetsl.model.loadModel(input_file)
    else:
        print("model first needs to be created, %s does not exist"%input_file)
        return
        
    input_shape = unetsl.model.getInputShape(model)
    output_shape = unetsl.model.getOutputShape(model)
    
    
    
    label_count = output_shape[0]
    
    print("in: ", input_shape)
    print("out: ", output_shape)
    
    normalize_samples = config[unetsl.NORMALIZE_SAMPLES]
    data_sources = unetsl.data.getDataSources(config[unetsl.DATA_SOURCES], normalize_samples)
    patch_size = input_shape
    stride = config[unetsl.STRIDE]
    batch_size = config[unetsl.BATCH_SIZE]
    validation_fraction = config[unetsl.VALIDATION_FRACTION]
    #TODO - these...
    #getDataGenerators(data_sources,n_labels, patch_size, stride, out_patch_size, batch_size, validation_split)
    training_generator, validation_generator = unetsl.data.getDataGenerators(
            data_sources, label_count, patch_size, stride, batch_size, validation_fraction
    )
    training_batches = training_generator.getNBatches()
    validation_batches = validation_generator.getNBatches()

    run_id = int(time.time()*1000)
    base = pathlib.Path(model_output_file).name.replace(".h5", "")
    e_filename = "training-log_%s-%08x.txt"%(base, run_id)
    b_filename = "batch-log_%s-%08x.txt"%(base, run_id)
    efile = pathlib.Path(e_filename)
    bfile = pathlib.Path(b_filename)
    while efile.exists() or bfile.exists():
        run_id += 1
        e_filename = "training-log_%s-%08x.txt"%(base, run_id)
        b_filename = "batch-log_%s-%08x.txt"%(base, run_id)
        efile = pathlib.Path(e_filename)
        bfile = pathlib.Path(b_filename)
    loggers = []
    logger = unetsl.model.LightLog(model_output_file, model, filename = str(efile))
    loggers.append(logger)
    tc = str(logger.file).replace(".txt", ".json")
    batch_logger = unetsl.model.BatchLog(model_output_file, model, filename = str(bfile))
    loggers.append(batch_logger)
    
    if len(config[unetsl.SAMPLES_TO_PREDICT])>0:
        reduction_type = 1
        
        known = [ "softmax", "sigmoid", "relu" ]
        at = config[unetsl.ACTIVATION]
        if at in known:
            reduction_type = known.index(at)
        
        epoch_predictions = unetsl.model.EpochPredictions(
                [ os.path.expandvars(s) for s in config[unetsl.SAMPLES_TO_PREDICT] ], 
                base, 
                model, 
                reduction_type=reduction_type, 
                n_gpus=n_gpus
        )
        loggers.append(epoch_predictions)
    config.save(pathlib.Path(tc))
    
    if config[unetsl.MULTI_GPU]:
        print("making multi-gpu")
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=n_gpus)
        
    lr = config[unetsl.LEARNING_RATE]
    loss_fun = unetsl.model.getLossFunction(config[unetsl.LOSS_FUNCTION])
    optimizer = unetsl.model.getOptimizer(config[unetsl.OPTIMIZER], lr)
    
    print("validation total: ", validation_batches, " train total: ", training_batches)
        
    
    unetsl.model.recompileModel(
            model, 
            optimizer, 
            loss_function=loss_fun)
    model.fit( x = (( x, y ) for (x,y) in training_generator),
                        steps_per_epoch=training_batches,
                        epochs=config[unetsl.EPOCHS],
                        validation_data=( (x, y) for (x,y) in validation_generator ),
                        validation_steps=validation_batches,
                        callbacks=loggers, 
                        verbose=2
                        )

@click.command()
@click.option("-c", "--config-file", required=True, type=click.Path(exists=True), prompt=True )
@click.option("-b", "--batch", is_flag=True)
@click.option("--gpus", envvar="GPUS", default=1, type=int)
@click.option("-D", "extended_options", multiple=True)
@click.option("--allow-growth", is_flag=True) #hack!
def main(config_file, batch, gpus, extended_options, allow_growth):
    
    config = unetsl.config.getDefaultConfigurationTool()
    
    config_path = pathlib.Path(config_file)
    config.load(config_path)
    config.updateWithExtendedOptions(extended_options)
    if not batch:
        if unetsl.cli_interface.configure(config):
            config.save(config_file)
        else:
            return;
    model_name = str(config_file).replace(".json", ".h5")
    trainModel(config, gpus, model_name)
    
    
    
    

if __name__=="__main__":
    main()
