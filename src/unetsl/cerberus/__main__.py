#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:35:51 2019

@author: smithm3
"""
import click
import unetsl.cerberus as cerberus
import unetsl.cerberus.cerberus_config as cerberus_config
import unetsl.cli_interface as client

import numpy

import unetsl
import json
import pathlib
import time

@click.group(chain=True)
def cerbs():
    pass

@cerbs.command("train")
@click.option("-c", "config_file")
@click.option("--gpus", envvar="GPUS", default=1, type=int)
@click.option("-b", "--batch", is_flag=True)
def trainCerberusModel(config_file, gpus, batch):
    cfg = cerberus_config.loadConfig(config_file)
    unet_cfg = cfg["unet"]
    train_cfg = cerberus_config.getTrainingConfig(cfg)
    
    if not batch: 
        if not client.configure(train_cfg):
            click.echo("canceled")
            return 0
        cerberus_config.saveConfig(cfg, pathlib.Path(config_file))
    
    
    
    loss_fns = cerberus.getLossFunctions(train_cfg[unetsl.LOSS_FUNCTION])
    input_file = train_cfg[unetsl.MODEL_FILE]
    model_output_file = train_cfg[unetsl.MODEL_OUT]
    
    if pathlib.Path(input_file).exists():
        model = unetsl.model.loadModel(input_file)
    else:
        click.echo("model first needs to be created, %s does not exist"%input_file)
        return
    head_configs = cfg["heads"]
    input_shape = cfg["unet"][unetsl.INPUT_SHAPE]
    pool_shape = cfg["unet"][unetsl.POOLING_SHAPE]
    heads = cerberus.loadHeads(head_configs, input_shape, pool_shape)
    
    output_shapes = cerberus.getOutputShapes(model)
    
    click.echo("%d model outputs"%(len(output_shapes)))
    for ops in output_shapes:
        click.echo("\t %s"%(ops, ))
    input_shapes = cerberus.getInputShapes(model)
    input_shape = input_shapes[-1]
    
    pool = unet_cfg[unetsl.POOLING_SHAPE]
    
    data_sources = unetsl.data.getDataSources(
            cfg[unetsl.DATA_SOURCES], 
            normalize_samples=train_cfg[unetsl.NORMALIZE_SAMPLES]
            )

    patch_size = input_shape
    stride = train_cfg[unetsl.STRIDE]
    batch_size = train_cfg[unetsl.BATCH_SIZE]
    validation_split = train_cfg[unetsl.VALIDATION_FRACTION]
    
    training_generator, validation_generator = unetsl.cerberus.getDataGenerators(
            data_sources, patch_size, stride, batch_size, validation_split, heads
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
    
    
    logger = unetsl.model.LightLog(model_output_file, model, filename = str(efile))
    tc = str(logger.file).replace(".txt", ".json")
    batch_logger = unetsl.model.BatchLog(model_output_file, model, filename = str(bfile))
    
    click.echo("# training log %s \n#batch log %s \n#config log %s"%(e_filename, b_filename, tc))
    
    cerberus_config.saveConfig(cfg, pathlib.Path(tc))
    
    lr = train_cfg[unetsl.LEARNING_RATE]
    
    optimizer = unetsl.model.getOptimizer(train_cfg[unetsl.OPTIMIZER], lr)
    
    click.echo("validation total: %s train total: %s"%(validation_batches, training_batches ))
    
    if train_cfg[unetsl.MULTI_GPU] and gpus>1:
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=gpus)
    
    model.compile(            
            optimizer=optimizer, 
            loss=loss_fns,
            metrics=['binary_accuracy', 'accuracy'], 
            loss_weights=train_cfg["loss weights"]
        )
    
    model.fit( (( x , y ) for (x,y) in training_generator),
                        steps_per_epoch=training_batches,
                        epochs=train_cfg[unetsl.EPOCHS],
                        validation_data=( (x,y) for (x,y) in validation_generator ),
                        validation_steps=validation_batches,
                        callbacks=[logger, batch_logger], 
                        verbose=2
                        )

@cerbs.command("inspect")
@click.option("-c", "config_file", prompt=True)
def inspectCerberusDataSources(config_file):
    from unetsl.data import VolumeViewer
    
    cfg = cerberus_config.loadConfig(config_file)
    data_sources = unetsl.data.getDataSources(cfg[unetsl.DATA_SOURCES], False)
    head_configs = cfg["heads"]
    input_shape = cfg["unet"][unetsl.INPUT_SHAPE]
    pool_shape = cfg["unet"][unetsl.POOLING_SHAPE]
    heads = cerberus.loadHeads(head_configs, input_shape, pool_shape)
    
    
    volumes = []
    DO_ALL=False
    for source in data_sources:
        data_generator = cerberus.getCerberusDataGenerator(heads, source, input_shape)
        for batch in data_generator:
            x = batch[0]
            y_dict = batch[1]
            
            # [ (c, z, y, x), (c, z, y, x) ... ] for n outputs.
            y_list = [y_dict[key] for key in y_dict]
            fig = 0
            if len(volumes)==0:
                volumes.append(VolumeViewer(fig, x[0]))
                for ys in y_list:
                    fig += 1
                    #batch = 1, channel
                    click.echo(ys.shape)
                    volumes.append(VolumeViewer(fig, ys[0], limits=(0,1)))
            
            else:
                datas = [x] + y_list
                for i, data in enumerate(datas):
                    volumes[i].setData(data[0])
            if not DO_ALL:
                check = input("return to continue, n for next, a for all: ")
                if check=="n" :
                    break
                if check=="a":
                    DO_ALL=True

def guessReductionType(key, layer):
    """
        For guess the reduction type just given the layer name as it appears
        in the output map, and the output layer.
        - softmax : categorical
        - sigmoid : one hot
        - linear : linear
    """
    return unetsl.predict.guessReduction(layer)

def guessShaper(key, layer):
    """
        uses the name to find a known shaper otherwise uses the default, up
        sample. Possibly not used if the output size is the desired depth.
        
    """
    kl = key.lower()
    if "crop" in kl:
        return "crop"
    
    return "upsample"
    

@cerbs.command("predict")
@click.argument( "model_file", type=click.Path(exists=True) )
@click.argument( "input_image", type=click.Path(exists=True) )
@click.argument( "output_image", type=click.Path(), default=None, required=False)
@click.option("-D", "extended_options", multiple=True)
@click.option("-b", "--batch", is_flag=True)
@click.option("--gpus", envvar="GPUS", default=1, type=int)
def predictionCommand(model_file, input_image, output_image, batch, extended_options, gpus):
    """
        -local implementation for using cerberus specific commands. eg: layer names,
        multiple reduction and expansion types
        
    """
    if output_image is None:
        img_name = pathlib.Path(input_image).name
        m_name = pathlib.Path(model_file).name
        output_image = "pred-%s-%s"%( m_name.replace(".h5", ""), img_name )
    
    config = {
            unetsl.predict.MODEL_KEY : model_file,
            unetsl.predict.IMG_KEY : input_image,
            unetsl.predict.OUT_KEY : output_image,
            unetsl.predict.DEBUG : False,
            unetsl.NORMALIZE_SAMPLES: False,
            unetsl.BATCH_SIZE : 16,
        }
    config.update(unetsl.config.parseExtendedOptions(extended_options))
    
    if batch:
        click.echo("batch mode")
    else:
        if not unetsl.cli_interface.configure(config):
            #cancelled
            return 0
    
    model = unetsl.model.loadModel(config[unetsl.predict.MODEL_KEY])
    
    output_map = unetsl.model.getOutputMap(model)
    
    rtm = {}
    sm = {}
    
    for key in output_map:
        rtm[key] = guessReductionType(key, output_map[key])
    for key in output_map:
        sm[key] = guessShaper(key, output_map[key])
    
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
        from tensorflow.keras.utils import multi_gpu_model
        model = multi_gpu_model(model, gpus=gpus)
        
    image, tags = unetsl.data.loadImage(config[unetsl.predict.IMG_KEY])
    print(image.shape)
    sample_normalize = config[unetsl.NORMALIZE_SAMPLES]
    
    predictor = unetsl.predict.MultiChannelPredictor(model, image)
    
    rtm = tune_config[unetsl.predict.REDUCTION_TYPE]
    predictor.reduction_types = tuple( rtm[key] for key in rtm )
    
    lsm = tune_config[unetsl.predict.LAYER_SHAPER]
    
    predictor.layer_shapers = tuple( unetsl.predict.getShaper( lsm[key] ) for key in lsm)
    predictor.batch_size = config[unetsl.BATCH_SIZE]
    predictor.debug = config[unetsl.predict.DEBUG]
    predictor.sample_normalize = config[unetsl.NORMALIZE_SAMPLES] 
    predictor.batch_size = config[unetsl.BATCH_SIZE]
    predictor.GPUS = gpus
    
    out, debug = predictor.predict()
    print("out shape: ", out.shape )
    if config[unetsl.predict.DEBUG]:
        print("debug shape: ", debug.shape)
    unetsl.data.saveImage(config[unetsl.predict.OUT_KEY], out, tags)
    
    if config[unetsl.predict.DEBUG]:
        click.echo("saving debug data as debug.tif")
        unetsl.data.saveImage("debug.tif", debug, tags)
    
    click.echo("finished cerberus prediction!")

@cerbs.command("create")
@click.option("-c", "config_file", prompt=True)
def createCerberusCommand(config_file):
    
    cfg = cerberus_config.loadConfig(config_file)
    
    unet_cfg = cfg["unet"]

    if client.configure(unet_cfg, "configure base unet: %s"%config_file, finish="next"):
        pass
    else:
        click.echo("cancelled")
        return
    
    for i, head_cfg in enumerate(cfg["heads"]):
        if client.configure(head_cfg, "configure head #%d"%i):
            pass
        else:
            click.echo("cancelled")
            return
    
    
    #TODO move to module file, so only the config file is used from here.
    cerb_model = cerberus.createCerberusUnet3dModel(
            unet_cfg[unetsl.INPUT_SHAPE], 
            pool_size=unet_cfg[unetsl.POOLING_SHAPE],
            kernel_shape = unet_cfg[unetsl.KERNEL_SHAPE], 
            depth=unet_cfg[unetsl.DEPTH],
            n_filters=unet_cfg[unetsl.N_FILTERS],
            spatial_dropout_rate=unet_cfg[unetsl.SPATIAL_DROPOUT_RATE], 
            head_configs= cfg["heads"]
        )
    
    out = unet_cfg[unetsl.MODEL_FILE]
    unetsl.model.saveModel(cerb_model, out)
    with open(config_file, 'w', encoding="utf8") as conf:
        json.dump(cfg, conf, indent="  ")

@cerbs.command("attach")
@click.option("-c", "config_file")
def attachDataSource(config_file):
    import unetsl.scripts.attach_data_source as attacher
    config = json.load(open(config_file, 'r', encoding="utf8"))
    config = attacher.manageConfig(config)
    ans = input("save config? ( y/N )")
    
    if ans.startswith("y"):
        with open(config_file, 'w', encoding="utf8") as conf:
            json.dump(config, conf, indent="  ")    

if __name__=="__main__":
    cerbs()
