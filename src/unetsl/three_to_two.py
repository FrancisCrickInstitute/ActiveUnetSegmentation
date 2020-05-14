#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
For segmeinting 3D image stacks to 2D images. This would help to skip
a project step for analysing movies.


Created on Fri Apr  3 15:27:27 2020

@author: smithm3
"""
import click
import unetsl
import unetsl.model
import unetsl

@click.group(chain=True)
def three_to_two():
    pass

def createFlattenedModel(config):
    input_shape = config["input shape"]
    model = unetsl.model.createUnet3dModel(
            config[unetsl.INPUT_SHAPE], 
            pool_size=config[unetsl.POOLING_SHAPE],
            n_labels = config[unetsl.N_LABELS],
            kernel_shape = config[unetsl.KERNEL_SHAPE], 
            depth=config[unetsl.DEPTH],
            n_filters=config[unetsl.N_FILTERS],
            activation_name=config[unetsl.ACTIVATION],
            spatial_dropout_rate=config[unetsl.SPATIAL_DROPOUT_RATE]
            
        )
    condenser = config["condenser"]
    n_labels = config["nlabels"]
    final_convolution = keras.layers.Conv3D(n_labels, condenser, padding="valid")(model.layers[-3].output)
    act = keras.layers.Activation(config["activation"])(final_convolution)
    model = keras.models.Model(inputs = model.layers[0].input, outputs=act)
    model.save(config["model file"]);
    
    


@three_to_two.command("create")
@click.option("-c", "--config-file", type=click.Path(), prompt=True)
@click.option("-D", "extended_options", multiple=True)
@click.option("-b", "--batch", is_flag=True)
def create(config_file, extended_options, batch):
    """
        This script will create and save a model and a config file.
    """
    
    
    config_path = pathlib.Path(config_file)
    
    config = unetsl.config.getDefaultConfigurationTool()
    if config_path.exists():
        config.load(config_path)
        backup_name = config_path.name.replace(".json", "-json.bak")
        backup_path = pathlib.Path(config_path.parent, backup_name)
        shutil.copy(str(config_path), str(backup_path))
    if not batch: 
        if not unetsl.cli_interface.configure(config, title="Update config: %s"%str(config_path)):
            print("cancelling...")
            sys.exit(0)
    
        config.save(config_path)
    config.updateWithExtendedOptions(extended_options)    
    model = createFlattenedModel(config)
    
    model.save(config[unetsl.MODEL_FILE]);

if __name__=="__main__":
    main()
