#!/usr/bin/env python3
"""
 BASED ON CODE FOUND HERE https://github.com/ellisdg/3DUnetCNN/tree/master/unet3d
 
"""
import sys
import pathlib
import shutil

import unetsl.config
import unetsl.cli_interface
import unetsl.model
import unetsl.data
import click

@click.command()
@click.option("-c", "--config-file", type=click.Path(), prompt=True)
@click.option("-D", "extended_options", multiple=True)
@click.option("-b", "--batch", is_flag=True)
def main(config_file, extended_options, batch):
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
    model.save(config[unetsl.MODEL_FILE]);

if __name__=="__main__":
    main()
