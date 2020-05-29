#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:31:51 2020

@author: smithm3
"""
import click
import unetsl.model
from tensorflow import keras



@click.command()
@click.argument("model_file")
@click.option("--layer", default=None)
def main(model_file, layer):
    model = unetsl.model.loadModel(model_file)
    layer_names = [layer.name for layer in model.layers]
    
    if layer is None:
        from unetsl.cli_interface import getLayersPrompt
        layer = getLayersPrompt(layer_names)
    
    selected_layer = None
    for model_layer in model.layers
        if layer == model_layer.name:
            selected_layer = model_layer
            break
    if selected_layer is None:
        print("could not find selected layer: ", layer)
        return -1
    

if __name__ == "__main__":
    main()