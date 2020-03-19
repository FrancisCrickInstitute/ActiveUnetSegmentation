#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unetsl.model
import sys
import click

import collections

def patchEndsofUnet(first, second, out="transferred.h5"):
    """
        Transfers the weights of the first model to corresponding layers in the
        second model. (If inverted the first model is the target.) This assumes
        that the first model has a lower depth than the second model.
        
        corresponding layers are found by keeping track of a shift. The first
        layer with weights, found and the next layer with the same shape is found
        in the second model shifted past the previously found layer.

	Inverted causes the second model's weights to be saved to the first model
        this assumes that the second model is a deeper network.

    """
    copied = []
    
    #removes all the '0's.
    named_list_first = collections.OrderedDict([ (l.name, l) for l in first.layers if len(l.get_weights())>0])
    named_list_second = collections.OrderedDict([ (l.name, l) for l in second.layers if len(l.get_weights())>0])
    
    first_names = [n for n in named_list_first.keys()]
    second_names_set = set(n for n in named_list_second.keys())
    
    for name in first_names:
        if name in second_names_set:
            fl = named_list_first[name]
            sl = named_list_second[name]
            fw = fl.get_weights()
            sw = sl.get_weights()
            if len( fw ) == len( sw ):
                found = True
                for j, w in enumerate( fw ):
                        if w.shape != sw[j].shape:
                            found = False
                            break
                if found:
                    sl.set_weights(fw)
                    copied.append("named copy layer: %s."%(name, ))
                    named_list_first.pop(name)
                    named_list_second.pop(name)
    first_layers = [ l for n, l in named_list_first.items() ]
    second_layers = [ l for n, l in named_list_second.items() ]
    shift = 0
    for i, layer in enumerate(first_layers):
        weights = layer.get_weights()
        if len(weights)>0:
            #copy-able weights.
            found = False
            
            while not found and shift < len(second_layers):
                otra = second.layers[shift]
                otra_weights = otra.get_weights()
                print(i, layer.name, shift, otra.name, ":: comparing")
                
                if len(otra_weights)==len(weights):
                    
                    found = True
                    #copy-able
                    for j, w in enumerate(weights):
                        if w.shape != otra_weights[j].shape:
                            found = False
                            shift += 1
                            break #only fail once
                    
                else:
                    shift += 1
            if found:
                copied.append("copying layer %d : %s to layer %d : %s."%(i, layer.name, shift, otra.name))
                otra.set_weights(weights)
            else:
                print("skipped layer: ", i, layer.name)
    else:
        second.save(out)    
    for cpd in copied:
        print(cpd)
        
def insertBottomOfUnet(first, second):
    """
      provided the first network has been trained on smaller data, it
      can be inserted into at the lower depth.
      
      Essentially building up longer range labelling.
      @param first trained network to be inserted into the second network.
      @param network that the new values will be inserted into.
    """
    pass

def showTrainableLayers(layers):
    for i, lay in enumerate(layers):
        if len(lay.get_weights())>0:
            print("%d %s (%s)"%(i, lay.name, ",".join(str(i.shape) for i in lay.get_weights())))
        else:
            print("\t**%d %s"%(i, lay.name))
            

@click.command()
@click.argument("origin", type=click.Path(exists=True))
@click.argument("target", type=click.Path(exists=True))
@click.argument("output", type=click.Path(), default="transferred.h5")
def main(origin, target, output):
    """
        origin: file that the weights will be taken from.
        
        target: file that weights will be copied too.
        
        output: destination to save file with updated weights. If not
        specified transferred.h5 will be used.
    """
    
    first = unetsl.model.loadModel(str(origin))
    print("load %s with %d layers"%(origin, len(first.layers)));
    second = unetsl.model.loadModel(str(target))
    
    showTrainableLayers(first.layers)
    showTrainableLayers(second.layers)
            
    print(len(first.layers), len(second.layers))
    
    patchEndsofUnet(first, second, out=output)

if __name__=="__main__":
    main()
