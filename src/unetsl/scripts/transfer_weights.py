#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import unetsl.model
import sys
import click

def patchEndsofUnet(first, second, out="transferred.h5", inverted=False):
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
    shift = 0
    for i, layer in enumerate(first.layers):
        weights = layer.get_weights()
        if len(weights)>0:
            #copy-able weights.
            found = False
            
            while not found and shift < len(second.layers):
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
                if inverted:
                    layer.set_weights(otra_weights)
                else:
                    otra.set_weights(weights)
            else:
                print("skipped layer: ", i, layer.name)
    if inverted:
        first.save(out)
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
@click.option("-i", "--inverted", default=False)
def main(origin, target, output, inverted):
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
    
    patchEndsofUnet(first, second, out=output, inverted=inverted)

if __name__=="__main__":
    main()
