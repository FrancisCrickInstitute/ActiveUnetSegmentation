#!/usr/bin/env python

from .masker import MaskerModel
import unetsl.data
import os, pathlib, sys, numpy
import click
@click.group(chain=True)
def masker():
    pass

@masker.command("train")
def train():
    model_name = "dog-tired.h5"
    model_path = pathlib.Path(model_name)
    mm = MaskerModel( (1, 384, 384, 384))
    mm.createModel()
    if model_path.exists():
        mm.loadWeights(model_path)
    print(mm.model.summary())
    images = [ unetsl.data.loadImage(os.path.join("images", img))[0] for img in os.listdir("images")]
    labels = [ unetsl.data.loadImage(os.path.join("labels", img))[0] for img in os.listdir("labels")]
    
    img_stack = mm.getInputData(images)
    lbl_stack = mm.getOutputData(labels)        
    mm.trainModel(img_stack, lbl_stack)

@masker.command("predict")
@click.argument("model_file")
@click.argument("img_file")
@click.argument("pred_file", default=None)
def predict(model_file, img_file, pred_file):
    model_file = pathlib.Path(model_file)
    img_file = pathlib.Path(img_file)
    
    mm = MaskerModel( (1, 384, 384, 384) )
    
    mm.loadModel(model_file)
    img_stack, tags = unetsl.data.loadImage(img_file)
    print(img_stack.shape)
    pred = mm.predictImages(img_stack)
    if pred_file != None:
        out_name = "pred-%s-%s"%(
                    model_file.name.replace(".h5", ""),
                    img_file.name )
    else:
        out_name = pred_file

    unetsl.data.saveImage( out_name, pred, tags)
    

@masker.command("inspect")
def inspect(model_file):
    mm = MaskerModel( (1, 384, 384, 384))
    mm.createModel()
    mm.model.summary()
    #if os.path.exists("dog-tired.h5"):
    #    mm.loadWeights("dog-tired.h5")
    images = [ unetsl.data.loadImage(os.path.join("images", img)) for img in os.listdir("images")]
    labels = [ unetsl.data.loadImage(os.path.join("labels", img)) for img in os.listdir("labels")]
    img_stack = numpy.concatenate([row[0] for row in images], axis=0)
    lbl_stack = numpy.concatenate([row[0] for row in labels], axis=0)
    
    unetsl.data.pyplot.ion()
    img_stack = mm.getInputData(img_stack)
    lbl_stack = mm.getOutputData(lbl_stack)
    v = unetsl.data.VolumeViewer(0, lbl_stack[0])
    v2 = unetsl.data.VolumeViewer(1, img_stack[0])
    input("enter to continue...")
    for i in range(lbl_stack.shape[0] - 1):
        v.setData(lbl_stack[i + 1])
        v2.setData(img_stack[i+1])
        input("continue ...")    
   


def main():
    masker()

if __name__=="__main__":
    main()
