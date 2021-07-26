#!/usr/bin/env python

from .masker import MaskerModel
import unetsl.data
import os, pathlib, sys, numpy
import click
@click.group(chain=True)
def masker():
    pass

@masker.command("create")
@click.argument("steady", default = 5)
def create(steady):
    print(steady)
    mm = MaskerModel( (1, 384, 384, 384, ), steady)
    model_name = "dog-tired-%s.h5"%steady
    mm.createModel()
    mm.saveModel(model_name)
    
@masker.command("train")
@click.argument("steady", default = 5)
@click.option("-r", "--restart", default=False, is_flag=True)
def train(steady, restart):
    model_name = "dog-tired-%s.h5"%steady
    model_path = pathlib.Path(model_name)
    mm = MaskerModel( (1, 384, 384, 384), steady)
    mm.createModel()
    if model_path.exists() and restart:
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
@click.argument("pred_file", default=None, required=False)
def predict(model_file, img_file, pred_file):
    model_file = pathlib.Path(model_file)
    img_file = pathlib.Path(img_file)
    
    mm = MaskerModel( (1, 384, 384, 384) )
    
    mm.loadModel(model_file)
        
    
    img_stack, tags = unetsl.data.loadImage(img_file)
    print(img_stack.shape)
    
    
    
    for i in range(len(img_stack)):
        pred = mm.predictImages(img_stack[i:i+1])
        if pred_file == None:
            out_name = "pred-%s-b%s-%s"%(
                    model_file.name.replace(".h5", ""), 
                    i,
                    img_file.name
                )
        else:
            out_name = pred_file
        unetsl.data.saveImage( out_name, pred, tags)


def lazyLoadImages( image_folder ):
    """ 
        Creates a generator from the image folder such that generating data 
        doesn't require loading all of the images in memory at once.
        
        Args:
            folder where images will be.
    """
    image_list = [  ]

import threading, time, sys

queued = []

def input_thread():
    
    while True:
        i = input("enter value")
        queued.append(i)
    sys.exit(0)

@masker.command("inspect")
@click.argument("model_file")
@click.argument("steady", default = 5)
def inspect(model_file, steady):
    mm = MaskerModel( (1, 384, 384, 384), steady)
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
    print(lbl_stack.shape)
    lbl_stack = lbl_stack.reshape((lbl_stack.shape[0], 3, lbl_stack.shape[1]//3, *lbl_stack.shape[2:]))
    v = unetsl.data.VolumeViewer(0, lbl_stack[0])
    v2 = unetsl.data.VolumeViewer(1, img_stack[0:1])
    for i in range(lbl_stack.shape[0] - 1):
        while True:
            if len(queued)==0:
                unetsl.data.pyplot.pause(0.1)
                continue
            else:
                queued.remove(queued[-1])
                break
        v.setData(lbl_stack[i+1])
        v2.setData(img_stack[i+1:i+2])
        



def main():
    masker()

if __name__=="__main__":
    #t = threading.Thread(target=input_thread)
    #t.start()
    main()
