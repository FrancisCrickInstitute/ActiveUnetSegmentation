#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
    For creating  simple keras model that takes in a "large" image and 
    produces a mask from that image.
    
    Saves the result as the original image + mask
    .
    Input -> (t, c, z, y, x)
    Output -> (t, c + 1, z, y, x) with 3 labels mask->not-mask->not-image
    
    
    
"""

import tensorflow.keras as keras
import tensorflow.keras.callbacks as callbacks
import numpy
import tensorflow.keras.backend as backend
import os
import os.path

import unetsl.data

import sys

import pathlib



def channelSummer( channel):

    def labelCount(y_true, y_pred):
        binned = backend.round(y_pred)
        values = backend.sum( backend.flatten( binned[ :, channel ] ), axis=0)
        totals = backend.sum( backend.flatten( y_true[ :, channel ] ), axis=0)
        
        #values = backend.sum(binned, axis=0) #samples
        #values = backend.sum(values, axis=1) #z
        #values = backend.sum(values, axis=1) #y
        #values = backend.sum(values, axis=1) #x
        
        
        return values/totals
    labelCount.__name__ = "%s%s"%(labelCount.__name__, channel)
    return labelCount


class EpochLog(callbacks.Callback):
    def __init__(self, tag):
        super().__init__()
        self.written = 0
        self.first = True
        self.file = "epoch-log-%s.txt"%tag
        
        with open(self.file, 'w', encoding="utf8") as lg:
            lg.write("#logging\n")
        
    def on_epoch_end(self, epoch, params):
        with open(self.file, 'a', encoding="utf8") as lg:
            if self.first:
                 lg.write("#epoch\t%s\n"% "\t".join(["%d.%s"%(i+2, l) for i, l in enumerate(params)]) )
                 self.first=False
            self.written += 1
            lg.write("%s\t%s\n"%( self.written,  "\t".join(
                                [ str(params[i]) for i in params ]
                            ) 
                        )
                    )

class BatchLog(callbacks.Callback):
    def __init__(self, tag, max_writes = 10000):
        super().__init__()
        self.first = True
        self.file = "batch-log-%s.txt"%tag
        self.written = 0
        self.max_writes = max_writes
        with open(self.file, 'w', encoding="utf8") as lg:
            lg.write("#logging\n")
    def on_batch_end(self, batch, logs = None):
        if self.written > self.max_writes:
            return
        with open(self.file, 'a', encoding="utf8") as lg:
            if self.first:
                 lg.write("#batch\t%s\n"% "\t".join(["%d.%s"%(i+2, l) for i, l in enumerate(logs)]))
                 self.first=False
            lg.write("%s\t%s\n"%( self.written,  "\t".join(
                                [ str(logs[i]) for i in logs ]
                            )
                        )
                    )
        self.written += 1
        
class MaskerModel:
    def __init__(self, input_shape, steady = None):
        """
            Requires input images to be smaller than input shape
            Args:
                input_shape: (c, z, y, x)
            
        """
        self.input_shape = input_shape
        self.steady = steady
    
    def getPaddedInput(self, img, cval=0):
        
        
        
        lz = ( self.input_shape[-3] - img.shape[-3] )//2
        ly = ( self.input_shape[-2] - img.shape[-2] )//2
        lx = ( self.input_shape[-1] - img.shape[-1] )//2
        #+1px if size is odd.
        rz = self.input_shape[-3] - lz*2  - img.shape[-3]
        ry = self.input_shape[-2] - ly*2 - img.shape[-2]
        rx = self.input_shape[-1] - lx*2 - img.shape[-1]
        pads = [(0,0) for i in img.shape[:-3]]
        
        pads.append((lz, lz+rz))
        pads.append((ly, ly+ry))
        pads.append((lx, lx+rx))
        
        
        return numpy.pad(img, pads, constant_values=cval)
        
    def getInputData(self, img):
        """
          Takes a list of images: [(n, c, z, y, x), ...] and returns it as a padded image
          n*len(images), c, z', y', x' where the new cnets are 
          
          Args:
              img : expects a list so that multiple sizes of images can be 
                    processed.
        """
        
        
        total = []
        for stack in img:
            total.append(self.getPaddedInput(stack))
        
        return numpy.concatenate(total, axis=0)
    
        
    def getOutputData(self, labelled_images):
        
        total = []        
        for labelled_img in labelled_images:
            labelled_img = unetsl.data.maxPool(self.getPaddedInput(labelled_img, cval=0), (3, 3, 3))
        
            background = 1*(labelled_img==1)
            border = 1*(labelled_img==0)
            foreground = 1 - background - border
        
            total.append(numpy.concatenate([background, foreground, border], axis=1))
        return numpy.concatenate(total, axis=0)
        
    def createModel(self):
        data_format = "channels_first"
        inp = keras.layers.Input(self.input_shape)
        print(inp)
        c = keras.layers.MaxPooling3D((3, 3, 3), data_format="channels_first")(inp)
        print(c)
        
        conv0 = keras.layers.Conv3D(
                    4, 
                    (8, 8, 8), 
                    padding='same', 
                    strides=(4, 4, 4), 
                    activation="relu",
                    data_format=data_format,
                    name = "contraction-0" ) 
        c = conv0(c)
        print(c)
        conv1 = keras.layers.Conv3D(
                    64, 
                    (8, 8, 8),
                    padding='same', 
                    strides=(2, 2, 2), 
                    activation="relu",
                    data_format=data_format, 
                    name = "contraction-1" )
        c = conv1(c)
        print(c)
        
        for i in range(self.steady):
            steady = keras.layers.Conv3D(
                        256, 
                        ( 3, 3, 3),
                        padding='same', 
                        strides=(1, 1, 1), 
                        activation="relu",
                        data_format=data_format, 
                        name = "steady-%s"%i )
            c = steady(c)
            drp = keras.layers.SpatialDropout3D(rate = 0.1, data_format=data_format);
            c = drp(c)
            print(c)
        
        
        tconv0 = keras.layers.Conv3DTranspose(
                filters=64,
                kernel_size=(4, 4, 4), 
                strides=(2, 2, 2), 
                data_format = data_format,
                activation = "relu",
                name = "expansion-0",
                padding = "same"
                )
        
        c = tconv0(c)
        print(c)
        
        tconv1 = keras.layers.Conv3DTranspose(
                filters = 8,
                kernel_size = (16, 16, 16), 
                strides = (4, 4, 4), 
                data_format = data_format,
                activation = "relu",
                name = "expansion-1",
                padding = "same"
                )
        
        c = tconv1(c)
        print(c)
        
        opl = keras.layers.Conv3D(
                    3, 
                    (1, 1, 1),
                    padding='same', 
                    strides=(1,1,1),
                    data_format=data_format, 
                    name = "final" )
        c = opl(c)
        
        activation = keras.layers.Softmax(axis=1, name="output")
        #activation = keras.layers.ReLU(name = "output")
        #activation = keras.layers.Activation("sigmoid", name="output")

        c = activation(c)

        print(c)
        self.model = keras.models.Model(inputs=[inp], outputs=[c])
        print(self.model.summary())
    
    def trainModel(self, images, labels):
        print(self.model.output)
        
              
        self.model.compile(
                optimizer=keras.optimizers.Adam(1e-6),
                loss = keras.losses.mean_squared_error,
                metrics = ["accuracy", "binary_accuracy", channelSummer(0), channelSummer(1), channelSummer(2)]
                )
        
        steps_per_epoch = images.shape[0]

        def trainGenerator():
            index = 0;
            
            while True:
                yield images[index:index+1], labels[index:index+1]
                index = (index + 1)%steps_per_epoch
        epochlog = EpochLog(self.steady)
        batchlog = BatchLog(self.steady)
        
        
        for i in range(1):        
            self.model.fit_generator(generator=trainGenerator(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=250, 
                        verbose=2,
                        callbacks = [epochlog, batchlog]
                        )
            self.model.save("dog-tired-%s.h5"%self.steady, include_optimizer=False)
    def predictImages(self, image, restore=True):
        padded_img = self.getInputData([image])
        #pred = self.getOutputData(image).astype("uint8")
        pred = self.model.predict(padded_img)
        if restore:
            pred = numpy.kron(pred, numpy.ones((3, 3,3))) 
            f, c, z, y, x = pred.shape
            f0, c0, z0, y0, x0 = image.shape
            print("o: ", image.shape, " p: ", pred.shape)
            dz = (z - z0)
            dy = (y - y0)
            dx = (x - x0)
            print(dz, dy, dx, "deltas")
            print(z0, y0, x0, "origins")
            pred = pred[:, :, 
                    dz//2 : dz//2 + z0, 
                    dy//2 : dy//2 + y0, 
                    dx//2 : dx//2 + x0
                ]
            pred = numpy.round(pred).astype("uint8")
            print(pred.shape, image.shape, "repeated")
            mask = 1 - pred[:, 1:2]
            pred = numpy.concatenate([mask, image],  axis=1)
        return pred.astype("float32")
        
    def loadWeights(self, model_file):
        """
            Loads weights from an existing model file. If the model file and 
            the loaded file have different shapes, then like named layers
            will be overwritten.
            
        """
        saved_model = keras.models.load_model(model_file)
        if self.steady==None:
            #implies that the model is 
            self.model = model
            
        weight_dict = { l.name : l.get_weights() for l in saved_model.layers}
        
        current = self.model.get_weights()
        
        for l in self.model.layers:
            w = l.get_weights()
            ow = weight_dict.get(l.name, None)
            if ow:
                p = len(w)
                cpd = 0
                for wi, owi in zip(w, ow):
                    if wi.shape == owi.shape:
                        wi[:] = owi[:]
                        cpd += 1
                print(cpd, " / ", p)
            l.set_weights(w)
        #sys.exit(0)
        #self.model.set_weights(current)
    def loadModel(self, model_file):
        self.model = keras.models.load_model(model_file)
    
