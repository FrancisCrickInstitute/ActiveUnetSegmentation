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
        #values = backend.sum( backend.flatten( binned ), axis=0)
        values = backend.sum(binned, axis=0) #samples
        values = backend.sum(values, axis=1) #z
        values = backend.sum(values, axis=1) #y
        values = backend.sum(values, axis=1) #x
        
        
        return values[channel]
    labelCount.__name__ = "%s%s"%(labelCount.__name__, channel)
    return labelCount


class EpochLog(callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.written = 0
        self.first = True
        self.file = "epoch-log.txt"
        with open(self.file, 'w', encoding="utf8") as lg:
            lg.write("#logging\n")
        
    def on_epoch_end(self, epoch, params):
        with open(self.file, 'a', encoding="utf8") as lg:
            if self.first:
                 lg.write("#epoch\t%s\n"% "\t".join([str(i) for i in params]))
                 self.first=False
            self.written += 1
            lg.write("%s\t%s\n"%( self.written,  "\t".join(
                                [ str(params[i]) for i in params ]
                            ) 
                        )
                    )

class BatchLog(callbacks.Callback):
    def __init__(self, max_writes = 10000):
        super().__init__()
        self.first = True
        self.file = "batch-log.txt"
        self.written = 0
        self.max_writes = max_writes
        with open(self.file, 'w', encoding="utf8") as lg:
            lg.write("#logging\n")
    def on_batch_end(self, batch, logs = None):
        if self.written > self.max_writes:
            return
        with open(self.file, 'a', encoding="utf8") as lg:
            if self.first:
                 lg.write("#batch\t%s\n"% "\t".join([str(i) for i in logs]))
                 self.first=False
            lg.write("%s\t%s\n"%( self.written,  "\t".join(
                                [ str(logs[i]) for i in logs ]
                            ) 
                        )
                    )
        self.written += 1
        
class MaskerModel:
    def __init__(self, input_shape):
        """
            Requires input images to be smaller than input shape
            Args:
                input_shape: (c, z, y, x)
            
        """
        self.input_shape = input_shape
    
    
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
          Takes an image: batch, c, z, y, x and returns it as a padded image
          batch, c, z', y', x' where the new cnets are 
        """
        return self.getPaddedInput(img)
    
        
    def getOutputData(self, labelled_img):
        
        z0 = labelled_img.shape[-3]
        zn = z0%3
        y0 = labelled_img.shape[-2]
        yn = y0%3
        x0 = labelled_img.shape[-1]
        xn = x0%3
        
        labelled_img = unetsl.data.maxPool(self.getPaddedInput(labelled_img, cval=0), (3, 3, 3))
        
        background = 1*(labelled_img==1)
        border = 1*(labelled_img==0)
        foreground = 1 - background - border
        #foreground = (1 - background)
        #background = self.getPaddedInput((background), cval=0)
        #foreground = self.getPaddedInput((foreground), cval=0)
        #border = self.getPaddedInput(numpy.zeros(labelled_img.shape), cval=1)
        
        op = numpy.concatenate([background, foreground, border], axis=1)
        
        
        print(op.shape)
        return op
        
    def createModel(self):
        data_format = "channels_first"
        inp = keras.layers.Input(self.input_shape)
        print(inp)
        c = keras.layers.MaxPooling3D((3, 3, 3), data_format="channels_first")(inp)
        print(c)
        
        conv0 = keras.layers.Conv3D(
                    16, 
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
        
        for i in range(4):
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
                optimizer=keras.optimizers.Adam(1e-5),
                loss = keras.losses.mean_squared_error,
                metrics = ["accuracy", "binary_accuracy", channelSummer(0), channelSummer(1), channelSummer(2)]
                )
        
        steps_per_epoch = images.shape[0]

        def trainGenerator():
            index = 0;
            
            while True:
                yield images[index:index+1], labels[index:index+1]
                index = (index + 1)%steps_per_epoch
        epochlog = EpochLog()
        batchlog = BatchLog()
        
        
        for i in range(100):        
            self.model.fit_generator(generator=trainGenerator(),
                        steps_per_epoch=steps_per_epoch,
                        epochs=100, 
                        verbose=2,
                        callbacks = [epochlog, batchlog]
                        )
            self.model.save("dog-tired.h5", include_optimizer=False)
    def predictImages(self, image, crop=False):
        padded_img = self.getInputData(image)
        #pred = self.getOutputData(image).astype("uint8")
        pred = self.model.predict(padded_img)
        #pred = padded_img
        
        if crop:
            f, c, z, y, x = pred.shape
            f0, c0, z0, y0, x0 = image.shape
            dz = (z - z0)//2
            dy = (y - y0)//2
            dx = (x - x0)//2
            pred = numpy.round(pred[:, :, dz : dz + z0, dy : dy + y0, dx : dx + x0]).astype("uint8")
       
        
        mask = (pred==numpy.max(pred, axis=1))*1
        #for i in range(c):
        #    pred[:, i] = pred[:, i]*factor
        #    factor = factor*2
        #pred = numpy.sum(pred, axis=1, keepdims=True, dtype='float32')
        return pred.astype("float32")
        #return mask.astype("uint8")
        #return numpy.round(pred).astype("uint8")
    def loadWeights(self, model_file):
        """
            Loads weights from an existing model file. If the model file and 
            the loaded file have different shapes, then like named layers
            will be overwritten.
            
        """
        saved_model = keras.models.load_model(model_file)
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
        
    
if __name__=="__main__":
    if "train" in sys.argv:
        mm = MaskerModel( (1, 384, 384, 384))
        mm.createModel()
        if os.path.exists("dog-tired.h5"):
            mm.loadWeights("dog-tired.h5")
        print(mm.model.summary())
        images = [ unetsl.data.loadImage(os.path.join("images", img)) for img in os.listdir("images")]
        labels = [ unetsl.data.loadImage(os.path.join("labels", img)) for img in os.listdir("labels")]
        img_stack = numpy.concatenate([row[0] for row in images], axis=0)
        lbl_stack = numpy.concatenate([row[0] for row in labels], axis=0)
        


        img_stack = mm.getInputData(img_stack)
        lbl_stack = mm.getOutputData(lbl_stack)
        
        mm.trainModel(img_stack, lbl_stack)
    elif "predict" in sys.argv:
        model_file = pathlib.Path(sys.argv[2])
        img_file = pathlib.Path(sys.argv[3])
        
        mm = MaskerModel( (1, 384, 384, 384) )
        mm.createModel()
        mm.loadWeights(str(model_file))
        img_stack, tags = unetsl.data.loadImage(img_file)
        print(img_stack.shape)
        pred = mm.predictImages(img_stack, crop = False)
        unetsl.data.saveImage(
                    "pred-%s-%s"%(
                        model_file.name.replace(".h5", ""), 
                        img_file.name
                        ),
                    pred, 
                    tags
                )
    elif "inspect" in sys.argv:
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