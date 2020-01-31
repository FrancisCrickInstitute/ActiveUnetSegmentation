# -*- coding: utf-8 -*-
import keras
import keras.engine as engine
import keras.layers as layers
import keras.optimizers as optimizers

import unetsl.cli_interface
import unetsl.model
import unetsl.data

import random
import sys
import numpy
from keras.models import load_model
from skimage.external.tifffile import TiffWriter
from keras import backend as K

from keras.utils import plot_model

def getDataGenerator(xdata, ydata, n_labels, indexes, patch, batch_size=1):
    """
       
    """
    print(ydata.shape)
    print(len(indexes), batch_size)
    xbatch = []
    ybatch = []
    while True:
        for index in indexes:
            x = xdata[
                        0,
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ]
            y = ydata[
                        0,
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ]
            xbatch.append(x)
            ybatch.append(y)
            if len(xbatch)==batch_size:
                yield numpy.array(xbatch),numpy.array(ybatch)
                xbatch = []
                ybatch = []

def tupleFromShape(shape):
    s = []
    for dim in shape:
        if dim.value:
            s.append(dim.value)
    return tuple(s)

def createModel():
    """
        Create a simple CNN model for tensorflow that will detect missing lines.
    """
    keras.backend.set_image_data_format("channels_first")
    input_shape = (1, 32, 32)
    kernel_shape = (16, 16)
    n_filters = 64
    input_layer = engine.Input(input_shape)
    active_1 = layers.Activation("relu")(input_layer)
    conv_1 = layers.Conv2D(n_filters, kernel_shape)(active_1)
    conv_2 = layers.Conv2D(2*n_filters, kernel_shape)(conv_1)
    pool_1 = layers.MaxPooling2D()(conv_2)
    
    s = tupleFromShape(pool_1.shape)
    p = 1
    for d in s:
        p *= d
    
    shaped_1 = layers.Reshape((p,))(pool_1)
    dense_1 = layers.Dense(1024)(shaped_1)
    shaped_2 = layers.Reshape((1, 32, 32))(dense_1)
    out = layers.Activation("sigmoid")(shaped_2)
    model = engine.Model(input_layer, out)
    plot_model(model, to_file='model.png')
    return model

def getData(input_fname, output_fname):
    inp, _ = sldata.loadImage( input_fname )
    
    op, _ = sldata.loadImage( output_fname )
    op = (op>0)*1
    patch = (1, 1, 32, 32)
    indexes = sldata.indexVolume(inp, patch, (0, 1, 32, 32))
    random.shuffle(indexes)
    ni = int(len(indexes)*0.8)
    nv = len(indexes) - ni
    batch_size = 128
    n_labels = 1
    input_gen = getDataGenerator(inp, op, n_labels, indexes[:ni], patch, batch_size)
    valid_gen = getDataGenerator(inp, op, n_labels, indexes[ni:], patch, batch_size)
    return input_gen, valid_gen

def creation():
    model = createModel()
    model.save("fixit_model.h5")
    
def sorensenDiceCoefLoss(y_exp, y_pred):
    y_exp_f = K.flatten(y_exp)
    y_pred_f = K.flatten(y_pred)
    
    return - 2*(K.sum(y_exp_f*y_pred_f)+ 1)/(1 + K.sum(y_exp_f) + K.sum(y_pred_f))

def training():
    
    model = keras.models.load_model("fixit_model.h5", custom_objects={"sorensenDiceCoefLoss": sorensenDiceCoefLoss})
    model.compile(optimizers.Adam(0.000001), sorensenDiceCoefLoss)
    train_gen, valid_gen = getData(sys.argv[2], sys.argv[3])
    for i in range(1000):
        model.fit_generator(generator=train_gen,
                            steps_per_epoch=10000,
                            epochs=10,
                            validation_data=valid_gen,
                            validation_steps=256,
                            )
    
        model.save("fixit_model-tt.h5")
    
def prediction():
    model = keras.models.load_model("fixit_model-tt.h5", custom_objects={"sorensenDiceCoefLoss": sorensenDiceCoefLoss})
    img, _ = unetsl.data.loadImage( sys.argv[2] )
    indexes = unetsl.data.indexVolume(img, (1, 1, 32, 32), (0, 1, 32, 32))
    out = numpy.zeros(img.shape, dtype="int8")
    
    for index in indexes:
        print(index)
        chunk = img[
                0:,
                index[1]:index[1] + 1,
                index[2]:index[2] + 32,
                index[3]:index[3] + 32
                ]
        pred = model.predict(chunk)
        pred = (pred>0.01)*1
        out[    0:,
            index[1]:index[1] + 1, 
            index[2] : index[2] + 32, 
            index[3] : index[3] + 32 ] |= pred
    
    with TiffWriter("%s"%(sys.argv[3])) as writer:
        writer.save(out)
if __name__=="__main__":
    print("usage: fixit_model c/t/p ")
    print("c - create mode. t - train model. p - predict image")
    
    if len(sys.argv)<2:
        sys.exit(-1)
    c = sys.argv[1]
    
    if c=='c':
        creation()
    elif c=='t':
        training()
    elif c=='p':
        prediction()
    
    
