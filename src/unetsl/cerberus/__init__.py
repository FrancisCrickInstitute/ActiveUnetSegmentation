# -*- coding: utf-8 -*-

from unetsl.cerberus.cerberus_config import getDefaultCerberusPredictionConfig, getTrainingConfig, HeadConstants
import click
import unetsl
import unetsl.model
import re
import numpy
import time

import keras.models
#import tensorflow.keras.models
import keras.layers

BITS="bits"
OFFSET="offset"
LABELLER="labeller"



class DataHead:
    def __init__(self, name, labeller, n_labels, bits, offset, depth, resampler):
        self.bits = bits;
        self.mask = (2**bits - 1)<<offset
        self.name = name
        self.labeller = labeller
        self.n_labels = n_labels
        self.depth = depth
        self.offset = offset
        self.resampler = resampler        
            
    def label(self, batch):
        """
            Recieves batch, which is the full y data [n, 1, z, y , x] where the
            channel will be bit-masked from creating a [n, n_labels, z', y', x'] batch. Returns an 
            an appropriate head, starting from bits -> bits.
            
        """
        relevant = (batch & self.mask) >> self.offset
        labelled_batch = numpy.array([ self.labeller(rn[0], self.n_labels) for rn in relevant])
        
        return self.resampler.toModelOutputSize(labelled_batch)    
    

class ResizeOperation:
    """
        This is an interface for resizing data to and from the size of an input
        image and the corresponding output image.
    """
    def toModelOutputSize(self, chunk):
        """
            For transforming the input segmentations/training data to the 
            correct size for the model
        """
        raise Exception("Operation not implemented: to model output size.")
        
    def toDataInputSize(self, chunk):
        """
            for transforming the output of the model back to an image with 
            the corresponding spatial dimensions of the image using for input data.
        """
        raise Exception("Operation not implemented: to data input size.")
    

class IdentitySizer(ResizeOperation):
    def toDataInputSize(self, chunk):
        return chunk
    def toModelOutputSize(self, chunk):
        return chunk

class Pooler(ResizeOperation):
    def __init__(self, pool_shape, operation):
        self.pool_shape = tuple(pool_shape)
        self.kron_glock = numpy.ones(self.pool_shape);
        self.op = operation
        
    def toModelOutputSize(self, chunk):
        return self.op(chunk, self.pool_shape)
    def toDataInputSize(self, chunk):
        return numpy.kron(chunk, self.kron_block)

class MaxPoolResize(Pooler):
    """
        Max pooler.
    """
    def __init__(self, input_shape, depth, pool):
        pool_shape = tuple( p**depth for p in pool )
        super().__init__(pool_shape, unetsl.data.maxPool)


class MinPoolResize(Pooler):
    """
        Finds the minimum pool, using a mask and stride of size pool. Applied
        depth number of times.
    """
    def __init__(self,input_shape, depth, pool):
        pool_shape = tuple( p**depth for p in pool )
        super().__init__(pool_shape, unetsl.data.minPool)
    

class CropResize:
    """
        Resize operation by cropping them image.
    """
    def __init__(self, input_shape, depth=None, pool=None, offset = None):
        zm = len(pool)
        
        self.view = tuple( i // p**depth for i, p in zip(input_shape[-zm:], pool) )
        self.input_shape = input_shape
        
        if offset is None:
            self.offset = tuple( (i - m)//2 for i, m in zip( input_shape[-zm:], self.view ) ) 
        
    def toDataInputSize(self, chunk):
        """
            Crops the supplied chunk down to the .view dimensions. 
        """
        op = numpy.zeros(self.input_shape, dtype=chunk.dtype)
        slc = self.getFullSlice(op)
        op[slc] = chunk
        return op
                
    def toModelOutputSize(self, chunk):
        """
            For transforming the input segmentations/training data to the 
            correct size for the model
        """
        slc = self.getFullSlice(chunk)
        return chunk[slc]
    
    def getFullSlice(self, arr):
        dims = len(arr.shape)
        pls = len(self.view)
        full = dims - pls
        lows = (0, )*full + self.offset
        span = arr.shape[:full] + self.view 
        highs = tuple(l + s for l, s in zip(lows, span))
        return tuple(slice(l, h) for l, h in zip(lows, highs) )
        
RESAMPLER_MAP = {
        "max pool" : MaxPoolResize,
        "min pool" : MinPoolResize,
        "crop" : CropResize
    }
    
class CerberusDataGenerator:
    def __init__(self, heads):
        self.heads = heads
    def getGenerator(self, basic_generator):
        for batch in basic_generator:
            if len(batch)==2:
                #standard x-y data set.
                #shape: sample, channels, z, y, z
                x = batch[0]
                #shape sample, 1, z, y, x
                y = batch[1]
                
                out = {}
                for head in self.heads:
                    out[head.name] = head.label(y)
                yield x, out
            
            else:
                raise Exception("weighted values, not supported")    
        


def getOutputLayer(unet_model, depth, level):
    """
        depth is the total depth of the model, the level is the attachment
        point/ level of output layer to be connected too.
        
    """
    layers = unet_model.layers
    
    if level==depth:
        #last contracting layer.
        bottom = "contraction-b-%d"%level
        for i in range(len(layers)):
            layer = layers[i]
            if layer.name==bottom:
                if layers[i+1].name=="unetsl-do-cb-%d"%level:
                    #after dropout layer
                    return layers[i+1]
                return layer
        #made it to here without finding output layer.
        raise Exception("invalid level selected %d."%level)
    
    expat = re.compile("expansion-b-(\\d+)")
    for i in range(len(layers)):
        me = expat.match(layers[i].name)
        if me:
            layer_depth = int(me.group(1))
            if layer_depth == level:
                layer = layers[i]
                if len(layers) > (i+1) and layers[i+1].name == "unetsl-do-eb-%d"%layer_depth:
                    return layers[i+1] #spatial dropout is in use.
                return layer
    raise Exception("Could not find a %d level output layer"%level)

def getDataGenerators( data_sources, patch_size, stride, batch_size, validation_fraction, heads):
    """
        Turn the data_sources into generators based on the geometries provided.
    """
    training_volumes = []
    validation_volumes = []
    
    train_fraction = 1-validation_fraction
    
    for source in data_sources:
        #updateGeometry(self, n_labels, patches, stride=None, out_patches=None)
        source.updateGeometry(1, patch_size, stride)
        source.generateIndexes();
        v1, v2 = source.split(train_fraction);
        print("train: ", v1) 
        print("  validation: ", v2)                
        training_volumes.append(v1)
        validation_volumes.append(v2)
    
    cdg = CerberusDataGenerator(heads)
    
    training_generator = unetsl.data.adInfinitum(
            [ 
                (
                        v.steps(batch_size), 
                        cdg.getGenerator(v.getDataGenerator(batch_size)) 
                ) for v in training_volumes 
            ]
        )
            
    validation_generator = unetsl.data.adInfinitum(
            [
                    (
                            v.steps(batch_size), 
                            cdg.getGenerator(v.getDataGenerator(batch_size))
                    ) for v in validation_volumes
            ]
        )
    
    return training_generator, validation_generator

def getCerberusDataGenerator(heads, data_source, patch_size, stride=None, batch_size=1):
    if stride is None:
        stride = [p for p in patch_size]
    data_source.updateGeometry(batch_size, patch_size, stride)
    cdg = CerberusDataGenerator(heads)
    return cdg.getGenerator(data_source.getDataGenerator(batch_size))



def loadHeads(head_configs, input_shape, pool):
    """
        head configs contains the local config for each head, the input shape
        are global configs dependent on the type of model.
        
    """
    results = []
    for hc in head_configs:
        labeller = unetsl.data.getLabeller(hc[unetsl.data.LABELLER])
        bits = hc[HeadConstants.bits]
        n_labels = hc[HeadConstants.n_labels]
        depth = hc["depth"]
        offset = hc[HeadConstants.offset]
        sizer = RESAMPLER_MAP[hc["resampler"]](input_shape, depth, pool)
        h = DataHead(
                hc["name"], 
                labeller,
                n_labels,
                bits,
                offset,
                depth,
                sizer
            )
        
        results.append(h)
        
    return results

def createCerberusUnet3dModel(input_shape, pool_size=(2, 2, 2), depth=3, 
                              n_filters=32,  
                              kernel_shape=(3,3,3), 
                              spatial_dropout_rate=0.0, head_configs={}
                              ):
    """
    Builds the 3D UNet Keras model and attached outputs defined in the head configs.
    
    Args:
        input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
         divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
         pool_size: Pool size for the max pooling operations.
         depth: indicates the depth of the U-shape for the model. An 
        increase in depth corresponds to a pool/contracting level.
    Return: Untrained 3D UNet Model with three outputs.
    """

    starting_model = unetsl.model.createUnet3dModel(
            input_shape, 
            pool_size=pool_size, 
            n_labels=1,
            depth=depth, 
            n_filters=n_filters, 
            activation_name="relu",                 
            kernel_shape=kernel_shape, 
            spatial_dropout_rate = spatial_dropout_rate
        )
    outputs = []
    
    for i, cfg in enumerate(head_configs):
        head_activation = cfg[HeadConstants.activation] 
        ol = getOutputLayer(starting_model, depth, cfg["depth"])
        n_labels = cfg["n_labels"]
        print("cerberus_%s"%cfg["name"])
        layer = keras.layers.Conv3D(n_labels, (1,1,1), name = "cerberus_%s"%cfg["name"] )(ol.output)
        cout_name = "%s"%cfg["name"]
        
        if head_activation=="softmax":
            activation_layer = keras.layers.Softmax(
                        axis=1, 
                        name=cout_name )(layer)
        else:
            activation_layer = keras.layers.Activation(
                                    head_activation, 
                                    name=cout_name
                                )(layer)
        outputs.append(activation_layer)
    return keras.models.Model(starting_model.inputs, outputs)

def getInputShapes(model):
    ls = []
    if isinstance(model.input,list):
        for inp in model.input:
            s = []
            for dim in inp.shape:
                if dim.value:
                    s.append(dim.value)
            ls.append(tuple(s))
        return ls
    else:
        return [unetsl.model.getInputShape(model)]
    
def getOutputShapes(model):
    """
       Gets the shapes of all of the outputs.
     
    """
    if isinstance(model.output, list):
        ls = []
        for inp in model.output:
            print(inp)
            s = []
            for dim in inp.shape:
                if dim.value:
                    s.append(dim.value)
            
            ls.append(tuple(s))
        return ls
    else:
        return [unetsl.model.getOutputShape(model)]
    
def getLossFunctions(loss_function_map):
    fns = {}
    for key in loss_function_map:
        fns[key] = unetsl.model.getLossFunction(loss_function_map[key])
    return fns

    

    