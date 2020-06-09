# -*- coding: utf-8 -*-

import tensorflow.keras as keras

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

import unetsl.data
import unetsl.predict




from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, Softmax, SpatialDropout3D, Input

from tensorflow.keras.optimizers import Adam, SGD

from tensorflow.keras.layers import concatenate

import tensorflow.keras.losses

import pathlib

import tensorflow
import tensorflow.keras

"""
Model Functions
"""


def weightedError(y_exp, y_pred):
    """
        Creates a weight per pixel based on the expected value. The weight 
        is an arbitrary chosen value that helped produce oversegmentation of
        membrane structures.
    """
    y_exp_f = K.flatten(y_exp)
    y_pred_f = K.flatten(y_pred)
    
    f = 25
    weight = (y_exp_f*f + 1)
    z = (y_pred_f - y_exp_f)*weight
    z = z*z    
    return tensorflow.reduce_sum(z)/(tensorflow.reduce_sum(weight*weight))

def sorensenDiceCoefLoss(y_exp, y_pred):
    """
        -2*(T*P + 1)/(1 + |T| + |P|)
        The 1 is a smoothing term.
    """
    y_exp_f = K.flatten(y_exp)
    y_pred_f = K.flatten(y_pred)
    return - 2*(K.sum(y_exp_f*y_pred_f)+ 1)/(1 + K.sum(y_exp_f) + K.sum(y_pred_f))


def jaccardIndexLoss(y_exp, y_pred):
    """
        -|T&P|/(|T| + |P| - |T&P|)
        Intersection divided by the union.
    """
    y_exp_f = K.abs(K.flatten(y_exp))
    y_pred_f = K.abs(K.flatten(y_pred))
    intersection = K.sum(y_exp_f*y_pred_f)
    return -(intersection)/(K.sum(y_exp_f) + K.sum(y_pred_f) - intersection)

def createWeightsC(y_c):
    """
        Creates the weight per channel. 
        
        Weighted diced loss, weighting function: returns 1/|y_c|**2 or 0.
    """
    w = K.sum(K.flatten(y_c))
    return tensorflow.cond(tensorflow.math.greater(w, 0), lambda: 1/tensorflow.square(w), lambda: 0.0)

def createWeightsN(y_n):
    """
        Finds the weight per channel for each y_n in a batch.
    """
    return tensorflow.map_fn(createWeightsC, y_n)
    
def numeratorC(weight, y_exp, y_pred):
    """
        finds the sums Ti*Pi*wi over the channels.
    """
    return weight*K.sum(K.flatten(y_exp)*K.flatten(y_pred))

def numerator(weights, y_exp, y_pred):
    """
        Finds the numerator per tensor in batch.
    """
    return tensorflow.map_fn(lambda x: numeratorC(*x), ( weights, y_exp, y_pred ), dtype=tensorflow.float32 )


def denominatorC(weight, y_exp, y_pred):
    """
        Sums wi*(|Ti| + |Pi|) over elements in the channel.
    """
    return weight*(K.sum(K.flatten(y_exp)) + K.sum(K.flatten(y_pred)))
    
def denominator(weights, y_exp, y_pred):
    """
        Finds denominator per tensor in batch.
    """
    return tensorflow.map_fn(lambda x: denominatorC(*x),(weights, y_exp, y_pred), dtype=tensorflow.float32)

def weightedDiceLoss(y_exp, y_pred, sigma = 0.01):
    """
        Calculates the weighted dice loss using Generalized Dice Loss (GDL) from
        https://arxiv.org/pdf/1707.03237.pdf
        
        Wij = 1/|T|**2
        
    """
    weights = tensorflow.map_fn(createWeightsN, y_exp)
    num = tensorflow.map_fn(lambda x: numerator(*x), (weights, y_exp, y_pred), dtype=tensorflow.float32)
    den = tensorflow.map_fn(lambda x: denominator(*x), (weights, y_exp, y_pred), dtype=tensorflow.float32)
    
    return -2*((K.sum(num) + sigma)/(K.sum(den) + sigma))
    
def logMse(y_exp, p_pred):
    return tensorflow.math.log( 1 + keras.losses.mse(y_exp, p_pred) )

def categoricalCrossEntropy(y_exp, y_pred):
    epsilon = 1e-6
    
    y_pred /= tensorflow.reduce_sum(y_pred, 1, True)
    y_pred = tensorflow.clip_by_value(y_pred, epsilon, 1 - epsilon)    
    return - tensorflow.reduce_sum(y_exp * tensorflow.log(y_pred), 1)
    
    
    

LOSS_FUNCTIONS = {
    "unetsl.model.sorensenDiceCoefLoss": sorensenDiceCoefLoss,
    "unetsl.model.weightedError": weightedError,
    "keras.losses.binary_crossentropy": keras.losses.binary_crossentropy,
    "keras.losses.mean_squared_error" : keras.losses.mean_squared_error,
    "keras.losses.categorical_crossentropy" : keras.losses.categorical_crossentropy,
    "unetsl.model.jaccardIndexLoss": jaccardIndexLoss, 
    "keras.losses.hinge": keras.losses.hinge,
    "unetsl.model.categoricalCrossEntropy" : categoricalCrossEntropy,
    "unetsl.model.weightedDiceLoss" : weightedDiceLoss,
    "unetsl.model.logMse" : logMse
    }


def getLossFunction(qualified_function_name):
    """
        Return:
            Loss function from the variable LOSS_FUNCTIONS or returns the name
            such that Keras can decide.
        
    """
    return LOSS_FUNCTIONS.get(qualified_function_name, qualified_function_name)

def getOptimizer(optimizer_name, learning_rate):
    """
        Either Adam or SGD optimizer with the supplied learning rate.
    """
    optimizers ={
            "keras.optimizers.Adam": Adam,
            "keras.optimizers.SGD": SGD
            }
    return optimizers[optimizer_name](learning_rate)


    
def saveModel(model, filepath):
    """
        saves the provided model to the path provided. Converts to string first
        because keras decided to stop supporting pathlib.Path
        
    """
    model.save(str(filepath))


def loadModel(model_file):
    """
        Loads a model from the supplied model, includes custom objects that map
        to local functinos. Uses str( model_file ) so a pathlib.Path will work
        here, even though keras broke that functionality.
        
    """
    custom_objects = {
            "weightedError": weightedError, 
            "sorensenDiceCoefLoss": sorensenDiceCoefLoss, 
            "dice_coefficient":weightedError,
            "dice_coefficient_loss":sorensenDiceCoefLoss, 
            "jaccardIndexLoss": jaccardIndexLoss,
            "categoricalCrossEntropy": categoricalCrossEntropy,
            "weightedDiceLoss": weightedDiceLoss
        }
    
    return load_model(str(model_file), custom_objects=custom_objects, compile=False)
    

def getInputShape(model):
    """
        Gets the shape when there is a single input.
        Return:
            Numeric dimensions, omits dimensions that have no value. eg batch 
            size.
    """
    s = []
    for dim in model.input.shape:
        if dim.value:
            s.append(dim.value)
        
    return tuple(s)

def getOutputShape(model, index = -1):
    """
        Gets the shape of a single output. For compatibility returns the shape
        of the last output if there are multiple outputs.
        Return:
            Numeric dimensions, omits dimensions that have no value. eg batch 
            size.
    """
    s = []
    if isinstance(model.output, list):
        shape = model.output[index].shape
    else:
        shape = model.output.shape
        
    for dim in shape:
        if dim.value:
            s.append(dim.value)
    return tuple(s)

def getOutputMap(model):
    """
        Gets the outputs as a map to the names.
        Return:
            Dictionary of ol.name: output pairs.
    """
    om = {}
    if isinstance(model.output, list):
        for ol in model.output:
            om[ol.name] = ol
    else:
        om[model.output.name] = model.output
    return om

def createUnet3dModel(
        input_shape, pool_size=(2, 2, 2), n_labels=2, depth=4, 
        n_filters=32, activation_name="relu", kernel_shape=(3,3,3), 
        spatial_dropout_rate=0.0
    ):
    """
    Builds the 3D UNet Keras model.
    Args:
        input_shape: shape that the network will train/predict. This does not
            affect the shape of the weights, but it will need to be
            consistent with with the convolution kernel & pooling.
        pool_size: The contraction size at each depth step.
        n_labels: Number of filters used for the output layer.
        depth: The number of max pooling layers - 1. Must be greater than 2. At
            each depth step the number of filters will double.
        n_filters: Number of filters for the first convolution layer,
            doubles at each subsequent convolution layer.
        activation_name: final layer activateion. "sigmoid" for binary labels,
            "softmax" for a softmax along the filter axis, and relu for linear
            output.
        kernel_shape: shape of kerneles used for convolution.
        spatial_dropout_rate: spatial drop outate used on the contracting 
            branch of the convolution layers.
    Returns:
        untrained 3d unet.
    """
    K.set_image_data_format("channels_first")
    inputs = [ Input(input_shape) ]
    current_layer = inputs[0]
    levels = list()
    # add levels with max pooling
    for layer_depth in range(depth):
        
        layer1 = Conv3D(
                    n_filters*(2**layer_depth), 
                    kernel_shape, 
                    padding='same', 
                    strides=(1, 1, 1), 
                    activation="relu", 
                    name = "contraction-a-%d"%layer_depth
                )(current_layer)
        
        if(spatial_dropout_rate>0):
            layer1 = SpatialDropout3D(
                        rate = spatial_dropout_rate, 
                        name = "unetsl-do-ca-%d"%layer_depth
                    )(layer1)
            
        layer2 = Conv3D(
                    n_filters*(2**layer_depth)*2, 
                    kernel_shape, 
                    padding='same', 
                    strides=(1, 1, 1), 
                    activation="relu", 
                    name = "contraction-b-%d"%layer_depth
                )(layer1)
        
        if(spatial_dropout_rate>0):
            layer2 = SpatialDropout3D(
                        rate = spatial_dropout_rate, 
                        name = "unetsl-do-cb-%d"%layer_depth
                    )(layer2)

        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])
        
    
    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = UpSampling3D(size=pool_size)(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        
        
        
        filters = levels[layer_depth][1]._keras_shape[1]
        
        current_layer = Conv3D(
                    filters , 
                    kernel_shape, 
                    padding='same', 
                    strides=(1, 1, 1), 
                    activation="relu", 
                    name = "expansion-a-%d"%layer_depth
                )(concat)
        
        #if(spatial_dropout_rate>0):
        #    current_layer = SpatialDropout3D(
        #                            rate = spatial_dropout_rate,
        #                            name="unetsl-do-ea-%d"%layer_depth
        #                        )(current_layer)
            
        current_layer = Conv3D(
                    filters , 
                    kernel_shape, 
                    padding='same', 
                    strides=(1, 1, 1), 
                    activation="relu", 
                    name = "expansion-b-%d"%layer_depth
                )(current_layer)
        
        #if(spatial_dropout_rate>0):
        #    current_layer = SpatialDropout3D(
        #                        rate = spatial_dropout_rate,
        #                        name="unetsl-do-eb-%d"%layer_depth    
        #                    )(current_layer)
        
    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    
    if activation_name=="softmax":
        #make sure the softmax is along the correct axis.
        act = Softmax(axis=1)(final_convolution)
    else:
        act = Activation(activation_name)(final_convolution)
    
    model = Model(inputs=inputs, outputs=act)
    
    return model


def recompileModel(
        model, optimizer, loss_function=keras.losses.binary_crossentropy, 
        metrics = ['binary_accuracy', 'accuracy']
    ):
    """
    Prepares a model for training. Delegates to keras.engine.Model.compile.
    Args:
        model: that will be compiled.
        optimizer: argument of same name.
        loss_function: argument named loss.
        metrics: argument of same name 
        
    """
    model.compile(optimizer=optimizer, loss=loss_function, metrics=metrics)


class CustomLogger:
    """
        For learning what the methods should do.
    """
    def __init__(self, *args, **kwargs):
        self.file = "explore-log.txt"
        pass
    def on_batch_begin(self, *args, **kwargs):
        self.log("on batch begin:\n\t%s \n\t%s \n"%(args, kwargs))
    def on_batch_end(self, *args, **kwargs):
        self.log("on batch end: \n\t%s \n\t%s \n"%(args, kwargs))
    def on_epoch_begin(self, *args, **kwargs):
        self.log("on epoch begin: \n\t%s \n\t%s \n"%(args, kwargs))
    def on_epoch_end(self, *args, **kwargs):
        self.log("on epoch end: \n\t%s \n\t%s \n"%(args, kwargs))
    def on_train_begin(self, *args, **kwargs):
        self.log("on train begin: \n\t %s \n\t %s \n"%(args, kwargs))
    def on_train_end(self, *args, **kwargs):
        self.log("on train end: \n\t %s \n\t %s \n"%(args, kwargs))
    def set_model(self, *args, **kwargs):
        self.log("model set: \n\t %s \n\t %s \n"%(args, kwargs))
    def set_params(self, *args, **kwargs):
        self.log("set params: \n\t %s \n\t %s \n"%(args, kwargs))
    def log(self, output):
        with open(self.file, 'a') as out:
            out.write(output)

class DefaultLogger:
    """
        Dummy logger, for extending. TODO replace with keras.
    """
    def __init__(self, *args, **kwargs):
        pass
    def on_batch_begin(self, *args, **kwargs):
        """
            This method has been removed from keras/tensorflow at some point 
            so it now just delegates to 'on_train_batch_begin'.
        """
        self.on_train_batch_begin(*args, **kwargs)
    def on_batch_end(self, *args, **kwargs):
        """
            This method has been removed from keras/tensorflow at some point 
            so it now just delegates to 'on_train_batch_begin'.
        """
        self.on_train_batch_end(*args, **kwargs)
    def on_epoch_begin(self, *args, **kwargs):
        pass
    def on_epoch_end(self, *args, **kwargs):
        pass
    def on_train_begin(self, *args, **kwargs):
        pass
    def on_train_end(self, *args, **kwargs):
        pass
    def on_train_batch_begin(self, *args, **kwargs):
        pass
    def on_train_batch_end(self, *args, **kwargs):
        pass
    def set_model(self, *args, **kwargs):
        pass
    def set_params(self, *args, **kwargs):
        pass
    def on_test_batch_begin(self, *args, **kwargs):
        pass
    def on_test_batch_end(self, *args, **kwargs):
        pass
    def on_test_begin(self, *args, **kwargs):
        pass
    def on_test_end(self, *args, **kwargs):
        pass
    

class LightLog(DefaultLogger):
    """
        Not very light. Call back used for saving the latest version of the 
        model, and the 'best' version.
    """
    def __init__(self, model_file, model, filename=None, best_index=-1):
        """
        Args:
            model_file: base name of the model file, the .h5 will be replaced
                with "-best.h5" and "-latest.h5", whcih the best and latest 
                models will be written to.
            filename: Name of file that training logs will be written to.
            best_index: index of metric that is used to compare the 'best model.'
        """
        self.loss = 1e6
        self.best_file = str(model_file).replace(".h5", "-best.h5")
        self.latest = str(model_file).replace(".h5", "-latest.h5")
        
        self.model = model
        print("saving last epoch as %s"%self.latest)
        self.header=None
        if filename is None:    
            pwd = pathlib.Path("training-log.txt")
            i = 0
            while pwd.exists():
                i += 1
                pwd = pathlib.Path("training-log-%d.txt"%i)
            
            self.file = pwd
            self.log("#training model: %s\n"%(str(model_file)))
        else:
            self.file=pathlib.Path(filename)
        self.log("#training model: %s\n"%(str(model_file)))
        self.keys = None
        self.best_index = best_index
    def on_epoch_end(self, epoch_no, parameters ):
        """
            On first call
        """
        #keys = ['binary_accuracy', 'val_binary_accuracy', 'loss', 'val_loss', 'val_acc', 'acc' ]
        if self.keys is None:
            self.keys = [key for key in parameters]
            self.keys.sort()
            
            if self.best_index >= len(self.keys):
                self.best_index = -1
            
            header = "#epoch\t%s\n"%"\t".join(self.keys)
            self.log(header)
            
        if self.best_index==-1:
            v = parameters["loss"]
        else:
            v = parameters.get(self.keys[self.best_index], -1)
        
        if v<self.loss:
            self.loss = v
            saveModel(self.model, self.best_file)
            
        values = "\t".join(str(parameters.get(s,-1)) for s in self.keys)
        self.log("%d\t%s\n"%(epoch_no, values))
        saveModel(self.model, pathlib.Path(self.latest))
    def set_model(self, model):
        #self.model = model
        for weight in model.weights:
            self.log("#%s\n"%weight)
        self.log("#input: %s\n"%model.input)
        self.log("#output: %s\n"%model.output)
        self.log("#optimiser: %s\n"%model.optimizer)
    def log(self, output):
        with self.file.open(mode="a") as out:
            out.write(output)

        
class BatchLog(DefaultLogger):
    """
        Class for monitoring 
    """
    monitor_steps = 5000
    rotation_steps = 1e9
    
    def __init__(self, modelfile, model, filename="batch-log.txt"):
        self.model = model
        self.batch_no = 0
        self.model_file_name = modelfile.replace(".h5", "-batch.h5")
        self.log_step=0
        self.header=None
        self.filename = filename
        self.file = pathlib.Path(filename)
        self.overwrite_batches=True
        self.stalling = 0
    
    def on_train_batch_end(self, batch_no, parameters):
        if self.batch_no>BatchLog.monitor_steps:
            return;
        if self.header==None:
            self.keys=[key for key in parameters]
            self.keys.sort()
            self.header = "#%s\t%s\n"%("batch_number", "\t".join("%d.%s"%(i+2, key) for i, key in enumerate(self.keys)))
            self.log(self.header)
        
        values = "\t".join(str(parameters.get(s,-1)) for s in self.keys)
        self.log("%d\t%s\n"%(self.batch_no, values))
        
        self.batch_no += 1
        
        if self.batch_no%BatchLog.monitor_steps==0:
            if not self.overwrite_batches:
                saveModel(self.model, self.model_file_name.replace("-batch", "-batch%s"%self.batch_no))
            else:
                saveModel(self.model, self.model_file_name)
        self.rotateLog()
    def set_model(self, model):
        #self.model = model
        for weight in model.weights:
            self.log("#%s\n"%weight)
        self.log("#input: %s\n"%model.input)
        self.log("#output: %s\n"%model.output)
        self.log("#optimiser: %s\n"%model.optimizer)

    def log(self, output):
        with self.file.open(mode="a") as out:
            out.write(output)
    def rotateLog(self):
        self.log_step += 1
        if(self.log_step%BatchLog.rotation_steps==0):
            self.file.rename(self.filename.replace(".txt", "_%05d.txt"%self.log_step))
            self.file = pathlib.Path(self.filename)

class EpochPredictions(DefaultLogger):
    def __init__(self, samples, base_name, model, prediction_folder = "predictions", sample_normalize = False, reduction_type=1, n_gpus=1):
        self.samples = samples
        self.base_name = base_name
        self.prediction_folder = pathlib.Path(prediction_folder)
        self.reduction_type = reduction_type
        self.sample_normalize = sample_normalize
        self.model = model
        self.gpus = n_gpus
        if not self.prediction_folder.exists():
            self.prediction_folder.mkdir()
           
        
    def set_model(self, model):
        self.model = model
        self.on_epoch_end(-1, {})
    def on_epoch_end(self, epoch_no, parameters ):
        prefix = "pred-" + self.base_name + "-e%d"%(epoch_no + 1)
        for sample in self.samples:
            sample_file = pathlib.Path(sample)
            simg, tags = unetsl.data.loadImage(str(sample_file))
            filename="%s-%s"%(prefix, sample_file.name)
            out = pathlib.Path(self.prediction_folder, filename)
            pimg, debug = unetsl.predict.predictImage(self.model, simg, self.reduction_type, sample_normalize = self.sample_normalize, GPUS=self.gpus)
            #model, image, categorical, output_index=output_index, sample_normalize=sample_normalize
            #def predictImage(model, image, categorical = False, stride=None, output_index=-1, sample_normalize=False):
            unetsl.data.saveImage(str(out), pimg, tags)
            #saveImage(file_name, data, tags={})
            