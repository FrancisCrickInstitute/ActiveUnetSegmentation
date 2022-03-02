import numpy

#from skimage.external.tifffile import TiffFile, TiffWriter
from tifffile import TiffFile, TiffWriter
import skimage.io
import skimage.morphology
import skimage.transform

import random
import json

import tensorflow
from tensorflow import nn

import tensorflow.keras as keras
from tensorflow.keras.utils import Sequence

import re

import pathlib
import math
import unetsl

import os.path

import scipy.ndimage

import matplotlib

from matplotlib import pyplot
from matplotlib.widgets import Slider

"""
Data keys
"""
SOURCE_TYPE = "source type"
PAIRED_DIRECTORY = "paired directories"
PAIRED_FILES = "paired files"
WEIGHTED_DIRECTORY = "weighted directories"
LABELS_TO_CATEGORY = "labels to category"
INPUT_FOLDERS = "input folders"
LABEL_FOLDERS = "label folders"
WEIGHTS_FOLDERS = "weights folders"
TRAINING_IMAGES = "training images"
LABEL_IMAGES = "label images"
ROTATIONS = "rotations"
CROP = "crop" 
LABELLER = "labeller"
REGION_LABELLER = "region labels"
MULTICLASS_LABELS = "multiclass labels"
CATEGORICAL_LABELS = "categorical labels"
DOUBLE_MEMBRANE_LABELS = "double membrane labels"
LINEAR_LABELS = "linear labels"

"""
Data Functions
"""   

class DataSource:
    def __init__(self):
        pass
    def getDataGenerators(self, stride, batch_size):
        pass
    def updateGeometry(self, *args):
        pass
    def split(self, *args):
        return self, None
    def steps(self, *args):
        return 0

class RequiredArgument(Exception):
    pass


class IndexedVolumeData:
    """
        Indexed volume pairs two volumes, images together and indexes over them
        returning each chunk. Usage
            ivd = Ivd(volume, labels, n_labels, patches, stride, labeller)
            ivd.generateIndexes()
            dg = ivd.getDataGenerator()
        
        The patches, stride, and indexes are optional, created for splitting
        the indexed volume.
        
        volume: an image with 5 dimensions. [channel, z, y, x]
        labels: image with the same zyx dimensions [n_labels, z, y, x].
        n_labels: number of output channels. 
        patches: size of input
        stride: distance between 
        
    """
    def __init__(self, volume, labels, n_labels=1, patches=(1,1,1,1), stride = None, indexes=None, labeller=None, normalize_samples=None, padding=None):
        if padding is not None:
            raise Exception("padding is not an accepted argument")
        else:
            self.padding = [0, 10, 10]
        if normalize_samples is None:
            raise RequiredArgument("normalize_samples is a required argument")
        if stride is None:
            stride = patches[:]
        
        if volume.shape[-3:] != labels.shape[-3:]:
            raise Exception("label and sample data differ in x,y,z dimensions %s != %s"%(volume.shape[-3:], labels.shape[-3:]))
        self.volume = volume
        self.labels = labels
        self.n_labels = n_labels
        self.patches = patches
        self.stride = stride
        self.indexes = indexes
        self.labeller=labeller
        self.normalize_samples=normalize_samples
        
    def updateGeometry(self, n_labels, patches, stride=None, padding=None):
        if stride is None:
            stride = patches[:]
        if self.volume.shape[0] != patches[0]:
            self.volume = splitIntoChannels(patches, self.volume)
        self.n_labels = n_labels
        self.patches = patches
        self.stride = stride
        
    def generateIndexes(self):
        self.indexes=indexVolume(self.volume, self.patches, self.stride, self.padding)
    def setIndexes(self, indexes):
        self.indexes=indexes
        
    def getDataGenerator(self, batch_size=1):
        if(self.indexes==None):
            self.generateIndexes()
        if self.padding is None:
            return getDataGenerator(self.volume, self.labels, self.n_labels, self.indexes, self.patches, batch_size=batch_size, labeller=self.labeller, normalize_samples = self.normalize_samples)
        else:
            return getPaddedDataGenerator(self.volume, self.labels, self.n_labels, self.indexes, self.patches, batch_size=batch_size, labeller=self.labeller, normalize_samples = self.normalize_samples, padding=self.padding)
        
    def size(self):
        return len(self.indexes)
    def steps(self, batch_size):
        n = len(self.indexes)
        batches = n//batch_size
        if n == batches*batch_size:
            return batches
        else:
            return batches + 1
    def split(self, f, shuffle=True):
        if(shuffle):
            random.shuffle(self.indexes);
        s1 = int(f*len(self.indexes))
        return (
            IndexedVolumeData(self.volume, self.labels, self.n_labels, self.patches, self.stride, self.indexes[:s1], self.labeller, self.normalize_samples),
            IndexedVolumeData(self.volume, self.labels, self.n_labels, self.patches, self.stride, self.indexes[s1:], self.labeller, self.normalize_samples)
            )
    def __str__(self):
        l = 0
        if self.indexes:
            l = len(self.indexes)
        return "%s shape: %s n_labels: %s indexes %s normalize: %s"%(
                self.__class__, self.volume.shape, self.n_labels, 
                l, self.normalize_samples)

class TimeSeriesDataGenerator:
    """
        Outdated broken until further notice.
    """
    def __init__(self, file_list, patch_size, out_patch, channels, batch, crop, stride):
        self.n = -1
        self.file_list = file_list
        self.patch_size = patch_size
        self.out_patch = out_patch
        self.channels = channels
        self.batch = batch
        self.crop = crop
        self.n_labels = out_patch[0]
        self.stride = stride
        self.loadFirstStack()
        
        
    def getCount(self):
        return self.n//self.batch
    
    def getGenerator(self):
        
        while True:
            loaded = list(self.loaded)
            self.loaded = []
            
            for i in range(len(self.file_list)):
                #process
                genx = []
                for j in range(self.channels):
                    genx.append(loaded[j].getDataGenerator())
                steps = loaded[self.before].steps(self.batch)
                for j in range(steps):
                    xbatch = []
                    ybatch = []
                    for k in range(self.batch):
                        """
                            x.shape 1, 1, z, y, x
                            y.shape 1, 2, z, y, x
                            
                            normally the batch would be the first index, but since
                            the 'time' is being stored as a channel that are
                            accumulated 
                        """
                        xs = []
                        ys = []
                        for c in range(self.channels):
                            x, y = genx[c].__next__()
                            """
                                we only segment 1 image. The other two are used at
                                different times.
                            """
                            xs.append(x[0,0])
                            if c == self.before:
                                #before is time_points//2 so the middle index.
                                ys = y[0]
                        xbatch.append(xs)
                        ybatch.append(ys)
                    yield numpy.array(xbatch), numpy.array(ybatch)
                
                
                #shift
                for j in range(len(loaded)-1):
                    loaded[j] = loaded[j+1]
                    
                dex = i + self.after + 1
                if dex < len(self.file_list):
                    img, _ = loadImage(self.file_list[dex][0], self.crop)
                    skel, _ = loadImage(self.file_list[dex][1], self.crop)
                    next_stack = IndexedVolumeData(img, skel, self.n_labels, self.patch_size, out_patches=self.out_patch, stride=self.stride, normalize_samples=self.normalize_samples)
                    loaded[-1] = next_stack
            self.loadFirstStack()
    
    def loadFirstStack(self):
        """
            loads the first stack of images.
        """
        self.loaded = []
        self.before = self.channels//2
        self.after = self.channels//2
        
        img, _ = loadImage(self.file_list[0][0], self.crop)
        skel, _ = loadImage(self.file_list[0][1], self.crop)
        current = IndexedVolumeData(img, skel, self.n_labels, self.patch_size, out_patches=self.out_patch, stride = self.stride, normalize_samples=self.normalize_samples)
        current.generateIndexes()
        self.n = len(self.file_list)*current.steps(1)
        
        for i in range(self.before):
            self.loaded.append(current)
        self.loaded.append(current)
        for i in range(self.after):
            img, _ = loadImage(self.file_list[1 + i][0], self.crop)
            skel, _ = loadImage(self.file_list[1 + i][1], self.crop)
            next_stack = IndexedVolumeData(img, skel, self.n_labels, self.patch_size, out_patches=self.out_patch, stride = self.stride, normalize_samples=self.normalize_samples)
            self.loaded.append(next_stack)
        
class RotatedIndexedVolumeData(IndexedVolumeData):
    def __init__(self, volume, labels, angle, n_labels=1, patches=(1, 1, 1, 1), stride = None, indexes=None, labeller=None, normalize_samples=None):
        """
            volume: full image data that will be indexed over
            labels: label volume data that will be labelled
            angle: rotation angle in radians
            n_labels: number of labels
            patches: size of volumes to be sampled
            stride: stride to be used for generating indexes.
            indexes: if the indexes were previously generated.
            
        """
        self.angle = angle
            
        IndexedVolumeData.__init__(self, volume, labels, n_labels, patches, stride, indexes, labeller, normalize_samples)
        
        if indexes:
            #if indexes were already generated can only assume angle/patches are correct.
            self.rotated_patch_size = getCropStride(self.patches, self.angle);

        
    def generateIndexes(self):
        self.rotated_patch_size = getCropStride(self.patches, self.angle)
        self.indexes=indexVolume(self.volume, self.rotated_patch_size, self.stride, self.padding)
    def getDataGenerator(self, batch_size=1):
        if(self.indexes==None):
            self.generateIndexes()
        large_patch_generator = getDataGenerator(self.volume, self.labels, self.n_labels, self.indexes, self.rotated_patch_size, batch_size=batch_size, labeller = self.labeller, normalize_samples = self.normalize_samples)
        offset = [(np - p)//2 for np, p in zip(self.rotated_patch_size, self.patches)]
        angle_deg = self.angle*180/math.pi
        for x_batch, y_batch in large_patch_generator:
            for sample_czyx in x_batch:
                for channel_zyx in sample_czyx:
                    for slice_yx in channel_zyx:
                        slice_yx[:, :] = skimage.transform.rotate(slice_yx, angle_deg, preserve_range=True)

            for sample_czyx in y_batch:
                for channel_zyx in sample_czyx:
                    for slice_yx in channel_zyx:
                        slice_yx[:, :] = skimage.transform.rotate(slice_yx, angle_deg, preserve_range=True, order=0)
                        #slice_yx[:, :] = rotate2DByPixels(slice_yx, angle_deg)
            
            x_batch = x_batch[:, 
                              offset[0]:offset[0] + self.patches[0], 
                              offset[1]:offset[1] + self.patches[1],
                              offset[2]:offset[2] + self.patches[2],
                              offset[3]:offset[3] + self.patches[3]
                              ]
            y_batch =y_batch[:, 
                              offset[0]:offset[0] + self.n_labels, 
                              offset[1]:offset[1] + self.patches[1],
                              offset[2]:offset[2] + self.patches[2],
                              offset[3]:offset[3] + self.patches[3]
                              ]
            yield x_batch, y_batch
    
    def split(self, f):
        random.shuffle(self.indexes);
        s1 = int(f*len(self.indexes))
        return (
            RotatedIndexedVolumeData(self.volume, self.labels, self.angle, self.n_labels, patches = self.patches, stride = self.stride, indexes = self.indexes[:s1], labeller = self.labeller, normalize_samples=self.normalize_samples),
            RotatedIndexedVolumeData(self.volume, self.labels, self.angle, self.n_labels, patches = self.patches, stride = self.stride, indexes = self.indexes[s1:], labeller = self.labeller, normalize_samples=self.normalize_samples)
            )

class WeightedIndexedVolumeData(IndexedVolumeData):
    def __init__(self, volume, labels, weights, n_labels=1, patches=(1,1,1), stride = None, indexes=None, labeller=None, normalize_samples = None):
        IndexedVolumeData.__init__(self, volume, labels, n_labels, patches, stride, indexes, labeller, normalize_samples)
        self.weights = weights
    def split(self, f):
        random.shuffle(self.indexes);
        s1 = int(f*len(self.indexes))
        
        return (
            WeightedIndexedVolumeData(self.volume, self.labels, self.weights, self.n_labels, patches = self.patches, stride = self.stride, indexes = self.indexes[:s1], labeller=self.labeller, normalize_samples=self.normalize_samples),
            WeightedIndexedVolumeData(self.volume, self.labels, self.weights, self.n_labels, patches = self.patches, stride = self.stride, indexes = self.indexes[s1:], labeller=self.labeller, normalize_samples=self.normalize_samples)
            )
    def getDataGenerator(self, batch_size=1):
        if(self.indexes==None):
            self.generateIndexes()
        return getWeightedDataGenerator(self.volume, self.labels, self.weights, self.n_labels, self.indexes, self.patches, batch_size=batch_size, labeller = self.labeller, normalize_samples = self.normalize_samples)
    

class InfiniteGenerator:
    def __init__(self, repeatingGenerators, randomize=True):
        """
            repeatingGenerators needs to be a list of tuples. [ ( n, gen), ...]
            n is the number of steps before repeating for a generator.
            gen is the generator
        """
        self.generators = repeatingGenerators
        self.batches = sum(c[0] for c in repeatingGenerators)
        
        
        gen_steps = [c[0] for c in repeatingGenerators]
        single_indexes = [i for i in range(len(gen_steps))]
        self.indexes = numpy.repeat(single_indexes, gen_steps)
        
        if randomize:
            numpy.random.shuffle(self.indexes)
        self.index = 0
        
    def __iter__(self):
        return self
    def getNBatches(self):
        return self.batches
    def generator(self):
        raise Exception("what are you doing!?")
        while True:
            #generatorLog("%d Top of the list"%tally)
            index = 0;
            for steps, generator in self.generators:
                for i in range(steps):
                    yield generator.__next__()
                index += 1
    def __next__(self):
        if self.index==self.batches:
            self.index = 0
        dex = self.indexes[self.index]
        self.index += 1
        return self.generators[dex][1].__next__()
            
    
def get_dims(n_chan):
    mx = int(math.sqrt(n_chan))
    
    factors = []
    for i in range(1, mx+1):
        if n_chan%i==0:
            factors.append((n_chan/i, i))
    factors.sort()
    return factors[-1]


class VolumeViewer:
    def __init__(self, figure_no, data, limits=None):
        """
            figure: int representing which matplotlib figure this should be
            data: (channel, z, y, x ) data. 
        """
        self.figure_no = figure_no
        self.channels=len(data)
        self.n_slices = len(data[0])
        self.slice = self.n_slices//2
        self.plots = []
        self.data=data
        limits = None
        self.initializeDisplay(limits)
    
    def initializeDisplay(self, limits):
        self.figure = pyplot.figure(self.figure_no)
        m,n = get_dims(self.channels)
        for c in range(self.channels):
            self.figure.add_subplot(m, n, (c+1) )
            slc = self.data[c, self.slice]
            mx = numpy.max(slc)
            mn = numpy.min(slc)
            if limits:
                orig = pyplot.imshow(self.data[c, self.slice], vmax=limits[1], vmin=limits[0])
            else:
                mn = numpy.min(self.data)
                mx = numpy.max(self.data)
                if mn == mx:
                    mn = 0
                    mx = 1
                orig = pyplot.imshow(self.data[c, self.slice], vmax = mx, vmin=mn)
                
            if mx==mn:
                mx = mn+1
            self.plots.append(orig)
        pyplot.subplots_adjust(left=0.1, bottom=0.25)
        axrs = pyplot.axes([0.2, 0.05, 0.65, 0.05], facecolor="blue")
        self.slider = Slider(axrs, "Slice", 0, self.n_slices-1, valinit=self.slice, valstep=1)
        self.slider.on_changed(self.setSlice)
        pyplot.show()
        
    def setData(self, data):
        self.data=data
        self.refresh()
        pass
    def setSlice(self, slc):
        slc = int(slc)
        if slc >= 0 and slc<self.n_slices:
            self.slice = slc
            self.refresh()
    def refresh(self):
        for c, plot in enumerate(self.plots):
            plot.set_data(self.data[c, self.slice])
        self.figure.canvas.draw()
        

def adInfinitum(infiniteGenerators):
    """
        Data generators are inifinite but repeat after so many steps, 
        this takes a finite number of steps from a generator then 
        proceeds to the next one.
        
    """
    return InfiniteGenerator(infiniteGenerators)

class Pooler:
    def __init__(self, shape, pool, operation):
        np = len(pool)
        in_dims = len(shape)
        skip = in_dims - np
        
        leaves = shape[:skip]
        ax = tuple()
        for i, p in enumerate(pool):
            leaves += ( shape[i + skip]//p, )
            leaves += ( p, ) 
            ax += ( skip + 2*i + 1, )
        self.ax = ax
        self.leaves = leaves
        self.op = operation
    
    def __call__(self, arr):
        return self.op(arr.reshape(self.leaves), self.ax)
    
def maxPool(arr, pool):
    """
        max pools arr in the pool dimensions.
    """
    return Pooler(arr.shape, pool, numpy.max)(arr)
    
def minPool(arr, pool):
    """
        min pools arr in the pool dimensions.
    """
    return Pooler(arr.shape, pool, numpy.min)(arr)



def rotate2DByPixels(in_img, angle_deg):
    angle=angle_deg*math.pi/180.0
    cx = in_img.shape[1]//2
    cy = in_img.shape[0]//2
    y, x = numpy.where(in_img!=0)
    
    out = numpy.zeros(in_img.shape,dtype="uint8")
    x = x - cx
    y = y - cy
    if x.shape[0]==0:
        return out
    
    angle = - angle
    xp = x*math.cos(angle) - y*math.sin(angle) + (cx)
    yp = x*math.sin(angle) + y*math.cos(angle) + (cy)
    
    
    
    mn = (0,0)
    mx = in_img.shape
    for dx in (0.15, 0.85):
        for dy in (0.15, 0.85):
            cnets = numpy.array([
                    ( yi + dy, xi + dx) for yi,xi in zip(yp, xp) if xi+dx>=mn[1] and xi+dx<mx[1] and yi+dy>=mn[0] and yi+dy<mx[0]
                            ], dtype="int")
            if len(cnets)>0:
                out[cnets[:,0], cnets[:, 1]] = 1
    
    return out

def normalizeImages(batch, sigma=1.0):
    std = batch.std(axis=(-3, -2, -1), keepdims=True)
    mn = batch.mean(axis=(-3, -2, -1), keepdims=True)
    
    std[numpy.where(std<1e-3)] = 1
    
    batch = sigma*(batch - mn)/std
    
    return batch

def normalizeBatch(batch, sigma=1.0):
    std = batch.std()
    mn = batch.mean()
    
    if  std>1e-3:
        return (batch - mn)*sigma/std
    else:
        return batch
    
def getMultiClassLabels(data, n_labels, fatten=False):
    """
    Translates a labelled volume into a set of binary labels.
    
    :param data: numpy array containing the label map with shape:  (1, ...).
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [n_labels] + list(data.shape)
    y = numpy.zeros(new_shape, numpy.int8)
    for label_index in range(n_labels):
        y[label_index] = (data>>label_index)&1
        if fatten:
            for i,sli in enumerate(y[label_index]):
                y[label_index, i]=skimage.morphology.dilation(sli)
    return y

def getLinearLabels(data, n_labels):
    """
    Doesn't change anything, keeps the values 1 to 1, currently stores at 8 bits.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param n_labels: integer values of the labels.
    :return: numpy array of shape: (n_samples, n_labels, ...)
    """
    new_shape = [1] + list(data.shape)
    y = numpy.zeros(new_shape, numpy.int8)
    y[0] = data*1
    return y

def skeletonToMultiClassRegions(data, n_labels):
    """
        performs a connected components, and labels the stack as different regions
        instead of 
    """

    #1 label for membrane, region labels for regions.
    regions = n_labels - 1
    new_shape = [n_labels] + list(data.shape)
    y = numpy.zeros(new_shape, numpy.int8)
    for i, slc in enumerate(data):
        labelled, count = scipy.ndimage.label((slc==0)*1)
        lim = n_labels
        if count<regions:
            lim = count+1
        elif count>regions:
            labelled[labelled>regions]=regions
        for j in range(lim):
            y[j, i] = (labelled==j)*1
    return y

def getCategoricalLabels(data, n_labels):
    """
        Similar to the multi-class labels, except labels are presumed to be unique
        and 0 is a label value. eg a binary image would be 2-label categries.
        
        n_labels has to be the n_non_zero_labels + 1. The 0 value will get changed
        to the highest value label
    """
    new_shape = [n_labels] + list(data.shape)
    y = numpy.zeros(new_shape, numpy.int8)
    for label_index in range(n_labels - 1):
        y[label_index] = (data>>label_index)&1
    y[ label_index - 1] = (data==0)*1
    
    return y
    
def getDoubleMembraneLabels(data, n_labels):
    """
    Translates a labelled volume into a set of binary labels.
    :param data: numpy array containing the label map with shape: (n_samples, 1, ...).
    :param labels: integer values of the labels.
    :return: binary numpy array of shape: (n_samples, n_labels, ...)
    """
    return getMultiClassLabels(data, n_labels, fatten=True)

labeller_map = {
        REGION_LABELLER : skeletonToMultiClassRegions,
        MULTICLASS_LABELS : getMultiClassLabels,
        CATEGORICAL_LABELS : getCategoricalLabels,
        DOUBLE_MEMBRANE_LABELS : getDoubleMembraneLabels,
        LINEAR_LABELS : getLinearLabels
    }

def fullRange(total, region, stride):
    """
      go from 0 to total-region, even if the last section overlaps.
    """
    x0 = 0
    while x0+region < total:
        yield x0
        x0 += stride
    
    if x0+region==total:
        yield x0
    else:
        x0 = total - region
        yield x0

def getPadding(total, region, stride):
    strided = total - region
    remains = strided%stride
    if remains<strided//4:
        return stride
    else:
        return stride
    
    
def paddedRange(total, region, stride, padding=None):
    """
        goes from 0 to total-region-padding so that the origin of an index
        can be shifted any region within padding.
        
        If padding is left as none, then the last section will be treated as
        padding. 
        
        This can also have overlap on the last frame.
        
    """
    
    if padding is None:
        padding = getPadding(total, region, stride)
    stridable = total - padding
    
    x0 = 0
    while x0+region < stridable:
        yield x0
        x0 += stride
    
    if x0+region==stridable:
        yield x0
    else:
        x0 = stridable - region
        yield x0


def getPaddedDataGenerator(xdata, ydata, n_labels, indexes, patch, batch_size=1, labeller=None, normalize_samples=False, padding=[0, 0, 0]):
    """
         Returns input batches, and output batches as sampled from the provided
         data. The data is expected to be (c, z, y, x) format and the return 
         is a tuple of (n, ci, zi, yi, xi), (n, co, zo, yo, xo) values. 
         
         This will repeat indefinitely with a period of len(indexes)
         xdata: input image
         ydata: output that will be labelled
         indexes: list of starting indexes. 
         patch: shape of the input data (c, z, y, x), note that the output
                data is (n_labels, z, y, x)
         
    """
    indexes = list(indexes)
    
    xbatch = []
    ybatch = []
    
    batches = len(indexes)//batch_size
    bonus = len(indexes) - batches*batch_size
    
    for i in range(bonus):
        indexes.append(indexes[i])
    pad = [ random.randint(0, r) for r in padding ]
    while True:
        for index in indexes:
            x = xdata[
                        0:patch[0],
                        index[1] + pad[0]:index[1] + pad[0] + patch[1],
                        index[2] + pad[1]:index[2] + pad[1] + patch[2],
                        index[3] + pad[2]:index[3] + pad[2] + patch[3]
                        ]
            y = labeller(ydata[
                        0,
                        index[1] + pad[0]:index[1] + pad[0] + patch[1],
                        index[2] + pad[1]:index[2] + pad[1] + patch[2],
                        index[3] + pad[2]:index[3] + pad[2] + patch[3]
                        ], n_labels)
            if(x.shape[-3:] != y.shape[-3:]):
                print("geometry doesn't match! x %s, y %s"%(x.shape[-3:], y.shape[-3:]))
            xbatch.append(x)
            ybatch.append(y)

            if len(xbatch)==batch_size:
                batch = numpy.array(xbatch)
                if normalize_samples:
                    batch = normalizeImages(batch)
                yield batch, numpy.array(ybatch)
                pad = [ random.randint(0, r) for r in padding ]
                xbatch = []
                ybatch = []
        #epoch, re-randomize. possibly should be in a callback.
        random.shuffle(indexes)
        pad = [ random.randint(0, r) for r in padding ]

def getDataGenerator(xdata, ydata, n_labels, indexes, patch, batch_size=1, labeller=None, normalize_samples=False, shuffle=False):
    """
         Returns input batches, and output batches as sampled from the provided
         data. The data is expected to be (c, z, y, x) format and the return 
         is a tuple of (n, ci, zi, yi, xi), (n, co, zo, yo, xo) values. 
         
         This will repeat indefinitely with a period of len(indexes)
         xdata: input image
         ydata: output that will be labelled
         indexes: list of starting indexes. 
         patch: shape of the input data (c, z, y, x), note that the output
                data is (n_labels, z, y, x)
         
    """
    indexes = list(indexes)
    xbatch = []
    ybatch = []
    
    batches = len(indexes)//batch_size
    bonus = len(indexes) - batches*batch_size
    
    for i in range(bonus):
        indexes.append(indexes[i])
    
    while True:
        for index in indexes:
            x = xdata[
                        0:patch[0],
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ]
            y = labeller(ydata[
                        0,
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ], n_labels)
            if(x.shape[-3:] != y.shape[-3:]):
                print("geometry doesn't match! x %s, y %s"%(x.shape[-3:], y.shape[-3:]))
            xbatch.append(x)
            ybatch.append(y)

            if len(xbatch)==batch_size:
                batch = numpy.array(xbatch)
                if normalize_samples:
                    batch = normalizeImages(batch)
                yield batch, numpy.array(ybatch)
                xbatch = []
                ybatch = []
        if shuffle:
            random.shuffle(indexes)

            
def getWeightedDataGenerator(xdata, ydata, weights, n_labels, indexes, patch, batch_size=1, cutoff=0.0, labeller=None, normalize_samples=False):
    """
       
    """
    
    xbatch = []
    ybatch = []
    weight_batch = []
    ybatches = [] #multi-output
    batches = len(indexes)//batch_size
    bonus = len(indexes) - batches*batch_size
    
    for i in range(bonus):
        indexes.append(indexes[i])
        
    #out_patch was broken and misleading.
    max_weights=patch[1]*patch[2]*patch[3]*3.0
    
    while True:
        #generatorLog("starting %d indexes in %d size batches"%(len(indexes), batch_size))
        for index in indexes:
            ws = numpy.sum(weights[
                        0:1,
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ]&0x127)/max_weights
            if ws<=cutoff:
                continue
            
            x = xdata[
                        0:patch[0],
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ]
            y = labeller(ydata[
                        0,
                        index[1]:index[1] + patch[1],
                        index[2]:index[2] + patch[2],
                        index[3]:index[3] + patch[3]
                        ], n_labels)
            if ( y.shape[-3:] != patch[-3:] ) or ( x.shape[-3:] != patch[-3:] ):
                raise Exception("incomplete data. z, y, x dimensions should be the same.")
            xbatch.append(x)
            ybatch.append(y)
                
            weight_batch.append(ws)
            if len(xbatch)==batch_size:
                batch = numpy.array(xbatch)
                if normalize_samples:
                    batch = normalizeImages(batch)
                    
                sm = 0
                for w in weight_batch:
                    sm += w
                    
                
                if len(ybatches)>0:
                    yield batch,[ numpy.array(yb) for yb in ybatches ], [ numpy.array(weight_batch)/sm for yb in ybatches]
                    ybatches.clear()
                else:
                    yield batch, numpy.array(ybatch), numpy.array(weight_batch)/sm
                xbatch = []
                ybatch = []
                weight_batch = []

                
def generatorLog(message):
    with open("generators-log.txt", 'a') as log:
        log.write(message)
        log.write("\n")

def getNumber(token):
    return float(num.findall(token)[0])
    
KNOWN_TAGS = {"x_resolution":float, "y_resolution":float, "spacing":getNumber, "unit":str, "channels": int, "slices": int, "frames":int}
digi = re.compile("\\d+")
num = re.compile(r"[\d\.]+")
def parseField(field):
    if any(tag in field for tag in KNOWN_TAGS):
        tokens = field.split()
        if digi.fullmatch(tokens[1]):
            """
               * <id> <tag> <type> <value>...
            """
            key = tokens[2]
            tp = tokens[3]
            
            if key in KNOWN_TAGS:
                #the resolutino is loaded/saved as inverse or some crap.
                numer= KNOWN_TAGS[key](num.findall(tokens[4])[0])
                denom = KNOWN_TAGS[key](num.findall(tokens[5])[0])
                value = numer/denom
                return key, value
        else:
            """
              Spacing and units.
              * <tag> <value>
            """
            key = tokens[1].replace(":", "")
            if key in KNOWN_TAGS:                
                value = KNOWN_TAGS[key](tokens[2])
                return key, value
    
    return None, None

def getImageJCalibration(img, tags = None):
    """
        Pulls apart the "info" string based on imagej style tiff stacks, and
        populates tags. 
        
        @param img Tifffile loaded data
        @param tags target dictionary for output.
        
        Return:
            returns tags, or a new dictionary if tags omitted or None.
    """
    if tags is None:
        tags = img.imagej_metadata
    else:
        for key in img.imagej_metadata:
            tags[key] = img.imagej_metadata[key]
    
    return tags
    
    

def loadImage(imageFile, swap_2d_time_series=True):
    """
        Loads the image file and returns it as 'frames, channels, z, y, x' data,
        or (1, channels, frames, y, x) if the image is 

        When the data is from a tiff file created by imagej, the format is 
        assumed to be (frames, slices, channels, ...)
        
        If the number of frames is > 1 but the number of slices is 1, then 
        slices and frames will get swapped because sometimes images are 
        2d time series. (1, c, frames, y, x) this produces behavior where, a
        2d+t network will work with consecutive frames, but a 2d w/out time 
        will not notice the difference.
        
        Args:
            imageFile: path to file to be loaded. converts to str.
            swap_2d_time_series: if the image has N time points and 1 z slice
                the file is reshaped to have N z-slices for using with 3d unets.
        
    """
    imageFile = str(imageFile)
    tags={}
    try:
        with TiffFile(imageFile) as tiff:
            data = []
            for p in tiff.pages:
                data.append(p.asarray())
            data = numpy.array(data)
            if tiff.is_imagej:
                #if tiff.is_rgb:
                #    print("warning: RGB LUT detected, summing along last axis!")
                #    data = numpy.sum(data, axis=-1)
                getImageJCalibration(tiff, tags)
                frames = tags.get("frames", 1)
                slices = tags.get("slices", 1)
                channels = tags.get("channels", 1)
                if swap_2d_time_series and slices==1 and frames>1:
                    print("replacing frames with slices")
                    slices = frames;
                    frames = 1
                data = data.reshape((frames, slices, channels, data.shape[-2], data.shape[-1]))
                data = numpy.rollaxis(data, 2, 1)
            if len(data.shape)==3:
                #non-imagej assumed to be single channel z-stack.
                data = data.reshape((1, 1, *data.shape[:]))
            elif len(data.shape)==4:
                #non-imagej assumed to be 1 frame of z , c,  y, x stack.
                data = numpy.rollaxis(data, 1, 0)                
                data = data.reshape((1, *data.shape[:]))
            return data, tags
            
    except Exception as error:
        print("defaulting to skimage.io because: %s"%error)
        print("Check if RGB LUT has been used!")
    
    return skimage.io.imread(imageFile), tags

def shapeThatThing(data):
    """
        @Deprecated
        
        Recieves an image as TZCYX and changes it to TCZYX. 
        For lower dimensioned images, they are broad cast to 
        higher dimension with the added axis of length 1.
        
    """
    dims = len(data.shape)    
    if dims==2:
        #single slice -> 1,1,1,Y,X
        return numpy.array([[[data]]])
    elif dims==3:
        #Z/T, Y, X ->  
        #single channel time series -> T, 1,1, Y, X
        t,h,w = data.shape
        arr = numpy.array([data])
        return arr.reshape(t, 1, 1, h, w)
    elif dims==4:
        data = numpy.array([data])
        data = numpy.rollaxis(data, 2, 1)
    elif dims==5:
        data = numpy.rollaxis(data, 2, 1)
    return data

def getResolution(tags):
    rx = 1
    ry = 1
    if "x_resolution" in tags and "y_resolution" in tags:
        rx = tags["x_resolution"]
        ry = tags["y_resolution"]
    return (rx, ry)
def getMetaData(tags):
    meta= {}
    if "unit" in tags:
        meta["unit"] = tags["unit"]
    if "spacing" in tags:
        meta["spacing"] = tags["spacing"]
    return meta
    
    
def saveImage(file_name, data, tags={}):
    file_name = str(file_name) #pathlib compat.
    resolution = getResolution(tags)
    metadata = getMetaData(tags)
    data = shapeThatThing(data)
    max_size = 2**35
    shape = data.shape
    total = 1
    for p in shape:
        total = p*total
    if total > max_size:
        out = data
        count = total//max_size
        
        if count*max_size<total:
            count += 1
        
        images_per_file = len(out)//count
        
        if images_per_file*count < len(out):
            images_per_file += 1
        
        for i in range(count):
            l = i*images_per_file
            t = l + images_per_file
            if t>len(out)+1:
                t = len(out)+1
            
            stack = data[l:t, 0:]
            part_name = file_name.replace(".tif", "-%d.tif"%i);
            
            
            with TiffWriter(part_name, imagej=True) as writer:
                writer.save(stack, resolution = resolution, metadata=metadata)
    else:
         with TiffWriter(file_name, imagej=True) as writer:
                writer.save(data, resolution = resolution, metadata=metadata)


class MalformedImageException(Exception):
    pass

def splitIntoChannels(input_shape, image):
    """
        @Deprecated
       By default images are loaded as (t, c, z, y, x) they're then parsed
       to be used by the model. This is a fallback incase the images were
       loaded with incorrect channel information.
       
       doesn't contain channel information it will be put into a (1, 1, z*t*c, y, x)
       array will attempt to separate the z*t*c based on the input shape.
       
        Args:
            input_shape : (c, z0, y0, x0) shape required by model
            image : a 5 dim ndarray, (n, c, z, y, x), that will be reshaped if 
                    the number of channels is not the same as required
                    by the input shape.
        Returns:
            image shaped (n, c, z, y, x)
    """
    channels = input_shape[0]
    if len(image.shape)==5:
        print("5 dim format, assuming [n, c, z, y, x] %s"%str(image.shape))
        if image.shape[1]!=input_shape[0]:
            print("\t\t[WARNING] updating geometry")
            if image.shape[-1] == channels:
                #input as channels last (n, z, y, x, c) 
                image = numpy.rollaxis(image, 4, 1)
            elif image.shape[-3]%channels == 0:
                #input as (n, 1, c*z, y, x)
                new_z = image.shape[-3]/channels
                y = image.shape[-2]
                x = image.shape[-1]
                
                image = numpy.reshape(image.shape[0], new_z, channels, y, x)
                image = numpy.rollaxis(image, 2, 1)
            else:
                raise MalformedImageException(
                        "Missmatched channels input (?, %s). actual shape: %s"%( input_shape, image.shape )
                    )
    else:
        raise MalformedImageException("To split into Channels, img must have 5 dim. actual shape: %s"%(image.shape, ) )
    return image




def rotateFileSource(indexedVolume, angle):
    """
        indexedVolume: indexed volume data source.
        angle: angle of rotation in radians.
    """
    return RotatedIndexedVolumeData(indexedVolume.volume, indexedVolume.labels, angle, labeller=indexedVolume.labeller, normalize_samples = indexedVolume.normalize_samples)



def getCropStride(stride, angle):
    """
        Gets the stride required for the corresponding rotation angle in
        radians.
    """
    angle = angle%(2*math.pi)
    
    new_stride = [int(s) for s in stride]
    cos = math.cos(angle)
    sin = math.sin(angle)
    
    if angle>=0 and angle<math.pi/2:
        new_stride[-1] = cos*stride[-1] + sin*stride[-2]
        new_stride[-2] = sin*stride[-1] + cos*stride[-2]
    elif angle>=math.pi/2 and angle<math.pi:
        new_stride[-1] = -cos*stride[-1] + sin*stride[-2]
        new_stride[-2] = sin*stride[-1] - cos*stride[-2]
    elif angle>=math.pi and angle<3*math.pi/2:
        new_stride[-1] = -cos*stride[-1] - sin*stride[-2]
        new_stride[-2] = - sin*stride[-1] - cos*stride[-2]
    elif angle>=1.5*math.pi:
        new_stride[-1] = + cos*stride[-1] - sin*stride[-2]
        new_stride[-2] = - sin*stride[-1] + cos*stride[-2]

    new_stride = [int(s + 0.5) for s in new_stride]
    return new_stride




def indexVolume(volume, patch_size, stride, padding):
    """
        The volume will be indexed by passing moving the patch size, across
    in increments of stride. Padding will be excluded, so that any index can
    have the padding value added to it, and they'll still be valid indexes.
    
    
    """
    res = []
    shape = [ v for v in volume.shape ] 
    
    if padding is not None:
        #removes the range so that the index can be offset anywhere between 0 and padding.
        for i, p in enumerate(padding):
            shape[1+i] -= p
            
    for i in range(len(patch_size)):
        if shape[i]<patch_size[i]:
            print("Volume is smaller than input patch! Cannot index.")
            raise Exception("volume size < patch size %s < %s"%(shape, patch_size))
    
    zs = (shape[1] - patch_size[1])//stride[1] + 1
    if zs*patch_size[1]<shape[1]:
        zs = zs + 1
    
    ys = (shape[2] - patch_size[2])//stride[2] + 1
    xs = (shape[3] - patch_size[3])//stride[3] + 1
    for k in range(zs):
        
        if k*stride[1] + patch_size[1] > shape[1]:
            zdex = shape[1] - patch_size[1]
        else:
            zdex = k*stride[1]
            
        for j in range(ys):
            ydex = j*stride[2]
            if ydex + patch_size[2] > shape[2]:
                ydex = shape[2] - patch_size[2]
            for i in range(xs):
                xdex = i*stride[3]
                if xdex + patch_size[3] > shape[3]:
                    xdex = shape[3] - patch_size[3]
                res.append((0, zdex, ydex, xdex))
    return res
    
def getFilePairs(original_folder, segmentation_folder):
    """
        Pairs the .tif files found in the original_folder with the .tif files
        found in the segmentation folder.
        - Assumes both folders share a corresponding first image, but truncates
        the input images down to the length segmentations.
    """
    grab = re.compile("(\\d+)\\.tif")
    input_folder = [ str(f) for f in pathlib.Path(original_folder).iterdir() if f.match("*tif")]
    n_inp = [(tuple(int(s) for s in grab.findall(n)), n) for n in input_folder]
    
    n_inp.sort()
    segmentation_folder = [ str(f) for f in pathlib.Path(segmentation_folder).iterdir() if f.match("*tif")]
    
    n_skel = [(tuple(int(s) for s in grab.findall(n)), n) for n in segmentation_folder]
    n_skel.sort()
    if len(n_inp)>len(n_skel):
        n_inp = n_inp[:len(n_skel)]
    
    pairs = list(zip([a[1] for a in n_inp], [b[1] for b in n_skel]))
    return pairs

def getWeightedFileGroups(original_folder, segmentation_folder, weights_folder):
    """
        Pairs the .tif files found in the original_folder with the .tif files
        found in the segmentation folder.
        - Assumes both folders share a corresponding first image, but truncates
        the input images down to the length segmentations.
    """
    original_folder = os.path.expandvars(original_folder)
    segmentation_folder = os.path.expandvars(segmentation_folder)
    weights_folder = os.path.expandvars(weights_folder)
    grab = re.compile("(\\d+)\\.tif")
    input_folder = [ str(f) for f in pathlib.Path(original_folder).iterdir() if f.match("*tif")]
    n_inp = [(tuple(int(s) for s in grab.findall(n)), n) for n in input_folder]
    
    n_inp.sort()
    segmentation_folder = [ str(f) for f in pathlib.Path(segmentation_folder).iterdir() if f.match("*tif")]
    n_skel = [(tuple(int(s) for s in grab.findall(n)), n) for n in segmentation_folder]
    n_skel.sort()

    weights_folder = [ str(f) for f in pathlib.Path(weights_folder).iterdir() if f.match("*tif")]
    n_ws = [(tuple(int(s) for s in grab.findall(n)), n) for n in weights_folder]
    n_ws.sort()

    shortest = min(map(len, (n_inp, n_skel, n_ws)))
    n_inp = n_inp[:shortest]
    n_ws = n_ws[:shortest]
    n_skel = n_skel[:shortest]
    pairs = list(zip([a[1] for a in n_inp], [b[1] for b in n_skel], [w[1] for w in n_ws]))
    return pairs



    
def getPairedDirectorySources(config, normalize_samples):
    """
        Directory 
    """
    img_pairs = []
        
    for in_folder, out_folder in zip(config[INPUT_FOLDERS], config[LABEL_FOLDERS]):
        img_pairs = img_pairs + getFilePairs(os.path.expandvars(in_folder), os.path.expandvars(out_folder))
    sources = []
    labeller = labeller_map[config[LABELLER]]
    rotations = []
    if ROTATIONS in config:
        rotations = config[ROTATIONS]
    for img, seg in img_pairs:
        source_ = fileSources(img, seg, labeller, normalize_samples)
        sources += source_
        
        for deg in rotations:
            angle = deg
            for source in source_:
                sources.append( rotateFileSource(source, angle))
    return sources

def getLabeller(labeller_name):
    return labeller_map[labeller_name]


def getPairedFileSources(config, normalize_samples):
    img_pairs = list(zip(config[TRAINING_IMAGES], config[LABEL_IMAGES]))
    sources = []
    labeller = config[LABELLER]
    for img, seg in img_pairs:
        file_sources = fileSource(img, seg, labeller, normalize_samples)
        for file_source in file_sources:
            sources.append( file_source )
            
    return sources

def fileSources(original_file, segmentation_file, labeller, normalize_samples):
    inp, _ = loadImage(original_file)
    lbl, _ = loadImage(segmentation_file)
    #new load images, this is always true!
    if len(inp.shape)==5 and len(lbl.shape)==5:
        v = []
        for inp_frame, lbl_frame in zip(inp, lbl):
            v.append(
                    IndexedVolumeData(
                            inp_frame, 
                            lbl_frame, 
                            labeller = labeller, 
                            normalize_samples=normalize_samples
                            )
                    )
    else:
        print("[Deprecated] This case is never true")
        v = [ 
            IndexedVolumeData(
                    inp, 
                    lbl, 
                    labeller = labeller, 
                    normalize_samples=normalize_samples
                    ) ]

    return v

def weightedFileSource(img, seg, weights, labeller, normalize_samples):
    inps, _ = loadImage(img)
    lbls, _ = loadImage(seg);
    wgts, _ = loadImage(weights)
    wivds = []
    for inp, lbl, wgt in zip(inps, lbls, wgts):
        wivds.append(WeightedIndexedVolumeData(inp, lbl, wgt, labeller=labeller, normalize_samples=normalize_samples))
    return wivds

def getWeightedDirectorySources(source_config, normalize_samples):
    labeller = labeller_map[source_config[LABELLER]]
    dir_groups = list(zip(
            source_config[INPUT_FOLDERS], 
            source_config[LABEL_FOLDERS], 
            source_config[WEIGHTS_FOLDERS]) )
    img_groups = []
    for im, lbl, wght in dir_groups:
        img_groups += getWeightedFileGroups(im, lbl, wght)
    sources = []
    for img, seg, weight in img_groups:
        sources += weightedFileSource(img, seg, weight, labeller, normalize_samples)
        
    return sources
    
SOURCE_DELEGATE = {
        PAIRED_DIRECTORY : getPairedDirectorySources,
        PAIRED_FILES : getPairedFileSources,
        WEIGHTED_DIRECTORY : getWeightedDirectorySources
        }

def getDataSources(data_source_configs, normalize_samples):
    """
        Data sources are sources of data. Their creation will be agnostic
        of batch size, stride, and validation which will be used when creating 
        creating generators.
        
    """
    sources = []
    for dsc in data_source_configs:
        source = SOURCE_DELEGATE[dsc[SOURCE_TYPE]](dsc, normalize_samples)
        sources += source
    return sources

def getDataSource(data_source_config, normalize_samples):
    """
        Data sources are sources of data. Their creation will be agnostic
        of batch size, stride, and validation which will be used when creating 
        creating generators.
        
    """
    source = SOURCE_DELEGATE[data_source_config[SOURCE_TYPE]](data_source_config, normalize_samples)
    return source

def getDataGenerators(data_sources,n_labels, patch_size, stride, batch_size, validation_fraction):
    """
        Turn the data_sources into generators based on the geometries provided.
    """
    training_volumes = []
    validation_volumes = []
    
    train_fraction = 1-validation_fraction
    
    for source in data_sources:
        #updateGeometry(self, n_labels, patches, stride=None, out_patches=None)
        source.updateGeometry(n_labels, patch_size, stride)
        source.generateIndexes();
        v1, v2 = source.split(train_fraction)
        training_volumes.append(v1)
        validation_volumes.append(v2)
    
    
    training_generator = adInfinitum(
            [ (v.steps(batch_size), v.getDataGenerator(batch_size) ) for v in training_volumes ]
            )
    validation_generator = adInfinitum([(v.steps(batch_size), v.getDataGenerator(batch_size)) for v in validation_volumes])
    
    return training_generator, validation_generator
