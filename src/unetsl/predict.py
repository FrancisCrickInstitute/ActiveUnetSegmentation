# -*- coding: utf-8 -*-
import unetsl.data
import numpy

def upsample(stack, shape):
    """
        upsamples the last three dimensions and keeps the other dimensions
        unchanged.
    """
    scale = [o//i for o,i in zip(shape[-3:], stack.shape[-3:])]
    return numpy.kron(stack, numpy.ones(tuple(scale)))

def createSliceTuple(origin, size):
    ret = []
    for i,j in zip(origin, size):
        ret.append(slice(i, j+i))
    return tuple(ret)

MODEL_KEY = "model file"
IMG_KEY = "image to predict"
OUT_KEY = "output image"
REDUCTION_TYPE = "reduction type"
DEBUG = "debug"
OUTPUT_INDEX = "output index"
LAYER_SHAPER = "layer shaper"
CATEGORICAL_REDUCTION = 0
MULTICLASS_REDUCTION = 1
LINEAR_REDUCTION = 2


def cropShaper(batch, target_shape):
    """
        The target shape should the shape of the destination image.
        
        A batch should be (N, C, Z, Y, X) and target_shape should be
        (N, C, Z*, Y*, X*)
        
        Args:
            batch (numpy.array): data to be reshaped in the space dimensions.
            target_shape ( [int, ...]): target shape, only last 3 dimensions are used.
            
        Return:
            A zero padded version of batch.
    """
    #possibly a list. 
    target_shape = tuple(target_shape[-3:])
    
    #total dimensions - should be 5
    dims = len(batch.shape)
    
    #patch dimensions, currently 3.
    pls = len(target_shape)
    
    fill = dims - pls

    view = batch.shape[-pls:]
    
    offset = tuple( (ti - vi)//2 for vi, ti in zip(view, target_shape) )
    
    lows = (0, )*fill + offset
    span = batch.shape[:fill] + view
    
    highs = tuple(l + s for l, s in zip(lows, span))
    
    slc = tuple(slice(l, h) for l, h in zip(lows, highs) )
    shape = batch.shape[ : fill ] + target_shape

    out = numpy.zeros(shape, dtype=batch.dtype)
    out[slc] = batch
    return out

def getShaper(shaper_name):
    """
        cannot know the shaper without knowing the head!
    """
    if shaper_name=="upsample":
        return upsample
    if shaper_name == "crop":
        return cropShaper

class LayerShaper:
    def __init__(self):
        pass
    def __call__(stack, shape):
        """
            shape is the desired output shape and stack is a batch of data, 
            expected shape. (n, c, z, y, x)
            (n, c, z, y, x)
        """
        pass

DEFAULT_REDUCTION_TYPES = {
        "Sigmoid" : MULTICLASS_REDUCTION,
        "Softmax" : CATEGORICAL_REDUCTION, 
        "Relu" : LINEAR_REDUCTION
        }
def guessReduction(output):
    try:
        for key in DEFAULT_REDUCTION_TYPES:
            if key in output.name:
                return DEFAULT_REDUCTION_TYPES[key]
    except:
        pass
    return MULTICLASS_REDUCTION

def guessOutputReductionTypes(model):
    guessed_types = {}
    om = unetsl.model.getOutputMap(model)
    for key in om:
        guessed_types[key] = guessReduction(om[key])
    print("guessed reductions")
    return tuple( guessed_types[key] for key in guessed_types)    
    
class MultiChannelPredictor:
    def __init__(self, model, image, reduction_types =[ ], stride=None, sample_normalize=False, batch_size=2, debug=False, GPUS=1, layer_shapers=[upsample,]):
        self.model = model
        self.image = image
        self.reduction_types = reduction_types
        self.stride = stride
        self.sample_normalize = sample_normalize
        self.batch_size = batch_size
        self.debug = debug 
        self.GPUS = GPUS
        self.layer_shapers=layer_shapers
    
    
    def predict(self):
        return self.predictImage(self.image);
    
    def predictImage(self, image):
        if len(self.reduction_types) < 1:
            self.reduction_types = guessOutputReductionTypes(self.model)
            
        return predictMultiChannelImage(
                self.model, 
                image, 
                reduction_type = self.reduction_types, 
                stride = self.stride, 
                sample_normalize = self.sample_normalize, 
                batch_size = self.batch_size, 
                debug = self.debug, 
                GPUS = self.GPUS, 
                shapers = self.layer_shapers
            )

def predictionToLabelledImage(prediction, reduction_type, labelled_shape):
    """
        recieves a prediction (labels, z, y, x) and returns a labeled image.
        (1, z, y, x), probabilities
    """
    if reduction_type==CATEGORICAL_REDUCTION:
        probabilities = numpy.max(prediction, axis=0)
        dexes = numpy.argmax(prediction, axis=0)
        labelled = 1<<numpy.array(dexes, dtype='uint8')
    elif reduction_type==MULTICLASS_REDUCTION:
        labelled = numpy.zeros(labelled_shape, dtype='uint8')
        for label in range(prediction.shape[0]):
            patch = numpy.array((prediction[label]>0.5)*(1<<label), dtype='uint16')
            labelled |= patch 
        probabilities = numpy.max(prediction, axis=0, keepdims=True)
    elif reduction_type==LINEAR_REDUCTION:
        labelled = numpy.array((prediction), dtype='uint16')
        probabilities = None
        
    return labelled, probabilities

def generateProbabilityWindow(input_shape, labelled_shape, stride):
    body = 1.0
    face = 0.75
    edge = 0.5
    corner = 0.25

    window = numpy.zeros(labelled_shape, dtype='float32') + edge
    r0 = (input_shape[1:] - stride)//2
    important = createSliceTuple(r0, stride)
    window[0][important] = body
    
    #corner at origin
    ox = 0
    oy = 0
    oz = 0
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    oy = r0[1] + stride[1]
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    oz = r0[0] + stride[0]
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    oy = 0
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    ox = r0[2] + stride[2]
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    oz = 0
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    oy = r0[1] + stride[1]
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    oz = r0[0] + stride[0]
    window[0][ createSliceTuple( (ox, oy, oz ), r0 ) ] = corner
    
    #xfaces
    xface_shape = (stride[0], stride[1], r0[2])
    window[0][ createSliceTuple( (r0[0], r0[1], 0), xface_shape ) ] = face
    window[0][ createSliceTuple( (r0[0], r0[1], stride[2] + r0[2]), xface_shape ) ] = face
    #yfaces
    yface_shape = (stride[0], r0[1], stride[2])
    window[0][ createSliceTuple((r0[0], 0, r0[2]), yface_shape)] = face
    window[0][ createSliceTuple( (r0[0], stride[1] + r0[1], r0[2]), yface_shape )] = face
    
    #zfaces
    zface_shape = (r0[0], stride[1], stride[2])
    window[0][ createSliceTuple( (0, r0[1], r0[2]), zface_shape ) ] = face
    window[0][ createSliceTuple( (stride[0] + r0[0], r0[1], r0[2]), zface_shape ) ] = face


    return window

def predictImage(model, image, reduction_type = MULTICLASS_REDUCTION, stride=None, output_index=-1, sample_normalize=False, batch_size=2, debug=False, GPUS=1, shaper=upsample):
    #input shape is (c, z, y, x)
    input_shape = unetsl.model.getInputShape(model)
    #TODO fix this.
    image = unetsl.data.splitIntoChannels(input_shape, image)
    
    #spatial dimensions
    patch_shape = numpy.array(input_shape[1:])
    if stride is None:
        stride = patch_shape//2
        for i, s in enumerate(stride):
            if s==0:
                stride[i] = 1
    else:
        stride=numpy.array(stride)
    
    #single channel output, input shape spatial dimension.
    out_shape = [1] + [s for s in image.shape[-3:]]
    labelled_shape = [1] + list(patch_shape)
    
    
    
    slices = []
    chunks = []
    
    for k in unetsl.data.fullRange(image.shape[-3], input_shape[-3], stride[-3]):
        for j in unetsl.data.fullRange(image.shape[-2], input_shape[-2], stride[-2]):
            for i in unetsl.data.fullRange(image.shape[-1], input_shape[-1], stride[-1]):
                slc = createSliceTuple((0, k, j, i), input_shape)
                slices.append(slc)
    
    
    while len(slices)%GPUS != 0:
        slices.append(slices[0])
    if batch_size < 0:
        batch_size = len(slices)
    else:
        if batch_size < GPUS:
            batch_size=GPUS
            
    #TODO improve the prediction window to not overwrite more edge-cases.
    window = generateProbabilityWindow(input_shape, labelled_shape, stride)
    out_stack = []
    debug_out_stack = []
    
    nslices = len(slices)
    nframes = image.shape[0]
    
    
    for n_frame, frame in enumerate(image):
        out = numpy.zeros(out_shape, dtype="uint8")
        debug_out = numpy.zeros(out_shape, dtype="float32")
        for j in range(0, len(slices), batch_size):
            to_send = batch_size
            if len(slices)<j + batch_size:
                to_send = len(slices) - j
                
            s_slices = slices[j:j+to_send]
            chunks = numpy.array(
                                  [ frame[slc] for slc in s_slices ], 
                                  dtype="uint16"
                                )
            if sample_normalize:
                chunks = unetsl.data.normalizeImages(chunks)
            
            predictions = model.predict(chunks)
            print(j + n_frame*nslices, " of ", nslices*nframes)
            if isinstance(predictions, list):
                predictions = shaper(predictions[output_index], patch_shape)
                
            for (slc, prediction) in zip(s_slices, predictions):
                labels, probs = predictionToLabelledImage(prediction, reduction_type, labelled_shape)
                if reduction_type==CATEGORICAL_REDUCTION:
                    #TODO include the window for probability comparisons.
                    org = out[slc]
                    old = debug_out[slc]
                    imp = (probs>old)*1
                    nimp = 1-imp
                    upd = (probs==old)*labels
                    debug_out[slc] =probs*imp + old*nimp
                    out[slc] = (nimp)*(org | upd ) + labels*imp
                elif reduction_type==MULTICLASS_REDUCTION:
    
                    original_probs = debug_out[slc]
                    improving = numpy.where(window>original_probs)
                    updating = numpy.where(window==original_probs)
                    
                    out[slc][improving] = labels[improving]
                    out[slc][updating] |= labels[updating]
                    
                    debug_out[slc][improving] = window[improving]
                    
                elif reduction_type==LINEAR_REDUCTION:
                    original_probs = debug_out[slc]
                    improving = numpy.where(window>original_probs)
                    updating = numpy.where(window==original_probs)
                    
                    out[slc][improving] = labels[improving]
                    out[slc][updating] = (labels[updating] + out[slc][updating])/2
                    debug_out[slc][improving] = window[improving]
                    
        out_stack.append(out)
        debug_out_stack.append(debug_out)                
    return numpy.array(out_stack), numpy.array(debug_out_stack)
    

def predictMultiChannelImage(model, image, reduction_type =[], stride=None, output_index=None, sample_normalize=False, batch_size=2, debug=False, GPUS=1, shapers=[upsample, ]):
    """
        The goal is to support multi-channel predictions where a channel denotes an output. Each channel will then have an associated
        reduction type and sizer.
        
        If not supplied the reduction type will be inferred by the op name of output tensor.
        
        The sizer operation will default to "upsample" as there is no way to infer the re-sizing operation
        from the information in the model. Possibly include the sizer type in the output-name.
        
        output_index needs to be indexes for reduction_type and shapers eg: if reduction_type and shapers are 
        dictionaries, the output index should be keys (probably strings) and if they're lists
        then the output index should be valid integers.
        
        Args:
            model ( tf.Model ): 
            image ( numpy.array ):
            reduction_type ():
            stride ()
            output_index
            sample_normalize
            batch_size
            debug
            GPUS
            shapers 
            
            
        Return:
            numpy array, numpy array:
                The first image is the prediction, after processing. The second image
                is a debug image, if applicable.
        
    """
    reduction_types = reduction_type
    #input shape is (c, z, y, x)
    input_shape = unetsl.model.getInputShape(model)
    image = unetsl.data.splitIntoChannels(input_shape, image)
    
    print("predicting with shape (n, c, z, y, x) :: ", image.shape)
        
    patch_shape = numpy.array(input_shape[1:])
    if stride is None:
        stride = patch_shape//2
        for i, s in enumerate(stride):
            if s==0:
                stride[i] = 1
    else:
        stride=numpy.array(stride)
    
    
    outputs = unetsl.model.getOutputMap(model)
    
    if output_index is None:
        output_index = [i for i in range(len(outputs))]
    
    noc = len(output_index)
    
    if len(reduction_type) == 0:
        reduction_types = guessOutputReductionTypes(model)
    if len(reduction_type)==noc:
        reduction_types = reduction_type
    elif len(reduction_type)==1:
        reduction_types = noc*reduction_type
        
    #output channels / spatial dimensions
    out_shape = [noc] + [s for s in image.shape[-3:]]
    labelled_shape = [1] + list(patch_shape)
    
    
    
    slices = []
    
    
    for k in unetsl.data.fullRange(image.shape[-3], input_shape[-3], stride[-3]):
        for j in unetsl.data.fullRange(image.shape[-2], input_shape[-2], stride[-2]):
            for i in unetsl.data.fullRange(image.shape[-1], input_shape[-1], stride[-1]):
                slc = createSliceTuple((0, k, j, i), input_shape)
                slices.append(slc)
    
    
    while len(slices)%GPUS != 0:
        slices.append(slices[0])
    if batch_size < 0:
        batch_size = len(slices)
    else:
        if batch_size < GPUS:
            batch_size=GPUS
    
    #TODO improve the prediction window to not overwrite more edge-cases.
    window = generateProbabilityWindow(input_shape, labelled_shape, stride)

    full_out_stack = []
    debug_out_stack = []
    
    
    for frame in image:
        full_out = numpy.zeros(out_shape, dtype="uint8")
        full_debug_out = numpy.zeros(out_shape, dtype="float32")
        for j in range(0, len(slices), batch_size):
            to_send = batch_size
            if len(slices)<j + batch_size:
                to_send = len(slices) - j
                
            s_slices = slices[j:j+to_send]
    
            chunks = numpy.array(
                                  [ frame[slc] for slc in s_slices ], 
                                  dtype="uint16"
                                )
            
            if sample_normalize:
                chunks = unetsl.data.normalizeImages(chunks)
            predictions = model.predict(chunks)
            for index, oi in enumerate(output_index):
                predictions[index] = shapers[oi](
                        predictions[index], 
                        labelled_shape )
            
            
            for ch, pred in enumerate(predictions):
                out = full_out[ch:ch+1]
                debug_out = full_debug_out[ch:ch+1]
                reduction_type = reduction_types[ch]        
                for (slc, prediction) in zip(s_slices, pred):
                    labels, probs = predictionToLabelledImage(prediction, reduction_type, labelled_shape)
                    
                    if reduction_type==CATEGORICAL_REDUCTION:
                        #TODO include the window for probability comparisons.
                        probs = probs*window
                        
                        org = out[slc]
                        old = debug_out[slc]
                        imp = (probs>old)*1
                        nimp = 1-imp
                        upd = (probs==old)*labels
                        debug_out[slc] =probs*imp + old*nimp
                        out[slc] = (nimp)*(org | upd ) + labels*imp
        
                    elif reduction_type==MULTICLASS_REDUCTION:
        
                        original_probs = debug_out[slc]
                                        
                        improving = numpy.where(window>original_probs)
                        updating = numpy.where(window==original_probs)
                        
                        out[slc][improving] = labels[improving]
                        out[slc][updating] |= labels[updating]
                        
                        debug_out[slc][improving] = window[improving]
                        
                    elif reduction_type==LINEAR_REDUCTION:
                        original_probs = debug_out[slc]
                        improving = numpy.where(window>original_probs)
                        updating = numpy.where(window==original_probs)
                        
                        out[slc][improving] = labels[improving]
                        out[slc][updating] = (labels[updating] + out[slc][updating])/2
                        debug_out[slc][improving] = window[improving]
            print( ( j+ batch_size), "completed of:", len(slices) )    
        debug_out_stack.append(debug_out)
        full_out_stack.append(full_out);
                
    return numpy.array(full_out_stack), numpy.array(debug_out_stack)

