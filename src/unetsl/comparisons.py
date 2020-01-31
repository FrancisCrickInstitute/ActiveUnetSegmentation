# -*- coding: utf-8 -*-

import skimage, numpy 
import imageio
import scipy.ndimage.filters as filters
import collections
import skimage.morphology
import skimage.filters
import re
import io

import unetsl.data

DEFAULT_CUTOFFS = ( 0.5, 0.8, 0.95)

point_kernel = numpy.array( 
                        [
                            [ 1,  1,  1],
                            [ 1, 10,  1],
                            [ 1,  1,  1]
                        ] )

endpoint_kernel = numpy.array( 
                        [
                            [ 1,  1,  1],
                            [ 1,  0,  1],
                            [ 1,  1,  1]
                        ] )


def getEndPoints(skeleton):
    points = filters.convolve( skeleton, point_kernel, mode='constant', cval=1.0)
    points = (points==11)*1
    
    return numpy.where(points)

def non_fix(seg):
    return skimage.morphology.skeletonize(seg)*1.0

def easy_fix(skeleton):
    """
        
        finds easy to fix points and eg two broken lines separated by 
        1px.
        
    """
    broken = filters.convolve( skeleton, point_kernel, mode='constant', cval=1.0)
    broken = (broken==11)*1
    
    ends = filters.convolve(broken, endpoint_kernel, mode='constant', cval=0.0)
    
    BORDER_FIX=True
    
    if BORDER_FIX:
        ends[0:1,:] = (ends[0:1, :]==1)*2
        ends[ ends.shape[-2] - 1:, : ] = (ends[ ends.shape[-2] - 1:, : ] !=0 )*2
        ends[:, 0:1] = (ends[ :, 0 : 1 ] !=0 )*2
        ends[:, ends.shape[-1] -1:] = (ends[:, ends.shape[-1] -1:] !=0 )*2
    
    fillers = 1*(ends==2)
    skeleton += fillers
    skeleton = (skeleton>0)*1.0
    skeleton = skimage.morphology.skeletonize(skeleton)*1.0
    
        
    
    return skeleton

def getColor(ji, cutoffs):
    bad = numpy.array([5, 5, 5],dtype="uint8")
    
    semi_bad = numpy.array( [ 10, 0, 55 ], dtype = "uint8" )
    dsb = numpy.array([10, 10, 10 ],dtype="uint8")
    
    decent = numpy.array( [ 125, 0, 20 ], dtype = "uint8" )
    dd = numpy.array([125, 20, 20 ],dtype="uint8")
    
    good = numpy.array([255, 165, 0],dtype="uint8")
    gd = numpy.array([0, 90, 30 ],dtype="uint8")
    
    if ji>cutoffs[-1]:
        delta = ( ji - cutoffs[-1] )/ ( 1.0 - cutoffs[-1] )
        return numpy.array([good + gd*delta], dtype="uint8")
    elif ji>cutoffs[-2]:
        delta = ( ji - cutoffs[-2] )/ ( cutoffs[-1] - cutoffs[-2] )
        return numpy.array([decent + dd*delta], dtype="uint8")
    elif ji>cutoffs[-3]:
        delta = ( ji - cutoffs[-3] )/ ( cutoffs[-2] - cutoffs[-3] )
        return numpy.array([semi_bad + dsb*delta], dtype="uint8")
    else:
        return bad

def postProcess(img):
    """
        blurs membrane errors.
    """
    white = numpy.array((255, 255, 255))
    me = numpy.all((img == white), axis=2)*1000
    me[0, :] = 0
    me[me.shape[0]-1, :] = 0
    me[:, me.shape[1]-1] = 0
    me[:, 0] = 0
    print("membrane ", me.shape)
    me_blr = ( 
               filters.gaussian_filter1d(me, sigma=3, axis=0) + 
               filters.gaussian_filter1d(me, sigma=3, axis=1) +
               filters.gaussian_filter(me, sigma=2) )
    me_blr[numpy.where(me_blr>255) == 255]
    out = img + numpy.reshape( me_blr, (*me_blr.shape, 1))
    out[numpy.where(out>255)] = 255 
    return out


def jaccardIndex(pred, truth, label_image=None, cutoffs=None):
    
    p_labels = skimage.measure.label(pred, background=1, connectivity=1)
    t_labels = skimage.measure.label(truth, background=1, connectivity=1)
    
    outer = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    t_sizes = collections.defaultdict(lambda: 0)
    p_sizes = collections.defaultdict(lambda: 0)
    
    for i in range(t_labels.shape[0]):
        for j in range(t_labels.shape[1]):
            nA = t_labels[i,j]
            nB = p_labels[i,j]
            
            outer[nA][nB] += 1
            t_sizes[nA] += 1
            p_sizes[nB] += 1
    
    
    
    
    membrane = outer[0][0]*1.0/(t_sizes[0] + p_sizes[0] - outer[0][0]) 
    best = {}
    for key in outer:
        ts = t_sizes[key]
        lm = outer[key]
        for ik in lm:
            lm[ik] = lm[ik]*1.0/(p_sizes[ik] + ts - lm[ik])
            
        best[key] = max(lm.values())
        
    #if outer[0] != membrane:
    #    print(".")
    #this fails some times!? 
    best[0] = membrane
    
    #label_image[:, :, 0] = p_labels
    #label_image[:, :, 1] = (pred != 0) * 255
    #label_image[:, :, 2] = pred[:, :]
    
    if label_image is not None:
        for i in range(pred.shape[0]):
            for j in range(pred.shape[1]):
                t_label = t_labels[i,j]
                p_label = p_labels[i,j]
                if t_label == 0:
                    if p_label == 0:
                        label_image[i,j] = [0x0, 0x0, 0x0]
                    else:
                        label_image[i,j] = [0xff, 0xff, 0xff]
                elif p_label == 0:
                    label_image[i,j] = [0, 0, 0]
                else:
                    ji = outer[t_label][p_label]
                    label_image[i, j] = getColor(ji, cutoffs)
        
    
    return best

def watershedFix(frame):
    thresh = (frame==0)*1
    
    #local_maxi = skimage.feature.peak_local_max(frame, indices=False, footprint=numpy.ones((5, 5)), labels=thresh)
    
    lbled = skimage.measure.label(thresh, background=0, connectivity=1)
    #print(lbled.shape, lbled.dtype)
    ws = skimage.morphology.watershed(-thresh, lbled)
    s = skimage.morphology.skeletonize(skimage.filters.sobel(ws)!=0)*1
    return s


def createJIImageComparison(prediction, truth, slice_index=0, cutoffs=None):
    """
        prediction : name of the prediction to be compared.
        truth : expected output.
        slice_index : sliced that will be renered.
        cutoffs : for creating different color levels.
    """
    pimg = skimage.io.imread(prediction)
    timg = skimage.io.imread(truth)
    
    if pimg.shape != timg.shape:
        pimg = numpy.reshape(pimg, timg.shape)
    
    p_slice = pimg[slice_index]
    p_skeleton = 1.0*(p_slice!=0)
        
    t_slice = timg[slice_index]
    t_skeleton = 1.0*(t_slice!=0)
    
    #p_skeleton = watershedFix(p_skeleton)
    #fix-em efore doing the regions
    p_skeleton = non_fix(p_skeleton)
    t_skeleton = non_fix(t_skeleton)
    
    p_skeleton = easy_fix(p_skeleton)
    t_skeleton = easy_fix(t_skeleton)
    
    ji_image = numpy.zeros((t_slice.shape[0], t_slice.shape[1], 3), dtype="uint8")
    jaccardIndex(p_skeleton, t_skeleton, ji_image, cutoffs=cutoffs)
    ji_image = postProcess(ji_image)
    op = io.BytesIO();
    imageio.imsave(op, ji_image, "png")
    op.seek(0)
    return op;

def getEpoch(prediction):
    pat = re.compile("-e(\\d+)")
    mo = pat.search(prediction)
    if mo:
        return int(mo.group(1))
    else:
        return -1
    
def compare(prediction, truth, ji_image_name=None, cutoffs=None):
    """
        
        Takes a prediction and the known ground truth and creates metrics for
        evaluation.
        
    """
    pimg = skimage.io.imread(prediction)
    timg = skimage.io.imread(truth)
    
    if pimg.shape != timg.shape:
        pimg = numpy.reshape(pimg, timg.shape)
    
    membrane = 0.0
    cutoffs = cutoffs
    regions = [0.0]*len(cutoffs)
    ends = 0
    mem_cor = 0.0
    over_mem = 0.0
    ji_stack = []
    ji_values = [] 
    for i in range(pimg.shape[0]):
        p_slice = pimg[i]
        p_skeleton = 1.0*(p_slice!=0)
        
        t_slice = timg[i]
        t_skeleton = 1.0*(t_slice!=0)
        
        pos = (p_skeleton*t_skeleton)
        over = p_skeleton - pos
        t_sum = numpy.sum(t_skeleton)*1.0
        mem_cor += numpy.sum(pos)/t_sum
        over_mem += numpy.sum(over)/t_sum
        
        #fix-em efore doing the regions
        p_skeleton = non_fix(p_skeleton)
        t_skeleton = non_fix(t_skeleton)
        
        p_skeleton = easy_fix(p_skeleton)
        t_skeleton = easy_fix(t_skeleton)
        
        ji_image = None
        if ji_image_name is not None:
            ji_image = numpy.zeros((t_slice.shape[0], t_slice.shape[1], 3), dtype="uint8")
            ji_stack.append(ji_image)
        ji = jaccardIndex(p_skeleton, t_skeleton, ji_image, cutoffs = cutoffs)
        ji_values.append(ji)
        
        mem = ji[0]
        ji[0] = 0
        regs = [0.0]*len(cutoffs)
        for k in ji:
            v = ji[k]
            for i, c in enumerate(cutoffs):
                if v>c:
                    regs[i] += 1
                
        ends += len(getEndPoints(p_skeleton)[0])
        membrane += mem
        for i in range(len(regions)):
            regions[i] += regs[i]/(len(ji)-1)
    mem_cor = mem_cor/pimg.shape[0]
    over_mem = over_mem/pimg.shape[0]
    
    if ji_image_name is not None:
        skimage.io.imsave(ji_image_name, numpy.array(ji_stack , dtype="uint8"))
    
    membrane = membrane/pimg.shape[0]
    for i, r in enumerate(regions):
        regions[i] = r/pimg.shape[0]
    epoch = getEpoch(prediction)
    values = [epoch, ends, mem_cor, over_mem, membrane]
    values += regions
    
    return values, ji_values

def evaluateSkeleton(prediction, ji_stack=None, cutoffs=DEFAULT_CUTOFFS):
    """
        
        Takes a prediction and the known ground truth and creates metrics for
        evaluation.
        
    """
    ji_stack = []
    pimg, tags = unetsl.data.loadImage(prediction)
    #channel, z, y, x
    pimg = pimg[0]
    
    membrane = 0.0
    cutoffs = cutoffs
    regions = [0.0]*len(cutoffs)
    ends = 0
    mem_cor = 0.0
    over_mem = 0.0
    ji_values = []
    
    p_slice = None
    #first channel is the skeleton
    for slc in range(pimg.shape[0] - 1):
        
        if p_slice is None:
            #only do it once.
            p_slice = pimg[slc]
            p_skeleton = 1.0*(p_slice!=0)
            #fix-em efore doing the regions
            p_skeleton = non_fix(p_skeleton)
            p_skeleton = easy_fix(p_skeleton)
        
        next_slice = pimg[slc + 1]
        next_skeleton = 1.0*(next_slice!=0)
        next_skeleton = non_fix(next_skeleton)
        next_skeleton = easy_fix(next_skeleton)
        
        ji_image = None
        if ji_stack is not None:
            ji_image = numpy.zeros((p_slice.shape[0], p_slice.shape[1], 3), dtype="uint8")
            ji_stack.append(ji_image)

        ji = jaccardIndex(p_skeleton, next_skeleton, ji_image, cutoffs = cutoffs)
        ji_values.append(ji)
        
        mem = ji[0]
        ji[0] = 0
        regs = [0.0]*len(cutoffs)
        for k in ji:
            v = ji[k]
            for i, c in enumerate(cutoffs):
                if v>c:
                    regs[i] += 1
                
        ends += len(getEndPoints(p_skeleton)[0])
        membrane += mem
        for i in range(len(regions)):
            regions[i] += regs[i]/(len(ji)-1)
        p_slice = next_slice
        p_skeleton = next_skeleton
        
    mem_cor = mem_cor/pimg.shape[0]
    over_mem = over_mem/pimg.shape[0]
    
    
    membrane = membrane/pimg.shape[0]
    for i, r in enumerate(regions):
        regions[i] = r/pimg.shape[0]
    epoch = getEpoch(prediction)
    values = [epoch, ends, mem_cor, over_mem, membrane]
    values += regions
    if ji_stack is not None:
        label_name="labelled/%s"%prediction.replace(".tif", "-labelled.tif")
        skimage.io.imsave(
                label_name, 
                numpy.array(
                        numpy.sum(ji_stack, axis=0)/len(ji_stack),
                        dtype="uint8"
                    )
            )
                
    return values, ji_values