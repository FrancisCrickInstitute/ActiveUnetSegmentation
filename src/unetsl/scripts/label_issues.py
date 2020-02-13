#!/usr/bin/env python

import sys, os
import unetsl.data

import numpy

import skimage.morphology
import skimage.draw
import skimage.measure

from skimage.feature import peak_local_max
from skimage.morphology import watershed

import scipy.ndimage.filters as filters
from scipy.ndimage.morphology import distance_transform_bf, distance_transform_cdt
from scipy.ndimage.measurements import watershed_ift
from scipy.ndimage import label as ndlabel

import collections

import click

DEBUG=False

if DEBUG:
    from matplotlib import pyplot


"""
Looks for known issues in the image stacks and creates a stack of 
labels.
"""

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
                            

def easy_fix(skeleton):
    """
        
        finds easy to fix points and eg two broken lines separated by 
        1px.
        
    """
    broken = filters.convolve( skeleton, point_kernel, mode='constant', cval=1.0)
    broken = (broken==11)*1
    
    fillers = 1*(filters.convolve(broken, endpoint_kernel, mode='constant', cval=0.0)==2)
    
    skeleton += fillers
    skeleton = (skeleton>0)*1.0
    skeleton = skimage.morphology.skeletonize(skeleton)*1.0
    
    return skeleton

border = { (-2, -2),
        (-2, -1),
        (-2, 0),
        (-2, 1),
        (-2, 2),
        (-1, -2),
        (-1, 2),
        (0, -2),
        (0, 2),
        (1, -2),
        (1, 2),
        (2, -2),
        (2, -1),
        (2, 0),
        (2, 1),
        (2, 2) }
single = { 
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)}

boxes = {}

def box(l):
    if l not in boxes:
        b = newBox(l)
        boxes[l] = b
    return boxes[l]
    

def newBox(l):
    if l==0:
        return [[0, 0]]
    v = [[0,0] for i in range(4*(2*l))]
    
    for i in range(2*l):
        v[4*i][0] = l
        v[4*i][1] = i - l
        
        v[4*i + 1][0] = i
        v[4*i + 1][1] = l
        
        v[4*i + 2][0] = -l
        v[4*i + 2][1] = l - i
        
        v[4*i + 3][0] = i - l
        v[4*i + 3][1] = - l
    
    return numpy.array(v)
        
def longRangeFix(skeleton, locations, cutoff=2):
    """
        If two broken points are close enough, draw a line to connect them.
        
        skeleton is the image to be repaired.
        locations y, x cooridnate pairs.
        cutoff maximum separation for a single axis.
        
    """
    for i in range(cutoff):
        box(i)
    locations.sort()
    
    low = 0;
    fixed = 0;
    
    
    for j in range(len(locations)):
        point = locations[j]
        if(point[0]<0):
            continue
        
        neighbors = []
        
        
        
        for i in range(low, len(locations)):
            if i==j:
                continue
            
            other = locations[i];
            if other[0]<0:
                #fixed.
                continue
            
            if other[0]<point[0]-cutoff:
                low = i
            elif other[0]>point[0] + cutoff:
                break;
            else:
                if other[1] >= point[1] - cutoff and other[1]<=point[1] + cutoff:
                    neighbors.append(other);
        
        
        
        if len(neighbors)==1:
            #draw a line and remove the points.
            other = neighbors[0]
            rr, cc = skimage.draw.line(other[0], other[1], point[0], point[1])
            skeleton[rr, cc] = 1
            
            point[0] = -1
            other[0] = -1
            fixed += 1
        
        elif len(neighbors)==0:
            connecting = {(point[0], point[1])}
            if DEBUG:
                pyplot.imshow(skeleton[point[0]-7: point[0] + 7, point[1]-7 : point[1] + 7])
                pyplot.show(False)    
            for j in range(cutoff):
                f, connecting = connectToMembrane(point, box(j), box(j+1), connecting, skeleton)
                if f>0:
                    fixed += 1
                    break;
                            
                            
    return skimage.morphology.skeletonize(skeleton)*1.0

import time

class Logger:
    def __init__(self, out):
        self.out = out
    def log(self, *arguments):
        if DEBUG:
            for arg in arguments:
                self.out.write("%d:: %s\n"%(int(time.time()), str(arg)))

def connectToMembrane(point, inner, outter, connecting, skeleton, logger = Logger(sys.stdout)):
    """
      point: is the center location, should be a broken end.
      inner: inner region, relative to point, that already has connected points marked
      outter: new region, relative to point, to be explored
      connecting: points that contain membrane, and are connected to the center,
                  actual position.
      skeleton: image that will be modified.
      
    """
    
    logger.log("searching: ", point, len(inner), len(outter), "connected by", len(connecting))
    sug = []
    toBeConnected = set()
    fixed = 0
    for pt in outter:
        logger.log("  outer: ", pt)
        x = pt[1] + point[1]
        y = pt[0] + point[0]
        #go through all of the 2px range points and find membrane.
        if hasMembrane(skeleton, (y, x)):
            logger.log("membrane found!", y, x)
            connected = False
            for ipt in single:
                ib = (y + ipt[0], x + ipt[1])
                logger.log("\tchecking if: ", ib, " in ", connecting)
                if ib in connecting:
                    #see if it is connected to outter region.
                    connected = True
                    break;
            if not connected:
                logger.log("suggestion: ",pt)
                sug.append(pt)
            else:
                toBeConnected.add((y, x))
    
    if len(sug)>=1:
        #check if the same structure.
        cutoff=5
        region = numpy.array(
                skeleton[
                        point[0] - cutoff : point[0] + cutoff, 
                        point[1] - cutoff : point[1] + cutoff 
                        ])
        labelled_regions, nlabels = ndlabel(region, structure = numpy.ones((3,3)))
        
        filtered_sug = []
        
        if all(dim == cutoff*2 for dim in region.shape):
            for suggestion in sug:
                px = labelled_regions[cutoff, cutoff]
                if suggestion[0]>=cutoff or suggestion[1]>=cutoff:
                    filtered_sug.append(suggestion)
                    continue
                if labelled_regions[suggestion[0] + cutoff, suggestion[1] + cutoff] == px:
                    logger.log("same structure", labelled_regions[suggestion[0] + cutoff, suggestion[1] + cutoff], " and " , px )
                    logger.log("loc: ", suggestion)
                    logger.log(labelled_regions)
                    if DEBUG:
                        input("...")
                else:
                    filtered_sug.append(suggestion)
            sug = filtered_sug
        else:
            #the size is too small to properly align connctivity. 
            pass
        logger.log("suggestions left: ", sug)
    if len(sug)>=1:
        p = ()
        distance = 1e6
        logger.log("suggestsions: ", sug)
        
        
        
        
        for s in sug:
            #check if the new point is connected to the original point.
            
            #find closest
            d2 = s[0]*s[0] + s[1]*s[1]
            if d2 < distance:
                p = s
                distance = d2
            logger.log("chose: ", p)
        rr, cc = skimage.draw.line(point[0], point[1], point[0] + p[0], point[1] + p[1])
        if DEBUG:
            skeleton[rr, cc] += 2
            pyplot.imshow(skeleton[point[0]-7: point[0] + 7, point[1]-7 : point[1] +7])
            pyplot.show(False)                
        skeleton[rr, cc] = 1
        fixed += 1
    
    if DEBUG:
        input("fixed %s or growing: %s"%(str(sug), str(toBeConnected)))
    return fixed, toBeConnected

def hasMembrane(img, pt):
    if pt[0]>=0 and pt[0]<len(img) and pt[1]>=0 and pt[1]<len(img[0]):
        return img[pt[0], pt[1]] != 0
    return False

class IntersectingRegions:
    def __init__(self, na, nb):
        self.na = na
        self.nb = nb
    def __hash__(self):
        return self.na + (self.nb<<16)
    def __eq__(self, other):
        return (self.na==other.na) and (self.nb==other.nb)
    

def labelSkeleton(img):
    return numpy.array(skimage.measure.label(img, background=1, connectivity=1), dtype="uint16")

def jaccardIndexByRegion(truthLabels, predictionLabels):
    """
        The canon is the ground truth labels, prediction is the predicted labels. 
        The two will be binaried and labeld, Then a histogram over a 2d map
        will measure the intersections. 
    """
    outer = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    sizer = collections.defaultdict(lambda: 0)
    
    for i in range(truthLabels.shape[0]):
        for j in range(truthLabels.shape[1]):
            nA = truthLabels[i,j]
            nB = predictionLabels[i,j]
            
            outer[nA][nB] += 1
            sizer[nA] += 1
    
    for key in outer:
        outer[key] = max([x for x in outer[key].values()])
    outer[0] = 0
    
    out = numpy.zeros(truthLabels.shape, dtype="uint16")
    
    for j in range(out.shape[0]):
        row = truthLabels[j]
        for i in range(out.shape[1]):
            out[j][i] = outer[row[i]]*255/sizer[row[i]]
    
    return out

def getEndPoints(skeleton):
    points = filters.convolve( skeleton, point_kernel, mode='constant', cval=1.0)
    points = (points==11)*1
    
    return numpy.where(points)

@click.command()
@click.argument("prediction", type=click.Path(exists=True))
@click.argument("destination", type=click.Path())
def main(prediction, destination):
    
    
    img, tags = unetsl.data.loadImage(prediction)
    out = numpy.zeros((4, *img.shape[-3:]), dtype="uint8") 
    count = 0
    
    skeletons = []
    
    labels = None
    skel = None
    total=0
    fix_total = 0
    simple_fix_total = 0
    c1 = img[0,0]
    for s_no, slic in enumerate(c1):
        """
            This loop runs as though the 'current' slice is the next slice.
            labels/skel are stale values for comparison.
        """
        
        next_skel = skimage.morphology.skeletonize((c1[s_no]!=0)*1.0)*1.0
        next_labels = labelSkeleton(next_skel)
        
        if labels is None:
            labels = next_labels
            
        if skel is None:
            skel = next_skel

        out[0, s_no, :, :] = skel[:, :]*255
        labelled = numpy.zeros(skel.shape)
        
        cnets_y, cnets_x = getEndPoints(skel)
        
        broken_points = len(cnets_x)
        total += broken_points
        
        #close border edge points.
        h = len(skel)
        w = len(skel[0])
        for y, x in zip(cnets_y, cnets_x):
            if x==1:
                skel[y, 0] = 1
            elif x==w-2:
                skel[ y, w-1]=1  
            elif y==1:
                skel[0,x] = 1
            elif y==h-2:
                skel[h-1,x] = 1
        
        
        
        for i in range(len(cnets_x)):
            rr, cc = skimage.draw.circle(cnets_y[i], cnets_x[i], 5, skel.shape)
            labelled[rr, cc] = 1
        
        skel = easy_fix(skel)
        
        
        cnets_y, cnets_x = getEndPoints(skel)
        
        after_simple = len(cnets_x)
        simple_fix_total += after_simple
        
        for i in range(len(cnets_x)):
            rr, cc = skimage.draw.circle(cnets_y[i], cnets_x[i], 5, skel.shape)
            labelled[rr, cc] = 2

        coordinates = [[y,x] for y,x in zip(cnets_y, cnets_x)]
        
        #skel = longRangeFix(skel, coordinates, cutoff=5)

        cnets_y, cnets_x = getEndPoints(skel)
        
        for i in range(len(cnets_x)):
            rr, cc = skimage.draw.circle(cnets_y[i], cnets_x[i], 5, skel.shape)
            labelled[rr, cc] = 4

        coordinates = [[y,x] for y,x in zip(cnets_y, cnets_x)]                
        #skel = longRangeFix(skel, coordinates, cutoff=5)
        
        cnets_y, cnets_x = getEndPoints(skel)
        
        after_long = len(cnets_x)
        
        fix_total += after_long
        

        for i in range(len(cnets_x)):
            rr, cc = skimage.draw.circle(cnets_y[i], cnets_x[i], 5, skel.shape)
            labelled[rr, cc] = 8
        
        out[1, s_no, :, :] = labelled[:,:]
        out[2, s_no, :, :] = skel[:, :]*255
        
        if s_no + 1 < len(c1):
            next_skel = skimage.morphology.skeletonize((c1[s_no+1]!=0)*1.0)*1.0
        else:
            next_skel = skimage.morphology.skeletonize((c1[s_no-1]!=0)*1.0)*1.0
        next_labels = labelSkeleton(next_skel)
        jdex = jaccardIndexByRegion(labels, next_labels)
        out[3, s_no, :, :] = jdex
        
        labels = next_labels
        skel = next_skel
        
    print("total: ", total, " broken ends to start with. ",simple_fix_total, " after initial pass. ", fix_total, " after fixing.")
        
    unetsl.data.saveImage(destination, out.reshape((1, *out.shape)))

def developingDistanceTransformWatershed():
    
    output_path = None
    if "-o" in sys.argv:
        output_path = sys.argv[ 1 + sys.argv.find("-o")]
    
    
    batch_mode = "-b" in sys.argv
    
    if output_path and os.path.exists(output_path):
        y = input("%s exists and could be overwritten proceed y/N"%sys.argv[2])
        if y not in ["Y", "y", "yes", "Yes"]:
            sys.exit(0)
    
    img, tags = unetsl.data.loadImage(sys.argv[1])
    c1 = img[0]
    
    out = []
    
    for s_no, slic in enumerate(c1):
        iskel = (slic==0)*1.0
        dist = distance_transform_cdt(iskel)
        #features = skimage.morphology.local_maxima(dist, numpy.ones((5,5)))
        features = skimage.morphology.label(peak_local_max(dist, min_distance=10, indices=False))
        wat = watershed_ift(numpy.array(dist, dtype="uint16"), numpy.array(features - (slic!=0)*1, dtype="int32"))
        out.append(numpy.array(dist, dtype="uint16"))
        out.append(features)
        out.append(wat)
    
    if output_path:
        unetsl.data.saveImage("dt-test.tif", numpy.array(out, dtype="uint16"))
    
    

def developingJaccardIndex():
    
    if len(sys.argv)!=3:
        sys.exit(0)
    
    if os.path.exists(sys.argv[2]):
        y = input("%s exists and could be overwritten proceed y/N"%sys.argv[2])
        if y not in ["Y", "y", "yes", "Yes"]:
            sys.exit(0)
    
    img, tags = unetsl.data.loadImage(sys.argv[1])
    c1 = img[0]
    
    out = []
    
    skeleton = skimage.morphology.skeletonize((c1[0]>0))
    labels = labelSkeleton(skeleton)
    for s_no, slic in enumerate(c1):
        s_next = s_no + 1
        if s_next == len(c1):
            s_next= s_no -1
        nextSkeleton = skimage.morphology.skeletonize((c1[s_next]>0))
        nextLabels = labelSkeleton(nextSkeleton)
        jdex = jaccardIndexByRegion(labels, nextLabels)
        out.append(jdex)
        skeleton = nextSkeleton
        labels = nextLabels
    unetsl.data.saveImage("test.tif", numpy.array(out))
    

if __name__=="__main__":
    main()
    #developingJaccardIndex();
    #developingDistanceTransformWatershed();
