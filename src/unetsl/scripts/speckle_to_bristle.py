#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pathlib
import sys

import unetsl.data
import skimage.draw

import random

SPECKLE_TAG="%start speckle%"
WIDTH=64

import numpy

class Box:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
    def contains(self, box):
        criteria = [
                box.x>self.x, 
                box.y>self.y, 
                box.x + box.w < self.x + self.w,
                box.y + box.h < self.y + self.h]
        return all(criteria)
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
class Speckle:
    def __init__(self):
        self.track = {}
        self.r = 8;
    def put(self, i, x, y):
        self.track[i] = Point(x,y)
    
    def overlaps(self, frame, box):
        if frame in self.track:
            pt = self.track[frame]
            lx = pt.x - self.r
            ly = pt.y - self.r
            hx = pt.x + self.r
            hy = pt.y + self.r
            limits = (
                    lx > box.x + box.w,
                    hx<box.x, 
                    ly>box.y + box.h, 
                    hy<box.y
                )
            if any(limits):
                return False
            return True
        return False
    
    def pointGenerator(self):
        for i in self.track:
            yield (i, self.track[i])
        
    
def loadSpeckles(speckle_file):
    with open(speckle_file) as sf:
        speckles = []
        speckle = {}
        for line in sf:
            if line[0]=="#":
                if SPECKLE_TAG in line:
                    speckle = Speckle()
                    speckles.append(speckle);
                continue
            vs = line.split()
            
            speckle.put(int(vs[2])-1, float(vs[0]), float(vs[1]))
        
        return speckles

if __name__=="__main__":
    print("usage: speckle_to_bristle.py speckles.csv source.tif skeleton.tif")
    speckles = loadSpeckles(sys.argv[1])
    
    original, _ = unetsl.data.loadImage(sys.argv[2])
    skeleton, _ = unetsl.data.loadImage(sys.argv[3])
    skeleton = (skeleton>0) * 1
    frames = original.shape[1]
    
    
    skeletons = []
    originals = []
    
    
    image_bounds = Box(0, 0, original.shape[-1], original.shape[-2])
    
    for speckle in speckles:
        
        #relabel skeleton.
        for frame, pt in speckle.pointGenerator():
            skel = skeleton[0, frame]    
            cc, rr = skimage.draw.circle(pt.y, pt.x, speckle.r, skel.shape)
            skel[cc, rr] = 2
            cc2, rr2 = skimage.draw.circle_perimeter(int(pt.y),int(pt.x), int(speckle.r), shape = skel.shape)
            skel[cc2, rr2] = 1
        
        
        for frame, pt in speckle.pointGenerator():
            skel = skeleton[0, frame]  
            
            #original location of the top left corner.
            x0 = int(pt.x - WIDTH//2)
            y0 = int(pt.y - WIDTH//2)

            
            for i in range(1):
                for j in range(1):
                    x = x0
                    y = y0
                    snap = Box(x, y, WIDTH, WIDTH)
                    
                    
                    if image_bounds.contains(snap):
                        sk = skel[y:y + WIDTH, x:x + WIDTH]
                        og = original[0, frame, y:y + WIDTH, x:x + WIDTH]
                        skeletons.append(sk)
                        originals.append(og)
    total = 0
    blank = numpy.zeros((WIDTH, WIDTH), dtype=skeletons[0].dtype)
    
    
    non_bristle = []
    while len(non_bristle) < len(skeletons):
        frame = int(random.random()*frames)
        looking = True
        while looking:
            x = int(random.random()*(image_bounds.w - WIDTH))
            y = int(random.random()*(image_bounds.h - WIDTH))
            b = Box(x, y, WIDTH, WIDTH);
            splat = any(speckle.overlaps(frame, b) for speckle in speckles)
            if not splat and image_bounds.contains(b) :
                looking = False
        nb = original[0, frame, y:y+WIDTH, x:x+WIDTH]
        non_bristle.append(nb)
        
    
    unetsl.data.saveImage("bristles.tif", numpy.array([originals]))
    unetsl.data.saveImage("non-bristles.tif", numpy.array([non_bristle]))
    #unetsl.data.saveImage("bristles-marked-skeleton.tif", numpy.array(skeletons, dtype="uint8"))
    