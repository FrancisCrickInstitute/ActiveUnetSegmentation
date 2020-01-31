#!/usr/bin/env python3

# -*- coding: utf-8 -*-

import click
import unetsl.comparisons
from matplotlib import pyplot
import numpy

def printRawValues(raw_values, output):
    N = 1000
    nms = [vdict[pt] for vdict in raw_values for pt in vdict]
    mn = min(nms)
    mx = max(nms)
    y = numpy.zeros((N, ), dtype="float")
    
    dp = (mx - mn)/N
    
    x = numpy.array([  dp * (i + 0.5) + mn for i in range(N) ])
    
    for pt in nms:
        i = int( (pt - mn)/dp)
        if i >= N:
            i = N-1;
        y[i] += 1
    
    for xi, yi in zip(x, y):
        output.write("%f\t%f\n"%(xi, yi))
    


def plotValues(raw_values):
    N = 1000
    nms = [vdict[pt] for vdict in raw_values for pt in vdict]
    mn = min(nms)
    mx = max(nms)
    y = numpy.zeros((N, ), dtype="float")
    
    dp = (mx - mn)/N
    
    x = numpy.array([  dp * (i + 0.5) + mn for i in range(N) ])
    
    for pt in nms:
        i = int( (pt - mn)/dp)
        if i >= N:
            i = N-1;
        y[i] += 1
    
    pyplot.plot(x, y)
    pyplot.show()
    

@click.command()
@click.argument("prediction", required=True, type=click.Path(exists=True))
@click.argument("truth", required=True, type=click.Path(exists=True))
@click.argument("save_labels", type=click.Path(), default=None)
def main(prediction, truth, save_labels=None):
    """
        
        Takes a prediction and the known ground truth and creates metrics for
        evaluation.
        
    """
    
    values, raw_values = unetsl.comparisons.compare(prediction,truth, save_labels, cutoffs=[0.5, 0.8, 0.95])
    
    #plotValues(raw_values)
    #print(" ".join([str(v) for v in values]))
    with open("log.txt", 'w', encoding="utf8") as f:
        printRawValues(raw_values, f)
    

if __name__=="__main__":
    main()
    