#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import unetsl.data
import unetsl.comparisons
import numpy


def plotValues(raw_values, plotname):
    from matplotlib import pyplot
    
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
    
    fig = pyplot.figure(0)
    pyplot.plot(x, y)
    fig.savefig(plotname)
    


def evaluateSkeleton(skeleton):
    return unetsl.comparisons.evaluateSkeleton(skeleton, cutoffs=[0.2, 0.4, 0.6])

@click.command()
@click.argument("prediction_type", required=True)
@click.argument("prediction", required=True, type=click.Path(exists=True))
def main(prediction_type, prediction):
    """
        Evaluates the quality of prediction
    """
    
    if prediction_type=='s':
        values, raw_values = evaluateSkeleton(prediction);
        plotValues(raw_values, "plots/%s"%prediction.replace(".tif", "-plot.png"))
        print(" ".join("%s"%v for v in values), prediction)
    else:
        print("only skeleton at the moment.")
        
if __name__=="__main__":
    main()