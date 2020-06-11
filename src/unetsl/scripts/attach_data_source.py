#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import unetsl.data
import unetsl.config
from unetsl.cli_interface import getFilePrompt

def getNumberList(inp):
    print(inp)
    values = inp.split();
    print(values)
    x = []
    
    for v in values:
        try:
            d = float(v.rstrip(","))
            x.append(d)
        except:
            print("cannot convert %s"%v)
    return x


def pairedDirectoryDataSource():
    input_folder = getFilePrompt("enter images directory: ")
    label_folder = getFilePrompt("enter labels directory: ")
    conf = {unetsl.data.SOURCE_TYPE : unetsl.data.PAIRED_DIRECTORY}
    conf[unetsl.data.INPUT_FOLDERS] = [input_folder]
    conf[unetsl.data.LABEL_FOLDERS] = [label_folder]
    entered = input("enter a list of angles, leave blank if none")
    conf[unetsl.data.ROTATIONS] = getNumberList(entered)
    
    conf[unetsl.data.LABELLER] = getLabeller()
    
    return conf

def getLabeller():
    labeller_set = False
    
    while not labeller_set:
        print("labellers: ")
        labellers = [l for l in unetsl.data.labeller_map]
        for i, lbr in enumerate(labellers):
            print("  - ", i, ": ", lbr)
        txt = input("select labeller: ")
        try:
            z = int(txt)
            if z < len(labellers):
                labeller = labellers[z]
                labeller_set = True
            else:
                print("enter a number between 0 and %d"%(len(labellers)-1), " to select a labeller")
        except:
            print("enter a number between 0 and %d"%(len(labellers)-1), " to select a labeller")
    return labeller

def weightedDirectoryDataSource():
    input_folder = getFilePrompt("enter images directory: ")
    label_folder = getFilePrompt("enter labels directory: ")
    weight_folder = getFilePrompt("enter weights directory: ")
    conf = {unetsl.data.SOURCE_TYPE : unetsl.data.WEIGHTED_DIRECTORY}
    conf[unetsl.data.INPUT_FOLDERS] = [input_folder]
    conf[unetsl.data.LABEL_FOLDERS] = [label_folder]
    conf[unetsl.data.WEIGHTS_FOLDERS] = [weight_folder]
    labelr = getLabeller()
    conf[unetsl.data.LABELLER] = labelr
    
    return conf



def removeMenu(config):
    removing = True
    sources = config[unetsl.DATA_SOURCES]
    while removing:
        print("enter index of source to remove: ")
        
        for i, source in enumerate(sources):
            print("%s. %s"%(i, source))
        t = input("...")
        try:
            index = int(t)
            if index < len(sources):
                print("removing: %s, %s"%(index, sources[i]))
                sources.remove(sources[index])
        except:
            removing = False
            
        

    

def mainMenu(config):
    print("enter type: ")
    print("1. paired directory")
    print("2. weighted directories")
    print("3. remove existing sources.")
    print("<anything else>. stop")
    t = input("...");
    if(len(t)==0):
        return -1
    if t[0] == '1':
        config[unetsl.DATA_SOURCES].append(pairedDirectoryDataSource())
        return 0
    elif t[0] == '2':
        config[unetsl.DATA_SOURCES].append(weightedDirectoryDataSource())
        return 0
    elif t[0] == '3':
        removeMenu(config)
    else:
        return -1

def manageConfig(config):
    adding = True
    while adding:
        ret = mainMenu(config)
        if ret==-1:
            adding=False
    
    return config
    
def main():
    
    if len(sys.argv)<3 and '-c' not in sys.argv:
        print("usage: attach_data_source -c model_config.json")
        sys.exit(0)
    cf = sys.argv[ sys.argv.index('-c') + 1]
    config = unetsl.config.getDefaultConfigurationTool()
    config.load(cf)
    existing = config.get(unetsl.DATA_SOURCES, [])
    config[unetsl.DATA_SOURCES] = existing
    adding = True
    while adding:
        ret = mainMenu(config)
        if ret==-1:
            adding=False
    s = input("save y/[n]?")
    if s in ["y", "Y", "yes", "Yes", "YES"]:
        config.save(cf)

if __name__=="__main__":
    main()
        
    