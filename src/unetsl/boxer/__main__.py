# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import .boxer

@click.command()
def boxer():
    pass

@boxer.command("create")
def create():
    boxer = boxer.BoxerModel();
    boxer.save("boxer.h5")
    
@boxer.command("train")
def train():
    boxer = boxer.BoxerModel();
    boxer.load("boxer.h5");
    boxer.train()
    boxer.save("boxer.h5")
    
if __name__=="__main__":
    main()
