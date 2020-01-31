#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from unetsl.data import getCropStride
import math

dt = 2*math.pi/100
stride = [1, 3, 64, 64]

for i in range(10000):
    s = getCropStride(stride, dt*i)
    for j in range(len(s)):
        if s[j]<stride[j]:
            raise Exception("Illegal value returned for angle %s resulting in %s stride"%(i*dt, s))
