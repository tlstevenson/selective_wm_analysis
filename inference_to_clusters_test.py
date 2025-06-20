#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:47:07 2025

@author: alex
"""
import inference_to_clusters_lib as itcl
import numpy as np

def process_hdf5_data_test():
    return

def PlotPortsTest():
    return

def AngleTest():
    print(f"Angle between axes is {itcl.Angle([0,1], [1,0])}")
    print(f"Angle between {[1,1]} and  {[0,1]} is {itcl.Angle([1,1], [0,1])}")
    
AngleTest()