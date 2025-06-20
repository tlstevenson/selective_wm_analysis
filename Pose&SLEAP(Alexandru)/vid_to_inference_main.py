#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 13:23:22 2025

@author: alex
"""
import vid_to_inference_lib as vtfl

vid_path = vtfl.GetFile()
while vid_path == None:
    vid_path = vtfl.GetFile()
    
centroid_path = vtfl.GetDirectory()
while centroid_path == None:
    centroid_path = vtfl.GetDirectory()
    
centered_path = vtfl.GetDirectory()
while centered_path == None:
    centered_path = vtfl.GetDirectory()
    
write_dir = vtfl.GetDirectory()
while write_dir == None:
    write_dir = vtfl.GetDirectory()
    
vtfl.RunInference(vid_path, centroid_path, centered_path, write_dir + "/InferenceResults(Selected).hdf5")