#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:08:25 2025

@author: alex
"""
import vid_to_inference_lib
import json
import sys


#NOT USE ABSOLUTE PATH
#json_path = "/Users/alex/Documents/HanksLabGithub/selective_wm_analysis/inference_paths.json"
json_path = sys.argv[1]
with open(json_path, "r") as file:
    data = json.load(file)
    vid_path = data["vid_path"]
    centroid_path = data["centroid_path"]
    center_path = data["center_path"]
    write_dir = data["write_dir"]
    print(write_dir)
    vid_to_inference_lib.RunInference(vid_path, centroid_path, center_path, write_dir)