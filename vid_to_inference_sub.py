#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:08:25 2025

@author: alex
"""
#import json
import sys
import init
import json
from sys_neuro_tools import sleap_utils_env
sleap_settings_path = sys.argv[1]
with open(sleap_settings_path, "r") as file:
    sleap_settings = json.load(file)
    vid_path = sleap_settings["vid_path"]
    centroid_path = sleap_settings["centroid_path"]
    center_path = sleap_settings["center_path"]
    write_dir = sleap_settings["write_dir"]
    sleap_utils_env.RunInference(vid_path, centroid_path, center_path, write_dir)