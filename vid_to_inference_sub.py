#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 13:08:25 2025

@author: alex
"""
#import json
#try:
"""try:
    import sys
except:
    sys.exit(2)
import init
try:
    import json
except:
    sys.exit(4)
try:
    from sys_neuro_tools import sleap_utils_env
except:
    sys.exit(5)
#except:
#    sys.exit(2)
try:
    sleap_settings_path = sys.argv[1]
except:
    sys.exit(6)
try:
    with open(sleap_settings_path, "r") as file:
        sleap_settings = json.load(file)
        vid_path = sleap_settings["vid_path"]
        centroid_path = sleap_settings["centroid_path"]
        center_path = sleap_settings["center_path"]
        write_dir = sleap_settings["write_dir"]
        sleap_utils_env.RunInference(vid_path, centroid_path, center_path, write_dir)
except:
    sys.exit(7)"""
"""import json
from sys_neuro_tools import sleap_utils_env
import subprocess
#import os
sleap_settings_path = sys.argv[1]
with open(sleap_settings_path, "r") as file:
    sleap_settings = json.load(file)
    vid_path = sleap_settings["vid_path"]
    #UNTESTED
    #new_name = vid_path[:-4] + "_reformat.mp4"
    #Reformats the given video
    #if not os.path.isfile(new_name):
        #command = f'ffmpeg -y -i {vid_path} -c:v libx264 -pix_fmt yuv420p -preset superfast -crf 23 {new_name}'
        #subprocess.run(command)
    centroid_path = sleap_settings["centroid_path"]
    center_path = sleap_settings["center_path"]
    write_dir = sleap_settings["write_dir"]
    sleap_utils_env.RunInference(vid_path, centroid_path, center_path, write_dir)
    #sleap_utils_env.RunInference(new_name, centroid_path, center_path, write_dir)"""
