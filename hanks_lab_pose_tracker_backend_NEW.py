#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:53:19 2026

@author: alex
"""
#%% Import statements
from pathlib import Path
import os

import inference_config_setup as ics
import sleap_vid_reformat as svr
import inference_launcher as slp_launcher
#%% Create a config file if necessary or extract old one
home = Path.home()
config_path = os.path.join(home, "hanks_pose_config")
config = ics.load_or_create_config(config_path)

#processed_vids_folder: any folder
#analysis_folder: any folder
#conda_env_path: use conda env list to find path
#inference_script_path: file path of inference_capsule_env.py
#sleap_python: Use conda activate sleap -> which python

#%%Select new videos
curr_vids = ["/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0001_r.mp4", 
             "/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0002_r.mp4"]
#Pop up a tkinter window to select all these until satistfied

#Format all videos
curr_format_vids = []
for vid in curr_vids:
    new_format_vid_path = svr.process_video(vid, config["processed_vids_folder"])
    if new_format_vid_path != None:
        curr_format_vids.append()

#%% Inference Selections
model_path = "PLACEHOLDER_get_from_(curr_sleap_model)"
#Here a tkinter window would have the option of changing to new one + updating json
#Here a tkinter button would trigger the start instead of having it start automatically
slp_launcher.run_inference(config_path, curr_format_vids)

