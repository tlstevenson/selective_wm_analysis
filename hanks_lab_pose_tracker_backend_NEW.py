#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 14:53:19 2026

@author: alex
"""
#%% Import statements
from pathlib import Path
import os
import init

import inference_config_setup as ics
import sleap_vid_reformat as svr
import inference_launcher as slp_launcher
import path_manager_NEW as pm
import disk_dataset_creator_NEW as ddc

#Imports that should be moved out
import subprocess
import sys
#%% Create a config file if necessary or extract old one
config = ics.load_or_create_config()

#processed_vids_folder: any folder
#analysis_folder: any folder
#conda_env_path: use conda env list to find path
#inference_script_path: file path of inference_capsule_env.py
#single_model_path: The path to the folder containing the full sleap model
#centroid_model_path: Same as above (centroid)
#centered_model_path: Same as above (centered_instance)
#disk_env_path: The location of the DISK conda environment
#disk_files_path: The location of the DISK repo

#TODO: conda_env_path needs to be renamed to sleap_env_path everywhere
#%%Select new videos

curr_vids = [r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0001.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0002.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0003.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0004.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0005.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0006.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0007.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0008.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0009.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0010.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0011.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0012.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0013.mp4",
             r"C:\\Users\hankslab\SLEAP\AlexSleap\CableVideos\Videos\Original\199\mov_0014.mp4"]
#Pop up a tkinter window to select all these until satistfied

#Format all videos
curr_format_vids = []
for vid in curr_vids:
    new_format_vid_path = svr.process_video(vid, config["processed_vids_folder"])
    if new_format_vid_path != None:
        curr_format_vids.append(new_format_vid_path)
        
#%%Should get to this point if FFmpeg is installed"
"""curr_format_vids = ["/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0001_r.mp4", 
                    "/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0002_r.mp4"]
print(curr_format_vids)"""
#%% Inference Selections
#Set up the write paths
#Here a tkinter window would have the option of changing to new one + updating json
#Here a tkinter button would trigger the start instead of having it start automatically
write_paths = []
for i in range(len(curr_format_vids)):
    write_paths.append(pm.get_mirrored_path_slp(config["processed_vids_folder"], curr_format_vids[i], config["analysis_folder"]))
    print(write_paths[i])
#%% Command to run inference on all files
slp_launcher.run_inference(curr_format_vids, write_paths)

#%%
analysis_files = [] #Store all valid reformatted analysis files
#Convert slp files to analysis h5 files in the same folder (TODO: Consider moving this out of here)
for file in write_paths:
    command = ["conda.bat", "run", "--no-capture-output", "-p", config["sleap_io_env_path"], 
               "python", os.path.join(os.path.dirname(os.path.abspath(__file__)),"assign_track_NEW.py"), file, pm.slp_to_h5(file)]
    #command = ["conda.bat", "run", "--no-capture-output", "-p", config["sleap_io_env_path"], 
    #           "conda", "list", "sleap-io"]
    try:
        process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True)
        if process.returncode == 0:
            print("=" * 50)
            print("Inference completed successfully!")
            analysis_files.append(pm.slp_to_h5(file))
        else:
            print("=" * 50)
            print(f"Inference failed with exit code {process.returncode}.")
            print(process.stdout)
            print(process.stderr)
    except Exception as e:
        print(f"Failed to launch subprocess: {e}")
        
#%%temporary: Test if the formatting worked
import h5py
with h5py.File('C:/Users/hankslab/Analysis/199/mov_0002_raw.h5', 'r') as f:
    print("Keys in H5 file:", list(f.keys()))

#%% Create dataset with analysis files
dataset_name = "Movie1_2_199"
print(pm.get_create_dataset_conf())
ddc.create_dataset(pm.get_create_dataset_conf(), dataset_name, analysis_files, config["disk_env_path"])
#%%
ddc.create_skeleton(dataset_name, analysis_files)