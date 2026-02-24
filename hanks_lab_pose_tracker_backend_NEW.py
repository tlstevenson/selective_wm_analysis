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
#%% Create a config file if necessary or extract old one
home = Path.home()
config_path = os.path.join(home, "hanks_pose_config")
config = ics.load_or_create_config(config_path)

#processed_vids_folder: any folder
#analysis_folder: any folder
#conda_env_path: use conda env list to find path
#inference_script_path: file path of inference_capsule_env.py
#single_model_path: The path to the folder containing the full sleap model
#centroid_model_path: Same as above (centroid)
#centered_model_path: Same as above (centered_instance)

#%% Generate the write path function
def get_mirrored_path(parent_folder, child_file, new_folder):
    """
    Finds the mirrored path of a child file in a new destination folder.
    """
    # Convert string paths to Path objects
    parent = Path(parent_folder)
    child = Path(child_file)
    new_dest = Path(new_folder)
    
    try:
        # Extract the relative path (e.g., 'subfolder/file.txt')
        relative_path = child.relative_to(parent)
        
        # Append the relative path to the new destination folder
        mirrored_path = new_dest / relative_path
        
        path_without_ext, ext = os.path.splitext(mirrored_path)
        mirrored_path = f"{path_without_ext[:-2]}_raw.h5" #Replace _r.mp4 with _raw.h5
        
        return mirrored_path
        
    except ValueError:
        # This triggers if the child file isn't actually inside the parent folder
        raise ValueError(f"The file '{child}' is not inside '{parent}'")

def get_manual_path(analysis_folder, animal_num, video_file):
    filename = os.path.basename(video_file)
    path_without_ext, ext = os.path.splitext(filename)
    return os.path.join(config["analysis_folder"], str(animal_num), f"{path_without_ext[:-2]}_raw.h5") #Replace _r.mp4 with _raw.h5

#Script to check that the current places exist and set or create files that don't
#What if you don't have rat number? Allow it to give it a name
#What if the rat number/name doesn't exist in processed videos folder (Auto makedir)

#What if you want to override file or parent directory name?


#%%Select new videos
curr_vids = ["/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0001.mp4", 
             "/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0002.mp4"]
#Pop up a tkinter window to select all these until satistfied

#Format all videos
curr_format_vids = []
for vid in curr_vids:
    new_format_vid_path = svr.process_video(vid, config["processed_vids_folder"])
    if new_format_vid_path != None:
        curr_format_vids.append(new_format_vid_path)
        
#%%Should get to this point if FFmpeg is installed
curr_format_vids = ["/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0001_r.mp4", 
                    "/Users/alex/Documents/HanksLab/HanksLabVideos/ReformattedVideos/199/mov_0002_r.mp4"]
print(curr_format_vids)

#%% Inference Selections
#Here a tkinter window would have the option of changing to new one + updating json
#Here a tkinter button would trigger the start instead of having it start automatically
write_paths = []
for i in range(len(curr_format_vids)):
    write_paths.append(get_manual_path(config["analysis_folder"], 199, curr_format_vids[i]))
    print(write_paths[i])
slp_launcher.run_inference(config_path, curr_format_vids, write_paths)