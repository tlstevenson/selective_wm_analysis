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
#%%Define traversal function
from pathlib import Path

def get_file_paths(directory_path):
    """Returns a list of strings containing the paths of all files in a directory."""
    path_obj = Path(directory_path)
    
    # .is_file() ensures we don't include subdirectories in the list
    return [str(file) for file in path_obj.iterdir() if file.is_file()]

# Example usage:
# files = get_file_paths("./my_folder")
#%%Select new videos
vid_folders = [r"E:/Tanner_Vids/ReformattedVideos/199",
               r"E:/Tanner_Vids/ReformattedVideos/274",
               r"E:/Tanner_Vids/ReformattedVideos/400",
               r"E:/Tanner_Vids/ReformattedVideos/402"]
curr_vids = []
for vid_folder in vid_folders:
    curr_vids = curr_vids + get_file_paths(vid_folder)
#%%Format all videos
curr_format_vids = []
for vid in curr_vids:
    new_format_vid_path = svr.process_video(vid, config["processed_vids_folder"])
    if new_format_vid_path != None:
        curr_format_vids.append(new_format_vid_path)
#%% Inference Selections (with reformatting)
#Set up the write paths
write_paths = []
for i in range(len(curr_format_vids)):
    write_paths.append(pm.get_mirrored_path_slp(config["processed_vids_folder"], curr_format_vids[i], config["analysis_folder"]))
print(write_paths)
#%% Inference Selections (without reformatting)
#Set up the write paths
#TODO: Remove hardcode path
write_paths = []
for i in range(len(curr_vids)):
    if "_r" in curr_vids[i]:
        write_paths.append(pm.get_mirrored_path_slp(r"E:/Tanner_Vids/ReformattedVideos", curr_vids[i], config["analysis_folder"]))
    else:
        write_paths.append(pm.get_mirrored_path_slp_raw(r"E:/Tanner_Vids/ReformattedVideos", curr_vids[i], config["analysis_folder"]))
    print(write_paths[-1])
#%% Command to run inference on all files
#slp_launcher.run_inference(curr_format_vids, write_paths)
slp_launcher.run_inference(curr_vids, write_paths)

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
#%% create the skeleton
ddc.create_skeleton(dataset_name, analysis_files)

#%%create proba missing_files
ddc.run_proba_missing_files(dataset_name, config["disk_env_path"])

#%%change config and run model training
ddc.update_training_config(pm.get_missing_conf(), dataset_name)
#%%
ddc.train_disk_model()

#%%
dataset_name = "Movie1_2_199"
checkpoints = "C:/Users/hankslab/repos/DISK/DISK/models/Movie1_2_199" #TODO: Remove hardcode
ddc.modify_test_config(pm.get_test_conf(), dataset_name, checkpoints)
#%%
ddc.run_test_fillmissing()

#%%
checkpoint = checkpoints
ddc.modify_impute_config(pm.get_impute_conf(), dataset_name, checkpoint)
#%%
ddc.run_impute_dataset()