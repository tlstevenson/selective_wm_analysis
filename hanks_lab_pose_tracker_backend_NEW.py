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
import basic_preprocessing_NEW as bpre
import analysis_file_statistics_NEW as afs

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

#FIXME: conda_env_path needs to be renamed to sleap_env_path everywhere
#%%Define traversal function
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
#%% Inference Selections (without reformatting)
#Set up the write paths
#TODO: Remove hardcode path
write_paths = []
for i in range(len(curr_vids)):
    write_paths.append(pm.get_mirrored_path_slp(r"E:/Tanner_Vids/ReformattedVideos", curr_vids[i], config["analysis_folder"], config["single_model_path"])) #TODO: Fix only works for single
    print(write_paths[-1])
#%% Command to run inference on all files
#slp_launcher.run_inference(curr_format_vids, write_paths)
slp_launcher.run_inference(curr_vids, write_paths)

#%%Temporary MAC only for directly using sleap-io to convert
analysis_folders = ["/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/199",
               "/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/274",
               "/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/400",
               "/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/402"]
write_paths = []
for folder in analysis_folders:
    files = get_file_paths(folder)
    for file in files:
        write_paths.append(file)

import sleap_io as sio
def convert_slp_to_analysis_h5(slp_path):
    """
    Converts a SLEAP prediction file with no tracks into a 
    valid Analysis HDF5 file for DISK.
    """
    if not os.path.exists(slp_path):
        raise Warning(f"The file {slp_path} does not exist or could not be found. Track assignment aborted.")
        return
    
    #TEMPORARY: Dont reformat if it already exists as an analysis file
    output_path = pm.slp_to_h5(slp_path)
    if os.path.exists(output_path):
        return str(output_path)
    
    # 1. Load the labels (very fast with sleap-io)
    labels = sio.load_slp(slp_path)
    
    # 2. Create a dummy track if none exist
    if len(labels.tracks) == 0:
        single_track = sio.Track(name="track_0")
        labels.tracks.append(single_track)
    else:
        single_track = labels.tracks[0]

    # 3. Force every instance into this track
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.track = single_track

    # 5. Save using the analysis-specific function
    # all_frames=True is vital for DISK to maintain the time-series continuity
    sio.io.main.save_analysis_h5(
        labels=labels, 
        filename=str(output_path), 
        all_frames=True
    )
    
    print(f"Successfully created: {output_path}")
    return str(output_path)

for file in write_paths:
    convert_slp_to_analysis_h5(file, pm.slp_to_h5(file))
#%%Permanent WINDOWS way to reformat videos

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
    
#%%View raw data
# Define your groups by folder
h5_folder_paths = [
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/199/h5", "199"), 
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/274/h5", "274"), 
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/400/h5", "400"),
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/402/h5", "402")
]

# Dynamically build the configuration list
VIDEO_CONFIG = []
for folder in h5_folder_paths:
    for file in get_file_paths(folder[0]):
        # Note: mapped to "filepath" to match downstream variables
        VIDEO_CONFIG.append({"filepath": file, "group": folder[1]})
    
print("Extracting data and building DataFrame...")
df, extracted_node_names = afs.process_all_videos(VIDEO_CONFIG)

print("Calculating standard statistics...")
stats_df = afs.calculate_stats(df)

csv_filename = "node_statistics_summary.csv"
stats_df.round(2).to_csv(csv_filename, index=False)
print(f"\nSaved detailed statistical table to: {csv_filename}")

print("Calculating outlier counts...")
outlier_df = afs.calculate_outlier_counts(df)

print("Generating Box Plots...")
afs.plot_boxplots_separate_figures(df)

#print("Generating Violin Plots...")
#afs.plot_violinplots_separate_figures(df)

print("Generating Outlier Bar Graphs...")
afs.plot_outlier_bar_graphs(outlier_df)
#%%Basic preprocessing steps
basic_pre_files = []
for file in analysis_files:
    #Add new path to basic files
    basic_pre_files.append(bpre.interpolate_and_save_h5(file, max_gap_length=15)[-1]) 
#%%View preprocessed data
# Dynamically build the configuration list
VIDEO_CONFIG = []
for folder in h5_folder_paths:
    for file in get_file_paths(folder[0]):
        # Note: mapped to "filepath" to match downstream variables
        VIDEO_CONFIG.append({"filepath": file, "group": folder[1]})
    
print("Extracting data and building DataFrame...")
df, extracted_node_names = afs.process_all_videos(VIDEO_CONFIG)

print("Calculating standard statistics...")
stats_df = afs.calculate_stats(df)

csv_filename = "node_statistics_summary.csv"
stats_df.round(2).to_csv(csv_filename, index=False)
print(f"\nSaved detailed statistical table to: {csv_filename}")

print("Calculating outlier counts...")
outlier_df = afs.calculate_outlier_counts(df)

print("Generating Box Plots...")
afs.plot_boxplots_separate_figures(df)

#print("Generating Violin Plots...")
#afs.plot_violinplots_separate_figures(df)

print("Generating Outlier Bar Graphs...")
afs.plot_outlier_bar_graphs(outlier_df)
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