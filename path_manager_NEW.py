#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:09:21 2026

@author: alex
"""
import os
import inference_config_setup as ics

def vid_to_h5_raw(path):
    path_without_ext, ext = os.path.splitext(path)
    print(f"Mirrored path naked: {path_without_ext}")
    print(f"Final path h5: {path_without_ext[:-2]}_raw.h5")
    return f"{path_without_ext[:-2]}_raw.h5" #Replace _r.mp4 with _raw.h5

def h5_raw_to_any(path, ending):
    return f"{path[:path.rfind('_')]}_{ending}.h5"

def get_mirrored_path(parent_folder, child_file, new_folder):
    """
    Finds the mirrored path of a child file in a new destination folder.
    """
    
    try:
        # Extract the relative path (e.g., 'subfolder/file.txt')
        vid_rel_path_in_dir = os.path.relpath(child_file, parent_folder)
        print(f"Relative path: {vid_rel_path_in_dir}")
        
        # Append the relative path to the new destination folder
        mirrored_path = os.path.join(new_folder, vid_rel_path_in_dir)
        print(f"Mirrored vid path: {mirrored_path}")
        return vid_to_h5_raw(mirrored_path) #Replace _r.mp4 with _raw.h5
        
    except ValueError:
        # This triggers if the child file isn't actually inside the parent folder
        raise ValueError(f"The file '{child_file}' is not inside '{parent_folder}'")

def get_manual_path(analysis_folder, animal_num, video_file):
    filename = os.path.basename(video_file)
    return os.path.join(analysis_folder, str(animal_num), vid_to_h5_raw(filename)) #Replace _r.mp4 with _raw.h5

def get_conf(conf_name):
    config = ics.load_or_create_config()
    return os.path.join(config["disk_env_path"], "DISK", "conf", conf_name)

def get_create_dataset_conf():
    get_conf("conf_create_dataset.yaml")
def get_proba_missing_files_conf(disk_env_path):
    get_conf("conf_proba_missing_files.yaml")
def get_impute_conf(disk_env_path):
    get_conf("conf_impute.yaml")
def get_missing_conf(disk_env_path):
    get_conf("conf_missing.yaml")
def get_test_conf(disk_env_path):
    get_conf("conf_test.yaml")