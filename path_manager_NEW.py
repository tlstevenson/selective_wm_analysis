#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:09:21 2026

@author: alex
"""
import os
import inference_config_setup as ics
import subprocess
import json

def vid_to_slp_raw(path):
    path_without_ext, ext = os.path.splitext(path)
    print(f"Mirrored path naked: {path_without_ext}")
    print(f"Final path slp: {path_without_ext[:-2]}_raw.slp")
    return f"{path_without_ext[:-2]}_raw.slp" #Replace _r.mp4 with _raw.slp

def slp_to_h5(path):
    path_without_ext, ext = os.path.splitext(path)
    print(f"Final path h5: {path[:path.rfind('_')]}_raw.h5")
    return f"{path[:path.rfind('_')]}_raw.h5"

def h5_raw_to_any(path, ending):
    return f"{path[:path.rfind('_')]}_{ending}.h5"

def get_mirrored_path_slp(parent_folder, child_file, new_folder):
    """
    Finds the mirrored path of a child file in a new destination folder.
    """
    
    try:
        # Extract the relative path (e.g., 'subfolder/file.txt')
        vid_rel_path_in_dir = os.path.relpath(child_file, parent_folder)
        
        # Append the relative path to the new destination folder
        mirrored_path = os.path.join(new_folder, vid_rel_path_in_dir)
        return vid_to_slp_raw(mirrored_path) #Replace _r.mp4 with _raw.h5
        
    except ValueError:
        # This triggers if the child file isn't actually inside the parent folder
        raise ValueError(f"The file '{child_file}' is not inside '{parent_folder}'")

def get_manual_path(analysis_folder, animal_num, video_file):
    filename = os.path.basename(video_file)
    return os.path.join(analysis_folder, str(animal_num), vid_to_slp_raw(filename)) #Replace _r.mp4 with _raw.h5

#%% DISK Configuration Getters
def get_conf(conf_name):
    config = ics.load_or_create_config()
    path = os.path.join(config["disk_files_path"], "DISK", "conf", conf_name)
    if os.path.exists(path):
        return path
    else:
        raise Exception(f"Path {path} of the config file does not exist.")

def get_create_dataset_conf():
    return get_conf("conf_create_dataset.yaml")
def get_proba_missing_files_conf():
    return get_conf("conf_proba_missing_files.yaml")
def get_impute_conf():
    return get_conf("conf_impute.yaml")
def get_missing_conf():
    return get_conf("conf_missing.yaml")
def get_test_conf():
    return get_conf("conf_test.yaml")
#%%
def get_disk_scripts_parent_dir():
    config = ics.load_or_create_config()
    path = os.path.join(config["disk_files_path"], "DISK")
    if os.path.exists(path):
        return path
    else:
        raise Exception(f"Path {path} of the config file does not exist.")
        
def get_disk_conda_path():
    config = ics.load_or_create_config()
    path = config["disk_env_path"]
    if os.path.exists(path):
        return path
    else:
        raise Exception(f"Path {path} of the config file does not exist.")
        
#%% UNTESTED Get environment path by name and conda executable
def get_conda_env_path(conda_path, env_name):
    """
    Returns the absolute path of a conda environment by name.
    
    :param conda_path: Path to the conda executable (e.g., '/miniconda3/bin/conda')
    :param env_name: The name of the environment to locate
    :return: Absolute path string or None if not found
    """
    try:
        # Run 'conda info --json' to get environment details
        result = subprocess.run(
            [conda_path, "info", "--json"],
            capture_output=True,
            text=True,
            check=True
        )
        
        data = json.loads(result.stdout)
        envs = data.get("envs", [])

        # Iterate through paths to find the one matching the env_name
        for env_path in envs:
            if os.path.basename(env_path) == env_name:
                return env_path
                
        return None

    except subprocess.CalledProcessError as e:
        print(f"Error executing conda: {e}")
        return None
    except json.JSONDecodeError:
        print("Failed to parse conda output.")
        return None