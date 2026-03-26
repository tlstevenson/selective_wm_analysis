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

def vid_to_slp(path):
    #.mp4 -> .proj.slp
    path_without_ext, ext = os.path.splitext(path)
    return f"{path_without_ext}.proj.slp"

def slp_to_h5(path):
    #.proj.slp -> .raw.h5
    path_without_ext, ext = os.path.splitext(path)
    path_without_type, my_type = os.path.splitext(path_without_ext)
    return f"{path_without_type}.raw.h5"

def h5_raw_to_interpol(path, interpol_info="15", interpol_type="quad"):
    #.raw.h5 -> e.g. .15.quad.h5
    path_without_ext, ext = os.path.splitext(path)
    path_without_type, my_type = os.path.splitext(path_without_ext)
    return f"{path_without_type}.{str(interpol_info)}.{interpol_type}.h5"

def h5_to_disk(path):
    #.raw.h5 -> .raw.disk.h5
    path_without_ext, ext = os.path.splitext(path)
    return f"{path_without_ext}.disk.h5"

def get_mirrored_path_slp(parent_folder, child_file, new_folder, model_path):
    """
    Finds the mirrored path of a child file in a new destination folder.
    """
    
    try:
        # Extract the relative path (e.g., 'subfolder/file.txt')
        vid_rel_path_in_dir = os.path.relpath(child_file, parent_folder)
        
        # Append the relative path to the new destination folder
        model_name = os.path.split(model_path)[-1]
        mirrored_path = os.path.join(new_folder, model_name, vid_rel_path_in_dir)
        return vid_to_slp(mirrored_path)
        
    except ValueError:
        # This triggers if the child file isn't actually inside the parent folder
        raise ValueError(f"The file '{child_file}' is not inside '{parent_folder}'")

def get_manual_path(analysis_folder, animal_num, video_file, model_path):
    filename = os.path.basename(video_file)
    model_name = os.path.split(model_path)[-1]
    return os.path.join(analysis_folder, model_name, str(animal_num), vid_to_slp(filename)) 

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
    
def convert_old_path_to_new(old_path):
    """
    Converts a path string from old format (e.g., file_raw.slp)
    to new format (e.g., file.raw.slp).
    """
    dir_name, file_name = os.path.split(old_path)
    name_no_ext, ext = os.path.splitext(file_name)

    # Find the last underscore in the filename 
    last_underscore_idx = name_no_ext.rfind('_')

    if last_underscore_idx != -1:
        # Replace ONLY the last underscore with a dot
        new_name_no_ext = name_no_ext[:last_underscore_idx] + '.' + name_no_ext[last_underscore_idx+1:]
        new_file_name = new_name_no_ext + ext
        return os.path.join(dir_name, new_file_name)
    
    return old_path # Returns the path as-is if no underscore is found

def migrate_directory_to_new_format(target_directory, dry_run=True):
    """
    Walks through a directory and renames all files matching the 
    old _suffix.ext format to the new .suffix.ext format.
    
    Set dry_run=False to actually execute the file renaming.
    """
    print(f"Starting migration in: {target_directory} (Dry Run: {dry_run})")
    
    for root, dirs, files in os.walk(target_directory):
        for file in files:
            old_filepath = os.path.join(root, file)
            new_filepath = convert_old_path_to_new(old_filepath)
            
            # If the converter changed the name, we have a match
            if old_filepath != new_filepath:
                if dry_run:
                    print(f"[DRY RUN] Would rename:\n  {old_filepath}\n  -> {new_filepath}\n")
                else:
                    try:
                        os.rename(old_filepath, new_filepath)
                        print(f"Renamed: {file} -> {os.path.basename(new_filepath)}")
                    except Exception as e:
                        print(f"Error renaming {old_filepath}: {e}")