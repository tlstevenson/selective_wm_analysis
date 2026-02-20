#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:38:32 2026

@author: alex
"""
import os
import json
import sys

#%% Config default setup

def load_or_create_config(config_path="config.json"):
    """
    Loads the JSON config if it exists. 
    If not, creates a default template and halts execution.
    """
    # Define the default paths you want in your template
    default_config = {
        "processed_vids_folder": "C:/path/to/reformatted_videos",
        "analysis_folder": "C:/path/to/analysis_folder",
        "conda_env_path": "C:/Users/hankslab/miniforge3/envs/sleapUpdated",
        "inference_script_path": "C:/path/to/vid_to_inference_main.py",
        "sleap_python": "C:/Users/hankslab/miniforge3/envs/sleapUpdated/python.exe",
        "single_model_path": "NoFile",
        "centroid_model_path": "C:/path/to/centroid_model",
        "centered_model_path": "C:/path/to/centered_model"
    }

    # Check if the file already exists
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at '{config_path}'.")
        print("Creating a default template...")
        
        # Write the default dictionary to the file
        with open(config_path, 'w') as file:
            # indent=4 makes the JSON file readable with line breaks and spacing
            json.dump(default_config, file, indent=4)
            
        print("Template created! Please open 'config.json', update it with your actual paths, and run this script again.")
        # Exit the script so it doesn't try to run with dummy "C:/path/to/..." variables
        sys.exit(0)

    # If it does exist, load and return it normally
    with open(config_path, 'r') as file:
        return json.load(file)