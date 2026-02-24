#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:38:32 2026

@author: alex
"""
import os
import json
import sys
from pathlib import Path

#%% Config default setup

def load_or_create_config(config_path="config.json"):
    """
    Loads the JSON config if it exists. 
    If not, creates a default template and halts execution.
    """
    user_home = Path.home();
    # Define the default paths you want in your template
    default_config = {
        "processed_vids_folder": os.path.join(user_home, "ReformattedVideos"),
        "analysis_folder": os.path.join(user_home, "Analysis"),
        "conda_env_path": "C:/path/to/sleap_env",
        "inference_script_path": os.path.join(Path(__file__).parent.resolve(), "inference_capsule_env.py"),
        "single_model_path": "C:/path/to/model",
        "centroid_model_path": "C:/path/to/model",
        "centered_model_path": "C:/path/to/model"
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
    manual_fields = ["conda_env_path", "single_model_path", "centroid_model_path", "centered_model_path"]
    defaults = ["C:/path/to/sleap_env", "C:/path/to/model", "C:/path/to/model", "C:/path/to/model"]
    with open(config_path, 'r') as file:
        config = json.load(file)
        for i in range(len(manual_fields)):
            if config[manual_fields[i]] == defaults[i]:
                raise Exception(f"Field {manual_fields[i]} is set to default value {defaults[i]}. Make sure that this and all other manual fields are valid.")
                sys.exit(0)
        return config