# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:02:26 2026

@author: cns-th-lab
"""

import os
import sys
from ruamel.yaml import YAML
import subprocess

def create_dataset(config_path, dataset_name, h5_files, conda_env_path):
    #Edit config file
    #Run python script from the conda environment
    update_disk_config(config_path, dataset_name, h5_files)
    programs_folder = os.path.join(conda_env_path, "DISK", )
    command = ["conda", "run", "--no-capture-output", "-p", conda_env_path, 
               "python", "create_dataset.py"]
    try:
        # Popen streams the output line-by-line
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=True, cwd=programs_folder) as process:
            # Iterate through the output as it is generated and print to console
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                
        # Ensure the process is fully complete before checking the exit code
        process.wait()
                
        if process.returncode == 0:
            print("=" * 50)
            print("Inference completed successfully!")
            return True
        else:
            print("=" * 50)
            print(f"Inference failed with exit code {process.returncode}.")
            return False
    except Exception as e:
        print(f"Failed to launch subprocess: {e}")
        return False

def update_disk_config(config_path, dataset_name, h5_files):
    """
    Updates a DISK .yaml template with specific parameters.
    
    Args:
        config_path (str): Path to the existing .yaml file.
        dataset_name (str): The name for the created dataset.
        h5_files (list): A list of strings containing paths to .h5 analysis files.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    # 1. Load the existing template
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with open(config_path, 'r') as f:
        config_data = yaml.load(f)

    # 2. Update values based on your requirements
    config_data['dataset_name'] = dataset_name
    
    # Frequencies (No downsampling)
    config_data['original_freq'] = 30
    config_data['subsampling_freq'] = 30
    
    # Length and Stride (Length 120, Stride 60)
    length = 120
    config_data['length'] = length
    config_data['stride'] = length // 2
    
    # File settings
    config_data['file_type'] = 'sleap_h5'
    
    # Update input files list
    # Ensure it's a list even if a single string is passed
    if isinstance(h5_files, str):
        h5_files = [h5_files]
    config_data['input_files'] = h5_files

    # 3. Write the changes back to the file
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f)
    
    print(f"Successfully updated {config_path} for dataset: {dataset_name}")

# --- Example Usage ---
# analysis_files = [
#     '/path/to/video1.analysis.h5',
#     '/path/to/video2.analysis.h5'
# ]
# update_disk_config("config.yaml", "my_new_experiment", analysis_files)