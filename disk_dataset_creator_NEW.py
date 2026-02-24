# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:02:26 2026

@author: cns-th-lab
"""

import os
import sys
import subprocess
import path_manager_NEW as pm

def create_dataset(config_path, dataset_name, h5_files, conda_env_path):
    #Edit config file
    #Run python script from the conda environment
    update_disk_dataset_config(config_path, dataset_name, h5_files)
    programs_folder = pm.get_disk_scripts_parent_dir()
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

def update_disk_dataset_config(config_path, dataset_name, h5_files):
    """Updates the YAML template using standard string replacement."""
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        return False

    with open(config_path, 'r') as f:
        lines = f.readlines()

    # Format the list of files for YAML: ['path1', 'path2']
    h5_list_str = str([os.path.abspath(f).replace("\\", "/") for f in h5_files])

    new_lines = []
    for line in lines:
        # Match keys at the start of the line and replace their values
        if line.strip().startswith('dataset_name:'):
            line = f'dataset_name: {dataset_name}\n'
        elif line.strip().startswith('original_freq:'):
            line = 'original_freq: 30\n'
        elif line.strip().startswith('subsampling_freq:'):
            line = 'subsampling_freq: 30\n'
        elif line.strip().startswith('length:'):
            line = 'length: 120\n'
        elif line.strip().startswith('stride:'):
            line = 'stride: 60\n'
        elif line.strip().startswith('file_type:'):
            line = 'file_type: sleap_h5\n'
        elif line.strip().startswith('input_files:') or line.strip().startswith('#input_files:'):
            line = f'input_files: {h5_list_str}\n'
        
        new_lines.append(line)

    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    return True