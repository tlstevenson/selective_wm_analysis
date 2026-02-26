# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 23:02:26 2026

@author: cns-th-lab
"""

import os
import sys
import subprocess
import path_manager_NEW as pm
import h5py

def create_dataset(config_path, dataset_name, h5_files, conda_env_path, create_skeleton="n\n"):
    #Edit config file
    #Run python script from the conda environment
    update_disk_dataset_config(config_path, dataset_name, h5_files)
    programs_folder = pm.get_disk_scripts_parent_dir()
    command = ["conda.bat", "run", "--no-capture-output", "-p", conda_env_path, 
               "python", "create_dataset.py"]
    try:
        # Popen streams the output line-by-line
        with subprocess.Popen(command, stdin=subprocess.PIPE, 
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                              text=True, bufsize=1, 
                              shell=True, cwd=programs_folder) as process:
            try:
                process.stdin.write(create_skeleton)
                process.stdin.flush()
            except Exception as e:
                print(f"Could not send input: {e}")
            #stdout, stderr = process.communicate(input=create_skeleton)
            # Iterate through the output as it is generated and print to console
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
            
            # Ensure the process is fully complete before checking the exit code
            process.wait()
            """
            process = subprocess.run(command, input = "y\n", 
                                     text=True, capture_output=True,
                                     check=True, cwd=programs_folder)"""
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
    print(h5_list_str)

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

def generate_disk_skeleton(h5_path, dataset_name):
    """
    Reads node names and edges from a SLEAP H5 file and writes a DISK skeleton YAML.
    """
    
    output_yaml_path = os.path.join(pm.get_disk_scripts_parent_dir(), 'datasets', dataset_name)
    
    if not os.path.exists(h5_path):
        print(f"Error: File {h5_path} not found.")
        return

    with h5py.File(h5_path, 'r') as f:
        # 1. Get Node Names
        nodes = [n.decode() if isinstance(n, bytes) else n for n in f['node_names'][:]]
        
        # 2. Get Edge Indices (the links)
        if 'edge_inds' not in f:
            print("Error: No 'edge_inds' found in H5. Did you define a skeleton in SLEAP?")
            return
        
        edges = f['edge_inds'][:]
        
    # 3. Format into DISK's expected YAML structure
    # We will put everything in one group called 'body' for simplicity
    yaml_content = "skeleton:\n  body:\n"
    for edge in edges:
        yaml_content += f"    - [{edge[0]}, {edge[1]}]\n"

    # 4. Write the file
    with open(output_yaml_path, 'w') as y_file:
        y_file.write(yaml_content)
    
    print(f"Successfully created skeleton file at: {output_yaml_path}")
    print(f"Total nodes: {len(nodes)}")
    print(f"Total links: {len(edges)}")