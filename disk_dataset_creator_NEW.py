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
import json

def create_dataset(config_path, dataset_name, h5_files, conda_env_path):
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
                process.stdin.write("n\n")
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
    create_skeleton(dataset_name, h5_files)

def create_skeleton(dataset_name, h5_files):
    programs_folder = pm.get_disk_scripts_parent_dir()
    #Build skeleton seperately
    skeleton_dir_path = os.path.join(programs_folder, "datasets", dataset_name)
    build_skeleton_from_h5(h5_files[0], skeleton_dir_path)

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

"""def generate_disk_skeleton(h5_path, dataset_name):
    ""
    Reads node names and edges from a SLEAP H5 file and writes a DISK skeleton YAML.
    ""
    
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
    print(f"Total links: {len(edges)}")"""
    
def build_skeleton_from_h5(h5_path, outputdir):
    # Setup paths
    print("Building skeletons")
    os.makedirs(outputdir, exist_ok=True)
    json_path = os.path.join(outputdir, 'node_groups.json')
    
    # Your requested color palette
    color_palette = ['orange', 'gold', 'grey', 'cornflowerblue', 'turquoise', 
                     'hotpink', 'purple', 'blue', 'seagreen', 'darkolivegreen']

    # 1. Read H5 file for nodes and edges
    try:
        with h5py.File(h5_path, 'r') as f:
            keypoints = [n.decode() if isinstance(n, bytes) else n for n in f['node_names'][:]]
            edges = f['edge_inds'][:] if 'edge_inds' in f else []
    except Exception as e:
        print(f"CRITICAL ERROR reading H5 file: {e}")
        return

    # 2. PHASE 1: Generate JSON and Quit
    if not os.path.exists(json_path):
        groups_template = {
            "body": {"color": color_palette[0], "nodes": []},
            "head":    {"color": color_palette[1], "nodes": []},
            "tail":    {"color": color_palette[2], "nodes": []}
        }
        
        with open(json_path, 'w') as jf:
            json.dump(groups_template, jf, indent=4)
            
        print(f"--- INIT PHASE COMPLETE ---")
        print(f"Generated group config at: {json_path}")
        print("\nPlease open 'node_groups.json' and add the following node names into the 'nodes' lists:")
        for i, name in enumerate(keypoints):
            print(f"  {i}: '{name}'")
        print("\nExiting script. Re-run once the JSON is populated.")
        sys.exit(0)

    # 3. PHASE 2: Group the SLEAP edges based on the populated JSON
    with open(json_path, 'r') as jf:
        groups = json.load(jf)

    # Create empty buckets for the edges
    grouped_edges = {g_name: [] for g_name in groups.keys()}
    ungrouped_edges = []

    # Sort each edge from SLEAP into one of the four groups
    for edge in edges:
        u, v = int(edge[0]), int(edge[1])
        name_u, name_v = keypoints[u], keypoints[v]
        
        assigned = False
        for g_name, g_data in groups.items():
            # If either node in the link belongs to a group, assign the link to that group
            if name_u in g_data['nodes'] or name_v in g_data['nodes']:
                grouped_edges[g_name].append((u, v))
                assigned = True
                break # Stop searching once sorted
        
        if not assigned:
            ungrouped_edges.append((u, v))

    # 4. Reformat to perfectly mimic DISK's eval() logic
    neighbor_links = []
    link_colors = []

    for g_name, edges_list in grouped_edges.items():
        if len(edges_list) == 0:
            continue # Skip empty groups
        
        # If there's only 1 link, DISK makes it a single tuple: (0, 2)
        # If there are multiple, DISK makes it a tuple of tuples: ((0, 2), (0, 6))
        if len(edges_list) == 1:
            formatted_group = edges_list[0]
        else:
            formatted_group = tuple(edges_list)
            
        neighbor_links.append(formatted_group)
        link_colors.append(groups[g_name]['color'])

    # Handle any links that were missed in the JSON to prevent crashes
    if ungrouped_edges:
        if len(ungrouped_edges) == 1:
            neighbor_links.append(ungrouped_edges[0])
        else:
            neighbor_links.append(tuple(ungrouped_edges))
        link_colors.append('white') # Safe fallback color

    # Find the center index (Defaults to 0 if body_m is missing)
    center = keypoints.index('body_m') if 'body_m' in keypoints else 0

    # 5. PHASE 3: Write to skeleton.py
    skeleton_file_path = os.path.join(outputdir, 'skeleton.py')
    
    with open(skeleton_file_path, 'w') as opened_file:
        txt = f"num_keypoints = {len(keypoints)}\n"
        txt += f"keypoints = {keypoints}\n"
        txt += f"center = {center}\n"
        txt += f"original_directory = '{outputdir.replace(chr(92), '/')}'\n"
        txt += f"neighbor_links = {neighbor_links}\n"
        txt += f"link_colors = {link_colors}\n"

        opened_file.write(txt)
        
    print(f"\n--- WRITING PHASE COMPLETE ---")
    print(f"Successfully generated: {skeleton_file_path}")