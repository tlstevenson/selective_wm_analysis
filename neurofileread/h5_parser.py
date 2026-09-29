# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:31:00 2026

@author: cns-th-lab
"""

import os
import h5py
import numpy as np

def get_h5_files_dir(label_dir_paths, sess_ids=[]):
    for path in label_dir_paths:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist for h5 searching.")

    label_files = []
    if len(sess_ids) > 0:
        for folder in label_dir_paths:
            folder_had_h5 = 0
            for file in os.listdir(folder):
                name, ext = os.path.splitext(file)
                if ext == ".h5" and int(str.removeprefix(name, "mov_")) in sess_ids:
                    label_files.append(os.path.join(folder, file))
                    folder_had_h5 += 1
            if folder_had_h5 < 2:
                raise ValueError(f"Specified folder {folder} did not have two h5 files with specified sessids")
    else:
        label_files = [os.path.join(folder, file) for folder in label_dir_paths for file in os.listdir(folder) if file.endswith(".h5")]
    return label_files

def get_port_file(rat_label_file, port_label_dirs):
    port_file = None
    name_no_ext = os.path.splitext(os.path.basename(rat_label_file))[0]
    my_sess = str.removeprefix(name_no_ext, "mov_")
    for port_dir in port_label_dirs:
        if os.path.exists(port_dir):
            for filename in os.listdir(port_dir):
                if my_sess in filename and ".h5" in filename:
                    port_file = os.path.join(port_dir, filename)
                    break
            if port_file != None:
                break
        else:
            print(f"Port directory {port_dir} not found. Continuing. CAUTION! Will output empty")

    if port_file == None:
        print(f"WARNING: File {os.path.basename(rat_label_file)} has no corresponding port label in provided port path directories.")
    return port_file

def extract_h5_metadata(filepath):
    with h5py.File(filepath, "r") as f:
        labels_dict = {
            "node_names": [n.decode("utf-8") for n in f["node_names"][:]],
            "edge_names": [[n1.decode("utf-8"), n2.decode("utf-8")] for n1, n2 in f["edge_names"][:]],
            "vid_path":"", 
            "sess": "", 
            "model_name": ""
        }
        labels_dict["edge_inds"] = [[labels_dict["node_names"].index(name_1), labels_dict["node_names"].index(name_2)] for name_1, name_2 in labels_dict["edge_names"]]
        
        vid_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(filepath))), os.path.basename(filepath))
        labels_dict["vid_path"] = os.path.splitext(vid_path)[0] + ".mp4"
        labels_dict["sess"] = str.removeprefix(os.path.splitext(os.path.basename(filepath))[0], "mov_")
        labels_dict["subj_id"] =  os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(filepath)))))
        labels_dict["model_name"] = os.path.basename(os.path.dirname(filepath))
        return labels_dict

def extract_h5_metadata_w_port(filepath, port_label_dirs):
    port_m_dict = extract_h5_metadata(filepath)
    port_file = get_port_file(filepath, port_label_dirs)
    if port_file == None:
        raise ValueError("WARNING!!! Port file not found")
    elif not os.path.exists(port_file):
        raise ValueError(f"Port label path at {port_file} does not exist")
    with h5py.File(port_file, "r") as g:
        port_m_dict["port_names"] = [n.decode("utf-8") for n in g["node_names"][:]]
        port_m_dict["port_filepath"] = port_file
    return port_m_dict

def extract_h5_data(filepath, target_project_structure={}):
    with h5py.File(filepath, "r") as f:
        data_dict = extract_h5_metadata(filepath)
        
        if target_project_structure != {}:
            if data_dict["node_names"] != target_project_structure["node_names"]:
                raise LookupError("Project structure not identical in nodes and indexing will fail.")
            if data_dict["edge_names"] != target_project_structure["edge_names"]:
                raise LookupError("Project structure not identical in edges and indexing will fail.")
        
        data_dict["scores"] = np.transpose(f["point_scores"][:], (2, 1, 0))
        data_dict["tracks"] = np.transpose(f["tracks"][:])
        return data_dict

def extract_h5_data_w_port(filepath, port_label_dirs, target_project_structure={}):
    port_data_dict = extract_h5_data(filepath, target_project_structure)
    port_file = get_port_file(filepath, port_label_dirs)
    try:
        if port_file == None:
            port_data_dict["port_tracks"] = [] # Fixed from labels_dict
            print("WARNING!!! Empty port list added as placeholder to port_tracks. DO NOT USE!")
            return port_data_dict
            
        with h5py.File(port_file, "r") as g:
            port_data_dict["port_names"] = [n.decode("utf-8") for n in g["node_names"][:]]
            if target_project_structure != {} and port_data_dict["port_names"] != target_project_structure.get("port_names", []): # Fixed from project_dict
                raise LookupError("Project structure not identical in port names and indexing will fail.")
            port_data_dict["port_tracks"] = np.transpose(g["tracks"][:])
        return port_data_dict
    except Exception as e:
        print(e)
        if port_file == None:
            raise ValueError("No port file found at specified locations!!!")