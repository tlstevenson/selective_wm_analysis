# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:31:00 2026

@author: cns-th-lab
"""

import os
import h5py
import numpy as np
import pandas as pd

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

def extract_h5_metadata_w_port(filepath, port_label_dirs): #TODO: Depracate this
    port_file = get_port_file(filepath, port_label_dirs)
    ans = extract_h5_data_w_port_f(filepath, port_file)
    return ans

def extract_h5_metadata_w_port_f(filepath, port_file):
    port_m_dict = extract_h5_metadata(filepath)
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

def extract_h5_data_w_port(filepath, port_label_dirs, target_project_structure={}): #TODO: Depracate
    port_file = get_port_file(filepath, port_label_dirs)
    ans = extract_h5_data_w_port_f(filepath, port_file, target_project_structure=target_project_structure)
    return ans
    
            
def extract_h5_data_w_port_f(filepath, port_file, target_project_structure={}):
    port_data_dict = extract_h5_data(filepath, target_project_structure)
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
            
def rat_file_paths_by_sess(sess_ids, par_dir, model_basename):
    label_paths = [] #Location of all rat .h5 files for given session ids
    rat_folders = [f for f in os.listdir(par_dir) if os.path.isdir(os.path.join(par_dir, f))]

    for sess_id in sess_ids:
        found_prediction_rat = False
        for rat_folder in rat_folders:
            prediction_path_rat = os.path.join(par_dir, rat_folder, r"Videos\predictions", model_basename, f"mov_{sess_id}.h5")

            #Mark whether port and rat files exist
            if os.path.exists(prediction_path_rat):
                found_prediction_rat = True
                label_paths.append(prediction_path_rat)
                break
        
        #Raise an error if any sepcified session does not have a prediction
        if not found_prediction_rat:
            raise FileExistsError(f"Rat prediction files not found for session id: {sess_id}")
            
    return label_paths
            
def port_file_paths_by_sess(sess_ids, par_dir, port_model_basename):
    port_label_paths = [] #Location of all port .h5 files for given session ids
    rat_folders = [f for f in os.listdir(par_dir) if os.path.isdir(os.path.join(par_dir, f))]

    for sess_id in sess_ids:
        found_prediction_port = False
        for rat_folder in rat_folders:
            prediction_path_port = os.path.join(par_dir, rat_folder, r"Videos\predictions", port_model_basename, f"mov_{sess_id}.h5")
            
            if os.path.exists(prediction_path_port):
                found_prediction_port = True
                port_label_paths.append(prediction_path_port)
                break
        
        #Raise an error if any sepcified session does not have a prediction
        if not found_prediction_port:
            raise FileExistsError(f"Rat and port prediction files not found for session id: {sess_id}")
    return port_label_paths
            
def rat_port_file_paths_by_sess(sess_ids, par_dir, model_basename, port_model_basename):
    port_label_paths = [] #Location of all port .h5 files for given session ids
    label_paths = [] #Location of all rat .h5 files for given session ids
    rat_folders = [f for f in os.listdir(par_dir) if os.path.isdir(os.path.join(par_dir, f))]

    for sess_id in sess_ids:
        found_prediction_rat = False
        found_prediction_port = False
        for rat_folder in rat_folders:
            prediction_path_rat = os.path.join(par_dir, rat_folder, r"Videos\predictions", model_basename, f"mov_{sess_id}.h5")
            print(prediction_path_rat)
            prediction_path_port = os.path.join(par_dir, rat_folder, r"Videos\predictions", port_model_basename, f"mov_{sess_id}.h5")
            
            #Mark whether port and rat files exist
            if os.path.exists(prediction_path_rat):
                found_prediction_rat = True
            
            if os.path.exists(prediction_path_port):
                found_prediction_port = True
            
            #Add both port and rat files only if both exist 
            if found_prediction_port and found_prediction_rat:
                label_paths.append(prediction_path_rat)
                port_label_paths.append(prediction_path_port)
                break
            elif found_prediction_port or found_prediction_rat:
                error_msg = """One of the following files was not found.
                                Rat file at path {prediction_path_rat}: {found_prediction_rat}
                                Port file at path {prediction_path_port}: {found_prediction_port}"""
                raise FileExistsError(error_msg)
            else:
                continue
        
        #Raise an error if any sepcified session does not have a prediction
        if not (found_prediction_rat and found_prediction_port):
            raise FileExistsError(f"Rat and port prediction files not found for session id: {sess_id}")
    return label_paths, port_label_paths

def generate_sleap_dataframe_f(rat_label_paths, port_label_paths):
    #Project data (should be shared across all .h5 files analyzed)
    project_dict = extract_h5_metadata_w_port_f(rat_label_paths[0], port_label_paths[0]) #TODO: May break as second parameter may need to be a list
    
    labels_dict = {} #Label data by session id
    idx = 0
    for file_idx in range(len(rat_label_paths)):
        comb_file_data = extract_h5_data_w_port_f(rat_label_paths[file_idx], port_label_paths[file_idx], target_project_structure=project_dict)
        for key in comb_file_data:
            if key not in labels_dict:
                print(f"WARNING!!! Adding new key {key}. This is ok at the start but not in middle.")
                labels_dict[key] = []
                labels_dict[key].append(comb_file_data[key])
            else:
                labels_dict[key].append(comb_file_data[key])
    
    #Crude check that all collumns added for all videos
    first_col_length = len(labels_dict[next(iter(labels_dict))])
    for key in labels_dict:
        if len(labels_dict[key]) != first_col_length:
            raise ValueError(f"Column {key} of labels_dict does not match the length of other columns.")
    
    #Create the dataframe after checking shape
    labels_df = pd.DataFrame(labels_dict)
    return labels_df