# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:59:20 2026

@author: cns-th-lab
"""
#%% IMport block
import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

# Imports for behavioral data
import init
from hankslab_db import db_access
from hankslab_db import (
    tonecatdelayresp_db as wm_db,
    basicRLtasks_db as bandit_db,
)
import neurofileread as nfr
import sleapstats as ss

#%%

#Parameters to set
sess_ids = [116498]
parent_dir = r"C:\Users\cns-th-lab\TannerVidsRenamed"
rat_folders = [f for f in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, f))]
model_basename = "260731_198_199x_234x_235x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x_occin"
port_model_basename = "260716_port_model"
label_paths = [] #Location of all rat .h5 files for given session ids
port_label_paths = [] #Location of all port .h5 files for given session ids

#Iterate through rat folders for each session id
for sess_id in sess_ids:
    found_prediction_rat = False
    found_prediction_port = False
    for rat_folder in rat_folders:
        prediction_path_rat = os.path.join(parent_dir, rat_folder, r"Videos\predictions", model_basename, f"mov_{sess_id}.h5")
        print(prediction_path_rat)
        prediction_path_port = os.path.join(parent_dir, rat_folder, r"Videos\predictions", port_model_basename, f"mov_{sess_id}.h5")
        
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

#%%%% Extract project and session specific data

#Project data (should be shared across all .h5 files analyzed)
project_dict = nfr.extract_h5_metadata_w_port_f(label_paths[0], port_label_paths[0]) #TODO: May break as second parameter may need to be a list

labels_dict = {} #Label data by session id

for file in label_paths:
    file_data = nfr.extract_h5_data(file, target_project_structure=project_dict)
    for key in file_data:
        if key not in labels_dict:
            print("WARNING!!! Adding new key. This is ok at the start but not in middle.")
            labels_dict[key] = []
            labels_dict[key].append(file_data[key])
        else:
            labels_dict[key].append(file_data[key])

#Crude check that all collumns added for all videos
first_col_length = len(labels_dict[next(iter(labels_dict))])
for key in labels_dict:
    if len(labels_dict[key]) != first_col_length:
        raise ValueError(f"Column {key} of labels_dict does not match the length of other columns.")

#Create the dataframe after checking shape
labels_df = pd.DataFrame(labels_dict)

#TODO: Also add ports to pd dataframe
#TODO: Begin ianterval extraction and trajectory clustering