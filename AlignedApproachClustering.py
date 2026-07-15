# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:16:27 2026

@author: Alexandru Tapus
"""

import init
from hankslab_db import db_access
import doric_utils as du
import numpy as np
import os
from pathlib import Path
from hankslab_db import tonecatdelayresp_db as wm_db, basicRLtasks_db as bandit_db

#%% Tanner imports
"""
import init
import pandas as pd
import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
from hankslab_db import db_access
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
import numpy as np
import matplotlib.pyplot as plt
import copy
import os.path as path
import pickle
import time"""

#%%
def get_file_paths(directory_path, extension="None"):
    """Returns a list of strings containing the paths of all files in a directory."""
    path_obj = Path(directory_path)
    
    if extension=="None":
        return [str(file) for file in path_obj.iterdir() if file.is_file()]
    else:
        return[str(file) for file in path_obj.iterdir() if file.is_file() and file.suffix==extension]
#%% Extract session indices by vid name
vid_folders = [r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos"]
analysis_sessions = []
for folder in vid_folders:
    videos = get_file_paths(folder, extension=".mp4")
    for video in videos:
        analysis_sessions.append(os.path.splitext(os.path.basename(video))[0])
        
#%% Get corresponding predictions folder
predictions_folder = r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x"
#Extracts all the predictions for the first video folder (Rat 483) [Later will change to match database code]
pose_label_files = [os.path.join(predictions_folder, f"{os.path.splitext(os.path.basename(video))[0]}.slp") for video in get_file_paths(vid_folders[0], extension=".mp4") if os.path.exists(os.path.join(predictions_folder, f"{os.path.splitext(os.path.basename(video))[0]}.slp"))]

#%% Get time series of fp data
trial_start_ts_dict = db_access.get_fp_trial_start_ts(analysis_sessions)

wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

for sess_id in analysis_sessions:
    #Messy block to check both databases for a bevahioral data slot for this session
    try:
        trial_data = wm_loc_db.get_behavior_data(sess_id)
    except:
        print("Wm data failed. Trying bandit.")
        try:
            trial_data = bandit_loc_db.get_behavior_data(sess_id)
        except Exception as e:
            print("Bandit failed.")
            print(e)
    trial_start_ts = trial_start_ts_dict[sess_id]
    trial_start_ts = trial_start_ts[:-1]
    cue_ts = trial_start_ts + trial_data['response_cue_time']
    cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
    response_ts = trial_start_ts + trial_data['response_time']


#%%Investigate video doric structure
doric_file_fp = r"C:\Users\cns-th-lab\Tanner_Alex_Vids\483\Session_0004.doric"
doric_file_vid = r"C:\Users\cns-th-lab\Tanner_Alex_Vids\483\Videos\483.2026-04-03.mov_0004.doric"
du.h5print(doric_file_vid)
time = du.h5read(doric_file_vid, ['DataAcquisition', 'BehaviorCamera', 'Video', 'Series0001', 'DMK-33UX290', 'Time'])
print(time)
print(time[0])
print(time[1])
time = np.array(time[0])
print(np.shape(time))

#%% Extract rate number and recording date from name (DEPRACATED)
#r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-07-28.mov_0001.mp4",
vid_list = [r"C:\Users\cns-th-lab\Tanner_Alex_Vids\483\Videos\483.2026-04-03.mov_0004.mp4"]
print(vid_list)
subj_date_data_dict = {}
subj_date_sess_dict = {}
for vid in vid_list:
    root, ext = os.path.splitext(os.path.basename(vid))
    root, mov = os.path.splitext(root)
    rat_num, date = os.path.splitext(root)   
    rat_num = int(rat_num)
    date = date[1:] #Since the . is included
    
    print(rat_num)
    print(date)
    print(type(rat_num))
    print(type(date))
    
    if not rat_num in subj_date_sess_dict.keys():
        subj_date_sess_dict[rat_num] = {}
    subj_dict_from_db = db_access.get_subj_sess_ids_by_date([rat_num], date) #Dictionary of ratnum: ids
    subj_date_sess_dict[rat_num][date] = subj_dict_from_db[rat_num]
    
    if not rat_num in subj_date_data_dict.keys():
        subj_date_data_dict[rat_num] = {}
    if(len(subj_dict_from_db[rat_num]) ==  1):
        subj_date_data_dict[rat_num][date] = db_access.get_session_data(subj_dict_from_db[rat_num])
    else:
        print("More than one id for a rat and date. Video session identity ambiguous.")
        subj_date_data_dict[rat_num][date] = []
    #subj_date_data_dict[rat_num] = {}
    #Get the session data that corresponds to that rat and day's attempt
    #subj_date_data_dict[rat_num][date] = db_access.get_session_data(db_access.get_subj_sess_ids_by_date([rat_num], date))
#Extract all nose poke times
#%%
for subj_id in subj_date_data_dict.keys():
    for date in subj_date_data_dict[subj_id].keys():
        print(subj_date_data_dict[subj_id][date].info())
        print(len(subj_date_data_dict[subj_id][date]["cpoke_in_time"]))
        print(len(subj_date_data_dict[subj_id][date]["cpoke_out_time"]))
        print(subj_date_data_dict[subj_id][date]["cpoke_in_time"])
        print(subj_date_data_dict[subj_id][date]["cpoke_out_time"])
        break
    break
#Extract all interpolated data at nose poke times
#Discard NaN sequences at Nose poke times

#%% Explore fp data
"""
for subj_id in subj_date_data_dict.keys():
    for date in subj_date_data_dict[subj_id].keys():
        print(subj_date_data_dict[subj_id][date].iloc[0]["parsed_events"].keys())
        print(subj_date_data_dict[subj_id][date].iloc[0]["parsed_events"]['States'].keys())
        print(subj_date_data_dict[subj_id][date].iloc[0]["parsed_events"]['Events'].keys())
        break
    break"""