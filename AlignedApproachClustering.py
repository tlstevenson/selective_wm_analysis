# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 18:16:27 2026

@author: Alexandru Tapus
"""

import init

import pyutils.utils as utils
from hankslab_db import db_access
import hankslab_db.tonecatdelayresp_db as db
import hankslab_db.pclicksdiscrim_db as discrim_db
import beh_analysis_helpers as bah
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sys_neuro_tools import plot_utils
import os

#%%
#Extract rate number and recording date from name
#r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-07-28.mov_0001.mp4",
vid_list = [r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-07-29.mov_0002.mp4",
            r"C:\Users\cns-th-lab\Tanner_Alex_Vids\199\Videos\199.2025-07-28.mov_0001.mp4",
            r"C:/Users/cns-th-lab/Tanner_Alex_Vids/199/Videos/199.2025-07-29.mov_0002.mp4"]
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
        print(subj_date_data_dict[subj_id][date])
        print(subj_date_data_dict[subj_id][date].info())
        print(len(subj_date_data_dict[subj_id][date]["cpoke_in_time"]))
        print(len(subj_date_data_dict[subj_id][date]["cpoke_out_time"]))
        print(subj_date_data_dict[subj_id][date]["cpoke_in_time"])
        print(subj_date_data_dict[subj_id][date]["cpoke_out_time"])
        break
    break
#Extract all interpolated data at nose poke times
#Discard NaN sequences at Nose poke times