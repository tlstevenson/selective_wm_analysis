# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:30:18 2023

@author: tanne
"""

# %% Declare imports and methods

import init
from hankslab_db import db_access
from hankslab_db import package_fp_data as pkg
from datetime import date, datetime, timezone
import glob
import pathlib
import os.path as path
import numpy as np

rec_date = date(2025,8,23)

# subj_ids = [198]
# subj_region_dict = {s: {'PL': 2, 'DLS': 1, 'DMS': 3 } for s in subj_ids}#,'TS': 4} #'PL': 1, 'DMS': 2  }#
# wavelength_dict = {490: 2, 420: 1, 465: 4, 405: 3} #420: 1, 465: 4, 490: 2, , 405: 3

subj_region_dict = {274: {'DLS-L': 1, 'DMS_L': 2, 'DMS-R': 3, 'DLS-R': 4},
                    400: {'DLS': 1, 'NAc': 2, 'PL': 3, 'DMS': 4},
                    402: {'DLS': 1, 'PL': 2, 'DMS': 3, 'TS': 4}}
wavelength_channel_dict = {1: 420, 2: 490, 3: 420, 4: 490}

subj_ids = list(subj_region_dict.keys())

target_dt = 0.005
new_format = True
print_struct = False
print_attr = False

comments = {s: {r: '' for r in subj_region_dict[s].keys()} for s in subj_ids}
# add comments below
# comments[191]['PL'] = 'Patch cord disconnected at some point'

data_dir = 'D:/Tanner'

subj_sess_ids = db_access.get_subj_sess_ids_by_date(subj_ids, rec_date.isoformat())

# to add individual sessions
# subj_ids = [191]
#subj_sess_ids = {198: [116588], 199: [116589]}

for subj_id in subj_ids:
    if not subj_id in subj_sess_ids.keys():
        continue
    
    if len(subj_sess_ids[subj_id]) > 1:
        print('Found {} sessions for subject {} on date {}. Please add them individually. Continuing...'.format(len(subj_sess_ids[subj_id]), subj_id, rec_date.isoformat()))
        continue
        
    region_dict = subj_region_dict[subj_id]
    
    # find the matching data file for the given data
    root_dir = path.join(data_dir, str(subj_id))
    subj_data_files = [path.join(root_dir, f) for f in glob.glob('*.doric', root_dir=root_dir)]
    file_times = sorted([datetime.fromtimestamp(pathlib.Path(f).stat().st_ctime, tz = timezone.utc) for f in subj_data_files])
    file_time_sel = [f_time.date() == rec_date for f_time in file_times]

    if sum(file_time_sel) > 1:
        print('Found {} data files for subject {} on date {}. Please add them individually. Continuing...'.format(sum(file_time_sel), subj_id, rec_date.isoformat()))
        continue
    elif sum(file_time_sel) == 1:
        data_file = np.array(subj_data_files)[np.array(file_time_sel)][0]

        pkg.package_doric_data(subj_id, subj_sess_ids[subj_id][0], region_dict, wavelength_channel_dict, comments_dict = comments[subj_id],
                               data_path = data_file, target_dt = target_dt, new_format = new_format,
                               print_file_struct = print_struct, print_attr = print_attr)

# sess_id = xxxxxx
# pkg.package_doric_data(subj_id, sess_id, region_dict, wavelength_dict, comment_dict = comments,
#                        initial_dir = path.join(data_dir, str(subj_id)), target_dt = target_dt, new_format = new_format,
#                        print_file_struct = print_struct, print_attr = print_attr)
