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
import os
import os.path as path
import numpy as np

rec_date = date(2025,8,23)

# subj_ids = [198]
# subj_region_dict = {s: {'PL': 2, 'DLS': 1, 'DMS': 3 } for s in subj_ids}#,'TS': 4} #'PL': 1, 'DMS': 2  }#
# wavelength_dict = {490: 2, 420: 1, 465: 4, 405: 3} #420: 1, 465: 4, 490: 2, , 405: 3

subj_region_dict = {274: {'DLS-L': 1, 'DMS-L': 2, 'DMS-R': 3, 'DLS-R': 4},
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
        
    sess_id = subj_sess_ids[subj_id][0]
    region_dict = subj_region_dict[subj_id]
    
    # find the matching data file for the given data
    root_dir = path.join(data_dir, str(subj_id))
    subj_data_files = [path.join(root_dir, f) for f in glob.glob('*.doric', root_dir=root_dir)]
    file_times = [datetime.fromtimestamp(pathlib.Path(f).stat().st_ctime, tz = timezone.utc) for f in subj_data_files]
    file_time_sel = [f_time.date() == rec_date for f_time in file_times]

    if sum(file_time_sel) > 1:
        print('Found {} data files for subject {} on date {}. Please add them individually. Continuing...'.format(sum(file_time_sel), subj_id, rec_date.isoformat()))
        continue
    elif sum(file_time_sel) == 1:
        data_file = np.array(subj_data_files)[np.array(file_time_sel)][0]

        pkg.package_doric_data(subj_id, sess_id, region_dict, wavelength_channel_dict, comments_dict = comments[subj_id],
                               data_path = data_file, target_dt = target_dt, new_format = new_format,
                               print_file_struct = print_struct, print_attr = print_attr)
        
        # rename file with session id
        new_name = path.join(root_dir, 'session_{}'.format(sess_id))
        os.rename(data_file, new_name)

# %% Add Manually

# sess_id = xxxxxx
# pkg.package_doric_data(subj_id, sess_id, region_dict, wavelength_dict, comment_dict = comments,
#                        initial_dir = path.join(data_dir, str(subj_id)), target_dt = target_dt, new_format = new_format,
#                        print_file_struct = print_struct, print_attr = print_attr)


# %% Rename old files

# subj_ids = [198,199,274,400,402,237,238,424,483]
# data_dir = 'D:/Tanner'

# for subj_id in subj_ids:
#     root_dir = path.join(data_dir, str(subj_id))
#     subj_data_files = [path.join(root_dir, f) for f in glob.glob('*.doric', root_dir=root_dir)]
#     file_times = [datetime.fromtimestamp(pathlib.Path(f).stat().st_ctime, tz = timezone.utc) for f in subj_data_files]
    
#     for data_file, file_time in zip(subj_data_files, file_times):
#         if sum([f_time.date() == file_time.date() for f_time in file_times]) > 1:
#             print('Found multiple sessions on the same day for subject {} on date {}. Please add them individually. Continuing...'.format(subj_id, file_time.date().isoformat()))
#             continue
        
#         subj_sess_ids = db_access.get_subj_sess_ids_by_date(subj_id, file_time.date().isoformat())
        
#         # rename file with session id
#         new_name = path.join(root_dir, 'session_{}'.format(subj_sess_ids[subj_id][0]))
#         os.rename(data_file, new_name)
        
        
# %% Update time data

from sys_neuro_tools import doric_utils as dor
from sys_neuro_tools import acq_utils as acq

target_dt = 0.005

subj_region_dict = {198: {'PL': 1, 'DMS': 2, 'DLS': 3},
                    199: {'PL': 1, 'DMS': 2, 'DLS': 3, 'TS': 4},
                    274: {'DLS-L': 1, 'DMS-L': 2, 'DMS-R': 3, 'DLS-R': 4},
                    424: {'PL': 1, 'NAc': 2, 'DMS': 3, 'DLS': 4},
                    237: {'PL': 1, 'NAc': 2, 'DMS': 3, 'TS': 4},
                    238: {'NAc': 1, 'DMS': 2, 'DLS': 3, 'TS': 4},
                    483: {'NAc-L': 1, 'NAc-R': 2, 'TS-L': 3, 'TS-R': 4},
                    400: {'DLS': 1, 'NAc': 2, 'PL': 3, 'DMS': 4},
                    402: {'DLS': 1, 'PL': 2, 'DMS': 3, 'TS': 4}}
wavelength_channel_dict = {1: 420, 2: 490, 3: 420, 4: 490}

subj_ids = list(subj_region_dict.keys())

data_dir = 'D:/Tanner'

for subj_id in subj_ids:
    
    subj_sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_id)
    
    root_dir = path.join(data_dir, str(subj_id))
    subj_data_files = [path.join(root_dir, f) for f in glob.glob('*.doric', root_dir=root_dir)]
    
    region_dict = subj_region_dict[subj_id]

    for sess_id in subj_sess_ids[subj_id]:
        data_path = [f for f in subj_data_files if str(sess_id) in f]
        
        dor_signal_path = '/DataAcquisition/FPConsole/Signals/Series0001/'
        ttl_name = 'ttl'

        signal_name_dict = {ttl_name: {'time': 'DigitalIO/Time', 'values': 'DigitalIO/DIO01'}}

        signal_name_dict.update({'{}_{}'.format(r, wavelength_channel_dict[c]):
                                 {'time': 'LockInAOUT0{}/Time'.format(c),
                                  'values': 'LockInAOUT0{}/AIN0{}'.format(c, region_dict[r])}
                                 for r in region_dict.keys() for c in wavelength_channel_dict.keys()})

        data = dor.get_specific_data(data_path, dor_signal_path, signal_name_dict)

        data, issues = dor.fill_missing_data(data, 'time')
        if len(issues) > 0:
            print('Issues found:\n{0}'.format('\n'.join(issues)))

        signal_data = {k:v for k,v in data.items() if k != ttl_name and 'values' in v}
        dec_time, dec_signals, dec_info = acq.decimate_data(signal_data, target_dt = target_dt)

        time_data = {'start': dec_time[0], 'end': dec_time[-1], 'dt': dec_info['decimated_dt'], 'length': len(dec_time), 'dec_info': dec_info}

        for region in region_dict.keys():
            db_access.update_fp_time_data(subj_id, sess_id, region, time_data)


