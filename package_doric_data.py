# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:30:18 2023

@author: tanne
"""

# %% Declare imports and methods

import init
import os.path as path
from pyutils import utils
from sys_neuro_tools import doric_utils as dor
from sys_neuro_tools import acq_utils as acq
import pickle
import matplotlib.pyplot as plt

def package_data(data, save_paths, ttl_name = 'ttl', target_dt = 0.005):
    ''' Decimate and save data to a pkl file '''

    # get trial start timestamps
    trial_start_ts, trial_nums = acq.parse_trial_times(data[ttl_name]['values'], data[ttl_name]['time'])

    signal_data = {k:v for k,v in data.items() if k != ttl_name}
    dec_time, dec_signals, dec_info = acq.decimate_data(signal_data, target_dt = target_dt)

    save_data = {'trial_start_ts': trial_start_ts, 'trial_nums': trial_nums,
                 'time': dec_time, 'signals': dec_signals, 'decimation': dec_info}

    if not utils.is_list(save_paths):
        save_paths = [save_paths]
        
    for save_path in save_paths:
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

    return save_data

# %% Declare file & look at structure
sess_id = 94072
filename = 'Session_{}'.format(sess_id)
load_path = path.join('E:', 'Data', 'Rat 179', filename+'.doric')

dor.h5print(load_path)

# %% Get specific data

data_path = '/DataAcquisition/FPConsole/Signals/Series0001/'

# for Neuroscience Studio 2.6.4
# signal_name_dict = {'ttl': {'time': 'DigitalIO/Time', 'values': 'DigitalIO/DIO01'},
#                     'DMS_405': {'time': 'AIN01xAOUT01-LockIn/Time', 'values': 'AIN01xAOUT01-LockIn/Values'},
#                     'DMS_490': {'time': 'AIN01xAOUT02-LockIn/Time', 'values': 'AIN01xAOUT02-LockIn/Values'},
#                     'PFC_405': {'time': 'AIN02xAOUT01-LockIn/Time', 'values': 'AIN02xAOUT01-LockIn/Values'},
#                     'PFC_490': {'time': 'AIN02xAOUT02-LockIn/Time', 'values': 'AIN02xAOUT02-LockIn/Values'}}

# declare which channels were PFC and DMS
PFC_ch = '1'
DMS_ch = '2'

signal_name_dict = {'ttl': {'time': 'DigitalIO/Time', 'values': 'DigitalIO/DIO01'},
                    'DMS_420': {'time': 'LockInAOUT01/Time', 'values': 'LockInAOUT01/AIN0' + DMS_ch},
                    'DMS_490': {'time': 'LockInAOUT02/Time', 'values': 'LockInAOUT02/AIN0' + DMS_ch},
                    'DMS_405': {'time': 'LockInAOUT03/Time', 'values': 'LockInAOUT03/AIN0' + DMS_ch},
                    'PFC_420': {'time': 'LockInAOUT01/Time', 'values': 'LockInAOUT01/AIN0' + PFC_ch},
                    'PFC_490': {'time': 'LockInAOUT02/Time', 'values': 'LockInAOUT02/AIN0' + PFC_ch},
                    'PFC_405': {'time': 'LockInAOUT03/Time', 'values': 'LockInAOUT03/AIN0' + PFC_ch}}

data = dor.get_specific_data(load_path, data_path, signal_name_dict)
# %% fill missing data
data, issues = dor.fill_missing_data(data, 'time')
if len(issues) > 0:
    print('Issues found:\n{0}'.format('\n'.join(issues)))

# %% Package data
save_paths = [path.join('E:', 'Data', 'Rat 179', filename+'.pkl'), path.join(utils.get_user_home(), 'fp_data', filename+'.pkl')]
dec_data = package_data(data, save_paths)

# %% Plot comparison of raw and decimated data

# fig, axs = plt.subplots(2,2, layout='constrained', figsize=(12,10))
# axs = axs.flatten()

# for i, signal_name in enumerate([k for k in data.keys() if k != 'ttl']):
#     ax = axs[i]
#     ax.plot(data[signal_name]['time'], data[signal_name]['values'], label='Raw Data')
#     ax.plot(dec_data['time'], dec_data['signals'][signal_name], label='Decimated Data')
#     ax.set_title(signal_name)

# lines, labels = axs[0].get_legend_handles_labels()
# fig.legend(lines, labels)