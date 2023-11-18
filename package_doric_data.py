# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:30:18 2023

@author: tanne
"""

import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..'))

from sys_neuro_tools import doric_utils as dor
from sys_neuro_tools import acq_utils as acq
import pyutils.utils as utils
import numpy as np
import pickle
import matplotlib.pyplot as plt

def test_package_data(load_path, save_path):

    signals_of_interest = ['405', '490']

    data, issues = dor.get_and_check_data(load_path)
    if len(issues) > 0:
        print('Issues found:\n{0}'.format('\n'.join(issues)))

    dec_data = package_data(data, signals_of_interest, save_path)

    fig, axs = plt.subplots(2,2, layout='constrained')
    axs = axs.flatten()

    for i, name in enumerate(dec_data['signals'].keys()):
        ax = axs[i]
        ax.plot(data[name]['Time'], data[name]['Values'], label='Raw Data')
        ax.plot(dec_data['time'], dec_data['signals'][name], label='Decimated Data')
        ax.set_title(name)

    lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels)


def package_data(data, signals_of_interest, save_path, target_dt = 0.005):
    '''
    Extract the relevant signals from a doric data file into a simpler pkl file, with optional decimation

    Parameters
    ----------
    data : Flattened data to package
    signals_of_interest : Names of the signals of interest
    save_path : Path to save
    target_dt : Target timestep. Determines decimation amount. Optional, default is 0.005.

    Returns
    -------
    The saved data structure

    '''

    # get trial start timestamps
    trial_start_ts, trial_nums = acq.parse_trial_times(data['TTL']['DIO01'], data['TTL']['Time'])

    # Extract relevant signals
    signals = {}
    time = []
    for name, signal in data.items():
        if any([soi in name for soi in signals_of_interest]):
            if len(time) == 0:
                time = signal['Time']
            else:
                # ensure the timestamps are the same
                if not np.array_equal(time, signal['Time']):
                    print('Time arrays are not equal')
            signals[name] = signal['Values']

    # decimate signals
    # calculate desired decimation factor
    current_dt = np.mean(np.diff(time))
    current_sf = np.round(1/current_dt)
    target_sf = np.round(1/target_dt)
    if current_sf > target_sf:
        decimation = int(current_sf/target_sf)
    else:
        decimation = 1

    # instead of using signal processing decimation methods, where there is assumed to be a true underlying signal
    # simply average over the number of bins given by the decimation factor to get the downsampled data
    if decimation > 1:
        dec_signals = {}
        # first do signals
        for name, signal in signals.items():
            start_idx = 0
            end_idx = decimation
            dec_signal = []
            while start_idx < len(signal):
                dec_signal.append(np.mean(signal[start_idx:end_idx]))
                start_idx = end_idx
                end_idx += decimation

                if end_idx > len(signal):
                    end_idx = len(signal)

            dec_signals[name] = np.array(dec_signal)

        # then do the time stamps where the new time value is centered on the decimation window
        if decimation % 2 == 0:
            start_t_offset = (decimation+1)/2 * current_dt
        else:
            start_t_offset = decimation/2 * current_dt

        dec_time = time[0] + start_t_offset + (np.arange(len(dec_signal))*decimation*current_dt)
        dec_time = np.array(dec_time)
    else:
        dec_signals = signals
        dec_time = time

    save_data = {'trial_start_ts': trial_start_ts, 'trial_nums': trial_nums,
                 'time': dec_time, 'signals': dec_signals, 'decimation': decimation}

    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    return save_data

filename = 'Test_0000'
load_path = path.join(utils.get_user_home(), 'downloads', filename+'.doric')
save_path = path.join(utils.get_user_home(), 'downloads', filename+'.pkl')

test_package_data(load_path, save_path)