# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 13:41:01 2024

@author: tanne
"""

# %% imports

import init
import hankslab_db.basicRLtasks_db as rl_db
import hankslab_db.tonecatdelayresp_db as wm_db
from pyutils import utils
import numpy as np
import fp_analysis_helpers as fpah
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
from hankslab_db import db_access
import pandas as pd
import time


# %% Load data
wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
rl_loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

reload = False
baseline_lpf = 0.0005

#sess_ids = {180: [101447]}
#sess_ids = {191: [101617]}

#fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, iso_lpf=10)
#sess_data = wm_loc_db.get_behavior_data(utils.flatten(sess_ids))

# sess_ids = {207: [100752]}
# fp_data, implant_info = fpah.load_fp_data(rl_loc_db, sess_ids)
# sess_data = rl_loc_db.get_behavior_data(utils.flatten(sess_ids))

# sess_ids = {400: [119285], 402: [119234]}
sess_ids = {179: [], 180: [], 191: [102208], 188: [102201, 102580],
            207: [], 199: [], 400: [119194], 402: [119009], 202: []}

fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, baseline_correction=True, tilt_t=True, filter_dropout_outliers=True, band_iso_fit=True, 
                                          irls_fit=True, reload=reload, baseline_lpf=baseline_lpf)
#sess_data = wm_loc_db.get_behavior_data(utils.flatten(sess_ids))

# %% View preprocessing

gen_title = 'Subject {}, Session {}'
sub_t = [0, np.inf] # [1100, 1120] #
dec = 2

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time']

        fpah.view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'], t, t_min=sub_t[0], t_max=sub_t[1], dec=dec,
                                    title=gen_title.format(subj_id, sess_id), plot_baseline_corr=True, plot_fband=True, plot_irls=True)

# %% define plotting routine

def plot_signal_details(processed_signals, t, title, lines_dict={}, dec=10, signal_types=None):

    regions = processed_signals.keys()

    t = t[::dec].copy()

    if signal_types is None:
        signal_types = ['dff_iso']
        
    for signal_type in signal_types:
        
        fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=[18,6*len(regions)], sharex=True)
        if len(regions) == 1:
            axs = [axs]
        
        plt.suptitle(title + ' - ' + signal_type)
    
        for i, region in enumerate(regions):

            ax = axs[i]
            ax.plot(t, processed_signals[region][signal_type][::dec], label='_')

            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors = prop_cycle.by_key()['color']

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color=colors[j], linestyles='dashed', transform=ax.get_xaxis_transform())

            ax.set_title(region)
            _, y_label = fpah.get_signal_type_labels(signal_type)
            ax.set_ylabel(y_label)
            ax.set_xlabel('Time (s)')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()


# %% create plots
subj_ids = list(sess_ids.keys())
dec=2
signal_types = ['z_dff_iso']

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:

        sess_fp = fp_data[subj_id][sess_id]
        trial_data = sess_data[sess_data['sessid'] == sess_id]

        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        first_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[0] if utils.is_list(x) else x)
        second_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[1] if utils.is_list(x) else np.nan)
        tone_ts = np.unique([*first_tone_ts, *second_tone_ts])
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        reward_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = reward_ts[trial_data['reward'] > 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, 'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts}

        plot_signal_details(sess_fp['processed_signals'], sess_fp['time'], 'Subject {}, Session {}'.format(subj_id, sess_id), lines_dict, dec=dec, signal_types=signal_types)


# %% Create cleaned up example data plot

subj_id = 207
sess_id = 100752
# subj_id = 180
# sess_id = 101447
sess_ids = {subj_id: [sess_id]}
sess_fp = fp_data[subj_id][sess_id]
trial_data = sess_data[sess_data['sessid'] == sess_id]
t_range = [1420, 1470] #[0, np.inf] #[2730, 2780] #[1155, 1195] # [1150, 1250] # 
regions = ['DMS', 'PL']
signal_type = 'dff_iso'
dec = 2
filt_f = {'DMS': 8, 'PL': 4}

region_colors = {'DMS': '#53C43B', 'PL': '#BB6ED8'}

t = sess_fp['time']

trial_start_ts = sess_fp['trial_start_ts'][:-1]
cue_ts = trial_start_ts + trial_data['response_cue_time']
reward_ts = trial_start_ts + trial_data['reward_time']
reward_ts = reward_ts[trial_data['reward'] > 0]

lines_dict = {'Response Cue': cue_ts, 'Reward Delivery': reward_ts}

t_sel = (t > t_range[0]) & (t < t_range[1])

fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=[8,4*len(regions)], sharex=True)

axs = np.array(axs).reshape((len(regions)))

for i, region in enumerate(regions):
    signal = sess_fp['processed_signals'][region][signal_type]
    filt_signal = fp_utils.filter_signal(signal, filt_f[region], 1/sess_fp['dec_info']['decimated_dt'])
    
    ax = axs[i]
    ax.plot(t[t_sel][::dec], filt_signal[t_sel][::dec], label='_', color=region_colors[region])
    
    for j, (name, lines) in enumerate(lines_dict.items()):
        lines = lines[(lines > t_range[0]) & (lines < t_range[1])]
        ax.vlines(lines, 0, 1, label=name, color='C'+str(j+1), linestyles='dashed', transform=ax.get_xaxis_transform())
    
    _, y_label = fpah.get_signal_type_labels(signal_type)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Time (s)')
    ax.legend()

fpah.save_fig(fig, fpah.get_figure_save_path('Example Signals', '', '_'.join(regions)), format='pdf')

# %%
regions = ['PL', 'DMS', 'DLS', 'TS']
signal_type = 'dff_iso'
signals = [sess_fp['processed_signals'][region][signal_type] for region in regions]
fpah.plot_power_spectra(signals, sess_fp['dec_info']['decimated_dt'], title='{}'.format(signal_type), signal_names=regions)


# %% Compare filtering methods for band-pass filters

from scipy.signal import butter, sosfiltfilt, firwin, filtfilt

signal_type = 'z_dff_iso_baseline'
bands = [[0,0.1], [0.1,1], [1,10], [10,20]]

orders = [2,3]

sess_ids = {402: [119234]}
fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, baseline_correction=True, filter_dropout_outliers=False, reload=False, baseline_lpf=0.0005)

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time']
        sr = 1/np.mean(np.diff(t))
        signals = fp_data[subj_id][sess_id]['processed_signals']
        
        regions = list(signals.keys())
        
        for region in regions:
            signal = signals[region][signal_type]
            
            fig, axs = plt.subplots(len(bands)+1, 1, sharex=True, layout='constrained')
            
            fig.suptitle('Subject {}, Session {}, Region {}'.format(subj_id, sess_id, region))
            
            axs[0].plot(t, signal, label='butter')
            axs[0].set_title('Original')
            
            nans = np.isnan(signal)

            if any(nans):
                signal, _ = fp_utils.fill_signal_nans(signal)

            for i, ax in enumerate(axs[1:]):
                band = bands[i]
                
                ax.set_title('{} - {}'.format(band[0], band[1]))

                # butterworth filter
                for order in orders:
                    if band[0] == 0:
                        sos = butter(order, band[1], btype='lowpass', fs=sr, output='sos')
                    else:
                        sos = butter(order, band, btype='bandpass', fs=sr, output='sos')
                        
                    band_signal = sosfiltfilt(sos, signal)
    
                    if any(nans):
                        band_signal[nans] = np.nan
                    
                    ax.plot(t, band_signal, label='Butter order {}'.format(order))
                
                # # FIR filter
                # if band[0] == 0:
                #     fall_width = 10**np.log10(band[1]*5) - band[1]
                #     numtaps = int(2.2*sr/fall_width)
                #     taps = firwin(numtaps, band[1], pass_zero='lowpass', fs=sr)
                # else:
                #     fall_width = band[0] - 10**np.log10(band[0]/5) + 10**np.log10(band[1]*5) - band[1]
                #     numtaps = int(2.2*sr/fall_width)
                #     taps = firwin(numtaps, band, pass_zero='bandpass', fs=sr)
               
                # band_signal = filtfilt(taps, [1], signal)

                # ax.plot(t, band_signal, label='FIR')
                
                ax.legend()
        
# %%
from scipy import signal
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import numpy as np

bands = [[0,0.1], [0.1,1], [1,10]]
orders = [2,3]
sr = 200

for order in orders:
    for band in bands:
        if band[0] == 0:
            b, a = butter(order, band[1], btype='lowpass', fs=sr)
        else:
            b, a = butter(order, band, btype='bandpass', fs=sr)
    
        w, h = signal.freqz(b, a, fs=sr, worN=10**np.linspace(np.log10(0.0001), np.log10(20), 512))
        plt.semilogx(w, 20 * np.log10(abs(h)), label='{}-{} Hz, Order {}'.format(band[0], band[1], order))
    
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude (dB)')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.legend()
plt.show()

# %%

subj_ids = [198, 199, 274, 400, 402] #179, 180, 182, 188, 191, 202, 207
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

ligs = ['490', '465']
isos = ['420', '405']
#sess_ids = {402: [119234], 400: [119194], 274: [119238]}
#fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, baseline_correction=True, filter_dropout_outliers=False, reload=False, baseline_lpf=0.0005)

bands1 = [[0,0.01], [0.01,0.1], [0.1,0.2], [0.2,0.4], [0.4,0.6], [0.6,0.8], [0.8,1], [1,2], [2,3], [3,4], [4,6], [6,8], [8,10], [10]]
band_names = ['{}-{}'.format(b[0], b[1]) if len(b) == 2 else '{}-'.format(b[0]) for b in bands1]
#bands2 = [[0,0.1], [0.1,2.5], [2.5,10]]
order = 3
dec=10

# build regression formula based on number of bands
param_names1 = ['b{}'.format(i) for i in range(len(bands1))]
expr1 = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names1)]) + ' + c'

param_names1 = param_names1 + ['c']

# Build the lambda string
lambda_str1 = 'lambda x, {}: {}'.format(', '.join(param_names1), expr1)

# Evaluate it
form1 = eval(lambda_str1)

bounds1 = (np.zeros(len(bands1)).tolist(), np.inf)
bounds1[0].append(-np.inf)

coeffs = []

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        
        fp_data = loc_db.get_sess_fp_data(sess_id)
        fp_data = fp_data['fp_data'][subj_id][sess_id]
        
        t = fp_data['time']
        sr = 1/np.mean(np.diff(t))
        
        raw_signals = fp_data['raw_signals']

        for region in raw_signals.keys():

            subj_lig = [k for k in raw_signals[region].keys() if k in ligs][0]
            subj_isos = [k for k in raw_signals[region].keys() if k in isos]

            raw_lig = raw_signals[region][subj_lig]
            
            dropout_ranges = []
            if sess_id in fpah.__sess_dropouts:

                if region in fpah.__sess_dropouts[sess_id]:
                    ranges = fpah.__sess_dropouts[sess_id][region]
                    
                    if not utils.is_list(ranges[0]):
                        ranges = [ranges]
                        
                    dropout_ranges.extend(ranges)
                    
                if 'all' in fpah.__sess_dropouts[sess_id]:
                    ranges = fpah.__sess_dropouts[sess_id]['all']
                    
                    if not utils.is_list(ranges[0]):
                        ranges = [ranges]
                        
                    dropout_ranges.extend(ranges)
                
            if len(dropout_ranges) > 0:

                t_drop = np.full_like(raw_lig, False, dtype=bool)
                for dropout_range in dropout_ranges:
                    t_sel = (t > dropout_range[0]) & (t < dropout_range[1])
                    t_drop[t_sel] = True
                
                raw_lig[t_drop] = np.nan
                    
            
            for iso in subj_isos:
                raw_iso = raw_signals[region][iso]
                
                if len(dropout_ranges) > 0:
                    raw_iso[t_drop] = np.nan

                raw_iso, nans = fp_utils.fill_signal_nans(raw_iso)
                    
                # separate iso into different frequency band components
                pred_mat1 = np.zeros((len(bands1), len(raw_iso)))
                for j, band in enumerate(bands1):
    
                    # butterworth filter
                    if band[0] == 0:
                        sos = butter(order, band[1], btype='lowpass', fs=sr, output='sos')
                    elif len(band) == 1:
                        sos = butter(order, band[0], btype='highpass', fs=sr, output='sos')
                    else:
                        sos = butter(order, band, btype='bandpass', fs=sr, output='sos')
                        
                    pred_mat1[j,:] = sosfiltfilt(sos, raw_iso)
                        
                params1 = curve_fit(form1, pred_mat1[:,~nans], raw_lig[~nans], bounds=bounds1)[0]
                
                band_params = {b: p for b,p in zip(band_names, params1[:-1])}
                band_params['c'] = params1[-1]
                
                fitted_iso = np.full_like(raw_iso, np.nan)
                fitted_iso[~nans] = form1(pred_mat1[:,~nans], *params1)
                
                # fig, ax = plt.subplots(1,1)
                # ax.plot(t[::dec], raw_lig[::dec], label='raw lig')
                # ax.plot(t[::dec], fitted_iso[::dec], label='fitted iso')
                # ax.set_title('Subject {}, Session {}, Region {}, Lig {}, Iso {}'.format(subj_id, sess_id, region, subj_lig, iso))
                # ax.legend()
                # plt.show()
                
                coeffs.append({'subjid': subj_id, 'sessid': sess_id, 'region': region, 'lig': subj_lig, 'iso': iso,
                               **band_params})
                
        print('Processed band loading for subj {} session {}'.format(subj_id, sess_id))
                
                
coeffs = pd.DataFrame.from_dict(coeffs)

# %% Plot coefficients

import seaborn as sb

def trim_region(region):
    idx = region.find('-')

    if idx != -1:
        return region[:idx]
    else:
        return region

plot_coeffs = coeffs.melt(id_vars=['subjid', 'sessid', 'region', 'lig', 'iso'], var_name='band')
plot_coeffs['subjid'] = plot_coeffs['subjid'].astype('category')

plot_coeffs['region'] = plot_coeffs['region'].apply(trim_region)
plot_coeffs['lig/iso'] = plot_coeffs['lig'] + '/' + plot_coeffs['iso']

regions = np.unique(plot_coeffs['region'])

fig, axs = plt.subplots(len(regions), 1, layout='constrained')

for i, region in enumerate(regions):
    reg_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    ax = axs[i]
    sb.stripplot(reg_coeffs, x='band', y='value', hue='subjid', ax=ax)
    ax.set_title(region)
    
fig, axs = plt.subplots(len(regions), 1, layout='constrained')

for i, region in enumerate(regions):
    reg_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    ax = axs[i]
    sb.stripplot(reg_coeffs, x='band', y='value', hue='lig/iso', ax=ax)
    ax.set_title(region)
    
# %% compare frequency band inclusion on recovered signal

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
from hankslab_db import db_access
import pandas as pd 

subj_ids = [199, 400, 402] # 198, 199, 274, 400, 402 #179, 180, 182, 188, 191, 202, 207
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

ligs = ['490', '465']
isos = ['420', '405']
#sess_ids = {402: [119234], 400: [119194], 274: [119238]}
#fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, baseline_correction=True, filter_dropout_outliers=False, reload=False, baseline_lpf=0.0005)

bands1 = [[0,0.01], [0.01,0.1], [0.1,0.2], [0.2,0.4], [0.4,0.6], [0.6,0.8], [0.8,1], [1,4], [4,8], [8,10], [10]]#[[0,0.05], [0.05,0.5], [0.5,1], [1,5], [5,10]]
bands2 = [[0,0.01], [0.01,0.1], [0.1,1], [1,10]]
order = 3
dec = 10

# build regression formula based on number of bands
param_names1 = ['b{}'.format(i) for i in range(len(bands1))]
expr1 = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names1)]) + ' + c'

param_names1 = param_names1 + ['c']

# Build the lambda string
lambda_str1 = 'lambda x, {}: {}'.format(', '.join(param_names1), expr1)

# Evaluate it
form1 = eval(lambda_str1)

bounds1 = (np.zeros(len(bands1)).tolist(), np.inf)
bounds1[0].append(-np.inf)

# build regression formula based on number of bands
param_names2 = ['b{}'.format(i) for i in range(len(bands2))]
expr2 = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names2)]) + ' + c'

param_names2 = param_names2 + ['c']

# Build the lambda string
lambda_str2 = 'lambda x, {}: {}'.format(', '.join(param_names2), expr2)

# Evaluate it
form2 = eval(lambda_str2)

bounds2 = (np.zeros(len(bands2)).tolist(), np.inf)
bounds2[0].append(-np.inf)

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        
        fp_data = loc_db.get_sess_fp_data(sess_id)
        fp_data = fp_data['fp_data'][subj_id][sess_id]
        
        t = fp_data['time']
        sr = 1/np.mean(np.diff(t))
        
        raw_signals = fp_data['raw_signals']
        
        regions = list(raw_signals.keys())
        
        fig, axs = plt.subplots(len(regions), 2, sharex=True, layout='constrained')
        fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(regions):

            lig = [k for k in raw_signals[region].keys() if k in ligs][0]
            iso = [k for k in raw_signals[region].keys() if k in isos][0]

            raw_lig = raw_signals[region][lig]
            raw_iso = raw_signals[region][iso]
            
            dropout_ranges = []
            if sess_id in fpah.__sess_dropouts:

                if region in fpah.__sess_dropouts[sess_id]:
                    ranges = fpah.__sess_dropouts[sess_id][region]
                    
                    if not utils.is_list(ranges[0]):
                        ranges = [ranges]
                        
                    dropout_ranges.extend(ranges)
                    
                if 'all' in fpah.__sess_dropouts[sess_id]:
                    ranges = fpah.__sess_dropouts[sess_id]['all']
                    
                    if not utils.is_list(ranges[0]):
                        ranges = [ranges]
                        
                    dropout_ranges.extend(ranges)
                
            if len(dropout_ranges) > 0:

                t_drop = np.full_like(raw_lig, False, dtype=bool)
                for dropout_range in dropout_ranges:
                    t_sel = (t > dropout_range[0]) & (t < dropout_range[1])
                    t_drop[t_sel] = True
                
                raw_lig[t_drop] = np.nan
                raw_iso[t_drop] = np.nan


            raw_iso, nans = fp_utils.fill_signal_nans(raw_iso)
                
            # separate iso into different frequency band components
            pred_mat1 = np.zeros((len(bands1), len(raw_iso)))
            for j, band in enumerate(bands1):

                # butterworth filter
                if band[0] == 0:
                    sos = butter(order, band[1], btype='lowpass', fs=sr, output='sos')
                elif len(band) == 1:
                    sos = butter(order, band[0], btype='highpass', fs=sr, output='sos')
                else:
                    sos = butter(order, band, btype='bandpass', fs=sr, output='sos')
                    
                pred_mat1[j,:] = sosfiltfilt(sos, raw_iso)
                    
            params1 = curve_fit(form1, pred_mat1[:,~nans], raw_lig[~nans], bounds=bounds1)[0]
            
            fitted_iso1 = np.full_like(raw_iso, np.nan)
            fitted_iso1[~nans] = form1(pred_mat1[:,~nans], *params1)
            
            pred_mat2 = np.zeros((len(bands2), len(raw_iso)))
            for j, band in enumerate(bands2):

                # butterworth filter
                if band[0] == 0:
                    sos = butter(order, band[1], btype='lowpass', fs=sr, output='sos')
                elif len(band) == 1:
                    sos = butter(order, band[0], btype='highpass', fs=sr, output='sos')
                else:
                    sos = butter(order, band, btype='bandpass', fs=sr, output='sos')
                    
                pred_mat2[j,:] = sosfiltfilt(sos, raw_iso)
                    
            params2 = curve_fit(form2, pred_mat2[:,~nans], raw_lig[~nans], bounds=bounds2)[0]
            
            fitted_iso2 = np.full_like(raw_iso, np.nan)
            fitted_iso2[~nans] = form2(pred_mat2[:,~nans], *params2)
            
            ax = axs[i,0]
            ax.plot(t[::dec], raw_lig[::dec], label='raw lig', alpha=0.6)
            ax.plot(t[::dec], fitted_iso1[::dec], label='fitted iso 1', alpha=0.6)
            ax.plot(t[::dec], fitted_iso2[::dec], label='fitted iso 2', alpha=0.6)
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,1]
            ax.plot(t[::dec], raw_lig[::dec]-fitted_iso1[::dec], label='dF 1', alpha=0.6)
            ax.plot(t[::dec], raw_lig[::dec]-fitted_iso2[::dec], label='dF 2', alpha=0.6)
            ax.legend()
            ax.set_title(region)
            
        plt.show()


# %% Plot processed signals

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from os import path
import pickle

subj_ids = [179, 180, 182, 188, 191, 202, 207, 198, 199, 274, 400, 402] # 179, 180, 182, 188, 191, 202, 207
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

r2_vals = []
r_vals = []

filename = 'preprocessing_iso_fit_metrics'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        
        if sess_id in fpah.__sess_ignore:
            continue
        
        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=True, filter_dropout_outliers=True, band_iso_fit=True)
        basic_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=False, tilt_t=False, filter_dropout_outliers=True, band_iso_fit=True)
        fp_data = fp_data[subj_id][sess_id]
        basic_data = basic_data[subj_id][sess_id]
        dt = fp_data['dec_info']['decimated_dt']

        # fig = fpah.view_processed_signals(fp_data['processed_signals'], fp_data['time'], plot_baseline_corr=True, plot_fband=True,
        #                             title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id))
        
        # plt.show()
        signals = fp_data['processed_signals']
        basic_signals = basic_data['processed_signals']

        regions = list(signals.keys())

        for region in regions:
            
            if subj_id in fpah.__region_ignore and region in fpah.__region_ignore[subj_id]:
                continue
            
            # calc R2 for each processing method to compare fit performance
            lig = signals[region]['filtered_lig']
            not_nan = ~np.isnan(lig)
            
            if all(~not_nan):
                continue
            
            lig = lig[not_nan]
            lig_baseline = signals[region]['baseline_lig'][not_nan]
            fitted_iso = basic_signals[region]['fitted_iso'][not_nan]
            fitted_fband_iso = basic_signals[region]['fitted_fband_iso'][not_nan]
            fitted_tilt_iso = signals[region]['fitted_iso'][not_nan]
            fitted_baseline_iso = signals[region]['fitted_baseline_corr_iso'][not_nan] + lig_baseline
            fitted_fband_tilt_iso = signals[region]['fitted_fband_iso'][not_nan]
            
            info = {'subjid': subj_id, 'sessid': sess_id, 'region': region}

            r2_vals.append({**info, 'basic': r2_score(lig, fitted_iso), 'tilt': r2_score(lig, fitted_tilt_iso), 'fband': r2_score(lig, fitted_fband_iso), 
                            'fband & tilt': r2_score(lig, fitted_fband_tilt_iso), 'baseline': r2_score(lig, fitted_baseline_iso), 'null': r2_score(lig, lig_baseline)})
            r_vals.append({**info, 'basic': pearsonr(lig, fitted_iso).statistic, 'tilt': pearsonr(lig, fitted_tilt_iso).statistic, 'fband': pearsonr(lig, fitted_fband_iso).statistic, 
                           'fband & tilt': pearsonr(lig, fitted_fband_tilt_iso).statistic, 'baseline': pearsonr(lig, fitted_baseline_iso).statistic, 'null': pearsonr(lig, lig_baseline).statistic})
        
r2_vals = pd.DataFrame.from_dict(r2_vals)
r_vals = pd.DataFrame.from_dict(r_vals)
        
r2_vals_sorted = r2_vals.sort_values(by='fband')
r_vals_sorted = r_vals.sort_values(by='fband')

r2_vals_sorted_sess = r2_vals.sort_values(by='sessid')
r_vals_sorted_sess = r_vals.sort_values(by='sessid')


with open(save_path, 'wb') as f:
    pickle.dump({'r2_vals': r2_vals,
                 'r_vals': r_vals},
                f)


# %%

import seaborn as sb

def trim_region(region):
    idx = region.find('-')

    if idx != -1:
        return region[:idx]
    else:
        return region

for vals, label in zip([r2_vals, r_vals], ['R2', 'Pearson R']):

    plot_vals = vals.copy()
    plot_vals['subjid'] = plot_vals['subjid'].astype('category')
    plot_vals['region'] = plot_vals['region'].apply(trim_region)
    
    plot_vals['basic'] = plot_vals['basic'] - plot_vals['null']
    plot_vals['tilt'] = plot_vals['tilt'] - plot_vals['null']
    plot_vals['fband'] = plot_vals['fband'] - plot_vals['null']
    plot_vals['fband & tilt'] = plot_vals['fband & tilt'] - plot_vals['null']
    plot_vals['baseline'] = plot_vals['baseline'] - plot_vals['null']
    
    regions = np.unique(plot_vals['region'])
    
    fig_cont, axs_cont = plt.subplots(len(regions), 3, layout='constrained')
    fig_cont.suptitle('Control Fit Comparisons, {}'.format(label))
    
    fig_tilt, axs_tilt = plt.subplots(len(regions), 3, layout='constrained')
    fig_tilt.suptitle('Freq Band w/ Tilt Fit Comparisons, {}'.format(label))
    
    for i, region in enumerate(regions):
        
        reg_vals = plot_vals[plot_vals['region'] == region]
        
        ax = axs_cont[i,0]
        sb.scatterplot(reg_vals, x='tilt', y='basic', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Tilt vs. Basic, {}'.format(region))
        
        ax = axs_cont[i,1]
        sb.scatterplot(reg_vals, x='fband', y='basic', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band vs. Basic, {}'.format(region))
        
        ax = axs_cont[i,2]
        sb.scatterplot(reg_vals, x='fband & tilt', y='fband', hue='subjid', ax=ax)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band w/ Tilt vs. Freq Band, {}'.format(region))

        handles_cont, labels_cont = ax.get_legend_handles_labels()
        ax.legend().remove()
        
        
        ax = axs_tilt[i,0]
        sb.scatterplot(reg_vals, x='fband & tilt', y='basic', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band w/ Tilt vs. Basic, {}'.format(region))
        
        ax = axs_tilt[i,1]
        sb.scatterplot(reg_vals, x='fband & tilt', y='tilt', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band w/ Tilt vs. Tilt, {}'.format(region))
        
        ax = axs_tilt[i,2]
        sb.scatterplot(reg_vals, x='fband & tilt', y='baseline', hue='subjid', ax=ax)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band w/ Tilt vs. Baseline, {}'.format(region))

        handles_tilt, labels_tilt = ax.get_legend_handles_labels()
        ax.legend().remove()
        
    fig_tilt.legend(handles_cont, labels_cont, loc='outside right upper')
    fig_tilt.legend(handles_tilt, labels_tilt, loc='outside right upper')
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.9)
