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
import beh_analysis_helpers as bah
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.signal import butter, sosfiltfilt
from hankslab_db import db_access
import pandas as pd
import time

from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from os import path
import pickle


# %% View preprocessing
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

#sess_ids = {402: [119793]}

subj_ids = [400, 402]
n_back = 2
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
sess_ids = bah.limit_sess_ids(sess_ids, n_back)

gen_title = 'Subject {}, Session {}'
sub_t = [0, np.inf] # [1100, 1120] #
dec = 2

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        fp_data, implant_info = fpah.load_fp_data(wm_loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=False, filter_dropout_outliers=False, band_iso_fit=False, 
                                                  baseline_band_iso_fit=True, irls_fit=False, reload=reload, baseline_lpf=baseline_lpf)
        t = fp_data[subj_id][sess_id]['time']

        fpah.view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'], t, t_min=sub_t[0], t_max=sub_t[1], dec=dec,
                                    title=gen_title.format(subj_id, sess_id), plot_baseline_corr=True, plot_fband=False, plot_baseline_fband=True, plot_irls=False)
        
        plt.show()

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

# %% Investigate power spectra method differences
import scipy.signal as sig

sess_ids = {199: [117450], 400: [119233], 274: [119464]}
signal_types = ['dff_iso_baseline', 'z_dff_iso_baseline', 'dff_iso_fband', 'z_dff_iso_fband']

x_lims = [0.0005, 10]

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        fp_data, implant_info = fpah.load_fp_data(wm_loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=True, filter_dropout_outliers=False, band_iso_fit=True, 
                                                  irls_fit=False, reload=False, baseline_lpf=0.0005)
        fp_data = fp_data[subj_id][sess_id]
        dt = fp_data['dec_info']['decimated_dt']
        
        # full signal
        n_sigs = len(signal_types)
        regions = list(fp_data['processed_signals'].keys())
        n_regs = len(regions)
        
        fig, axs = plt.subplots(n_sigs, 4, sharey=True, sharex=True, layout='constrained', figsize=(20,n_sigs*5))
        
        for i, signal_type in enumerate(signal_types):
            for region in regions:
                if not region in fp_data['processed_signals']:
                    continue
                
                signal = fp_data['processed_signals'][region][signal_type]
                signal, _ = fp_utils.fill_signal_nans(signal)
                n = len(signal)
                
                # full periodogram
                ps = np.abs(np.fft.rfft(signal*np.hanning(n)))**2/(n/dt)
                freqs = np.fft.rfftfreq(n, dt)
                
                freq_sel = (freqs > x_lims[0]) & (freqs < x_lims[1])
                ax = axs[i,0]
                ax.loglog(freqs[freq_sel], ps[freq_sel], alpha=0.7, label=region)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density (V^2/Hz)')
                ax.set_title('Full Periodogram ({})'.format(signal_type))
                ax.legend()
                
                # welch without detrending
                nperseg = round(1/dt*2/0.005)

                freqs, ps = sig.welch(signal, fs=1/dt, nperseg=nperseg, scaling='density', detrend=False)
                freq_sel = (freqs > x_lims[0]) & (freqs < x_lims[1])
                ax = axs[i,1]
                ax.loglog(freqs[freq_sel], ps[freq_sel], alpha=0.7, label=region)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density (V^2/Hz)')
                ax.set_title('Welch Periodogram ({})'.format(signal_type))
                ax.legend()
                
                # welch with constant detrending
                freqs, ps = sig.welch(signal, fs=1/dt, nperseg=nperseg, scaling='density', detrend='constant')
                freq_sel = (freqs > x_lims[0]) & (freqs < x_lims[1])
                ax = axs[i,2]
                ax.loglog(freqs[freq_sel], ps[freq_sel], alpha=0.7, label=region)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density (V^2/Hz)')
                ax.set_title('Welch Periodogram Constant Detrend ({})'.format(signal_type))
                ax.legend()
                
                # welch with linear detrending
                freqs, ps = sig.welch(signal, fs=1/dt, nperseg=nperseg, scaling='density', detrend='linear')
                freq_sel = (freqs > x_lims[0]) & (freqs < x_lims[1])
                ax = axs[i,3]
                ax.loglog(freqs[freq_sel], ps[freq_sel], alpha=0.7, label=region)
                ax.set_xlabel('Frequency (Hz)')
                ax.set_ylabel('Power Spectral Density (V^2/Hz)')
                ax.set_title('Welch Periodogram Linear Detrend ({})'.format(signal_type))
                ax.legend()
                
        plt.show()

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

# %% Investigate frequency band regression coefficients

subj_ids = [179, 180, 182, 188, 191, 202, 207, 198, 199, 274, 400, 402] #179, 180, 182, 188, 191, 202, 207
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

ligs = ['490', '465']
isos = ['420', '415', '405']

bands = [[0,0.01], [0.01,0.1], [0.1,0.2], [0.2,0.4], [0.4,0.6], [0.6,0.8], [0.8,1], [1,2], [2,3], [3,4], [4,6], [6,8], [8,10], [10]]
band_names = ['{}-{}'.format(b[0], b[1]) if len(b) == 2 else '{}-'.format(b[0]) for b in bands]

order = 3
baseline_lpf = 0.0005

# build regression formula based on number of bands
param_names = ['b{}'.format(i) for i in range(len(bands))]
expr = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names)]) + ' + c'

param_names = param_names + ['c']

# Build the lambda string
lambda_str = 'lambda x, {}: {}'.format(', '.join(param_names), expr)

# Evaluate it
form = eval(lambda_str)

bounds = (np.zeros(len(bands)).tolist(), np.inf)
bounds[0].append(-np.inf)

filename = 'preprocessing_fband_fit_coeffs'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

coeffs = []

for subj_id in sess_ids.keys():
    reload = subj_id in [274, 400, 402]
    
    for sess_id in sess_ids[subj_id]:
        
        fp_data = loc_db.get_sess_fp_data(sess_id, reload=reload)
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
            
            trend_pad_len = int((1/(2*baseline_lpf))*sr)
            baseline_lig = fp_utils.filter_signal(raw_lig, baseline_lpf, sr, trend_pad_len=trend_pad_len)
            baseline_corr_lig = raw_lig - baseline_lig
            
            for iso in subj_isos:
                raw_iso = raw_signals[region][iso]
                
                if len(dropout_ranges) > 0:
                    raw_iso[t_drop] = np.nan

                raw_iso, nans = fp_utils.fill_signal_nans(raw_iso)
                
                baseline_iso = fp_utils.filter_signal(raw_iso, baseline_lpf, sr, trend_pad_len=trend_pad_len)
                baseline_corr_iso = raw_iso - baseline_iso
                    
                # separate iso into different frequency band components
                pred_mat = np.zeros((len(bands), len(raw_iso)))
                for j, band in enumerate(bands):
    
                    # butterworth filter
                    if band[0] == 0:
                        sos = butter(order, band[1], btype='lowpass', fs=sr, output='sos')
                    elif len(band) == 1:
                        sos = butter(order, band[0], btype='highpass', fs=sr, output='sos')
                    else:
                        sos = butter(order, band, btype='bandpass', fs=sr, output='sos')
                        
                    pred_mat[j,:] = sosfiltfilt(sos, baseline_corr_iso)
                        
                params = curve_fit(form, pred_mat[:,~nans], baseline_corr_lig[~nans], bounds=bounds)[0]
                
                band_params = {b: p for b,p in zip(band_names, params[:-1])}
                band_params['c'] = params[-1]
                
                # fitted_iso = np.full_like(raw_iso, np.nan)
                # fitted_iso[~nans] = form(pred_mat[:,~nans], *params)
                
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

with open(save_path, 'wb') as f:
    pickle.dump(coeffs, f)

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

subj_ids = np.unique(plot_coeffs['subjid'])
wavelengths = np.unique(plot_coeffs['lig/iso'])

fig, axs = plt.subplots(len(regions), 1, layout='constrained')

for i, region in enumerate(regions):
    reg_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    ax = axs[i]
    sb.stripplot(reg_coeffs, x='band', y='value', hue='subjid', hue_order=subj_ids, ax=ax, alpha=0.5)
    ax.set_title(region)
    ax.set_xlabel('Frequency Band (Hz)')
    ax.set_ylabel('Coefficient Value')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    
fig.legend(handles, labels, loc='outside right upper')
    
fig, axs = plt.subplots(len(regions), 1, layout='constrained')

for i, region in enumerate(regions):
    reg_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    ax = axs[i]
    sb.stripplot(reg_coeffs, x='band', y='value', hue='lig/iso', hue_order=wavelengths, ax=ax, alpha=0.5)
    ax.set_title(region)
    ax.set_xlabel('Frequency Band (Hz)')
    ax.set_ylabel('Coefficient Value')
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()
    
fig.legend(handles, labels, loc='outside right upper')
    
# %% compare frequency band inclusion on recovered signal

subj_ids = [402] # 198, 199, 274, 400, 402 #179, 180, 182, 188, 191, 202, 207
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

ligs = ['490', '465']
isos = ['420', '415', '405']

#sess_ids = {402: [119234], 400: [119194], 274: [119238]}
#fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, baseline_correction=True, filter_dropout_outliers=False, reload=False, baseline_lpf=0.0005)

bands1 = [[0,0.01], [0.01,0.1], [0.1,1], [1,10]] #[[0,0.05], [0.05,0.5], [0.5,1], [1,5], [5,10]]
bands2 = [[0,0.1], [0.1,1], [1,10]]
order = 3
dec = 10
baseline_lpf = 0.0005

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

            trend_pad_len = int((1/(2*baseline_lpf))*sr)
            baseline_lig = fp_utils.filter_signal(raw_lig, baseline_lpf, sr, trend_pad_len=trend_pad_len)
            baseline_iso = fp_utils.filter_signal(raw_iso, baseline_lpf, sr, trend_pad_len=trend_pad_len)
            baseline_corr_lig = raw_lig - baseline_lig
            baseline_corr_iso = raw_iso - baseline_iso
                
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
                    
                pred_mat1[j,:] = sosfiltfilt(sos, baseline_corr_iso)
                    
            params1 = curve_fit(form1, pred_mat1[:,~nans], baseline_corr_lig[~nans], bounds=bounds1)[0]
            
            fitted_iso1 = np.full_like(baseline_corr_iso, np.nan)
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
                    
                pred_mat2[j,:] = sosfiltfilt(sos, baseline_corr_iso)
                    
            params2 = curve_fit(form2, pred_mat2[:,~nans], baseline_corr_lig[~nans], bounds=bounds2)[0]
            
            fitted_iso2 = np.full_like(baseline_corr_iso, np.nan)
            fitted_iso2[~nans] = form2(pred_mat2[:,~nans], *params2)
            
            ax = axs[i,0]
            ax.plot(t[::dec], baseline_corr_lig[::dec], label='raw lig', alpha=0.6)
            ax.plot(t[::dec], fitted_iso1[::dec], label='fitted iso 1', alpha=0.6)
            ax.plot(t[::dec], fitted_iso2[::dec], label='fitted iso 2', alpha=0.6)
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,1]
            ax.plot(t[::dec], baseline_corr_lig[::dec]-fitted_iso1[::dec], label='dF 1', alpha=0.6)
            ax.plot(t[::dec], baseline_corr_lig[::dec]-fitted_iso2[::dec], label='dF 2', alpha=0.6)
            ax.legend()
            ax.set_title(region)
            
        plt.show()


# %% Calculate metrics for how well the isosbestic fits to the ligand channel for multiple processing methods

subj_ids = [179, 180, 182, 188, 191, 202, 207, 198, 199, 274, 400, 402]
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

iso_bands = [[0,0.01], [0.01,0.1], [0.1,1], [1,10]]
baseline_iso_bands = [[0,0.1], [0.1,1], [1,10]]

r2_vals = []
r_vals = []

filename = 'preprocessing_iso_fit_metrics'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        
        if sess_id in fpah.__sess_ignore:
            continue
        
        basic_fband_fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=False, tilt_t=False, filter_dropout_outliers=True, band_iso_fit=True, 
                                                   baseline_band_iso_fit=False, iso_bands=iso_bands)
        tilt_fband_fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=False, tilt_t=True, filter_dropout_outliers=True, band_iso_fit=True, 
                                                  baseline_band_iso_fit=False, iso_bands=iso_bands)
        baseline_fband_fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=False, filter_dropout_outliers=True, band_iso_fit=False, 
                                                      baseline_band_iso_fit=True, iso_bands=baseline_iso_bands)
        
        dt = basic_fband_fp_data[subj_id][sess_id]['dec_info']['decimated_dt']
        
        basic_fband_signals = basic_fband_fp_data[subj_id][sess_id]['processed_signals']
        tilt_fband_signals = tilt_fband_fp_data[subj_id][sess_id]['processed_signals']
        baseline_fband_signals = baseline_fband_fp_data[subj_id][sess_id]['processed_signals']

        regions = list(basic_fband_signals.keys())

        for region in regions:
            
            if subj_id in fpah.__region_ignore and region in fpah.__region_ignore[subj_id]:
                continue
            
            # calc R2 for each processing method to compare fit performance
            lig = basic_fband_signals[region]['filtered_lig']
            not_nan = ~np.isnan(lig)
            
            if all(~not_nan):
                continue
            
            lig = lig[not_nan]
            lig_baseline = baseline_fband_signals[region]['baseline_lig'][not_nan]
            fitted_iso = basic_fband_signals[region]['fitted_iso'][not_nan]
            fitted_fband_iso = basic_fband_signals[region]['fitted_fband_iso'][not_nan]
            fitted_tilt_iso = tilt_fband_signals[region]['fitted_iso'][not_nan]
            fitted_fband_tilt_iso = tilt_fband_signals[region]['fitted_fband_iso'][not_nan]
            fitted_baseline_iso = baseline_fband_signals[region]['fitted_baseline_corr_iso'][not_nan] + lig_baseline
            fitted_baseline_fband_iso = baseline_fband_signals[region]['fitted_baseline_fband_iso'][not_nan] + lig_baseline
            
            
            info = {'subjid': subj_id, 'sessid': sess_id, 'region': region}

            r2_vals.append({**info, 'basic': r2_score(lig, fitted_iso), 'tilt': r2_score(lig, fitted_tilt_iso), 'fband': r2_score(lig, fitted_fband_iso), 
                            'fband & tilt': r2_score(lig, fitted_fband_tilt_iso), 'baseline': r2_score(lig, fitted_baseline_iso), 
                            'baseline & fband': r2_score(lig, fitted_baseline_fband_iso), 'null': r2_score(lig, lig_baseline)})
            r_vals.append({**info, 'basic': pearsonr(lig, fitted_iso).statistic, 'tilt': pearsonr(lig, fitted_tilt_iso).statistic, 'fband': pearsonr(lig, fitted_fband_iso).statistic, 
                           'fband & tilt': pearsonr(lig, fitted_fband_tilt_iso).statistic, 'baseline': pearsonr(lig, fitted_baseline_iso).statistic, 
                           'baseline & fband': pearsonr(lig, fitted_baseline_fband_iso).statistic, 'null': pearsonr(lig, lig_baseline).statistic})
        
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


# %% plot comparisons

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
    plot_vals['baseline & fband'] = plot_vals['baseline & fband'] - plot_vals['null']
    
    regions = np.unique(plot_vals['region'])
    
    fig_cont, axs_cont = plt.subplots(len(regions), 4, layout='constrained')
    fig_cont.suptitle('Control Fit Comparisons, {}'.format(label))
    
    fig_main, axs_main = plt.subplots(len(regions), 4, layout='constrained')
    fig_main.suptitle('Main Fit Comparisons, {}'.format(label))
    
    for i, region in enumerate(regions):
        
        reg_vals = plot_vals[plot_vals['region'] == region]
        
        ax = axs_cont[i,0]
        sb.scatterplot(reg_vals, x='fband', y='basic', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band vs. Basic, {}'.format(region))
        
        ax = axs_cont[i,1]
        sb.scatterplot(reg_vals, x='baseline', y='basic', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Baseline vs. Basic, {}'.format(region))
        
        ax = axs_cont[i,2]
        sb.scatterplot(reg_vals, x='fband & tilt', y='fband', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band w/ Tilt vs. Freq Band, {}'.format(region))
        
        ax = axs_cont[i,3]
        sb.scatterplot(reg_vals, x='fband & tilt', y='baseline', hue='subjid', ax=ax)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Freq Band w/ Tilt vs. Baseline, {}'.format(region))

        handles_cont, labels_cont = ax.get_legend_handles_labels()
        ax.legend().remove()
        
        
        ax = axs_main[i,0]
        sb.scatterplot(reg_vals, x='baseline & fband', y='basic', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Baseline & Freq Band vs. Basic, {}'.format(region))
        
        ax = axs_main[i,1]
        sb.scatterplot(reg_vals, x='baseline & fband', y='baseline', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Baseline & Freq Band vs. Baseline, {}'.format(region))
        
        ax = axs_main[i,2]
        sb.scatterplot(reg_vals, x='baseline & fband', y='fband', hue='subjid', ax=ax, legend=False)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Baseline & Freq Band vs. Freq Band, {}'.format(region))
        
        ax = axs_main[i,3]
        sb.scatterplot(reg_vals, x='baseline & fband', y='fband & tilt', hue='subjid', ax=ax)
        plot_utils.plot_unity_line(ax)
        ax.set_title('Baseline & Freq Band vs. Freq Band w/ Tilt, {}'.format(region))
        

        handles_main, labels_main = ax.get_legend_handles_labels()
        ax.legend().remove()
        
    fig_cont.legend(handles_cont, labels_cont, loc='outside right upper')
    fig_main.legend(handles_main, labels_main, loc='outside right upper')
    # plt.tight_layout()
    # plt.subplots_adjust(right=0.9)


# %% Investigate frequency bands across regions

freq_bands = [[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,2], [2,6], [6,10]]
order = 3
dec = 2

sess_ids = {199: [117450], 400: [119785], 402: [120009], 274: [119740]}
signal_type = 'z_dff_iso_baseline_fband'
reload = True

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        fp_data, implant_info = fpah.load_fp_data(rl_loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=False, 
                                                  filter_dropout_outliers=False, band_iso_fit=False, baseline_band_iso_fit=True,
                                                  irls_fit=False, reload=False, baseline_lpf=0.0005)
        
        trial_data = rl_loc_db.get_behavior_data(sess_id, reload=reload)

        fp_data = fp_data[subj_id][sess_id]
        t = fp_data['time']
        dt = fp_data['dec_info']['decimated_dt']
        sr = 1/dt
        
        trial_start_ts = fp_data['trial_start_ts'][:-1]

        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Response': response_ts, #'Cpoke Out': cpoke_out_ts, 
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        regions = list(fp_data['processed_signals'].keys())
        n_regions = len(regions)
        n_bands = len(freq_bands)
        
        fig_split, axs_split = plt.subplots(n_bands, 1, sharey=False, sharex=True, layout='constrained', figsize=(20,n_bands*5))
        fig_split.suptitle('Subject {}, Session {}, Frequency Band Signals'.format(subj_id, sess_id))
        
        fig_recon, axs_recon = plt.subplots(n_regions, 1, sharey=False, sharex=True, layout='constrained', figsize=(20,n_regions*5))
        fig_recon.suptitle('Subject {}, Session {}, Frequency Band Reconstruction'.format(subj_id, sess_id))
        
        sig_recon = {r: np.zeros_like(t) for r in regions}
        sig_full = {}
        
        for i, band in enumerate(freq_bands):
            ax = axs_split[i]
            
            for region in regions:
                if not region in fp_data['processed_signals']:
                    continue
                
                signal = fp_data['processed_signals'][region][signal_type]
                sig_full[region] = signal 
                signal, nans = fp_utils.fill_signal_nans(signal)
                
                if band[0] == 0:
                    sos = butter(order, band[1], btype='lowpass', fs=sr, output='sos')
                elif len(band) == 1:
                    sos = butter(order, band[0], btype='highpass', fs=sr, output='sos')
                else:
                    sos = butter(order, band, btype='bandpass', fs=sr, output='sos')
                    
                sig_band = sosfiltfilt(sos, signal)
                sig_band[nans] = np.nan
                
                sig_recon[region] = sig_recon[region] + sig_band
                
                ax.plot(t[::dec], sig_band[::dec], label=region, alpha=0.6)
                
            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, linestyles='dashed', color='C{}'.format(j+n_regions),
                          transform=ax.get_xaxis_transform()) 
            
            ax.legend()
            ax.set_title('{} - {} Hz'.format(band[0], band[1]))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Band Signal (dF/F)')
            
        for i, region in enumerate(regions):
            ax = axs_recon[i]
            
            ax.set_title(region)
            ax.plot(t[::dec], sig_full[region][::dec], label='True Signal')
            ax.plot(t[::dec], sig_recon[region][::dec], label='Reconstructed')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal (dF/F)')
            ax.legend()