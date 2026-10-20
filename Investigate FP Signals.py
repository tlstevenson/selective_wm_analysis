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

wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
rl_loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')


# %% View preprocessing

reload = False
baseline_lpf = 0.0005
allow_neg_coeffs = False

bands = [[0,0.5], [0.5,5], [5]]

#sess_ids = {180: [101447]}
#sess_ids = {191: [101617]}

#fp_data, implant_info = fpah.load_fp_data(wm_loc_db, sess_ids, iso_lpf=10)
#sess_data = wm_loc_db.get_behavior_data(utils.flatten(sess_ids))

# sess_ids = {400: [119785]} #199: [117251], , 402: [119793]}
# fp_data, implant_info = fpah.load_fp_data(rl_loc_db, sess_ids)
# sess_data = rl_loc_db.get_behavior_data(utils.flatten(sess_ids))

# subj_ids = [421,422]#,235,419,421,422] #198,199,237,238,274,400,402,424,483
# sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)

# sess_ids = {subj: [sess for sess in sess_ids[subj] if sess in fpah.__sess_dropouts] for subj in subj_ids}

sess_ids = {422: [130182]}
# sess_ids = {237: [124598, 124771, 124940, 124950, 124973, 125128, 125155, 125179],
#  238: [124607, 125109, 125146],
#  274: [118992, 119187],
#  400: [119194, 119247, 119285, 119785, 119974, 120001, 120103],
#  402: [120066],
#  424: [125129],
#  483: [124606, 124655, 124735, 124746, 124769, 124948, 124957, 125000, 125102, 125164]}

# n_back = 4
# sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# sess_ids = {402: [119234], 400: [119194,119713,119247], 274: [119238]}

gen_title = 'Subject {}, Session {}'
sub_t = [0, np.inf] # [1100, 1120] #
dec = 2

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        fp_data, implant_info = fpah.load_fp_data(wm_loc_db, {subj_id: [sess_id]}, traditional_fit=True, baseline_iso_fit=True, band_iso_fit=False,
                                                  baseline_band_iso_fit=True, irls_fit=False, baseline_irls_fit=False, tilt_t=False, reload=reload,
                                                  save_processed_signals=False) #allow_neg_coeffs=allow_neg_coeffs, iso_bands=bands, baseline_lpf=baseline_lpf,
        t = fp_data[subj_id][sess_id]['time']

        fpah.view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'], t, t_min=sub_t[0], t_max=sub_t[1], dec=dec,
                                    title=gen_title.format(subj_id, sess_id), plot_traditional=True, plot_baseline=True, plot_fband=False,
                                    plot_baseline_fband=True, plot_irls=True, plot_baseline_irls=False)

        plt.show()

# %% update timestamps
# reload = False
# subj_ids = [400, 402, 424, 483]
# sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)

# skip_sess = [119000,119194,119233,119247,119285,119429,
#              119479,119713,119741,119785,119830,119974]
# skip_subj = []

# completed_sess = []

# for subj_id in sess_ids.keys():
#     if subj_id in skip_subj:
#         continue

#     for sess_id in sess_ids[subj_id]:

#         if sess_id in skip_sess:
#             continue

#         fp_data = wm_loc_db.get_sess_fp_data(sess_id, reload=reload)
#         fp_data = fp_data['fp_data'][subj_id][sess_id]
#         t = fp_data['time']
#         dec_info = fp_data['dec_info']
#         fix_t = t - dec_info['initial_dt']

#         time_data = {'start': fix_t[0], 'end': fix_t[-1], 'dt': dec_info['decimated_dt'], 'length': len(t), 'dec_info': dec_info}

#         for region in fp_data['raw_signals'].keys():
#             db_access.update_fp_time_data(subj_id, sess_id, region, time_data)

#         completed_sess.append(sess_id)

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

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+1), linestyles='dashed', transform=ax.get_xaxis_transform())

            ax.set_title(region)
            _, y_label = fpah.get_signal_type_labels(signal_type)
            ax.set_ylabel(y_label)
            ax.set_xlabel('Time (s)')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()


# %% create plots

sess_ids = {274: [119418]}

subj_ids = list(sess_ids.keys())
dec=2
signal_types = ['dff_iso_baseline_fband']

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:

        fp_data, implant_info = fpah.load_fp_data(wm_loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=False, filter_dropout_outliers=False, band_iso_fit=False,
                                                  baseline_band_iso_fit=True, irls_fit=False, reload=reload, baseline_lpf=baseline_lpf, allow_neg_coeffs=allow_neg_coeffs)
        sess_fp = fp_data[subj_id][sess_id]
        trial_data = wm_loc_db.get_behavior_data(sess_id)

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

        lines_dict = {#'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, 'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts,
                      #'Response': response_ts, 'Reward': reward_ts
                     }

        plot_signal_details(sess_fp['processed_signals'], sess_fp['time'], 'Subject {}, Session {}'.format(subj_id, sess_id), lines_dict, dec=dec, signal_types=signal_types)


# %% Create cleaned up example data plot

subj_id = 400
sess_id = 120001
sess_ids = {subj_id: [sess_id]}
fp_data, implant_info = fpah.load_fp_data(rl_loc_db, sess_ids, baseline_correction=True, tilt_t=False, filter_dropout_outliers=False, band_iso_fit=False,
                                          baseline_band_iso_fit=True, irls_fit=False, reload=False)
trial_data = rl_loc_db.get_behavior_data(sess_id)

sess_fp = fp_data[subj_id][sess_id]

# %%
t_range = [651, 681] #[0, np.inf] #[2730, 2780] #[1155, 1195] # [1150, 1250] #
regions = ['PL', 'NAc', 'DMS', 'DLS']
signal_type = 'dff_iso_baseline_fband'
dec = 2
filt_f = {'DMS': 10, 'DLS': 10, 'NAc': 10, 'PL': 5}

region_colors = {'DMS': '#53C43B', 'DLS': '#E87A2C', 'NAc': '#47BBBF', 'PL': '#BB6ED8'}

t = sess_fp['time']

trial_start_ts = sess_fp['trial_start_ts'][:-1]
cue_ts = trial_start_ts + trial_data['response_cue_time']
outcome_ts = trial_start_ts + trial_data['reward_time']
reward_ts = outcome_ts[trial_data['reward'] > 0]
unreward_ts = outcome_ts[trial_data['reward'] == 0]

lines_dict = {'Response Cue': cue_ts, 'Reward Delivery': reward_ts, 'Unreward Tone': unreward_ts}
lines_colors = {'Response Cue': '#FFBB00', 'Reward Delivery': '#3399CC', 'Unreward Tone': '#FC606D'}

t_sel = (t > t_range[0]) & (t < t_range[1])

fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=[9,3*len(regions)], sharex=True)

axs = np.array(axs).reshape((len(regions)))

for i, region in enumerate(regions):
    signal = sess_fp['processed_signals'][region][signal_type]
    filt_signal = fp_utils.filter_signal(signal, filt_f[region], 1/sess_fp['dec_info']['decimated_dt'])

    ax = axs[i]
    ax.plot(t[t_sel][::dec], filt_signal[t_sel][::dec], label='_', color=region_colors[region])

    for j, (name, lines) in enumerate(lines_dict.items()):
        lines = lines[(lines > t_range[0]) & (lines < t_range[1])]
        ax.vlines(lines, 0, 1, label=name, color=lines_colors[name], linestyles='dashed', transform=ax.get_xaxis_transform())

    _, y_label = fpah.get_signal_type_labels(signal_type)
    ax.set_title(region)
    ax.set_ylabel(y_label)
    ax.set_xlabel('Time (s)')

    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

fig.legend(handles, labels, loc='outside right center')

fpah.save_fig(fig, fpah.get_figure_save_path('Example Signals', '', '_'.join(regions)), format='pdf')

# %% Investigate power spectra method differences
import scipy.signal as sig

sess_ids = {400: [119000]}
signal_types = ['dff_iso_baseline', 'dff_iso_fband', 'dff_iso_baseline_fband']

x_lims = [0.0005, 10]

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        fp_data, implant_info = fpah.load_fp_data(wm_loc_db, {subj_id: [sess_id]}, baseline_correction=True, tilt_t=True, filter_dropout_outliers=False, band_iso_fit=True,
                                                  baseline_band_iso_fit=True, irls_fit=False, reload=False, baseline_lpf=0.0005)
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

subj_ids = [198, 199, 274, 400, 402, 237, 238, 424, 483] #179, 180, 182, 188, 191, 202, 207
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)
loc_db = rl_db.LocalDB_BasicRLTasks('twoArmBandit')

recalculate = False

# freq_vals = [0] + list(np.logspace(np.log10(0.01), np.log10(10), 15))
# freq_vals = np.round(freq_vals, 3)
# bands = list(zip(freq_vals[:-1], freq_vals[1:]))
# bands.append([10])

bands = [[0,0.001], [0.001,0.01], [0.01,0.1], [0.1,0.2], [0.2,0.3], [0.3,0.4], [0.4,0.5], [0.5,0.6], [0.6,0.7], [0.7,0.8],
         [0.8,0.9], [0.9,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,8], [8,9], [9,10], [10]]
band_names = ['{}-{}'.format(b[0], b[1]) if len(b) == 2 else '{}-'.format(b[0]) for b in bands]

filename = 'preprocessing_fband_fit_coeffs'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        coeff_dict = saved_data['coeff_dict']
        saved_bands = saved_data['bands']

        if saved_bands != bands:
            recalculate = True

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    coeff_dict = {}

for subj_id in sess_ids.keys():
    if subj_id not in coeff_dict:
        coeff_dict[subj_id] = {}

    for sess_id in sess_ids[subj_id]:

        if sess_id in fpah.__sess_ignore:
            continue

        if sess_id in coeff_dict[subj_id] and not recalculate:
            continue

        if sess_id not in coeff_dict[subj_id]:
            coeff_dict[subj_id][sess_id] = []

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, band_iso_fit=True, baseline_band_iso_fit=True,
                               tilt_t=False, filter_dropout_outliers=False, iso_bands=bands,
                               allow_neg_coeffs=False, ignore_sess_start=True, interp_reg_coeffs=True)

        subj_regions = list(fp_data[subj_id][sess_id]['processed_signals'].keys())

        for region in subj_regions:
            processed_signals = fp_data[subj_id][sess_id]['processed_signals'][region]
            lig = processed_signals['lig']
            iso = processed_signals['iso']
            fit_info = processed_signals['iso_fit_info']

            for i in range(len(fit_info['baseline_fband'])):
                fband_params = {b: p for b,p in zip(band_names, fit_info['fband'][i]['params'][:-1])}
                fband_params['c'] = fit_info['fband'][i]['params'][-1]

                baseline_fband_params = {b: p for b,p in zip(band_names, fit_info['baseline_fband'][i]['params'][:-1])}
                baseline_fband_params['c'] = fit_info['baseline_fband'][i]['params'][-1]

                coeff_dict[subj_id][sess_id].append({'subjid': subj_id, 'sessid': sess_id, 'region': region, 'lig': lig, 'iso': iso, 'type': 'fband',
                               **fband_params})
                coeff_dict[subj_id][sess_id].append({'subjid': subj_id, 'sessid': sess_id, 'region': region, 'lig': lig, 'iso': iso, 'type': 'baseline_fband',
                               **baseline_fband_params})

        print('Processed band loading for subj {} session {}'.format(subj_id, sess_id))

        with open(save_path, 'wb') as f:
            pickle.dump({'coeff_dict': coeff_dict, 'bands': bands}, f)

# convert dictionary to pandas dataframe for plotting
pd_data = []

for subj_id in coeff_dict.keys():
    for sess_id in coeff_dict[subj_id].keys():
        pd_data.extend(coeff_dict[subj_id][sess_id])

coeffs = pd.DataFrame.from_dict(pd_data)

# %% Plot coefficients

import seaborn as sb

def trim_region(region):
    idx = region.find('-')

    if idx != -1:
        return region[:idx]
    else:
        return region

rescale = True
plot_c = False
region_order = ['PL', 'NAc', 'DMS', 'DLS', 'TS']
id_cols = ['subjid', 'sessid', 'region', 'lig', 'iso', 'type']
band_cols = [c for c in coeffs.columns if c not in id_cols]
if not plot_c:
    band_cols = [c for c in band_cols if c != 'c']

plot_type = 'baseline_fband' # 'fband', 'baseline_fband'

plot_coeffs = coeffs[coeffs['type'] == plot_type]

if rescale:
    max_coeffs = plot_coeffs[band_cols].max(axis=1)
    coeff_sel = max_coeffs > 1e-5
    plot_coeffs.loc[coeff_sel, band_cols] = plot_coeffs.loc[coeff_sel, band_cols].div(max_coeffs[coeff_sel], axis=0)

plot_coeffs = plot_coeffs.melt(id_vars=id_cols, value_vars=band_cols, var_name='band')
plot_coeffs['subjid'] = plot_coeffs['subjid'].astype('category')

plot_coeffs['region'] = plot_coeffs['region'].apply(trim_region)
plot_coeffs['lig/iso'] = plot_coeffs['lig'] + '/' + plot_coeffs['iso']

subj_ids = np.unique(plot_coeffs['subjid'])
wavelengths = np.unique(plot_coeffs['lig/iso'])
regions = np.unique(plot_coeffs['region'])
regions = [r for r in region_order if r in regions]

fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=(12,3*len(regions)))

for i, region in enumerate(regions):
    reg_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    ax = axs[i]
    sb.stripplot(reg_coeffs, x='band', y='value', hue='subjid', hue_order=subj_ids, ax=ax, alpha=0.5)
    sb.pointplot(reg_coeffs, x='band', y='value', hue='subjid', hue_order=subj_ids, ax=ax, alpha=0.5, dodge=True)
    ax.set_title(region)
    ax.set_xlabel('Frequency Band (Hz)')
    ax.set_ylabel('Coefficient Value')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

fig.legend(handles, labels, loc='outside right upper')

fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=(12,3*len(regions)))

for i, region in enumerate(regions):
    reg_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    ax = axs[i]
    sb.stripplot(reg_coeffs, x='band', y='value', hue='lig/iso', hue_order=wavelengths, ax=ax, alpha=0.5)
    sb.pointplot(reg_coeffs, x='band', y='value', hue='lig/iso', hue_order=wavelengths, ax=ax, alpha=0.5, dodge=True)
    ax.set_title(region)
    ax.set_xlabel('Frequency Band (Hz)')
    ax.set_ylabel('Coefficient Value')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

fig.legend(handles, labels, loc='outside right upper')

# %%
# calculate and plot correlations between adjacent frequency band coefficients
fband_corrs = []
plot_coeffs = coeffs[coeffs['type'] == plot_type]
for subj_id in subj_ids:
    subj_coeffs = plot_coeffs[plot_coeffs['subjid'] == subj_id]
    subj_regions = np.unique(subj_coeffs['region'])
    for region in subj_regions:
        region_coeffs = subj_coeffs[subj_coeffs['region'] == region]
        region_results = {'subjid': str(subj_id), 'region': region}
        for band1, band2 in zip(band_cols[:-1], band_cols[1:]):
            region_results['{}/{}'.format(band1, band2)] = pearsonr(region_coeffs[band1], region_coeffs[band2]).statistic

        fband_corrs.append(region_results)

for region in regions:
    region_coeffs = plot_coeffs[plot_coeffs['region'] == region]
    region_results = {'subjid': 'all', 'region': region}
    for band1, band2 in zip(band_cols[:-1], band_cols[1:]):
        region_results['{}/{}'.format(band1, band2)] = pearsonr(region_coeffs[band1], region_coeffs[band2]).statistic

    fband_corrs.append(region_results)

fband_corrs = pd.DataFrame.from_dict(fband_corrs)
band_corr_cols = ['{}/{}'.format(band1, band2) for band1, band2 in zip(band_cols[:-1], band_cols[1:])]
fband_corrs = fband_corrs.melt(id_vars=['subjid', 'region'], value_vars=band_corr_cols, var_name='band_corr')
fband_corrs['subjid'] = fband_corrs['subjid'].astype('category')
fband_corrs['region'] = fband_corrs['region'].apply(trim_region)
subj_order = [str(s) for s in subj_ids]+['all']

fig, axs = plt.subplots(len(regions), 1, layout='constrained', figsize=(12,3*len(regions)))

for i, region in enumerate(regions):
    reg_corrs = fband_corrs[fband_corrs['region'] == region]
    ax = axs[i]
    plot_utils.plot_dashlines(0, dir='h', ax=ax)
    sb.lineplot(reg_corrs, x='band_corr', y='value', hue='subjid', hue_order=subj_order, ax=ax, alpha=0.5, markers=True, errorbar=None)
    ax.set_title(region)
    ax.set_xlabel('Frequency Bands (Hz)')
    ax.set_ylabel('Pearson r')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

fig.legend(handles, labels, loc='outside right upper')


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

freq_bands = [[0,0.1], [0.1,1], [1,10]]#[[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,2], [2,6], [6,10]]
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

        # fig_recon, axs_recon = plt.subplots(n_regions, 1, sharey=False, sharex=True, layout='constrained', figsize=(20,n_regions*5))
        # fig_recon.suptitle('Subject {}, Session {}, Frequency Band Reconstruction'.format(subj_id, sess_id))

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

        # for i, region in enumerate(regions):
        #     ax = axs_recon[i]

        #     ax.set_title(region)
        #     ax.plot(t[::dec], sig_full[region][::dec], label='True Signal')
        #     ax.plot(t[::dec], sig_recon[region][::dec], label='Reconstructed')
        #     ax.set_xlabel('Time (s)')
        #     ax.set_ylabel('Signal (dF/F)')
        #     ax.legend()