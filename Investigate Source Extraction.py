# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 14:22:49 2026

@author: Tanner
"""

# %% imports

import init
import pandas as pd
import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
from hankslab_db import db_access
from hankslab_db import tonecatdelayresp_db as wm_db, basicRLtasks_db as bandit_db
import fp_analysis_helpers as fpah
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import os.path as path
import pickle

from caiman.source_extraction.cnmf.deconvolution import estimate_time_constant, GetSn
from scipy.signal import lfilter
from scipy.linalg import toeplitz

# %% Declare common functions and settings

# used for saving plots
wm_beh_name = 'Single Tone WM'
wm_sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=7)

bandit_beh_name = 'Two-armed Bandit'
bandit_sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2)

tasks = ['wm', 'bandit']
beh_names = {'wm': 'Single Tone WM', 'bandit': 'Probabilistic Bandit'}

# optionally limit sessions based on subject ids
subj_ids = [198, 199, 274, 400, 402, 237, 238, 424, 483]
wm_sess_ids = {k: [s for s in v if not s in fpah.__sess_ignore] for k, v in wm_sess_ids.items() if k in subj_ids}
bandit_sess_ids = {k: [s for s in v if not s in fpah.__sess_ignore] for k, v in bandit_sess_ids.items() if k in subj_ids}

subj_ids = np.unique(list(wm_sess_ids.keys())+list(bandit_sess_ids.keys()))

wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

implant_info = db_access.get_fp_implant_info(subj_ids)

all_regions = ['PL', 'NAc', 'DMS', 'DLS', 'TS']
region_colors = {'PL': '#BB6ED8', 'NAc': '#47BBBF', 'DMS': '#53C43B', 'DLS': '#E87A2C', 'TS': '#E6B900'}

tilt_t = False
traditional_fit = True
baseline_iso_fit = False
baseline_band_iso_fit = True
band_iso_fit = False

def sort_subj_regions(subj_regions):
    return fpah.sort_subj_regions(subj_regions, all_regions)

def get_subj_region_mapping(subj_regions):
    return {r: [k for k in all_regions if k in r][0] for r in subj_regions}

def get_subj_regions(subj_id):
    subj_regions = list(implant_info[subj_id].keys())

    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]

    return sort_subj_regions(subj_regions)

# %% Investigate source extraction whitening methods

plot_fits = False

dec = 2
ar_orders = [1,2,3,4]
filtered_opts = [False, True]
g_types = ['full'] #, 'real'

region_lpf = {'PL': 4, 'NAc': 6}
region_lags = {'PL': 10, 'NAc': 4, 'DMS': 1, 'DLS': 1, 'TS': 1}
region_ar_lags = {'PL': 2, 'NAc': 1, 'DMS': 0.15, 'DLS': 0.15, 'TS': 0.15}

recalculate = False
reprocess_sess_ids = []
reprocess_reg = []

signal_type = 'dff_iso_baseline_fband'

exp_func2 = lambda x,a,tau1,tau2: a*np.exp(-x/tau1) + (1-a)*np.exp(-x/tau2)
bounds2 = ([0]*3,
          [1, np.inf, np.inf])

sim_noise = np.random.randn(500000)

def simulate_ar_corr(g, n_lags, noise=None, n=100000):

    p = len(g)

    if noise is None:
        noise = np.random.randn(n)
    else:
        n = len(noise)

    x = np.zeros(n)

    for t in range(p, n):
        ar = noise[t]
        for i in range(p):
            ar += g[i]*x[t-i-1]
        x[t] = ar

    sim_ar2_corr, _ = fp_utils.correlate(x, x, 1, neg_lag=0, pos_lag=n_lags-1)
    return sim_ar2_corr

def fit_ar_yw(x, p, lags):
    tmp_x, nans = fp_utils.fill_signal_nans(x, extrapolate_edges=False)
    ext_nans = np.isnan(tmp_x)
    sn = GetSn(tmp_x[~ext_nans])

    lags += p
    xc, _ = fp_utils.correlate(x, x, 1, neg_lag=p, pos_lag=lags, mode='covariance')
    xc = xc[:, np.newaxis]

    A = toeplitz(c=np.ravel(xc[p + np.arange(lags)]),
                 r=np.ravel(xc[p + np.arange(p)])) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, xc[p+1:], rcond=None)[0]
    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))
    # stabilize unstable roots
    mask = np.abs(gr) > 1
    gr[mask] = gr[mask] / np.abs(gr[mask]) * 0.99

    g = -np.poly(gr)[1:]

    return g.flatten()

def get_real_g(g):

    gr = np.roots(np.concatenate([np.array([1]), -g.flatten()]))

    gr_new = []

    for r in gr:
        if np.imag(r) > 0:
            m = np.abs(r)
            gr_new.extend([m, m])

        elif np.imag(r) == 0:
            gr_new.append(np.real(r))

    g_new = -np.poly(gr_new)[1:].flatten()

    return g_new

def extract_ar_source(x, g):
    # inverse filter to recover the source of the AR process
    a = np.concatenate([[1], -g])

    return lfilter(a, [1], x)


tasks = ['wm', 'bandit']

filename = 'wm_bandit_ar_source_param_sweep'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        ar_source_results = saved_data['ar_source_results']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    ar_source_results = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    subj_reprocess_reg = [s for s in subj_regions if reg_mapping[s] in reprocess_reg]

    if not subj_id in ar_source_results:
        ar_source_results[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids[::2]:

            if sess_id in ar_source_results[subj_id][task] and not sess_id in reprocess_sess_ids and len(subj_reprocess_reg) == 0:
                continue

            print('Comparing AR source extraction for subject {}, task {}, session {}'.format(subj_id, task, sess_id))

            ar_source_results[subj_id][task][sess_id] = {
                r: {
                    ar_order: {
                        filtered: {
                            g_type: {}
                            for g_type in g_types
                            if ar_order > 1 or g_type == 'full'
                        }
                        for filtered in filtered_opts
                    }
                    for ar_order in ar_orders}
                for r in subj_regions}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']

            if plot_fits:
                n_rows = len(ar_orders)
                fig, axs = plt.subplots(n_rows, len(subj_regions), figsize=(5*len(subj_regions),4*n_rows), layout='constrained')
                axs = axs.reshape((n_rows, len(subj_regions)))
                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

            for reg_i, region in enumerate(subj_regions):
                if not region in fp_data['processed_signals']:
                    continue
                raw_signal = fp_data['processed_signals'][region][signal_type]

                if all(np.isnan(raw_signal)):
                    continue

                if reg_mapping[region] in region_lpf:
                    lpf_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
                else:
                    lpf_signal = raw_signal

                dec_lpf_signal, _ = fp_utils.decimate(lpf_signal, t, dec, dt=dt)

                if plot_fits:
                    for order_i, ar_order in enumerate(ar_orders):
                        ax = axs[order_i, reg_i]
                        ax.set_title('{} AR({})'.format(region, ar_order))
                        ax.set_xlabel('Lag (s)')
                        if reg_i == 0:
                            ax.set_ylabel('Autocorr')
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)

                for filtered in filtered_opts:

                    if filtered and not reg_mapping[region] in region_lpf:
                        continue

                    if filtered:
                        signal = lpf_signal
                    else:
                        signal = raw_signal

                    dec_signal, _ = fp_utils.decimate(signal, t, dec, dt=dt)

                    corr_lag_lim = region_lags[reg_mapping[region]]
                    ar_lags = int(region_ar_lags[reg_mapping[region]] / (dec*dt))

                    auto_corr, corr_lags = fp_utils.correlate(dec_signal, dec_signal, dt*dec, neg_lag=0, pos_lag=corr_lag_lim)

                    if plot_fits:
                        for order_i, ar_order in enumerate(ar_orders):
                            ax = axs[order_i, reg_i]
                            ax.plot(corr_lags, auto_corr, label='Autocorr, filt={}'.format(filtered), alpha=0.5)

                    for order_i, ar_order in enumerate(ar_orders):

                        full_g = fit_ar_yw(dec_signal, ar_order, ar_lags)

                        for g_type in ar_source_results[subj_id][task][sess_id][region][ar_order][filtered].keys():

                            if g_type == 'real':
                                g = get_real_g(full_g)
                            else:
                                g = full_g

                            source = extract_ar_source(dec_lpf_signal, g)

                            resid_autocorr, _ = fp_utils.correlate(source, source, dt*dec, neg_lag=0, pos_lag=corr_lag_lim)

                            sim_autocorr = simulate_ar_corr(g, len(corr_lags), noise=sim_noise)

                            q_sel = np.full(len(corr_lags), False)
                            first_cross = np.argmax((resid_autocorr < 0) | np.isclose(resid_autocorr, 0))
                            q_sel[first_cross:] = True
                            Q = np.nansum(resid_autocorr[q_sel]**2)

                            ar_source_results[subj_id][task][sess_id][region][ar_order][filtered][g_type] = {
                                'g': g,
                                'Q': Q,
                                'first_cross': first_cross,
                                'autocorr': auto_corr,
                                'corr_lags': corr_lags,
                                'simulated_autocorr': sim_autocorr,
                                'residual_autocorr': resid_autocorr
                            }

                            if plot_fits:
                                ax = axs[order_i, reg_i]
                                ax.plot(corr_lags, sim_autocorr, label='sim, {}, filt={}'.format(g_type, filtered), alpha=0.5)
                                ax.plot(corr_lags, resid_autocorr, label='resid, {}, filt={}, Q={:.3f}'.format(g_type, filtered, Q), alpha=0.5)

                if plot_fits:
                    for order_i, ar_order in enumerate(ar_orders):
                        ax = axs[order_i, reg_i]
                        ax.legend(fontsize=8)

            if plot_fits:
                plt.show()


        with open(save_path, 'wb') as f:
            pickle.dump({'ar_source_results': ar_source_results}, f)

# %% Compare whitening method results

ar_compare_df = []

for subj_id, subj_dict in ar_source_results.items():
    subj_region_mapping = get_subj_region_mapping(get_subj_regions(subj_id))
    for task, task_dict in subj_dict.items():
        for sess_id, sess_dict in task_dict.items():
            for region, region_dict in sess_dict.items():
                for ar_order, order_dict in region_dict.items():
                    for filtered, filt_dict in order_dict.items():
                        for g_type, result_dict in filt_dict.items():

                            if len(result_dict) == 0:
                                continue

                            autocorr = np.asarray(result_dict['autocorr'])
                            sim_autocorr = np.asarray(result_dict['simulated_autocorr'])

                            n = min(len(autocorr), len(sim_autocorr))
                            autocorr = autocorr[:n]
                            sim_autocorr = sim_autocorr[:n]

                            mse = np.nanmean((autocorr - sim_autocorr)**2)

                            ar_compare_df.append({'subj_id': subj_id, 'task': task, 'sess_id': sess_id, 'region': region,
                                'gen_region': subj_region_mapping[region], 'ar_order': ar_order, 'filtered': 'filtered' if filtered else 'unfiltered',
                                'g_type': g_type, 'Q': result_dict['Q'], 'autocorr_mse': mse
                            })

ar_compare_df = pd.DataFrame(ar_compare_df)

# nicer labels for plotting
ar_compare_df['subj_id'] = ar_compare_df['subj_id'].astype('category')
ar_compare_df['model'] = (
    'AR(' + ar_compare_df['ar_order'].astype(str) + '), ' +
    ar_compare_df['filtered'] + ', ' +
    ar_compare_df['g_type'].astype(str)
)

# Plot per-subject Q and MSE

metric_labels = [('Q', 'Residual autocorr Q'), ('autocorr_mse', 'Autocorr simulation MSE')]
n_rows = len(metric_labels)*2
n_cols = len(all_regions)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,3*n_rows), layout='constrained', sharey=False)
fig.suptitle('AR Whitening Order Comparison by Subject')
for i1, (metric, ylabel) in enumerate(metric_labels):
    for i2, filt in enumerate(['filtered', 'unfiltered']):
        for j, region in enumerate(all_regions):
            plot_df = ar_compare_df[(ar_compare_df['filtered'] == filt) & (ar_compare_df['gen_region'] == region)]
            ax = axs[i1*2+i2,j]
            sb.pointplot(plot_df, x='ar_order', y=metric, hue='subj_id',
                errorbar='se', dodge=True, ax=ax)

            ax.set_xlabel('AR Order')
            ax.set_ylabel(ylabel)
            ax.set_title('{}, {}, {}'.format(region, ylabel, filt))
            ax.legend(ncols=2, fontsize=6)

n_rows = len(metric_labels)
fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols,3*n_rows), layout='constrained', sharey=False)
fig.suptitle('AR Whitening Order Comparison, Subject Avg')
for i, (metric, ylabel) in enumerate(metric_labels):
    for j, region in enumerate(all_regions):
        plot_df = ar_compare_df[ar_compare_df['gen_region'] == region]
        ax = axs[i,j]
        sb.pointplot(plot_df, x='ar_order', y=metric, hue='filtered', errorbar='se', ax=ax)

        ax.set_xlabel('AR Order')
        ax.set_ylabel(ylabel)
        ax.set_title('{}, {}'.format(region, ylabel))
        ax.legend()

# %% Compare AR whitening methods on specific sessions

comparison_options = {'PL': [{'filtered': False, 'ar_order': 1}, {'filtered': False, 'ar_order': 2}],
                      'NAc': [{'filtered': False, 'ar_order': 1}, {'filtered': False, 'ar_order': 2}],
                      'DMS': [{'filtered': False, 'ar_order': 1}, {'filtered': False, 'ar_order': 2}],
                      'DLS': [{'filtered': False, 'ar_order': 1}, {'filtered': False, 'ar_order': 2}],
                      'TS': [{'filtered': False, 'ar_order': 1}, {'filtered': False, 'ar_order': 2}]}
sess_ids = {199: [116828, 117565], 237: [124638, 124973], 400: [119713, 119974], 402: [119481, 119742], 424: [124644, 124772], 483: [124924, 124957]} #
loc_db = bandit_loc_db

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for sess_id in sess_ids[subj_id]:
        task = 'wm' if sess_id in wm_sess_ids[subj_id] else 'bandit'

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        t = fp_data['time']
        dt = fp_data['dec_info']['decimated_dt']

        n_rows = len(subj_regions)
        fig, axs = plt.subplots(n_rows, 1, figsize=(20,4*n_rows), layout='constrained', sharex=True)
        fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(subj_regions):
            if not region in fp_data['processed_signals']:
                continue
            raw_signal = fp_data['processed_signals'][region][signal_type]

            if all(np.isnan(raw_signal)):
                continue

            if reg_mapping[region] in region_lpf:
                lpf_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
            else:
                lpf_signal = raw_signal

            dec_lpf_signal, dec_t = fp_utils.decimate(lpf_signal, t, dec, dt=dt)

            region_options = comparison_options[reg_mapping[region]]
            ax = axs[i]
            ax.set_title(region)
            ax.plot(dec_t, utils.z_score(dec_lpf_signal), label='Signal', alpha=0.5)

            for opt in region_options:
                opt_g = ar_source_results[subj_id][task][sess_id][region][opt['ar_order']][opt['filtered']]['full']['g']
                source = extract_ar_source(dec_lpf_signal, opt_g)
                ax.plot(dec_t, utils.z_score(source), label='AR({}), {}'.format(opt['ar_order'], opt['filtered']), alpha=0.5)

            ax.legend()

        plt.show()

# %% Compare AR(1), AR(2), and pseudo AR(1) source extraction methods

from caiman.source_extraction.cnmf.deconvolution import GetSn
from scipy.signal import lfilter
from scipy.linalg import toeplitz

plot_fits = False

dec = 2

region_lpf = {'PL': 4, 'NAc': 6}
region_lags = {'PL': 4, 'NAc': 2, 'DMS': 1, 'DLS': 1, 'TS': 1}
#region_ar_lags = {'PL': 4, 'NAc': 0.5, 'DMS': 0.3, 'DLS': 0.3, 'TS': 0.3}
region_ar_lags = {'PL': 2, 'NAc': 0.5, 'DMS': 0.15, 'DLS': 0.15, 'TS': 0.15}
xcorr_lag = 1

recalculate = False
reprocess_sess_ids = []
reprocess_reg = []

signal_type = 'dff_iso_baseline_fband'

#filename = 'wm_bandit_ar_source_params_xcorr_pl_lag1'
filename = 'wm_bandit_ar_source_params_xcorr_pl_lag2'
#filename = 'wm_bandit_ar_source_params_xcorr_pl_lag4_DS_lag03'

sim_noise = np.random.randn(500000)

def simulate_ar_corr(g, n_lags, noise=None, n=100000):

    p = len(g)

    if noise is None:
        noise = np.random.randn(n)
    else:
        n = len(noise)

    x = np.zeros(n)

    for t in range(p, n):
        ar = noise[t]
        for i in range(p):
            ar += g[i]*x[t-i-1]
        x[t] = ar

    sim_corr, _ = fp_utils.correlate(x, x, 1, neg_lag=0, pos_lag=n_lags-1)
    return sim_corr

def get_ar_roots(g):
    return np.roots(np.concatenate([[1], -np.array(g)]))

def fit_ar_yw(x, p, lags):
    tmp_x, nans = fp_utils.fill_signal_nans(x, extrapolate_edges=False)
    ext_nans = np.isnan(tmp_x)
    sn = GetSn(tmp_x[~ext_nans])

    ac, _ = fp_utils.correlate(x, x, 1, neg_lag=p, pos_lag=lags+p, mode='covariance')
    return fit_acf_ar_yw(ac, p, lags+p, sn=sn)

def fit_acf_ar_yw(ac, p, lags, sn=0):
    ac = ac[:, np.newaxis]

    A = toeplitz(c=np.ravel(ac[p + np.arange(lags)]),
                 r=np.ravel(ac[p + np.arange(p)])) - sn**2 * np.eye(lags, p)
    g = np.linalg.lstsq(A, ac[p+1:lags+p+1], rcond=None)[0]
    gr = get_ar_roots(g.flatten())
    # stabilize unstable roots
    mask = np.abs(gr) > 1
    gr[mask] = gr[mask] / np.abs(gr[mask]) * 0.99

    g = -np.poly(gr)[1:]

    return g.flatten()

def get_real_g(g):

    gr = get_ar_roots(g)

    gr_new = []

    for r in gr:
        if np.imag(r) > 0:
            m = np.abs(r)
            gr_new.extend([m, m])

        elif np.imag(r) == 0:
            gr_new.append(np.real(r))

    g_new = -np.poly(gr_new)[1:].flatten()

    return g_new

def extract_ar_source(x, g):
    # inverse filter to recover the source of the AR process
    a = np.concatenate([[1], -np.array(g)])

    return lfilter(a, [1], x)

def get_ar_modes(g, dt):
    gr = get_ar_roots(g)

    modes = []

    for r in gr:
        # only include one complex conjugate pair
        if np.imag(r) > 0 or np.imag(r) == 0:
            mag = np.abs(r)
            theta = np.angle(r)
            if theta == np.pi:
                theta = 0
            tau = -dt/np.log(mag)
            freq = theta/(2*np.pi*dt)

            modes.append({'root': r, 'mag': mag, 'theta': theta, 'tau': tau, 'freq': freq})

    return modes

process_names = ['true', 'ar1', 'ar2', 'ar2to1']
tasks = ['wm', 'bandit']

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        ar_source_xcorr = saved_data['ar_source_xcorr']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    ar_source_xcorr = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    subj_reprocess_reg = [s for s in subj_regions if reg_mapping[s] in reprocess_reg]

    if not subj_id in ar_source_xcorr:
        ar_source_xcorr[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in ar_source_xcorr[subj_id][task] and not sess_id in reprocess_sess_ids and len(subj_reprocess_reg) == 0:
                continue

            print('Comparing AR source extraction for subject {}, task {}, session {}'.format(subj_id, task, sess_id))

            ar_source_xcorr[subj_id][task][sess_id] = {'modes': {p: {} for p in process_names},
                                                       'autocorrs': {p: {} for p in process_names},
                                                       'autocorr_lags': {},
                                                       'xcorrs': {p: {} for p in process_names},
                                                       'xcorr_lags': {}}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']

            if plot_fits:
                n_rows = 1+len(process_names)
                fig, axs = plt.subplots(n_rows, len(subj_regions), figsize=(5*len(subj_regions),4*n_rows), layout='constrained')
                axs = axs.reshape((n_rows, len(subj_regions)))
                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

            reg_sources = {p: {} for p in process_names}

            for reg_i, region in enumerate(subj_regions):
                if not region in fp_data['processed_signals']:
                    continue
                raw_signal = fp_data['processed_signals'][region][signal_type]

                if all(np.isnan(raw_signal)):
                    continue

                if reg_mapping[region] in region_lpf:
                    filt_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
                else:
                    filt_signal = raw_signal

                dec_raw_signal, _ = fp_utils.decimate(raw_signal, t, dec, dt=dt)
                dec_filt_signal, _ = fp_utils.decimate(filt_signal, t, dec, dt=dt)
                reg_sources['true'][region] = dec_filt_signal

                corr_lag_lim = region_lags[reg_mapping[region]]
                ar_lags = int(region_ar_lags[reg_mapping[region]] / (dec*dt))

                auto_corr, corr_lags = fp_utils.correlate(dec_raw_signal, dec_raw_signal, dt*dec, neg_lag=0, pos_lag=corr_lag_lim)
                ar_source_xcorr[subj_id][task][sess_id]['autocorrs']['true'][region] = auto_corr
                ar_source_xcorr[subj_id][task][sess_id]['autocorr_lags'][region] = corr_lags

                # calculate parameters
                g_ar1 = fit_ar_yw(dec_raw_signal, 1, ar_lags)
                ar1_resid = extract_ar_source(dec_filt_signal, g_ar1)
                ar1_modes = get_ar_modes(g_ar1, dt*dec)
                reg_sources['ar1'][region] = ar1_resid
                ar_source_xcorr[subj_id][task][sess_id]['modes']['ar1'][region] = ar1_modes
                sim_autocorr_ar1 = simulate_ar_corr(g_ar1, len(corr_lags), noise=sim_noise)
                ar_source_xcorr[subj_id][task][sess_id]['autocorrs']['ar1'][region] = sim_autocorr_ar1

                g_ar2 = fit_ar_yw(dec_raw_signal, 2, ar_lags)
                ar2_resid = extract_ar_source(dec_filt_signal, g_ar2)
                ar2_modes = get_ar_modes(g_ar2, dt*dec)
                reg_sources['ar2'][region] = ar2_resid
                ar_source_xcorr[subj_id][task][sess_id]['modes']['ar2'][region] = ar2_modes
                sim_autocorr_ar2 = simulate_ar_corr(g_ar2, len(corr_lags), noise=sim_noise)
                ar_source_xcorr[subj_id][task][sess_id]['autocorrs']['ar2'][region] = sim_autocorr_ar2

                g_ar2to1 = [max([m['mag'] for m in ar2_modes])]
                ar2to1_resid = extract_ar_source(dec_filt_signal, g_ar2to1)
                ar2to1_modes = get_ar_modes(g_ar2to1, dt*dec)
                reg_sources['ar2to1'][region] = ar2to1_resid
                ar_source_xcorr[subj_id][task][sess_id]['modes']['ar2to1'][region] = ar2to1_modes
                sim_autocorr_ar2to1 = simulate_ar_corr(g_ar2to1, len(corr_lags), noise=sim_noise)
                ar_source_xcorr[subj_id][task][sess_id]['autocorrs']['ar2to1'][region] = sim_autocorr_ar2to1

                # plot autocorrelations
                if plot_fits:
                    ax = axs[0, reg_i]
                    plot_utils.plot_dashlines(0, dir='h', ax=ax)

                    ax.plot(corr_lags, auto_corr, label='Signal', alpha=0.75)

                    ax.plot(corr_lags, sim_autocorr_ar1, label='Sim AR(1)', alpha=0.5)
                    ax.plot(corr_lags, sim_autocorr_ar2, label='Sim AR(2)', alpha=0.5)
                    ax.plot(corr_lags, sim_autocorr_ar2to1, label='Sim AR(2)->AR(1)', alpha=0.5)

                    resid_autocorr_ar1, _ = fp_utils.correlate(ar1_resid, ar1_resid, dt*dec, neg_lag=0, pos_lag=region_lags[reg_mapping[region]])
                    resid_autocorr_ar2, _ = fp_utils.correlate(ar2_resid, ar2_resid, dt*dec, neg_lag=0, pos_lag=region_lags[reg_mapping[region]])
                    resid_autocorr_ar2to1, _ = fp_utils.correlate(ar2to1_resid, ar2to1_resid, dt*dec, neg_lag=0, pos_lag=region_lags[reg_mapping[region]])

                    ax.plot(corr_lags, resid_autocorr_ar1, label='Resid AR(1)', alpha=0.5)
                    ax.plot(corr_lags, resid_autocorr_ar2, label='Resid AR(2)', alpha=0.5)
                    ax.plot(corr_lags, resid_autocorr_ar2to1, label='Resid AR(2)->AR(1)', alpha=0.5)

                    ax.set_title('{} Autocorrelations'.format(region))
                    ax.legend(fontsize=8)

            # calculate cross-correlations for each option
            if plot_fits:
                for p_i, p in enumerate(process_names):
                    for reg_i, region_i in enumerate(subj_regions):
                        ax = axs[1+p_i, reg_i]
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=ax)
                        ax.set_title('{}, Residual Xcorr {}'.format(region_i, p))
                        ax.set_xlabel('Lag (s)')
                        if reg_i == 0:
                            ax.set_ylabel('Pearson r')

            sess_xcorrs = {}
            for p_i, p in enumerate(process_names):
                sess_xcorrs[p] = {r: {} for r in subj_regions}

                for reg_i, region_i in enumerate(subj_regions):

                    if region_i not in reg_sources[p]:
                        continue

                    resid_i = reg_sources[p][region_i]

                    for reg_j, region_j in enumerate(subj_regions):

                        if region_j not in reg_sources[p]:
                            continue

                        resid_j = reg_sources[p][region_j]

                        if not region_j in sess_xcorrs[p][region_i]:
                            xcorr, xcorr_lags = fp_utils.correlate(resid_i, resid_j, dt*dec, neg_lag=xcorr_lag, pos_lag=xcorr_lag)

                            sess_xcorrs[p][region_i][region_j] = xcorr
                            sess_xcorrs[p][region_j][region_i] = np.flip(xcorr)

                        if plot_fits and reg_i != reg_j:
                            ax = axs[1+p_i, reg_i]
                            ax.plot(xcorr_lags, sess_xcorrs[p][region_i][region_j], label=region_j, alpha=0.6)
                            ax.legend(fontsize=8)


            ar_source_xcorr[subj_id][task][sess_id]['xcorrs'] = sess_xcorrs
            ar_source_xcorr[subj_id][task][sess_id]['xcorr_lags'] = xcorr_lags

            if plot_fits:
                plt.show()

            with open(save_path, 'wb') as f:
                pickle.dump({'ar_source_xcorr': ar_source_xcorr}, f)


# %% Aggregate average source correlations

stacked_autocorrs = {s: {p: {r: [] for r in get_subj_regions(s)} for p in process_names} for s in subj_ids}
stacked_autocorrs['all'] = {p: {r: [] for r in all_regions} for p in process_names}

stacked_xcorrs = {s: {p: {r: {r: [] for r in get_subj_regions(s)} for r in get_subj_regions(s)} for p in process_names} for s in subj_ids}
stacked_xcorrs['all'] = {p: {r: {r: [] for r in all_regions} for r in all_regions} for p in process_names}

autocorr_lags = {r: [] for r in all_regions}
xcorr_lags = []

ar_modes = {s: {p: [] for p in process_names} for s in subj_ids}
ar_modes['all'] = {p: [] for p in process_names}

process_labels = {'true': 'Signal', 'ar1': 'AR(1)', 'ar2': 'AR(2)', 'ar2to1': 'AR(2)->AR(1)'}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    subj_unique_regions = np.unique(list(reg_mapping.values()))

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:
            sess_autocorr_lags = ar_source_xcorr[subj_id][task][sess_id]['autocorr_lags']
            xcorr_lags = ar_source_xcorr[subj_id][task][sess_id]['xcorr_lags']

            for p in process_names:

                autocorrs = ar_source_xcorr[subj_id][task][sess_id]['autocorrs'][p]
                xcorrs = ar_source_xcorr[subj_id][task][sess_id]['xcorrs'][p]

                all_region_autocorrs = {r: [] for r in subj_unique_regions}
                all_region_xcorrs = {r1: {r2: [] for r2 in subj_unique_regions} for r1 in subj_unique_regions}

                for region_i in subj_regions:

                    if region_i not in xcorrs or region_i not in autocorrs:
                        continue

                    autocorr_lags[reg_mapping[region_i]] = sess_autocorr_lags[region_i]

                    stacked_autocorrs[subj_id][p][region_i].append(autocorrs[region_i])
                    all_region_autocorrs[reg_mapping[region_i]].append(autocorrs[region_i])

                    if p != 'true':
                        max_ar_mode = {'tau': 0}
                        for ar_mode in ar_source_xcorr[subj_id][task][sess_id]['modes'][p][region_i]:
                            if ar_mode['tau'] > max_ar_mode['tau']:
                                max_ar_mode = ar_mode

                        if max_ar_mode['tau'] > 0:
                            max_ar_mode = max_ar_mode.copy()
                            max_ar_mode['region'] = region_i
                            max_ar_mode['general_region'] = reg_mapping[region_i]
                            max_ar_mode['task'] = task
                            max_ar_mode['subj_id'] = subj_id
                            ar_modes[subj_id][p].append(max_ar_mode)
                            ar_modes['all'][p].append(max_ar_mode)

                    for region_j in subj_regions:

                        if region_j not in xcorrs[region_i]:
                            continue

                        stacked_xcorrs[subj_id][p][region_i][region_j].append(xcorrs[region_i][region_j])

                        # don't map bilateral implants to autocorrelations
                        if reg_mapping[region_i] != reg_mapping[region_j] or region_i == region_j:
                            all_region_xcorrs[reg_mapping[region_i]][reg_mapping[region_j]].append(xcorrs[region_i][region_j])

                for region_i in subj_unique_regions:

                    if len(all_region_autocorrs[region_i]) > 0:
                        avg_autocorr = np.mean(np.stack(all_region_autocorrs[region_i], axis=0), axis=0)
                        stacked_autocorrs['all'][p][region_i].append(avg_autocorr)

                    for region_j in subj_unique_regions:

                        if len(all_region_xcorrs[region_i][region_j]) == 0:
                            continue

                        # average bilateral regions per session
                        avg_xcorr = np.mean(np.stack(all_region_xcorrs[region_i][region_j], axis=0), axis=0)
                        stacked_xcorrs['all'][p][region_i][region_j].append(avg_xcorr)

# convert to stacked matrices
subjs = []
subjs.extend(subj_ids)
subjs.append('all')
for subj_id in subjs:
    if subj_id == 'all':
        subj_regions = all_regions
    else:
        subj_regions = get_subj_regions(subj_id)

    for p in process_names:
        for region_i in subj_regions:
            stacked_autocorrs[subj_id][p][region_i] = np.stack(stacked_autocorrs[subj_id][p][region_i], axis=0)

            for region_j in subj_regions:
                stacked_xcorrs[subj_id][p][region_i][region_j] = np.stack(stacked_xcorrs[subj_id][p][region_i][region_j], axis=0)


# %% plot average auto and cross-correlations for each processing method

plot_ind_subj = False
plot_meta_subj = True

plot_subj = []
if plot_ind_subj:
    plot_subj.extend(subj_ids)
if plot_meta_subj:
    plot_subj.append('all')

# plot correlations
for subj_id in plot_subj:

    if subj_id == 'all':
        subj_regions = all_regions
        reg_colors = region_colors
    else:
        subj_regions = get_subj_regions(subj_id)
        reg_colors = {r: 'C{}'.format(i) for i, r in enumerate(subj_regions)}

    reg_mapping = get_subj_region_mapping(subj_regions)

    fig_autocorr, axs_autocorr = plt.subplots(1, len(subj_regions), figsize=(5*len(subj_regions),4), layout='constrained', sharey=True)
    fig_autocorr.suptitle('Source Deconvolution Autocorrelation Comparison, Subject {}'.format(subj_id))

    n_rows = len(process_names)
    fig_xcorr, axs_xcorr = plt.subplots(n_rows, len(subj_regions), figsize=(5*len(subj_regions),4*n_rows), layout='constrained', sharey=True)
    fig_xcorr.suptitle('Source Deconvolution Crosscorrelation Comparison, Subject {}'.format(subj_id))

    for reg_i, region in enumerate(subj_regions):

        # plot average autocorrelations
        ax = axs_autocorr[reg_i]
        plot_utils.plot_dashlines(0, dir='h', ax=ax)

        for p in process_names:
            autocorrs = stacked_autocorrs[subj_id][p][region]
            plot_utils.plot_shaded_error(autocorr_lags[reg_mapping[region]], np.mean(autocorrs, axis=0),
                                         y_err=utils.stderr(autocorrs, axis=0), ax=ax, label=process_labels[p], alpha=0.6)

        ax.set_title('{} Autocorrelations'.format(region))
        ax.legend(fontsize=10)
        if reg_i == 0:
            ax.set_ylabel('Pearson r')

        # plot average cross-correlations
        for p_i, p in enumerate(process_names):
            ax = axs_xcorr[p_i, reg_i]
            plot_utils.plot_dashlines(0, dir='h', ax=ax)
            plot_utils.plot_dashlines(0, dir='v', ax=ax)

            for region_j in subj_regions:
                if region != region_j:
                    xcorrs = stacked_xcorrs[subj_id][p][region][region_j]
                    plot_utils.plot_shaded_error(xcorr_lags, np.mean(xcorrs, axis=0), y_err=utils.stderr(xcorrs, axis=0),
                                                 ax=ax, label=region_j, alpha=0.6, color=reg_colors[region_j])

            #ax.legend(fontsize=10)
            ax.set_title('{}, Residual Xcorr {}'.format(region, process_labels[p]))
            ax.set_xlabel('Lag (s)')
            ax.grid(True, axis='y')
            if reg_i == 0:
                ax.set_ylabel('Pearson r')

# plot AR parameters
for subj_id in plot_subj:

    if subj_id == 'all':
        subj_regions = all_regions
        palette = region_colors
        region_col = 'general_region'
    else:
        subj_regions = get_subj_regions(subj_id)
        palette = {r: 'C{}'.format(i) for i, r in enumerate(subj_regions)}
        region_col = 'region'

    n_cols = len(process_names)
    fig, axs = plt.subplots(1, n_cols, figsize=(5*n_cols,5), layout='constrained')
    fig.suptitle('Source Deconvolution Parameter Comparison, Subject {}'.format(subj_id))

    p_ax = 0
    for p in process_names:
        if p == 'true':
            continue

        param_df = pd.DataFrame(ar_modes[subj_id][p])

        if p == 'ar2':
            ax = axs[p_ax]
            sb.boxplot(param_df, x=region_col, y='tau', order=subj_regions, hue=region_col, whis=(0,100), ax=ax, log_scale=(False, True), palette=palette)
            ax.set_title('{} Taus'.format(process_labels[p]))
            ax.set_xlabel('Region')
            ax.set_ylabel('Tau (s)')
            ax.legend(fontsize=10)
            p_ax += 1
            ax = axs[p_ax]
            sb.boxplot(param_df, x=region_col, y='freq', order=subj_regions, hue=region_col, whis=(0,100), ax=ax, palette=palette)
            ax.set_title('{} Frequency'.format(process_labels[p]))
            ax.set_xlabel('Region')
            ax.set_ylabel('Frequency (Hz)')
            ax.legend(fontsize=10)
            p_ax += 1
            # sb.scatterplot(param_df, x='freq', y='tau', hue=region_col, hue_order=subj_regions, ax=ax, palette=palette)
            # ax.set_yscale('log')
        else:
            ax = axs[p_ax]
            sb.boxplot(param_df, x=region_col, y='tau', order=subj_regions, hue=region_col, whis=(0,100), ax=ax, log_scale=(False, True), palette=palette)
            ax.set_title('{} Taus'.format(process_labels[p]))
            ax.set_xlabel('Region')
            ax.set_ylabel('Tau (s)')
            ax.legend(fontsize=10)
            p_ax += 1

param_df = pd.DataFrame(ar_modes['all']['ar2to1'])
p = sb.catplot(param_df, x='subj_id', y='tau', hue='task', col='general_region', sharex=False)
p.set(yscale='log')

# %% Compare averaged AR parameter values over sessions to AR parameter values calculated from the average ACF

# parameters are essentially the same, will move forward with simply calculating them from the average autocorrelation

avg_sess_param_dfs = {}
avg_acf_param_dfs = {p: [] for p in process_names if p != 'true'}
avg_acf_g_dfs = {s: {r: {p: [] for p in process_names if p != 'true'} for r in get_subj_regions(s)} for s in subj_ids}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for region in subj_regions:

        acf = np.mean(stacked_autocorrs[subj_id]['true'][region], axis=0)
        acf_lags = autocorr_lags[reg_mapping[region]]
        dt = np.mean(np.diff(acf_lags))

        ar_lags = sum(acf_lags <= region_ar_lags[reg_mapping[region]])

        # calculate parameters
        acf_ar1 = np.concat(([acf[1]], acf))
        g_ar1 = fit_acf_ar_yw(acf_ar1, 1, ar_lags)
        ar1_mode = get_ar_modes(g_ar1, dt)[0]
        ar1_mode['region'] = region
        ar1_mode['general_region'] = reg_mapping[region]
        ar1_mode['subj_id'] = subj_id
        avg_acf_param_dfs['ar1'].append(ar1_mode)
        avg_acf_g_dfs[subj_id][region]['ar1'] = g_ar1

        acf_ar2 = np.concat((np.flip(acf[1:3]), acf))
        g_ar2 = fit_acf_ar_yw(acf_ar2, 2, ar_lags)
        ar2_modes = get_ar_modes(g_ar2, dt)
        ar2_mode = {'tau': 0}
        for ar_mode in ar2_modes:
            if ar_mode['tau'] > ar2_mode['tau']:
                ar2_mode = ar_mode
        ar2_mode['region'] = region
        ar2_mode['general_region'] = reg_mapping[region]
        ar2_mode['subj_id'] = subj_id
        avg_acf_param_dfs['ar2'].append(ar2_mode)
        avg_acf_g_dfs[subj_id][region]['ar2'] = g_ar2

        g_ar2to1 = [max([m['mag'] for m in ar2_modes])]
        ar2to1_mode = get_ar_modes(g_ar2to1, dt)[0]
        ar2to1_mode['region'] = region
        ar2to1_mode['general_region'] = reg_mapping[region]
        ar2to1_mode['subj_id'] = subj_id
        avg_acf_param_dfs['ar2to1'].append(ar2to1_mode)
        avg_acf_g_dfs[subj_id][region]['ar2to1'] = g_ar2to1

for p in process_names:
    if p == 'true':
        continue

    param_df = pd.DataFrame(ar_modes['all'][p])
    param_df['log_tau'] = param_df['tau'].apply(lambda t: np.log(t))

    avg_params = param_df.groupby(['subj_id', 'region'])[['log_tau', 'freq']].agg(['median', 'mean']).reset_index()
    avg_params = avg_params.rename(columns={'log_tau': 'tau'}, level=0)
    avg_params.loc[:,('tau','median')] = avg_params.loc[:,('tau','median')].apply(lambda t: np.exp(t))
    avg_params.loc[:,('tau','mean')] = avg_params.loc[:,('tau','mean')].apply(lambda t: np.exp(t))

    avg_sess_param_dfs[p] = avg_params

    avg_acf_param_dfs[p] = pd.DataFrame(avg_acf_param_dfs[p])

# %% Calculate and plot source cross-correlations using average parameter values

# looks identical to the xcorrs with parameters calculated per session

process_name = 'ar2to1'
stacked_single_xcorrs = {s: {r: {r: [] for r in get_subj_regions(s)} for r in get_subj_regions(s)} for s in subj_ids}
stacked_single_xcorrs['all'] = {r: {r: [] for r in all_regions} for r in all_regions}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    subj_unique_regions = np.unique(list(reg_mapping.values()))

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']

            sess_resids = {}

            for region in subj_regions:
                if not region in fp_data['processed_signals']:
                    continue
                raw_signal = fp_data['processed_signals'][region][signal_type]

                if all(np.isnan(raw_signal)):
                    continue

                if reg_mapping[region] in region_lpf:
                    filt_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
                else:
                    filt_signal = raw_signal

                dec_filt_signal, _ = fp_utils.decimate(filt_signal, t, dec, dt=dt)

                g = avg_acf_g_dfs[subj_id][region][process_name]
                sess_resids[region] = extract_ar_source(dec_filt_signal, g)

            finished_regions = {r: {r: False for r in subj_regions} for r in subj_regions}
            all_region_xcorrs = {r1: {r2: [] for r2 in subj_unique_regions} for r1 in subj_unique_regions}

            for region_i in subj_regions:
                for region_j in subj_regions:

                    if not region_i in sess_resids or not region_j in sess_resids or finished_regions[region_i][region_j]:
                        continue

                    xcorr, xcorr_lags = fp_utils.correlate(sess_resids[region_i], sess_resids[region_j], dt*dec, neg_lag=xcorr_lag, pos_lag=xcorr_lag)
                    flipped_xcorr = np.flip(xcorr)

                    stacked_single_xcorrs[subj_id][region_i][region_j].append(xcorr)
                    stacked_single_xcorrs[subj_id][region_j][region_i].append(flipped_xcorr)

                    # don't map bilateral implants to autocorrelations
                    if reg_mapping[region_i] != reg_mapping[region_j] or region_i == region_j:
                        all_region_xcorrs[reg_mapping[region_i]][reg_mapping[region_j]].append(xcorr)
                        all_region_xcorrs[reg_mapping[region_j]][reg_mapping[region_i]].append(flipped_xcorr)

                    finished_regions[region_i][region_j] = True
                    finished_regions[region_j][region_i] = True

            for region_i in subj_unique_regions:
                for region_j in subj_unique_regions:

                    if len(all_region_xcorrs[region_i][region_j]) == 0:
                        continue

                    # average bilateral regions per session
                    avg_xcorr = np.mean(np.stack(all_region_xcorrs[region_i][region_j], axis=0), axis=0)
                    stacked_single_xcorrs['all'][region_i][region_j].append(avg_xcorr)

    for region_i in subj_regions:
        for region_j in subj_regions:
            stacked_single_xcorrs[subj_id][region_i][region_j] = np.stack(stacked_single_xcorrs[subj_id][region_i][region_j], axis=0)

for region_i in all_regions:
    for region_j in all_regions:
        stacked_single_xcorrs['all'][region_i][region_j] = np.stack(stacked_single_xcorrs['all'][region_i][region_j], axis=0)

plot_ind_subj = True
plot_meta_subj = True

plot_subj = []
if plot_ind_subj:
    plot_subj.extend(subj_ids)
if plot_meta_subj:
    plot_subj.append('all')

# plot correlations
for subj_id in plot_subj:

    if subj_id == 'all':
        subj_regions = all_regions
        reg_colors = region_colors
    else:
        subj_regions = get_subj_regions(subj_id)
        reg_colors = {r: 'C{}'.format(i) for i, r in enumerate(subj_regions)}

    reg_mapping = get_subj_region_mapping(subj_regions)

    fig, axs = plt.subplots(1, len(subj_regions), figsize=(5*len(subj_regions),4), layout='constrained', sharey=True)
    fig.suptitle('Source Deconvolution Crosscorrelation, Avg {} Parameters, Subject {}'.format(process_labels[process_name], subj_id))

    for reg_i, region_i in enumerate(subj_regions):

        ax = axs[reg_i]
        plot_utils.plot_dashlines(0, dir='h', ax=ax)
        plot_utils.plot_dashlines(0, dir='v', ax=ax)

        for region_j in subj_regions:
            if region_i != region_j:
                xcorrs = stacked_single_xcorrs[subj_id][region_i][region_j]
                plot_utils.plot_shaded_error(xcorr_lags, np.mean(xcorrs, axis=0), y_err=utils.stderr(xcorrs, axis=0),
                                             ax=ax, label=region_j, alpha=0.6, color=reg_colors[region_j])

        ax.legend(fontsize=10)
        ax.set_title(region_i)
        ax.set_xlabel('Lag (s)')
        if reg_i == 0:
            ax.set_ylabel('Pearson r')
