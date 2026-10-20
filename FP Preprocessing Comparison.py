# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:53:01 2024

@author: tanne
"""

import init
from hankslab_db import db_access
import hankslab_db.basicRLtasks_db as db
from pyutils import utils
import numpy as np
import fp_analysis_helpers as fpah
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt
import numpy as  np
from scipy.signal import medfilt, butter, sosfiltfilt
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize, OptimizeWarning
from sklearn.linear_model import LinearRegression
import warnings
import time


# %% Declare methods

def get_all_processed_signals(raw_lig, raw_iso, t, smooth_fit, vary_t, lig_lpf=10, iso_lpf=4, smooth_lpf=0.1):
    ''' Gets all possible processed signals and intermediaries for the given raw signals.
        Will check to see if any signals should be excluded. Also will also optionally exclude signals before and after the behavior.'''


    # initialize signal variables
    empty_signal = np.full_like(raw_lig, np.nan)
    baseline_lig = empty_signal.copy()
    baseline_iso = empty_signal.copy()
    baseline_corr_lig = empty_signal.copy()
    baseline_corr_iso = empty_signal.copy()
    fitted_iso = empty_signal.copy()
    fitted_baseline_corr_iso = empty_signal.copy()
    dff_iso = empty_signal.copy()
    dff_iso_baseline = empty_signal.copy()

    # fit iso to ligand
    denoised_lig = fp_utils.filter_signal(raw_lig, lig_lpf, fs)
    denoised_iso = fp_utils.filter_signal(raw_iso, iso_lpf, fs)
    
    baseline_lig = fp_utils.filter_signal(raw_lig, 0.0005, fs, filter_type='lowpass')
    baseline_iso = fp_utils.filter_signal(raw_iso, 0.0005, fs, filter_type='lowpass')
    
    # calculate traditional iso dF/F
    if smooth_fit:
        smooth_lig = fp_utils.filter_signal(raw_lig, smooth_lpf, fs, filter_type='lowpass')
        smooth_iso = fp_utils.filter_signal(raw_iso, smooth_lpf, fs, filter_type='lowpass')
        _, fit_params = fp_utils.fit_signal(smooth_iso, smooth_lig, t, vary_t=vary_t)
        
        if vary_t:
            s_to_fit = np.vstack((denoised_iso[None,:], t[None,:]))
        else:
            s_to_fit = denoised_iso
            
        fitted_iso = fit_params['formula'](s_to_fit, *fit_params['params'])
    else:
        fitted_iso, _ = fp_utils.fit_signal(denoised_iso, denoised_lig, t, vary_t=vary_t)

    dff_iso = ((denoised_lig - fitted_iso)/baseline_lig)*100

    # baseline correction to approximate photobleaching
    try:
        # baseline_lig = fit_baseline(raw_lig)
        # baseline_iso = fit_baseline(raw_iso)
        
        baseline_corr_lig = denoised_lig - baseline_lig
        baseline_corr_iso = denoised_iso - baseline_iso

        # baseline_corr_lig = fp_utils.filter_signal(denoised_lig, 0.001, fs, filter_type='highpass', order=1)
        # baseline_corr_iso = fp_utils.filter_signal(denoised_iso, 0.001, fs, filter_type='highpass', order=1)

        # scale the isosbestic signal to best fit the ligand-dependent signal
        if smooth_fit:
            smooth_lig = fp_utils.filter_signal(baseline_corr_lig, smooth_lpf, fs, filter_type='lowpass')
            smooth_iso = fp_utils.filter_signal(baseline_corr_iso, smooth_lpf, fs, filter_type='lowpass')
            _, fit_params = fp_utils.fit_signal(smooth_iso, smooth_lig, t, vary_t=False)
    
            fitted_baseline_corr_iso = fit_params['formula'](baseline_corr_iso, *fit_params['params'])
        else:
            fitted_baseline_corr_iso, _ = fp_utils.fit_signal(baseline_corr_iso, baseline_corr_lig, t, vary_t=False)

        # shift the fitted baseline corrected iso by the ligand baseline to get the reference
        dff_iso_baseline = ((baseline_corr_lig - fitted_baseline_corr_iso)/baseline_lig)*100

    except RuntimeError as error:
        print(str(error))

    return {'raw_lig': raw_lig,
            'raw_iso': raw_iso,
            'filtered_lig': denoised_lig,
            'baseline_lig': baseline_lig,
            'baseline_iso': baseline_iso,
            'baseline_corr_lig': baseline_corr_lig,
            'baseline_corr_iso': baseline_corr_iso,
            'fitted_iso': fitted_iso,
            'fitted_baseline_corr_iso': fitted_baseline_corr_iso,
            'z_dff_iso': utils.z_score(dff_iso),
            'dff_iso': dff_iso,
            'dff_iso_baseline': dff_iso_baseline}


#%% Get fp data signals

sess_ids = db_access.get_fp_data_sess_ids(subj_ids=[198,199,191,202,180])
# sess_ids = {199: [116668]}
# sess_ids = {202: [101965], 191: [102208, 102406]} #{179: [92692], 180: [102327]} #{202: [101965, 101912], 191: [102208, 100301, 101667]}
# sess_ids = {188: [102246]} #, 100406, 100551, 100673]}
# subj_id = 202 # 191
# sess_id = 101965 # 102208

# sess_ids = {subj_id: [sess_id]}

loc_db = db.LocalDB_BasicRLTasks('')
# get fiber photometry data
fp_data = loc_db.get_sess_fp_data(utils.flatten(sess_ids))
# separate into different dictionaries

fp_data = fp_data['fp_data']


# %% Process signals

isos = np.array(['420','405'])
ligs = np.array(['490','465'])

lig_lpf = 10
iso_lpf = 4
smooth_lpf = 0.1

smooth_fits = [False, True]
vary_ts = [False, True]

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        raw_signals = fp_data[subj_id][sess_id]['raw_signals']
        t = fp_data[subj_id][sess_id]['time']
        fs = 1/(t[1] - t[0])

        fp_data[subj_id][sess_id]['processed_signals'] = {o1: {o2: {} for o2 in [False, True]} for o1 in [False, True]}

        n_regions = len(raw_signals.keys())

        for region in raw_signals.keys():
            
            lig_sel = np.array([k in raw_signals[region].keys() for k in ligs])
            iso_sel = np.array([k in raw_signals[region].keys() for k in isos])

            if sum(lig_sel) >= 1:
                lig = ligs[lig_sel][0]
            else:
                raise Exception('No ligand wavelength found')

            if sum(iso_sel) >= 1:
                iso = isos[iso_sel][0]
            else:
                raise Exception('No isosbestic wavelength found')
            
            raw_lig = raw_signals[region][lig]
            raw_iso = raw_signals[region][iso]
            
                    
            for smooth_fit in smooth_fits:
                for vary_t in vary_ts:
                    fp_data[subj_id][sess_id]['processed_signals'][smooth_fit][vary_t][region] = get_all_processed_signals(raw_lig, raw_iso, t, smooth_fit, vary_t, 
                                                                                                                           lig_lpf=lig_lpf, iso_lpf=iso_lpf, smooth_lpf=smooth_lpf)

# %% Plot processed signals

gen_title = 'Baseline Comparison, Subject {}, Session {}, Smooth Fit: {}, Vary t: {}, Lig LPF: {}, Iso LPF: {}, Smooth LPF: {}'
sub_t = [0,np.inf] # [1100, 1120] #
dec = 10

smooth_fit=False
vary_t=False
plot_baseline_corr=True
filter_outliers=True

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time']

        fig = fpah.view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'][smooth_fit][vary_t], t, t_min=sub_t[0], t_max=sub_t[1], dec=dec,
                                    title=gen_title.format(subj_id, sess_id, smooth_fit, vary_t, lig_lpf, iso_lpf, smooth_lpf), plot_baseline_corr=plot_baseline_corr, filter_outliers=filter_outliers)
        
        plt.show()
        #plt.close(fig)


# %% Compare dF/F for no options vs all options

dec = 10

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time'][::dec]

        regions = list(fp_data[subj_id][sess_id]['raw_signals'].keys())
        processed_signals = fp_data[subj_id][sess_id]['processed_signals']
        
        fig, axs = plt.subplots(len(regions), 1, figsize=(12, 3*len(regions)), layout='constrained')
        
        if len(regions) == 1:
            axs = [axs]
        
        for i, region in enumerate(regions):

            ax = axs[i]
            fig.suptitle('Subj {}, Sess {}'.format(subj_id, sess_id))
            ax.plot(t, processed_signals[False][False][region]['dff_iso'][::dec], label='Traditional', alpha=0.5)
            ax.plot(t, processed_signals[False][True][region]['dff_iso'][::dec], label='Vary t', alpha=0.5)
            ax.plot(t, processed_signals[False][False][region]['dff_iso_baseline'][::dec], label='Baseline Corr Trad', alpha=0.5)
            ax.plot(t, processed_signals[True][False][region]['dff_iso_baseline'][::dec], label='Baseline Corr Smooth Fit', alpha=0.5)

            # ax.plot(t, processed_signals[True][False][region]['dff_iso'][::dec], label='Smooth Fit', alpha=0.5)
            # ax.plot(t, processed_signals[True][True][region]['dff_iso'][::dec], label='Vary t & Smooth Fit', alpha=0.5)
            ax.set_title(region)
            ax.set_ylabel('dF/F')
            ax.legend(loc='upper right')

# %% Process signals

from sklearn.decomposition import FastICA
from scipy.optimize import curve_fit

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        raw_signals = fp_data[subj_id][sess_id]['raw_signals']
        t = fp_data[subj_id][sess_id]['time']
        fs = 1/(t[1] - t[0])

        #fp_data[subj_id][sess_id]['processed_signals'] = {}
        #fp_data[subj_id][sess_id]['processed_signals_denoised'] = {}

        n_regions = len(raw_signals.keys())

        for region in raw_signals.keys():

            raw_lig = raw_signals[region][lig]
            raw_iso = raw_signals[region][iso]

            denoised_lig = fp_utils.filter_signal(raw_lig, 10, fs)
            denoised_iso = fp_utils.filter_signal(raw_iso, 10, fs)
            
            fitted_iso = fp_utils.fit_signal(denoised_iso, denoised_lig, t)
            
            classic_dff = (denoised_lig - fitted_iso)/fitted_iso * 100
            
            ica = FastICA()
            signal_ica = ica.fit_transform(np.vstack([denoised_lig, fitted_iso]).T)
            mix_mat = ica.mixing_
            mix_loading = np.abs(np.sum(mix_mat, axis=0))
            
            if mix_loading[0] > mix_loading[1]:
                noise_idx = 0
                sig_idx = 1
            else:
                noise_idx = 1
                sig_idx = 0
            
            form = lambda x, a, b: a*x + b
            params = curve_fit(form, signal_ica[:,noise_idx], fitted_iso)[0]
            fitted_shared_signal = form(signal_ica[:,noise_idx], *params)

            params = curve_fit(form, signal_ica[:,sig_idx], denoised_lig-fitted_iso)[0]
            fitted_unshared_signal = form(signal_ica[:,sig_idx], *params)
            
            ica1_dff = (denoised_lig - fitted_shared_signal)/fitted_iso * 100
            ica2_dff = (fitted_unshared_signal-np.mean(fitted_unshared_signal))/fitted_iso * 100
            
            t_min = 1200 # -np.inf # 
            t_max = 1220 # np.inf # 
            plot_t = t.copy()
            plot_t[(t < t_min) | (t > t_max)] = np.nan
            
            fig, axs = plt.subplots(2,2, figsize=(10,8), layout='constrained')
            
            ax = axs[0,0]
            ax.plot(plot_t, denoised_lig, label='Lig', alpha=0.6)
            ax.plot(plot_t, fitted_iso, label='Fitted Iso', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[0,1]
            ax.plot(plot_t, denoised_lig, label='Lig', alpha=0.6)
            ax.plot(plot_t, fitted_shared_signal, label='IC1', alpha=0.6)
            #ax.plot(plot_t, fitted_unshared_signal, label='IC2', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[1,0]
            ax.plot(plot_t, denoised_lig - fitted_iso, label='Iso Subtracted', alpha=0.6)
            ax.plot(plot_t, denoised_lig - fitted_shared_signal, label='IC1 Subtracted', alpha=0.6)
            ax.plot(plot_t, fitted_unshared_signal-np.mean(fitted_unshared_signal), label='IC2 Subtracted', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[1,1]
            ax.plot(plot_t, classic_dff, label='Classic dF/F', alpha=0.6)
            ax.plot(plot_t, ica1_dff, label='ICA1 dF/F', alpha=0.6)
            ax.plot(plot_t, ica2_dff, label='ICA2 dF/F', alpha=0.6)
            ax.legend(loc='upper right')
            
            #fp_data[subj_id][sess_id]['processed_signals'][region] = get_all_processed_signals(raw_lig, raw_iso, t)
            #fp_data[subj_id][sess_id]['processed_signals_denoised'][region] = get_all_processed_signals(denoised_lig, denoised_iso, t)

# %% Compare dF/F by choice of baseline LPF

# 0.0005 seems to better fit the overall baseline than 0.0002, but they produce very similar results

loc_db = db.LocalDB_BasicRLTasks('')

sess_ids = {400: [119713, 119974], 402: [119793, 119983], 274: [119776, 119966]}
baseline_lpf1 = 0.0005
baseline_lpf2 = 0.0002
iso_bands = [[0,0.1], [0.1,1], [1,10]]
dec = 2

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        fp_data1, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, baseline_lpf=baseline_lpf1, tilt_t=False, 
                                        filter_dropout_outliers=False, band_iso_fit=False, baseline_band_iso_fit=True, iso_bands=iso_bands)
        fp_data2, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, baseline_lpf=baseline_lpf2, tilt_t=False, 
                                        filter_dropout_outliers=False, band_iso_fit=False, baseline_band_iso_fit=True, iso_bands=iso_bands)
        fp_data1 = fp_data1[subj_id][sess_id]
        fp_data2 = fp_data2[subj_id][sess_id]
        processed_sigs1 = fp_data1['processed_signals']
        processed_sigs2 = fp_data2['processed_signals']
        
        t = fp_data1['time']

        regions = list(fp_data1['processed_signals'].keys())

        fig, axs = plt.subplots(len(regions), 4, figsize=(10,8), layout='constrained', sharex=True)

        for i, region in enumerate(regions):
            
            if subj_id in fpah.__region_ignore and region in fpah.__region_ignore[subj_id]:
                continue
            
            reg_sigs1 = processed_sigs1[region]
            reg_sigs2 = processed_sigs2[region]

            ax = axs[i,0]
            ax.plot(t[::dec], reg_sigs1['filtered_lig'][::dec], label='Lig', alpha=0.6, color='C0')
            ax.plot(t[::dec], reg_sigs1['baseline_lig'][::dec], label='Baseline {}'.format(baseline_lpf1), alpha=0.6, color='C0', linestyle='--')
            ax.plot(t[::dec], reg_sigs2['baseline_lig'][::dec], label='Baseline {}'.format(baseline_lpf2), alpha=0.6, color='C0', linestyle=':')
            ax.plot(t[::dec], reg_sigs1['filtered_iso'][::dec], label='Iso', alpha=0.6, color='C1')
            ax.plot(t[::dec], reg_sigs1['baseline_iso'][::dec], label='Baseline {}'.format(baseline_lpf1), alpha=0.6, color='C1', linestyle='--')
            ax.plot(t[::dec], reg_sigs2['baseline_iso'][::dec], label='Baseline {}'.format(baseline_lpf2), alpha=0.6, color='C1', linestyle=':')
            ax.legend(loc='upper right')
            
            ax = axs[i,1]
            ax.plot(t[::dec], reg_sigs1['baseline_corr_lig'][::dec], label='Lig {}'.format(baseline_lpf1), alpha=0.6)
            ax.plot(t[::dec], reg_sigs1['baseline_corr_iso'][::dec], label='Iso {}'.format(baseline_lpf1), alpha=0.6)
            ax.plot(t[::dec], reg_sigs1['fitted_baseline_fband_iso'][::dec], label='Fitted Iso {}'.format(baseline_lpf1), alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[i,2]
            ax.plot(t[::dec], reg_sigs2['baseline_corr_lig'][::dec], label='Lig {}'.format(baseline_lpf2), alpha=0.6)
            ax.plot(t[::dec], reg_sigs2['baseline_corr_iso'][::dec], label='Iso {}'.format(baseline_lpf2), alpha=0.6)
            ax.plot(t[::dec], reg_sigs2['fitted_baseline_fband_iso'][::dec], label='Fitted Iso {}'.format(baseline_lpf2), alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[i,3]
            ax.plot(t[::dec], reg_sigs1['dff_iso_baseline_fband'][::dec], label='dF/F {}'.format(baseline_lpf1), alpha=0.6)
            ax.plot(t[::dec], reg_sigs2['dff_iso_baseline_fband'][::dec], label='dF/F {}'.format(baseline_lpf2), alpha=0.6)
            ax.legend(loc='upper right')
        

# %% compare frequency band inclusion on recovered signal
loc_db = db.LocalDB_BasicRLTasks('')

# with baseline correction, the bands [[0,0.1], [0.1,1], [1]] seem to be the minimal set of the most general purpose frequencies
# Splitting [0,0.1] into [0,0.01], [0.01,0.1] seems to have the largest effect, but that seems to lead to increased filtering artifacts
# No effects of splitting [1] into [1,5], [5]. Marginal effects of splitting [0.1,1] into [0.1,0.5], [0.5,1]
# No difference between [1] and [1,10]

# Best outcome is [[0,0.5], [0.5,5], [5]]. Expanding the lowest frequency band to 0.5 improves edge effects and reduces the artifacts of sharp changes

#subj_ids = [198,199,237,238,274,400,402,424,483] # 198, 199, 274, 400, 402 #179, 180, 182, 188, 191, 202, 207
#sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)

sess_ids = {400: [119247]}
# sess_ids = {198: [116588], 237: [125128], 238: [124965,125171], 274: [119464], 400: [119233,119247,119830,120001,120103], 
#             402: [119793,120009], 424: [124738,125129], 483: [124582,124622,124769,124948,125000]} #{402: [119234], 400: [119194,119713,119247], 274: [119238]}

# bands1 = [[0,0.1], [0.1,0.5], [0.5,1], [1,5], [5,8], [8]] #[[0,0.05], [0.05,0.5], [0.5,1], [1,5], [5,10]]
# freq_vals = [0] + list(np.logspace(np.log10(0.01), np.log10(10), 10))
# freq_vals = np.round(freq_vals, 3)
# bands1 = list(zip(freq_vals[:-1], freq_vals[1:]))
#bands1 = [[0,0.01], [0.01,0.1], [0.1,0.5], [0.5,1], [1,5], [5,8], [8]]
#bands1 = [[0,0.01], [0.01,0.1], [0.1,0.5], [0.5,1], [1,10]]
bands1 = [[0,0.5], [0.5,5], [5]]
bands2 = [[0,0.1], [0.1,1], [1]]
#bands2 = [[0,0.01], [0.01,0.1], [0.1,1], [1,10]]

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id][::2]:
        
        fp_data1, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, band_iso_fit=False, baseline_band_iso_fit=True, 
                               tilt_t=False, filter_dropout_outliers=False, iso_bands=bands1, 
                               allow_neg_coeffs=False, ignore_sess_start=True)
        
        fp_data2, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, band_iso_fit=False, baseline_band_iso_fit=True, 
                               tilt_t=False, filter_dropout_outliers=False, iso_bands=bands2, 
                               allow_neg_coeffs=False, ignore_sess_start=True)
        
        fp_data1 = fp_data1[subj_id][sess_id]
        fp_data2 = fp_data2[subj_id][sess_id]
        
        t = fp_data1['time']
        sr = 1/np.mean(np.diff(t))

        regions = list(fp_data1['processed_signals'].keys())
        
        alpha = 0.5
        fig, axs = plt.subplots(len(regions), 2, sharex=True, layout='constrained')
        fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(regions):
            processed_signals1 = fp_data1['processed_signals'][region]
            processed_signals2 = fp_data2['processed_signals'][region]
            
            ax = axs[i,0]
            ax.plot(t[::dec], processed_signals1['baseline_corr_lig'][::dec], label='baseline lig', alpha=alpha, color='C3')
            ax.plot(t[::dec], processed_signals1['fitted_baseline_corr_iso'][::dec], label='fitted baseline iso', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals1['fitted_baseline_fband_iso'][::dec], label='fitted iso 1', alpha=alpha, color='C1')
            ax.plot(t[::dec], processed_signals2['fitted_baseline_fband_iso'][::dec], label='fitted iso 2', alpha=alpha, color='C2')
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,1]
            ax.plot(t[::dec], processed_signals1['dff_iso_baseline'][::dec], label='baseline dF/F', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals1['dff_iso_baseline_fband'][::dec], label='baseline fband dF/F 1', alpha=alpha, color='C1')
            ax.plot(t[::dec], processed_signals2['dff_iso_baseline_fband'][::dec], label='baseline fband dF/F 2', alpha=alpha, color='C2')
            ax.legend()
            ax.set_title(region)
            
            # ax = axs[i,2]
            # ax.plot(t[::dec], processed_signals1['filtered_lig'][::dec], label='lig', alpha=alpha)
            # ax.plot(t[::dec], processed_signals1['fitted_fband_iso'][::dec], label='fitted iso 1', alpha=alpha)
            # ax.plot(t[::dec], processed_signals2['fitted_fband_iso'][::dec], label='fitted iso 2', alpha=alpha)
            # ax.set_title(region)
            # ax.legend()
            
            # ax = axs[i,3]
            # ax.plot(t[::dec], processed_signals1['dff_iso_fband'][::dec], label='fband dF/F 1', alpha=alpha)
            # ax.plot(t[::dec], processed_signals2['dff_iso_fband'][::dec], label='fband dF/F 2', alpha=alpha)
            # ax.legend()
            # ax.set_title(region)
            
        plt.show()


# %% compare low pass filter value on recovered signal
loc_db = db.LocalDB_BasicRLTasks('')

#subj_ids = [198,199,237,238,274,400,402,424,483] # 198, 199, 274, 400, 402 #179, 180, 182, 188, 191, 202, 207
#sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)

#sess_ids = {400: [119247]}
sess_ids = {198: [116588], 237: [125128], 238: [124965,125171], 274: [119464], 400: [119233,119247,119830,120001,120103], 
            402: [119793,120009], 424: [124738,125129], 483: [124582,124622,124769,124948,125000]} #{402: [119234], 400: [119194,119713,119247], 274: [119238]}

bands = [[0,0.5], [0.5,5], [5]]
lpf1 = 20
lpf2 = 10

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id][::2]:
        
        fp_data1, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, band_iso_fit=False, baseline_band_iso_fit=True, 
                               tilt_t=False, iso_bands=bands, lig_lpf=lpf1, iso_lpf=lpf1,
                               allow_neg_coeffs=False, ignore_sess_start=True)
        
        fp_data2, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, band_iso_fit=False, baseline_band_iso_fit=True, 
                               tilt_t=False, iso_bands=bands, lig_lpf=lpf2, iso_lpf=lpf2,
                               allow_neg_coeffs=False, ignore_sess_start=True)
        
        fp_data1 = fp_data1[subj_id][sess_id]
        fp_data2 = fp_data2[subj_id][sess_id]
        
        t = fp_data1['time']
        sr = 1/np.mean(np.diff(t))

        regions = list(fp_data1['processed_signals'].keys())
        
        alpha = 0.5
        fig, axs = plt.subplots(len(regions), 3, sharex=True, layout='constrained', figsize=(18,5*len(regions)))
        fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(regions):
            processed_signals1 = fp_data1['processed_signals'][region]
            processed_signals2 = fp_data2['processed_signals'][region]
            
            ax = axs[i,0]
            ax.plot(t[::dec], processed_signals1['baseline_corr_lig'][::dec], label='baseline lig 1', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals2['baseline_corr_lig'][::dec], label='baseline lig 2', alpha=alpha, color='C1')
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,1]
            ax.plot(t[::dec], processed_signals1['fitted_baseline_fband_iso'][::dec], label='fitted iso 1', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals2['fitted_baseline_fband_iso'][::dec], label='fitted iso 2', alpha=alpha, color='C1')
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,2]
            ax.plot(t[::dec], processed_signals1['dff_iso_baseline_fband'][::dec], label='baseline fband dF/F 1', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals2['dff_iso_baseline_fband'][::dec], label='baseline fband dF/F 2', alpha=alpha, color='C1')
            ax.legend()
            ax.set_title(region)
            
        plt.show()

# %% compare fband with tilt to baseline subtracted fband

# baseline subtracted fband still outperforms the fband with tilt in overall smoothing of the baseline

loc_db = db.LocalDB_BasicRLTasks('')

#subj_ids = [198,199,237,238,274,400,402,424,483] # 198, 199, 274, 400, 402 #179, 180, 182, 188, 191, 202, 207
#sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)

#sess_ids = {400: [119247]}
sess_ids = {198: [116588], 237: [125128], 238: [124965,125171], 274: [119464], 400: [119233,119247,119830,120001,120103], 
            402: [119793,120009], 424: [124738,125129], 483: [124582,124622,124769,124948,125000]} #{402: [119234], 400: [119194,119713,119247], 274: [119238]}

bands = [[0,0.5], [0.5,5], [5]]

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id][::2]:
        
        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, band_iso_fit=True, baseline_band_iso_fit=True, 
                               tilt_t=True, iso_bands=bands, allow_neg_coeffs=False, ignore_sess_start=True)
        
        fp_data = fp_data[subj_id][sess_id]
        
        t = fp_data['time']
        sr = 1/np.mean(np.diff(t))

        regions = list(fp_data['processed_signals'].keys())
        
        alpha = 0.5
        fig, axs = plt.subplots(len(regions), 3, sharex=True, layout='constrained', figsize=(20,4*len(regions)))
        fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(regions):
            processed_signals = fp_data['processed_signals'][region]
            
            ax = axs[i,0]
            ax.plot(t[::dec], processed_signals['filtered_lig'][::dec], label='Lig', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals['fitted_fband_iso'][::dec], label='Fband Iso', alpha=alpha, color='C1')
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,1]
            ax.plot(t[::dec], processed_signals['baseline_corr_lig'][::dec], label='Baseline Lig', alpha=alpha, color='C0')
            ax.plot(t[::dec], processed_signals['fitted_baseline_fband_iso'][::dec], label='Baseline Fband Iso', alpha=alpha, color='C2')
            ax.set_title(region)
            ax.legend()
            
            ax = axs[i,2]
            ax.plot(t[::dec], processed_signals['dff_iso_fband'][::dec], label='Fband dF/F', alpha=alpha, color='C1')
            ax.plot(t[::dec], processed_signals['dff_iso_baseline_fband'][::dec], label='Baseline Fband dF/F', alpha=alpha, color='C2')
            ax.legend()
            ax.set_title(region)
            
        plt.show()

# %% Compare subspace-based regression to traditional regression

from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

loc_db = db.LocalDB_BasicRLTasks('')

sess_ids = {400: [119713]}
baseline_lpf = 0.0002
iso_bands = [[0,0.1], [0.1,1], [1,10]]
dec = 2

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, baseline_lpf=baseline_lpf, tilt_t=False, 
                                    filter_dropout_outliers=False, band_iso_fit=False, baseline_band_iso_fit=True, iso_bands=iso_bands)
        fp_data = fp_data[subj_id][sess_id]
        processed_sigs = fp_data['processed_signals']
        
        t = fp_data['time']
        dt = fp_data['dec_info']['decimated_dt']

        regions = list(fp_data['processed_signals'].keys())

        fig, axs = plt.subplots(len(regions), 3, figsize=(10,8), layout='constrained', sharex=True)

        for i, region in enumerate(regions):
            
            if subj_id in fpah.__region_ignore and region in fpah.__region_ignore[subj_id]:
                continue
            
            reg_sigs = processed_sigs[region]

            # Perform single latent projection
            # Stack and z-score
            lig = utils.z_score(reg_sigs['baseline_corr_lig'])
            iso = utils.z_score(reg_sigs['baseline_corr_iso'])
            lig_fbands = utils.z_score(fp_utils.decompose_signal_fbands(lig, iso_bands, 1/dt), axis=1)
            iso_fbands = utils.z_score(fp_utils.decompose_signal_fbands(reg_sigs['baseline_corr_iso'], iso_bands, 1/dt), axis=1)

            nans = np.isnan(lig) | np.isnan(iso)
            
            shared_latent1 = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            shared_latent2 = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            
            # PCA subspace
            # Stack channels
            X = np.vstack([lig[~nans], iso[~nans]])
        
            # PCA finds shared subspace when only 2 channels
            pca_latents = PCA(n_components=2).fit_transform(X.T)
            shared_latent1[~nans] = pca_latents[:,0]
            shared_latent2[~nans] = pca_latents[:,1]
        
            # Project shared latent back onto ligand channel
            alpha1 = np.dot(lig[~nans], shared_latent1[~nans]) / np.dot(lig[~nans], lig[~nans])
            alpha2 = np.dot(lig[~nans], shared_latent2[~nans]) / np.dot(lig[~nans], lig[~nans])
            shared_est1 = alpha1 * shared_latent1
            shared_est2 = alpha2 * shared_latent2
            
            lig_clean1 = lig - shared_est1
            lig_clean2 = lig - shared_est2
            
            dff_latent1 = (lig_clean1/(reg_sigs['baseline_lig'] + shared_est1)) * 100
            dff_latent2 = (lig_clean2/(reg_sigs['baseline_lig'] + shared_est2)) * 100
            
            # plot results
            ax = axs[i,0]
            ax.plot(t[::dec], lig[::dec], label='Z-scored Lig', alpha=0.6)
            ax.plot(t[::dec], iso[::dec], label='Z-scored Iso', alpha=0.6)
            ax.plot(t[::dec], shared_est1[::dec], label='Fit Shared Latent1', alpha=0.6)
            ax.plot(t[::dec], shared_est2[::dec], label='Fit Shared Latent2', alpha=0.6)
            ax.legend(loc='upper right')
            
            ax = axs[i,1]
            ax.plot(t[::dec], reg_sigs['dff_iso_baseline_fband'][::dec], label='Regression dF/F', alpha=0.6)
            ax.plot(t[::dec], dff_latent1[::dec], label='Latent dF/F 1', alpha=0.6)
            ax.plot(t[::dec], dff_latent2[::dec], label='Latent dF/F 2', alpha=0.6)
            ax.legend(loc='upper right')

            # Shared latent
            # latent_u_partial = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            # latent_v_partial = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            
            # latent_u_full = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            # latent_v_full = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            
            # cca = CCA(n_components=1)
            # U_partial, V_partial = cca.fit_transform(iso_fbands[:,~nans].T, lig[~nans][:,None])
            # latent_u_partial[~nans] = U_partial.ravel()
            # latent_v_partial[~nans] = V_partial.ravel()
            
            # U_full, V_full = cca.fit_transform(iso_fbands[:,~nans].T, lig_fbands[:,~nans].T)
            # latent_u_full[~nans] = U_full.ravel()
            # latent_v_full[~nans] = V_full.ravel()

            # fit_latent_u_partial, _ = fp_utils.fit_signal(latent_u_partial, reg_sigs['baseline_corr_lig'], t)
            # fit_latent_v_partial, _ = fp_utils.fit_signal(latent_v_partial, reg_sigs['baseline_corr_lig'], t)
            
            # fit_latent_u_full, _ = fp_utils.fit_signal(latent_u_full, reg_sigs['baseline_corr_lig'], t)
            # fit_latent_v_full, _ = fp_utils.fit_signal(latent_v_full, reg_sigs['baseline_corr_lig'], t)
            
            # dff_latent_u_partial = ((reg_sigs['baseline_corr_lig'] - fit_latent_u_partial)/(reg_sigs['baseline_lig'] + fit_latent_u_partial)) * 100
            # dff_latent_v_partial = ((reg_sigs['baseline_corr_lig'] - fit_latent_v_partial)/(reg_sigs['baseline_lig'] + fit_latent_v_partial)) * 100
            
            # dff_latent_u_full = ((reg_sigs['baseline_corr_lig'] - fit_latent_u_full)/(reg_sigs['baseline_lig'] + fit_latent_u_full)) * 100
            # dff_latent_v_full = ((reg_sigs['baseline_corr_lig'] - fit_latent_v_full)/(reg_sigs['baseline_lig'] + fit_latent_v_full)) * 100

            # # plot results
            # ax = axs[i,0]
            # ax.plot(t[::dec], lig[::dec], label='Z-scored Lig', alpha=0.6)
            # ax.plot(t[::dec], iso[::dec], label='Z-scored Iso', alpha=0.6)
            # ax.plot(t[::dec], latent_u_partial[::dec], label='Latent Lig Partial', alpha=0.6)
            # #ax.plot(t[::dec], latent_v_partial[::dec]+0.2, label='Latent Iso Partial', alpha=0.6)
            # ax.plot(t[::dec], latent_u_full[::dec], label='Latent Lig Full', alpha=0.6)
            # #ax.plot(t[::dec], latent_v_full[::dec]+0.4, label='Latent Iso Full', alpha=0.6)
            # ax.legend(loc='upper right')
            
            # ax = axs[i,1]
            # ax.plot(t[::dec], reg_sigs['baseline_corr_lig'][::dec], label='Lig', alpha=0.6)
            # ax.plot(t[::dec], reg_sigs['baseline_corr_iso'][::dec], label='Iso', alpha=0.6)
            # ax.plot(t[::dec], reg_sigs['fitted_baseline_fband_iso'][::dec], label='Fitted Iso', alpha=0.6)
            # ax.plot(t[::dec], fit_latent_u_partial[::dec], label='Fittent Latent Lig Partial', alpha=0.6)
            # #ax.plot(t[::dec], fit_latent_v_partial[::dec]+0.2, label='Fittent Latent Iso Partial', alpha=0.6)
            # ax.plot(t[::dec], fit_latent_u_full[::dec], label='Fittent Latent Lig Full', alpha=0.6)
            # #ax.plot(t[::dec], fit_latent_v_full[::dec]+0.4, label='Fittent Latent Iso Full', alpha=0.6)
            # ax.legend(loc='upper right')
            
            # ax = axs[i,2]
            # ax.plot(t[::dec], reg_sigs['dff_iso_baseline_fband'][::dec], label='Regression dF/F', alpha=0.6)
            # ax.plot(t[::dec], dff_latent_u_partial[::dec], label='Latent Lig dF/F Partial', alpha=0.6)
            # #ax.plot(t[::dec], dff_latent_v_partial[::dec]+0.2, label='Latent Iso dF/F Partial', alpha=0.6)
            # ax.plot(t[::dec], dff_latent_u_full[::dec], label='Latent Lig dF/F Full', alpha=0.6)
            # #ax.plot(t[::dec], dff_latent_v_full[::dec]+0.4, label='Latent Iso dF/F Full', alpha=0.6)
            # ax.legend(loc='upper right')
        
        
# %% Compare wavelet-decomposition regression to frequency band decomposition regression

import pywt

loc_db = db.LocalDB_BasicRLTasks('')

sess_ids = {400: [119713,119247]}
baseline_lpf = 0.0005
level = 6
f_max = 10
freq_vals = [0] + [f_max / (2 ** l) for l in range(level, -1, -1)]
freq_vals = np.round(freq_vals, 3)

freq_vals2 = [0] + list(np.logspace(np.log10(0.05), np.log10(f_max), level+1))
freq_vals2 = np.round(freq_vals2, 3)

iso_bands = list(zip(freq_vals[:-1], freq_vals[1:]))
#iso_bands = [[0,0.1], [0.1,1], [1,10]]
dec = 2

wavelet1 = 'db4'
#wavelet2 = 'sym5'
level = 10

pywt.central_frequency(wavelet1)

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=True, baseline_lpf=baseline_lpf, tilt_t=False, 
                                    filter_dropout_outliers=False, band_iso_fit=False, baseline_band_iso_fit=True, iso_bands=iso_bands)
        fp_data = fp_data[subj_id][sess_id]
        processed_sigs = fp_data['processed_signals']
        
        fpah.view_processed_signals(processed_sigs, fp_data['time'], 
                                    plot_baseline_corr=True, plot_fband=False, plot_baseline_fband=True,
                                    title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id))
        
        t = fp_data['time']
        dt = fp_data['dec_info']['decimated_dt']

        regions = list(fp_data['processed_signals'].keys())

        fig_sig, axs_sig = plt.subplots(len(regions), 2, figsize=(10,3*len(regions)), layout='constrained', sharex=True)
        fig_sig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))
        
        fig_mra, axs_mra = plt.subplots(len(regions), 2, figsize=(10,3*len(regions)), layout='constrained', sharex=True)
        fig_mra.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(regions):
            
            if subj_id in fpah.__region_ignore and region in fpah.__region_ignore[subj_id]:
                continue
            
            reg_sigs = processed_sigs[region]

            filled_sig, nans = fp_utils.fill_signal_nans(reg_sigs['baseline_corr_iso'])

            # pad the signal on each end with the extrapolated linear trend over that pad length before or after the edge
            tot_pad = -len(filled_sig) % 2**level
            left_pad_len = tot_pad // 2
            right_pad_len = tot_pad - left_pad_len
            filled_sig = fp_utils.multiscale_pad(filled_sig, [left_pad_len, right_pad_len])

            # Perform SWT on baseline subtracted iso
            start = time.perf_counter()
            iso_swt1 = pywt.mra(filled_sig, wavelet1, level=level)
            print('Performed {} MRA in {:.1f} s'.format(wavelet1, time.perf_counter()-start))
            # start = time.perf_counter()
            # iso_swt2 = pywt.mra(filled_sig, wavelet2, level=level)
            # print('Performed {} MRA in {:.1f} s'.format(wavelet2, time.perf_counter()-start))
            
            # plot wavelet transformation results
            ax = axs_mra[i,0]
            ax.set_title('{} Interp Iso'.format(region))
            ax.plot(filled_sig[::dec])
            
            ax = axs_mra[i,1]
            ax.set_title('{}, {}'.format(region, wavelet1))
            for j in range(level+1):
                ax.plot(utils.z_score(iso_swt1[j])[::dec], label='Level {}'.format(level-j+1), alpha=0.4)
            ax.legend(loc='upper right')
            
            # ax = axs_mra[i,1]
            # ax.set_title('{}, {}'.format(region, wavelet2))
            # ax.plot(t[::dec], reg_sigs['baseline_corr_lig'][::dec], label='Lig', alpha=0.4)
            # ax.plot(t[::dec], reg_sigs['baseline_corr_iso'][::dec], label='Iso', alpha=0.4)
            # for j in range(level+1):
            #     ax.plot(t[~nans][::dec], utils.z_score(iso_swt2[j,:])[::dec], label='Level {} ({:.1f})'.format(level-j+1, params2[j]), alpha=0.4)
            # ax.legend(loc='upper right')
            
            iso_swt1 = np.vstack(iso_swt1)
            iso_swt1 = iso_swt1[:,left_pad_len:-right_pad_len]
            iso_swt1 = iso_swt1[:,~nans]
            
            # iso_swt2 = np.vstack(iso_swt2)
            # iso_swt2 = iso_swt2[:,left_pad_len:-right_pad_len]
            # iso_swt2 = iso_swt2[:,~nans]
            
            # dynamically create the formula to optimize
            param_names = ['b{}'.format(i) for i in range(level+1)]

            expr = ' + '.join(['{}*x[{},:]'.format(p,i) for i,p in enumerate(param_names)]) + ' + c'

            param_names = param_names + ['c']

            # Build the lambda string and evaluate it 
            form = eval('lambda x, {}: {}'.format(', '.join(param_names), expr))

            bounds = (np.zeros(level+1).tolist(), np.inf)
            bounds[0].append(-np.inf)

            start = time.perf_counter()
            params1 = curve_fit(form, iso_swt1, reg_sigs['baseline_corr_lig'][~nans], bounds=bounds)[0]
            fitted_iso_swt1 = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            fitted_iso_swt1[~nans] = form(iso_swt1, *params1)
            print('Fit {} MRA iso in {:.1f} s'.format(wavelet1, time.perf_counter()-start))
            
            # start = time.perf_counter()
            # params2 = curve_fit(form, iso_swt2, reg_sigs['baseline_corr_lig'][~nans], bounds=bounds)[0]
            # fitted_iso_swt2 = np.full_like(reg_sigs['baseline_corr_lig'], np.nan)
            # fitted_iso_swt2[~nans] = form(iso_swt2, *params2)
            # print('Fit {} MRA iso in {:.1f} s'.format(wavelet2, time.perf_counter()-start))
            
            dff_swt1 = ((reg_sigs['baseline_corr_lig'] - fitted_iso_swt1)/(reg_sigs['baseline_lig'] + fitted_iso_swt1)) * 100
            # dff_swt2 = ((reg_sigs['baseline_corr_lig'] - fitted_iso_swt2)/(reg_sigs['baseline_lig'] + fitted_iso_swt2)) * 100
            
            # plot results
            ax = axs_sig[i,0]
            ax.set_title(region)
            ax.plot(t[::dec], reg_sigs['baseline_corr_lig'][::dec], label='Lig', alpha=0.4)
            ax.plot(t[::dec], reg_sigs['fitted_baseline_corr_iso'][::dec], label='Baseline Iso', alpha=0.4)
            ax.plot(t[::dec], reg_sigs['fitted_baseline_fband_iso'][::dec], label='Fband Iso', alpha=0.4)
            ax.plot(t[::dec], fitted_iso_swt1[::dec], label='Fitted SWT Iso - {}'.format(wavelet1), alpha=0.4)
            # ax.plot(t[::dec], fitted_iso_swt2[::dec], label='Fitted SWT Iso - {}'.format(wavelet2), alpha=0.4)
            ax.legend(loc='upper right')
            
            ax = axs_sig[i,1]
            ax.set_title(region)
            ax.plot(t[::dec], reg_sigs['dff_iso_baseline_fband'][::dec], label='Fband dF/F', alpha=0.4)
            ax.plot(t[::dec], dff_swt1[::dec], label='SWT df/F - {}'.format(wavelet1), alpha=0.4)
            # ax.plot(t[::dec], dff_swt2[::dec], label='SWT df/F - {}'.format(wavelet2), alpha=0.4)
            ax.legend(loc='upper right')

            

# %%

# import numpy as np

# # LMS Adaptive Filter Function
# def lms_filter(desired_signal, noise_reference, mu=0.01, filter_order=5):
#     """
#     Perform adaptive noise cancellation using LMS filter.

#     Parameters:
#     - desired_signal: Signal containing both the desired signal and noise (d[n])
#     - noise_reference: Reference noise signal (x[n])
#     - mu: Learning rate (step size) for LMS algorithm
#     - filter_order: Number of filter taps (higher values allow more complex noise cancellation)

#     Returns:
#     - cleaned_signal: Filter output with noise removed
#     """
#     N = len(desired_signal)
#     w = np.zeros(filter_order)  # Initialize filter weights
#     x_buf = np.zeros(filter_order)  # Buffer to hold past values of x(n)
#     cleaned_signal = np.zeros(N)  # Output signal

#     for n in range(N):
#         # Update input buffer (shift values, insert new one)
#         x_buf[1:] = x_buf[:-1]
#         x_buf[0] = noise_reference[n] if n < len(noise_reference) else 0

#         # Compute filter output (estimated noise)
#         noise_estimate = np.dot(w, x_buf)

#         # Compute error signal (cleaned signal)
#         e = desired_signal[n] - noise_estimate
#         cleaned_signal[n] = e

#         # Update filter weights using LMS rule
#         w += 2 * mu * e * x_buf  # Gradient descent step

#     return cleaned_signal

# # Simulated Example
# n = 1000
# np.random.seed(42)
# t = np.linspace(0, 1, n)  # Time vector
# true_signal = 3*np.sin(2 * np.pi * 5 * t)  # Desired signal (10 Hz sine wave)
# noise = np.random.randn(n)  # Random noise
# mixed_signal = true_signal + noise  # Signal + noise
# reference_noise = noise + np.random.randn(n) * 0.1  # Reference noise with some variation

# # Apply LMS filter
# # filtered_signal = lms_filter(mixed_signal, reference_noise, mu=0.01, filter_order=5)

# # # Plot results
# # plt.figure(figsize=(10, 5))
# # plt.plot(t, mixed_signal, label="Noisy Signal", alpha=0.6)
# # plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
# # plt.plot(t, true_signal, label="True Signal", linestyle="dashed")
# # plt.legend()
# # plt.xlabel("Time [s]")
# # plt.ylabel("Amplitude")
# # plt.title("LMS Adaptive Noise Cancellation")
# # plt.show()

# from sklearn.decomposition import FastICA
# from sys_neuro_tools import fp_utils
# from scipy.optimize import curve_fit

# # Apply ICA
# ica = FastICA()
# X_ica = ica.fit_transform(np.vstack([mixed_signal, reference_noise]).T)

# # Extract the first independent component (assumed to be the shared signal)
# extracted_shared_signal = X_ica[:, 0]
# extracted_unshared_signal = X_ica[:, 1]

# form = lambda x, a, b: a*x + b
# params = curve_fit(form, extracted_shared_signal, reference_noise)[0]
# fitted_shared_signal = form(extracted_shared_signal, *params)

# subtracted_shared_signal = mixed_signal - fitted_shared_signal

# params = curve_fit(form, extracted_unshared_signal, mixed_signal)[0]
# fitted_unshared_signal = form(extracted_unshared_signal, *params)

# plt.figure(figsize=(10, 5))
# plt.plot(t, mixed_signal, label="Noisy Signal", alpha=0.6)
# plt.plot(t, fitted_shared_signal, label="Shared Noise", alpha=0.6)
# plt.plot(t, subtracted_shared_signal, label="Filtered Signal", linewidth=2, alpha=0.6)
# plt.plot(t, fitted_unshared_signal, label="Unshared Signal", alpha=0.6)
# plt.plot(t, true_signal, label="True Signal", linestyle="dashed")
# #plt.plot(t, noise, label="Noise", alpha=0.6)
# #plt.plot(t, extracted_shared_signal, label="Shared Signal", alpha=0.6)
# #plt.plot(t, fitted_shared_signal, label="Fitted Shared Signal", alpha=0.6)
# #plt.plot(t, extracted_unshared_signal, label="Unshared Signal", alpha=0.6)
# plt.legend()
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.title("LMS Adaptive Noise Cancellation")
# plt.show()