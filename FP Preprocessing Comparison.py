# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 10:53:01 2024

@author: tanne
"""

import init
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


# %% Declare methods

def get_all_processed_signals(raw_lig, raw_iso, t, smooth_fit, vary_t, noise_lpf=10, smooth_lpf=0.1):
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
    denoised_lig = fp_utils.filter_signal(raw_lig, noise_lpf, fs)
    denoised_iso = fp_utils.filter_signal(raw_iso, noise_lpf, fs)
    
    baseline_lig = fp_utils.filter_signal(raw_lig, 0.001, fs, filter_type='lowpass')
    baseline_iso = fp_utils.filter_signal(raw_iso, 0.001, fs, filter_type='lowpass')
    
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
        smooth_lig = fp_utils.filter_signal(baseline_corr_lig, smooth_lpf, fs, filter_type='lowpass')
        smooth_iso = fp_utils.filter_signal(baseline_corr_iso, smooth_lpf, fs, filter_type='lowpass')
        _, fit_params = fp_utils.fit_signal(smooth_iso, smooth_lig, t, vary_t=False)

        fitted_baseline_corr_iso = fit_params['formula'](baseline_corr_iso, *fit_params['params'])
        
        #fitted_baseline_corr_iso, _ = fp_utils.fit_signal(baseline_corr_iso, baseline_corr_lig, t, vary_t=False)
        # shift the fitted baseline corrected iso by the ligand baseline to get the reference
        dff_iso_baseline = ((baseline_corr_lig - fitted_baseline_corr_iso)/baseline_lig)*100

    except RuntimeError as error:
        print(str(error))

    return {'raw_lig': raw_lig,
            'raw_iso': raw_iso,
            'baseline_lig': baseline_lig,
            'baseline_iso': baseline_iso,
            'baseline_corr_lig': baseline_corr_lig,
            'baseline_corr_iso': baseline_corr_iso,
            'fitted_iso': fitted_iso,
            'fitted_baseline_corr_iso': fitted_baseline_corr_iso,
            'dff_iso': dff_iso,
            'dff_iso_baseline': dff_iso_baseline}


#%% Get fp data signals

sess_ids = {202: [101965], 191: [102208, 102406]} #{179: [92692], 180: [102327]} #{202: [101965, 101912], 191: [102208, 100301, 101667]}
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

smooth_fit = False
vary_t = False
noise_lpf = 10
smooth_lpf = 0.1

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        raw_signals = fp_data[subj_id][sess_id]['raw_signals']
        t = fp_data[subj_id][sess_id]['time']
        fs = 1/(t[1] - t[0])

        if not 'processed_signals' in fp_data[subj_id][sess_id]:
            fp_data[subj_id][sess_id]['processed_signals'] = {}
            
        fp_data[subj_id][sess_id]['processed_signals'].update({smooth_fit: {vary_t: {}}})

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
            
            fp_data[subj_id][sess_id]['processed_signals'][smooth_fit][vary_t][region] = get_all_processed_signals(raw_lig, raw_iso, t, smooth_fit, vary_t, noise_lpf=noise_lpf, smooth_lpf=smooth_lpf)

# %% Plot processed signal

gen_title = 'Baseline Comparison, Subject {}, Session {}, Smooth Fit: {}, Vary t: {}, Noise LPF: {}, Smooth LPF: {}'
sub_t = [0, np.inf] # [1100, 1120] #
dec = 10

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time']

        fpah.view_processed_signals(fp_data[subj_id][sess_id]['processed_signals'][smooth_fit][vary_t], t, t_min=sub_t[0], t_max=sub_t[1], dec=dec,
                                    title=gen_title.format(subj_id, sess_id, smooth_fit, vary_t, noise_lpf, smooth_lpf))


# %% Compare dF/F for no options vs all options

dec = 10

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        t = fp_data[subj_id][sess_id]['time'][::dec]

        regions = list(fp_data[subj_id][sess_id]['raw_signals'].keys())
        processed_signals = fp_data[subj_id][sess_id]['processed_signals']
        
        fig, axs = plt.subplots(len(regions), 1, figsize=(8, 3*len(regions)), layout='constrained')
        
        for i, region in enumerate(regions):

            ax = axs[i]
            ax.plot(t, processed_signals[False][True][region]['dff_iso'][::dec], label='Trad. dF/F', alpha=0.6)
            ax.plot(t, processed_signals[True][True][region]['dff_iso'][::dec], label='Updated dF/F', alpha=0.6)
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