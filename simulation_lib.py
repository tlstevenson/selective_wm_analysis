# -*- coding: utf-8 -*-
'''
Created on Thu Feb  6 15:28:33 2025

@author: good_

Edited by Alex Truong
'''

import init
import hankslab_db.basicRLtasks_db as db

from hankslab_db import db_access
from pyutils import utils
import matplotlib.pyplot as plt

import numpy as np
from sys_neuro_tools import plot_utils, fp_utils
import statsmodels.api as sm

from scipy.optimize import curve_fit, minimize, OptimizeWarning
from scipy.ndimage import gaussian_filter1d
import warnings

import json

import pickle

import pandas as pd
import os

#%% load the param data calculated from actual recordings 
with open('param_range_comb.json', 'rb') as f:
    param_range_comb = json.load(f)

a_range = [0.1, 10]
a_sd = 2
b_range = [1, 1e5]
b_sd = 100
c_range = [-3e-6, 4e-6]
c_sd = 5e-7
d_range = [1, 10]
d_sd = 5

param_vals = lambda rng, sd: {'max_value': max(rng), 'min_value': min(rng),
                              'mean_plus_SD4': max(rng), 'mean_minus_SD4': min(rng), 'SD': sd}
custom_param_range = {'exp_linear': {'param_0': param_vals(a_range, a_sd),
                                     'param_1': param_vals(b_range, b_sd),
                                     'param_2': param_vals(c_range, c_sd),
                                     'param_3': param_vals(d_range, d_sd)}}

#%% generate true signal and noise 
                                  
def make_signal(time, n_terms=10, amp_range=[0.1, 1], f_range=[0.01, 10]):
    
    signal = np.zeros_like(time) # creates a NumPy array filled with zeros, having the same length as time
   
    for i in range(n_terms):  # accumulates sine wave at random amp & freq for the # of times specified by n_terms 
        rand_amp = np.random.uniform(amp_range[0], amp_range[1])
        rand_f = np.exp(np.random.uniform(np.log(f_range[0]), np.log(f_range[1])))# pick from logarithmic space, then change it back using exp, to pick more from lower freq range
        rand_phase = np.random.rand(1)
        signal = signal + rand_amp * np.sin(2 * np.pi * (rand_f * time + rand_phase))  # this produces sin waves at a random frequency and phase

    # amplitude normalization
    current_max_amp = np.max(np.abs(signal)) # find the maximum absolute amplitude
    if current_max_amp < 1e-12:
        current_max_amp = 1  # avoid divide by zero or skip normalization
    signal = signal / current_max_amp # scale to make |amplitude| <= 1
            
    return signal

def generate_gaussian_noise(time, fs, snr, smooth_sigma=5):

    rng = np.random.default_rng()
    noise = rng.standard_normal(len(time))
    noise = gaussian_filter1d(noise, sigma=smooth_sigma)
    noise /= np.std(noise)
    noise /= snr
    return noise
    
#%% generate artifact

def make_art(time, max_art_count=10, art_duration_range=[1,50], n_terms=5, amp_range=[0.1, 1], f_range=[0.01, 5]):  
    
    # randomly determine n (number of selected time points). 
    n = np.random.randint(1, max_art_count+1) # return random integers from low (inclusive) to high (exclusive)
    
    art_sig_comb = np.zeros_like(time)
    
    fs = 1/(time[1] - time[0])
    
    selected_t_idx = np.random.choice((time * fs).astype(int), n, replace=False)   # picked timepoint (index) to add artifacts. replace=False argument ensures that each selected time point is unique.
    selected_m_idx = np.random.uniform(art_duration_range[0], art_duration_range[1] + 1, size=n) * fs  # return random integers from low (inclusive) to high (exclusive).
    selected_m_idx = selected_m_idx.astype(int)

    for t_idx, m_idx in zip(selected_t_idx, selected_m_idx):   # zip create pairs of selected_t and m    
    
        # ensure m_idx does not exceed the available range in art_sig_com. if exceeds, use the available length instead 
        valid_m_idx = min(m_idx, len(art_sig_comb) - t_idx)
    
        art_t = np.linspace(0, valid_m_idx*1/fs, num=valid_m_idx, endpoint=False)   # define artifact t. linspace produce 50 values by default evenly spaced between the specified range 
        
        art_sig = make_signal(time=art_t, f_range = f_range, n_terms=n_terms, amp_range=amp_range)
        
        art_sig_comb[t_idx:t_idx+valid_m_idx] = art_sig_comb[t_idx:t_idx+valid_m_idx] + art_sig # add the artifact signal from t_idx for the duration set by m_idx
        
        
        # amplitude normalization with safety check to avoid divide-by-zero
        current_max_amp = np.max(np.abs(art_sig_comb))
        if current_max_amp < 1e-12:
            current_max_amp = 1  # avoid dividing by zero or near-zero
        art_sig_comb = art_sig_comb / current_max_amp
    return art_sig_comb

#%% get params for baseline

def param_selector (form_name, use_custom_params=True):   # add a way to be able to adjust the max and min value 
    
    # select params from the dataset (param_range_comb), which was produced from the recorded FP data
    
    params = custom_param_range if use_custom_params else param_range_comb
    
    # print("use_custom_params actually is:", use_custom_params) # debug for ensuring right parameter selection
    
    selected_params = {}
    for param in params[form_name].keys(): 
        df = params[form_name][param]
        limit_min = max(df['mean_minus_SD4'], df['min_value']) # min value is 0 when the param > 0. 
        limit_max = min(df['mean_plus_SD4'], df['max_value']) # this should always be mean_plus_SD4 if using Z = 10 as a cut off  

        # pick a param randomly from the range 
        
        if form_name == 'exp_linear' and param == 'param_1':  
            selected_params[param] = np.exp(np.random.uniform(np.log(limit_min), np.log(limit_max))) # pick from logged ver, then change it back using exp 
        
        elif form_name == 'exp_linear' and param == 'param_3':  
            selected_params[param] = np.random.uniform(1, limit_max)
        
        else: 
            selected_params[param] = np.random.uniform(limit_min, limit_max)
    
    return selected_params


def param_jitter (selected_params, SD_frac, form_name, use_custom_params=True):    
    '''
    add jitter for polynomial parameters, 
    picking from the normal distribution made with SD for each param (eg, param_0)'
    '''
    
    params = custom_param_range if use_custom_params else param_range_comb
    
    jit_selected_params = {}
    jitter = np.random.normal(loc=0, scale=SD_frac)
    for param in selected_params.keys():
        df = params[form_name][param]
        prev_val = selected_params[param]
        
        # DEBUG PRINT
        #print(f'{param}: SD = {df['SD']}, SD_frac = {SD_frac}')
        
        jitter = np.random.normal(loc=0, scale=SD_frac)
        
        # exponential decay should be inversely proportional to the rest of the parameters
        if form_name == 'exp_linear' and param == 'param_1':
            new_val = np.exp(np.log(prev_val) - np.log(df['SD'])*jitter)
        # always make intercept be smaller for iso
        elif form_name == 'exp_linear' and param in ['param_0', 'param_3']:
            new_val = prev_val - df['SD']*np.abs(jitter)
        # elif form_name == 'exp_linear' and param == 'param_3':
            
        else:
            new_val = prev_val + df['SD']*jitter
       
        # set the same min and max as the input (lig) params     
        limit_min = max(df['mean_minus_SD4'], df['min_value']) 
        limit_max = min(df['mean_plus_SD4'], df['max_value'])
        
        if form_name == 'exp_linear' and param in ['param_3']:  
            clipped = np.clip(new_val, 1, limit_max)
        else:
            clipped = np.clip(new_val, limit_min, limit_max)

        #print(f'[JITTER DEBUG] {param}: before={before:.4g}, jitter={jitter:.4g}, after={after:.4g}, clipped={clipped:.4g}')
        
        jit_selected_params[param] = clipped

    return jit_selected_params

#%% baseline formula 
def get_baseline_form (form_type):  
    
    match form_type:
        
        case 'double_exp_decay':
            baseline_form = lambda x, a, b, c, d, e: a*np.exp(-x/b) + c*np.exp(-x/(b*d)) + e
            # double exponential decay 
            
            # specifying the boundary for each parameter (only for the exponential ones) 
            #               a       b      c     d     e 
            bounds = ([0      ,      0, -np.inf, 0, -np.inf],
                      [ np.inf, np.inf,  np.inf, 1,  np.inf])
          
        case 'exp_linear':  #combination of the exponential decay term and a linear term 
            baseline_form = lambda x, a, b, c, d: a*np.exp(-x/b) - c*x +d 
                        
            #               a       b      c     d     
            bounds = ([0      ,      0, -np.inf,  -np.inf],
                      [ np.inf, np.inf,  np.inf,  np.inf])
              
        case 'polynomial':  # works for different degrees, depending on the number of coeffs you provide
                            # 2 coeffs -> linear, 3 coeffs -> quadratic,  4 coeffs -> cubic, 5 coeffs -> quartic, 6 coeffs -> quintic 
            baseline_form = lambda x, *coeffs: np.polyval(coeffs, x) 
           
            bounds = None 
 
    return {'formula': baseline_form, 'bounds': bounds}     


#%% generate baseline

def generate_baseline (form_type, SD_frac, time):  
    
    # for time, the input is in second. converting to an index when apply 
    
    while True:
        lig_params = param_selector (form_type, use_custom_params=True)
        iso_params = param_jitter (lig_params, SD_frac, form_type, use_custom_params=True)
    
        # params.values() seems to change the order of params. Thus, sorting based on the key - # may move this to combining function 
        lig_param_list = [lig_params[key] for key in sorted(lig_params.keys(), key=lambda x: int(x.split('_')[1]))]
        iso_param_list = [iso_params[key] for key in sorted(iso_params.keys(), key=lambda x: int(x.split('_')[1]))]
        
       	# get the one baseline_form to use
        baseline_form = get_baseline_form(form_type)
       
        # generate baselines 
        baseline_lig = baseline_form ['formula'] (np.arange(len(time)), *lig_param_list)
        baseline_iso = baseline_form ['formula'] (np.arange(len(time)), *iso_param_list)
    
        if np.all(baseline_lig > 0) and np.all(baseline_iso > 0):   # if either of the baselines are 0 or negative, generator another set of params and generate baselines 
            return {
                    'lig_params':lig_params,
                    'iso_params': iso_params,
                    'baseline_lig': baseline_lig,
                    'baseline_iso': baseline_iso
                    }


#%% combine baseline, signal, artifact and noise

def simulate_signal (time, true_sig, art, noise, sim_type, form_type, current_SD_frac, alpha=0, SNR=1, SAR=10, scale=0.1, rms_scale=False):
    
    baselines = generate_baseline (form_type, current_SD_frac, time)
    baseline_lig = baselines['baseline_lig']
    baseline_iso = baselines['baseline_iso']
    
    # rescale signal amplitudes (signal to noise ratio, signal to artifact ratio)
    
    # rescale true_sig to desired scale (define the signal RMS scale)
    scaled_true_sig = true_sig * scale
    
    if rms_scale:
        true_rms = np.sqrt(np.mean(scaled_true_sig ** 2))
    
        # rescale noise and art to get exact RMS ratios
        noise_rms = np.sqrt(np.mean(noise ** 2))
        
        if noise_rms < 1e-12 or np.isnan(noise_rms):
            noise_rms = 1
        
        noise = noise / noise_rms * (true_rms / SNR)
    
        art_rms = np.sqrt(np.mean(art ** 2))
        
        if art_rms < 1e-12 or np.isnan(art_rms):
            art_rms = 1
            
        art = art / art_rms * (true_rms / SAR)
    else:
        noise = noise * scale / SNR
    
        art_rms = np.sqrt(np.mean(art ** 2))
        
        if art_rms < 1e-12 or np.isnan(art_rms):
            art_rms = 1
            
        art = art * scale / SAR
    
    # debug to check for NaNs or 0
    #print('scaled_true_sig stats: min =', np.min(scaled_true_sig), ', max =', np.max(scaled_true_sig))
    #print('artifact stats: min =', np.min(art), ', max =', np.max(art))
    #print('baseline_iso stats: min =', np.min(baseline_iso), ', max =', np.max(baseline_iso))
    #print('baseline_lig stats: min =', np.min(baseline_lig), ', max =', np.max(baseline_lig))

    #print('Number of points where scaled_true_sig < -1:', np.sum(scaled_true_sig < -1))
    #print('Number of points where art < -1:', np.sum(art < -1))
    
    match sim_type:
        case 1:
            raw_lig = baseline_lig * (((scaled_true_sig + art) + 1) + noise)
            raw_iso = baseline_iso * (((scaled_true_sig * alpha + art) + 1) + noise)
        
        case 2:
            raw_lig = baseline_lig * ((scaled_true_sig + 1) * (art + 1) + noise)
            raw_iso = baseline_iso * ((scaled_true_sig * alpha + 1) * (art + 1) + noise)

    return {'raw_lig': raw_lig, 'raw_iso': raw_iso, 'scaled_true_sig': scaled_true_sig, 'artifact': art, 'noise': noise, **baselines}


def simulate_n_signals(n, time,
                       param_name, param_range, param_step,
                       f_range_sig, 
                       f_range_noise,
                       max_art_count, art_duration_range, f_range_art,
                       form_type, sim_type,
                       SD_frac_default,
                       alpha_default,
                       SNR_default,
                       SAR_default,
                       scale,
                       fs,
                       smooth_sigma):
    
    simulated_signals = {}
    
    for param_value in np.arange(param_range[0], param_range[1]+param_step, param_step):
       simulated_signals[param_value] = []

       for _ in range(n):
           
           # Override parameter depending on param_name
           current_SD_frac = param_value if param_name == 'SD_frac' else SD_frac_default
           current_alpha = param_value if param_name == 'alpha' else alpha_default
           current_SNR = param_value if param_name == 'SNR' else SNR_default
           current_SAR = param_value if param_name == 'SAR' else SAR_default
           
           # Generate signal, noise, artifacts
           true_sig = make_signal(time, f_range = f_range_sig)
           # noise = make_signal(time, f_range = f_range_noise)      # this was used for noise composed of sinusoids
           noise = generate_gaussian_noise(time, fs, snr=current_SNR, smooth_sigma=smooth_sigma)
           art = make_art(time, max_art_count, art_duration_range, f_range = f_range_art)

           signal_data = simulate_signal(time, true_sig, art, noise, 
                                              sim_type, form_type, current_SD_frac,
                                              current_alpha,
                                              current_SNR, current_SAR,
                                              scale=scale)

           simulated_signals[param_value].append(signal_data)

    return simulated_signals


#%% process_signals 

clamp_total_points = 0
clamp_total_calls = 0

def process_signals(raw_lig, raw_iso, baseline_iso, time, fs, lpf_general = 10, lpf_baseline = 0.0005, iso_bands=[[0,0.1], [0.1,1], [1,10]]):
    """
    Run OLS (basic), IRLS, LPF Baseline Subtraction, Frequency Band,
    and Frequency Band + LPF Baseline Subtraction fits for comparison.
    Returns dictionary with keys:
    'OLS', 'IRLS', 'LPF_only', 'FreqBand', 'FreqBand_LPF'

    iso_bands: list of lists, optional frequency bands for frequency band methods.

    Notes:
    - filtered with lpf_general for OLS, IRLS, FreqBand
    - baseline subtraction uses lpf_baseline
    """
    global clamp_total_points, clamp_total_calls

    epsilon = 1e-6  # avoid zero denominators
    results = {}

    # --- Filter signals with general LPF ---
    filtered_lig = fp_utils.filter_signal(raw_lig, cutoff_f=lpf_general, sr=fs)
    filtered_iso = fp_utils.filter_signal(raw_iso, cutoff_f=lpf_general, sr=fs)

    # --- OLS (standard) ---
    fitted_ols, fit_info_ols = fp_utils.fit_signal(filtered_iso, filtered_lig, time, vary_t=False)
    fitted_ols_baseline = fit_info_ols['formula'](baseline_iso, *fit_info_ols['params'])

    denom = np.clip(fitted_ols, epsilon, None)
    dff_ols = ((filtered_lig - fitted_ols) / denom)

    clamp_total_points += np.sum(denom == epsilon)
    clamp_total_calls += 1

    results['OLS'] = {
        'raw_lig': raw_lig,
        'fitted_iso': fitted_ols,
        'fitted_iso_baseline': fitted_ols_baseline,
        'dff': dff_ols,
        'fit_params': fit_info_ols['params'],
        'filt_t': np.arange(len(time)),
    }

    # --- IRLS ---
    not_nans = ~np.isnan(filtered_lig) & ~np.isnan(filtered_iso)
    fitted_irls = np.full_like(filtered_lig, np.nan)

    exog = sm.add_constant(filtered_iso[not_nans])
    endog = filtered_lig[not_nans]
    rlm_mod = sm.RLM(endog, exog, M=sm.robust.norms.TukeyBiweight(c=3))
    rlm_res = rlm_mod.fit()

    fitted_irls[not_nans] = rlm_res.fittedvalues
    irls_params = [float(rlm_res.params[1]), float(rlm_res.params[0])]
    irls_formula = lambda x, a, b: a * x + b
    irls_baseline = irls_formula(baseline_iso, *irls_params)

    denom = np.clip(fitted_irls, epsilon, None)
    dff_irls = ((filtered_lig - fitted_irls) / denom)

    clamp_total_points += np.sum(denom == epsilon)
    clamp_total_calls += 1

    results['IRLS'] = {
        'raw_lig': raw_lig,
        'fitted_iso': fitted_irls,
        'fitted_iso_baseline': irls_baseline,
        'dff': dff_irls,
        'fit_params': irls_params,
        'filt_t': np.arange(len(time)),
    }

    # --- LPF Baseline Subtraction ---
    trend_pad_len = int((1 / (2 * lpf_baseline)) * fs)

    # compute LPF baseline using baseline LPF
    baseline_lig = fp_utils.filter_signal(filtered_lig, cutoff_f=lpf_baseline, sr=fs, trend_pad_len=trend_pad_len)
    baseline_iso = fp_utils.filter_signal(filtered_iso, cutoff_f=lpf_baseline, sr=fs, trend_pad_len=trend_pad_len)

    # baseline corrected signals for fitting
    baseline_corr_lig = filtered_lig - baseline_lig
    baseline_corr_iso = filtered_iso - baseline_iso

    # fit isosbestic to baseline corrected signals
    fitted_baseline_iso, fit_info_lpf = fp_utils.fit_signal(
        baseline_corr_iso, baseline_corr_lig, time, vary_t=False
    )

    # compute dF/F using baseline in denominator
    denom = np.clip(baseline_lig + fitted_baseline_iso, epsilon, None)
    dff_lpf = ((baseline_corr_lig - fitted_baseline_iso) / denom)

    clamp_total_points += np.sum(denom == epsilon)
    clamp_total_calls += 1

    results['LPF_only'] = {
        'raw_lig': raw_lig,
        'raw_iso': raw_iso,
        'baseline_lig': baseline_lig,
        'baseline_iso': baseline_iso,
        'corr_lig': baseline_corr_lig,
        'corr_iso': baseline_corr_iso,
        'fitted_iso': fitted_baseline_iso,
        'dff': dff_lpf,
        'fit_params': fit_info_lpf['params'],
        'filt_t': np.arange(len(time)),
    }

    # --- Frequency Band ---
    # use general filtered signals for fitting
    fitted_fband_iso, fit_info_fband = fp_utils.fit_signal(
        filtered_iso, filtered_lig, time, vary_t=False, fit_bands=True, f_bands=iso_bands
    )

    denom = np.clip(fitted_fband_iso, epsilon, None)
    dff_fband = ((filtered_lig - fitted_fband_iso) / denom)

    clamp_total_points += np.sum(denom == epsilon)
    clamp_total_calls += 1

    results['FreqBand'] = {
        'raw_lig': raw_lig,
        'raw_iso': raw_iso,
        'fitted_iso': fitted_fband_iso,
        'dff': dff_fband,
        'fit_params': fit_info_fband['params'],
        'filt_t': np.arange(len(time)),
    }

    # --- Frequency Band + LPF Baseline Subtraction ---
    baseline_corr_lig = filtered_lig - baseline_lig
    baseline_corr_iso = filtered_iso - baseline_iso

    fitted_fband_lpf_iso, fit_info_fband_lpf = fp_utils.fit_signal(
        baseline_corr_iso, baseline_corr_lig, time, vary_t=False, fit_bands=True, f_bands=iso_bands
    )

    denom = np.clip(baseline_lig + fitted_fband_lpf_iso, epsilon, None)
    dff_fband_lpf = ((baseline_corr_lig - fitted_fband_lpf_iso) / denom)

    clamp_total_points += np.sum(denom == epsilon)
    clamp_total_calls += 1

    results['FreqBand_LPF'] = {
        'raw_lig': raw_lig,
        'raw_iso': raw_iso,
        'baseline_lig': baseline_lig,
        'baseline_iso': baseline_iso,
        'corr_lig': baseline_corr_lig,
        'corr_iso': baseline_corr_iso,
        'fitted_iso': fitted_fband_lpf_iso,
        'dff': dff_fband_lpf,
        'fit_params': fit_info_fband_lpf['params'],
        'filt_t': np.arange(len(time)),
    }

    return results


#%% calculate explained variance

def ev(true_sig, dff):
    
    numerator = np.var(true_sig - dff)
    denominator = np.var(true_sig)
    return 1 - (numerator / denominator)


#%% exclude outliners as needed 

def remove_outliers (data, threshold=2):
    data = np.array(data)
    mean, std = np.mean(data), np.std(data)
    z_scores = (data - mean) / std
    return data[np.abs(z_scores) < threshold] 


#%% plot ev - line plot 

def plot_ev_results(ev_results, DV_name, exclude_outliers=False, alpha_default=None, SD_frac_default=None, SNR_default=None, SAR_default=None, method_labels=None):
    plt.figure(figsize=(6, 5))
    
    DVs = sorted([float(k) for k in ev_results.keys()])
    methods = list(next(iter(ev_results.values())).keys())
    
    for method in methods:
        y_values = []
        y_errors = []
        
        for DV in DVs:
            data = ev_results[DV][method]
            n_sims = len(data)
            if exclude_outliers:
                clean_data = remove_outliers(data)
            else:
                clean_data = data
                
            y_values.append(np.mean(clean_data))
            y_errors.append(np.std(clean_data) / np.sqrt(len(clean_data)))

        if method_labels is None:
            method_label = method
        else:
            method_label = method_labels[method]
        
        plt.errorbar(DVs, y_values, yerr=y_errors, marker='o', capsize=5, label=method_label)
        
    plt.xlabel(DV_name)
    plt.ylabel('EV')
    
    if exclude_outliers:
        title = f'EV Results - n = {n_sims} per x, Outlier excluded at z=2, varying {DV_name}\n'
    else:
        title = f'EV Results - n = {n_sims} per x, Outlier not excluded, varying {DV_name}\n'
       
    match DV_name:
        case 'alpha':
            title += f'SD_frac = {SD_frac_default}, SNR = {SNR_default}, SAR = {SAR_default}'
        case 'SD_frac':
            title += f'alpha = {alpha_default}, SNR = {SNR_default}, SAR = {SAR_default}'
        case 'SNR':
            title += f'alpha = {alpha_default}, SD_frac = {SD_frac_default}, SAR = {SAR_default}'
        case 'SAR':
            title += f'alpha = {alpha_default}, SD_frac = {SD_frac_default}, SNR = {SNR_default}'
            
    plt.title(title)
    
    plt.legend()
    plt.grid()
    plt.ylim(-0.05, 1.05)
    plt.show()

#%% helper for plotting fitted iso consistently across methods

def get_plot_fitted_iso(entry, method_key):
    """
    Return a fluorescence-like fitted iso for plotting.
    LPF-based methods store residuals and need baseline added back.
    """
    if entry is None or not isinstance(entry, dict):
        return None

    fitted_iso = entry.get('fitted_iso', None)
    if fitted_iso is None:
        return None

    if method_key in ['LPF_only', 'FreqBand_LPF']:
        baseline_iso = entry.get('baseline_iso', None)
        if baseline_iso is None:
            return None
        return baseline_iso + fitted_iso

    return fitted_iso

#%% plot the signals processed with all current methods
def plot_comparative_figures(
    raw_lig, raw_iso, baseline_iso, time, true_sig, fs=200,
    suptitle_text=None, ev=None, dv=None, param_name=None, extra_title=None):
    """
    Plot raw signals, fitted isos, and dF/F across all processing methods returned by process_signals.
    Automatically skips missing data. Works with single or multiple entries per method.
    """
    import matplotlib.pyplot as plt

    # --- Run processing ---
    results = process_signals(raw_lig, raw_iso, baseline_iso, time, fs)

    # --- Methods to plot ---
    method_labels = ['OLS', 'IRLS', 'LPF_only', 'FreqBand', 'FreqBand_LPF']
    n_methods = len(method_labels)

    fig, axes = plt.subplots(n_methods, 2, figsize=(12, 3 * n_methods), sharex=True)
    if n_methods == 1:
        axes = axes.reshape(1, 2)
        
    # --- Subhead titles for left/right panels ---
    fig.subplots_adjust(top=0.92)  # make room for suptitle
    axes[0, 0].set_title("Raw Ligand vs Fitted Iso", fontsize=12)
    axes[0, 1].set_title("True Signal vs dF/F", fontsize=12)

    for idx, method_key in enumerate(method_labels):
        entries = results.get(method_key, None)

        # Skip if missing or empty
        if entries is None or len(entries) == 0:
            axes[idx, 0].set_visible(False)
            axes[idx, 1].set_visible(False)
            continue

        # Ensure entries is always a list
        if isinstance(entries, dict):
            entries = [entries]

        ax_left = axes[idx, 0]
        ax_right = axes[idx, 1]

        # Plot raw ligand and true signal once per method
        ax_left.plot(time, raw_lig, label='raw_lig', color='skyblue', alpha=0.7)
        ax_right.plot(time, true_sig, label='true_sig', color='orange', alpha=0.7)

        for entry in entries:
            if not isinstance(entry, dict):
                continue  # skip if something went wrong
                
            if 'fitted_iso' in entry and entry['fitted_iso'] is not None:
                plot_iso = entry['fitted_iso']
                if method_key in ['LPF_only', 'FreqBand_LPF']:
                    plot_iso = plot_iso + baseline_iso
                ax_left.plot(time, plot_iso, label=f'{method_key}', color='orange', alpha=0.7)
            if 'dff' in entry and entry['dff'] is not None:
                ax_right.plot(time, entry['dff'], label=f'{method_key}', color='skyblue', alpha=0.7)

        ax_left.set_ylabel(method_key, fontsize=8)
        ax_left.legend(loc='upper right')
        ax_right.legend(loc='upper right')

    # --- Set x labels ---
    for ax in axes[-1, :]:
        ax.set_xlabel('Time (s)')

    # --- Suptitle with EV info ---
    main_title = "Signal Comparison Across Processing Methods"
    if ev is not None and dv is not None:
        if param_name is not None:
            main_title += f" | Lowest EV = {ev:.2f} | {param_name} = {float(dv):.2f}"
        else:
            main_title += f" | Lowest EV = {ev:.2f} | DV = {float(dv):.2f}"
    if suptitle_text is not None:
        main_title += f" | {suptitle_text}"
    if extra_title is not None:
        main_title += f" | {extra_title}"

    fig.suptitle(main_title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


#%% for troubleshooting / visualization, plot some of the signals as needed 

def plot_signals(processed_signals, true_sig, t, ev, fs=200, title='Signal Overview'):
  
    raw_iso = processed_signals['raw_iso']
    fitted_iso = processed_signals['fitted_iso']
    raw_lig = processed_signals['raw_lig']

    fig, axs = plt.subplots(4, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    plt.suptitle(f'{title} (EV = {ev:.2f})')

    axs[0].plot(t, true_sig, color='black', label='True Signal')
    axs[0].set_ylabel('True Sig (V)')
    axs[0].set_title('True Signal')
    axs[0].legend()

    axs[1].plot(t, raw_iso, color='red', label='Raw Iso')
    axs[1].set_ylabel('Raw Iso (V)')
    axs[1].set_title('Raw Isosbestic Signal')
    axs[1].legend()

    axs[2].plot(t, fitted_iso, color='blue', label='Fitted Iso')
    axs[2].set_ylabel('Fitted Iso (V)')
    axs[2].set_title('Fitted Isosbestic Signal')
    axs[2].legend()

    axs[3].plot(t, raw_lig, color='green', label='Raw Ligand')
    axs[3].set_ylabel('Raw Lig (V)')
    axs[3].set_title('Raw Ligand Signal')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend()

    return fig

#%% saving signals in a table with each row corresponding to a signal

def signal_to_row(signal_dict, DV, param_name, method=None, ev=None, index=None, signal_keys=None):
    """
    Flatten one simulated signal and related arrays into a single row dict.
    """
    row = {
        'DV': float(DV),
        'param_name': param_name,
        'index': index,
        'method': method,
        'ev': ev,
    }

    # Add baseline params
    for k, v in signal_dict.get('lig_params', {}).items():
        row[f'lig_{k}'] = v
    for k, v in signal_dict.get('iso_params', {}).items():
        row[f'iso_{k}'] = v

    # Default keys to store
    default_signal_keys = ['raw_lig', 'raw_iso', 'baseline_iso', 'scaled_true_sig', 'artifact', 'noise']
    keys = signal_keys if signal_keys is not None else default_signal_keys

    for key in keys:
        val = signal_dict.get(key, None)
        row[key] = np.asarray(val) if val is not None else None

    return row


def signals_to_dataframe(simulated_signals, ev_results, param_name, signal_keys=None):
    """
    Convert nested simulated_signals and ev_results into a long-form pandas DataFrame.
    """
    all_rows = []
    for DV, signal_list in simulated_signals.items():
        methods_for_dv = ev_results.get(DV, {})
        for idx, signal in enumerate(signal_list):
            for method, ev_list in methods_for_dv.items():
                ev_val = ev_list[idx]
                row = signal_to_row(signal, DV, param_name, method=method, ev=ev_val, index=idx, signal_keys=signal_keys)
                all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)


def save_dataframe(df, filepath):
    """
    Ensure directory exists and save DataFrame to pickle.
    """
    folder = os.path.dirname(filepath)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    df.to_pickle(filepath)


def load_or_create_dataframe(filepath):
    """
    Load a pickled DataFrame if present, else return an empty DataFrame.
    """
    if os.path.exists(filepath):
        return pd.read_pickle(filepath)
    else:
        return pd.DataFrame()

#%% retrieving signals

def retrieve_signals(path_or_df, DV=None, method=None, ev_thresh=None, index=None, signal_keys=None):
    """
    Load a saved DataFrame (or accept a DataFrame), filter rows by DV/method/ev_thresh/index,
    and return (filtered_df, matched_signals).
    """
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        if not os.path.exists(path_or_df):
            raise FileNotFoundError(f"File not found: {path_or_df}")
        df = pd.read_pickle(path_or_df)

    # Apply filters
    if DV is not None:
        df = df[df['DV'] == float(DV)]
    if method is not None:
        df = df[df['method'] == method]
    if ev_thresh is not None:
        df = df[df['ev'] >= ev_thresh]
    if index is not None:
        df = df[df['index'] == index]

    if df.empty:
        return df, []

    metadata_cols = {'DV', 'param_name', 'index', 'method', 'ev'}
    param_prefixes = ('lig_', 'iso_')

    auto_signal_cols = [c for c in df.columns if c not in metadata_cols and not c.startswith(param_prefixes)]

    if signal_keys is not None:
        signal_cols = [c for c in signal_keys if c in df.columns]
    else:
        signal_cols = auto_signal_cols

    matched_signals = df[signal_cols].to_dict(orient='records')

    return df, matched_signals
