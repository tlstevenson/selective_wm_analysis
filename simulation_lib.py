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

from scipy.optimize import curve_fit, minimize, OptimizeWarning
import warnings

import json

#%% load the param data calculated from actual recordings 
with open('param_range_comb.json', 'rb') as f:
    param_range_comb = json.load(f)

a_range = [0.1,10]
a_sd = 2
b_range = [1,1e5]
b_sd = 100
c_range = [-3e-6, 4e-6]
c_sd = 5e-7
d_range = [1,10]
d_sd = 3

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
    
    selected_params = {}
    for param in params[form_name].keys(): 
        df = params[form_name][param]
        limit_min = max(df['mean_minus_SD4'], df['min_value']) # min value is 0 when the param > 0. 
        limit_max = min(df['mean_plus_SD4'], df['max_value']) # this should always be mean_plus_SD4 if using Z = 10 as a cut off  

        # pick a param randomly from the range 
        
        if form_name == 'exp_linear' and param == 'param_1':  
            selected_params[param] = np.exp(np.random.uniform(np.log(limit_min), np.log(limit_max))) # pick from logged ver, then change it back using exp 
        
        elif form_name == 'exp_linear' and param == 'param_3':  
            selected_params[param] = np.random.uniform(0, limit_max)
        
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
        else:
            new_val = prev_val + df['SD']*jitter
       
        # set the same min and max as the input (lig) params     
        limit_min = max(df['mean_minus_SD4'], df['min_value']) 
        limit_max = min(df['mean_plus_SD4'], df['max_value'])
        
        if form_name == 'exp_linear' and param in ['param_3']:  
            clipped = np.clip(new_val, 0, limit_max)
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
        lig_params = param_selector (form_type)
        iso_params = param_jitter (lig_params, SD_frac, form_type)
    
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
                       scale):
    
    simulated_signals = {}
    
    for param_value in np.arange(param_range[0], param_range[1]+param_step, param_step):
       simulated_signals[param_value] = []

       for _ in range(n):
           true_sig = make_signal(time, f_range = f_range_sig)
           noise = make_signal(time, f_range = f_range_noise)
           art = make_art(time, max_art_count, art_duration_range, f_range = f_range_art)

           # Override parameter depending on param_name
           current_SD_frac = param_value if param_name == 'SD_frac' else SD_frac_default
           current_alpha = param_value if param_name == 'alpha' else alpha_default
           current_SNR = param_value if param_name == 'SNR' else SNR_default
           current_SAR = param_value if param_name == 'SAR' else SAR_default

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

def process_signals(raw_lig, raw_iso, baseline_iso, time, fs, smooth_fit=True, vary_t=True, filt_denom=True, smooth_lpf=0.1): 
    global clamp_total_points, clamp_total_calls

    epsilon = 1e-6  # minimum denominator to avoid division by zero or near zero

    if smooth_fit: 
        smooth_lig = fp_utils.filter_signal(raw_lig, smooth_lpf, fs)   
        smooth_iso = fp_utils.filter_signal(raw_iso, smooth_lpf, fs) 

        if vary_t:
            fitted_smooth_iso, smooth_fit_info = fp_utils.fit_signal(smooth_iso, smooth_lig, time, vary_t=True)
            s_to_fit = np.vstack((raw_iso[None], time[None])) 
            baseline_to_fit = np.vstack((baseline_iso[None], time[None])) 

        else: 
            fitted_smooth_iso, smooth_fit_info = fp_utils.fit_signal(smooth_iso, smooth_lig, time, vary_t=False)
            s_to_fit = raw_iso
            baseline_to_fit = baseline_iso

        # for smooth fit, get new fitted iso , using the smooth_fit_info 
        fitted_iso = smooth_fit_info['formula'](s_to_fit, *smooth_fit_info['params'])

        # also regress baseline to see how close baseline_iso goes to baseline_lig 
        fitted_iso_baseline = smooth_fit_info['formula'](baseline_to_fit, *smooth_fit_info['params'])
        
        # Remove any portion of signals that go below 0
        filt_t = np.full(len(raw_lig), True)
        filt_thresh = np.argmax(fitted_iso < epsilon)
        # handles condition where there ar no signals below epsilon
        if fitted_iso[filt_thresh] < epsilon:
            filt_t[filt_thresh:] = False

        # clamp denominator to avoid division by zero or near-zero
        denom = fitted_smooth_iso if filt_denom else fitted_iso
        denom = np.clip(denom, epsilon, None)

        # count how many points were clamped (denominator was below epsilon)
        num_clamped = np.sum(denom == epsilon)
        clamp_total_points += num_clamped
        clamp_total_calls += 1

        dff = ((raw_lig - fitted_iso) / denom)

        # added raw_lig
        return {
            'raw_lig': raw_lig,
            'smooth_lig':smooth_lig,
            'smooth_iso':smooth_iso,
            'fitted_smooth_iso':fitted_smooth_iso,
            'fitted_iso': fitted_iso,
            'fitted_iso_baseline':fitted_iso_baseline,
            'dff': dff,
            'fit_params': smooth_fit_info['params'],
            'filt_t': filt_t}

    else:
        if vary_t:
            fitted_iso, fit_info = fp_utils.fit_signal(raw_iso, raw_lig, time, vary_t=True)
            baseline_to_fit = np.vstack((baseline_iso[None], time[None])) 

        else: 
            fitted_iso, fit_info = fp_utils.fit_signal(raw_iso, raw_lig, time, vary_t=False)  
            baseline_to_fit = baseline_iso       

        # also regress baseline to see how close baseline_iso goes to baseline_lig 
        fitted_iso_baseline = fit_info['formula'](baseline_to_fit, *fit_info['params'])
        
        # Remove any portion of signals that go below 0
        filt_t = np.full(len(raw_lig), True)
        filt_thresh = np.argmax(fitted_iso < epsilon)
        # handles condition where there ar no signals below epsilon
        if fitted_iso[filt_thresh] < epsilon:
            filt_t[filt_thresh:] = False

        if filt_denom:
            smooth_fitted_iso = fp_utils.filter_signal(fitted_iso, smooth_lpf, fs)  
            denom = np.clip(smooth_fitted_iso, epsilon, None)
        else:
            denom = np.clip(fitted_iso, epsilon, None)

        num_clamped = np.sum(denom == epsilon)
        clamp_total_points += num_clamped
        clamp_total_calls += 1

        dff = ((raw_lig - fitted_iso) / denom)

        #Added raw_lig
        return {
            'raw_lig': raw_lig,
            'fitted_iso': fitted_iso,
            'fitted_iso_baseline':fitted_iso_baseline,
            'dff': dff,
            'fit_params': fit_info['params'],
            'filt_t': filt_t
        }
    
''' no clamping version
def process_signals (raw_lig, raw_iso, baseline_iso, time, fs, smooth_fit=True, vary_t=True, filt_denom=True, smooth_lpf=0.1): 
    
    if smooth_fit: 
        smooth_lig = fp_utils.filter_signal(raw_lig, smooth_lpf, fs)   
        smooth_iso = fp_utils.filter_signal(raw_iso, smooth_lpf, fs) 
    
        if vary_t:
            fitted_smooth_iso, smooth_fit_info = fp_utils.fit_signal(smooth_iso, smooth_lig, time, vary_t=True)
            s_to_fit = np.vstack((raw_iso[None], time[None])) 
            baseline_to_fit = np.vstack((baseline_iso[None], time[None])) 
    
        else: 
            fitted_smooth_iso, smooth_fit_info = fp_utils.fit_signal(smooth_iso, smooth_lig, time, vary_t=False)
            s_to_fit = raw_iso
            baseline_to_fit = baseline_iso
            
        #for smooth fit, get new fitted iso , using the smooth_fit_info 
        fitted_iso = smooth_fit_info['formula'](s_to_fit, *smooth_fit_info['params'])
                        
        #also regress baseline to see how close baseline_iso goes to baseline_lig 
        fitted_iso_baseline = smooth_fit_info['formula'](baseline_to_fit, *smooth_fit_info['params'])
        
        if filt_denom:
            denom = fp_utils.filter_signal(fitted_iso, smooth_lpf, fs)  
        else:
            denom = fitted_iso
        
        dff = ((raw_lig - fitted_iso) / denom)
    
        return {
                'smooth_lig':smooth_lig,
                'smooth_iso':smooth_iso,
                'fitted_smooth_iso':fitted_smooth_iso,
                'fitted_iso': fitted_iso,
                'fitted_iso_baseline':fitted_iso_baseline,
                'dff': dff,
                'fit_params': smooth_fit_info['params']}
    
    else:
        if vary_t:
            fitted_iso, fit_info   = fp_utils.fit_signal(raw_iso, raw_lig, time, vary_t=True)
            baseline_to_fit = np.vstack((baseline_iso[None], time[None])) 
    
        else: 
            fitted_iso, fit_info   = fp_utils.fit_signal(raw_iso, raw_lig, time, vary_t=False)  
            baseline_to_fit = baseline_iso       
            
    
        #also regress baseline to see how close baseline_iso goes to baseline_lig 
        fitted_iso_baseline = fit_info['formula'](baseline_to_fit, *fit_info['params'])       
        
        if filt_denom:
            smooth_fitted_iso = fp_utils.filter_signal(fitted_iso, smooth_lpf, fs)  
            denom = smooth_fitted_iso
        else:
            denom = fitted_iso
        
        dff = ((raw_lig - fitted_iso) / denom)
    
        return {
                'fitted_iso': fitted_iso,
                'fitted_iso_baseline':fitted_iso_baseline,
                'dff': dff,
                'fit_params': fit_info['params']
                 }
'''

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


#%% plot ev  - line plot 

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


#%% plot the signals that are processed with each of the 4 different processing steps against each other

def plot_comparative_figures(raw_lig, raw_iso, baseline_iso, time, true_sig, fs=200, smooth_lpf=0.1, suptitle_text=None, ev=None, dv=None, param_name=None):
    processing_conditions = [
        (True, True),
        (True, False),
        (False, True),
        (False, False)
    ]

    results = []
    for smooth_fit, vary_t in processing_conditions:
        result = process_signals(
            raw_lig, raw_iso, baseline_iso, time, fs,
            smooth_fit=smooth_fit,
            vary_t=vary_t,
            filt_denom=True,
            smooth_lpf=smooth_lpf
        )
        results.append({
            'smooth_fit': smooth_fit,
            'vary_t': vary_t,
            'raw_lig': raw_lig,
            'raw_iso': raw_iso,
            'baseline_iso': baseline_iso,
            **result
        })

    # Prepare EV/DV text for titles if provided
    ev_dv_str = ''
    if ev is not None and dv is not None:
        label = param_name if param_name is not None else 'DV'
        ev_dv_str = f' | Lowest EV = {ev:.2f} | {label} = {float(dv):.2f}'

    # === FIGURE 1: 4x2 GRID === #
    fig1, axs = plt.subplots(4, 2, figsize=(14, 12), sharex=True)

    main_title_fig1 = 'Signal Comparison Across Processing Methods' + ev_dv_str
    fig1.suptitle(main_title_fig1, fontsize=16, y=0.98)

    for i, res in enumerate(results):
        label = f'smooth={res['smooth_fit']}, vary_t={res['vary_t']}'

        axs[i, 0].plot(time, res['raw_lig'], label='raw_lig', alpha=0.6)
        axs[i, 0].plot(time, res['fitted_iso'], label='fitted_iso', alpha=0.6)
        if res['smooth_fit']:
            axs[i, 0].plot(time, res['smooth_lig'], label='smooth_lig', linestyle='--')
            axs[i, 0].plot(time, res['smooth_iso'], label='smooth_iso', linestyle='--')
        axs[i, 0].set_ylabel(label)
        axs[i, 0].legend(loc='upper right')

        axs[i, 1].plot(time, true_sig, label='true_sig', color='blue', alpha=0.7)
        axs[i, 1].plot(time, res['dff'], label='dF/F', color='green', alpha=0.6)
        axs[i, 1].legend(loc='upper right')

    axs[3, 0].set_xlabel('Time (s)')
    axs[3, 1].set_xlabel('Time (s)')

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # leave space for suptitle
    plt.show()

    # === FIGURE 2: Overlayed comparisons === #
    fig2, axs2 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    main_title_fig2 = 'Overlayed Signal Comparisons Across Methods' + ev_dv_str
    fig2.suptitle(main_title_fig2, fontsize=16, y=0.98)

    for res in results:
        label = f'smooth={res['smooth_fit']}, vary_t={res['vary_t']}'
        axs2[0].plot(time, res['fitted_iso'], label=label)
    axs2[0].set_title('Fitted Iso across Methods')
    axs2[0].set_ylabel('Fitted Iso (V)')
    axs2[0].legend()

    axs2[1].plot(time, true_sig, label='true_sig', color='black', linewidth=2)
    for res in results:
        label = f'dF/F (smooth={res['smooth_fit']}, vary_t={res['vary_t']})'
        axs2[1].plot(time, res['dff'], label=label, linestyle='--')
    axs2[1].set_title('dF/F across Methods vs. True Signal')
    axs2[1].set_ylabel('dF/F')
    axs2[1].set_xlabel('Time (s)')
    axs2[1].legend()

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

