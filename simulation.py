# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 15:28:33 2025

@author: good_
"""

import init
import hankslab_db.basicRLtasks_db as db

from hankslab_db import db_access
from pyutils import utils
import matplotlib.pyplot as plt

import numpy as np
from sys_neuro_tools import plot_utils, fp_utils

from scipy.optimize import curve_fit, minimize, OptimizeWarning
import warnings

import pickle

#%%load the param data calculated from actual recordings 
with open("param_range_comb.pkl", "rb") as f:
    param_range_comb = pickle.load(f)


#%% define recording length

fs = 200  # Sampling freq. 
duration = 500
total_t = np.linspace(0, duration, num=fs*duration, endpoint=False)   # for the entire duration 

#%% generate true signal and noise 
                                  
def make_signal(time, max_amp, n_terms = 2, amp_range = [0.1, 1], f_range = [0.1, 10]):
    
    signal = np.zeros_like(time) #Creates a NumPy array filled with zeros, having the same length as time
   
    for i in range(n_terms):  # accumulating sine wave at random amp & freq for the # of times specified by n_terms 
        random_amp = np.random.uniform(amp_range[0], amp_range[1])
        random_f = np.exp(np.random.uniform(np.log(f_range[0]), np.log(f_range[1])))# pick from loggged ver, then change it back using exp , to pick more from the lower freq range ( I think)
        signal = signal + random_amp * np.sin(2 * np.pi * random_f * time)  # this produces sin waves at the random freq 

    current_max_amp = np.max(np.abs(signal))  # Find the maximum absolute amplitude
    signal = signal * (max_amp / current_max_amp)  # scale to make the amp smaller than the max_amp_sig 
            
    return signal

    
#%% generate artifact

def make_art(time, max_amp, max_art_count=10, duration_range=[1,50], n_terms = 2, amp_range = [0.1, 1], f_range = [0.01, 5]):  
    
    # Randomly determine n (number of selected time points). 
    n = np.random.randint(1, max_art_count+1) # Return random integers from low (inclusive) to high (exclusive)
    
    art_sig_comb = np.zeros_like(time)
    
    selected_t_idx = np.random.choice((time * fs).astype(int), n, replace=False)   # picked timepoint (index) to add artifacts. replace=False argument ensures that each selected time point is unique.
    selected_m_idx = np.random.uniform(duration_range[0], duration_range[1] + 1, size=n)*fs  # Return random integers from low (inclusive) to high (exclusive).
    selected_m_idx = selected_m_idx.astype(int)

    for t_idx, m_idx in zip(selected_t_idx, selected_m_idx):   #zip create pairs of selected_t and m    
    
        # Ensure m_idx does not exceed the available range in art_sig_com. If exceeds, use the available length instead 
        valid_m_idx = min(m_idx, len(art_sig_comb) - t_idx)
    
        art_t = np.linspace(0, valid_m_idx*1/fs, num=valid_m_idx, endpoint=False)   # define artifact t. linspace produce 50 values by default evenlly spaced between the specified range 
        
        art_sig = make_signal(time=art_t, max_amp = max_amp, f_range = f_range, n_terms=n_terms, amp_range=amp_range)
        
        art_sig_comb[t_idx:t_idx+valid_m_idx] = art_sig_comb[t_idx:t_idx+valid_m_idx] + art_sig #Add the artifact signal from t_idx for the duration set by m_idx

    return art_sig_comb
    


#%% get params for baseline

def param_selector (form_name):   # add a way to be able to adjust the max and min value 
    
    'select params from the dataset (param_range_comb), which was produced from the recorded FP data'
    selected_params = {}
    for param in param_range_comb[form_name].keys(): 
        df = param_range_comb[form_name][param]
        limit_min = max(df['mean_minus_SD4'], df['min_value']) # min value is 0 when the param > 0. 
        limit_max = min(df['mean_plus_SD4'], df['max_value']) # this should always be mean_plus_SD4 if using Z = 10 as a cut off  

        #pick a param randomly from the range 
        
        if form_name == 'exp_linear' and param in [ 'param_1']:  
            selected_params[param] = np.exp(np.random.uniform(np.log(limit_min), np.log(limit_max))) # pick from loggged ver, then change it back using exp 
        
        if form_name == 'exp_linear' and param in [ 'param_3']:  
            selected_params[param] = np.random.uniform(0, limit_max)
        
        else: 
            selected_params[param] = np.random.uniform(limit_min, limit_max)
    
    return selected_params


def param_jitter (selected_params, SD_frac, form_name):    
    
    '''
    add jitter for polynomial parameters, 
    picking from the normal distribution made with SD for each param (eg, param_0)'
    '''
    
    jit_selected_params = {}
    for param in selected_params.keys():
        df = param_range_comb[form_name][param]
        jitter = (np.random.normal(loc=0, scale=df['SD']*SD_frac))   
        
        # set the same min and max as the input (lig) params     
        limit_min = max(df['mean_minus_SD4'], df['min_value']) 
        limit_max = min(df['mean_plus_SD4'], df['max_value'])
        
        if form_name == 'exp_linear' and param in [ 'param_3']:  
            jit_selected_params[param] = np.clip(selected_params[param] + jitter, 0, limit_max)

        else: 
            jit_selected_params[param] = np.clip(selected_params[param] + jitter, limit_min, limit_max)

    return jit_selected_params


#%% baseline formula 
def get_baseline_form (form_type):  
    
    match form_type:
        
        case 'double_exp_decay':
            baseline_form = lambda x, a, b, c, d, e: a*np.exp(-x/b) + c*np.exp(-x/(b*d)) + e
            #double exponential decay 
            
            # specifying the boundary for each parameter (only for the exponential ones) 
            #               a       b      c     d     e 
            bounds = ([0      ,      0, -np.inf, 0, -np.inf],
                      [ np.inf, np.inf,  np.inf, 1,  np.inf])
          
        case 'exp_linear':  #combination of the exponential decay term and a linear term 
            baseline_form = lambda x, a, b, c, d: a*np.exp(-x/b) + c*x +d 
                        
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
    
    'for time, the input is in second. converting to an index when apply' 
    
    while True:
        lig_params = param_selector (form_type)
        iso_params = param_jitter (lig_params, SD_frac, form_type)
    
        # params.values() seems to change the order of params. Thus, sorting based on the key - # may move this to combining function 
        lig_param_list = [lig_params[key] for key in sorted(lig_params.keys(), key=lambda x: int(x.split("_")[1]))]
        iso_param_list = [iso_params[key] for key in sorted(iso_params.keys(), key=lambda x: int(x.split("_")[1]))]
        
       	#get the one baseline_form to use
        baseline_form = get_baseline_form(form_type)
       
        #generate baselines 
        baseline_lig = baseline_form ['formula'] (np.arange(len(time)), *lig_param_list)
        baseline_iso = baseline_form ['formula'] (np.arange(len(time)), *iso_param_list)
    
        if np.all(baseline_lig > 0) and np.all(baseline_iso > 0):   # if either of the baselines are 0 or negative, generator another set of params and generate baselines 
            return {
                    "lig_params":lig_params,
                    "iso_params": iso_params,
                    "baseline_lig": baseline_lig,
                    "baseline_iso" :baseline_iso
                    }

#baselines  = generate_baseline ('exp_linear', 0.5, total_t)


#%% combine baseline, signal, artifact and noise

def simulate_signal (mul, time, baseline_lig, baseline_iso, true_sig, art, noise, simu_type):
    
    match simu_type:
        case 1:
            raw_lig = baseline_lig * ( (true_sig + art) + 1)  + noise  
            raw_iso = baseline_iso * ( (true_sig * mul + art) + 1)  + noise 
        
        case 2:
            raw_lig = baseline_lig * (true_sig + 1)*(art + 1)  + noise  
            raw_iso = baseline_iso * (true_sig*mul + 1)*(art + 1)  + noise  

    return raw_lig, raw_iso


#raw_lig, raw_iso = simulate_signal (0.1, total_t, baselines['baseline_lig'], baselines['baseline_iso'], true_sig, art, noise, simu_type=1)


#%% simulate signals 
def simulate_n_signals (n, time, 
                        max_amp_sig, f_range_sig ,  # for true signal 
                        max_amp_noise, f_range_noise,  # for noise 
                        max_amp_art, max_art_count, duration_range, f_range_art,  # for artifacts  
                        form_type, SD_frac_range, SD_frac,  # for generate_baseline
                        mul_range, mul, sim_type, # to combine with baseline 
                        DV ): 
    
    simulated_signals = {}
    
    match DV: 
        case 'SD_frac': 
            for SD_frac in np.arange(SD_frac_range[0], SD_frac_range[1], 0.1): 
                
                SD_frac_key = f"{SD_frac:.2f}"  # Convert to string with 2 decimal places
                simulated_signals[SD_frac_key] = []
        
                for _ in range(n):  
                    true_sig = make_signal(time, max_amp_sig, f_range = f_range_sig) 
                    noise  = make_signal(time, max_amp_noise, f_range = f_range_noise)  
                    art = make_art(time, max_amp_art, max_art_count, duration_range, f_range= f_range_art) 
                    
                    baselines  = generate_baseline (form_type, SD_frac, time)
            
                    raw_lig, raw_iso = simulate_signal (mul, time, baselines['baseline_lig'], baselines['baseline_iso'], true_sig, art, noise, sim_type)
                    
                    simulated_signals[SD_frac_key].append({'true_sig':true_sig, "raw_lig": raw_lig, "raw_iso": raw_iso, "baseline_lig":baselines['baseline_lig'], "baseline_iso":baselines['baseline_iso']})


        case 'mul': 
            for mul in np.arange(mul_range[0], mul_range[1], 0.1): 
                 
                mul_key = f"{mul:.2f}"  # Convert to string with 2 decimal places
                simulated_signals[mul_key] = []
        
                for _ in range(n):  
                    true_sig = make_signal(time, max_amp_sig, f_range = f_range_sig) 
                    noise  = make_signal(time, max_amp_noise, f_range = f_range_noise)  
                    art = make_art(time, max_amp_art, max_art_count, duration_range, f_range= f_range_art) 
                    
                    baselines  = generate_baseline (form_type, SD_frac, time)
            
                    raw_lig, raw_iso = simulate_signal (mul, time, baselines['baseline_lig'], baselines['baseline_iso'], true_sig, art, noise, sim_type)
                    
                    simulated_signals[mul_key].append({'true_sig':true_sig, "raw_lig": raw_lig, "raw_iso": raw_iso, "baseline_lig":baselines['baseline_lig'], "baseline_iso":baselines['baseline_iso']})


    return simulated_signals
    

simulated_signals = simulate_n_signals(n = 10, time = total_t, 
                        max_amp_sig = 0.3, f_range_sig = [0.5, 10] ,  # for true signal 
                        max_amp_noise = 0.1, f_range_noise = [1, 10],  # for noise 
                        max_amp_art = 0.1, max_art_count = 5, duration_range= [1,50], f_range_art = [0.01, 5],  # for artifacts. When setting the max amp, make sure : s(t) + a(t) + 1 > 0 
                        form_type = 'exp_linear', 
                        SD_frac_range = [0, 0.8], SD_frac = 0,
                        mul_range = [0, 0.8], mul = 0, 
                        sim_type = 2, # to combine with baseline 
                        DV = 'SD_frac')  # pick the dependnt variable to vary. 



#%%tourbleshoot for specific signals 
# test_list = simulated_signals['0.30']
# test_signals= test_list[192]



#%% process_signals 

def process_signals (raw_lig, raw_iso, baseline_iso, time, fs, smooth_fit, vary_t, filt_denom = True ): 
    
    # denoised_lig = fp_utils.filter_signal(raw_lig, 10, fs)
    # denoised_iso= fp_utils.filter_signal(raw_iso, 10, fs)  
    
    denoised_lig = raw_lig
    denoised_iso = raw_iso
    
    if smooth_fit: 
        smooth_lig = fp_utils.filter_signal(raw_lig, 0.1, fs)   
        smooth_iso = fp_utils.filter_signal(raw_iso, 0.1, fs) 
    
        if vary_t:
            fitted_smooth_iso, smooth_fit_info = fp_utils.fit_signal(smooth_iso, smooth_lig, time, vary_t=True)
            s_to_fit = np.vstack((denoised_iso[None], time[None])) 
            baseline_to_fit = np.vstack((baseline_iso[None], time[None])) 

        else: 
            fitted_smooth_iso, smooth_fit_info = fp_utils.fit_signal(smooth_iso, smooth_lig, time, vary_t=False)
            s_to_fit = denoised_iso
            baseline_to_fit = baseline_iso
            
        #for smooth fit, get new fitted iso , using the smooth_fit_info 
        fitted_iso = smooth_fit_info['formula'](s_to_fit, *smooth_fit_info['params'])
                        
        #also regress baseline to see how close baseline_iso goes to baseline_lig 
        fitted_baseline_iso = smooth_fit_info['formula'](baseline_to_fit, *smooth_fit_info['params'])
        
        
        if filt_denom:
            filt_fitted_iso = fp_utils.filter_signal(fitted_iso, 0.1, fs)  
            dff_iso = ((denoised_lig - fitted_iso)/filt_fitted_iso)
        else:
            dff_iso = ((denoised_lig - fitted_iso)/fitted_iso)
            
        #cut off values below 0.1 
        idx = np.where(fitted_iso < 0.1)[0]
        cutoff_idx = idx[0] if len(idx) > 0 else None
        
        raw_lig = raw_lig[:cutoff_idx]
        raw_iso = raw_iso[:cutoff_idx]
        denoised_lig = denoised_lig[:cutoff_idx]
        smooth_lig = smooth_lig[:cutoff_idx]
        smooth_iso = smooth_iso[:cutoff_idx]
        fitted_smooth_iso = fitted_smooth_iso[:cutoff_idx]
        fitted_iso = fitted_iso[:cutoff_idx]
        fitted_baseline_iso = fitted_baseline_iso[:cutoff_idx]
        dff_iso = dff_iso[:cutoff_idx]

        
        return {
                # 'raw_lig': raw_lig,  
                # 'raw_iso': raw_iso,
                # 'denoised_lig':denoised_lig,
                # 'smooth_lig':smooth_lig,
                # 'smooth_iso':smooth_iso,
                # 'fitted_smooth_iso':fitted_smooth_iso,
                # 'fitted_iso': fitted_iso,
                # 'fitted_baseline_iso':fitted_baseline_iso,
                'dff_iso': dff_iso
                 }


    else:
        if vary_t:
            fitted_iso, fit_info   = fp_utils.fit_signal(denoised_iso, denoised_lig, time, vary_t=True)
            baseline_to_fit = np.vstack((baseline_iso[None], time[None])) 

        else: 
            fitted_iso, fit_info   = fp_utils.fit_signal(denoised_iso, denoised_lig, time, vary_t=False)  
            baseline_to_fit = baseline_iso       
            

        #also regress baseline to see how close baseline_iso goes to baseline_lig 
        fitted_baseline_iso = fit_info['formula'](baseline_to_fit, *fit_info['params'])       
        
        if filt_denom:
            filt_fitted_iso = fp_utils.filter_signal(fitted_iso, 0.1, fs)  
            dff_iso = ((denoised_lig - fitted_iso)/filt_fitted_iso)
        else:
            dff_iso = ((denoised_lig - fitted_iso)/fitted_iso)
        
        
        #cut off values below 0.1 
        idx = np.where(fitted_iso < 0.1)[0]
        cutoff_idx = idx[0] if len(idx) > 0 else None
        
        raw_lig = raw_lig[:cutoff_idx]
        raw_iso = raw_iso[:cutoff_idx]
        denoised_lig = denoised_lig[:cutoff_idx]
        fitted_iso = fitted_iso[:cutoff_idx]
        fitted_baseline_iso = fitted_baseline_iso[:cutoff_idx]
        dff_iso = dff_iso[:cutoff_idx]

        return {
                # 'raw_lig': raw_lig,  
                # 'raw_iso': raw_iso,
                # 'denoised_lig':denoised_lig,
                # 'fitted_iso': fitted_iso,
                # 'fitted_baseline_iso':fitted_baseline_iso,
                'dff_iso': dff_iso
                 }


#%% process signals 
processed_signals = {}

for DV, signal_list in simulated_signals.items():  # Iterate over SD_frac keys
    processed_signals[DV] = {}  # Create nested dictionary for each SD_frac

    for smooth_fit in [True, False]:
        for vary_t in [True, False]:
            key = f"smooth_fit_{smooth_fit}_vary_t_{vary_t}"
            processed_signals[DV][key] = []  # Store results in list

            for entry in signal_list:  # Iterate over stored signal lists
                raw_lig = entry["raw_lig"]
                raw_iso = entry["raw_iso"]
                baseline_iso = entry["baseline_iso"]

                # Process signals and store results
                processed_output = process_signals(raw_lig, raw_iso, baseline_iso, total_t, fs,  
                                                   smooth_fit=smooth_fit, vary_t=vary_t)
                processed_signals[DV][key].append(processed_output)


#%% for troubleshooting. process potentially problematic simulated signals. 
# processed_signals = {}
# for smooth_fit in [True, False]:
#     for vary_t in [True, False]:
#         key = f"smooth_fit_{smooth_fit}_vary_t_{vary_t}"
#         processed_signals[key] = process_signals(test_signals['raw_lig'], test_signals['raw_iso'], test_signals['baseline_iso'], total_t, fs, 
#                                             smooth_fit=smooth_fit, vary_t=vary_t)




#%% calculate explained variance 

def ev(true_sig, dff_iso):
    
    # if the dff_iso is cut off, use the same length for true sig 
    if len(true_sig) != len(dff_iso):
        true_sig = true_sig[:len(dff_iso)] 
    
    numerator = np.var(true_sig - dff_iso)
    denominator = np.var(true_sig)
    return 1 - (numerator / denominator)


ev_results = {}

for DV, method in processed_signals.items():  # Iterate over SD_frac keys
    ev_results[DV] = {}  # Create nested dictionary for each SD_frac
    
    # Extract the correct true_sig for each signal in simulated_signals
    signal_list = simulated_signals[DV]  # List of signals for this SD_frac

    for method_key, processed_list in method.items():  # Iterate over different methods
        ev_results[DV][method_key] = []  # Store results

        for signal_entry, processed_entry in zip(signal_list, processed_list):  
            true_sig = signal_entry["true_sig"]  # Extract true signal from simulated_signals
            dff_iso = processed_entry["dff_iso"]  # Extract dff_iso from processed_signals

            # Compute evaluation metric
            ev_output = ev(true_sig, dff_iso)  

            # Store result
            ev_results[DV][method_key].append(ev_output)

#%% For troubleshooting. Testing ev  
# ev_results = {}

# for key, signals in processed_signals.items():
#     ev_results[key] = ev(test_signals['true_sig'], signals["dff_iso"])



#%% exclude outliners as needed 

def outliers (data, threshold=2):
    data = np.array(data)
    mean, std = np.mean(data), np.std(data)
    z_scores = (data - mean) / std
    return data[np.abs(z_scores) < threshold] 


#%% plot ev  - line plot 

def plot_ev_results(ev_results):
    plt.figure(figsize=(10, 10))
    
    DVs = sorted([float(k) for k in ev_results.keys()])

    # Extract methods
    methods = list(next(iter(ev_results.values())).keys())
    
    
    for method in methods:
        y_values = []
        y_errors = []
        
        for DV in DVs:
            data = ev_results[f"{DV:.2f}"][method]
            clean_data = outliers(data)  # Remove outliers
            #clean_data = data   # use this, when not excluding outliners
            
            y_values.append(np.mean(clean_data))
            y_errors.append(np.std(clean_data) / np.sqrt(len(clean_data)))

        plt.errorbar(DVs, y_values, yerr=y_errors, marker='o', capsize=5, label=method)
        
    plt.xlabel("SD fraction")
    plt.ylabel("EV")
    plt.title("EV Results - n = 500 per x , Outlier excluded at z2 ")
    plt.legend()
    #plt.ylim(0.3, 1.1)
    plt.grid()
    plt.show()


plot_ev_results(ev_results)

#%% For troubleshooting / visualization, plot some of the signals as needed 

def plot_signals(processed_signals, true_sig, baseline_lig, baseline_iso, t, title, ev):
    fig, axs = plt.subplots(4,1, figsize=(9, 10), sharex=True, constrained_layout=True)
    plt.suptitle(title)
    
    # if the dff_iso is cut off, use the same length for time, true signal 
    if len(processed_signals['fitted_iso']) != len(t):
        t = t[:len(processed_signals['fitted_iso'])] 
        true_sig = true_sig[:len(processed_signals['fitted_iso'])] 
        baseline_lig = baseline_lig[:len(processed_signals['fitted_iso'])] 
        baseline_iso = baseline_iso[:len(processed_signals['fitted_iso'])] 
        
    
   
    ax = axs[0]  
    ax.plot(t, processed_signals['raw_lig'], label='Raw Lig', alpha=0.5)
    ax.plot(t, processed_signals['raw_iso'], label='Raw iso', alpha=0.5)
    ax.plot(t, baseline_lig, label='Baseline_lig', alpha=0.5)
    ax.plot(t, baseline_iso, label='Baseline_iso', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescent Signal (V)')
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    ax.set_title('Simulated ligand and iso signals')
    ax.legend(loc='upper right')
    
    if 'smooth_lig' in processed_signals:
        ax = axs[1] 
        ax.plot(t, processed_signals['smooth_lig'], label='smooth_lig', alpha=0.5)
        ax.plot(t, processed_signals['smooth_iso'], label='smooth_iso', alpha=0.5)
        ax.plot(t, processed_signals['fitted_smooth_iso'], label='fitted_smooth_iso', alpha=0.5)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Fluorescent Signal (V)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.set_title('Smooth fit')
        ax.legend(loc='upper right')
        

    ax = axs[2] 
    ax.plot(t, processed_signals['denoised_lig'], label='denoised_lig', alpha=0.5)
    ax.plot(t, processed_signals['fitted_iso'], label='fitted_iso', alpha=0.5)
    ax.plot(t, baseline_lig, label='Baseline_lig', alpha=0.5)
    ax.plot(t, processed_signals['fitted_baseline_iso'], label='fitted_baseline_iso', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fluorescent Signal (V)')
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    ax.set_title('Denoised lig and fitted iso')
    ax.legend(loc='upper right')
    
    ax = axs[3] 
    ax.plot(t, true_sig, label='true signal', alpha=0.5)
    ax.plot(t, processed_signals['dff_iso'], label='dF/F', alpha=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('dF/F or Fluorescent Signal (V)')
    ax.xaxis.set_tick_params(which='both', labelbottom=True)
    ax.set_title(f'dF/F vs. true signal: Explained variance = {ev}')
    ax.legend(loc='upper right')
    #ax.set_xlim(290, 300)

    
    return fig


for key, signals in processed_signals.items():
    plot_signals(signals, test_signals['true_sig'], test_signals['baseline_lig'], test_signals['baseline_iso'], total_t, key, ev_results[key])
    #plt.savefig(f"Type1_{key}.png", dpi=300)
    plt.show()



# Save fig 

# for key, signals in processed_signals.items():
#     plot_signals(signals, baselines, true_sig, total_t, key, ev_results[key])
#     plt.savefig(f"Type1_{key}_290-300.png", dpi=300)
#     plt.show()



















