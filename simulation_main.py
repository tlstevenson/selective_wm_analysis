# -*- coding: utf-8 -*-
'''
Created on Thu Jun 26 12:23:54 2025

@author: atruo - Alex Truong

relevant data/graphs stored here: https://docs.google.com/document/d/1VdKfUvVSoFxo4LlaEtMhj-IyLhX59fA6npD_ieepOV0/edit?usp=sharing
'''

import matplotlib.pyplot as plt
import numpy as np
import simulation_lib as sl
import random

#%% define recording length

fs = 30  # sampling freq. 
duration = 500
total_t = np.linspace(0, duration, num=fs*duration, endpoint=False)   # for the entire duration 


#%% simulate signals

n = 200
time = total_t

# param_name = 'alpha'
# param_range = [0, 1]
# param_step = 0.1

param_name = 'SD_frac'
param_range = [0, 0.5]
param_step = 0.05

# param_name = 'SNR'
# param_range = [0.5, 5]
# param_step = 0.5

# param_name = 'SAR'
# param_range = [0.5, 5]
# param_step = 0.5

f_range_sig = [0.1, 10]
f_range_noise = [0.5, 10]
max_art_count = 5
art_duration_range = [1, 50]
f_range_art = [0.05, 10]
form_type = 'exp_linear'
sim_type = 2
SD_frac_default = 0.05
alpha_default = 0.05
SNR_default = 10
SAR_default = 1
scale = 0.1

simulated_signals = sl.simulate_n_signals(n, time,
    param_name, param_range, param_step,
    f_range_sig,
    f_range_noise,
    max_art_count, art_duration_range, f_range_art,
    form_type, sim_type,
    SD_frac_default,
    alpha_default,
    SNR_default,
    SAR_default, 
    scale
)


#%%troubleshoot for specific signals 
# test_list = simulated_signals['0.30']
# test_signals= test_list[192]


#%% for troubleshooting. process potentially problematic simulated signals. 
# processed_signals = {}
# for smooth_fit in [True, False]:
#     for vary_t in [True, False]:
#         key = f'smooth_fit_{smooth_fit}_vary_t_{vary_t}'
#         processed_signals[key] = process_signals(test_signals['raw_lig'], test_signals['raw_iso'], test_signals['baseline_iso'], total_t, fs, 
#                                             smooth_fit=smooth_fit, vary_t=vary_t)


#%% process signals 
processed_signals = {}
smooth_lpf = 0.1

# Count total steps for progress tracking
n_signals_per_DV = len(next(iter(simulated_signals.values())))
n_DVs = len(simulated_signals)
n_methods = 4  # smooth_fit=True/False × vary_t=True/False
total_steps = n_DVs * n_methods * n_signals_per_DV

step = 0  # to track progress

for DV, signal_list in simulated_signals.items():  # iterate over SD_frac keys
    processed_signals[DV] = {}  # create nested dictionary for each SD_frac

    for smooth_fit in [True, False]:
        for vary_t in [True, False]:
            key = f'smooth_fit_{smooth_fit}_vary_t_{vary_t}'
            processed_signals[DV][key] = []  # store results in list

            for entry in signal_list:  # iterate over stored signal lists
                raw_lig = entry['raw_lig']
                raw_iso = entry['raw_iso']
                baseline_iso = entry['baseline_iso']

                # process signals and store results
                processed_output = sl.process_signals(
                    raw_lig, raw_iso, baseline_iso, total_t, fs,
                    smooth_fit=smooth_fit, vary_t=vary_t, smooth_lpf=smooth_lpf
                )
                processed_signals[DV][key].append(processed_output)

                # update progress
                step += 1
                
                print(f'Processing signal {step}/{total_steps}...', end='\r')


#%% report average clamping after processing signals

if sl.clamp_total_calls > 0:
    avg_clamped = sl.clamp_total_points / sl.clamp_total_calls
    print(f'[process_signals] Total number of clamped denominator points: {sl.clamp_total_points}')
    print(f'[process_signals] Average number of clamped denominator points per call: {avg_clamped:.2f}')
else:
    print('[process_signals] Clamp function was never called.')


#%% calculate explained variance ev_results

ev_results = {}

for DV, method in processed_signals.items():  # iterate over SD_frac keys
    ev_results[DV] = {}  # create nested dictionary for each SD_frac
    
    # extract the correct true_sig for each signal in simulated_signals
    signal_list = simulated_signals[DV]  # list of signals for this SD_frac

    for method_key, processed_list in method.items():  # iterate over different methods
        ev_results[DV][method_key] = []  # store results

        for signal_entry, processed_entry in zip(signal_list, processed_list):
            filt_t = processed_entry['filt_t']
            scaled_true_sig = signal_entry['scaled_true_sig']  # extract true signal from simulated_signals
            dff_iso = processed_entry['dff']  # extract dff_iso from processed_signals

            # compute evaluation metric
            ev_output = sl.ev(scaled_true_sig[filt_t], dff_iso[filt_t])  

            # store result
            ev_results[DV][method_key].append(ev_output)


#%% for troubleshooting. Testing ev  
# ev_results = {}

# for key, signals in processed_signals.items():
#     ev_results[key] = ev(test_signals['true_sig'], signals['dff_iso'])


#%% debugging, testing if jitter = 0 
#_ = sl.generate_baseline('exp_linear', 0.5, total_t)


#%% debugging for signal = 0


 # get one existing simulated signal from the first DV group (e.g., SNR = 0.1)
#DV_key = list(simulated_signals.keys())[0]  # e.g., '0.10'
#example_idx = 0  # index of signal to plot
#signal_data = simulated_signals[DV_key][example_idx]

 # extract components to plot
#scaled_true_sig = signal_data['scaled_true_sig']
#baseline_lig = signal_data['baseline_lig']
#baseline_iso = signal_data['baseline_iso']

 
#plt.figure(figsize=(12, 5))
#plt.plot(total_t, scaled_true_sig, label='Scaled True Signal', linewidth=2)
#plt.plot(total_t, baseline_lig, label='Baseline LIG', alpha=0.8)
#plt.plot(total_t, baseline_iso, label='Baseline ISO', alpha=0.8)
#plt.xlabel('Time (s)')
#plt.ylabel('Signal Amplitude')
#plt.title(f'Simulated Signal Components (DV = {DV_key}, Index = {example_idx})')
#plt.legend()
#plt.grid(True)
#plt.tight_layout()
#plt.show()


#%% plot statistical test results and signals

method_labels = {'smooth_fit_False_vary_t_False': 'Classic',
                 'smooth_fit_True_vary_t_False': 'Smoothed Fit',
                 'smooth_fit_False_vary_t_True': 'Rotated Fit',
                 'smooth_fit_True_vary_t_True': 'Smoothed & Rotated Fit'}

sl.plot_ev_results(ev_results, param_name, exclude_outliers=False, method_labels=method_labels,
                   alpha_default=alpha_default, SD_frac_default=SD_frac_default, SNR_default=SNR_default, SAR_default=SAR_default)  # explained variance
sl.plot_ev_results(ev_results, param_name, exclude_outliers=True, method_labels=method_labels,
                   alpha_default=alpha_default, SD_frac_default=SD_frac_default, SNR_default=SNR_default, SAR_default=SAR_default)

# plot signal that gave the abs minimum EV value across the entire DV range

min_ev = float('inf')
min_idx = None
min_method = None
min_DV = None

# search for the lowest EV value across all DV groups
for DV, evs_for_dv in ev_results.items():
    for method_key, ev_list in evs_for_dv.items():
        for idx, ev_val in enumerate(ev_list):
            if ev_val < min_ev:
                min_ev = ev_val
                min_idx = idx
                min_method = method_key
                min_DV = DV
                
correct_ev = ev_results[min_DV][min_method][min_idx]
                                
# grab the corresponding signal
sim_list = simulated_signals[min_DV]
i = min_idx

print(f'Lowest EV = {correct_ev:.4f} from DV group: {min_DV:.4f}, method: {min_method}, index: {i}')

# print baseline parameters for lowest EV signal
lig_params = sim_list[i]['lig_params']
iso_params = sim_list[i]['iso_params']

print('\nBaseline Parameters for Lowest EV Case:')
print('Ligand Parameters:')
for k, v in lig_params.items():
    print(f'  {k}: {v:.4g}')
print('Isosbestic Parameters:')
for k, v in iso_params.items():
    print(f'  {k}: {v:.4g}')


# plot using the lowest-EV signal
sl.plot_comparative_figures(
    raw_lig=sim_list[i]['raw_lig'],
    raw_iso=sim_list[i]['raw_iso'],
    baseline_iso=sim_list[i]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[i]['scaled_true_sig'],
    fs=fs,
    ev=min_ev,
    dv=min_DV,
    param_name=param_name
)

 
#%% plot random signal that is not the lowest EV

ev_list = ev_results[min_DV][min_method]
non_min_indices = [j for j in range(len(ev_list)) if j != min_idx]

other_idx = random.choice(non_min_indices)
other_ev = ev_results[min_DV][min_method][other_idx]

print(f'Other EV = {other_ev:.4f} from DV group: {min_DV:.4f}, method: {min_method}, index: {other_idx}')

sl.plot_comparative_figures(
    raw_lig=sim_list[other_idx]['raw_lig'],
    raw_iso=sim_list[other_idx]['raw_iso'],
    baseline_iso=sim_list[other_idx]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[other_idx]['scaled_true_sig'],
    fs=fs,
    ev=other_ev,
    dv=min_DV,
    param_name=param_name
)   


#%% plot true sig, raw iso, raw lig, and fitted iso for worst EV signal

lowest_processed = processed_signals[min_DV][min_method][i].copy()
lowest_processed['raw_lig'] = simulated_signals[min_DV][i]['raw_lig']
lowest_processed['raw_iso'] = simulated_signals[min_DV][i]['raw_iso']

true_sig = simulated_signals[min_DV][i]['scaled_true_sig']

fig = sl.plot_signals(
    processed_signals=lowest_processed,
    true_sig=true_sig,
    t=total_t,
    ev=min_ev,
    fs=fs,
    title=f'Lowest EV: {float(min_DV):.2f} | {min_method} | idx={i}'
)

fig, ax = plt.subplots(1, 1, figsize=(5, 3), layout='constrained')
ax.plot(total_t, lowest_processed['raw_iso'], label='Isosbestic', alpha=0.6)
ax.plot(total_t, lowest_processed['raw_lig'], label='Ligand', alpha=0.6)
ax.set_xlabel('Time')
ax.legend()

#%% plot signal, artifact and noise from the lowest EV signal

# get signal from simulated_signals with the lowest EV
sim_list = simulated_signals[min_DV]
i = min_idx
lowest_ev_signal = sim_list[i]

# extract artifact and noise
artifact = lowest_ev_signal['artifact']
noise = lowest_ev_signal['noise']
true_sig = simulated_signals[min_DV][i]['scaled_true_sig']

# plot them separately
fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharey=True, layout='constrained')
axs[0].plot(total_t, true_sig, color='blue')
axs[0].set_title('Signal')
axs[0].set_xlabel('Time')
axs[1].plot(total_t, artifact, color='red')
axs[1].set_title('Artifact')
axs[1].set_xlabel('Time')
axs[2].plot(total_t, noise, color='orange')
axs[2].set_title('Noise')
axs[2].set_xlabel('Time')

fig.suptitle('Signal, Artifact and Noise (Lowest EV Signal)')



#%% regression coefficients for all methods at lowest EV index
print(f'\nRegression Coefficients for All Methods at DV = {min_DV}, index = {min_idx}:')

# iterate over all 4 combinations of smooth_fit and vary_t
for smooth_fit in [True, False]:
    for vary_t in [True, False]:
        method_key = f'smooth_fit_{smooth_fit}_vary_t_{vary_t}'
        
        try:
            entry = processed_signals[min_DV][method_key][min_idx]
            fit_params = entry.get('fit_params', None)
        except (IndexError, KeyError):
            fit_params = None

        print(f'\nMethod: {method_key}')
        if fit_params is not None:
            for idx, coeff in enumerate(fit_params):
                print(f'  Param {idx + 1}: {coeff:.6g}')
        else:
            print('  No fit_params found.')
            
            