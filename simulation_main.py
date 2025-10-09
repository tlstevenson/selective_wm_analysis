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

import pandas as pd
import os
from tqdm import tqdm

#%% define recording length

fs = 30  # sampling freq. 
duration = 500
total_t = np.linspace(0, duration, num=fs*duration, endpoint=False)   # for the entire duration 

#%% define save file path

# table_file = "signal_results_table.pkl"

#%% simulate signals

n = 50 # was 200
time = total_t

param_name = 'alpha'
param_range = [0, 1]
param_step = 0.1

# param_name = 'SD_frac'
# param_range = [0, 0.5]
# param_step = 0.05

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
scale = 0.1 # was 0.1
lpf_default = 0.1
lowpass_1mHz_cutoff = 0.001

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

# save the simulated signals after generation
# sl.save_simulated_signals(savefile, param_name, simulated_signals)
# print(f"Simulated signals saved to {savefile}")

#%% troubleshoot for specific signals 
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

# Count total steps for progress tracking
n_signals_per_DV = len(next(iter(simulated_signals.values())))
n_DVs = len(simulated_signals)
n_methods = 4 + 1 + 1  # 4 OLS methods, 1 LPF-only, 1 IRLS
total_steps = n_DVs * n_methods * n_signals_per_DV

for DV, signal_list in tqdm(simulated_signals.items(), desc="DV groups"):
    processed_signals[DV] = {
        'smooth_fit_True_vary_t_True': [],
        'smooth_fit_True_vary_t_False': [],
        'smooth_fit_False_vary_t_True': [],
        'smooth_fit_False_vary_t_False': [],
        'lowpass_1mHz': [],
        'irls_regression': []
    }

    for idx, entry in enumerate(tqdm(signal_list, desc="signals", leave=False)):
        raw_lig = entry['raw_lig']
        raw_iso = entry['raw_iso']
        baseline_iso = entry['baseline_iso']

        # Single call returns dict with OLS, LPF_only, IRLS
        proc_all = sl.process_signals(raw_lig, raw_iso, baseline_iso, total_t, fs, lpf=lpf_default)

        # Map OLS outputs
        ols_map = {
            'smooth_fit_True_vary_t_True': proc_all['OLS'].get('smooth1_varyt1'),
            'smooth_fit_True_vary_t_False': proc_all['OLS'].get('smooth1_varyt0'),
            'smooth_fit_False_vary_t_True': proc_all['OLS'].get('smooth0_varyt1'),
            'smooth_fit_False_vary_t_False': proc_all['OLS'].get('smooth0_varyt0'),
        }

        # LPF-only (same result for both smooth flags)
        lpf_only = proc_all.get('LPF_only')

        # IRLS
        irls_out = proc_all.get('IRLS')

        # --- Validation helper ---
        def validate_and_append(key, out):
            if out is None:
                print(f"⚠️ Warning: DV={DV}, method={key}, index={idx} returned None")
                out = {}
            if 'dff' not in out:
                out['dff'] = np.zeros_like(total_t, dtype=float)
                print(f"⚠️ Missing 'dff': DV={DV}, method={key}, index={idx}")
            if 'filt_t' not in out:
                out['filt_t'] = np.array([], dtype=int)
                print(f"⚠️ Missing 'filt_t': DV={DV}, method={key}, index={idx}")
            processed_signals[DV][key].append(out)

        # Append all methods
        for k, v in ols_map.items():
            validate_and_append(k, v)

        validate_and_append('lowpass_1mHz', lpf_only)
        validate_and_append('irls_regression', irls_out)


#%% report average clamping after processing signals

if sl.clamp_total_calls > 0:
    avg_clamped = sl.clamp_total_points / sl.clamp_total_calls
    print(f'[process_signals] Total number of clamped denominator points: {sl.clamp_total_points}')
    print(f'[process_signals] Average number of clamped denominator points per call: {avg_clamped:.2f}')
else:
    print('[process_signals] Clamp function was never called.')


#%% calculate explained variance ev_results

ev_results = {}

for DV, method_dict in processed_signals.items():  # iterate over DV keys
    ev_results[DV] = {}  # create nested dictionary for each DV
    
    signal_list = simulated_signals[DV]  # list of signals for this DV

    for method_key, processed_list in method_dict.items():  # iterate over methods
        ev_results[DV][method_key] = []  # store results

        for idx, (signal_entry, processed_entry) in enumerate(zip(signal_list, processed_list)):
            # ensure filt_t exists and is non-empty; otherwise mark EV as NaN
            filt_t = processed_entry.get('filt_t', None)
            if filt_t is None or len(filt_t) == 0:
                ev_output = np.nan
                print(f"⚠️ Empty filt_t: DV={DV}, method={method_key}, index={idx}")
            else:
                # extract true signal and processed dff safely
                scaled_true_sig = signal_entry.get('scaled_true_sig', np.zeros_like(total_t, dtype=float))
                dff_iso = processed_entry.get('dff', np.zeros_like(total_t, dtype=float))

                try:
                    ev_output = sl.ev(scaled_true_sig[filt_t], dff_iso[filt_t])
                except Exception as e:
                    ev_output = np.nan
                    print(f"⚠️ EV computation failed: DV={DV}, method={method_key}, index={idx}, error={e}")

                if np.isnan(ev_output):
                    print(f"⚠️ NaN EV: DV={DV}, method={method_key}, index={idx}")

            # store result
            ev_results[DV][method_key].append(ev_output)


#%% saving signals

# =============================================================================
# # Convert to DataFrame
# df_new = sl.signals_to_dataframe(simulated_signals, ev_results, param_name)
# 
# # Load existing table (if any)
# df_existing = sl.load_or_create_dataframe(table_file)
# 
# # Append and drop duplicates (optional, based on DV+index+method)
# df_combined = pd.concat([df_existing, df_new], ignore_index=True)
# df_combined.drop_duplicates(subset=['DV', 'index', 'method'], inplace=True)
# 
# # Save
# sl.save_dataframe(df_combined, table_file)
# print(f"✅ Appended and saved to: {table_file}")
# =============================================================================


#%% retrieving signals
# =============================================================================
# 
# df_filtered, signals = sl.retrieve_signals(
#     table_file,
#     DV=0.3,
#     method="smooth_fit_True_vary_t_True",
#     ev_thresh=0.8
# )
# 
# =============================================================================


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

method_labels = {
    'smooth_fit_False_vary_t_False': 'Classic',
    'smooth_fit_True_vary_t_False': 'Smoothed Fit',
    'smooth_fit_False_vary_t_True': 'Rotated Fit',
    'smooth_fit_True_vary_t_True': 'Smoothed & Rotated Fit',
    'lowpass_1mHz': 'LPF',
    'irls_regression': 'IRLS Regression'
}

# Plot EV results
sl.plot_ev_results(ev_results, param_name, exclude_outliers=False, method_labels=method_labels,
                   alpha_default=alpha_default, SD_frac_default=SD_frac_default,
                   SNR_default=SNR_default, SAR_default=SAR_default)
sl.plot_ev_results(ev_results, param_name, exclude_outliers=True, method_labels=method_labels,
                   alpha_default=alpha_default, SD_frac_default=SD_frac_default,
                   SNR_default=SNR_default, SAR_default=SAR_default)


# Find signal with the absolute minimum EV across all DVs and methods
min_ev = float('inf')
min_idx = None
min_method = None
min_DV = None

for DV, evs_for_dv in ev_results.items():
    for method_key, ev_list in evs_for_dv.items():
        for idx, ev_val in enumerate(ev_list):
            if ev_val < min_ev:
                min_ev = ev_val
                min_idx = idx
                min_method = method_key
                min_DV = DV

correct_ev = ev_results[min_DV][min_method][min_idx]

# Grab the corresponding signal
sim_list = simulated_signals[min_DV]
i = min_idx

print(f'Lowest EV = {correct_ev:.4f} from DV group: {min_DV:.4f}, method: {min_method}, index: {i}')

# Print baseline parameters for lowest EV signal
lig_params = sim_list[i]['lig_params']
iso_params = sim_list[i]['iso_params']

print('\nBaseline Parameters for Lowest EV Case:')
print('Ligand Parameters:')
for k, v in lig_params.items():
    print(f'  {k}: {v:.4g}')
print('Isosbestic Parameters:')
for k, v in iso_params.items():
    print(f'  {k}: {v:.4g}')



print("DEBUG: raw_lig =", sim_list[i].get('raw_lig') is None)
print("DEBUG: raw_iso =", sim_list[i].get('raw_iso') is None)
print("DEBUG: baseline_iso =", sim_list[i].get('baseline_iso') is None)
print("DEBUG: true_sig =", sim_list[i].get('scaled_true_sig') is None)
print("DEBUG: time length =", len(total_t))


# Plot using the lowest-EV signal
sl.plot_comparative_figures(
    raw_lig=sim_list[i]['raw_lig'],
    raw_iso=sim_list[i]['raw_iso'],
    baseline_iso=sim_list[i]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[i]['scaled_true_sig'],
    fs=fs,
    ev=min_ev,
    dv=min_DV,
    param_name=param_name,
    extra_title=None
)


#%% debugging for missing EV values for certain methods at different DV values

for dv, methods_dict in ev_results.items():
    for method_key, ev_list in methods_dict.items():
        for i, val in enumerate(ev_list):
            if val is None or (isinstance(val, float) and np.isnan(val)):
                print(f"⚠️ EV missing: method={method_key}, DV={dv}, i={i}")

                # pull from processed_signals
                dff = processed_signals[dv][method_key][i]['dff']
                filt_t = processed_signals[dv][method_key][i]['filt_t']

                # pull the matching true signal
                true_sig = simulated_signals[dv][i]['scaled_true_sig']

                print(f"    len dff={len(dff)}, NaN count={np.isnan(dff).sum()}, "
                      f"len filt_t={len(filt_t)}, filt_t sum={filt_t.sum()}, "
                      f"len true_sig={len(true_sig)}")
#%% debugging               
                
# for dv, methods_dict in ev_results.items():
#     for method_key, ev_list in methods_dict.items():
#         arr = np.array(ev_list, dtype=float)
        
#         # Check for NaNs and their indices
#         nan_indices = [i for i, v in enumerate(arr) if np.isnan(v)]
#         if nan_indices:
#             print(f"NaNs inside EV array: DV={dv}, method={method_key}, indices={nan_indices}")
        
#         # Count NaNs and infs
#         nan_count = np.isnan(arr).sum()
#         inf_count = np.isinf(arr).sum()
        
#         # Get min/max safely (ignoring NaNs and infs)
#         finite_arr = arr[np.isfinite(arr)]
#         min_val = min(finite_arr) if finite_arr.size > 0 else np.nan
#         max_val = max(finite_arr) if finite_arr.size > 0 else np.nan
        
#         # Print combined info
#         print(f"DV={dv} | method={method_key} | len={len(arr)} | "
#               f"NaNs={nan_count}, infs={inf_count} | min={min_val:.4f}, max={max_val:.4f}")


#%% plot lowest EV signal: fitted iso and dF/F comparisons across all methods

# grab the lowest EV signal across all DVs and methods
sim_list = simulated_signals[min_DV]
i = min_idx
lowest_ev_signal = sim_list[i]

# extract raw signals
raw_lig = lowest_ev_signal['raw_lig']
raw_iso = lowest_ev_signal['raw_iso']
true_sig = lowest_ev_signal['scaled_true_sig']

# --- Figure 1: Fitted Iso Comparison ---
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')
ax1.plot(total_t, raw_iso, label='Raw Iso', alpha=0.6)
ax1.plot(total_t, raw_lig, label='Raw Lig', alpha=0.6)

# overlay fitted iso from all processed methods
for method_key, processed_entry in processed_signals[min_DV].items():
    if method_key == 'OLS':
        for subkey, res in processed_entry.items():
            fitted_iso = res.get('fitted_iso', None)
            if fitted_iso is not None:
                label = f"OLS {subkey.replace('smooth', 'sm').replace('varyt', 'vt')}"
                ax1.plot(total_t, fitted_iso, label=label, alpha=0.7)
    else:
        res = processed_entry[i]
        fitted_iso = res.get('fitted_iso', None)
        if fitted_iso is not None:
            label = method_key.replace('_', ' ').title()
            ax1.plot(total_t, fitted_iso, label=label, alpha=0.7)

ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')
ax1.set_title(f"Overlayed Signal Comparisons Across Methods | Lowest EV = {min_ev:.4f} | {param_name}={float(min_DV):.2f}")
ax1.legend(loc='upper right', fontsize=8)

# --- Figure 2: dF/F Comparison ---
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')
ax2.plot(total_t, true_sig, label='True Signal', color='black', linestyle='--', alpha=0.8)

# overlay dF/F from all processed methods
for method_key, processed_entry in processed_signals[min_DV].items():
    if method_key == 'OLS':
        for subkey, res in processed_entry.items():
            dff = res.get('dff', None)
            if dff is not None:
                label = f"OLS {subkey.replace('smooth', 'sm').replace('varyt', 'vt')}"
                ax2.plot(total_t, dff, label=label, alpha=0.7)
    else:
        res = processed_entry[i]
        dff = res.get('dff', None)
        if dff is not None:
            label = method_key.replace('_', ' ').title()
            ax2.plot(total_t, dff, label=label, alpha=0.7)

ax2.set_xlabel('Time')
ax2.set_ylabel('dF/F')
ax2.set_title(f"Overlayed Signal Comparisons Across Methods | Lowest EV = {min_ev:.4f} | {param_name}={float(min_DV):.2f}")
ax2.legend(loc='upper right', fontsize=8)


#%% plot lowest EV signal (4 panels: true, raw iso, fitted iso, raw lig)

# Grab the lowest EV signal across all DVs and methods
sim_list = simulated_signals[min_DV]
i = min_idx
lowest_ev_signal = sim_list[i]

# extract raw signals
raw_lig = lowest_ev_signal['raw_lig']
raw_iso = lowest_ev_signal['raw_iso']
true_sig = lowest_ev_signal['scaled_true_sig']

# --- Determine the method that produced this lowest EV signal ---
method_name = 'Unknown'
fitted_iso = None

# Loop over all processed signals for this DV to find the method corresponding to index i
for method_key, processed_entry in processed_signals[min_DV].items():
    if method_key == 'OLS':
        for subkey, res in processed_entry.items():
            if i < len(res['dff']):  # check index exists
                fitted_iso = res.get('fitted_iso', None)
                method_name = method_labels.get(subkey, subkey)
                break
        if fitted_iso is not None:
            break
    else:
        res = processed_entry[i]
        fitted_iso = res.get('fitted_iso', None)
        method_name = method_labels.get(method_key, method_key)
        break

# --- Plot the 4 panels ---
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True, layout='constrained')

axs[0].plot(total_t, true_sig, color='black', alpha=1.0)
axs[0].set_title('True Signal')

axs[1].plot(total_t, raw_iso, color='red', alpha=1.0)
axs[1].set_title('Raw Isosbestic')

if fitted_iso is not None:
    axs[2].plot(total_t, fitted_iso, color='blue', alpha=1.0)
axs[2].set_title('Fitted Isosbestic')

axs[3].plot(total_t, raw_lig, color='green', alpha=1.0)
axs[3].set_title('Raw Ligand')
axs[3].set_xlabel('Time')

# Super title
fig.suptitle(
    f"Lowest EV at {param_name}: {float(min_DV):.2f} | {method_name} | idx = {i} (EV = {min_ev:.4f})",
    fontsize=14
)

plt.show()


#%% plot artifact and noise for the lowest EV signal

artifact = lowest_ev_signal['artifact']
noise = lowest_ev_signal['noise']

fig, ax = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')

ax.plot(total_t, artifact, color='purple', alpha=0.8, label='Artifact')
ax.plot(total_t, noise, color='orange', alpha=0.8, label='Noise')

ax.set_xlabel('Time')
ax.set_ylabel('Signal')
ax.set_title('Artifact and Noise (Lowest EV Signal)')
ax.legend(loc='upper right')
plt.show()


#%% regression coefficients for all methods at lowest EV index
print(f'\nRegression Coefficients for All Methods at DV = {min_DV}, index = {min_idx}:')

# iterate over all 4 combinations of smooth_fit and vary_t + 2 LPF-only methods + IRLS
method_keys = []

# OLS with smooth_fit T/F and vary_t T/F
for smooth_fit in [True, False]:
    for vary_t in [True, False]:
        method_keys.append(f'smooth_fit_{smooth_fit}_vary_t_{vary_t}')

# LPF 1mHz
method_keys.append('lowpass_1mHz')

# IRLS 
method_keys.append('irls_regression')

# iterate and print coefficients
for method_key in method_keys:
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
        

#%% plot random signal that is not the lowest EV

all_methods = list(ev_results[min_DV].keys())
other_method = random.choice(all_methods)

ev_list = ev_results[min_DV][other_method]
non_min_indices = [j for j in range(len(ev_list)) if j != min_idx]

other_idx = random.choice(non_min_indices)
other_ev = ev_list[other_idx]

print(f'Other EV = {other_ev:.4f} from DV group: {min_DV:.4f}, method: {other_method}, index: {other_idx}')

sl.plot_comparative_figures(
    raw_lig=sim_list[other_idx]['raw_lig'],
    raw_iso=sim_list[other_idx]['raw_iso'],
    baseline_iso=sim_list[other_idx]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[other_idx]['scaled_true_sig'],
    fs=fs,
    ev=other_ev,
    dv=min_DV,
    param_name=param_name,
    extra_title=f'Random non-min EV ({other_method})'
)


#%% plot random signal that is not the lowest EV (two figures: fitted iso & dF/F)

# grab list of simulated signals for this DV
sim_list = simulated_signals[min_DV]

# choose a random method
all_methods = list(ev_results[min_DV].keys())
other_method = random.choice(all_methods)

# choose a random index that is not the lowest EV
ev_list = ev_results[min_DV][other_method]
non_min_indices = [j for j in range(len(ev_list)) if j != min_idx]
other_idx = random.choice(non_min_indices)
other_ev = ev_list[other_idx]

print(f"Random non-min EV = {other_ev:.4f} | DV = {min_DV:.2f}, method = {other_method}, index = {other_idx}")

# extract signals
raw_lig = sim_list[other_idx]['raw_lig']
raw_iso = sim_list[other_idx]['raw_iso']
true_sig = sim_list[other_idx]['scaled_true_sig']

# --- Figure 1: Fitted Iso Comparison ---
fig1, ax1 = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')
ax1.plot(total_t, raw_iso, label='Raw Iso', alpha=0.6)
ax1.plot(total_t, raw_lig, label='Raw Lig', alpha=0.6)

# overlay fitted iso from all processed methods
for method_key, processed_entry in processed_signals[min_DV].items():
    if method_key == 'OLS':
        for subkey, res in processed_entry.items():
            fitted_iso = res.get('fitted_iso', None)
            if fitted_iso is not None:
                label = f"OLS {subkey.replace('smooth', 'sm').replace('varyt', 'vt')}"
                ax1.plot(total_t, fitted_iso, label=label, alpha=0.7)
    else:
        res = processed_entry[other_idx]
        fitted_iso = res.get('fitted_iso', None)
        if fitted_iso is not None:
            label = method_key.replace('_', ' ').title()
            ax1.plot(total_t, fitted_iso, label=label, alpha=0.7)

ax1.set_xlabel('Time')
ax1.set_ylabel('Signal')
ax1.set_title(f"Overlayed Signal Comparisons Across Methods | Random non-min EV = {other_ev:.4f} | {param_name}={float(min_DV):.2f}")
ax1.legend(loc='upper right', fontsize=8)

# --- Figure 2: dF/F Comparison ---
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4), layout='constrained')
ax2.plot(total_t, true_sig, label='True Signal', color='black', linestyle='--', alpha=0.8)

# overlay dF/F from all processed methods
for method_key, processed_entry in processed_signals[min_DV].items():
    if method_key == 'OLS':
        for subkey, res in processed_entry.items():
            dff = res.get('dff', None)
            if dff is not None:
                label = f"OLS {subkey.replace('smooth', 'sm').replace('varyt', 'vt')}"
                ax2.plot(total_t, dff, label=label, alpha=0.7)
    else:
        res = processed_entry[other_idx]
        dff = res.get('dff', None)
        if dff is not None:
            label = method_key.replace('_', ' ').title()
            ax2.plot(total_t, dff, label=label, alpha=0.7)

ax2.set_xlabel('Time')
ax2.set_ylabel('dF/F')
ax2.set_title(f"Overlayed Signal Comparisons Across Methods | Random non-min EV = {other_ev:.4f} | {param_name}={float(min_DV):.2f}")
ax2.legend(loc='upper right', fontsize=8)


#%% plotting the signal for the largest delta EV between any two methods

all_methods = list(next(iter(ev_results.values())).keys())  # get all methods

max_delta_ev = -np.inf
max_delta_info = {
    "DV": None,
    "index": None,
    "method1": None,
    "method2": None,
    "ev1": None,
    "ev2": None,
}

# Loop through all DVs and their signals
for DV, methods_dict in ev_results.items():
    n_signals = len(next(iter(methods_dict.values())))  # number of signals in this DV

    for idx in range(n_signals):
        # compare all method pairs
        for i, method1 in enumerate(all_methods):
            for method2 in all_methods[i+1:]:
                if method1 not in methods_dict or method2 not in methods_dict:
                    continue

                ev1 = methods_dict[method1][idx]
                ev2 = methods_dict[method2][idx]
                delta = abs(ev1 - ev2)

                if delta > max_delta_ev:
                    max_delta_ev = delta
                    max_delta_info = {
                        "DV": DV,
                        "index": idx,
                        "method1": method1,
                        "method2": method2,
                        "ev1": ev1,
                        "ev2": ev2,
                    }

# extract signal info
DV_target = max_delta_info["DV"]
index_target = max_delta_info["index"]
sim_list = simulated_signals[DV_target]

title = (f"Largest ΔEV ({max_delta_info['method1']} vs {max_delta_info['method2']}) = "
         f"{max_delta_ev:.4f} at DV = {float(DV_target):.2f}, index = {index_target}")

sl.plot_comparative_figures(
    raw_lig=sim_list[index_target]['raw_lig'],
    raw_iso=sim_list[index_target]['raw_iso'],
    baseline_iso=sim_list[index_target]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[index_target]['scaled_true_sig'],
    fs=fs,
    ev=max_delta_ev,
    dv=DV_target,
    param_name=param_name,
    suptitle_text=None,
    extra_title=title
)

