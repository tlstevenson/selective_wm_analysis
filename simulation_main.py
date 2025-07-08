# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:23:54 2025

@author: atruo

relevant data/graphs stored here: https://docs.google.com/document/d/1VdKfUvVSoFxo4LlaEtMhj-IyLhX59fA6npD_ieepOV0/edit?usp=sharing
"""
import init

import matplotlib.pyplot as plt

import numpy as np

import json
import simulation_lib as sl

import random

#%% define recording length

fs = 20  # Sampling freq. 
duration = 500
total_t = np.linspace(0, duration, num=fs*duration, endpoint=False)   # for the entire duration 


# %% Simulate baselines

a_mean = 4
a_sd = 1
b_mean = 100
b_sd = 20
c_mean = 2e-6
c_sd = 2e-7
d_mean = 10
d_sd = 1


#%% simulate signals

n = 50
time = total_t
param_name = 'SNR'
param_range = [0.1, 5]
param_step = 0.2
f_range_sig = [0.1, 10]
f_range_noise = [1, 10]
max_art_count = 5
art_duration_range = [1, 50]
f_range_art = [0.01, 10]
form_type = 'exp_linear'
sim_type = 2
SD_frac_default = 0.1
alpha_default = 0
SNR_default = 10
SAR_default = 1
scale = 0.01

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
#         key = f"smooth_fit_{smooth_fit}_vary_t_{vary_t}"
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
                processed_output = sl.process_signals(
                    raw_lig, raw_iso, baseline_iso, total_t, fs,
                    smooth_fit=smooth_fit, vary_t=vary_t, smooth_lpf=smooth_lpf
                )
                processed_signals[DV][key].append(processed_output)

                # Update progress
                step += 1
                
                print(f"Processing signal {step}/{total_steps}...", end="\r")
                
                


#%% report average clamping after processing signals

if sl.clamp_total_calls > 0:
    avg_clamped = sl.clamp_total_points / sl.clamp_total_calls
    print(f"[process_signals] Total number of clamped denominator points: {sl.clamp_total_points}")
    print(f"[process_signals] Average number of clamped denominator points per call: {avg_clamped:.2f}")
else:
    print("[process_signals] Clamp function was never called.")


#%% calculate explained variance ev_results

ev_results = {}

for DV, method in processed_signals.items():  # Iterate over SD_frac keys
    ev_results[DV] = {}  # Create nested dictionary for each SD_frac
    
    # Extract the correct true_sig for each signal in simulated_signals
    signal_list = simulated_signals[DV]  # List of signals for this SD_frac

    for method_key, processed_list in method.items():  # Iterate over different methods
        ev_results[DV][method_key] = []  # Store results

        for signal_entry, processed_entry in zip(signal_list, processed_list):  
            scaled_true_sig = signal_entry["scaled_true_sig"]  # Extract true signal from simulated_signals
            dff_iso = processed_entry["dff"]  # Extract dff_iso from processed_signals

            # Compute evaluation metric
            ev_output = sl.ev(scaled_true_sig, dff_iso)  

            # Store result
            ev_results[DV][method_key].append(ev_output)


#%% for troubleshooting. Testing ev  
# ev_results = {}

# for key, signals in processed_signals.items():
#     ev_results[key] = ev(test_signals['true_sig'], signals["dff_iso"])

#%% debugging, testing if jitter = 0 
#_ = sl.generate_baseline('exp_linear', 0.5, total_t)

#%% debugging for signal = 0


# Grab one existing simulated signal from the first DV group (e.g., SNR = 0.1)
DV_key = list(simulated_signals.keys())[0]  # e.g., '0.10'
example_idx = 0  # index of signal to plot
signal_data = simulated_signals[DV_key][example_idx]

# Extract components to plot
scaled_true_sig = signal_data["scaled_true_sig"]
baseline_lig = signal_data["baseline_lig"]
baseline_iso = signal_data["baseline_iso"]

# Plot
plt.figure(figsize=(12, 5))
plt.plot(total_t, scaled_true_sig, label='Scaled True Signal', linewidth=2)
plt.plot(total_t, baseline_lig, label='Baseline LIG', alpha=0.8)
plt.plot(total_t, baseline_iso, label='Baseline ISO', alpha=0.8)
plt.xlabel("Time (s)")
plt.ylabel("Signal Amplitude")
plt.title(f"Simulated Signal Components (DV = {DV_key}, Index = {example_idx})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% plot statistical test results and signals

sl.plot_ev_results(ev_results, param_name, exclude_outliers=False)  # explained variance
sl.plot_ev_results(ev_results, param_name, exclude_outliers=True)

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

print(f"Lowest EV = {correct_ev:.4f} from DV group: {min_DV:.4f}, method: {min_method}, index: {i}")

# Print baseline parameters for lowest EV signal
lig_params = sim_list[i]["lig_params"]
iso_params = sim_list[i]["iso_params"]

print("\nBaseline Parameters for Lowest EV Case:")
print("Ligand Parameters:")
for k, v in lig_params.items():
    print(f"  {k}: {v:.4g}")
print("Isosbestic Parameters:")
for k, v in iso_params.items():
    print(f"  {k}: {v:.4g}")


# plot using the lowest-EV signal
sl.plot_comparative_figures(
    raw_lig=sim_list[i]['raw_lig'],
    raw_iso=sim_list[i]['raw_iso'],
    baseline_iso=sim_list[i]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[i]['scaled_true_sig'],
    fs=fs,
    ev=min_ev,
    dv=min_DV
)

 
#%%
# plot random signal that is not the lowest-EV

ev_list = ev_results[min_DV][min_method]
non_min_indices = [j for j in range(len(ev_list)) if j != min_idx]

other_idx = random.choice(non_min_indices)
other_ev = ev_results[min_DV][min_method][other_idx]

print(f"Other EV = {other_ev:.4f} from DV group: {min_DV:.4f}, method: {min_method}, index: {other_idx}")

sl.plot_comparative_figures(
    raw_lig=sim_list[other_idx]['raw_lig'],
    raw_iso=sim_list[other_idx]['raw_iso'],
    baseline_iso=sim_list[other_idx]['baseline_iso'],
    time=total_t,
    true_sig=sim_list[other_idx]['scaled_true_sig'],
    fs=fs,
    ev=other_ev,
    dv=min_DV
)


#%%

lowest_processed = processed_signals[min_DV][min_method][i].copy()
lowest_processed["raw_lig"] = simulated_signals[min_DV][i]["raw_lig"]
lowest_processed["raw_iso"] = simulated_signals[min_DV][i]["raw_iso"]


fig = sl.plot_signals(
    processed_signals=lowest_processed,
    true_sig=true_sig,
    t=total_t,
    ev=min_ev,
    fs=fs,
    window_sec=500,
    title=f"Lowest EV: {float(min_DV):.2f} | {min_method} | idx={i}"
)
plt.show()


#%% Plot artifact and noise from the lowest EV signal

# Get signal from simulated_signals with the lowest EV
sim_list = simulated_signals[min_DV]
i = min_idx
lowest_ev_signal = sim_list[i]

# Extract artifact and noise
artifact = lowest_ev_signal["artifact"]
noise = lowest_ev_signal["noise"]

# Plot them separately
plt.figure(figsize=(12, 5))
plt.plot(total_t, artifact, label='Artifact', color='orange')
plt.plot(total_t, noise, label='Noise', color='purple', alpha=0.7)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Artifact and Noise (Lowest EV Signal)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#%% Regression Coefficients for All Methods at Lowest EV Index
print(f"\nRegression Coefficients for All Methods at DV = {min_DV}, index = {min_idx}:")

# Iterate over all 4 combinations of smooth_fit and vary_t
for smooth_fit in [True, False]:
    for vary_t in [True, False]:
        method_key = f"smooth_fit_{smooth_fit}_vary_t_{vary_t}"
        
        try:
            entry = processed_signals[min_DV][method_key][min_idx]
            fit_params = entry.get("fit_params", None)
        except (IndexError, KeyError):
            fit_params = None

        print(f"\nMethod: {method_key}")
        if fit_params is not None:
            for idx, coeff in enumerate(fit_params):
                print(f"  Param {idx + 1}: {coeff:.6g}")
        else:
            print("  No fit_params found.")

#%% plotting lig - fit_iso
fig2, ax2 = plt.subplots()
my_raw = np.array(processed_signals[min_DV]["raw_lig"])
my_fit = np.array(processed_signals[min_DV]["fitted_iso"])
ax2.plot(time, my_raw)
ax2.plot(time, my_fit)
ax2.plot(time, my_raw-my_fit)
#%%

# fig, ax = plt.subplots(4,2)

# idx = 0

#for vary_t_var in [True, False]:
#    for vary_smooth in [True, False]:
#        signal_list = process_signals(raw_lig, raw_iso, baseline_iso, time, fs, smooth_fit = vary_smooth, vary_t = vary_t_var, filt_denom=True, smooth_lpf=0.1)
        
#        if vary_t_var and vary_smooth:
#        elif vary_t_var and not vary_smooth:
#       elif not vary_t_var and vary_smooth:
#        else:
                
#        ax[idx%4, idx//4 + 1].plot(x,y)
#        ax[idx%4, idx//4].plot(x,y)
        
#        idx = idx + 2
                                


"""
# Plotting: Show 4-panel plots for multiple signals across conditions
max_examples = 2  # Number of signals to visualize per condition
window_sec = 5    # How many seconds to show at the start of the signal

max_plots = 2
plots_created = 0

for DV in processed_signals:
    for method_key in processed_signals[DV]:
        signal_list = processed_signals[DV][method_key]
        sim_list = simulated_signals[DV]
        ev_list = ev_results[DV][method_key]

        for i in range(min(len(signal_list), max_examples)):
            if plots_created >= max_plots:
                break

            processed = signal_list[i]
            print(f"sim_list[{i}] keys:", sim_list[i].keys())
            true_sig = sim_list[i]["true_sig"]
            ev_val = ev_list[i]

            fig = sl.plot_signals(
                processed_signals=processed,
                true_sig=true_sig,
                t=total_t,
                ev=ev_val,
                fs=fs,
                window_sec=5,
                title=f"{DV} | {method_key} | Example {i}"
            )
            plt.show()
            plots_created += 1
        
        if plots_created >= max_plots:
            break
    if plots_created >= max_plots:
        break
"""


# for key, signals in processed_signals.items():
#    plot_signals(signals, test_signals['true_sig'], test_signals['baseline_lig'], test_signals['baseline_iso'], total_t, key, ev_results[key])
#    plt.savefig(f"Type1_{key}.png", dpi=300)
#    plt.show()


# Save fig 

# for key, signals in processed_signals.items():
#     plot_signals(signals, baselines, true_sig, total_t, key, ev_results[key])
#     plt.savefig(f"Type1_{key}_290-300.png", dpi=300)
#     plt.show()
