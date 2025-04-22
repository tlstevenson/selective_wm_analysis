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
import json


    
#%% 

def get_baseline_form (form_type): 
    
    match form_type:
        
        #excluding double_exp_decay as it does not seem to fit well to the data 
        
        case 'double_exp_decay':
            baseline_form = lambda x, a, b, c, d, e: a*np.exp(-x/b) + c*np.exp(-x/(b*d)) + e
            #double exponential decay 
            
            # specifying the boundary for each parameter (only for exponential ones) 
            #               a       b      c     d     e 
            bounds = ([0      ,      0, -np.inf, 0, -np.inf],
                      [ np.inf, np.inf,  np.inf, 1,  np.inf])
          
        case 'exp_linear':
            baseline_form = lambda x, a, b, c, d: a*np.exp(-x/b) + c*x +d 
            #combination of the exponential decay term and a linear term 
            
            #               a       b      c     d     
            bounds = ([0      ,      0, -np.inf,  -np.inf],
                      [ np.inf, np.inf,  np.inf,  np.inf])
              
        case 'polynomial':
            baseline_form = lambda x, *coeffs: np.polyval(coeffs, x) 
            # works for different degrees, depending on the number of coeffs you provide
            # 2 coeffs -> linear, 3 coeffs -> quadratic,  4 coeffs -> cubic, 5 coeffs -> quartic, 6 coeffs -> quintic 
            bounds = None 
 
    return {'formula': baseline_form, 'bounds': bounds}           
      

        
def fit_baseline(signal, form_type, degree = None): 
   
    # get the formula  
    baseline_form = get_baseline_form(form_type) 
    x = np.arange(len(signal))  # this just create an array that contains integars. 
    nans = np.isnan(signal)
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        warnings.simplefilter('error', category=OptimizeWarning)

        try:
            if form_type == 'polynomial':
               # for polynomial, curve_fit doesn't necessary work well without specifying the initial guess.  
               # also testing different degrees 
               params = np.polyfit( x[~nans], signal[~nans], degree)  
               #baseline = np.polyval(params, x)   # if needed 
                    
               return  {'params': params }    #'formula': baseline_form['formula'],

            else: 
                params = curve_fit(baseline_form['formula'], x[~nans] , signal[~nans], bounds=baseline_form['bounds'])[0]
        
                return  {'params': params}   #'formula': baseline_form['formula'],
               # return baseline_form['formula'](x, *params)  # if need the fitted BL 


        except (OptimizeWarning, RuntimeError):
            return {'params': np.nan}  # Return NaN when the fitting was not successful # might not work well for dife degree of polynomial 
            

#%%  fit the actual signal to BL see what are the realistic params to estimate decay 


variant_subj = {'3.6 CAG': [182, 202], '3.8 CAG': [179, 180, 188], '3.8 Syn': [191, 207]}

#testing w. just one sbj for now 
#variant_subj = {'3.8 CAG': [180],}

subj_variant = {v: k for k, vs in variant_subj.items() for v in vs}
subj_ids = utils.flatten(variant_subj)

# getting all the session ids for all the subj
sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids)

#to testing w. only a few sessions 
#sess_ids = {179: sess_ids[179][:1]}
#sess_ids = {180:sess_ids[180][16:17]}

loc_db = db.LocalDB_BasicRLTasks('')
fp_data = loc_db.get_sess_fp_data(utils.flatten(sess_ids))
fp_data = fp_data['fp_data']

form_type = ['polynomial']   #form_type = ['double_exp_decay', 'exp_linear', 'polynomial']


#%% fit the data 

baseline_fit_comb = {}  

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        raw_signals = fp_data[subj_id][sess_id]['raw_signals']
        t = fp_data[subj_id][sess_id]['time']
        fs = 1/(t[1] - t[0]) 
        
        for region in raw_signals.keys():
            for isolig in raw_signals[region].keys():
                
                raw_signal = raw_signals[region][isolig]
                smooth_signal = fp_utils.filter_signal(raw_signal, 0.1, fs) 
                
                for form in form_type:
                    
                    if form == 'polynomial':
                        
                       for degree in range(4, 5):    # change the degree if needed 
                           baseline_fit_info = fit_baseline(smooth_signal, form, degree)
                           form_name =  f"polynomial{degree}"   
                           baseline_fit_comb.setdefault(subj_id, {}).setdefault(sess_id, {}).setdefault(region, {}).setdefault(isolig, {})[form_name] = baseline_fit_info  
                                                     
                    else: 
                        baseline_fit_info = fit_baseline(smooth_signal, form)
                        form_name = form
                        baseline_fit_comb.setdefault(subj_id, {}).setdefault(sess_id, {}).setdefault(region, {}).setdefault(isolig, {})[form_name] = baseline_fit_info

                        # plt.plot(t, raw_signal, label='raw signal', alpha=0.5)
                        # plt.plot(t, smooth_signal, label='0.1Hz filt signal', alpha=0.5)
                        # plt.plot(t, baseline, label='baseline', alpha=0.5)
                        # plt.xlabel('Time (s)')
                        # plt.ylabel('Fluorescent Signal (V)')
                        # plt.title(f'sbj{subj_id}, session {sess_id}{region}{isolig}: Formula {form} degree3')
                        # plt.plot(dpi=600)  
                        # plt.legend() 
                        # plt.show()

#%%flatten the dictionary so that I can easily filter and plot ?      

baseline_df = []

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:
        for region in baseline_fit_comb[subj_id][sess_id].keys():
            for isolig in baseline_fit_comb[subj_id][sess_id][region].keys():
                for form_name in baseline_fit_comb[subj_id][sess_id][region][isolig].keys():
                    try:
                        data = baseline_fit_comb[subj_id][sess_id][region][isolig][form_name]
                        if isinstance(data, tuple):  # Handle tuple case
                            data = data[0]
                        params = np.atleast_1d(data['params']) if 'params' in data else None
                    except (KeyError, IndexError, TypeError):
                        params = None  # Default if retrieval fails

                    baseline_df.append({
                        'subj_id': subj_id,
                        'sess_id': sess_id,
                        'region': region,
                        'isolig': isolig,
                        'form_name': form_name,
                        **{f'param_{i}': params[i] if params is not None and i < params.shape[0] and not np.isnan(params[i]) else None for i in range(7)}
                    })


#%% function to plot histogram to see the distribution of params  


def plot_distr (df, region, isolig, form_name, title): 
    
    filtered_list = [
    item for item in df
    if (region is None or item.get('region') == region)  # Ignore  if None
    and (isolig is None or item.get('isolig') == isolig)  # Ignore if None
    and item.get('form_name') == form_name
    ]
    param_values = {f"param_{i}": [item[f"param_{i}"] for item in filtered_list if f"param_{i}" in item] for i in range(6)}

    fig, axs = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
    plt.suptitle(title)
    axs = axs.flatten()
    plot_idx = 0
    
    param_range = {}
    
    for param, values in param_values.items():
        values = [v for v in values if v is not None] # Filter out None values

        if values:  # Ensure there are valid values to plot
        
            mean_val = np.mean(values)
            std_dev = np.std(values, ddof=1)  # Sample standard deviation
            z_scores = [(v - mean_val) / std_dev if std_dev != 0 else 0 for v in values]
            mean_SD4 = mean_val + std_dev * 4
            mean_plus_SD4 = mean_val + std_dev * 4
            mean_minus_SD4 = mean_val - std_dev * 4

            
            # Filter values where -10 <= Z-score <= 10
            filt_values = [v for v, z in zip(values, z_scores) if -10 <= z <= 10]
            max_value = max(filt_values)
            min_value = min(filt_values)
            
            #saving the param range to use them later 
            param_range[param] = {
                "max_value" : max_value, 
                "min_value" : min_value,
                "mean_plus_SD4": mean_plus_SD4,
                "mean_minus_SD4": mean_minus_SD4,
                "SD": std_dev
                }
         
            # to plot using log ( this doesn't work for negative values ) 
            bin_edges = np.logspace(np.log10(min(filt_values)), np.log10(max(filt_values)), num=100)
            axs[plot_idx].hist(filt_values, bins=bin_edges, edgecolor='black', alpha=0.7)
            
            #axs[plot_idx].hist(filt_values, bins=100, edgecolor='black', alpha = 0.7)
            axs[plot_idx].set_xscale('log')  
            axs[plot_idx].set_xlabel("Values")
            axs[plot_idx].set_ylabel("Number of sessions")
            axs[plot_idx].set_title(f"{param}")
            axs[plot_idx].text(0.95, 0.95, f"Max: {max_value}\nMin: {min_value} \nmean+SD4: {mean_plus_SD4} \nmean-SD4: {mean_minus_SD4}", 
                   transform=axs[plot_idx].transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right')
            
            plot_idx += 1  # Move to the next subplot
            
    return fig, param_range

#%%    plot  histograms

#form_names = ['double_exp_decay', 'exp_linear','polynomial3','polynomial4','polynomial5' ]
form_names = ['exp_linear']

 
param_range_comb = {}  # run without this when combining the data from multiple baseline types

for form_name in form_names:
    fig, param_range = plot_distr(baseline_df, None, None, form_name, title = (f'Combined region and iso/lig, {form_name}'))
    plt.show()
    
    param_range_comb[form_name] = param_range
    

     
#%% save dictionaries using pickle  as needed 

# Save data to a file
with open("param_range_comb.pkl", "wb") as f:
    pickle.dump(param_range_comb, f)
    
with open("param_range_comb.json", "w") as f:
    json.dump(param_range_comb, f)
    
# to load it  
with open("exp_linear_baseline_df.pkl", "rb") as f:
    baseline_df = pickle.load(f)


    
