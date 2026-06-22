# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 13:22:41 2025

@author: tanne
"""

import init
import pandas as pd
from pyutils import utils
import hankslab_db.basicRLtasks_db as db
from hankslab_db import db_access
#import beh_analysis_helpers as beh
#import fp_analysis_helpers as fpah
#from fp_analysis_helpers import Alignment as Align
from sys_neuro_tools import plot_utils, fp_utils
import agents
import training_helpers as th
import sim_helpers as sh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
from os import path
import pickle
from scipy.stats import pearsonr
from pathlib import Path

script_dir = Path(__file__).parent.resolve()




subj_ids = [198, 199, 274, 400, 402]#[179, 188, 191, 207] # 182

cluster_path = path.join('/Users/tiffanyma/model_fits/probabilistic_bandit/fits', 'fit_models_sep_rates_tszma.json')
save_path = path.join(script_dir, 'fit_models_new.json')
all_models={}
print(f"Cluster file at: {cluster_path}")
print(f"File exists: {path.exists(cluster_path)}")

if path.exists(cluster_path):
    all_models = agents.load_model(cluster_path)
    print(f"Loaded models from: {cluster_path}")
    print(f"Type of all_models: {type(all_models)}")
    print(f"Subjects found: {list(all_models.keys())}")
elif path.exists(save_path):
    all_models = agents.load_model(save_path)
    print(f"Loaded models from local save: {save_path}")
else:
    all_models = {}
    print("No fit results found in cluster path or local save path")

# merge in corrected Persev (fixed) fits from rerun file
persev_rerun_path = path.join('/Users/tiffanyma/model_fits/probabilistic_bandit/fits',
                               'fit_models_sep_rates_persev_rerun.json')
if path.exists(persev_rerun_path):
    all_models_persev = agents.load_model(persev_rerun_path)
    for subj in all_models_persev.keys():
        if subj in all_models:
            all_models[subj].update(all_models_persev[subj])
        else:
            all_models[subj] = all_models_persev[subj]
    print(f"Merged corrected Persev fits from: {persev_rerun_path}")
else:
    print("No Persev rerun file found — skipping merge")

print(f"Final all_models keys: {list(all_models.keys())}")
    


# load data
# sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=2)

# # start from the third session (so index=2)-->do not account for the first two sessions
# sess_ids = {subj: sess[2:] for subj, sess in sess_ids.items()}

sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids, protocol='ClassicRLTasks', stage_num=2)

# get session data
reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

all_sess = th.define_choice_outcome(all_sess)
    

#%%
ignore_subj = ['182']
# plot accuracy and total LL
subjids = list(all_models.keys())
subjids = [s for s in subjids if 'meta' not in s.lower() and not any(ign in s for ign in ignore_subj)]


ignore_models = ['Basic - Value']#'Basic - Value'['Q/Persev/Fall', 'SI/Persev', 'Q SI', 'RL SI'] # ['basic - value only']
ignore_any_match = False
plot_best_fit_only = True

model_names = list(all_models[subjids[0]].keys())
model_names.sort(key=str.lower)

if ignore_any_match:
    model_names = [n for n in model_names if not any(im in n for im in ignore_models)]
else:
    model_names = [n for n in model_names if not n in ignore_models]
    model_names = [n for n in model_names if not n.endswith('_cv')]

# build dataframe with accuracy and LL per fit
fit_mets = []
for subj in subjids:
    for model_name in model_names:
        if model_name in all_models[subj]:
            for i in range(len(all_models[subj][model_name])):
                model = all_models[subj][model_name][i]['model'].model
                perf = all_models[subj][model_name][i]['perf']
                
                fit_mets.append({'subjid': subj, 'model': '{} ({})'.format(model_name, th.count_params(model)), 'n_params': th.count_params(model), **perf})
            
fit_mets = pd.DataFrame(fit_mets)

#added for debugging purposes
print("NaN check:")
print(f"ll_total NaNs: {fit_mets['ll_total'].isna().sum()}")
print(f"ll_avg NaNs: {fit_mets['ll_avg'].isna().sum()}")
if fit_mets[['ll_total', 'll_avg']].isna().any().any():
    print("\nRows with NaN values:")
    print(fit_mets[fit_mets[['ll_total', 'll_avg']].isna().any(axis=1)])


fit_mets['n_trials'] = (fit_mets['ll_total']/fit_mets['ll_avg']).fillna(0).astype(int)
fit_mets['norm_llh'] = np.exp(fit_mets['ll_avg'])
fit_mets['bic'] = th.calc_bic(fit_mets['ll_total'], fit_mets['n_params'], fit_mets['n_trials'])
fit_mets['ll_total'] = -fit_mets['ll_total']
fit_mets['ll_avg'] = -fit_mets['ll_avg']
fit_mets['acc'] = fit_mets['acc']*100

model_names = fit_mets['model'].unique().tolist()
model_names.sort(key=str.lower)

best_model_counts = {m: {'norm_llh': 0, 'acc': 0, 'bic': 0} for m in model_names}

# calculate percent difference from best fitting model per subject
fit_mets[['diff_ll_avg', 'diff_norm_llh', 'diff_acc', 'diff_bic']] = 0.0
for subj in subjids:
    subj_sel = fit_mets['subjid'] == subj
    subj_mets = fit_mets[subj_sel]
    best_ll_avg = subj_mets['ll_avg'].min()
    best_norm_llh = subj_mets['norm_llh'].max()
    best_acc = subj_mets['acc'].max()
    best_bic = subj_mets['bic'].min()

    fit_mets.loc[subj_sel, 'diff_ll_avg'] = (subj_mets['ll_avg'] - best_ll_avg)/best_ll_avg*100
    fit_mets.loc[subj_sel, 'diff_norm_llh'] = -(subj_mets['norm_llh'] - best_norm_llh)/best_norm_llh*100
    fit_mets.loc[subj_sel, 'diff_acc'] = -(subj_mets['acc'] - best_acc)/best_acc*100
    fit_mets.loc[subj_sel, 'diff_bic'] = (subj_mets['bic'] - best_bic)/best_bic*100
    
    # count best models
    best_ll_names = subj_mets[subj_mets['norm_llh'] == best_norm_llh]['model'].unique()
    best_acc_names = subj_mets[subj_mets['acc'] == best_acc]['model'].unique()
    best_bic_names = subj_mets[subj_mets['bic'] == best_bic]['model'].unique()
    for name in best_ll_names:
        best_model_counts[name]['norm_llh'] += 1
        
    for name in best_acc_names:
        best_model_counts[name]['acc'] += 1
        
    for name in best_bic_names:
        best_model_counts[name]['bic'] += 1
    
best_model_counts = pd.DataFrame(best_model_counts).transpose().reset_index().rename(columns={'index': 'model'})

if plot_best_fit_only:
    fit_mets = fit_mets.loc[fit_mets.groupby(['subjid', 'model'])['norm_llh'].idxmax()]

perf_cols = ['diff_ll_avg', 'diff_norm_llh', 'diff_acc', 'diff_bic']

avg_diffs = fit_mets.groupby(['subjid', 'model'])[perf_cols].min().reset_index()
avg_diffs = avg_diffs.groupby('model')[perf_cols].mean().reset_index()

ax_height = max(len(model_names)/5, 3)
# Plot model fit performances
fig, axs = plt.subplots(2, 1, figsize=(10,ax_height*2), layout='constrained')

sb.stripplot(fit_mets, y='model', x='norm_llh', hue='subjid', ax=axs[0], palette='colorblind')
sb.stripplot(fit_mets, y='model', x='acc', hue='subjid', ax=axs[1], palette='colorblind')

fig.suptitle('Model Performance Comparison - Values')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[0].set_xlabel('Avg p(correct) per trial')
axs[1].set_xlabel('Accuracy (%)')
axs[0].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
axs[1].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))


# plot performance differences from best fit
fig, axs = plt.subplots(3, 1, figsize=(10,ax_height*3), layout='constrained')

for ax in axs:
    plot_utils.plot_x0line(ax=ax)

sb.stripplot(fit_mets, y='model', x='diff_norm_llh', hue='subjid', ax=axs[0], palette='colorblind')
sb.stripplot(fit_mets, y='model', x='diff_acc', hue='subjid', ax=axs[1], palette='colorblind')
sb.stripplot(fit_mets, y='model', x='diff_bic', hue='subjid', ax=axs[2], palette='colorblind')

fig.suptitle('Model Performance Comparison - % Worse from Best Model')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[2].set_title('Model BIC')
axs[0].set_xlabel('% Worse Avg p(correct) per trial')
axs[1].set_xlabel('% Worse Accuracy')
axs[2].set_xlabel('% Worse BIC')
axs[0].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
axs[1].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
axs[2].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))


# plot best model counts
fig, axs = plt.subplots(3, 1, figsize=(10,ax_height*3), layout='constrained')

sb.barplot(best_model_counts, y='model', x='norm_llh', ax=axs[0], errorbar=None)
sb.barplot(best_model_counts, y='model', x='acc', ax=axs[1], errorbar=None)
sb.barplot(best_model_counts, y='model', x='bic', ax=axs[2], errorbar=None)

fig.suptitle('Best Model Performance Counts')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[2].set_title('Model BIC')
axs[0].set_xlabel('# Best Models')
axs[1].set_xlabel('# Best Models')
axs[2].set_xlabel('# Best Models')

# plot average best model differences
fig, axs = plt.subplots(3, 1, figsize=(10,ax_height*3), layout='constrained')

sb.barplot(avg_diffs, y='model', x='diff_norm_llh', ax=axs[0], errorbar=None)
sb.barplot(avg_diffs, y='model', x='diff_acc', ax=axs[1], errorbar=None)
sb.barplot(avg_diffs, y='model', x='diff_bic', ax=axs[2], errorbar=None)

fig.suptitle('Average Model Difference from Best Model per Subject')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[2].set_title('Model BIC')
axs[0].set_xlabel('% Worse Avg p(correct) per trial')
axs[1].set_xlabel('% Worse Accuracy')
axs[2].set_xlabel('% Worse BIC')


# %% Get FP data and analyze peaks

fp_sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2, subj_ids=subj_ids)
implant_info = db_access.get_fp_implant_info(subj_ids)

filename = 'two_arm_bandit_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path):
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        aligned_signals = saved_data['aligned_signals']
        aligned_metadata = saved_data['metadata']

alignments = [Align.cue, Align.reward] #  
signal_type = 'dff_iso' # , 'z_dff_iso'

filter_props = {Align.cue: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}},
                Align.reward: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}}}

peak_find_props = {Align.cue: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                               'PL': {'min_dist': 0.2, 'peak_tmax': 1.5, 'peak_edge_buffer': 0.2, 'lim_peak_width_to_edges': True}},
                   Align.reward: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                                  'PL': {'min_dist': 0.5, 'peak_tmax': 3.5, 'peak_edge_buffer': 0.2, 'lim_peak_width_to_edges': False}}}

sides = ['contra', 'ipsi']
regions = ['DMS', 'PL']

ignored_signals = {'PL': [],
                   'DMS': []}

t = aligned_signals['t']

peak_metrics = []

for subj_id in subj_ids:
    print('Analyzing peaks for subj {}'.format(subj_id))
    for sess_id in fp_sess_ids[subj_id]:
        
        trial_data = all_sess[all_sess['sessid'] == sess_id]
        rewarded = trial_data['rewarded'].to_numpy()
        responded = ~np.isnan(trial_data['response_time']).to_numpy()
        choice = trial_data['choice']
        choice_np = choice.to_numpy()
        reward_time = trial_data['reward_time'].to_numpy()[:,None]
        stays = choice[:-1].to_numpy() == choice[1:].to_numpy()
        switches = np.insert(~stays, 0, False)
        stays = np.insert(stays, 0, False)
        prev_rewarded = np.insert(rewarded[:-1], 0, False)
        prev_unrewarded = np.insert(~rewarded[:-1], 0, False)
        
        resp_rewarded = rewarded[responded]
        
        for region in regions:
            if sess_id in ignored_signals[region]:
                continue

            region_side = implant_info[subj_id][region]['side']
            choice_side = choice.apply(lambda x: fpah.get_implant_rel_side(x, region_side) if not x == 'none' else 'none').to_numpy()

            for align in alignments:
                if not align in aligned_signals[subj_id][sess_id][signal_type]:
                    continue

                t_r = t[align][region]
                mat = aligned_signals[subj_id][sess_id][signal_type][align][region]

                # calculate peak properties on a trial-by-trial basis
                contra_choices = choice_side == 'contra'
                contra_choices = contra_choices[responded]
                
                resp_trial = 1
                for i in range(mat.shape[0]):
                    if responded[i]:
                        metrics = fpah.calc_peak_properties(mat[i,:], t_r, 
                                                            filter_params=filter_props[align][region],
                                                            peak_find_params=peak_find_props[align][region],
                                                            fit_decay=False)

                        peak_metrics.append(dict([('subj_id', subj_id), ('sess_id', sess_id), ('signal_type', signal_type), 
                                                 ('align', align.name), ('region', region), ('trial', resp_trial),
                                                 ('rewarded', rewarded[i]), ('side', choice_side[i]), ('abs_side', choice_np[i]),
                                                 ('reward_time', reward_time[i]), ('RT', trial_data['RT'].iloc[i]),
                                                 ('cpoke_out_latency', trial_data['cpoke_out_latency'].iloc[i]), *metrics.items()]))
                        
                        resp_trial += 1
                            

peak_metrics = pd.DataFrame(peak_metrics)
# drop unused columns
peak_metrics.drop(['decay_tau', 'decay_params', 'decay_form'], axis=1, inplace=True)

# filter peak metric outliers
# make subject ids categories

ignore_outliers = True
ignore_any_outliers = True
outlier_thresh = 10

t_min = 0.02
t_max = {a: {r: peak_find_props[a][r]['peak_tmax'] - t_min for r in regions} for a in alignments} 

parameters = ['peak_time', 'peak_height'] #, 'decay_tau'

filt_peak_metrics = peak_metrics.copy()

# remove outliers on a per-subject basis:
if ignore_outliers:
    
    # first get rid of peaks with times too close to the edges of the peak window (10ms from each edge)
    peak_sel = np.full(len(peak_metrics), False)
    for align in alignments:    
        for region in regions:
            align_region_sel = (peak_metrics['align'] == align) & (peak_metrics['region'] == region)
            sub_peak_metrics = peak_metrics[align_region_sel]
            peak_sel[align_region_sel] = ((sub_peak_metrics['peak_height'] > 0) & 
                                          (sub_peak_metrics['peak_time'] > t_min) &
                                          (sub_peak_metrics['peak_time'] < t_max[align][region]))
            
    # look at potentially problematic peaks
    # t = aligned_signals['t']
    # rem_peak_info = peak_metrics[~peak_sel]
    # rem_peak_info =  rem_peak_info[rem_peak_info['signal_type'] == 'dff_iso']
    # rem_subj_ids = np.unique(rem_peak_info['subj_id'])
    # for subj_id in rem_subj_ids:
    #     subj_peak_info = rem_peak_info[rem_peak_info['subj_id'] == subj_id]
    #     for _, row in subj_peak_info.iterrows():
    #         mat = aligned_signals[row['subj_id']][row['sess_id']]['dff_iso'][row['align']][row['region']]
    #         _, ax = plt.subplots(1,1)
    #         ax.set_title('{} - {}, {} {}-aligned, trial {}'.format(row['subj_id'], row['sess_id'], row['region'], row['align'], row['trial']))
    #         ax.plot(t[row['align']][row['region']], mat[row['trial'], :])
    #         plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
    #         peak_idx = np.argmin(np.abs(t[row['align']][row['region']] - row['peak_time']))
    #         ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
    #         ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')

    filt_peak_metrics = filt_peak_metrics[peak_sel]
    
    # first add iqr multiple columns
    for param in parameters:
        filt_peak_metrics['iqr_mult_'+param] = np.nan
    
    # calculate iqr multiple for potential outliers
    outlier_grouping = ['subj_id', 'sess_id']
    
    # compute IQR on different groups of trials based on the alignment and region
    for align in alignments:
        # separate peaks by outcome at time of reward
        if align == Align.reward:
            align_outlier_grouping = outlier_grouping+['rewarded']
        else:
            align_outlier_grouping = outlier_grouping
            
        for region in regions:
            # separate peaks by side for DMS since very sensitive to choice side
            if region == 'DMS':
                region_outlier_grouping = align_outlier_grouping+['side']
            else:
                region_outlier_grouping = align_outlier_grouping
                
            align_region_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['region'] == region)
            
            filt_peak_metrics.loc[align_region_sel, :] = fpah.calc_iqr_multiple(filt_peak_metrics[align_region_sel], region_outlier_grouping, parameters)
    
    # then remove outlier values
    if ignore_any_outliers:

        outlier_sel = np.full(len(filt_peak_metrics), False)
        for param in parameters:
            outlier_sel = outlier_sel | (np.abs(filt_peak_metrics['iqr_mult_'+param]) >= outlier_thresh)
            
        filt_peak_metrics.loc[outlier_sel, parameters] = np.nan
        
    else:
        for param in parameters:
            outlier_sel = np.abs(filt_peak_metrics['iqr_mult_'+param]) >= outlier_thresh
            
            if any(outlier_sel):
                # look at outlier peaks
                # t = aligned_signals['t']
                # rem_peak_info = filt_peak_metrics[outlier_sel]
                # rem_peak_info =  rem_peak_info[rem_peak_info['signal_type'] == 'dff_iso']
                # rem_subj_ids = np.unique(rem_peak_info['subj_id'])
                # for subj_id in rem_subj_ids:
                #     subj_peak_info = rem_peak_info[rem_peak_info['subj_id'] == subj_id]
                #     for _, row in subj_peak_info.iterrows():
                #         mat = aligned_signals[row['subj_id']][row['sess_id']]['dff_iso'][row['align']][row['region']]
                #         _, ax = plt.subplots(1,1)
                #         ax.set_title('{} - {}, {} {}-aligned, trial {}'.format(row['subj_id'], row['sess_id'], row['region'], row['align'], row['trial']))
                #         ax.plot(t[row['align']][row['region']], mat[row['trial'], :])
                #         plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
                #         peak_idx = np.argmin(np.abs(t[row['align']][row['region']] - row['peak_time']))
                #         ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
                #         ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')
            
                filt_peak_metrics.loc[outlier_sel, param] = np.nan

# %% Compare model fit outputs

subjids = [198, 199, 274, 400, 402, 237, 238, 424, 483]#[179, 188, 191] # [179, 188, 191, 207]

sides = ['contra', 'ipsi']
regions = ['DMS', 'PL']

use_fp_sess_only = True
plot_ind_sess = True
plot_output = True
plot_output_diffs = False
plot_agent_states = True
plot_rpes = True
plot_fp_peak_amps = False
plot_fp_rpe_corr = False

is_bayes_model = False

#['Q - Same Alpha Only, K Fixed', 'Q - All Alpha Free, All K Fixed']
#['Q - All Alpha Shared, All K Fixed', 'Q - All Alpha Shared, All K Free']
#['Q - All Alpha Shared, All K Fixed', 'Q - Same Alpha Only Shared, K Fixed']
#['Q - All Alpha Shared, All K Fixed', 'Q - All Alpha Shared, Counter D/R K=-1']
#['Q - All Alpha Shared, All K Free', 'SI - Free Same/Diff Rew Evidence']
#['Q - All Alpha Shared, All K Fixed', 'SI - Separate Rew/Unrew Evidence']
# compare_model_info = {'Q SI - Alpha Free, K Free': {'agent_names': ['State', 'Value', 'Belief']}, 
#                       'Q SI - Separate High Alphas, Const Low K, High K Free': {'agent_names': ['State', 'Value', 'Belief']}}
# compare_model_info = {'Q SI - Alpha Free, K Free': {'agent_names': ['State', 'Value', 'Belief']}, 
#                       'Q SI - Alpha Free, K Free, Belief Update First': {'agent_names': ['State', 'Value', 'Belief']}}

compare_model_info = {'Q - Alpha Rew/Unrew Shared, All K Fixed': {'agent_names': ['Value']},
                      'Q/Persev - Alpha Rew/Unrew Shared, All K Fixed': {'agent_names': ['Value', 'Persev']},
                      #'SI - Separate Rew/Unrew Evidence': {'agent_names': ['Value']},
                      #'SI/Persev - Separate Rew/Unrew Evidence': {'agent_names': ['Value', 'Persev']}}
                     }

               
compare_models = list(compare_model_info.keys())
model_outputs = {s: {m: {} for m in compare_models} for s in subj_ids}

plot_sess_ids = fp_sess_ids if use_fp_sess_only else sess_ids
    
for subj in subjids: 

    # Format model inputs to re-run fit models
    sess_data = all_sess[all_sess['sessid'].isin(plot_sess_ids[subj])]
    # filter out no responses
    sess_data = sess_data[sess_data['hit']==True]
    
    ## Create 3-D inputs tensor and 3-D labels tensor
    n_sess = len(plot_sess_ids[subj])
    max_trials = np.max(sess_data.groupby('sessid').size())

    # use all trials for evaluation
    inputs = torch.zeros(n_sess, max_trials, 3)
    left_choice_labels = torch.zeros(n_sess, max_trials, 1)
    trial_mask = torch.zeros(n_sess, max_trials, 1)

    # populate tensors from behavioral data
    for i, sess_id in enumerate(plot_sess_ids[subj]):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data) 
        
        left_choice_labels[i, :n_trials-1, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)
        inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['chose_left'], trial_data['chose_right'], trial_data['rewarded_int']]).T).type(torch.float)
        trial_mask[i, :n_trials-1, :] = 1
        
    for model_name in compare_models:
        
        # get best model
        best_model_idx = 0
        for i in range(len(all_models[str(subj)][model_name])):
            if all_models[str(subj)][model_name][i]['perf']['norm_llh'] > all_models[str(subj)][model_name][best_model_idx]['perf']['norm_llh']:
                best_model_idx = i    

        model = all_models[str(subj)][model_name][best_model_idx]['model'].model.clone()

        # run model
        output, agent_states, fit_perf = th.eval_model(model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                       output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
        
        state_diff_hist = torch.stack([torch.stack(agent.state_diff_hist, dim=1) for agent in model.agents], dim=-1).numpy()
        #state_delta_hist = torch.stack([torch.stack(agent.state_delta_hist, dim=1) for agent in model.agents], dim=-1).numpy()
        
        output_dict = {'model': model, 'output': output, 'agent_states': agent_states, 'perf': fit_perf,
                       'agent_state_diff_hist': state_diff_hist}
        
        if isinstance(model.agents[0], agents.QValueStateInferenceAgent):
            value_hist = torch.stack(model.agents[0].v_hist[1:], dim=1).numpy()
            belief_hist = torch.stack(model.agents[0].belief_hist[1:], dim=1).numpy()
            agent_states = np.insert(agent_states, 1, value_hist, axis=3)
            agent_states = np.insert(agent_states, 2, belief_hist, axis=3)
            
        if isinstance(model.agents[0], agents.BayesianAgent):
            output_dict['full_nll'] = torch.stack(model.agents[0].nll_hist_full, dim=1).numpy()
            output_dict['stay_nll'] = torch.stack(model.agents[0].nll_hist_stay, dim=1).numpy()
            output_dict['rew_kl_div'] = model.agents[0].get_kl_divergence(p_dist='reward')
            output_dict['rew_ent'] = model.agents[0].get_entropy(p_dist='reward')
            output_dict['ent_nll_diff'] = output_dict['stay_nll']/output_dict['rew_ent']
            
            is_bayes_model = True
        
        model_outputs[subj][model_name] = output_dict


if plot_ind_sess:
    for subj in subjids: 
    
        # Format model inputs to re-run fit models
        sess_data = all_sess[all_sess['sessid'].isin(plot_sess_ids[subj])]
        # filter out no responses
        sess_data = sess_data[sess_data['hit']==True]
                
        # compare output similarities between the models
        for i, sess_id in enumerate(plot_sess_ids[subj]):
            trial_data = sess_data[sess_data['sessid'] == sess_id]
            
            # output_diffs = {}
            # for model_name in compare_models:
            #     output_diffs[model_name] = {}
            #     outputs = np.stack([mo['output'][i,:,0] for mo in model_outputs[subj][model_name]], axis=1)
            #     output_diffs[model_name]['outputs'] = outputs
            #     output_diffs[model_name]['diff'] = np.mean(np.abs(np.diff(outputs, axis=1)), axis=1)
            #     output_diffs[model_name]['avg'] = np.mean(outputs, axis=1)
            #     # assuming the value agent is the first agent
            #     agent_states = np.stack([mo['agent_states'][i,:,:,0] for mo in model_outputs[subj][model_name]], axis=1)
            #     output_diffs[model_name]['agent_states'] = agent_states
            #     output_diffs[model_name]['avg_agent_states'] = np.mean(agent_states, axis=1)
            #     output_diffs[model_name]['state_diffs'] = np.stack([mo['agent_state_diff_hist'][i,:,:,0] for mo in model_outputs[subj][model_name]], axis=1)
            #     output_diffs[model_name]['avg_state_diffs'] = np.mean(output_diffs[model_name]['state_diffs'], axis=1)
            #     output_diffs[model_name]['trans_state_diffs'] = utils.rescale(output_diffs[model_name]['state_diffs'], 0, 1, axis=1)
            #     output_diffs[model_name]['trans_avg_state_diffs'] = np.mean(output_diffs[model_name]['trans_state_diffs'], axis=1)
            
            output_diffs = np.abs(np.diff(np.stack([model_outputs[subj][m]['output'][i,:,0] for m in compare_models], axis=1), axis=1))
            n_agents = np.max([model_outputs[subj][m]['agent_states'].shape[3] for m in compare_models])
    
            n_rows = 0
            if plot_output:
                n_rows += 1
            if plot_output_diffs:
                n_rows += 1
            if plot_agent_states:
                n_rows += n_agents
            if plot_rpes:
                n_rows += 1
                if is_bayes_model:
                    n_rows += 4
            if plot_fp_peak_amps:
                n_rows += len(regions)
                
            fig, axs = plt.subplots(n_rows, 1, figsize=(15,n_rows*4), layout='constrained')
                
            # get block transitions
            block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
            block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
            block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
                
            fig.suptitle('Subj {} Session {} Model Comparison'.format(subj, sess_id))
    
            # label trials from 1 to the last trial
            x = np.arange(len(trial_data))+1
            
            ax_idx = 0
            
            if plot_output:
                ax = axs[ax_idx]
                # plot model outputs
                for j, model_name in enumerate(compare_models):
                    ax.plot(x[1:], model_outputs[subj][model_name]['output'][i,:len(trial_data)-1,0], color='C{}'.format(j), alpha=0.6, label=model_name)
                    
                ax.set_ylabel('p(Choose Left)')
                ax.set_xlabel('Trial')
                ax.set_title('Model Outputs', fontsize=10)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                ax.axhline(y=0.5, color='black', linestyle='dashed')
                ax.margins(x=0.01)
                
                th._draw_choices(trial_data, ax)
                th._draw_blocks(block_switch_trials, block_rates, ax)
                
                ax_idx += 1
            
            # Plot output diffs between models
            if plot_output_diffs:
                ax = axs[ax_idx]
                
                ax.plot(x[1:], output_diffs[:len(trial_data)-1])
        
                ax.set_ylabel('Output Diffs')
                ax.set_xlabel('Trial')
                ax.set_title('Avg Model Output Differences', fontsize=10)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.axhline(y=0, color='black', linestyle='dashed')
                ax.margins(x=0.01)
                
                th._draw_choices(trial_data, ax)
                th._draw_blocks(block_switch_trials, block_rates, ax)
                
                ax_idx += 1
                
            if plot_agent_states:
                for j in range(n_agents):
                    ax = axs[ax_idx]
                    
                    for k, model_name in enumerate(compare_models):
                        agent_names = compare_model_info[model_name]['agent_names']
                        if j < len(agent_names):
                            ax.plot(x[1:], model_outputs[subj][model_name]['agent_states'][i,:len(trial_data)-1,0,j], color='C{}'.format(k), alpha=0.6, label='{}, {} left'.format(model_name, agent_names[j]))
                            ax.plot(x[1:], model_outputs[subj][model_name]['agent_states'][i,:len(trial_data)-1,1,j], color='C{}'.format(k), alpha=0.6, linestyle='dotted', label='{}, {} right'.format(model_name, agent_names[j]))
                        
                    ax.set_ylabel('Agent State Values')
                    ax.set_xlabel('Trial')
                    ax.set_title('Agent State Values', fontsize=10)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                    #ax.axhline(y=0, color='black', linestyle='dashed')
                    ax.grid(axis='y')
                    ax.margins(x=0.01)
                    
                    th._draw_choices(trial_data, ax)
                    th._draw_blocks(block_switch_trials, block_rates, ax)
                
                    ax_idx += 1
    
            if plot_rpes:
                if is_bayes_model:
                    all_rpes = {'Point RPE': 'agent_state_diff_hist', 'B(reward) KL Divergence': 'rew_kl_div', 'Outcome Neg Log Likelihood': 'stay_nll', 'Dist Entropy': 'rew_ent', 'NLL/Entropy': 'ent_nll_diff'} # 'Outcome Log Likelihood': 'full_ll', 
                else:
                    all_rpes = {'RPE': 'agent_state_diff_hist'}
                
                for label, met in all_rpes.items():
                    
                    ax = axs[ax_idx]
                    # plot all model RPEs for each side
                    for j, model_name in enumerate(compare_models):
                        data = model_outputs[subj][model_name][met]
                        if len(data.shape) == 4:
                            data = data[i,:len(trial_data),:,0]
                        else:
                            data = data[i,:len(trial_data),:]
                        ax.plot(x, data[:,0], color='C{}'.format(j), alpha=0.6, label='{}, left'.format(model_name))
                        ax.plot(x, data[:,1], color='C{}'.format(j), alpha=0.6, linestyle='dotted', label='{}, right'.format(model_name))
                        
                    ax.set_ylabel('RPEs')
                    ax.set_xlabel('Trial')
                    ax.set_title('Model {}'.format(label), fontsize=10)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                    #ax.axhline(y=0, color='black', linestyle='dashed')
                    ax.grid(axis='y')
                    ax.margins(x=0.01)
                
                    th._draw_choices(trial_data, ax)
                    th._draw_blocks(block_switch_trials, block_rates, ax)
                    
                    ax_idx += 1

            if plot_fp_peak_amps:
                for j, region in enumerate(regions):
                    ax = axs[ax_idx+j]
                        
                    sess_region_metrics = filt_peak_metrics[(filt_peak_metrics['sess_id'] == sess_id) & (filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == Align.reward)].sort_values('trial')
                    peak_amps = sess_region_metrics['peak_height'].to_numpy()
                    peak_trials = sess_region_metrics['trial'].to_numpy()
                    
                    for k, side in enumerate(sides):
                        for rewarded in [True, False]:
                            side_outcome_sel = (sess_region_metrics['side'] == side) & (sess_region_metrics['rewarded'] == rewarded)
                            color = 'C{}'.format(k+3)
                            # change color lightness based on outcome
                            color = utils.change_color_lightness(color, -0.30) if rewarded else utils.change_color_lightness(color, 0.30)
                            rew_label = 'rew' if rewarded else 'unrew'
                            ax.vlines(x=peak_trials[side_outcome_sel], ymin=0, ymax=peak_amps[side_outcome_sel], color=color, label='{} choice, {}'.format(side, rew_label))
                        
                    _, y_label = fpah.get_signal_type_labels(signal_type)
                    ax.set_ylabel(y_label)
                    ax.set_xlabel('Trial')
                    ax.set_title('{} - Reward Peak Amplitude'.format(region), fontsize=10)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                    #ax.axhline(y=0, color='black', linestyle='dashed')
                    ax.grid(axis='y')
                    ax.margins(x=0.01)
                    
                    th._draw_choices(trial_data, ax)
                    th._draw_blocks(block_switch_trials, block_rates, ax)
                    
            if plot_fp_rpe_corr:
                if is_bayes_model:
                    all_rpes = {'Point RPE': 'agent_state_diff_hist', 'B(reward) KL Divergence': 'rew_kl_div', 'Outcome NLL': 'stay_nll', 'Dist Entropy': 'rew_ent', 'NLL/Entropy': 'ent_nll_diff'} # 'Outcome Log Likelihood': 'full_ll', 
                else:
                    all_rpes = {'RPE': 'agent_state_diff_hist'}
                
                n_rows = len(regions)
                n_cols = len(all_rpes)
                
                for r, rewarded in enumerate([True, False]):
                    rew_label = 'Rewarded' if rewarded else 'Unrewarded'
                    
                    fig, axs = plt.subplots(n_rows, n_cols, figsize=(3*n_cols,3*n_rows), layout='constrained')
                    fig.suptitle('{} Trials FP Peak/Model RPE Correlations - Subj {} Session {}'.format(rew_label, subj, sess_id))
                    _, y_label = fpah.get_signal_type_labels(signal_type)
                    
                    for j, region in enumerate(regions):
    
                        sess_region_metrics = filt_peak_metrics[(filt_peak_metrics['sess_id'] == sess_id) & (filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == Align.reward)].sort_values('trial')
                        peak_amps = sess_region_metrics['peak_height'].to_numpy()
                        peak_trials = sess_region_metrics['trial'].to_numpy()
                        
                        region_side = implant_info[subj][region]['side']
                        
                        for k, (met_label, met) in enumerate(all_rpes.items()):
                            
                            rpe_data = model_outputs[subj][model_name][met]
                            if len(rpe_data.shape) == 4:
                                rpe_data = rpe_data[i,:len(trial_data),:,0]
                            else:
                                rpe_data = rpe_data[i,:len(trial_data),:]
                            
                            ax = axs[j,k]
                            ax.set_title('{} {} Corr'.format(region, met_label))
                            
                            for s, rel_side in enumerate(sides):
                                abs_side = fpah.get_implant_abs_side(rel_side, region_side)
                                side_idx = 0 if abs_side == 'left' else 1

                                side_outcome_sel = (sess_region_metrics['side'] == rel_side) & (sess_region_metrics['rewarded'] == rewarded)
                                
                                if np.sum(side_outcome_sel) > 2:
                                    side_outcome_trial_idx = peak_trials[side_outcome_sel]-1
                                    
                                    fp_amps = peak_amps[side_outcome_sel]
                                    nan_sel = np.isnan(fp_amps)
                                    model_rpe = rpe_data[side_outcome_trial_idx, side_idx]
                                    
                                    corr, p_val = pearsonr(fp_amps[~nan_sel], model_rpe[~nan_sel])
                                    
                                    color = 'C{}'.format(s)
                                    #marker = 'o' if rewarded else 'x'
                                    
                                    ax.scatter(model_rpe, fp_amps, color=color, alpha=0.5, #marker=marker, 
                                               label='{} choice - R$^2$={:.3f} (p={:.3f})'.format(rel_side, corr, p_val))
                        
                            ax.set_ylabel(y_label)
                            ax.set_xlabel(met_label)
                            ax.legend(fontsize=8, borderaxespad=0)
                            
            plt.show(block=False)
                
# plot comparison of fit performance for each model

# first build dataframe
if len(compare_models) > 1:
    model_fit_comparison = []
    for subj in subjids:
        for model_name in compare_models:
            
            model_fit_comparison.append({'subj': subj, 'model': model_name, **model_outputs[subj][model_name]['perf']})
            
    model_fit_comparison = pd.DataFrame(model_fit_comparison)
    
    # then plot
    metrics = ['norm_llh'] #, 'bic', 'acc'
    metric_labels = ['Normalized Likelihood', 'BIC', 'Accuracy']
    n_cols =  len(metrics)
    fig, axs = plt.subplots(1, n_cols, figsize=(3*n_cols,3), layout='constrained')
    axs = np.resize(np.array(axs), n_cols)
    fig.suptitle('Fit Performance Comparison: {} vs {}'.format(compare_models[0], compare_models[1]))
    
    for i, metric in enumerate(metrics):
        pivot_metrics = model_fit_comparison.pivot(index='subj', columns='model', values=metric).reset_index()
        ax = axs[i]
        sb.scatterplot(pivot_metrics, x=compare_models[0], y=compare_models[1], ax=ax)
        plot_utils.plot_unity_line(ax)
        ax.set_title(metric_labels[i])
        ax.set_xlabel(compare_models[0])
        ax.set_ylabel(compare_models[1])
    
# %% Plot model fits

import math

# rebuild model_names from CV keys
model_names_cv = set()
for subj in subjids:
    for key in all_models[subj].keys():
        if key.endswith('_cv'):
            model_names_cv.add(key[:-3])
model_names_cv = sorted(list(model_names_cv), key=str.lower)

# build per-fold NLL (same logic as CV Model Performance Plot cell)
cv_fit_mets_by_type = []
for subj in subjids:
    for mn in model_names_cv:
        cv_key = mn + '_cv'
        
        if cv_key in all_models[subj]:
            fit_repeats = all_models[subj][cv_key]
            
            if len(fit_repeats) == 0:
                continue
            
            n_folds = fit_repeats[0]['n_folds']
            
            best_nll_per_fold = []
            for fold_idx in range(n_folds):
                fold_nlls = []
                for repeat in fit_repeats:
                    fold_results = [f for f in repeat['folds'] if f['fold_idx'] == fold_idx]
                    if len(fold_results) > 0:
                        nll = fold_results[0]['nll']
                        if not math.isnan(nll):
                            fold_nlls.append(nll)
                
                if len(fold_nlls) > 0:
                    best_nll_per_fold.append(min(fold_nlls))
            
            if len(best_nll_per_fold) == n_folds:
                avg_nll_per_fold = np.mean(best_nll_per_fold)
                
                model_obj = fit_repeats[0]['model'].model
                n_params = th.count_params(model_obj)
                model_name_with_params = '{} ({})'.format(mn, n_params)
                
                # extract model type prefix (everything before " - ")
                model_type = mn.split(' - ')[0].strip()
                
                cv_fit_mets_by_type.append({
                    'subjid': subj,
                    'model': model_name_with_params,
                    'model_type': model_type,
                    'avg_nll_per_fold': avg_nll_per_fold,
                    'n_folds': n_folds,
                    'n_params': n_params,
                })

cv_fit_mets_by_type = pd.DataFrame(cv_fit_mets_by_type)

# calculate percent difference from best model per subject (across ALL models)
cv_subjids_by_type = sorted(cv_fit_mets_by_type['subjid'].unique().tolist())
cv_fit_mets_by_type['diff_avg_nll'] = 0.0
for subj in cv_subjids_by_type:
    subj_sel = cv_fit_mets_by_type['subjid'] == subj
    subj_mets = cv_fit_mets_by_type[subj_sel]
    best_avg_nll = subj_mets['avg_nll_per_fold'].min()
    cv_fit_mets_by_type.loc[subj_sel, 'diff_avg_nll'] = (subj_mets['avg_nll_per_fold'] - best_avg_nll) / best_avg_nll * 100

# generate separate plots for each individual model
all_cv_model_names = sorted(cv_fit_mets_by_type['model'].unique().tolist(), key=str.lower)

for model_name in all_cv_model_names:
    model_data = cv_fit_mets_by_type[cv_fit_mets_by_type['model'] == model_name]

    # plot 1 - raw average NLL per fold, one dot per subject
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5), layout='constrained')
    sb.stripplot(model_data, y='model', x='avg_nll_per_fold', hue='subjid',
                 ax=ax, palette='colorblind', order=[model_name],
                 hue_order=cv_subjids_by_type)
    fig.suptitle('{} - CV Average NLL per Fold'.format(model_name), fontsize=14)
    ax.set_xlabel('Average NLL per Fold', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
    plt.show()

    # plot 2 - percent difference from best model, one dot per subject
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5), layout='constrained')
    plot_utils.plot_x0line(ax=ax)
    sb.stripplot(model_data, y='model', x='diff_avg_nll', hue='subjid',
                 ax=ax, palette='colorblind', order=[model_name],
                 hue_order=cv_subjids_by_type)
    fig.suptitle('{} - % Worse than Best Model per Subject'.format(model_name), fontsize=14)
    ax.set_title('One dot per subject/reward rate', fontsize=12)
    ax.set_xlabel('% Worse Average NLL per Fold', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
    plt.show()

    # plot 3 - percent difference averaged across subjects, one dot per model
    avg_data = model_data.groupby('model')['diff_avg_nll'].mean().reset_index()
    fig, ax = plt.subplots(1, 1, figsize=(12, 2.5), layout='constrained')
    plot_utils.plot_x0line(ax=ax)
    sb.stripplot(avg_data, y='model', x='diff_avg_nll',
                 ax=ax, order=[model_name], color='steelblue', size=8)
    fig.suptitle('{} - Average % Worse than Best Model'.format(model_name), fontsize=14)
    ax.set_title('One dot per model (averaged across subjects)', fontsize=12)
    ax.set_xlabel('Avg % Worse Average NLL per Fold', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    plt.show()
    

# %% Example Session Fit: Best Q vs Best SI Model

# find the best Q-family and SI-family model from CV results
q_types = [t for t in cv_fit_mets_by_type['model_type'].unique() if t.startswith('Q')]
si_types = [t for t in cv_fit_mets_by_type['model_type'].unique() if t.startswith('SI')]

# get the model with the lowest average % worse across subjects for each family
q_data = cv_fit_mets_by_type[cv_fit_mets_by_type['model_type'].isin(q_types)]
si_data = cv_fit_mets_by_type[cv_fit_mets_by_type['model_type'].isin(si_types)]

best_q_model_label = q_data.groupby('model')['diff_avg_nll'].mean().idxmin()
best_si_model_label = si_data.groupby('model')['diff_avg_nll'].mean().idxmin()

# strip the " (N)" parameter count suffix to get the original model name
best_q_model = best_q_model_label.rsplit(' (', 1)[0]
best_si_model = best_si_model_label.rsplit(' (', 1)[0]

print('Best Q model:  {} -> {}'.format(best_q_model_label, best_q_model))
print('Best SI model: {} -> {}'.format(best_si_model_label, best_si_model))

compare_models_poster = [best_q_model, best_si_model]
compare_labels_poster = [best_q_model_label, best_si_model_label]
compare_colors = ['#1f77b4', '#ff7f0e']  # blue for Q, orange for SI

n_example_sess = 1  # number of example sessions to plot per subject

for subj in subjids:
    # extract integer subject id from string key (e.g. '274 (50/10)' -> 274)
    subj_int = int(subj.split(' ')[0]) if ' ' in subj else int(subj)
    
    sess_data = all_sess[all_sess['subjid'] == subj_int]
    sess_data = sess_data[sess_data['hit'] == True]
    
    if len(sess_data) == 0:
        print('No session data found for subject {}'.format(subj))
        continue
    
    sess_id_list = sess_data['sessid'].unique().tolist()
    n_sess = len(sess_id_list)
    max_trials = sess_data.groupby('sessid').size().max()
    
    # build input tensors
    inputs = torch.zeros(n_sess, max_trials, 3)
    left_choice_labels = torch.zeros(n_sess, max_trials, 1)
    trial_mask = torch.zeros(n_sess, max_trials, 1)
    
    for i, sess_id in enumerate(sess_id_list):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data)
        left_choice_labels[i, :n_trials-1, :] = torch.from_numpy(
            np.array(trial_data['chose_left'].iloc[1:])[:,None]).type(torch.float)
        inputs[i, :n_trials, :] = torch.from_numpy(
            np.array([trial_data['chose_left'], trial_data['chose_right'], 
                       trial_data['rewarded_int']]).T).type(torch.float)
        trial_mask[i, :n_trials-1, :] = 1
    
    # run both models
    model_outputs_poster = {}
    for model_name in compare_models_poster:
        # try regular key first, then fall back to CV key
        if model_name in all_models[subj]:
            model_key = model_name
        elif model_name + '_cv' in all_models[subj]:
            model_key = model_name + '_cv'
        else:
            print('Model {} not found for subject {}'.format(model_name, subj))
            continue
        
        # find best fit repeat
        best_idx = 0
        fit_repeats = all_models[subj][model_key]
        
        if model_key.endswith('_cv'):
            # for CV models, pick the repeat with the lowest total NLL
            valid_repeats = [(i, r) for i, r in enumerate(fit_repeats) if not math.isnan(r['total_nll'])]
            if len(valid_repeats) == 0:
                print('No valid CV repeats for {} subject {}'.format(model_name, subj))
                continue
            best_idx = min(valid_repeats, key=lambda x: x[1]['total_nll'])[0]
        else:
            for i in range(len(fit_repeats)):
                if fit_repeats[i]['perf']['norm_llh'] > fit_repeats[best_idx]['perf']['norm_llh']:
                    best_idx = i
        
        model = fit_repeats[best_idx]['model'].model.clone()
        output, _, fit_perf = th.eval_model(model, inputs, left_choice_labels, 
                                             trial_mask=trial_mask,
                                             output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
        model_outputs_poster[model_name] = {'output': output, 'perf': fit_perf}
    
    if len(model_outputs_poster) < 2:
        continue
    
    # plot example sessions
    for i in range(min(n_example_sess, n_sess)):
        sess_id = sess_id_list[i]
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data)
        x = np.arange(n_trials) + 1
        
        # block transitions
        block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial'].values
        block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'] + 1)
        block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob'].values
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 5), layout='constrained')
        
        # draw colored background bands for blocks
        for b in range(len(block_switch_trials) - 1):
            b_start = block_switch_trials[b]
            b_end = block_switch_trials[b + 1]
            # color based on which side is favored
            rate_str = str(block_rates[b]) if b < len(block_rates) else ''
            # Use light blue for left-favored, light red for right-favored
            if '75' in rate_str:
                band_color = '#d4e6f1'  # light blue
            elif '10' in rate_str:
                band_color = '#fadbd8'  # light red
            else:
                band_color = '#f0f0f0'  # neutral gray
            ax.axvspan(b_start, b_end, color=band_color, alpha=0.4, zorder=0)
        
        # smoothed behavioral choice curve (rolling average)
        chose_left = trial_data['chose_left'].values.astype(float)
        smooth_window = 10
        kernel = np.ones(smooth_window) / smooth_window
        smoothed_choices = np.convolve(chose_left, kernel, mode='same')
        ax.plot(x, smoothed_choices, color='#888888', alpha=0.6, linewidth=1.5, 
                linestyle='-', label='Behavior (smoothed)', zorder=1)
        
        # plot model outputs
        for j, (model_name, label) in enumerate(zip(compare_models_poster, compare_labels_poster)):
            if model_name in model_outputs_poster:
                ax.plot(x[1:], model_outputs_poster[model_name]['output'][i,:n_trials-1,0], 
                        color=compare_colors[j], alpha=0.85, linewidth=2.5, label=label, zorder=2+j)
        
        ax.axhline(y=0.5, color='black', linestyle='dashed', alpha=0.2, linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel('p(Choose Left)', fontsize=14)
        ax.set_xlabel('Trial', fontsize=14)
        ax.set_title('Subject {} — Session {} — Best Q vs SI Model'.format(subj, sess_id), fontsize=15)
        ax.tick_params(axis='both', labelsize=12)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0, framealpha=0.9)
        ax.margins(x=0.01)
        ax.set_axisbelow(True)
        
        plt.show()

# %% Debug NaN CV results - 274 (50/10) | Q - All Free
debug_subj = '274 (50/10)'
debug_model_name = 'Q - All Free'
debug_cv_key = debug_model_name + '_cv'

# get session data directly for subject 274 with 50/10 reward rate
debug_sess_data = all_sess[all_sess['subjid'] == 274]
debug_sess_data = debug_sess_data[debug_sess_data['block_prob'] == '50/10']
debug_sess_data = debug_sess_data[debug_sess_data['hit'] == True]

print('n_sessions:', debug_sess_data['sessid'].nunique())

# get training data in the same format as fit_model_cv
debug_training_data = th.get_model_training_data(debug_sess_data, basic_model=False)
inputs = debug_training_data['inputs']
labels = debug_training_data['labels']
trial_mask_eval = debug_training_data['trial_mask_eval']
n_trials_per_sess = debug_training_data['n_trials_per_sess']

# get the fold masks the same way as fit_model_cv
loss_output_transforms = th.get_loss_output_transforms(basic_model=False)
fold_masks = th.get_cv_fold_masks(debug_training_data['trial_mask_train'],
                                   trial_mask_eval, n_trials_per_sess, n_folds=3)

# load the saved CV model parameters - no refitting, just using saved params
cv_result = all_models[debug_subj][debug_cv_key][0]
model = cv_result['model'].model

print('Checking parameters for NaN/Inf:')
for name, param in model.named_parameters():
    has_nan = torch.isnan(param.data).any()
    has_inf = torch.isinf(param.data).any()
    print('  {} | NaN: {} | Inf: {} | Values: {}'.format(name, has_nan, has_inf, param.data))

print('\nRunning eval_model on each fold:')
for fold_idx, (fold_train_mask, fold_test_mask) in enumerate(fold_masks):
    print('\n  Fold {}:'.format(fold_idx + 1))
    print('  inputs NaN: {}'.format(torch.isnan(inputs).any()))
    print('  labels NaN: {}'.format(torch.isnan(labels.float()).any()))
    
    _, _, fold_perf = th.eval_model(model, inputs, labels,
                                     trial_mask=fold_test_mask,
                                     output_transform=loss_output_transforms['eval_output_transform'])
    
    print('  Fold NLL: {}'.format(-fold_perf['ll_total']))
    print('  Fold Acc: {:.2f}%'.format(fold_perf['acc'] * 100))
    print('  ll_total: {}'.format(fold_perf['ll_total']))
    print('  ll_avg: {}'.format(fold_perf['ll_avg']))



# %% Debug NaN CV results - 274 (50/10) | Bayes - No Switch Scatter
debug_subj = '274 (50/10)'
debug_model_name = 'Bayes - No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates'
debug_cv_key = debug_model_name + '_cv'

# get session data directly for subject 274 with 50/10 reward rate
debug_sess_data = all_sess[all_sess['subjid'] == 274]
debug_sess_data = debug_sess_data[debug_sess_data['block_prob'] == '50/10']
debug_sess_data = debug_sess_data[debug_sess_data['hit'] == True]

# get training data
debug_training_data = th.get_model_training_data(debug_sess_data, basic_model=False)
inputs = debug_training_data['inputs']
labels = debug_training_data['labels']
trial_mask_eval = debug_training_data['trial_mask_eval']
n_trials_per_sess = debug_training_data['n_trials_per_sess']

# get fold masks
loss_output_transforms = th.get_loss_output_transforms(basic_model=False)
fold_masks = th.get_cv_fold_masks(debug_training_data['trial_mask_train'],
                                   trial_mask_eval, n_trials_per_sess, n_folds=3)

# load saved CV model parameters
cv_result = all_models[debug_subj][debug_cv_key][0]
model = cv_result['model'].model

print('Checking parameters for NaN/Inf:')
for name, param in model.named_parameters():
    has_nan = torch.isnan(param.data).any()
    has_inf = torch.isinf(param.data).any()
    print('  {} | NaN: {} | Inf: {} | Values: {}'.format(name, has_nan, has_inf, param.data))

print('\nRunning eval_model on each fold:')
for fold_idx, (fold_train_mask, fold_test_mask) in enumerate(fold_masks):
    print('\n  Fold {}:'.format(fold_idx + 1))
    print('  inputs NaN: {}'.format(torch.isnan(inputs).any()))
    print('  labels NaN: {}'.format(torch.isnan(labels.float()).any()))
    
    _, _, fold_perf = th.eval_model(model, inputs, labels,
                                     trial_mask=fold_test_mask,
                                     output_transform=loss_output_transforms['eval_output_transform'])
    
    print('  Fold NLL: {}'.format(-fold_perf['ll_total']))
    print('  Fold Acc: {:.2f}%'.format(fold_perf['acc'] * 100))
    print('  ll_total: {}'.format(fold_perf['ll_total']))
    print('  ll_avg: {}'.format(fold_perf['ll_avg']))

# %% CV Model Performance Plot
import math

# rebuild model_names from CV keys to include SI models
model_names = set()
for subj in subjids:
    for key in all_models[subj].keys():
        if key.endswith('_cv'):
            model_names.add(key[:-3])  # strip _cv suffix
model_names = sorted(list(model_names), key=str.lower)

keep_models = [
    'Q - All Alpha Free, All K Fixed, Diff K=0.5',
    'Q - All Alpha Shared, All K Fixed',
    'Q - Alpha Rew/Unrew Shared, All K Fixed',
    'SI - All Separate Evidence',
    'SI - Free Same/Diff Rew Evidence',
    'SI - Shared Rew Evidence, Unrew Fixed 0',
    'Bayes - No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates',
]
model_names = [m for m in model_names if any(k in m for k in keep_models)]

print('Total models:', len(model_names))
# build per-fold NLL by looking at each fold across all fit repeats
# for each fold, take the best NLL across all fit repeats
cv_fit_mets = []
for subj in subjids:
    for model_name in model_names:
        cv_key = model_name + '_cv'
        
        if cv_key in all_models[subj]:
            fit_repeats = all_models[subj][cv_key]
            
            if len(fit_repeats) == 0:
                continue
            
            # get number of folds from first repeat
            n_folds = fit_repeats[0]['n_folds']
            
            # for each fold, find the best NLL across all fit repeats
            best_nll_per_fold = []
            for fold_idx in range(n_folds):
                fold_nlls = []
                for repeat in fit_repeats:
                    # find the fold result for this fold_idx
                    fold_results = [f for f in repeat['folds'] if f['fold_idx'] == fold_idx]
                    if len(fold_results) > 0:
                        nll = fold_results[0]['nll']
                        if not math.isnan(nll):
                            fold_nlls.append(nll)
                
                if len(fold_nlls) > 0:
                    best_nll_per_fold.append(min(fold_nlls))
            
            if len(best_nll_per_fold) == n_folds:
                avg_nll_per_fold = np.mean(best_nll_per_fold) #average across folds

                # get best repeat by total NLL for BIC computation
                valid_repeats = [r for r in fit_repeats if not math.isnan(r['total_nll'])]
                best_repeat = min(valid_repeats, key=lambda r: r['total_nll']) if valid_repeats else fit_repeats[0]

                # total test trials across folds (needed for BIC)
                n_trials_test = sum(
                    int(round(abs(f['perf']['ll_total'] / f['perf']['ll_avg'])))
                    for f in best_repeat['folds']
                    if f['perf'].get('ll_avg', 0) != 0
                )

                # get number of parameters from saved model
                model_obj = best_repeat['model'].model
                n_params = th.count_params(model_obj)
                model_name_with_params = '{} ({})'.format(model_name, n_params)

                # BIC = k*ln(n) + 2*NLL
                bic = (n_params * np.log(n_trials_test) + 2 * best_repeat['total_nll']
                       if n_trials_test > 0 else np.nan)

                cv_fit_mets.append({
                    'subjid': subj,
                    'model': model_name_with_params,
                    'avg_nll_per_fold': avg_nll_per_fold,
                    'n_folds': n_folds,
                    'n_params': n_params,
                    'bic': bic,
                })

cv_fit_mets = pd.DataFrame(cv_fit_mets)
print('CV fit metrics shape:', cv_fit_mets.shape)
print('Models found:', cv_fit_mets['model'].nunique())

# sort model names and subject ids alphabetically
cv_model_names = sorted(cv_fit_mets['model'].unique().tolist(), key=str.lower)
cv_subjids = sorted(cv_fit_mets['subjid'].unique().tolist())

# calculate percent difference from best model per subject
cv_fit_mets['diff_avg_nll'] = 0.0
cv_fit_mets['diff_bic'] = 0.0
for subj in cv_subjids:
    subj_sel = cv_fit_mets['subjid'] == subj
    subj_mets = cv_fit_mets[subj_sel]
    best_avg_nll = subj_mets['avg_nll_per_fold'].min()
    best_bic = subj_mets['bic'].min()
    cv_fit_mets.loc[subj_sel, 'diff_avg_nll'] = (subj_mets['avg_nll_per_fold'] - best_avg_nll) / best_avg_nll * 100
    cv_fit_mets.loc[subj_sel, 'diff_bic'] = (subj_mets['bic'] - best_bic) / best_bic * 100

# average percent difference across subjects per model
avg_cv_diffs = cv_fit_mets.groupby('model')['diff_avg_nll'].mean().reset_index()

ax_height = max(len(cv_model_names)/5, 3)

# plot 1 - raw average NLL per fold, one dot per subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_height), layout='constrained')
sb.stripplot(cv_fit_mets, y='model', x='avg_nll_per_fold', hue='subjid',
             ax=ax, palette='colorblind', order=cv_model_names,
             hue_order=cv_subjids)
fig.suptitle('CV Model Performance - Average NLL per Fold')
ax.set_title('Average NLL per Fold')
ax.set_xlabel('Average NLL per Fold')
ax.set_ylabel('Model')
ax.legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# plot 2 - percent difference from best model, one dot per subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_height), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(cv_fit_mets, y='model', x='diff_avg_nll', hue='subjid',
             ax=ax, palette='colorblind', order=cv_model_names,
             hue_order=cv_subjids)
fig.suptitle('CV Model Performance - % Worse than Best Model per Subject')
ax.set_title('One dot per subject/reward rate')
ax.set_xlabel('% Worse Average NLL per Fold')
ax.set_ylabel('Model')
ax.legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# plot 3 - percent difference averaged across subjects, one dot per model
avg_cv_diffs_subj = cv_fit_mets.groupby('model')['diff_avg_nll'].mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(12, ax_height), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(avg_cv_diffs_subj, y='model', x='diff_avg_nll',
             ax=ax, order=cv_model_names, color='steelblue', size=8)

fig.suptitle('CV Model Performance - Average % Worse than Best Model')
ax.set_title('One dot per model (averaged across subjects)')
ax.set_xlabel('Avg % Worse Average NLL per Fold')
ax.set_ylabel('Model')
plt.show()

# plot 4 - percent difference in BIC from best model, one dot per subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_height), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(cv_fit_mets, y='model', x='diff_bic', hue='subjid',
             ax=ax, palette='colorblind', order=cv_model_names,
             hue_order=cv_subjids)
fig.suptitle('CV Model Performance - % Worse BIC than Best Model per Subject')
ax.set_title('One dot per subject/reward rate')
ax.set_xlabel('% Worse BIC')
ax.set_ylabel('Model')
ax.legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# plot 6 - percent difference in BIC averaged across subjects, one dot per model
avg_cv_bic_diffs = cv_fit_mets.groupby('model')['diff_bic'].mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(12, ax_height), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(avg_cv_bic_diffs, y='model', x='diff_bic',
             ax=ax, order=cv_model_names, color='steelblue', size=8)

fig.suptitle('CV Model Performance - Average % Worse BIC than Best Model')
ax.set_title('One dot per model (averaged across subjects)')
ax.set_xlabel('Avg % Worse BIC')
ax.set_ylabel('Model')
plt.show()

#%%
# Plot 2: Average across subjects — one dot per model
# ══════════════════════════════════════════════════════════
avg_cv_diffs_subj = cv_fit_mets.groupby('model')['diff_avg_nll'].mean().reset_index()
 
fig, ax = plt.subplots(1, 1, figsize=(10, ax_height), layout='constrained')
 
ax.axvline(0, color='#444441', linewidth=1.0, linestyle='--', alpha=0.5)
ax.set_axisbelow(True)
ax.grid(axis='x', color=GRID_COLOR, linewidth=0.8)
 
# Highlight best models in accent color, others in neutral
colors = [ACCENT if m == model_order[0] else '#5F5E5A' for m in model_order]
 
sb.stripplot(
    avg_cv_diffs_subj, y='model', x='diff_avg_nll',
    ax=ax, order=model_order, palette=colors,
    size=10, alpha=0.9, edgecolor='white', linewidth=0.8,
)
 
ax.set_xlabel('Average % worse than best model', fontweight='500')
ax.set_ylabel('')
ax.set_title('Cross-validated model performance averaged across subjects',
             fontweight='500', loc='left', pad=12)
 
# Annotate best model
best_model = model_order[0]
best_val = avg_cv_diffs_subj[avg_cv_diffs_subj['model'] == best_model]['diff_avg_nll'].values[0]
ax.annotate(
    'Best model',
    xy=(best_val, 0),
    xytext=(best_val + 1.5, -0.4),
    fontsize=10, color=ACCENT, fontweight='500',
    arrowprops=dict(arrowstyle='->', color=ACCENT, lw=0.8,
                    connectionstyle='arc3,rad=0.2'),
)
 
plt.savefig('cv_averaged.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('cv_averaged.pdf', bbox_inches='tight', facecolor='white')
plt.show()
#%%
#inspect model parameters 
import math

subj_to_inspect = '198 (75/10)'
model_to_inspect = 'Q/Persev (fixed) - All Alpha Shared, All K Fixed'
cv_key = model_to_inspect + '_cv'

# get the best fit repeat (lowest total NLL)
fit_repeats = all_models[subj_to_inspect][cv_key]
valid_repeats = [(i, r) for i, r in enumerate(fit_repeats) if not math.isnan(r['total_nll'])]
best_repeat_idx, best_repeat = min(valid_repeats, key=lambda x: x[1]['total_nll'])

print('Best repeat: {} | Total NLL: {:.3f}'.format(best_repeat_idx, best_repeat['total_nll']))

# get the model from the last fold of the best repeat
model = best_repeat['model'].model

print('\nModel parameters:')
for name, param in model.named_parameters():
    print('  {} : {:.4f}'.format(name, param.data.item()))


# %% Perseverative Alpha Analysis
# Examine fitted perseverative alpha values to assess whether the free parameter is needed

import math

# --- Part 1: extract fitted alpha_p from best CV repeat per subject/model ---
persev_alpha_rows = []
for subj in subjids:
    for model_name, fits in all_models[subj].items():
        if 'free alpha' not in model_name or not model_name.endswith('_cv'):
            continue
        if not isinstance(fits, list) or len(fits) == 0:
            continue
        # select best CV repeat by lowest total_nll
        valid_fits = [f for f in fits if isinstance(f, dict) and 'total_nll' in f
                      and not math.isnan(f['total_nll'])]
        if len(valid_fits) == 0:
            continue
        best_fit = min(valid_fits, key=lambda f: f['total_nll'])
        model_obj = best_fit['model'].model
        base_name = model_name[:-3]  # strip _cv suffix for display
        for agent in model_obj.agents:
            if isinstance(agent, agents.PerseverativeAgent):
                persev_alpha_rows.append({
                    'subjid': subj,
                    'model': base_name,
                    'alpha_p': agent.alpha.a.item(),
                })

persev_alpha_df = pd.DataFrame(persev_alpha_rows)
persev_model_names = sorted(persev_alpha_df['model'].unique().tolist(), key=str.lower)
persev_subjids = sorted(persev_alpha_df['subjid'].unique().tolist())

# plot: free alpha_p per model, one dot per subject
fig, ax = plt.subplots(1, 1, figsize=(12, max(len(persev_model_names) / 2.5, 3)), layout='constrained')
sb.stripplot(persev_alpha_df, y='model', x='alpha_p', hue='subjid',
             ax=ax, palette='colorblind', order=persev_model_names,
             hue_order=persev_subjids, size=8)
ax.axvline(0.5, color='gray', linestyle='--', linewidth=1.0, label='α=0.5')
fig.suptitle('Free Perseverative Alpha (Best Fit per Subject)', fontsize=14)
ax.set_xlabel('Perseverative Alpha', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# --- Part 2: compare CV NLL of free alpha vs fixed models ---
persev_cv_rows = []
for subj in subjids:
    for model_name in all_models[subj].keys():
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]
        if 'Persev' not in base_name:
            continue
        if 'free alpha' in base_name:
            alpha_type = 'free'
        elif '(fixed)' in base_name:
            alpha_type = 'fixed'
        else:
            continue

        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        n_folds = fit_repeats[0]['n_folds']
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for repeat in fit_repeats
                         for f in repeat['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) == n_folds:
            persev_cv_rows.append({
                'subjid': subj,
                'model': base_name,
                'alpha_type': alpha_type,
                'avg_nll_per_fold': np.mean(best_nll_per_fold),
            })

persev_cv_df = pd.DataFrame(persev_cv_rows)

# compute difference: free alpha NLL - fixed alpha NLL (negative = free alpha is better)
diff_rows = []
for subj in persev_cv_df['subjid'].unique():
    subj_data = persev_cv_df[persev_cv_df['subjid'] == subj]
    free_data = subj_data[subj_data['alpha_type'] == 'free']
    fixed_data = subj_data[subj_data['alpha_type'] == 'fixed']

    for _, free_row in free_data.iterrows():
        fixed_name = (free_row['model']
                      .replace('/Persev (free alpha)', '/Persev (fixed)')
                      .replace('Persev (free alpha)', 'Persev (fixed)'))
        matching_fixed = fixed_data[fixed_data['model'] == fixed_name]
        if len(matching_fixed) > 0:
            fixed_nll = matching_fixed.iloc[0]['avg_nll_per_fold']
            free_nll = free_row['avg_nll_per_fold']
            # use the free alpha model name as label (strip alpha type for compactness)
            pair_label = free_row['model']
            diff_rows.append({
                'subjid': subj,
                'model_pair': pair_label,
                'nll_diff': free_nll - fixed_nll,
                'pct_diff': (free_nll - fixed_nll) / fixed_nll * 100,
            })

persev_diff_df = pd.DataFrame(diff_rows)
model_pairs = sorted(persev_diff_df['model_pair'].unique().tolist(), key=str.lower)
diff_subjids = sorted(persev_diff_df['subjid'].unique().tolist())

# plot: % change in CV NLL from using free vs fixed perseverative alpha (per subject)
fig, ax = plt.subplots(1, 1, figsize=(12, max(len(model_pairs) / 2.5, 3)), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(persev_diff_df, y='model_pair', x='pct_diff', hue='subjid',
             ax=ax, palette='colorblind', order=model_pairs,
             hue_order=diff_subjids, size=8)
fig.suptitle('Free vs Fixed Perseverative Alpha: % Change in CV NLL', fontsize=14)
ax.set_title('Negative = free alpha improves fit, one dot per subject', fontsize=12)
ax.set_xlabel('% Change in Avg NLL per Fold (Free − Fixed)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(loc='upper right', fontsize=10, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# plot: same but averaged across subjects
avg_diff = persev_diff_df.groupby('model_pair')['pct_diff'].mean().reset_index()
fig, ax = plt.subplots(1, 1, figsize=(12, max(len(model_pairs) / 2.5, 3)), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(avg_diff, y='model_pair', x='pct_diff',
             ax=ax, order=model_pairs, color='steelblue', size=8)
fig.suptitle('Free vs Fixed Perseverative Alpha: Avg % Change in CV NLL (across subjects)', fontsize=14)
ax.set_title('One dot per model (averaged across subjects)', fontsize=12)
ax.set_xlabel('Avg % Change in Avg NLL per Fold (Free − Fixed)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)


# %% Fixed Model Comparison - Boxplots (Q and SI separately)
import math

# rebuild model names from CV keys, excluding free alpha models
fixed_model_names_cv = set()
for subj in subjids:
    for key in all_models[subj].keys():
        if key.endswith('_cv') and 'free alpha' not in key:
            fixed_model_names_cv.add(key[:-3])
fixed_model_names_cv = sorted(list(fixed_model_names_cv), key=str.lower)

# build per-fold NLL for fixed models only
fixed_cv_mets = []
for subj in subjids:
    for mn in fixed_model_names_cv:
        cv_key = mn + '_cv'

        if cv_key not in all_models[subj]:
            continue

        fit_repeats = all_models[subj][cv_key]
        if len(fit_repeats) == 0:
            continue

        n_folds = fit_repeats[0]['n_folds']
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for repeat in fit_repeats
                         for f in repeat['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) == n_folds:
            avg_nll_per_fold = np.mean(best_nll_per_fold)
            model_type = mn.split(' - ')[0].strip()

            fixed_cv_mets.append({
                'subjid': subj,
                'model': mn,
                'model_type': model_type,
                'avg_nll_per_fold': avg_nll_per_fold,
            })

fixed_cv_df = pd.DataFrame(fixed_cv_mets)

# calculate % difference from best fixed model per subject (within fixed models only)
fixed_cv_subjids = sorted(fixed_cv_df['subjid'].unique().tolist())
fixed_cv_df['diff_avg_nll'] = 0.0
for subj in fixed_cv_subjids:
    subj_sel = fixed_cv_df['subjid'] == subj
    subj_mets = fixed_cv_df[subj_sel]
    best_avg_nll = subj_mets['avg_nll_per_fold'].min()
    fixed_cv_df.loc[subj_sel, 'diff_avg_nll'] = (subj_mets['avg_nll_per_fold'] - best_avg_nll) / best_avg_nll * 100

# split into Q and SI model families, excluding specified models
exclude_models = [
    'Q - Same Alpha Only, K Fixed',
    'Q/Persev (fixed) - Same Alpha Only, K Fixed',
    'SI - Separate Same/Diff Evidence',
    'SI - Shared Evidence',
]

q_df = fixed_cv_df[fixed_cv_df['model_type'].str.startswith('Q') &
                   ~fixed_cv_df['model'].isin(exclude_models)]
si_df = fixed_cv_df[fixed_cv_df['model_type'].str.startswith('SI') &
                    ~fixed_cv_df['model'].isin(exclude_models)]

q_model_names = sorted(q_df['model'].unique().tolist(), key=str.lower)
si_model_names = sorted(si_df['model'].unique().tolist(), key=str.lower)

# compute shared x-axis limits across both plots
combined_max = max(q_df['diff_avg_nll'].max(), si_df['diff_avg_nll'].max())
combined_min = min(q_df['diff_avg_nll'].min(), si_df['diff_avg_nll'].min())
x_margin = (combined_max - combined_min) * 0.05
x_lim = (combined_min - x_margin, combined_max + x_margin)

# Q models boxplot
fig, ax = plt.subplots(1, 1, figsize=(14, max(len(q_model_names) / 1.5, 6)), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(q_df, y='model', x='diff_avg_nll', ax=ax, order=q_model_names, color='steelblue')
ax.set_xlim(x_lim)
fig.suptitle('Q Models - % Worse than Best Fixed Model', fontsize=18)
ax.set_xlabel('% Worse than Best Fixed Model (Avg NLL per Fold)', fontsize=16)
ax.set_ylabel('Model', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
plt.show()

# SI models boxplot
fig, ax = plt.subplots(1, 1, figsize=(14, max(len(si_model_names) / 1.5, 6)), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(si_df, y='model', x='diff_avg_nll', ax=ax, order=si_model_names, color='steelblue')
ax.set_xlim(x_lim)
fig.suptitle('SI Models - % Worse than Best Fixed Model', fontsize=18)
ax.set_xlabel('% Worse than Best Fixed Model (Avg NLL per Fold)', fontsize=16)
ax.set_ylabel('Model', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
plt.show()

# combined Q vs SI boxplot on the same scale
# recalculate % difference from the single best model across ALL fixed models per subject
fixed_cv_df['diff_avg_nll_combined'] = 0.0
for subj in fixed_cv_subjids:
    subj_sel = fixed_cv_df['subjid'] == subj
    subj_mets = fixed_cv_df[subj_sel]
    best_avg_nll = subj_mets['avg_nll_per_fold'].min()
    fixed_cv_df.loc[subj_sel, 'diff_avg_nll_combined'] = (subj_mets['avg_nll_per_fold'] - best_avg_nll) / best_avg_nll * 100

# keep only Q and SI families, add family label for hue
combined_df = fixed_cv_df[fixed_cv_df['model_type'].str.startswith('Q') |
                           fixed_cv_df['model_type'].str.startswith('SI')].copy()
combined_df['family'] = combined_df['model_type'].apply(lambda x: 'Q' if x.startswith('Q') else 'SI')
all_model_names = sorted(combined_df['model'].unique().tolist(), key=str.lower)

fig, ax = plt.subplots(1, 1, figsize=(14, max(len(all_model_names) / 1.5, 8)), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(combined_df, y='model', x='diff_avg_nll_combined', hue='family',
           ax=ax, order=all_model_names, palette={'Q': 'steelblue', 'SI': 'darkorange'}, legend=True)
fig.suptitle('Q vs SI Models - % Worse than Best Fixed Model (All Families)', fontsize=18)
ax.set_title('% difference relative to single best model across all subjects', fontsize=14)
ax.set_xlabel('% Worse than Best Fixed Model (Avg NLL per Fold)', fontsize=16)
ax.set_ylabel('Model', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
ax.legend(title='Family', fontsize=13, title_fontsize=13, loc='upper right')
plt.show()


# %% Example Trial Diagram - Q/Persev (fixed) All Alpha Shared, All K Fixed

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# --- fitted parameter values from subject 198 (75/10), best non-CV fit ---
alpha   = 0.71  # shared learning rate (alpha_same/diff_rew/unrew all equal)
alpha_p = 0.98  # fixed perseverative alpha

# before trial (example starting state)
Q_L  = 0.50;  Q_R  = 0.30
VP_L = 0.20;  VP_R = 0.10   # perseverative values (n_vals=2: separate for left/right)

# trial: choose Left, rewarded
# Q update: chosen(left) moves toward k_same_rew=1; unchosen(right) moves toward k_diff_rew=0
Q_L_new  = round(Q_L  + alpha   * (1.0 - Q_L),  2)
Q_R_new  = round(Q_R  + alpha   * (0.0 - Q_R),  2)
VP_L_new = round(VP_L + alpha_p * (1.0 - VP_L), 2)   # chose left -> input=1 for left
VP_R_new = round(VP_R + alpha_p * (0.0 - VP_R), 2)   # chose left -> input=0 for right

fig, ax = plt.subplots(figsize=(16, 8))
ax.set_xlim(0, 16)
ax.set_ylim(0, 8)
ax.axis('off')

# colors
C_BEFORE = '#D6E4F0'   # light blue
C_EVENT  = '#D5F5E3'   # light green
C_AFTER  = '#FDEBD0'   # light orange
C_EQ     = '#F5EEF8'   # light purple
C_TEXT   = '#1C2833'
C_ARROW  = '#2C3E50'

def draw_box(ax, x, y, w, h, color, title, lines, title_fs=14, line_fs=13):
    box = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.15',
                          facecolor=color, edgecolor='#7F8C8D', linewidth=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h - 0.3, title, ha='center', va='top',
            fontsize=title_fs, fontweight='bold', color=C_TEXT)
    for i, line in enumerate(lines):
        ax.text(x + w/2, y + h - 0.85 - i * 0.55, line, ha='center', va='top',
                fontsize=line_fs, color=C_TEXT, fontfamily='monospace')

def draw_arrow(ax, x1, x2, y):
    ax.annotate('', xy=(x2, y), xytext=(x1, y),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.5))

def draw_arrow_down(ax, x, y1, y2):
    ax.annotate('', xy=(x, y2), xytext=(x, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=2.5))

# --- Before Trial box ---
draw_box(ax, 0.3, 4.5, 3.5, 3.2, C_BEFORE, 'Before Trial t',
         [f'Q_L   = {Q_L:.2f}',
          f'Q_R   = {Q_R:.2f}',
          f'VP_L  = {VP_L:.2f}',
          f'VP_R  = {VP_R:.2f}'])

draw_arrow(ax, 3.8, 5.0, 6.1)

# --- Trial Event box ---
draw_box(ax, 5.0, 4.5, 3.8, 3.2, C_EVENT, 'Trial t',
         ['Choice:  Left Port',
          'Outcome: Rewarded ✓',
          f'α = {alpha},  αp = {alpha_p}'])

draw_arrow(ax, 8.8, 10.0, 6.1)

# --- After Trial box ---
draw_box(ax, 10.0, 4.5, 3.8, 3.2, C_AFTER, 'After Trial t',
         [f'Q_L   = {Q_L_new:.2f}  (+{Q_L_new-Q_L:.2f})',
          f'Q_R   = {Q_R_new:.2f}  ({Q_R_new-Q_R:.2f})',
          f'VP_L  = {VP_L_new:.2f}  (+{VP_L_new-VP_L:.2f})',
          f'VP_R  = {VP_R_new:.2f}  ({VP_R_new-VP_R:.2f})'])

# --- arrows down to equations ---
draw_arrow_down(ax, 4.95, 4.5, 3.7)
draw_arrow_down(ax, 11.9, 4.5, 3.7)

# --- Update Equations box ---
draw_box(ax, 0.3, 0.3, 15.2, 3.2, C_EQ, 'Update Equations',
         [f'Q_L  ← Q_L  + α × (k_same_rew − Q_L)  =  {Q_L:.2f} + {alpha} × (1.0 − {Q_L:.2f})  =  {Q_L_new:.2f}   [chosen, rewarded → target k=1]',
          f'Q_R  ← Q_R  + α × (k_diff_rew  − Q_R)  =  {Q_R:.2f} + {alpha} × (0.0 − {Q_R:.2f})  =  {Q_R_new:.2f}   [unchosen → target k=0, decays]',
          f'VP_L ← VP_L + αp × (1 − VP_L)          =  {VP_L:.2f} + {alpha_p} × (1.0 − {VP_L:.2f})  =  {VP_L_new:.2f}   [chose left → persev. value ↑]',
          f'VP_R ← VP_R + αp × (0 − VP_R)          =  {VP_R:.2f} + {alpha_p} × (0.0 − {VP_R:.2f})  =  {VP_R_new:.2f}   [did not choose right → persev. value ↓]'],
         title_fs=14, line_fs=11)

fig.suptitle('Example Trial — Q/Persev (fixed): All Alpha Shared, All K Fixed  [Subject 198 (75/10)]',
             fontsize=16, fontweight='bold', y=1.01)
plt.tight_layout()
plt.show()


# %% Fixed vs Free Perseverative Alpha - NLL Comparison (Q and SI families)
import math

# build CV NLL for all persev models (both fixed and free alpha) across Q and SI families
persev_nll_rows = []
for subj in subjids:
    for model_name in all_models[subj].keys():
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]

        # only include models with Persev in the name, Q or SI family
        if 'Persev' not in base_name:
            continue
        family = base_name.split('/')[0].strip()
        if family not in ('Q', 'SI'):
            continue

        # determine alpha type
        if 'free alpha' in base_name:
            alpha_type = 'Free'
        elif '(fixed)' in base_name:
            alpha_type = 'Fixed'
        else:
            continue

        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        n_folds = fit_repeats[0]['n_folds']
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for repeat in fit_repeats
                         for f in repeat['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) == n_folds:
            # strip alpha type label from model name for cleaner y-axis
            label = (base_name
                     .replace('/Persev (free alpha)', '/Persev')
                     .replace('/Persev (fixed)', '/Persev'))
            persev_nll_rows.append({
                'subjid': subj,
                'model': label,
                'alpha_type': alpha_type,
                'family': family,
                'avg_nll_per_fold': np.mean(best_nll_per_fold),
            })

persev_nll_df = pd.DataFrame(persev_nll_rows)

# shared x-axis limits across both plots
x_min = persev_nll_df['avg_nll_per_fold'].min()
x_max = persev_nll_df['avg_nll_per_fold'].max()
x_margin = (x_max - x_min) * 0.05
x_lim = (x_min - x_margin, x_max + x_margin)

for family in ('Q', 'SI'):
    fam_df = persev_nll_df[persev_nll_df['family'] == family]
    model_names_sorted = sorted(fam_df['model'].unique().tolist(), key=str.lower)

    fig, ax = plt.subplots(1, 1, figsize=(14, max(len(model_names_sorted) / 1.2, 6)),
                           layout='constrained')
    plot_utils.plot_x0line(ax=ax)
    sb.boxplot(fam_df, y='model', x='avg_nll_per_fold', hue='alpha_type',
               ax=ax, order=model_names_sorted,
               palette={'Fixed': 'steelblue', 'Free': 'darkorange'},
               hue_order=['Fixed', 'Free'])
    ax.set_xlim(x_lim)
    fig.suptitle(f'{family} Models — CV NLL: Fixed vs Free Perseverative Alpha', fontsize=18)
    ax.set_title('One box per model, distribution across subjects', fontsize=14)
    ax.set_xlabel('Avg NLL per Fold', fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(title='Alpha Type', fontsize=13, title_fontsize=13, loc='upper right')
    plt.show()


# %% With vs Without Perseveration - NLL Comparison (Q and SI families)
import math

# build CV NLL for all Q and SI models, label whether they have perseveration or not
persev_comp_rows = []
for subj in subjids:
    for model_name in all_models[subj].keys():
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]

        # only Q and SI families — split on '/' first, then on ' ' to get just the prefix
        family = base_name.split('/')[0].split(' ')[0].strip()
        if family not in ('Q', 'SI'):
            continue

        # determine perseveration type
        if 'Persev (fixed)' in base_name:
            persev_type = 'With Persev (fixed)'
        elif 'Persev (free alpha)' in base_name:
            # use best fixed alpha only for this comparison
            continue
        elif 'Persev' not in base_name:
            persev_type = 'No Persev'
        else:
            continue

        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        n_folds = fit_repeats[0]['n_folds']
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for repeat in fit_repeats
                         for f in repeat['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) == n_folds:
            # strip persev label for cleaner y-axis matching
            label = base_name.replace('/Persev (fixed)', '')
            persev_comp_rows.append({
                'subjid': subj,
                'model': label,
                'persev_type': persev_type,
                'family': family,
                'avg_nll_per_fold': np.mean(best_nll_per_fold),
            })

persev_comp_df = pd.DataFrame(persev_comp_rows)

# compute % difference: (with_persev - no_persev) / no_persev * 100
# negative = perseveration improves fit
persev_diff_rows = []
for subj in persev_comp_df['subjid'].unique():
    for family in ('Q', 'SI'):
        subj_fam = persev_comp_df[(persev_comp_df['subjid'] == subj) &
                                   (persev_comp_df['family'] == family)]
        with_df  = subj_fam[subj_fam['persev_type'] == 'With Persev (fixed)']
        without_df = subj_fam[subj_fam['persev_type'] == 'No Persev']

        for _, row in with_df.iterrows():
            match = without_df[without_df['model'] == row['model']]
            if len(match) > 0:
                nll_no_persev   = match.iloc[0]['avg_nll_per_fold']
                nll_with_persev = row['avg_nll_per_fold']
                pct_diff = (nll_with_persev - nll_no_persev) / nll_no_persev * 100
                persev_diff_rows.append({
                    'subjid': subj,
                    'model': row['model'],
                    'family': family,
                    'pct_diff': pct_diff,
                })

persev_diff_df = pd.DataFrame(persev_diff_rows)

# shared x-axis limits across both plots
x_min = persev_diff_df['pct_diff'].min()
x_max = persev_diff_df['pct_diff'].max()
x_margin = (x_max - x_min) * 0.05
x_lim = (x_min - x_margin, x_max + x_margin)

for family in ('Q', 'SI'):
    fam_df = persev_diff_df[persev_diff_df['family'] == family]
    matched_models = sorted(fam_df['model'].unique().tolist(), key=str.lower)

    fig, ax = plt.subplots(1, 1, figsize=(14, max(len(matched_models) / 1.2, 6)),
                           layout='constrained')
    plot_utils.plot_x0line(ax=ax)
    sb.boxplot(fam_df, y='model', x='pct_diff',
               ax=ax, order=matched_models, color='steelblue')
    ax.set_xlim(x_lim)
    fig.suptitle(f'{family} Models — % Change in CV NLL: With vs Without Perseveration', fontsize=18)
    ax.set_title('Negative = perseveration improves fit', fontsize=14)
    ax.set_xlabel('% Change in Avg NLL per Fold (With Persev − No Persev)', fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    plt.show()


# %% With vs Without Perseveration — 5 Focal Models
import math

focal_persev_models_keep = [
    'Q - All Alpha Shared, All K Fixed',
    'Q - Alpha Rew/Unrew Shared, All K Fixed',
    'SI - Free Same/Diff Rew Evidence',
    'SI - Shared Rew Evidence, Unrew Fixed 0',
    'Q+SI - All Alpha Shared, All K Fixed + Shared Rew Evidence, Unrew Fixed 0',
]

# Build CV NLL for focal models with and without fixed persev
focal_persev_rows = []
for subj in subjids:
    for model_name in all_models[subj].keys():
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]

        if 'Persev (fixed)' in base_name:
            persev_type = 'With Persev (fixed)'
            label = base_name.replace('/Persev (fixed)', '')
        elif 'Persev' not in base_name:
            persev_type = 'No Persev'
            label = base_name
        else:
            continue  # skip free alpha persev

        if label not in focal_persev_models_keep:
            continue

        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        n_folds = fit_repeats[0]['n_folds']
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for repeat in fit_repeats
                         for f in repeat['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) == n_folds:
            valid_repeats = [r for r in fit_repeats if not math.isnan(r['total_nll'])]
            best_repeat = min(valid_repeats, key=lambda r: r['total_nll']) if valid_repeats else fit_repeats[0]
            n_trials_test = sum(
                int(round(abs(f['perf']['ll_total'] / f['perf']['ll_avg'])))
                for f in best_repeat['folds']
                if f['perf'].get('ll_avg', 0) != 0
            )
            n_params = th.count_params(best_repeat['model'].model)
            bic = (n_params * np.log(n_trials_test) + 2 * best_repeat['total_nll']
                   if n_trials_test > 0 else np.nan)

            focal_persev_rows.append({
                'subjid': subj,
                'model': label,
                'persev_type': persev_type,
                'avg_nll_per_fold': np.mean(best_nll_per_fold),
                'bic': bic,
            })

focal_persev_df = pd.DataFrame(focal_persev_rows)

# compute % difference: (with_persev - no_persev) / no_persev * 100
focal_persev_diff_rows = []
for subj in focal_persev_df['subjid'].unique():
    subj_data = focal_persev_df[focal_persev_df['subjid'] == subj]
    with_df    = subj_data[subj_data['persev_type'] == 'With Persev (fixed)']
    without_df = subj_data[subj_data['persev_type'] == 'No Persev']

    for _, row in with_df.iterrows():
        match = without_df[without_df['model'] == row['model']]
        if len(match) > 0:
            nll_no   = match.iloc[0]['avg_nll_per_fold']
            nll_with = row['avg_nll_per_fold']
            focal_persev_diff_rows.append({
                'subjid': subj,
                'model': row['model'],
                'pct_diff': (nll_with - nll_no) / nll_no * 100,
            })

focal_persev_diff_df = pd.DataFrame(focal_persev_diff_rows)
focal_persev_subjids = sorted(focal_persev_diff_df['subjid'].unique().tolist())

# compute BIC difference: (with_persev - no_persev)
focal_persev_bic_diff_rows = []
for subj in focal_persev_df['subjid'].unique():
    subj_data = focal_persev_df[focal_persev_df['subjid'] == subj]
    with_df    = subj_data[subj_data['persev_type'] == 'With Persev (fixed)']
    without_df = subj_data[subj_data['persev_type'] == 'No Persev']

    for _, row in with_df.iterrows():
        match = without_df[without_df['model'] == row['model']]
        if len(match) > 0 and not np.isnan(row['bic']) and not np.isnan(match.iloc[0]['bic']):
            focal_persev_bic_diff_rows.append({
                'subjid': subj,
                'model': row['model'],
                'bic_diff': row['bic'] - match.iloc[0]['bic'],
            })

focal_persev_bic_diff_df = pd.DataFrame(focal_persev_bic_diff_rows)

# use focal_persev_models_keep order so models appear in a consistent sequence
focal_persev_order = [m for m in focal_persev_models_keep
                      if m in focal_persev_diff_df['model'].unique()
                      and not m.startswith('Q+SI')]

ax_h_fp = max(len(focal_persev_order) / 1.5, 4)

# Plot 1: one dot per subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_persev_diff_df, y='model', x='pct_diff', hue='subjid',
             ax=ax, palette='colorblind', order=focal_persev_order,
             hue_order=focal_persev_subjids, size=8)
fig.suptitle('With vs Without Perseveration — 5 Focal Models', fontsize=16)
ax.set_title('Negative = perseveration improves fit (one dot per subject)', fontsize=12)
ax.set_xlabel('% Change in Avg NLL per Fold (With Persev − No Persev)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(title='Subject', loc='upper right', fontsize=9,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# Plot 2: averaged across subjects
focal_persev_avg = focal_persev_diff_df.groupby('model')['pct_diff'].mean().reset_index()

fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_persev_avg, y='model', x='pct_diff',
             ax=ax, order=focal_persev_order, color='steelblue', size=10)
fig.suptitle('With vs Without Perseveration — 5 Focal Models', fontsize=16)
ax.set_title('Averaged across subjects (one dot per model)', fontsize=12)
ax.set_xlabel('Avg % Change in Avg NLL per Fold (With Persev − No Persev)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
plt.show()

# Plot 3: boxplot — all 5 focal models in one plot
fig, ax = plt.subplots(1, 1, figsize=(14, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(focal_persev_diff_df, y='model', x='pct_diff',
           ax=ax, order=focal_persev_order, color='steelblue')
fig.suptitle('5 Focal Models — % Change in CV NLL: With vs Without Perseveration', fontsize=16)
ax.set_title('Negative = perseveration improves fit', fontsize=13)
ax.set_xlabel('% Change in Avg NLL per Fold (With Persev − No Persev)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
plt.show()

# Plot 4: BIC boxplot — all focal models
bic_order = [m for m in focal_persev_order if m in focal_persev_bic_diff_df['model'].unique()]
fig, ax = plt.subplots(1, 1, figsize=(14, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(focal_persev_bic_diff_df, y='model', x='bic_diff',
           ax=ax, order=bic_order, color='steelblue')
fig.suptitle('Focal Models — BIC Change: With vs Without Perseveration', fontsize=16)
ax.set_title('Negative = perseveration improves fit', fontsize=13)
ax.set_xlabel('BIC Change (With Persev − No Persev)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
plt.show()

# Plot 5: BIC parsimony ranking — best BIC per model (min across with/without persev)
# for each subject, take the lower BIC variant of each model, then compute % worse than best
focal_best_bic_rows = []
for subj in focal_persev_df['subjid'].unique():
    subj_data = focal_persev_df[focal_persev_df['subjid'] == subj]
    for model in subj_data['model'].unique():
        model_rows = subj_data[subj_data['model'] == model]
        best_bic = model_rows['bic'].min()
        if not np.isnan(best_bic):
            focal_best_bic_rows.append({'subjid': subj, 'model': model, 'best_bic': best_bic})

focal_best_bic_df = pd.DataFrame(focal_best_bic_rows)

# % worse than lowest BIC model per subject
for subj in focal_best_bic_df['subjid'].unique():
    sel = focal_best_bic_df['subjid'] == subj
    best = focal_best_bic_df.loc[sel, 'best_bic'].min()
    focal_best_bic_df.loc[sel, 'pct_worse_bic'] = (focal_best_bic_df.loc[sel, 'best_bic'] - best) / abs(best) * 100

bic_rank_order = [m for m in focal_persev_models_keep
                  if not m.startswith('Q+SI') and m in focal_best_bic_df['model'].unique()]

# find the overall winner (lowest avg % worse BIC)
avg_pct_worse = focal_best_bic_df.groupby('model')['pct_worse_bic'].mean()
best_model = avg_pct_worse.idxmin()

fig, ax = plt.subplots(1, 1, figsize=(14, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(focal_best_bic_df, y='model', x='pct_worse_bic',
           ax=ax, order=bic_rank_order, color='steelblue')
sb.stripplot(focal_best_bic_df, y='model', x='pct_worse_bic',
             ax=ax, order=bic_rank_order, color='black', size=5, alpha=0.6)
fig.suptitle('Focal Models — Most Parsimonious Model (Best BIC)', fontsize=16)
ax.set_title('Most parsimonious: "{}"\n(best BIC per model = min across with/without persev)'.format(best_model), fontsize=11)
ax.set_xlabel('% Worse BIC than Best Model per Subject', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
plt.show()

# Plot 6: raw BIC values — with vs without perseveration, one panel per model
bic_raw_df = focal_persev_df[focal_persev_df['model'].isin(bic_rank_order)].copy()
persev_order_labels = ['No Persev', 'With Persev (fixed)']
n_models = len(bic_rank_order)
fig, axs = plt.subplots(1, n_models, figsize=(4 * n_models, 4), layout='constrained')
if n_models == 1:
    axs = [axs]
fig.suptitle('Raw BIC: With vs Without Perseveration', fontsize=15)
for ax, model in zip(axs, bic_rank_order):
    model_data = bic_raw_df[bic_raw_df['model'] == model]
    sb.boxplot(model_data, x='persev_type', y='bic', order=persev_order_labels,
               ax=ax, palette={'No Persev': 'steelblue', 'With Persev (fixed)': 'darkorange'})
    sb.stripplot(model_data, x='persev_type', y='bic', order=persev_order_labels,
                 ax=ax, color='black', size=5, alpha=0.6)
    ax.set_title(model, fontsize=9, wrap=True)
    ax.set_xlabel('')
    ax.set_ylabel('BIC' if ax == axs[0] else '')
    ax.tick_params(axis='x', labelsize=9)
plt.show()

# %% With vs Without Perseveration — 5 Focal Models, by Reward Condition
import re

def _extract_rew_cond(subjid_str):
    m = re.search(r'(75/10|50/10)', str(subjid_str))
    return m.group(1) if m else 'Unknown'

focal_persev_diff_df['rew_cond'] = focal_persev_diff_df['subjid'].apply(_extract_rew_cond)

rew_order = ['75/10', '50/10']

# average per model per reward condition
focal_persev_avg_rew = (focal_persev_diff_df
                        .groupby(['model', 'rew_cond'])['pct_diff']
                        .mean().reset_index())

# Plot 1: one dot per subject, colored by reward condition
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_persev_diff_df, y='model', x='pct_diff', hue='rew_cond',
             ax=ax, palette='colorblind', order=focal_persev_order,
             hue_order=rew_order, size=8, dodge=True)
fig.suptitle('With vs Without Perseveration — 5 Focal Models by Reward Condition', fontsize=16)
ax.set_title('Negative = perseveration improves fit (one dot per subject)', fontsize=12)
ax.set_xlabel('% Change in Avg NLL per Fold (With Persev − No Persev)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(title='Reward Condition', loc='upper right', fontsize=10,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# Plot 2: averaged across subjects, one dot per model per reward condition
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_fp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_persev_avg_rew, y='model', x='pct_diff', hue='rew_cond',
             ax=ax, palette='colorblind', order=focal_persev_order,
             hue_order=rew_order, size=10, dodge=True)
fig.suptitle('With vs Without Perseveration — 5 Focal Models by Reward Condition', fontsize=16)
ax.set_title('Averaged across subjects (one dot per model per condition)', fontsize=12)
ax.set_xlabel('Avg % Change in Avg NLL per Fold (With Persev − No Persev)', fontsize=13)
ax.set_ylabel('Model', fontsize=13)
ax.tick_params(axis='both', labelsize=12)
ax.legend(title='Reward Condition', loc='upper right', fontsize=10,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% Fixed vs Free Perseverative Alpha - % Difference in NLL per Individual Subject
# reuses persev_nll_df built in the Fixed vs Free Perseverative Alpha cell above

# compute % difference: (free - fixed) / fixed * 100 per subject per model
free_fixed_diff_rows = []
for subj in persev_nll_df['subjid'].unique():
    for family in ('Q', 'SI'):
        subj_fam = persev_nll_df[(persev_nll_df['subjid'] == subj) &
                                  (persev_nll_df['family'] == family)]
        free_data  = subj_fam[subj_fam['alpha_type'] == 'Free']
        fixed_data = subj_fam[subj_fam['alpha_type'] == 'Fixed']

        for _, free_row in free_data.iterrows():
            match = fixed_data[fixed_data['model'] == free_row['model']]
            if len(match) > 0:
                nll_fixed = match.iloc[0]['avg_nll_per_fold']
                nll_free  = free_row['avg_nll_per_fold']
                pct_diff  = (nll_free - nll_fixed) / nll_fixed * 100
                free_fixed_diff_rows.append({
                    'subjid': subj,
                    'model': free_row['model'],
                    'family': family,
                    'pct_diff': pct_diff,
                })

free_fixed_diff_df = pd.DataFrame(free_fixed_diff_rows)

# shared x-axis limits across both plots
x_min = free_fixed_diff_df['pct_diff'].min()
x_max = free_fixed_diff_df['pct_diff'].max()
x_margin = (x_max - x_min) * 0.05
x_lim = (x_min - x_margin, x_max + x_margin)

for family in ('Q', 'SI'):
    fam_df = free_fixed_diff_df[free_fixed_diff_df['family'] == family]
    model_names_sorted = sorted(fam_df['model'].unique().tolist(), key=str.lower)
    subj_order = sorted(fam_df['subjid'].unique().tolist())

    fig, ax = plt.subplots(1, 1, figsize=(14, max(len(model_names_sorted) / 1.2, 6)),
                           layout='constrained')
    plot_utils.plot_x0line(ax=ax)
    sb.stripplot(fam_df, y='model', x='pct_diff', hue='subjid',
                 ax=ax, order=model_names_sorted, hue_order=subj_order,
                 palette='colorblind', size=9)
    ax.set_xlim(x_lim)
    fig.suptitle(f'{family} Models — % Difference in CV NLL: Free vs Fixed Perseverative Alpha',
                 fontsize=18)
    ax.set_title('Negative = free alpha improves fit, one dot per subject', fontsize=14)
    ax.set_xlabel('% Change in Avg NLL per Fold (Free − Fixed)', fontsize=16)
    ax.set_ylabel('Model', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend(title='Subject', fontsize=11, title_fontsize=12,
              loc='upper right', bbox_to_anchor=(1.14, 1.))
    plt.show()

# %% Compare fitted parameters: Fixed vs Free Perseverative Alpha
# Extracts Q-learning alphas, perseverative alpha, and beta weights
# from the best CV repeat for each subject x model pair

param_rows = []
for subj, subj_models in all_models.items():
    cv_keys = [k for k in subj_models.keys() if k.endswith('_cv')]
    for cv_key in cv_keys:
        base_name = cv_key[:-3]

        # only process models that have a perseverative component (fixed or free)
        if 'Persev' not in base_name:
            continue

        if 'free alpha' in base_name:
            alpha_type = 'Free'
        elif 'fixed' in base_name:
            alpha_type = 'Fixed'
        else:
            continue

        family = base_name.split('/')[0].split(' ')[0].strip()

        fits = subj_models[cv_key]
        valid_fits = [f for f in fits if isinstance(f, dict) and 'total_nll' in f
                      and not math.isnan(f['total_nll'])]
        if len(valid_fits) == 0:
            continue
        best_fit = min(valid_fits, key=lambda f: f['total_nll'])
        model_obj = best_fit['model'].model  # SummationModule

        row = {'subjid': subj, 'model': base_name, 'alpha_type': alpha_type, 'family': family}

        # extract per-agent parameters
        for agent in model_obj.agents:
            if isinstance(agent, agents.PerseverativeAgent):
                row['alpha_p'] = agent.alpha.a.item()
            elif isinstance(agent, agents.QValueAgent):
                row['alpha_same_rew']   = agent.alpha_same_rew.a.item()
                row['alpha_same_unrew'] = agent.alpha_same_unrew.a.item()
                row['alpha_diff_rew']   = agent.alpha_diff_rew.a.item()
                row['alpha_diff_unrew'] = agent.alpha_diff_unrew.a.item()

        # extract beta weights (one per agent): model_obj.beta.weight shape = (1, n_agents)
        betas = model_obj.beta.weight.detach().view(-1).tolist()
        for i, b in enumerate(betas):
            row[f'beta_{i}'] = b

        param_rows.append(row)

param_df = pd.DataFrame(param_rows)

# ---- print table: for each subject, compare fixed vs free parameters side-by-side ----
param_cols = ['alpha_p', 'alpha_same_rew', 'alpha_same_unrew', 'alpha_diff_rew', 'alpha_diff_unrew',
              'beta_0', 'beta_1']
param_cols_present = [c for c in param_cols if c in param_df.columns]

# build a short label: strip '/Persev (fixed)' / '/Persev (free alpha)' so fixed & free share same base
def _base_label(name):
    return (name.replace('/Persev (free alpha)', '/Persev')
                .replace('/Persev (fixed)', '/Persev'))

param_df['model_base'] = param_df['model'].apply(_base_label)

print('\n=== Fixed vs Free: Parameter Comparison (best CV repeat) ===\n')
for family in ('Q', 'SI'):
    fam_df = param_df[param_df['family'] == family]
    for base in sorted(fam_df['model_base'].unique()):
        subset = fam_df[fam_df['model_base'] == base].sort_values(['subjid', 'alpha_type'])
        if len(subset) == 0:
            continue
        print(f'--- {base} ---')
        display_cols = ['subjid', 'alpha_type'] + param_cols_present
        display_cols = [c for c in display_cols if c in subset.columns]
        print(subset[display_cols].to_string(index=False, float_format='{:.3f}'.format))
        print()

# ---- scatter: Q-learning alpha (same-rewarded) fixed vs free, one dot per subject per model ----
for family in ('Q', 'SI'):
    fam_df = param_df[param_df['family'] == family].copy()
    if 'alpha_same_rew' not in fam_df.columns:
        continue

    fixed_df = fam_df[fam_df['alpha_type'] == 'Fixed'][['subjid', 'model_base', 'alpha_same_rew']].rename(
        columns={'alpha_same_rew': 'alpha_same_rew_fixed'})
    free_df  = fam_df[fam_df['alpha_type'] == 'Free'][['subjid', 'model_base', 'alpha_same_rew']].rename(
        columns={'alpha_same_rew': 'alpha_same_rew_free'})
    merged = pd.merge(fixed_df, free_df, on=['subjid', 'model_base'])

    if len(merged) == 0:
        continue

    model_bases = sorted(merged['model_base'].unique())
    n_models = len(model_bases)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), layout='constrained')
    if n_models == 1:
        axes = [axes]

    subj_order = sorted(merged['subjid'].unique().tolist())
    palette = sb.color_palette('colorblind', n_colors=len(subj_order))
    subj_color = {s: palette[i] for i, s in enumerate(subj_order)}

    for ax, base in zip(axes, model_bases):
        sub = merged[merged['model_base'] == base]
        for _, r in sub.iterrows():
            ax.scatter(r['alpha_same_rew_fixed'], r['alpha_same_rew_free'],
                       color=subj_color[r['subjid']], s=80, zorder=3,
                       label=str(r['subjid']))
        lo = min(sub['alpha_same_rew_fixed'].min(), sub['alpha_same_rew_free'].min()) - 0.05
        hi = max(sub['alpha_same_rew_fixed'].max(), sub['alpha_same_rew_free'].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, zorder=1)  # identity line
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel('α_same_rew (fixed α_p)', fontsize=13)
        ax.set_ylabel('α_same_rew (free α_p)', fontsize=13)
        ax.set_title(base, fontsize=12)
        ax.tick_params(labelsize=12)

    # deduplicated legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=subj_color[s],
                           markersize=9, label=str(s)) for s in subj_order]
    axes[-1].legend(handles=handles, title='Subject', fontsize=10, title_fontsize=11,
                    loc='upper left')

    fig.suptitle(f'{family} Models — Q-learning α_same_rew: Fixed vs Free α_p', fontsize=15)
    plt.show()

# ---- perseverative alpha distribution (free models only) ----
free_param_df = param_df[param_df['alpha_type'] == 'Free']
if 'alpha_p' in free_param_df.columns and len(free_param_df) > 0:
    for family in ('Q', 'SI'):
        fam_df = free_param_df[free_param_df['family'] == family]
        if len(fam_df) == 0:
            continue
        model_names_sorted = sorted(fam_df['model_base'].unique().tolist(), key=str.lower)
        subj_order = sorted(fam_df['subjid'].unique().tolist())
        fig, ax = plt.subplots(1, 1, figsize=(10, max(len(model_names_sorted) / 1.5, 4)),
                               layout='constrained')
        sb.stripplot(fam_df, y='model_base', x='alpha_p', hue='subjid',
                     ax=ax, order=model_names_sorted, hue_order=subj_order,
                     palette='colorblind', size=9)
        ax.set_xlim(-0.05, 1.05)
        ax.axvline(x=0.98, color='gray', linestyle='--', lw=1, label='fixed α_p=0.98')
        fig.suptitle(f'{family} Models — Fitted Perseverative Alpha (Free Models)', fontsize=15)
        ax.set_xlabel('Perseverative Alpha (α_p)', fontsize=14)
        ax.set_ylabel('Model', fontsize=14)
        ax.tick_params(labelsize=13)
        ax.legend(title='Subject', fontsize=10, title_fontsize=11,
                  loc='upper right', bbox_to_anchor=(1.15, 1.))
        plt.show()

# %% Matched LL Comparison: Fixed vs Free Perseverative Agent by Reward Condition
import re
import math

def _extract_reward_cond(subjid_str):
    m = re.search(r'\((\d+/\d+)\)', str(subjid_str))
    return m.group(1) if m else 'unknown'

# collect per-subject, per-model avg NLL per fold for fixed and free persev models
matched_ll_rows = []
for subj in subjids:
    reward_cond = _extract_reward_cond(subj)
    for model_name in all_models[subj].keys():
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]
        if 'Persev' not in base_name:
            continue
        family = base_name.split('/')[0].strip()
        if family not in ('Q', 'SI'):
            continue
        if 'free alpha' in base_name:
            alpha_type = 'Free'
        elif '(fixed)' in base_name:
            alpha_type = 'Fixed'
        else:
            continue

        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        n_folds = fit_repeats[0]['n_folds']
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for repeat in fit_repeats
                         for f in repeat['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) == n_folds:
            label = (base_name
                     .replace('/Persev (free alpha)', '/Persev')
                     .replace('/Persev (fixed)', '/Persev'))
            matched_ll_rows.append({
                'subjid': subj,
                'reward_cond': reward_cond,
                'model': label,
                'alpha_type': alpha_type,
                'family': family,
                'avg_nll_per_fold': np.mean(best_nll_per_fold),
            })

matched_ll_df = pd.DataFrame(matched_ll_rows)

# compute matched % NLL difference (free − fixed) per subject per model
# negative = free perseverative alpha improves fit
matched_diff_rows = []
for subj in matched_ll_df['subjid'].unique():
    reward_cond = _extract_reward_cond(subj)
    for family in ('Q', 'SI'):
        subj_fam = matched_ll_df[(matched_ll_df['subjid'] == subj) &
                                  (matched_ll_df['family'] == family)]
        free_data  = subj_fam[subj_fam['alpha_type'] == 'Free']
        fixed_data = subj_fam[subj_fam['alpha_type'] == 'Fixed']

        for _, free_row in free_data.iterrows():
            match_fixed = fixed_data[fixed_data['model'] == free_row['model']]
            if len(match_fixed) > 0:
                nll_fixed = match_fixed.iloc[0]['avg_nll_per_fold']
                nll_free  = free_row['avg_nll_per_fold']
                matched_diff_rows.append({
                    'subjid': subj,
                    'reward_cond': reward_cond,
                    'model': free_row['model'],
                    'family': family,
                    'nll_diff': nll_free - nll_fixed,
                    'pct_diff': (nll_free - nll_fixed) / nll_fixed * 100,
                })

matched_diff_df = pd.DataFrame(matched_diff_rows)

# --- Plot 1: scatter fixed NLL vs free NLL, colored by reward condition ---
# points below unity line = free perseverative alpha fits better
for family in ('Q', 'SI'):
    fam_df = matched_ll_df[matched_ll_df['family'] == family]
    model_names_fam = sorted(fam_df['model'].unique().tolist(), key=str.lower)

    for model_label in model_names_fam:
        model_data = fam_df[fam_df['model'] == model_label]
        fixed_pts = model_data[model_data['alpha_type'] == 'Fixed'][
            ['subjid', 'reward_cond', 'avg_nll_per_fold']].rename(
            columns={'avg_nll_per_fold': 'nll_fixed'})
        free_pts = model_data[model_data['alpha_type'] == 'Free'][
            ['subjid', 'avg_nll_per_fold']].rename(
            columns={'avg_nll_per_fold': 'nll_free'})
        merged = pd.merge(fixed_pts, free_pts, on='subjid')
        if len(merged) == 0:
            continue

        reward_conds = sorted(merged['reward_cond'].unique())
        palette = sb.color_palette('colorblind', n_colors=len(reward_conds))
        cond_color = {c: palette[i] for i, c in enumerate(reward_conds)}

        fig, ax = plt.subplots(1, 1, figsize=(6, 5), layout='constrained')
        for _, row in merged.iterrows():
            ax.scatter(row['nll_fixed'], row['nll_free'],
                       color=cond_color[row['reward_cond']], s=90, zorder=3,
                       label=row['reward_cond'])

        lo = min(merged['nll_fixed'].min(), merged['nll_free'].min()) * 0.99
        hi = max(merged['nll_fixed'].max(), merged['nll_free'].max()) * 1.01
        ax.plot([lo, hi], [lo, hi], 'k--', lw=1, label='unity')
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel('Fixed α_p — Avg NLL per Fold', fontsize=13)
        ax.set_ylabel('Free α_p — Avg NLL per Fold', fontsize=13)
        ax.set_title(model_label, fontsize=12)

        seen = set()
        handles_leg, labels_leg = [], []
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles_leg.append(h)
                labels_leg.append(l)
        ax.legend(handles_leg, labels_leg, title='Reward Cond', fontsize=11, title_fontsize=12)
        fig.suptitle(f'{family} — Matched LL: Fixed vs Free Perseverative Agent', fontsize=14)
        plt.show()

# --- Plot 2: % NLL difference split by reward condition, one column per model ---
for family in ('Q', 'SI'):
    fam_diff = matched_diff_df[matched_diff_df['family'] == family]
    if len(fam_diff) == 0:
        continue
    model_names_fam = sorted(fam_diff['model'].unique().tolist(), key=str.lower)
    reward_conds = sorted(fam_diff['reward_cond'].unique())

    n_models = len(model_names_fam)
    fig, axs = plt.subplots(1, n_models, figsize=(5 * n_models, 5),
                            layout='constrained', squeeze=False)
    fig.suptitle(f'{family} — Matched NLL Difference: Free − Fixed Perseverative Agent\n'
                 'Negative = free α_p improves fit', fontsize=14)

    for j, model_label in enumerate(model_names_fam):
        ax = axs[0, j]
        model_diff = fam_diff[fam_diff['model'] == model_label]
        plot_utils.plot_x0line(ax=ax)
        sb.stripplot(model_diff, x='reward_cond', y='pct_diff', hue='reward_cond',
                     ax=ax, order=reward_conds, palette='colorblind', size=9, legend=False)
        ax.set_title(model_label, fontsize=11)
        ax.set_xlabel('Reward Condition', fontsize=12)
        ax.set_ylabel('% Change NLL (Free − Fixed)' if j == 0 else '', fontsize=12)
        ax.tick_params(labelsize=11)
    plt.show()

# --- Plot 3: all models together, reward condition as dodge hue ---
for family in ('Q', 'SI'):
    fam_diff = matched_diff_df[matched_diff_df['family'] == family]
    if len(fam_diff) == 0:
        continue
    model_names_fam = sorted(fam_diff['model'].unique().tolist(), key=str.lower)
    reward_conds = sorted(fam_diff['reward_cond'].unique())

    fig, ax = plt.subplots(1, 1, figsize=(14, max(len(model_names_fam) / 1.5, 5)),
                           layout='constrained')
    plot_utils.plot_x0line(ax=ax)
    sb.stripplot(fam_diff, y='model', x='pct_diff', hue='reward_cond',
                 ax=ax, order=model_names_fam, hue_order=reward_conds,
                 palette='colorblind', size=9, dodge=True)
    fig.suptitle(f'{family} Models — Matched LL: Free vs Fixed Perseverative Agent by Reward Condition',
                 fontsize=14)
    ax.set_title('Negative = free α_p improves fit; one dot per subject', fontsize=12)
    ax.set_xlabel('% Change in Avg NLL per Fold (Free − Fixed)', fontsize=13)
    ax.set_ylabel('Model', fontsize=13)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(title='Reward Condition', fontsize=11, title_fontsize=12,
              loc='upper right', bbox_to_anchor=(1.14, 1.))
    plt.show()

# %% Fixed vs Free c: CV NLL and BIC Comparison
import math, re

si_c_models = [
    'SI - Shared Rew Evidence, Unrew Fixed 0',
    'SI - Fixed c: 75/10 Condition',
    'SI - Fixed c: 50/10 Condition',
    'SI/Persev (free alpha) - Shared Rew Evidence, Unrew Fixed 0',
    'SI/Persev (free alpha) - Fixed c: 75/10 Condition',
    'SI/Persev (free alpha) - Fixed c: 50/10 Condition',
    'SI/Persev (fixed) - Shared Rew Evidence, Unrew Fixed 0',
    'SI/Persev (fixed) - Fixed c: 75/10 Condition',
    'SI/Persev (fixed) - Fixed c: 50/10 Condition',
]

si_c_label_map = {
    'SI - Shared Rew Evidence, Unrew Fixed 0':                    'SI - Free c',
    'SI - Fixed c: 75/10 Condition':                              'SI - Fixed c (75/10)',
    'SI - Fixed c: 50/10 Condition':                              'SI - Fixed c (50/10)',
    'SI/Persev (free alpha) - Shared Rew Evidence, Unrew Fixed 0':'SI/Persev(free α) - Free c',
    'SI/Persev (free alpha) - Fixed c: 75/10 Condition':          'SI/Persev(free α) - Fixed c (75/10)',
    'SI/Persev (free alpha) - Fixed c: 50/10 Condition':          'SI/Persev(free α) - Fixed c (50/10)',
    'SI/Persev (fixed) - Shared Rew Evidence, Unrew Fixed 0':     'SI/Persev(fixed α) - Free c',
    'SI/Persev (fixed) - Fixed c: 75/10 Condition':               'SI/Persev(fixed α) - Fixed c (75/10)',
    'SI/Persev (fixed) - Fixed c: 50/10 Condition':               'SI/Persev(fixed α) - Fixed c (50/10)',
}

def _rew_rate_si(subjid_str):
    m = re.search(r'\((\d+/\d+)\)', str(subjid_str))
    return m.group(1) if m else 'all'

def _c_type(model_name):
    if '75/10' in model_name:
        return 'Fixed c (75/10)'
    elif '50/10' in model_name:
        return 'Fixed c (50/10)'
    return 'Free c'

def _persev_type(model_name):
    if 'Persev (free alpha)' in model_name:
        return 'Free α_p'
    elif 'Persev (fixed)' in model_name:
        return 'Fixed α_p'
    return 'No Persev'

si_c_cv_rows = []

for subj in subjids:
    for model_name in si_c_models:
        cv_key = model_name + '_cv'
        if cv_key not in all_models[subj]:
            continue
        fit_repeats = all_models[subj][cv_key]
        if not fit_repeats:
            continue

        n_folds = fit_repeats[0]['n_folds']

        # best repeat by total NLL
        valid_repeats = [r for r in fit_repeats if not math.isnan(r['total_nll'])]
        if not valid_repeats:
            continue
        best_repeat = min(valid_repeats, key=lambda r: r['total_nll'])

        # best NLL per fold across all repeats
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for r in fit_repeats
                         for f in r['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) != n_folds:
            continue

        avg_nll = np.mean(best_nll_per_fold)
        total_nll = best_repeat['total_nll']

        # total test trials across folds (for BIC)
        n_trials_test = sum(
            int(round(abs(f['perf']['ll_total'] / f['perf']['ll_avg'])))
            for f in best_repeat['folds']
            if f['perf'].get('ll_avg', 0) != 0
        )

        n_params = th.count_params(best_repeat['model'].model)
        bic = (n_params * np.log(n_trials_test) + 2 * total_nll
               if n_trials_test > 0 else np.nan)

        si_c_cv_rows.append({
            'subjid':           subj,
            'rew_rate':         _rew_rate_si(subj),
            'model':            model_name,
            'model_label':      si_c_label_map[model_name],
            'c_type':           _c_type(model_name),
            'persev':           _persev_type(model_name),
            'avg_nll_per_fold': avg_nll,
            'total_nll':        total_nll,
            'n_trials':         n_trials_test,
            'n_params':         n_params,
            'bic':              bic,
        })

si_c_cv_df = pd.DataFrame(si_c_cv_rows)
print('SI c-model CV metrics:', si_c_cv_df.shape)
print(si_c_cv_df[['model_label', 'n_params']].drop_duplicates().to_string())

# % worse than best model per subject
si_c_cv_df['diff_avg_nll'] = 0.0
si_c_cv_df['diff_bic']     = 0.0
for subj in si_c_cv_df['subjid'].unique():
    sel = si_c_cv_df['subjid'] == subj
    best_nll = si_c_cv_df.loc[sel, 'avg_nll_per_fold'].min()
    best_bic = si_c_cv_df.loc[sel, 'bic'].min()
    si_c_cv_df.loc[sel, 'diff_avg_nll'] = (
        (si_c_cv_df.loc[sel, 'avg_nll_per_fold'] - best_nll) / best_nll * 100)
    si_c_cv_df.loc[sel, 'diff_bic'] = (
        (si_c_cv_df.loc[sel, 'bic'] - best_bic) / best_bic * 100)

model_order_si = [si_c_label_map[m] for m in si_c_models
                  if si_c_label_map[m] in si_c_cv_df['model_label'].values]
model_order_si = list(dict.fromkeys(model_order_si))
rew_order_si   = sorted(si_c_cv_df['rew_rate'].unique())
ax_h_si        = max(len(model_order_si) / 2.5, 4)

# Plot 1: raw avg NLL and BIC
fig, axs = plt.subplots(1, 2, figsize=(18, ax_h_si), layout='constrained')
sb.stripplot(si_c_cv_df, y='model_label', x='avg_nll_per_fold', hue='rew_rate',
             ax=axs[0], order=model_order_si, hue_order=rew_order_si,
             palette='colorblind', size=8)
axs[0].set_title('CV Average NLL per Fold')
axs[0].set_xlabel('Average NLL per Fold')
axs[0].set_ylabel('Model')
axs[0].legend(title='Rew Rate', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))

sb.stripplot(si_c_cv_df, y='model_label', x='bic', hue='rew_rate',
             ax=axs[1], order=model_order_si, hue_order=rew_order_si,
             palette='colorblind', size=8)
axs[1].set_title('CV BIC')
axs[1].set_xlabel('BIC')
axs[1].set_ylabel('')
axs[1].legend(title='Rew Rate', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))
fig.suptitle('SI Fixed vs Free c — CV Model Comparison (Raw)', fontsize=14)
plt.show()

# Plot 2: % worse than best model
fig, axs = plt.subplots(1, 2, figsize=(18, ax_h_si), layout='constrained')
for ax in axs:
    plot_utils.plot_x0line(ax=ax)

sb.stripplot(si_c_cv_df, y='model_label', x='diff_avg_nll', hue='rew_rate',
             ax=axs[0], order=model_order_si, hue_order=rew_order_si,
             palette='colorblind', size=8)
axs[0].set_title('% Worse than Best Model — NLL')
axs[0].set_xlabel('% Worse Avg NLL per Fold')
axs[0].set_ylabel('Model')
axs[0].legend(title='Rew Rate', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))

sb.stripplot(si_c_cv_df, y='model_label', x='diff_bic', hue='rew_rate',
             ax=axs[1], order=model_order_si, hue_order=rew_order_si,
             palette='colorblind', size=8)
axs[1].set_title('% Worse than Best Model — BIC')
axs[1].set_xlabel('% Worse BIC')
axs[1].set_ylabel('')
axs[1].legend(title='Rew Rate', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))
fig.suptitle('SI Fixed vs Free c — % Worse than Best Model', fontsize=14)
plt.show()

# Plot 3: % worse NLL and BIC colored by subject (ignoring reward rate)
si_c_cv_df['base_subj'] = si_c_cv_df['subjid'].apply(
    lambda s: re.sub(r'\s*\(\d+/\d+\)', '', str(s)).strip())
base_subj_order = sorted(si_c_cv_df['base_subj'].unique())

fig, axs = plt.subplots(1, 2, figsize=(18, ax_h_si), layout='constrained')
for ax in axs:
    plot_utils.plot_x0line(ax=ax)

sb.stripplot(si_c_cv_df, y='model_label', x='diff_avg_nll', hue='base_subj',
             ax=axs[0], order=model_order_si, hue_order=base_subj_order,
             palette='colorblind', size=8)
axs[0].set_title('% Worse than Best Model — NLL')
axs[0].set_xlabel('% Worse Avg NLL per Fold')
axs[0].set_ylabel('Model')
axs[0].legend(title='Subject', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))

sb.stripplot(si_c_cv_df, y='model_label', x='diff_bic', hue='base_subj',
             ax=axs[1], order=model_order_si, hue_order=base_subj_order,
             palette='colorblind', size=8)
axs[1].set_title('% Worse than Best Model — BIC')
axs[1].set_xlabel('% Worse BIC')
axs[1].set_ylabel('')
axs[1].legend(title='Subject', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))
fig.suptitle('SI Fixed vs Free c — % Worse than Best Model (colored by subject)', fontsize=14)
plt.show()

# %% Fixed vs Free c: Fitted Parameter Comparison (p_stay and c_same_rew)

si_c_param_rows = []

for subj in subjids:
    for model_name in si_c_models:
        cv_key = model_name + '_cv'
        if cv_key not in all_models[subj]:
            continue
        fit_repeats = all_models[subj][cv_key]
        valid_repeats = [r for r in fit_repeats if not math.isnan(r['total_nll'])]
        if not valid_repeats:
            continue
        best_repeat = min(valid_repeats, key=lambda r: r['total_nll'])
        model_obj   = best_repeat['model'].model

        si_agent = next((a for a in model_obj.agents
                         if isinstance(a, agents.StateInferenceAgent)), None)
        if si_agent is None:
            continue

        # convert UnitParam raw values [-1, 1] to probabilities [0, 1]
        p_stay_prob  = 0.5 * (1 + si_agent.p_stay.a.item())
        c_rew_prob   = 0.5 * (1 + si_agent.c_same_rew.a.item())

        si_c_param_rows.append({
            'subjid':      subj,
            'rew_rate':    _rew_rate_si(subj),
            'model':       model_name,
            'model_label': si_c_label_map[model_name],
            'c_type':      _c_type(model_name),
            'persev':      _persev_type(model_name),
            'p_stay':      p_stay_prob,
            'c_same_rew':  c_rew_prob,
        })

si_c_param_df = pd.DataFrame(si_c_param_rows)

# focus on no-persev models for cleaner parameter comparison
no_persev_df    = si_c_param_df[si_c_param_df['persev'] == 'No Persev']
no_persev_order = [si_c_label_map[m] for m in si_c_models[:3]]
rew_order_p     = sorted(si_c_param_df['rew_rate'].unique())
ax_h_p          = max(len(no_persev_order) / 1.5, 3)

fig, axs = plt.subplots(1, 2, figsize=(14, ax_h_p), layout='constrained')

# p_stay
sb.stripplot(no_persev_df, y='model_label', x='p_stay', hue='rew_rate',
             ax=axs[0], order=no_persev_order, hue_order=rew_order_p,
             palette='colorblind', size=9)
axs[0].axvline(0.5, color='gray', linestyle='--', lw=1, label='P(stay) = 0.5')
axs[0].set_xlim(0, 1)
axs[0].set_title('Fitted P(stay)')
axs[0].set_xlabel('P(stay)')
axs[0].set_ylabel('Model')
axs[0].legend(title='Rew Rate', fontsize=9, loc='upper right', bbox_to_anchor=(1.01, 1.))

# c_same_rew — show all 3 models; fixed models cluster at their fixed value,
# free model shows the fitted value; reference lines show the two fixed values
sb.stripplot(no_persev_df, y='model_label', x='c_same_rew', hue='rew_rate',
             ax=axs[1], order=no_persev_order, hue_order=rew_order_p,
             palette='colorblind', size=9)
axs[1].axvline(0.75, color='gray', linestyle='--', lw=1.2, label='Fixed 75/10 = 0.75')
axs[1].axvline(0.50, color='gray', linestyle=':',  lw=1.2, label='Fixed 50/10 = 0.50')
axs[1].set_xlim(0, 1)
axs[1].set_title('P(rew | same choice, high state) = 0.5·(1 + c_same_rew)')
axs[1].set_xlabel('P(rew | same, high)')
axs[1].set_ylabel('')
axs[1].legend(title='Rew Rate / Reference', fontsize=9, loc='upper right',
              bbox_to_anchor=(1.01, 1.))

fig.suptitle('SI Fixed vs Free c — Fitted Parameters (No Persev)', fontsize=14)
plt.show()

# %% CV Model Comparison — Q, SI, RL SI (5/26/2026)
import math

new_run_keep = [
    'Q - All Alpha Free, All K Fixed, Diff K=0.5',
    'SI - All Separate Evidence',
    'SI - Free Same/Diff Rew Evidence',
    'RL SI - All Separate Evidence',
    'RL SI - Free Same/Diff Rew Evidence',
]

new_run_cv_rows = []

for subj in subjids:
    for model_name in list(all_models[subj].keys()):
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]
        if not any(k in base_name for k in new_run_keep):
            continue

        fit_repeats = all_models[subj][model_name]
        if not fit_repeats:
            continue

        n_folds = fit_repeats[0]['n_folds']

        # best NLL per fold across all repeats
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [f['nll'] for r in fit_repeats
                         for f in r['folds']
                         if f['fold_idx'] == fold_idx and not math.isnan(f['nll'])]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) != n_folds:
            continue

        avg_nll = np.mean(best_nll_per_fold)

        # best repeat by total NLL for BIC
        valid_repeats = [r for r in fit_repeats if not math.isnan(r['total_nll'])]
        if not valid_repeats:
            continue
        best_repeat = min(valid_repeats, key=lambda r: r['total_nll'])

        n_trials_test = sum(
            int(round(abs(f['perf']['ll_total'] / f['perf']['ll_avg'])))
            for f in best_repeat['folds']
            if f['perf'].get('ll_avg', 0) != 0
        )

        n_params = th.count_params(best_repeat['model'].model)
        bic = (n_params * np.log(n_trials_test) + 2 * best_repeat['total_nll']
               if n_trials_test > 0 else np.nan)

        new_run_cv_rows.append({
            'subjid':           subj,
            'model':            '{} ({})'.format(base_name, n_params),
            'avg_nll_per_fold': avg_nll,
            'n_params':         n_params,
            'bic':              bic,
        })

new_run_cv_df = pd.DataFrame(new_run_cv_rows)
print('New run CV metrics:', new_run_cv_df.shape)
print('Models found:', new_run_cv_df['model'].nunique())

new_run_model_names = sorted(new_run_cv_df['model'].unique().tolist(), key=str.lower)
new_run_subjids     = sorted(new_run_cv_df['subjid'].unique().tolist())

# % worse than best model per subject
new_run_cv_df['diff_avg_nll'] = 0.0
new_run_cv_df['diff_bic']     = 0.0
for subj in new_run_subjids:
    sel      = new_run_cv_df['subjid'] == subj
    best_nll = new_run_cv_df.loc[sel, 'avg_nll_per_fold'].min()
    best_bic = new_run_cv_df.loc[sel, 'bic'].min()
    new_run_cv_df.loc[sel, 'diff_avg_nll'] = (
        (new_run_cv_df.loc[sel, 'avg_nll_per_fold'] - best_nll) / best_nll * 100)
    new_run_cv_df.loc[sel, 'diff_bic'] = (
        (new_run_cv_df.loc[sel, 'bic'] - best_bic) / best_bic * 100)

ax_h_new = max(len(new_run_model_names) / 5, 4)

# Plot 1: raw avg NLL per fold
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_new), layout='constrained')
sb.stripplot(new_run_cv_df, y='model', x='avg_nll_per_fold', hue='subjid',
             ax=ax, palette='colorblind', order=new_run_model_names,
             hue_order=new_run_subjids)
fig.suptitle('CV Model Performance')
ax.set_title('Average NLL per Fold')
ax.set_xlabel('Average NLL per Fold')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# Plot 2: % worse NLL
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_new), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(new_run_cv_df, y='model', x='diff_avg_nll', hue='subjid',
             ax=ax, palette='colorblind', order=new_run_model_names,
             hue_order=new_run_subjids)
fig.suptitle('CV Model Performance')
ax.set_title('% Worse than Best Model — NLL')
ax.set_xlabel('% Worse Avg NLL per Fold')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# Plot 3: raw BIC
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_new), layout='constrained')
sb.stripplot(new_run_cv_df, y='model', x='bic', hue='subjid',
             ax=ax, palette='colorblind', order=new_run_model_names,
             hue_order=new_run_subjids)
fig.suptitle('CV Model Performance')
ax.set_title('BIC')
ax.set_xlabel('BIC')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# Plot 4: % worse BIC
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_new), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(new_run_cv_df, y='model', x='diff_bic', hue='subjid',
             ax=ax, palette='colorblind', order=new_run_model_names,
             hue_order=new_run_subjids)
fig.suptitle('CV Model Performance')
ax.set_title('% Worse than Best Model — BIC')
ax.set_xlabel('% Worse BIC')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% CV Model Comparison — 5 Base Models + Fixed Persev Variants: Build Data
import math

# The 6 base models (5 focal + Q+SI) and their persev counterparts
focal_base_models = [
    'Bayes - No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates',
    'Q - All Alpha Shared, All K Fixed',
    'Q - Alpha Rew/Unrew Shared, All K Fixed',
    'SI - Free Same/Diff Rew Evidence',
    'SI - Shared Rew Evidence, Unrew Fixed 0',
    'Q+SI - All Alpha Shared, All K Fixed + Shared Rew Evidence, Unrew Fixed 0',
]
focal_persev_models = [m.replace('Bayes - ', 'Bayes/Persev (fixed) - ')
                        .replace('Q - ', 'Q/Persev (fixed) - ')
                        .replace('SI - ', 'SI/Persev (fixed) - ')
                        .replace('Q+SI - ', 'Q+SI/Persev (fixed) - ')
                       for m in focal_base_models]
focal_keep = focal_base_models + focal_persev_models

# use all loaded subjects (all 9) rather than the hardcoded subjids list
_cv_subjids = [s for s in all_models.keys() if 'meta' not in s.lower()]

# Build CV metrics for the focal model set
focal_cv_rows = []
for subj in _cv_subjids:
    for model_name in list(all_models[subj].keys()):
        if not model_name.endswith('_cv'):
            continue
        base_name = model_name[:-3]
        if not any(k == base_name for k in focal_keep):
            continue

        fit_repeats = all_models[subj][model_name]
        if not fit_repeats:
            continue

        n_folds = fit_repeats[0]['n_folds']

        # use per-trial NLL (-ll_avg) so units match non-CV plots
        best_nll_per_fold = []
        for fold_idx in range(n_folds):
            fold_nlls = [-f['perf']['ll_avg'] for r in fit_repeats
                         for f in r['folds']
                         if f['fold_idx'] == fold_idx
                         and not math.isnan(f['nll'])
                         and f['perf'].get('ll_avg', 0) != 0]
            if fold_nlls:
                best_nll_per_fold.append(min(fold_nlls))

        if len(best_nll_per_fold) != n_folds:
            continue

        avg_nll_per_trial = np.mean(best_nll_per_fold)

        valid_repeats = [r for r in fit_repeats if not math.isnan(r['total_nll'])]
        if not valid_repeats:
            continue
        best_repeat = min(valid_repeats, key=lambda r: r['total_nll'])

        n_params = th.count_params(best_repeat['model'].model)
        model_label = '{} ({})'.format(base_name, n_params)

        focal_cv_rows.append({
            'subjid': subj,
            'model': model_label,
            'base_name': base_name,
            'nll_per_trial': avg_nll_per_trial,
            'n_params': n_params,
            'rew_cond': re.search(r'(75/10|50/10)', str(subj)).group(1) if re.search(r'(75/10|50/10)', str(subj)) else 'Unknown',
        })

focal_cv_df = pd.DataFrame(focal_cv_rows)
print('Focal CV metrics:', focal_cv_df.shape)
print('Models found:', focal_cv_df['model'].nunique())
print(focal_cv_df['base_name'].unique().tolist())

# % worse than best model per subject (best = lowest NLL per trial across all 12 models)
focal_subjids = sorted(focal_cv_df['subjid'].unique().tolist())
focal_cv_df['diff_avg_nll'] = 0.0
for subj in focal_subjids:
    sel = focal_cv_df['subjid'] == subj
    best_nll = focal_cv_df.loc[sel, 'nll_per_trial'].min()
    focal_cv_df.loc[sel, 'diff_avg_nll'] = (
        (focal_cv_df.loc[sel, 'nll_per_trial'] - best_nll) / best_nll * 100)

# Build model display order: pair each base model with its persev counterpart
# so related models sit adjacent on the y-axis
focal_model_order = []
for base, persev in zip(focal_base_models, focal_persev_models):
    base_matches   = focal_cv_df[focal_cv_df['base_name'] == base]['model'].unique().tolist()
    persev_matches = focal_cv_df[focal_cv_df['base_name'] == persev]['model'].unique().tolist()
    focal_model_order.extend(base_matches + persev_matches)

# average % worse across subjects per model
focal_avg_diff = focal_cv_df.groupby('model')[['diff_avg_nll', 'nll_per_trial']].mean().reset_index()

ax_h_focal = max(len(focal_model_order) / 3, 5)

# %% CV Model Comparison — 6 Models + Fixed Persev: Plot 1 - NLL per Trial
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_focal), layout='constrained')
sb.stripplot(focal_cv_df, y='model', x='nll_per_trial', hue='subjid',
             ax=ax, palette='colorblind', order=focal_model_order,
             hue_order=focal_subjids)
fig.suptitle('CV Model Comparison — 6 Models + Fixed Persev (9 subjects)')
ax.set_title('NLL per Trial (avg across folds, one dot per subject)')
ax.set_xlabel('NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% CV Model Comparison — 6 Models + Fixed Persev: Plot 2 - % Worse per Subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_focal), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_cv_df, y='model', x='diff_avg_nll', hue='subjid',
             ax=ax, palette='colorblind', order=focal_model_order,
             hue_order=focal_subjids)
fig.suptitle('CV Model Comparison — 6 Models + Fixed Persev (9 subjects)')
ax.set_title('% Worse than Best Model (one dot per subject)')
ax.set_xlabel('% Worse NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% CV Model Comparison — 6 Models + Fixed Persev: Plot 3 - Average % Worse
# exclude Bayes models from this plot
_no_bayes_order = [m for m in focal_model_order if 'Bayes' not in m]
_no_bayes_avg   = focal_avg_diff[focal_avg_diff['model'].isin(_no_bayes_order)]

ax_h_no_bayes = max(len(_no_bayes_order) / 3, 5)
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_no_bayes), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(_no_bayes_avg, y='model', x='diff_avg_nll',
             ax=ax, order=_no_bayes_order, color='steelblue', size=9)
fig.suptitle('CV Model Comparison — No Bayes, Fixed Persev (9 subjects)')
ax.set_title('Average % Worse than Best Model (one dot per model, averaged across subjects)')
ax.set_xlabel('Avg % Worse NLL per Trial')
ax.set_ylabel('Model')
plt.show()

# %% CV Model Comparison — 6 Models + Fixed Persev: By Reward Condition, Build Data
_rew_order = ['75/10', '50/10']

# % worse per subject, computed within each reward condition separately
focal_cv_df['diff_avg_nll_rew'] = 0.0
for subj in focal_subjids:
    sel = focal_cv_df['subjid'] == subj
    best_nll = focal_cv_df.loc[sel, 'nll_per_trial'].min()
    focal_cv_df.loc[sel, 'diff_avg_nll_rew'] = (
        (focal_cv_df.loc[sel, 'nll_per_trial'] - best_nll) / best_nll * 100)

# average % worse per model per reward condition
focal_avg_diff_rew = (focal_cv_df.groupby(['model', 'rew_cond'])[['diff_avg_nll_rew', 'nll_per_trial']]
                      .mean().reset_index())

# %% CV Model Comparison — 6 Models + Fixed Persev: By Reward Condition, Plot 1 - NLL per Trial
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_focal), layout='constrained')
sb.stripplot(focal_cv_df, y='model', x='nll_per_trial', hue='rew_cond',
             ax=ax, palette='colorblind', order=focal_model_order,
             hue_order=_rew_order, dodge=True, size=7)
fig.suptitle('CV Model Comparison — 6 Models + Fixed Persev (by Reward Condition)')
ax.set_title('NLL per Trial (avg across folds, one dot per subject)')
ax.set_xlabel('NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Reward Condition', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.18, 1.))
plt.show()

# %% CV Model Comparison — 6 Models + Fixed Persev: By Reward Condition, Plot 2 - % Worse per Subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_focal), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_cv_df, y='model', x='diff_avg_nll_rew', hue='rew_cond',
             ax=ax, palette='colorblind', order=focal_model_order,
             hue_order=_rew_order, dodge=True, size=7)
fig.suptitle('CV Model Comparison — 6 Models + Fixed Persev (by Reward Condition)')
ax.set_title('% Worse than Best Model (one dot per subject)')
ax.set_xlabel('% Worse NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Reward Condition', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.18, 1.))
plt.show()

# %% CV Model Comparison — 6 Models + Fixed Persev: By Reward Condition, Plot 3 - Average % Worse
# exclude Bayes, use boxplot
_no_bayes_order_rew = [m for m in focal_model_order if 'Bayes' not in m and 'Q+SI' not in m]
_no_bayes_cv_df     = focal_cv_df[focal_cv_df['model'].isin(_no_bayes_order_rew)]
ax_h_no_bayes_rew   = max(len(_no_bayes_order_rew) / 3, 5)

fig, ax = plt.subplots(1, 1, figsize=(14, ax_h_no_bayes_rew), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.boxplot(_no_bayes_cv_df, y='model', x='diff_avg_nll_rew', hue='rew_cond',
           ax=ax, palette='colorblind', order=_no_bayes_order_rew,
           hue_order=_rew_order, dodge=True)
fig.suptitle('Fixed vs. Np Persev (by Reward Condition)')
ax.set_title('Average % Worse than Best Model (averaged across subjects)')
ax.set_xlabel('Avg % Worse NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Reward Condition', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.18, 1.))
plt.show()

# %% Non-CV Model Comparison — 5 Focal Models, Build Data

_focal_base = [
    'Bayes - No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates',
    'Q - All Alpha Shared, All K Fixed',
    'Q - Alpha Rew/Unrew Shared, All K Fixed',
    'SI - Free Same/Diff Rew Evidence',
    'SI - Shared Rew Evidence, Unrew Fixed 0',
]
_focal_persev = [m.replace('Bayes - ', 'Bayes/Persev (fixed) - ')
                  .replace('Q - ', 'Q/Persev (fixed) - ')
                  .replace('SI - ', 'SI/Persev (fixed) - ')
                 for m in _focal_base]
_focal_all = _focal_base + _focal_persev

focal_noncv_rows = []
for subj in subjids:
    for model_name in _focal_all:
        if model_name not in all_models[subj]:
            continue
        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        # best repeat = highest ll_total (log likelihood, higher = better)
        valid = [r for r in fit_repeats if 'perf' in r and not np.isnan(r['perf']['ll_total'])]
        if not valid:
            continue
        best = max(valid, key=lambda r: r['perf']['ll_total'])

        model_obj = best['model'].model
        n_params = th.count_params(model_obj)
        # correct for serialization bug in old models: alpha_p requires_grad is wrongly True
        # after loading. For Persev (fixed) models alpha should always be fixed (False).
        # Only subtract if it is incorrectly True.
        if 'Persev (fixed)' in model_name:
            persev_agents = [a for a in model_obj.agents if isinstance(a, agents.PerseverativeAgent)]
            if any(a.alpha.requires_grad for a in persev_agents):
                n_params -= 1
        nll_per_trial = -best['perf']['ll_avg']  # ll_avg is negative, flip to get NLL

        focal_noncv_rows.append({
            'subjid':   subj,
            'model':    '{} ({})'.format(model_name, n_params),
            'base_name': model_name,
            'nll_per_trial': nll_per_trial,
            'n_params': n_params,
        })

focal_noncv_df = pd.DataFrame(focal_noncv_rows)
print('Non-CV focal metrics:', focal_noncv_df.shape)
print('Models found:', focal_noncv_df['model'].nunique())

focal_noncv_subjids = sorted(focal_noncv_df['subjid'].unique().tolist())

# % worse than best model per subject
focal_noncv_df['diff_nll'] = 0.0
for subj in focal_noncv_subjids:
    sel = focal_noncv_df['subjid'] == subj
    best_nll = focal_noncv_df.loc[sel, 'nll_per_trial'].min()
    focal_noncv_df.loc[sel, 'diff_nll'] = (
        (focal_noncv_df.loc[sel, 'nll_per_trial'] - best_nll) / best_nll * 100)

# build model order: pair each base model with its persev counterpart
focal_noncv_order = []
for base, persev in zip(_focal_base, _focal_persev):
    base_labels   = focal_noncv_df[focal_noncv_df['base_name'] == base]['model'].unique().tolist()
    persev_labels = focal_noncv_df[focal_noncv_df['base_name'] == persev]['model'].unique().tolist()
    focal_noncv_order.extend(base_labels + persev_labels)

focal_noncv_avg_diff = focal_noncv_df.groupby('model')['diff_nll'].mean().reset_index()

ax_h_noncv = max(len(focal_noncv_order) / 3, 5)

# %% Non-CV Model Comparison — 5 Focal Models: Plot 1 - NLL per Trial
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_noncv), layout='constrained')
sb.stripplot(focal_noncv_df, y='model', x='nll_per_trial', hue='subjid',
             ax=ax, palette='colorblind', order=focal_noncv_order,
             hue_order=focal_noncv_subjids)
fig.suptitle('Non-CV Model Comparison — 5 Base Models + Fixed Persev')
ax.set_title('NLL per Trial — Best Fit (one dot per subject)')
ax.set_xlabel('NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% Non-CV Model Comparison — 5 Focal Models: Plot 2 - % Worse per Subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_noncv), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_noncv_df, y='model', x='diff_nll', hue='subjid',
             ax=ax, palette='colorblind', order=focal_noncv_order,
             hue_order=focal_noncv_subjids)
fig.suptitle('Non-CV Model Comparison — 5 Base Models + Fixed Persev')
ax.set_title('% Worse than Best Model (one dot per subject)')
ax.set_xlabel('% Worse NLL per Trial')
ax.set_ylabel('Model')
ax.legend(title='Subject', loc='upper right', fontsize=8,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% Non-CV Model Comparison — 5 Focal Models: Plot 3 - Average % Worse
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_noncv), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(focal_noncv_avg_diff, y='model', x='diff_nll',
             ax=ax, order=focal_noncv_order, color='steelblue', size=9)
fig.suptitle('Non-CV Model Comparison — 5 Base Models + Fixed Persev')
ax.set_title('Average % Worse than Best Model (one dot per model, averaged across subjects)')
ax.set_xlabel('Avg % Worse NLL per Trial')
ax.set_ylabel('Model')
plt.show()

# %% Non-CV With vs Without Perseveration — 5 Focal Models, Build Data

# compute % change in NLL per trial: (with_persev - no_persev) / no_persev * 100
# reuses focal_noncv_df from the Non-CV Build Data cell
noncv_persev_diff_rows = []
for subj in focal_noncv_df['subjid'].unique():
    subj_data = focal_noncv_df[focal_noncv_df['subjid'] == subj]
    for base, persev in zip(_focal_base, _focal_persev):
        # find matching model labels (which include param count suffix)
        base_rows   = subj_data[subj_data['base_name'] == base]
        persev_rows = subj_data[subj_data['base_name'] == persev]
        if base_rows.empty or persev_rows.empty:
            continue
        nll_base   = base_rows.iloc[0]['nll_per_trial']
        nll_persev = persev_rows.iloc[0]['nll_per_trial']
        noncv_persev_diff_rows.append({
            'subjid':   subj,
            'model':    base,
            'pct_diff': (nll_persev - nll_base) / nll_base * 100,
        })

noncv_persev_diff_df = pd.DataFrame(noncv_persev_diff_rows)
noncv_persev_subjids = sorted(noncv_persev_diff_df['subjid'].unique().tolist())
noncv_persev_order   = [m for m in _focal_base if m in noncv_persev_diff_df['model'].unique()]
noncv_persev_avg     = noncv_persev_diff_df.groupby('model')['pct_diff'].mean().reset_index()
ax_h_ncp = max(len(noncv_persev_order) / 1.5, 4)

# %% Non-CV With vs Without Perseveration — Plot 1: one dot per subject
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_ncp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(noncv_persev_diff_df, y='model', x='pct_diff', hue='subjid',
             ax=ax, palette='colorblind', order=noncv_persev_order,
             hue_order=noncv_persev_subjids, size=8)
fig.suptitle('With vs Without Perseveration — 5 Focal Models (Non-CV)')
ax.set_title('Negative = perseveration improves fit (one dot per subject)')
ax.set_xlabel('% Change in NLL per Trial (With Persev − No Persev)')
ax.set_ylabel('Model')
ax.tick_params(axis='both', labelsize=12)
ax.legend(title='Subject', loc='upper right', fontsize=9,
          framealpha=0.5, bbox_to_anchor=(1.14, 1.))
plt.show()

# %% Non-CV With vs Without Perseveration — Plot 2: averaged across subjects
fig, ax = plt.subplots(1, 1, figsize=(12, ax_h_ncp), layout='constrained')
plot_utils.plot_x0line(ax=ax)
sb.stripplot(noncv_persev_avg, y='model', x='pct_diff',
             ax=ax, order=noncv_persev_order, color='steelblue', size=10)
fig.suptitle('With vs Without Perseveration — 5 Focal Models (Non-CV)')
ax.set_title('Averaged across subjects (one dot per model)')
ax.set_xlabel('Avg % Change in NLL per Trial (With Persev − No Persev)')
ax.set_ylabel('Model')
ax.tick_params(axis='both', labelsize=12)
plt.show()

# %% Best-Fit Parameters — 5 Focal Models (no CV), Build Data
import re

def _extract_rew_cond_param(subjid_str):
    m = re.search(r'(75/10|50/10)', str(subjid_str))
    return m.group(1) if m else 'Unknown'

def _extract_focal_params(model_obj):
    """Extract interpretable parameter values from a fitted SummationModule."""
    params = {}

    bayes_attrs = ['init_high_rew_mean', 'init_low_rew_mean', 'init_rew_sig',
                   'init_switch_mean', 'init_switch_sig', 'switch_scatter_sig',
                   'stay_bias_lam', 'outcome_inference_lam', 'unreward_inference_lam',
                   'imperfect_update_alpha', 'forget_alpha']
    q_attrs  = ['alpha_same_rew', 'alpha_same_unrew', 'alpha_diff_rew', 'alpha_diff_unrew']
    si_attrs = ['p_stay', 'c_same_rew', 'c_diff_rew', 'c_same_unrew', 'c_diff_unrew']

    for agent in model_obj.agents:
        if isinstance(agent, agents.QValueAgent):
            seen_ids = set()
            for attr in q_attrs:
                obj = getattr(agent, attr)
                if id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    params[attr] = obj.a.item()

        elif isinstance(agent, agents.StateInferenceAgent):
            seen_ids = set()
            for attr in si_attrs:
                obj = getattr(agent, attr)
                if id(obj) not in seen_ids:
                    seen_ids.add(id(obj))
                    params[attr] = obj.a.item()

        elif isinstance(agent, agents.PerseverativeAgent):
            params['alpha_p'] = agent.alpha.a.item()

        elif isinstance(agent, agents.BayesianAgent):
            for attr in bayes_attrs:
                obj = getattr(agent, attr, None)
                if obj is not None and isinstance(obj, agents.UnitParam):
                    params[attr] = obj.a.item()

    # beta weights (PositiveConstraint = softplus, already constrained positive)
    betas = model_obj.beta.weight.data.squeeze()
    if betas.dim() == 0:
        params['beta'] = betas.item()
    else:
        for i, b in enumerate(betas.tolist()):
            params[f'beta_{i}'] = b

    # bias
    if hasattr(model_obj, 'bias') and model_obj.bias is not None:
        params['bias'] = model_obj.bias.item()

    return params

# the 5 base models and their persev counterparts (no _cv suffix)
focal_param_models = focal_base_models + focal_persev_models  # reuse lists from earlier cell

focal_param_rows = []
for subj in subjids:
    for model_name in focal_param_models:
        if model_name not in all_models[subj]:
            continue
        fit_repeats = all_models[subj][model_name]
        if not isinstance(fit_repeats, list) or len(fit_repeats) == 0:
            continue

        # best repeat = highest ll_total (perf stores log likelihood, higher = better)
        valid = [r for r in fit_repeats if 'perf' in r and not np.isnan(r['perf']['ll_total'])]
        if not valid:
            continue
        best = max(valid, key=lambda r: r['perf']['ll_total'])

        model_obj = best['model'].model
        param_vals = _extract_focal_params(model_obj)

        for param_name, param_val in param_vals.items():
            focal_param_rows.append({
                'subjid':    subj,
                'model':     model_name,
                'rew_cond':  _extract_rew_cond_param(subj),
                'param':     param_name,
                'value':     param_val,
            })

focal_param_df = pd.DataFrame(focal_param_rows)
print('Focal param rows:', focal_param_df.shape)
print('Models found:', focal_param_df['model'].unique().tolist())

# %% Best-Fit Parameters — 5 Focal Models (no CV), Plots
# One figure per model, one dot per subject colored by subject ID
focal_param_subjids = sorted(focal_param_df['subjid'].unique().tolist())

for base, persev in zip(focal_base_models, focal_persev_models):
    for model_name in [base, persev]:
        mdf = focal_param_df[focal_param_df['model'] == model_name]
        if mdf.empty:
            print(f'No data for {model_name}')
            continue

        param_names = sorted(mdf['param'].unique().tolist())
        ax_h = max(len(param_names) / 1.5, 3)

        fig, ax = plt.subplots(1, 1, figsize=(8, ax_h), layout='constrained')
        sb.stripplot(mdf, y='param', x='value', hue='subjid',
                     ax=ax, palette='colorblind', order=param_names,
                     hue_order=focal_param_subjids, size=9)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(0.5, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
        ax.axvline(1, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        fig.suptitle(model_name, fontsize=12)
        ax.set_xlabel('Parameter Value')
        ax.set_ylabel('Parameter')
        ax.legend(title='Subject', fontsize=9, loc='upper right',
                  bbox_to_anchor=(1.3, 1.))
        plt.show()

# %% Best-Fit Parameters — By Reward Condition (y axis is parameter, x axis is parameter value in log scale)
# One figure per model
# Each parameter: bar = mean per condition, dots = individual subjects,
# line connects 75/10 mean to 50/10 mean to show shift between conditions

_rew_palette_param = {'75/10': 'steelblue', '50/10': 'darkorange'}
_rew_order_param   = ['75/10', '50/10']
_dodge_offset      = 0.2   # seaborn dodge offset for 2 hue groups (bar width=0.8, /2 = 0.4, /2 = 0.2)

for base, persev in zip(focal_base_models, focal_persev_models):
    for model_name in [base, persev]:
        mdf = focal_param_df[focal_param_df['model'] == model_name].copy()
        if mdf.empty:
            print(f'No data for {model_name}')
            continue

        param_names = sorted(mdf['param'].unique().tolist())
        ax_h = max(len(param_names) / 1.5, 3)

        fig, ax = plt.subplots(1, 1, figsize=(9, ax_h), layout='constrained')

        # bar: mean per (param, rew_cond) with SE error bars
        sb.barplot(mdf, y='param', x='value', hue='rew_cond',
                   ax=ax, palette=_rew_palette_param, order=param_names,
                   hue_order=_rew_order_param, dodge=True,
                   errorbar='se', capsize=0.2, alpha=0.5)

        # scatterplot: individual subject dots on top
        sb.stripplot(mdf, y='param', x='value', hue='rew_cond',
                     ax=ax, palette=_rew_palette_param, order=param_names,
                     hue_order=_rew_order_param, dodge=True,
                     size=6, jitter=False, alpha=0.9)

        # connecting lines: for each parameter draw a line from 75/10 mean to 50/10 mean
        means = mdf.groupby(['param', 'rew_cond'])['value'].mean().reset_index()
        for i, param in enumerate(param_names):
            m75 = means.loc[(means['param'] == param) & (means['rew_cond'] == '75/10'), 'value']
            m50 = means.loc[(means['param'] == param) & (means['rew_cond'] == '50/10'), 'value']
            if len(m75) > 0 and len(m50) > 0:
                ax.plot([m75.values[0], m50.values[0]],
                        [i - _dodge_offset, i + _dodge_offset],
                        color='gray', linewidth=1.5, zorder=5, alpha=0.8)

        # symlog scale handles 0 and negative values (e.g. bias, fixed params at 0)
        ax.set_xscale('symlog', linthresh=0.01)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.axvline(1, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        # deduplicate legend entries (barplot + scatterlot both add legends)
        handles, labels = ax.get_legend_handles_labels()
        seen, uniq_h, uniq_l = set(), [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); uniq_h.append(h); uniq_l.append(l)
        ax.legend(uniq_h, uniq_l, title='Reward Condition', fontsize=9,
                  loc='upper right', bbox_to_anchor=(1.3, 1.))

        fig.suptitle(model_name, fontsize=11)
        ax.set_xlabel('Parameter Value (symlog scale)')
        ax.set_ylabel('Parameter')
        plt.show()

