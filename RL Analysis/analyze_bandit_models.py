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
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
from sys_neuro_tools import plot_utils, fp_utils
from modeling import agents
import modeling.training_helpers as th
import modeling.sim_helpers as sh
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


# %% Load data

subj_ids = [179, 188, 191, 207] # 182

save_path = path.join(script_dir, 'fit_models_local.json')
if path.exists(save_path):
    all_models = agents.load_model(save_path)
else:
    all_models = {}

# load data
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=2)

# start from the third session (so index=2)-->do not account for the first two sessions
sess_ids = {subj: sess[2:] for subj, sess in sess_ids.items()}

# get session data
reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

all_sess = th.define_choice_outcome(all_sess)
    
# %% Investigate Fit Results

ignore_subj = ['182']
# plot accuracy and total LL
subjids = list(all_models.keys())
subjids = [s for s in subjids if not s in ignore_subj]

ignore_models = ['Q/Persev/Fall', 'SI/Persev', 'Q SI', 'RL SI'] # ['basic - value only']

model_names = list(all_models[subjids[0]].keys())
model_names.sort(key=str.lower)

model_names = [n for n in model_names if not any(im in n for im in ignore_models)]

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
fit_mets['n_trials'] = (fit_mets['ll_total']/fit_mets['ll_avg']).astype(int)
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

subjids = [179, 188, 191] # [179, 188, 191, 207]

sides = ['contra', 'ipsi']
regions = ['DMS', 'PL']

use_fp_sess_only = True
plot_ind_sess = True
plot_output = True
plot_output_diffs = False
plot_agent_states = True
plot_rpes = True
plot_fp_peak_amps = True
plot_fp_rpe_corr = True

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

compare_model_info = {'Bayes - No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates': {'agent_names': ['p(reward)']}} 
#                       'Q SI - Alpha Free, K Free, Belief Update First': {'agent_names': ['State', 'Value', 'Belief']}}

               
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

use_simple_plot = True

limit_mask = False
n_limit_hist = 2

n_plots = 3

model_name = 'Q/Persev - All Alpha Shared, All K Fixed'
agent_names = ['Value', 'Perseverative'] # ['State'] #

subjids = [188] # [179, 188, 191, 207]

for subj in subjids: 

    sess_data = all_sess[all_sess['subjid'] == subj]
    # filter out no responses
    sess_data = sess_data[sess_data['hit']==True]
    
    training_data = th.get_model_training_data(sess_data, limit_mask=limit_mask, n_limit_hist=n_limit_hist)
        
    best_model_idx = 0
    for i in range(len(all_models[str(subj)][model_name])):
        if all_models[str(subj)][model_name][i]['perf']['norm_llh'] > all_models[str(subj)][model_name][best_model_idx]['perf']['norm_llh']:
            best_model_idx = i    

    model = all_models[str(subj)][model_name][best_model_idx]['model'].model.clone()
    
    output, agent_states = th.run_model(model, training_data['two_side_inputs'], output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
    betas = model.beta.weight[0].detach().numpy()
    
    # add to the agent states for hybrid value/state inference agents
    if isinstance(model.agents[0], agents.QValueStateInferenceAgent):
        value_hist = torch.stack(model.agents[0].v_hist[1:], dim=1).numpy()
        belief_hist = torch.stack(model.agents[0].belief_hist[1:], dim=1).numpy()
        agent_states = np.insert(agent_states, 1, value_hist, axis=3)
        agent_states = np.insert(agent_states, 2, belief_hist, axis=3)
        betas = np.insert(betas, 1, np.array([1,1]))

    if use_simple_plot:
        th.plot_simple_multi_val_fit_results(sess_data, output, agent_states, 
                                             agent_names, betas=betas, use_ratio=False,
                                             title_prefix='Subj {}, {} - '.format(subj, model_name))
    else:
        th.plot_multi_val_fit_results(sess_data, output, agent_states, agent_names, n_sess=n_plots, trial_mask=training_data['trial_mask_train'],
                                      betas=betas, title_prefix='Subj {}, {} - '.format(subj, model_name))