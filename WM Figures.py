# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:16:38 2024

@author: tanne
"""

# %% imports

import init
from hankslab_db import db_access
import hankslab_db.tonecatdelayresp_db as db
from pyutils import utils
import numpy as np
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
from sys_neuro_tools import plot_utils, fp_utils
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
import pandas as pd
import pickle
import os.path as path
import seaborn as sb
from scipy import stats
import copy
import warnings


# %% Declare subject information

# get all session ids for given protocol
sess_ids = utils.flatten(db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp'))
sess_ids.extend(utils.flatten(db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp2')))

sess_info = db_access.get_sess_protocol_stage(sess_ids)
# update stage number for rat 188 since different numbers are used for the same stage due to adding a stage in the middle of recording
sess_info.loc[(sess_info['subjid'] == 188) & (sess_info['protocol'] == 'ToneCatDelayResp2'), 'startstage'] = 9
sess_info['proto_stage'] = sess_info.apply(lambda x: '{}_{}'.format(x['protocol'], x['startstage']), axis=1)

sess_ids = sess_info[['subjid', 'sessid']].groupby('subjid')['sessid'].apply(list).to_dict()
subj_ids = np.unique(sess_info['subjid'])

reload = False
loc_db = db.LocalDB_ToneCatDelayResp()
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)
variants = np.unique(sess_data['task_variant'])

# %% Set up variables
signal_types = ['z_dff_iso']
alignments = [Align.tone, Align.cue, Align.reward]
xlims = {Align.tone: {'DMS': [-1,2], 'PL': [-3,5]},
         Align.cue: {'DMS': [-1,2], 'PL': [-3,5]},
         Align.reward: {'DMS': [-1,2], 'PL': [-3,15]}}

regions = ['PL', 'DMS'] 
recalculate = False

filename = 'sel_wm_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        aligned_signals = pickle.load(f)
else:
    aligned_signals = {subjid: {sessid: {sig_type: {align: {region: {'norm': {}, 'raw': {}} for region in regions} 
                                                    for align in alignments}
                                         for sig_type in signal_types}
                                for sessid in sess_ids[subjid]} 
                       for subjid in subj_ids}
                       

# %% Build signal matrices aligned to alignment points

# choose 405 over 420 when there are sessions with both for 3.6
isos = {182: ['405', '420'], 202: ['405', '420'], 179: ['420', '405'],
        180: ['420', '405'], 188: ['420', '405'], 191: ['420', '405'], 207: ['420', '405']}

baseline_lims = [-0.1, 0]

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:
        if sess_id in fpah.__sess_ignore:
            continue

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, isos=isos[subj_id], fit_baseline=False)
        fp_data = fp_data[subj_id][sess_id]

        trial_data = sess_data[sess_data['sessid'] == sess_id]
     
        rewarded = trial_data['reward'] > 0
        responded = ~np.isnan(trial_data['response_time'])
        n_tones = trial_data['n_tones']
        variant = trial_data['task_variant']
        incongruent = trial_data['incongruent']
        one_tone = n_tones == 1
        two_tone = n_tones == 2

        ts = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        reward_ts = trial_start_ts + trial_data['reward_time']
        first_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[0] if utils.is_list(x) else x)
        second_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[1] if utils.is_list(x) else np.nan)

        for signal_type in signal_types:
            for align in alignments:
                for region in fp_data['processed_signals'].keys():
                    if region in regions:

                        signal = fp_data['processed_signals'][region][signal_type]
                        lims = xlims[align][region]

                        match align:
                            case Align.tone:
                                poke_in_baseline = np.nanmean(fp_utils.build_signal_matrix(signal, ts, cpoke_in_ts, -baseline_lims[0], baseline_lims[1])[0], axis=1)[:,None]
                                first_mat, _ = fp_utils.build_signal_matrix(signal, ts, first_tone_ts, -lims[0], lims[1])
                                second_mat, _ = fp_utils.build_signal_matrix(signal, ts, second_tone_ts, -lims[0], lims[1])
                                
                                first_mat_norm = first_mat - poke_in_baseline
                                second_mat_norm = second_mat - poke_in_baseline
                                
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['one_tone_rewarded'] = first_mat[responded & one_tone & rewarded,:]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['one_tone_unrewarded'] = first_mat[responded & one_tone & ~rewarded,:]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['one_tone_rewarded'] = first_mat_norm[responded & one_tone & rewarded,:]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['one_tone_unrewarded'] = first_mat_norm[responded & one_tone & ~rewarded,:]
                                
                                for v in variants:
                                    v_sel = variant == v
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['first_tone_var_'+v] = first_mat[responded & two_tone & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['second_tone_var_'+v] = second_mat[responded & two_tone & v_sel,:]
                                    
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['first_tone_hit_cong_var_'+v] = first_mat[responded & two_tone & ~incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['first_tone_hit_incong_var_'+v] = first_mat[responded & two_tone & incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['first_tone_miss_cong_var_'+v] = first_mat[responded & two_tone & ~incongruent & ~rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['first_tone_miss_incong_var_'+v] = first_mat[responded & two_tone & incongruent & ~rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['second_tone_hit_cong_var_'+v] = second_mat[responded & two_tone & ~incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['second_tone_hit_incong_var_'+v] = second_mat[responded & two_tone & incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['second_tone_miss_cong_var_'+v] = second_mat[responded & two_tone & ~incongruent & ~rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['second_tone_miss_incong_var_'+v] = second_mat[responded & two_tone & incongruent & ~rewarded & v_sel,:]
                                    
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['first_tone_var_'+v] = first_mat_norm[responded & two_tone & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['second_tone_var_'+v] = second_mat_norm[responded & two_tone & v_sel,:]
                                    
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['first_tone_hit_cong_var_'+v] = first_mat_norm[responded & two_tone & ~incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['first_tone_hit_incong_var_'+v] = first_mat_norm[responded & two_tone & incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['first_tone_miss_cong_var_'+v] = first_mat_norm[responded & two_tone & ~incongruent & ~rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['first_tone_miss_incong_var_'+v] = first_mat_norm[responded & two_tone & incongruent & ~rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['second_tone_hit_cong_var_'+v] = second_mat_norm[responded & two_tone & ~incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['second_tone_hit_incong_var_'+v] = second_mat_norm[responded & two_tone & incongruent & rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['second_tone_miss_cong_var_'+v] = second_mat_norm[responded & two_tone & ~incongruent & ~rewarded & v_sel,:]
                                    aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['second_tone_miss_incong_var_'+v] = second_mat_norm[responded & two_tone & incongruent & ~rewarded & v_sel,:]
                                    
                                    

                            case Align.cue:
                                # only look at trials where the cue happened before poking out
                                
                                mat, t = fp_utils.build_signal_matrix(signal, ts, cue_ts, -lims[0], lims[1])
                                baseline_sel = (t >= baseline_lims[0]) & (t <= baseline_lims[1])
                                mat_norm = mat - np.nanmean(mat[:, baseline_sel], axis=1)[:,None]
                                
                                # poke_out_after_cue_sel = cpoke_out_ts > cue_ts
                                # trial_sel = responded & poke_out_after_cue_sel
                                trial_sel = responded
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['all'] = mat[trial_sel, :]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['rewarded'] = mat[trial_sel & rewarded, :]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['unrewarded'] = mat[trial_sel & ~rewarded, :]
                                
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['all'] = mat_norm[trial_sel, :]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['rewarded'] = mat_norm[trial_sel & rewarded, :]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['unrewarded'] = mat_norm[trial_sel & ~rewarded, :]
                                
                            case Align.reward:
                                mat, _ = fp_utils.build_signal_matrix(signal, ts, reward_ts, -lims[0], lims[1])
                                cue_baseline = np.nanmean(fp_utils.build_signal_matrix(signal, ts, cue_ts, -baseline_lims[0], baseline_lims[1])[0], axis=1)[:,None]
                                
                                mat_norm = mat - cue_baseline 

                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['rewarded'] = mat[responded & rewarded, :]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['raw']['unrewarded'] = mat[responded & ~rewarded, :]
                                
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['rewarded'] = mat_norm[responded & rewarded, :]
                                aligned_signals[subj_id][sess_id][signal_type][align][region]['norm']['unrewarded'] = mat_norm[responded & ~rewarded, :]
                                    
                                
aligned_signals['t'] = {align: {region: [] for region in regions} for align in alignments}
dt = fp_data['dec_info']['decimated_dt']
for align in alignments:
    for region in regions:
        aligned_signals['t'][align][region] = np.arange(xlims[align][region][0], xlims[align][region][1]+dt, dt)

with open(save_path, 'wb') as f:
    pickle.dump(aligned_signals, f)

# %% Stack aligned signals

ignored_sessions = sess_info.loc[sess_info['proto_stage'] == 'ToneCatDelayResp2_7', 'sessid'].to_numpy()

ignored_signals = {'PL': [],
                   'DMS': []}

ignored_subjects = []

plot_regions = ['DMS', 'PL']
alignments = [Align.tone, Align.cue, Align.reward]
signal_types = ['z_dff_iso'] # 'dff_iso',
norm_types = ['raw', 'norm']

t = aligned_signals['t']
stacked_signals = {s: {a: {r: {n: {} for n in norm_types} 
                           for r in plot_regions} 
                       for a in alignments} 
                   for s in signal_types}

for signal_type in signal_types:
    for align in alignments:                
        for region in plot_regions:
            for norm_type in norm_types:
                
                groups = aligned_signals[subj_ids[0]][sess_ids[subj_ids[0]][0]][signal_type][align][region][norm_type].keys()
                
                for group in groups:
                    
                    stacked_mats = np.vstack([aligned_signals[subj_id][sess_id][signal_type][align][region][norm_type][group] 
                                              for subj_id in subj_ids if not subj_id in ignored_subjects
                                              for sess_id in sess_ids[subj_id] if (not sess_id in ignored_sessions and
                                                                                   not sess_id in ignored_signals[region] and 
                                                                                   signal_type in aligned_signals[subj_id][sess_id] and
                                                                                   align in aligned_signals[subj_id][sess_id][signal_type] and
                                                                                   group in aligned_signals[subj_id][sess_id][signal_type][align][region][norm_type])])
            
                    stacked_signals[signal_type][align][region][norm_type][group] = stacked_mats
                                                  

# define for plotting methods below
def calc_error(mat, use_se):
    if use_se:
        return utils.stderr(mat, axis=0)
    else:
        return np.nanstd(mat, axis=0, ddof=1)
# %% Plot average traces for the groups
    
plot_regions = ['DMS']
signal_type = 'z_dff_iso'
norm_type = 'norm' # 'raw' # 
norm_to_zero = False
baseline_lims = [-0.1, 0]

# plot cue and reward
aligns = [Align.cue, Align.reward]
groups = {Align.cue: ['rewarded', 'unrewarded'], Align.reward: ['rewarded', 'unrewarded']}
group_labels = {'all': 'All', 'rewarded': 'Correct', 'unrewarded': 'Incorrect'}
all_color = '#08AB36'
rew_color = '#BC141A'
unrew_color = '#1764AB'
colors = {'all': all_color, 'rewarded': rew_color, 'unrewarded': unrew_color}

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,0.8]},
             Align.reward: {'DMS': [-0.1,0.8], 'PL': [-0.3,10]}}

#width_ratios = [1.8,10.3]
width_ratios = [0.7,0.9]

n_rows = len(plot_regions)
n_cols = 2
t = aligned_signals['t']

x_label = 'Time (s)'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 4*n_rows), sharey='row', width_ratios=width_ratios)
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs)

    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_all_outcome_{}_{}'.format('_'.join(aligns), signal_type, '_'.join(plot_regions))

    for i, region in enumerate(plot_regions):
        for j, align in enumerate(aligns):
            match align:
                case Align.cue:
                    title = 'Response Cue'
                    
                case Align.reward:
                    title = 'Reward Delivery'

            ax = axs[i,j]
            
            region_signals = stacked_signals[signal_type][align][region][norm_type]
            t_r = t[align][region]
            t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
    
            for group in groups[align]:
                
                act = region_signals[group]
                if norm_to_zero:
                    baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                    act = act - np.nanmean(act[:, baseline_sel], axis=1)[:,None]

                error = calc_error(act, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[group], color=colors[group], plot_x0=False)
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)

            ax.set_xlabel(x_label)
            ax.legend()
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')
        
# %% plot single tone outcome

plot_regions = ['DMS']
signal_type = 'z_dff_iso'
norm_type = 'norm' # 'raw' # 
norm_to_zero = False
baseline_lims = [-0.1, 0]

align = Align.tone
groups = ['one_tone_rewarded', 'one_tone_unrewarded']
group_labels = {'one_tone_rewarded': 'Correct', 'one_tone_unrewarded': 'Incorrect'}
colors = {'one_tone_rewarded': rew_color, 'one_tone_unrewarded': unrew_color}

plot_lims = {'DMS': [-0.1,0.6], 'PL': [-0.5,1.25]}

n_rows = len(plot_regions)
n_cols = 1
t = aligned_signals['t']

x_label = 'Time (s)'
title = 'Single Tone'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_one_tone_outcome_{}_{}'.format(align, signal_type, '_'.join(plot_regions))

    for i, region in enumerate(plot_regions):
            
        ax = axs[i]
        
        region_signals = stacked_signals[signal_type][align][region][norm_type]
        t_r = t[align][region]
        t_sel = (t_r > plot_lims[region][0]) & (t_r < plot_lims[region][1])

        for group in groups:
            act = region_signals[group]
            if norm_to_zero:
                baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                act = act - np.nanmean(act[:, baseline_sel], axis=1)[:,None]
            
            error = calc_error(act, True)
            
            plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[group], color=colors[group], plot_x0=False)

        ax.set_title('{} {}'.format(region, title))
        plot_utils.plot_dashlines([0, 0.3], ax=ax)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend()
        
    fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')
    
# %% plot two tone by variant

plot_regions = ['DMS']
signal_type = 'z_dff_iso'
norm_type = 'norm' # 'raw' # 
norm_to_zero = False
baseline_lims = [-0.1, 0]

align = Align.tone
gen_plot_groups = ['second_tone_var_'] # 'first_tone_var_', 
var_labels = {'first': 'First', 'last': 'Last'}
groups = [[g+v for v in variants] for g in gen_plot_groups]
group_labels = {g+v: var_labels[v] for v in variants for g in gen_plot_groups}

plot_lims = {'DMS': [-0.4,0.8], 'PL': [-0.5,1]}
y_lim = [-0.25, 0.15]

n_rows = len(plot_regions)
n_cols = len(groups)
t = aligned_signals['t']

x_label = 'Time (s)'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(4*n_cols, 4*n_rows), sharey='row')
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
        
    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_two_tone_variant_{}_{}'.format(align, signal_type, '_'.join(plot_regions))

    for i, region in enumerate(plot_regions):
        for j, sub_groups in enumerate(groups):     
            ax = axs[i,j]
            
            title = 'First Tone' if j == 0 else 'Second Tone'
            region_signals = stacked_signals[signal_type][align][region][norm_type]
            t_r = t[align][region]
            t_sel = (t_r > plot_lims[region][0]) & (t_r < plot_lims[region][1])
    
            for group in sub_groups:
                act = region_signals[group]
                if norm_to_zero:
                    baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                    act = act - np.nanmean(act[:, baseline_sel], axis=1)[:,None]
                
                error = calc_error(act, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[group], plot_x0=False)
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines([0, 0.3], ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
    
            ax.set_xlabel(x_label)
            ax.legend(title='Response Rule')
            ax.set_ylim(y_lim)
        
    fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')
    
# %% plot two tone by variant, for incongruent hits and misses

plot_regions = ['DMS', 'PL']
signal_type = 'z_dff_iso'
norm_type = 'norm' # 'raw' # 
norm_to_zero = False
baseline_lims = [-0.1, 0]

align = Align.tone
gen_plot_groups = [['first_tone_hit_incong_var_', 'first_tone_miss_incong_var_'], ['second_tone_hit_incong_var_', 'second_tone_miss_incong_var_']]
var_labels = {'first': 'First', 'last': 'Last'}
groups = [[sub_g+v for v in variants for sub_g in g] for g in gen_plot_groups]
gen_group_labels = {'first_tone_hit_incong_var_': '{}, Correct', 'first_tone_miss_incong_var_': '{}, Incorrect', 
                    'second_tone_hit_incong_var_': '{}, Correct', 'second_tone_miss_incong_var_': '{}, Incorrect'}
group_labels = {k+v: l.format(var_labels[v]) for v in variants for k,l in gen_group_labels.items()}

plot_lims = {'DMS': [-0.4,0.8], 'PL': [-0.5,1]}

n_rows = len(plot_regions)
n_cols = len(groups)
t = aligned_signals['t']

x_label = 'Time (s)'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(8, 4*n_rows), sharey='row')
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
        
    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_two_tone_incong_variant_{}_{}'.format(align, signal_type, '_'.join(plot_regions))

    for i, region in enumerate(plot_regions):
        for j, sub_groups in enumerate(groups):     
            ax = axs[i,j]
            
            title = 'First Tone' if j == 0 else 'Second Tone'
            region_signals = stacked_signals[signal_type][align][region][norm_type]
            t_r = t[align][region]
            t_sel = (t_r > plot_lims[region][0]) & (t_r < plot_lims[region][1])
    
            for group in sub_groups:
                act = region_signals[group]
                if norm_to_zero:
                    baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                    act = act - np.nanmean(act[:, baseline_sel], axis=1)[:,None]
                
                error = calc_error(act, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[group], plot_x0=False)
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines([0, 0.3], ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
    
            ax.set_xlabel(x_label)
            ax.legend(title='Response Rule')
        
    fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')

