# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 15:16:38 2024

@author: tanne
"""

# %% imports

import init
from hankslab_db import db_access
import hankslab_db.basicRLtasks_db as db
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

from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf


# %% Declare subject information

# get all session ids for given protocol
sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2)
subj_ids = list(sess_ids.keys())

reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)
implant_info = db_access.get_fp_implant_info(subj_ids)

# %% Set up variables
signal_types = ['dff_iso', 'z_dff_iso']
alignments = [Align.cue, Align.reward]
regions = ['DMS', 'PL']
xlims = {'DMS': [-1,2], 'PL': [-3,15]}
recalculate = False

filename = 'two_arm_bandit_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        aligned_signals = saved_data['aligned_signals']
        aligned_metadata = saved_data['metadata']
else:
    aligned_signals = {subjid: {sessid: {sig_type: {align: {region: [] for region in regions} 
                                                    for align in alignments}
                                         for sig_type in signal_types}
                                for sessid in sess_ids[subjid]} 
                       for subjid in subj_ids}
                       

# %% Build signal matrices aligned to alignment points

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:
        if sess_id in fpah.__sess_ignore:
            continue

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, fit_baseline=False)
        fp_data = fp_data[subj_id][sess_id]

        trial_data = sess_data[sess_data['sessid'] == sess_id]

        ts = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        reward_ts = trial_start_ts + trial_data['reward_time']

        for signal_type in signal_types:
            for align in alignments:
                for region in fp_data['processed_signals'].keys():
                    if region in regions:
                        signal = fp_data['processed_signals'][region][signal_type]

                        match align:
                            case Align.cue:
                                align_ts = cue_ts
                                
                            case Align.reward:
                                align_ts = reward_ts

                        lims = xlims[region]
                        
                        mat, t = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1])
                        aligned_signals[subj_id][sess_id][signal_type][align][region] = mat

aligned_signals['t'] = {region: [] for region in regions}
dt = fp_data['dec_info']['decimated_dt']
for region in regions:
    aligned_signals['t'][region] = np.arange(xlims[region][0], xlims[region][1]+dt, dt)

with open(save_path, 'wb') as f:
    pickle.dump({'aligned_signals': aligned_signals,
                 'metadata': {'signal_types': signal_types,
                             'alignments': alignments,
                             'regions': regions,
                             'xlims': xlims}}, f)

# %% Analyze aligned signals

ignored_signals = {'PL': [96556, 101853, 101906, 101958, 102186, 102235, 102288, 102604],
                   'DMS': [96556, 102604]}

ignored_subjects = [182] # [179]

rew_hist_n_back = 10
rew_rate_n_back = 3
bah.get_rew_rate_hist(sess_data, n_back=rew_rate_n_back, kernel='uniform')

# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
n_rew_hist_bins = 4
rew_hist_bin_edges = np.linspace(-0.001, 1.001, n_rew_hist_bins+1)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{:.0f}-{:.0f}%'.format(abs(b.left)*100, b.right*100) for b in rew_hist_bins}

alignments = [Align.cue, Align.reward]
signal_types = ['z_dff_iso', 'dff_iso']

analyze_peaks = True

filter_props = {Align.cue: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 2}},
                Align.reward: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}}}

peak_find_props = {Align.cue: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.4, 'peak_edge_buffer': 0.08},
                               'PL': {'min_dist': 0.2, 'peak_tmax': 1.5, 'peak_edge_buffer': 0.2}},
                   Align.reward: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08},
                                  'PL': {'min_dist': 0.5, 'peak_tmax': 3.5, 'peak_edge_buffer': 0.2}}}

sides = ['contra', 'ipsi']

t = aligned_signals['t']
stacked_signals = {s: {a: {r: {} for r in regions} 
                       for a in alignments} 
                   for s in signal_types}
peak_metrics = []

def stack_mat(stacked_mats, key, mat):
    if not key in stacked_mats:
        stacked_mats[key] = np.zeros((0, mat.shape[1]))
    else:
        stacked_mats[key] = np.vstack((stacked_mats[key], mat))

for subj_id in subj_ids:
    if subj_id in ignored_subjects:
        continue
    print('Analyzing peaks for subj {}'.format(subj_id))
    for sess_id in sess_ids[subj_id]:
        
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        rewarded = trial_data['rewarded'].to_numpy()
        responded = ~np.isnan(trial_data['response_time']).to_numpy()
        rew_hist = pd.cut(trial_data['rew_rate_hist_all'], rew_hist_bins)
        choice = trial_data['choice']

        for signal_type in signal_types:
            if not signal_type in aligned_signals[subj_id][sess_id]:
                continue
            for align in alignments:
                if not align in aligned_signals[subj_id][sess_id][signal_type]:
                    continue

                for region in regions:
                    if sess_id in ignored_signals[region]:
                        continue
                    
                    t_r = t[region]
                    mat = aligned_signals[subj_id][sess_id][signal_type][align][region]
                    region_side = implant_info[subj_id][region]['side']
                    choice_side = choice.apply(lambda x: fpah.get_implant_side_type(x, region_side) if not x == 'none' else 'none').to_numpy()
                    
                    # calculate peak properties on a trial-by-trial basis
                    if analyze_peaks:
                        contra_choices = choice_side == 'contra'
                        
                        for i in range(mat.shape[0]):
                            if responded[i]:
                                metrics = fpah.calc_peak_properties(mat[i,:], t_r, 
                                                                    filter_params=filter_props[align][region],
                                                                    peak_find_params=peak_find_props[align][region],
                                                                    fit_decay=False)
                                
                                if i < rew_hist_n_back:
                                    buffer = np.full(rew_hist_n_back-i, False)
                                    rew_hist_vec = np.flip(np.concatenate((buffer, rewarded[:i])))
                                    contra_hist_vec = np.flip(np.concatenate((buffer, contra_choices[:i])))
                                    ipsi_hist_vec = np.flip(np.concatenate((buffer, ~contra_choices[:i])))
                                else:
                                    rew_hist_vec = np.flip(rewarded[i-rew_hist_n_back:i])
                                    contra_hist_vec = np.flip(contra_choices[i-rew_hist_n_back:i])
                                    ipsi_hist_vec = np.flip(~contra_choices[i-rew_hist_n_back:i])
    
                                peak_metrics.append(dict([('subj_id', subj_id), ('sess_id', sess_id), ('signal_type', signal_type), 
                                                         ('align', align.name), ('region', region), ('trial', i),
                                                         ('rewarded', rewarded[i]), ('rew_hist_bin', rew_hist.iloc[i]), 
                                                         ('side', choice_side[i]), ('rew_hist', rew_hist_vec),
                                                         ('contra_hist', contra_hist_vec), ('ipsi_hist', ipsi_hist_vec),
                                                         *metrics.items()]))
                    
                    # normalize all grouped matrices to the pre-event signal of the lowest reward rate
                    baseline_mat = mat[(rew_hist == rew_hist_bins[0]) & responded,:]
                    if baseline_mat.shape[0] > 0:
                        baseline_sel = (t_r >= -0.1) & (t_r < 0)
                        baseline = np.nanmean(baseline_mat[:,baseline_sel])
                    else:
                        baseline = 0
                        
                    mat = mat - baseline
                    
                    # group trials together and stack across sessions
                    for rew_bin in rew_hist_bins:
                        rew_sel = rew_hist == rew_bin
                        bin_str = rew_hist_bin_strs[rew_bin]
                        
                        stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded', mat[rew_sel & responded & rewarded,:])
                        stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded', mat[rew_sel & responded & ~rewarded,:])
                        
                        for side in sides:
                            side_sel = choice_side == side
                            stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded_'+side, mat[rew_sel & responded & rewarded & side_sel,:])
                            stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded_'+side, mat[rew_sel & responded & ~rewarded & side_sel,:])

                        match align:
                            case Align.cue:
                                
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str, mat[rew_sel & responded,:])

                                for side in sides:
                                    side_sel = choice_side == side
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_'+side, mat[rew_sel & responded & side_sel,:])

                            #case Align.reward:

peak_metrics = pd.DataFrame(peak_metrics)

# %% declare common plotting stuff

def calc_error(mat, use_se):
    if use_se:
        return utils.stderr(mat, axis=0)
    else:
        return np.nanstd(mat, axis=0, ddof=1)
    
side_labels = {'ipsi': 'Ipsi', 'contra': 'Contra'}
rew_labels = {'rewarded': 'rew', 'unrewarded': 'unrew'}
#bin_labels = {b:'{:.0f}-{:.0f}'.format(np.abs(np.ceil(b.left*rew_rate_n_back)), np.floor(b.right*rew_rate_n_back)) for b in rew_hist_bins}
bin_labels = {b:'{:.0f}'.format(np.mean([np.abs(np.ceil(b.left*rew_rate_n_back)), np.floor(b.right*rew_rate_n_back)])) for b in rew_hist_bins}
group_labels_dict = {'rew_hist_{}_{}'.format(rew_hist_bin_strs[rew_bin], rk): '{} {}'.format(bin_labels[rew_bin], rv)
                for rew_bin in rew_hist_bins
                for rk, rv in rew_labels.items()}
group_labels_dict.update({'rew_hist_{}'.format(rew_hist_bin_strs[rew_bin]): bin_labels[rew_bin]
                for rew_bin in rew_hist_bins})
group_labels_dict.update({'rew_hist_{}_{}'.format(rew_hist_bin_strs[rew_bin], side_type): '{} {}'.format(bin_labels[rew_bin], side_label)
                for rew_bin in rew_hist_bins 
                for side_type, side_label in side_labels.items()})
group_labels_dict.update({'rew_hist_{}_{}_{}'.format(rew_hist_bin_strs[rew_bin], rk, side_type): '{} {} {}'.format(bin_labels[rew_bin], side_label, rv)
                for rew_bin in rew_hist_bins 
                for side_type, side_label in side_labels.items()
                for rk, rv in rew_labels.items()})

# %% Plot average traces for the groups
plot_regions = ['DMS']
plot_aligns = [Align.cue, Align.reward]

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
gen_groups = {Align.cue: ['rew_hist_{}'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
groups = {a: [group.format(rew_hist_bin_strs[rew_bin]) for group in gen_groups[a] for rew_bin in rew_hist_bins] for a in plot_aligns}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.1,0.8], 'PL': [-0.5,10]}}

#width_ratios = [2,10.5]
width_ratios = [0.7,0.9]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 4*n_rows), sharey='row', width_ratios=width_ratios)
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs)

    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_reward_hist_{}_back_{}'.format('_'.join(plot_aligns), rew_rate_n_back, signal_type)

    for i, region in enumerate(plot_regions):
        for j, align in enumerate(plot_aligns):
            match align:
                case Align.cue:
                    title = 'Response Cue'
                    legend_cols = 1
                    
                case Align.reward:
                    title = 'Reward Delivery'
                    legend_cols = 2

            ax = axs[i,j]
            
            region_signals = stacked_signals[signal_type][align][region]
    
            for group, color in zip(groups[align], colors[align]):
                act = region_signals[group]
                t_r = t[region]
                t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                error = calc_error(act, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels_dict[group], color=color, plot_x0=False)
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
            #else:
                # ax.yaxis.set_tick_params(which='both', labelleft=True)
            ax.legend(ncols=legend_cols, loc='upper right', title='# Rewards in last {} Trials'.format(rew_rate_n_back))

            ax.set_xlabel(x_label)
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
        
# %% Plot average traces for the groups by side
plot_regions = ['DMS']
plot_aligns = [Align.cue, Align.reward]

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}
gen_groups = {Align.cue: ['rew_hist_{}_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.1,0.8], 'PL': [-0.5,10]}}

#width_ratios = [2,10.5]
width_ratios = [0.7,0.9]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    for side_type in sides:
        groups = {a: [group.format(rew_hist_bin_strs[rew_bin], side_type) for group in gen_groups[a] for rew_bin in rew_hist_bins] for a in plot_aligns}
    
        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 4*n_rows), sharey='row', width_ratios=width_ratios)
        
        if n_rows == 1 and n_cols == 1:
            axs = np.array(axs)
    
        axs = axs.reshape((n_rows, n_cols))
        
        fig.suptitle('{} {}'.format(side_type, signal_title))
        
        plot_name = '{}_reward_hist_{}_back_{}_{}'.format('_'.join(plot_aligns), rew_rate_n_back, side_type, signal_type)
    
        for i, region in enumerate(plot_regions):
            for j, align in enumerate(plot_aligns):
                match align:
                    case Align.cue:
                        title = 'Response Cue'
                        legend_cols = 1
                        
                    case Align.reward:
                        title = 'Reward Delivery'
                        legend_cols = 2
    
                ax = axs[i,j]
                
                region_signals = stacked_signals[signal_type][align][region]
        
                for group, color in zip(groups[align], colors[align]):
                    act = region_signals[group]
                    t_r = t[region]
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    error = calc_error(act, True)
                    
                    plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels_dict[group], color=color, plot_x0=False)
        
                ax.set_title('{} {}'.format(region, title))
                plot_utils.plot_dashlines(0, ax=ax)
        
                if j == 0:
                    ax.set_ylabel(y_label)
                #else:
                    # ax.yaxis.set_tick_params(which='both', labelleft=True)
                ax.legend(ncols=legend_cols, loc='upper right', title='# Rewards in last {} Trials'.format(rew_rate_n_back))
    
                ax.set_xlabel(x_label)
                
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

# %% compare peak height dffs with z-scores
cols = ['subj_id', 'sess_id', 'align', 'region', 'group_label', 'rewarded', 'peak_height']
sub_metrics = peak_metrics.loc[peak_metrics['signal_type'] == 'dff_iso', cols].reset_index()
sub_metrics.rename(columns={'peak_height': 'height_dff'}, inplace=True)
sub_metrics['height_zdff'] = peak_metrics.loc[peak_metrics['signal_type'] == 'z_dff_iso', 'peak_height'].to_numpy()
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}

sub_subj_ids = np.unique(sub_metrics['subj_id'])

n_regions = len(regions)
n_aligns = len(alignments)

for subj_id in sub_subj_ids:
    fig, axs = plt.subplots(n_regions, n_aligns, figsize=(4*n_aligns, 4*n_regions), layout='constrained')
    fig.suptitle('Peak Heights, {}'.format(subj_id))

    for i, region in enumerate(regions):
        for j, align in enumerate(alignments):

            match align:
                case Align.cue:
                    region_metrics = sub_metrics[(sub_metrics['region'] == region) & (sub_metrics['align'] == align) 
                                                  & (sub_metrics['subj_id'] == subj_id)]
                case Align.reward:
                    # only show rewarded peaks heights in reward alignment
                    region_metrics = sub_metrics[(sub_metrics['region'] == region) & (sub_metrics['align'] == align) 
                                                  & (sub_metrics['subj_id'] == subj_id) & sub_metrics['rewarded']]
                    
            ax = axs[i,j]
            for k, group in enumerate(np.unique(region_metrics['group_label'])):
                group_metrics = region_metrics[region_metrics['group_label'] == group]
                ax.scatter(group_metrics['height_dff'], group_metrics['height_zdff'], label=group, alpha=0.5, c='C'+str(k))
                
            for k, group in enumerate(np.unique(region_metrics['group_label'])):
                group_metrics = region_metrics[region_metrics['group_label'] == group]
                ax.scatter(np.nanmean(group_metrics['height_dff']), np.nanmean(group_metrics['height_zdff']), c='C'+str(k), marker='x', s=500, linewidths=3, label='_')
                
            ax.set_title('{} {}'.format(region, align_labels[align]))
            ax.set_xlabel('dF/F')
            ax.set_ylabel('z-scored dF/F')
            ax.legend()
            
# %% prep peak properties for analysis

# make subject ids categories
peak_metrics['subj_id'] = peak_metrics['subj_id'].astype('category')
peak_metrics['rew_hist_bin_label'] = peak_metrics['rew_hist_bin'].apply(lambda x: bin_labels[x])

ignore_outliers = True
outlier_thresh = 10

t_min = 0.01
t_max = {a: {r: peak_find_props[a][r]['peak_tmax'] - 0.01 for r in regions} for a in alignments} 

parameters = ['peak_time', 'peak_height', 'peak_width'] #, 'decay_tau'

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
    #         ax.plot(t[row['region']], mat[row['trial'], :])
    #         plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
    #         peak_idx = np.argmin(np.abs(t[row['region']] - row['peak_time']))
    #         ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
    #         ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')

    filt_peak_metrics = filt_peak_metrics[peak_sel]
    
    # first add iqr multiple columns
    for param in parameters:
        filt_peak_metrics['iqr_mult_'+param] = np.nan
    
    # calculate iqr multiple for potential outliers
    outlier_grouping = ['subj_id', 'sess_id', 'signal_type']
    
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
            #         ax.plot(t[row['region']], mat[row['trial'], :])
            #         plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
            #         peak_idx = np.argmin(np.abs(t[row['region']] - row['peak_time']))
            #         ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
            #         ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')
        
            filt_peak_metrics.loc[outlier_sel, param] = np.nan


# %% Plot peak properties

parameters = ['peak_time', 'peak_height', 'peak_width'] # 'decay_tau'
parameter_labels = {'peak_time': 'Time to peak (s)', 'peak_height': 'Peak height ({})',
                    'peak_width': 'Peak FWHM (s)', 'decay_tau': 'Decay τ (s)'}

subj_order = np.unique(filt_peak_metrics['subj_id'])

for signal_type in signal_types:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    for align in alignments:
        peak_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['signal_type'] == signal_type)
        
        match align:
            case Align.cue:
                base_title = 'Response Cue Transients - {}, {}'
                x_label = '# Rewards in last {} Trials'.format(rew_rate_n_back)
            case Align.reward:
                base_title = 'Reward Transients - {}, {}'
                x_label = '# Rewards in last {} Trials'.format(rew_rate_n_back)
                peak_sel = peak_sel & filt_peak_metrics['rewarded']
    
        for region in regions:
            region_metrics = filt_peak_metrics[peak_sel & (filt_peak_metrics['region'] == region)]
    
            # group by reward history group, colored by subject
            fig, axs = plt.subplots(1,4, figsize=(16,4), layout='constrained', width_ratios=[1,1,1,0.3])
            fig.suptitle(base_title.format(region, signal_label))
    
            for i, param in enumerate(parameters):
                ax = axs[i]
                sb.stripplot(data=region_metrics, x='rew_hist_bin_label', y=param, hue='subj_id', ax=ax,
                             hue_order=subj_order, alpha=0.6, legend=False, jitter=0.25)
                ax.set_xlabel(x_label)
                if param == 'peak_height':
                    ax.set_ylabel(parameter_labels[param].format(y_label))
                else:
                    ax.set_ylabel(parameter_labels[param])
    
            # add legend
            ax = axs[-1]
            colors = sb.color_palette()
            patches = [Patch(label=subj, color=colors[i], alpha=0.8) for i, subj in enumerate(subj_order)]
            ax.legend(patches, subj_order, loc='center', frameon=False, title='Subjects')
            ax.set_axis_off()


# %% make combined comparison figures per peak property

parameters = ['peak_height'] # 'peak_time', 'peak_height', 'peak_width', 'decay_tau'
parameter_titles = {'peak_time': 'Time to Peak', 'peak_height': 'Peak Height',
                    'peak_width': 'Peak Width', 'decay_tau': 'Decay τ'}
parameter_labels = {'peak_time': 'Time to peak (s)', 'peak_height': 'Peak height ({})',
                    'peak_width': 'Peak FWHM (s)', 'decay_tau': 'Decay τ (s)'}

rew_hist_rew_colors = plt.cm.seismic(np.linspace(0.6,1,len(rew_hist_bins)))
rew_hist_palette = sb.color_palette(rew_hist_rew_colors) 

split_by_side = True
sides = ['contra', 'ipsi']
n_regions = len(regions)
align_order = ['cue', 'reward']
n_aligns = len(align_order)
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}
hatch_order = ['//\\\\', '']
line_order = ['dashed', 'solid']

for signal_type in signal_types:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    for param in parameters:
        
        plot_name = '{}_reward_hist_{}_{}_back_{}'.format('_'.join(align_order), param, rew_rate_n_back, signal_type)
    
        # Compare responses across alignments grouped by region
        fig, axs = plt.subplots(n_regions, n_aligns, figsize=(4*n_aligns, 4*n_regions), layout='constrained', sharey='row')
        fig.suptitle('{}, {}'.format(parameter_titles[param], signal_label))
    
        for i, region in enumerate(regions):
            for j, align in enumerate(align_order):
                peak_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['signal_type'] == signal_type) & (filt_peak_metrics['region'] == region)
    
                match align:
                    #case Align.cue:

                    case Align.reward:
                        peak_sel = peak_sel & filt_peak_metrics['rewarded']

                region_metrics = filt_peak_metrics[peak_sel]
                
                ax = axs[i,j]
                ax.set_title('{} {}'.format(region, align_labels[align]))
                
                if split_by_side:
                    # plot reward history group averages in boxplots
                    sb.boxplot(data=region_metrics, x='rew_hist_bin_label', y=param,
                               hue='side', hue_order=sides, ax=ax, showfliers=False, legend=False)
                    
                    # update colors and fills of boxes
                    for k, patch in enumerate(ax.patches):
                        # Left boxes first, then right boxes
                        if k < len(rew_hist_bins):
                            # add hatch to cues
                            patch.set_hatch(hatch_order[int(k/len(rew_hist_bins))])

                        patch.set_facecolor(rew_hist_palette[k % len(rew_hist_bins)])

                    # add subject averages for each alignment with lines connecting them
                    subj_avgs = region_metrics.groupby(['subj_id', 'rew_hist_bin_label', 'side']).agg({param: np.nanmean}).reset_index()
                    
                    group_labels = np.unique(region_metrics['rew_hist_bin_label'])
                    region_subj_ids = np.unique(region_metrics['subj_id'])
                    dodge = [-0.2, 0.2]
                    for k, (side, d) in enumerate(zip(sides, dodge)):
                        x = np.arange(len(group_labels)) + d
                        for subj_id in region_subj_ids:
                            subj_avg = subj_avgs[(subj_avgs['subj_id'] == subj_id) & (subj_avgs['side'] == side)]
                            y = [subj_avg.loc[subj_avg['rew_hist_bin_label'] == g, param] for g in group_labels]
                            ax.plot(x, y, color='black', marker='o', linestyle=line_order[k], alpha=0.7)

                    # Add the custom legend to the figure (or to one of the subplots)
                    legend_patches = [Patch(facecolor='none', edgecolor='black', hatch=hatch_order[k], label=side_labels[s]) for k, s in enumerate(sides)]
                    ax.legend(handles=legend_patches, frameon=False)
                else:
                    # plot reward history group averages in boxplots
                    sb.boxplot(data=region_metrics, x='rew_hist_bin_label', y=param,
                               hue='rew_hist_bin_label', palette=rew_hist_palette,
                               ax=ax, showfliers=False, legend=False)
    
                    # add subject averages for each alignment with lines connecting them
                    subj_avgs = region_metrics.groupby(['subj_id', 'rew_hist_bin_label']).agg({param: np.nanmean}).reset_index()
            
                    group_labels = np.unique(region_metrics['rew_hist_bin_label'])
                    region_subj_ids = np.unique(region_metrics['subj_id'])
                    for subj_id in region_subj_ids:
                        subj_avg = subj_avgs[subj_avgs['subj_id'] == subj_id]
                        y = [subj_avg.loc[subj_avg['rew_hist_bin_label'] == g, param] for g in group_labels]
        
                        ax.plot(np.arange(len(group_labels)), y, color='black', marker='o', linestyle='dashed', alpha=0.7)
                        
                if param == 'peak_height':
                    ax.set_ylabel(parameter_labels[param].format(y_label))
                else:
                    ax.set_ylabel(parameter_labels[param])
                ax.set_xlabel('# Rewards in last {} Trials'.format(rew_rate_n_back))
                
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')


# %% Perform statistical tests on the peak properties

# look at each region and property separately
parameters = ['peak_height'] # ['peak_time', 'peak_height', 'peak_width', 'decay_tau'] #
regions = ['DMS', 'PL']
aligns = [Align.cue, Align.reward]
signals = ['dff_iso']
include_side = False

for signal_type in signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    for param in parameters:
        for region in regions:
            for align in aligns:
                
                match align:
                    case Align.cue:
                        region_metrics = filt_peak_metrics[(filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == align) 
                                                      & (filt_peak_metrics['signal_type'] == signal_type)]
                        #re_form = '1 + side'
                        #vc = {'subj_id': '0 + C(subj_id)', 'side': '0 + C(side)'}
                    case Align.reward:
                        # only show rewarded peaks heights in reward alignment
                        region_metrics = filt_peak_metrics[(filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == align) 
                                                      & (filt_peak_metrics['signal_type'] == signal_type) & filt_peak_metrics['rewarded']]
                        #re_form = '1 + side'
                        #vc = {'subj_id': '0 + C(subj_id)'}
                        
                # drop nans
                region_metrics = region_metrics[~np.isnan(region_metrics[param])]
                
                if include_side:
                    vc_form={'subj_id': '0 + C(subj_id)', 'side': '0 + C(side)'}
                else:
                    vc_form={'subj_id': '0 + C(subj_id)'}
        
                # group_lm = ols(param+' ~ C(group_label)', data=region_metrics).fit()
                # subj_lm = ols(param+' ~ C(subj_id)', data=region_metrics).fit()
                # group_subj_lm = ols(param+' ~ C(group_label)*C(subj_id)', data=region_metrics).fit()
                # subj_group_lm = ols(param+' ~ C(subj_id)*C(group_label)', data=region_metrics).fit()
                
                # # print('{} {}-aligned variant model fit:\n {}\n'.format(region, align, variant_lm.summary()))
                # # print('{} {}-aligned variant & subject model fit:\n {}\n'.format(region, align, variant_subj_lm.summary()))
                
                # print('{}: {} {}, {}-aligned reward history model ANOVA:\n {}\n'.format(param, signal_type, region, align, anova_lm(group_lm)))
                # print('{}: {} {}, {}-aligned subject model ANOVA:\n {}\n'.format(param, signal_type, region, align, anova_lm(subj_lm)))
                # print('{}: {} {}, {}-aligned reward history & subject model ANOVA:\n {}\n'.format(param, signal_type, region, align, anova_lm(group_subj_lm)))
                # print('{}: {} {}, {}-aligned subject & reward history model ANOVA:\n {}\n'.format(param, signal_type, region, align, anova_lm(subj_group_lm)))
                
                # print('{}: {} {}, {}-aligned comparison between reward history & reward history/subject models:\n {}\n'.format(param, signal_type, region, align, anova_lm(group_lm, group_subj_lm)))
                # print('{}: {} {}, {}-aligned comparison between subject & subject/reward history models:\n {}\n'.format(param, signal_type, region, align, anova_lm(subj_lm, subj_group_lm)))
                
                        
                # mem = sm.MixedLM.from_formula(param+' ~ C(group_label)', groups='subj_id', data=region_metrics, missing='drop')
                # print('{}: {} {}, {}-aligned fixed variants, random subjects:\n {}\n'.format(param, signal_type, region, align, mem.fit().summary()))
                
                # Create N mixed effects models where the first reward history group is rotated to compare it with all other groups
                rew_hist_groups = np.unique(region_metrics['rew_hist_bin_label'])
                rew_hist_vals = region_metrics['rew_hist_bin_label'].to_numpy()
                param_vals = region_metrics[param].to_numpy()
                subj_id_vals = region_metrics['subj_id'].to_numpy()
                side_vals = region_metrics['side'].to_numpy()
                
                for first_group in rew_hist_groups:
                    group_mapping = {first_group: '0'}
                    for other_group in rew_hist_groups[~np.isin(rew_hist_groups, first_group)]:
                        group_mapping.update({other_group: 'bin_'+other_group})
                        
                    group_vals = np.array([group_mapping[g] for g in rew_hist_vals])
                    
                    model_data = pd.DataFrame.from_dict({'val': param_vals, 'subj_id': subj_id_vals, 'group': group_vals, 'side': side_vals})

                    mem = sm.MixedLM.from_formula('val ~ C(group)', groups='subj_id', vc_formula=vc_form, data=model_data, missing='drop')
                        
                    print('{}: {} {}, {}-aligned Mixed-effects Model, \'{}\' group compared against other groups:\n {}\n'.format(
                           param, signal_type, region, align, first_group, mem.fit().summary()))
                    
                
# %% perform n-back regression on peak height

# look at each region and property separately
parameters = ['peak_height'] # ['peak_time', 'peak_height', 'peak_width', 'decay_tau'] #
regions = ['DMS', 'PL']
aligns = [Align.cue, Align.reward]
signals = ['z_dff_iso'] # 'dff_iso', 

include_current_side = False
include_side_reward_hist = True
fit_subj_separate = False
lim_n_back = 4 # rew_hist_n_back

for signal_type in signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    for param in parameters:
        for region in regions:
            for align in aligns:
                
                match align:
                    case Align.cue:
                        region_metrics = filt_peak_metrics[(filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == align) 
                                                      & (filt_peak_metrics['signal_type'] == signal_type)]
                    case Align.reward:
                        # only show rewarded peaks heights in reward alignment
                        region_metrics = filt_peak_metrics[(filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == align) 
                                                      & (filt_peak_metrics['signal_type'] == signal_type) & filt_peak_metrics['rewarded']]
                        
                # drop nans
                region_metrics = region_metrics[~np.isnan(region_metrics[param])]
                        
                # build predictor matrix labels
                if include_side_reward_hist:
                    predictor_labels = ['rewarded contra ({})'.format(i) for i in range(-1, -lim_n_back-1, -1)]
                    predictor_labels.extend(['rewarded ipsi ({})'.format(i) for i in range(-1, -lim_n_back-1, -1)])
                else:
                    predictor_labels = ['reward ({})'.format(i) for i in range(-1, -lim_n_back-1, -1)]
                
                if include_current_side:
                    predictor_labels.append('contra choice')
                    
                param_vals = region_metrics[param].reset_index(drop=True)
                
                # build predictor matrix
                if include_side_reward_hist:
                    predictor_vals = np.vstack(region_metrics.apply(lambda r: np.array(r['contra_hist'][:lim_n_back]) * np.array(r['rew_hist'][:lim_n_back]), axis=1))
                    predictor_vals = np.hstack([predictor_vals, np.vstack(region_metrics.apply(lambda r: np.array(r['ipsi_hist'][:lim_n_back]) * np.array(r['rew_hist'][:lim_n_back]), axis=1))])
                else:
                    predictor_vals = np.vstack(region_metrics['rew_hist'].apply(lambda x: x[:lim_n_back]))
                
                if include_current_side:
                    predictor_vals = np.hstack([predictor_vals, (region_metrics['side'].to_numpy()[:,None] == 'contra')])
                    
                predictors = pd.DataFrame(predictor_vals, columns=predictor_labels)
                predictors = sm.add_constant(predictors)

                mem = sm.MixedLM(param_vals, predictors, groups=region_metrics['subj_id']).fit()
                print('{}: {} {}, {}-aligned All Subjects Mixed-effects Regression Results:\n{}\n'.format(param, signal_type, region, align, mem.summary()))
                
                params = mem.params
                cis = mem.conf_int(0.05)
                
                if fit_subj_separate:
                    for subj_id in np.unique(region_metrics['subj_id']):
                        subj_sel = region_metrics['subj_id'].to_numpy() == subj_id
    
                        mem = sm.OLS(param_vals[subj_sel], predictors.loc[subj_sel,:]).fit()
                        print('{}: {} {}, {}-aligned Subject {} OLS Regression Results:\n{}\n'.format(param, signal_type, region, align, subj_id, mem.summary()))



    # # plot regression coefficients over trials back
    # fig, axs = plt.subplots(1, len(reg_groups), layout='constrained', figsize=(4*len(reg_groups), 4), sharey=True)
    # fig.suptitle('Choice Regression Coefficients by Block Reward Rate (Rat {})'.format(subj_id))

    # x_vals = np.arange(n_back)+1

    # for i, group in enumerate(reg_groups):

    #     fit_res = reg_results[i]
    #     params = fit_res.params
    #     cis = fit_res.conf_int(0.05)

    #     ax = axs[i]
    #     ax.set_title('Block Rate: {}'.format(group))
    #     plot_utils.plot_dashlines(0, dir='h', ax=ax)

    #     # plot constant
    #     key = 'const'
    #     ax.errorbar(0, params[key], yerr=np.abs(cis.loc[key,:] - params[key]).to_numpy()[:,None], fmt='o', capsize=4, label='bias')

    #     row_labels = params.index.to_numpy()
    #     for j, pred_label in enumerate(label_list):
    #         pred_label = pred_label.replace(' ({})', '')
    #         pred_row_labels = [label for label in row_labels if pred_label == re.sub(r' \(.*\)', '', label)]

    #         pred_params = params[pred_row_labels].to_numpy()
    #         pred_cis = cis.loc[pred_row_labels,:].to_numpy()

    #         ax.errorbar(x_vals, pred_params, yerr=np.abs(pred_cis - pred_params[:,None]).T, fmt='o-', capsize=4, label=pred_label)

    #     if i == 0:
    #         ax.set_ylabel('Regresssion Coefficient for Choosing Left')
    #     ax.set_xlabel('Trials Back')
    #     ax.legend(loc='best')