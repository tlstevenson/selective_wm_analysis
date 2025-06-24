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
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import pickle
import os.path as path
import seaborn as sb
from scipy import stats
import copy
import warnings

from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.multitest as smm


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
                       
ignored_signals = {'PL': [96556, 101853, 101906, 101958, 102186, 102235, 102288, 102604],
                   'DMS': [96556, 102604]}

ignored_subjects = [182] # [179]

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
                match align:
                    case Align.cue:
                        align_ts = cue_ts
                        # mask out reward
                        reward_ts_mask = reward_ts.to_numpy(copy=True)
                        reward_ts_mask[~trial_data['rewarded']] = np.nan
                        if np.isnan(reward_ts_mask[-1]):
                            reward_ts_mask[-1] = np.inf
                        reward_ts_mask = pd.Series(reward_ts_mask).bfill().to_numpy()
                        
                        mask_lims = np.hstack((np.full_like(align_ts, 0)[:, None], reward_ts_mask[:, None]))
                        
                    case Align.reward:
                        align_ts = reward_ts
                        # mask out next reward
                        next_reward_ts = reward_ts[1:].to_numpy(copy=True)
                        next_reward_ts[~trial_data['rewarded'][1:]] = np.nan
                        next_reward_ts = pd.Series(np.append(next_reward_ts, np.inf))
                        next_reward_ts = next_reward_ts.bfill().to_numpy()

                        mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], next_reward_ts[:, None]))
                
                for region in fp_data['processed_signals'].keys():
                    if region in regions:
                        signal = fp_data['processed_signals'][region][signal_type]

                        lims = xlims[region]
                        
                        mat, t = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)
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
    
# %% modify sess_data "RT" column

sess_data['RT'] = sess_data['RT'].fillna(sess_data['response_time'] - sess_data['response_cue_time'])

# %% Analyze aligned signals

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
        cpoke_in_time = sess_data['cpoke_in_time'].fillna(0)
        RT = sess_data['RT'].fillna(0)
        L_or_R = sess_data['choice']
        reward_time = sess_data['reward_time']
        cpoke_out_time = sess_data['cpoke_out_time'].fillna(0)
        
        resp_rewarded = rewarded[responded]

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
                        contra_choices = contra_choices[responded]
                        
                        resp_idx = 0
                        for i in range(mat.shape[0]):
                            if responded[i]:
                                metrics = fpah.calc_peak_properties(mat[i,:], t_r, 
                                                                    filter_params=filter_props[align][region],
                                                                    peak_find_params=peak_find_props[align][region],
                                                                    fit_decay=False)
                                
                                if resp_idx < rew_hist_n_back:
                                    buffer = np.full(rew_hist_n_back-resp_idx, False)
                                    rew_hist_vec = np.flip(np.concatenate((buffer, resp_rewarded[:resp_idx])))
                                    contra_hist_vec = np.flip(np.concatenate((buffer, contra_choices[:resp_idx])))
                                    ipsi_hist_vec = np.flip(np.concatenate((buffer, ~contra_choices[:resp_idx])))
                                else:
                                    rew_hist_vec = np.flip(resp_rewarded[resp_idx-rew_hist_n_back:resp_idx])
                                    contra_hist_vec = np.flip(contra_choices[resp_idx-rew_hist_n_back:resp_idx])
                                    ipsi_hist_vec = np.flip(~contra_choices[resp_idx-rew_hist_n_back:resp_idx])
    
                                peak_metrics.append(dict([('subj_id', subj_id), ('sess_id', sess_id), ('signal_type', signal_type), 
                                                         ('align', align.name), ('region', region), ('trial', i),
                                                         ('rewarded', rewarded[i]), ('rew_hist_bin', rew_hist.iloc[i]), 
                                                         ('side', choice_side[i]), ('rew_hist', rew_hist_vec),
                                                         ('contra_hist', contra_hist_vec), ('ipsi_hist', ipsi_hist_vec), 
                                                         ('cpoke_in_time', cpoke_in_time[i]), ('RT', RT[i]),
                                                         ('L_or_R', L_or_R[i]), ('reward_time', reward_time[i]),
                                                         ('cpoke_out_time', cpoke_out_time[i]), 
                                                         *metrics.items()]))
                                
                                resp_idx += 1
                    
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

# %% Plot average traces across all groups
plot_regions = ['DMS', 'PL']
plot_aligns = [Align.cue, Align.reward]
plot_signals = ['z_dff_iso']

all_color = '#08AB36'
rew_color = '#BC141A'
unrew_color = '#1764AB'

separate_outcome = False

if separate_outcome:
    gen_groups = {Align.cue: {'rewarded': 'rew_hist_{}_rewarded', 'unrewarded': 'rew_hist_{}_unrewarded'},
                  Align.reward: {'rewarded': 'rew_hist_{}_rewarded', 'unrewarded': 'rew_hist_{}_unrewarded'}}
    group_labels = {'rewarded': 'Rewarded', 'unrewarded': 'Unrewarded'}

    colors = {'rewarded': rew_color, 'unrewarded': unrew_color}
else:
    gen_groups = {Align.cue: {'all': 'rew_hist_{}'},
                  Align.reward: {'all': 'rew_hist_{}_rewarded'}}
    group_labels = {'all': '_'}

    colors = {'all': all_color}
    
groups = {a: {k: [v.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for k, v in gen_groups[a].items()} for a in plot_aligns}

cue_to_reward = np.nanmedian(sess_data['reward_time'] - sess_data['response_cue_time'])

plot_lims = {Align.cue: {'DMS': [-0.1,0.8], 'PL': [-1,8]},
             Align.reward: {'DMS': [-0.1,1.2], 'PL': [-1,12]}}

width_ratios = [np.diff(plot_lims[align]['DMS'])[0] for align in plot_aligns]
#width_ratios = [2,10.5]
#width_ratios = [0.7,0.9]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

for signal_type in plot_signals:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), sharey='row', width_ratios=width_ratios)
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs)

    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_all_trials_outcome_{}'.format('_'.join(plot_aligns), signal_type)

    for i, region in enumerate(plot_regions):
        for j, align in enumerate(plot_aligns):
            match align:
                case Align.cue:
                    title = 'Response Cue'
                    
                case Align.reward:
                    title = 'Reward Delivery'

            ax = axs[i,j]
            
            region_signals = stacked_signals[signal_type][align][region]
            t_r = t[region]
            
            # normalize to pre-cue levels on a trial-by-trial basis
            cue_signals = stacked_signals[signal_type][Align.cue][region]
            baseline_sel = (t_r >= -0.1) & (t_r < 0)
            
            for key, stack_groups in groups[align].items():
                stacked_signal = np.zeros((0,len(t_r)))
                
                for group in stack_groups:
                    baseline = np.nanmean(cue_signals[group][:,baseline_sel], axis=1)[:,None]
                    stacked_signal = np.vstack((stacked_signal, region_signals[group] - baseline))

                t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                error = calc_error(stacked_signal, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(stacked_signal, axis=0)[t_sel], error[t_sel], ax, label=group_labels[key], color=colors[key], plot_x0=False)
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
            #else:
                # ax.yaxis.set_tick_params(which='both', labelleft=True)
            ax.legend(loc='upper right')

            ax.set_xlabel(x_label)
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

# %% Plot average traces by reward history
plot_regions = ['PL'] #'DMS', 
plot_aligns = [Align.cue, Align.reward] #
plot_signal_types = ['z_dff_iso']

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
gen_groups = {Align.cue: ['rew_hist_{}'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
groups = {a: [group.format(rew_hist_bin_strs[rew_bin]) for group in gen_groups[a] for rew_bin in rew_hist_bins] for a in plot_aligns}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,0.5], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.2,1], 'PL': [-0.5,10]}}

#width_ratios = [1,2]
#width_ratios = [0.7,0.9]
#width_ratios = [1]
width_ratios = [np.diff(plot_lims[Align.cue][plot_regions[0]])[0], np.diff(plot_lims[Align.reward][plot_regions[0]])[0]]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 3.5*n_rows), sharey='row', width_ratios=width_ratios)
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs)

    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = 'reward_hist_{}_back_{}_{}_{}'.format(rew_rate_n_back, signal_type, '_'.join(plot_aligns), '_'.join(plot_regions))

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
plot_regions = ['DMS'] #, 'PL'
plot_aligns = [Align.cue, Align.reward]
plot_signal_types = ['z_dff_iso']

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}
gen_groups = {Align.cue: ['rew_hist_{}_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,8]},
             Align.reward: {'DMS': [-0.1,1], 'PL': [-1,12]}}

width_ratios = [np.diff(plot_lims[Align.cue][plot_regions[0]])[0], np.diff(plot_lims[Align.reward][plot_regions[0]])[0]]
#width_ratios = [2,10.5]
#width_ratios = [0.7,0.9]

n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

# plot each side on its own row and alignment in its own column. Each region gets its own figure

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    for region in plot_regions:

        fig, axs = plt.subplots(2, n_cols, layout='constrained', figsize=(5*n_cols, 7), sharey=True, width_ratios=width_ratios)

        axs = np.array(axs).reshape((2, n_cols))
        
        fig.suptitle('{} {}'.format(region, signal_title))
        
        plot_name = 'side_reward_hist_{}_back_{}_{}_{}'.format(rew_rate_n_back, signal_type, region, '_'.join(plot_aligns))
    
        for i, side_type in enumerate(sides):
            groups = {a: [group.format(rew_hist_bin_strs[rew_bin], side_type) for group in gen_groups[a] for rew_bin in rew_hist_bins] for a in plot_aligns}
            
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
# cols = ['subj_id', 'sess_id', 'align', 'region', 'group_label', 'rewarded', 'peak_height']
# sub_metrics = peak_metrics.loc[peak_metrics['signal_type'] == 'dff_iso', cols].reset_index()
# sub_metrics.rename(columns={'peak_height': 'height_dff'}, inplace=True)
# sub_metrics['height_zdff'] = peak_metrics.loc[peak_metrics['signal_type'] == 'z_dff_iso', 'peak_height'].to_numpy()
# align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}

# sub_subj_ids = np.unique(sub_metrics['subj_id'])

# n_regions = len(regions)
# n_aligns = len(alignments)

# for subj_id in sub_subj_ids:
#     fig, axs = plt.subplots(n_regions, n_aligns, figsize=(4*n_aligns, 4*n_regions), layout='constrained')
#     fig.suptitle('Peak Heights, {}'.format(subj_id))

#     for i, region in enumerate(regions):
#         for j, align in enumerate(alignments):

#             match align:
#                 case Align.cue:
#                     region_metrics = sub_metrics[(sub_metrics['region'] == region) & (sub_metrics['align'] == align) 
#                                                   & (sub_metrics['subj_id'] == subj_id)]
#                 case Align.reward:
#                     # only show rewarded peaks heights in reward alignment
#                     region_metrics = sub_metrics[(sub_metrics['region'] == region) & (sub_metrics['align'] == align) 
#                                                   & (sub_metrics['subj_id'] == subj_id) & sub_metrics['rewarded']]
                    
#             ax = axs[i,j]
#             for k, group in enumerate(np.unique(region_metrics['group_label'])):
#                 group_metrics = region_metrics[region_metrics['group_label'] == group]
#                 ax.scatter(group_metrics['height_dff'], group_metrics['height_zdff'], label=group, alpha=0.5, c='C'+str(k))
                
#             for k, group in enumerate(np.unique(region_metrics['group_label'])):
#                 group_metrics = region_metrics[region_metrics['group_label'] == group]
#                 ax.scatter(np.nanmean(group_metrics['height_dff']), np.nanmean(group_metrics['height_zdff']), c='C'+str(k), marker='x', s=500, linewidths=3, label='_')
                
#             ax.set_title('{} {}'.format(region, align_labels[align]))
#             ax.set_xlabel('dF/F')
#             ax.set_ylabel('z-scored dF/F')
#             ax.legend()
            
# %% prep peak properties for analysis

parameter_titles = {'peak_time': 'Time to Peak', 'peak_height': 'Peak Amplitude',
                    'peak_width': 'Peak Width', 'decay_tau': 'Decay τ'}
parameter_labels = {'peak_time': 'Time to Peak (s)', 'peak_height': 'Peak Amplitude ({})',
                    'peak_width': 'Peak FWHM (s)', 'decay_tau': 'Decay τ (s)'}
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}

# make subject ids categories
peak_metrics['subj_id'] = peak_metrics['subj_id'].astype('category')
peak_metrics['rew_hist_bin_label'] = peak_metrics['rew_hist_bin'].apply(lambda x: bin_labels[x])
peak_metrics['align_label'] = peak_metrics['align'].apply(lambda x: align_labels[x])

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
           # if np.random.random() < 0.01
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
            
# %% value = cue peak amplitude / x_axis:y_axis = previous trial('pre'):current trial('post') / categories = rewarded or unrewarded & same or different side choice

#construct dateframe
cue_amp = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'L_or_R', 'side']]
cue_amp = cue_amp[(cue_amp['region'] == 'DMS') & (cue_amp['signal_type'] == 'z_dff_iso') & (cue_amp['align'] == 'cue')]

#build up categories
amp_all = []

for item in np.unique(cue_amp['sess_id']):
    
    new_amp = cue_amp[(cue_amp['sess_id'] == item)]
    new_table = [new_amp[:-1].reset_index(), new_amp[1:].reset_index()]
    
    prefixes = ['pre_', 'post_']
    for i, df in enumerate(new_table):
        df.columns = [prefixes[i] + col for col in df.columns]
        
    amp_raw_table = pd.concat(new_table, axis = 1)
    
    amp_all.append(amp_raw_table)

amp_done = pd.concat(amp_all, ignore_index = True)

def comp_values(row):
    if row['pre_side'] == row['post_side']:
        return 'same'
    elif row['pre_side'] < row['post_side']:
        return 'contra->ipsi'
    else:
        return 'ipsi->contra'

side_compare = amp_done[['pre_side', 'post_side']]
side_compare['side_switch'] = side_compare.apply(comp_values, axis = 1)

amp_done['side_switch'] = side_compare['side_switch']
pre_rew_sel = amp_done['pre_rewarded']

plot_side = side_compare['side_switch']

check_table = amp_done[['pre_L_or_R', 'post_L_or_R', 'pre_side', 'post_side', 'side_switch']]

cus_palette = {'contra' : 'red', 'ipsi' : 'green'}

#separate subj-id plots
included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_done
    else:
        plot_mets = amp_done[amp_done['pre_subj_id'] == subj]
            
#plot
    pre_rew_sel = amp_done['pre_rewarded']
    fig, axs = plt.subplots(2, 2, figsize = (8, 8), layout = 'constrained', sharex = True, sharey = True)
    fig.suptitle('Cue Peak Amplitude, Comparing Previous & Current Trials, {}'.format(subj))

    ax = axs[0,0]
        
    sb.scatterplot(data = plot_mets[(pre_rew_sel) & (plot_mets['side_switch'] == 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_side', palette = cus_palette, ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Same, Pre-Rewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')
    
    ax = axs[0,1]
        
    sb.scatterplot(data = plot_mets[(~pre_rew_sel) & (plot_mets['side_switch'] == 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_side', palette = cus_palette, ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Same, Pre-Unrewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')

    ax = axs[1,0]
       
    sb.scatterplot(data = plot_mets[(pre_rew_sel) & (plot_mets['side_switch'] != 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'side_switch', hue_order = ['contra->ipsi', 'ipsi->contra'], ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Different, Pre-Rewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')
    
    ax = axs[1,1]

    sb.scatterplot(data = plot_mets[(~pre_rew_sel) & (plot_mets['side_switch'] != 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'side_switch', hue_order = ['contra->ipsi', 'ipsi->contra'], ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Different, Pre-Unrewarded')
    ax.set_xlabel('Pre_Peak_Amplitude')
    ax.set_ylabel('Post_Peak_Amplitude')
            
# %% value = reward peak amplitude / x_axis:y_axis = previous trial('pre'):current trial('post') / filtering condition: rewarded trial only / categories = same or different side choice

#construct dataframe
rew_amp = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'L_or_R', 'side']]
rew_amp = rew_amp[(rew_amp['region'] == 'DMS') & (rew_amp['signal_type'] == 'z_dff_iso') & (rew_amp['align'] == 'reward')]

amp_all = []

for item in np.unique(rew_amp['sess_id']):
    
    new_amp = rew_amp[(rew_amp['sess_id'] == item)]
    new_table = [new_amp[:-1].reset_index(), new_amp[1:].reset_index()]
    
    prefixes = ['pre_', 'post_']
    for i, df in enumerate(new_table):
        df.columns = [prefixes[i] + col for col in df.columns]
        
    amp_raw_table = pd.concat(new_table, axis = 1)
    
    amp_all.append(amp_raw_table)

amp_done = pd.concat(amp_all, ignore_index = True)

#filtering condition: post_rewarded_trials only
amp_done = amp_done[(amp_done['post_rewarded'] == True)]

#build up categories
def comp_values(row):
    if row['pre_side'] == row['post_side']:
        return 'same'
    else:
        return '{} -> {}'.format(row['pre_side'], row['post_side'])

amp_done['side_switch'] = amp_done.apply(comp_values, axis = 1)

plot_side = amp_done['side_switch']

cus_palette = {'contra' : 'brown', 'ipsi' : 'green'}

#separate subj-id plots
included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

#debug
check_table = amp_done[['pre_subj_id','pre_L_or_R', 'post_L_or_R', 'pre_side', 'post_side', 'side_switch']]
check_table = check_table[check_table['pre_subj_id'] == 179]

order = ['contra -> ipsi', 'ipsi -> contra']

for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_done
    else:
        plot_mets = amp_done[amp_done['pre_subj_id'] == subj]
            
#plot
    pre_rew_sel = amp_done['pre_rewarded']
    fig, axs = plt.subplots(2, 2, figsize = (8, 8), layout = 'constrained', sharex = True, sharey = True)
    # fig.suptitle('Reward Peak Amplitude, Comparing Previous & Current(Post) Trials, {}'.format(subj))
    fig.suptitle('Reward Peak Amplitude, Comparing Previous & Current(post) Trials, Post_Rewarded Trials Only, {}'.format(subj))

    ax = axs[0,0]
        
    sb.scatterplot(data = plot_mets[(pre_rew_sel) & (plot_mets['side_switch'] == 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_side', palette = cus_palette, ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Same, Pre-Rewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')
    
    ax = axs[1,0]

    sb.scatterplot(data = plot_mets[(pre_rew_sel) & (plot_mets['side_switch'] != 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'side_switch', hue_order = order, ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Different, Pre-Rewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')
    
    ax = axs[0,1]
        
    sb.scatterplot(data = plot_mets[(~pre_rew_sel) & (plot_mets['side_switch'] == 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_side', palette = cus_palette, ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Same, Pre-Unrewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')
    
    ax = axs[1,1]

    sb.scatterplot(data = plot_mets[(~pre_rew_sel) & (plot_mets['side_switch'] != 'same')], x = 'pre_peak_height', y = 'post_peak_height', hue = 'side_switch', hue_order = order, ax = ax, alpha = 0.5)
    
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    
    ax.set_title('Different, Pre-Unrewarded')
    ax.set_xlabel('Pre_Peak_Height')
    ax.set_ylabel('Post_Peak_Height')
    

# %% value = reward peak amplitude / stirpplot / categories = same or different side choice

# rough dataframe building
rew_peak = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'L_or_R', 'side', 'peak_time']]
rew_peak = rew_peak[(rew_peak['region'] == 'DMS') & (rew_peak['align'] == 'reward') & (rew_peak['signal_type'] == 'z_dff_iso')]

amp_done = rew_peak[['subj_id', 'side',  'rewarded', 'peak_height', 'peak_time']]

def type_define(row):
    if row['rewarded'] == True:
        return'{}, rewarded'.format(row['side'])
    else:
        return'{}, unrewarded'.format(row['side'])

amp_done['catagory'] = amp_done.apply(type_define, axis = 1)
order = ['contra, rewarded', 'contra, unrewarded', 'ipsi, rewarded', 'ipsi, unrewarded']

amp_done['catagory'] = pd.Categorical(amp_done['catagory'], categories = order, ordered = True)

# subj_id loop
included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_done
    else:
        plot_mets = amp_done[amp_done['subj_id'] == subj]
        
    fig, axs = plt.subplots(1, 1, figsize = (8, 4), layout = 'constrained', sharex = True, sharey = True)
    fig.suptitle('Reward Peak Amplitude Distribution, Comparing Side Choices Switch & Current Trial Rewarded or Not, {}'.format(subj))
    sb.swarmplot(data = plot_mets, x = 'catagory', y = 'peak_height', hue = 'catagory', ax = axs, alpha = 0.5, legend = False, order = order, size = 1.5)

# %% value = reward peak amplitude / x_axis:y_axis = peak time:peak height / categories = rewarded or unrewarded & same or different side choice

#construct dataframe
rew_peak = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'L_or_R', 'side', 'peak_time']]
rew_peak = rew_peak[(rew_peak['region'] == 'DMS') & (rew_peak['align'] == 'reward') & (rew_peak['signal_type'] == 'z_dff_iso')]

amp_done = rew_peak[['subj_id', 'side',  'rewarded', 'peak_height', 'peak_time']]

#build up categories
def type_define(row):
    if row['rewarded'] == True:
        return'{}, rewarded'.format(row['side'])
    else:
        return'{}, unrewarded'.format(row['side'])

amp_done['catagory'] = amp_done.apply(type_define, axis = 1)
order = ['contra, rewarded', 'contra, unrewarded', 'ipsi, rewarded', 'ipsi, unrewarded']

included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

hue_palette = {'contra, rewarded' : 'blue', 'contra, unrewarded' : 'red', 
               'ipsi, rewarded' : 'orange', 'ipsi, unrewarded' : 'green'}

#separate subj-id plots
for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_done
    else:
        plot_mets = amp_done[(amp_done['subj_id'] == subj)]

#plot
    fig, axs = plt.subplots(2, 2, figsize = (8, 8), layout = 'constrained', sharex = True, sharey = True)
    fig.suptitle('Reward Peak Amplitude vs. Peak Time, {}'.format(subj))
    
    ax = axs[0,0]
    sb.scatterplot(data = plot_mets[plot_mets['catagory'] == 'contra, rewarded'], x = 'peak_time', y = 'peak_height', hue = 'catagory', palette = hue_palette, ax = ax, alpha = 0.5, legend = False)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Contra, Rewarded')
    ax.set_xlabel('Peak Time')
    ax.set_ylabel('Peak Height')
    
    ax = axs[0,1]
    sb.scatterplot(data = plot_mets[plot_mets['catagory'] == 'contra, unrewarded'], x = 'peak_time', y = 'peak_height', hue = 'catagory', palette = hue_palette, ax = ax, alpha = 0.5, legend = False)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Contra, Unrewarded')
    ax.set_xlabel('Peak Time')
    ax.set_ylabel('Peak Height')
    
    ax = axs[1,0]
    sb.scatterplot(data = plot_mets[plot_mets['catagory'] == 'ipsi, rewarded'], x = 'peak_time', y = 'peak_height', hue = 'catagory', palette = hue_palette, ax = ax, alpha = 0.5, legend = False)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Ipsi, Rewarded')
    ax.set_xlabel('Peak Time')
    ax.set_ylabel('Peak Height')
    
    ax = axs[1,1]
    sb.scatterplot(data = plot_mets[plot_mets['catagory'] == 'ipsi, unrewarded'], x = 'peak_time', y = 'peak_height', hue = 'catagory', palette = hue_palette, ax = ax, alpha = 0.5, legend = False)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Ipsi, Unrewarded')
    ax.set_xlabel('Peak Time')
    ax.set_ylabel('Peak Height')
    
# %% check with specific peaks shape (unrew: peakheight > 3 or 4)

def print_dic_structure(d, indent = 0):
    for key, value in d.items():
        print(" " * indent+ str(key))
        if isinstance(value, dict):
            print_dic_structure(value, indent + 1)
print_dic_structure(aligned_signals)


# look at potentially problematic peaks
t = aligned_signals['t']
#rem_peak_info = peak_metrics[~peak_sel]
rem_peak_info = filt_peak_metrics[(filt_peak_metrics['signal_type'] == 'z_dff_iso') 
                                  & (filt_peak_metrics['region'] == 'DMS') 
                                  & (filt_peak_metrics['align'] == 'reward') 
                                  & (filt_peak_metrics['rewarded'] == False)
                                  & (filt_peak_metrics['peak_height'] >= 4)].groupby("subj_id").apply(lambda x: x.nlargest(5, 'peak_height'))
rem_subj_ids = np.unique(rem_peak_info['subj_id'])

for subj_id in rem_subj_ids:
    subj_peak_info = rem_peak_info[rem_peak_info['subj_id'] == subj_id]
    
    for _, row in subj_peak_info.iterrows():
         #if np.random.random() < 0.1:
            mat = aligned_signals[row['subj_id']][row['sess_id']]['z_dff_iso'][row['align']][row['region']]
            
            _, ax = plt.subplots(1,1)
            ax.set_title('{} - {}, {} {}-aligned, trial {}'.format(row['subj_id'], row['sess_id'], row['region'], row['align'], row['trial']))
            ax.plot(t[row['region']], mat[row['trial'], :])
            plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
            peak_idx = np.argmin(np.abs(t[row['region']] - row['peak_time']))
            ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
            peak_end_idx = np.argmin(np.abs(t[row['region']] - row['peak_end_time']))
            ax.plot(row['peak_end_time'], mat[row['trial'], peak_end_idx], marker=7, markersize=10, color='C1')
            peak_start_idx = np.argmin(np.abs(t[row['region']] - row['peak_start_time']))
            ax.plot(row['peak_start_time'], mat[row['trial'], peak_start_idx], marker=7, markersize=10, color='C1')
            ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')
            start_point = mat[row['trial'], peak_start_idx]
            end_point = mat[row['trial'], peak_end_idx]
            ax.plot([row['peak_start_time'], row['peak_end_time']], [start_point, end_point], color = 'black', linestyle = '-')
            ax.set_xlabel('Time')
            ax.set_ylabel('Peak Height')

# %% value = reward peak amplitude / x_axis:y_axis = previous trial('pre'):current trial('post') / categories = current[(contra/ipsi)&(rew/unrew)] / color = pre[(contra/ipsi)&(rew/unrew)]

#construct dataframe
rew_amp = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'L_or_R', 'side', 'peak_time']]
rew_amp = rew_amp[(rew_amp['region'] == 'DMS') & (rew_amp['align'] == 'reward') & (rew_amp['signal_type'] == 'z_dff_iso')]

#build up categories
amp_done = []

for item in np.unique(rew_amp['sess_id']):
    
    new_amp = rew_amp[(rew_amp['sess_id'] == item)]
    new_table = [new_amp[:-1].reset_index(), new_amp[1:].reset_index()]
    
    prefixes = ['pre_', 'post_']
    for i, df in enumerate(new_table):
        df.columns = [prefixes[i] + col for col in df.columns]
        
    amp_raw_table = pd.concat(new_table, axis = 1)
    
    amp_done.append(amp_raw_table)

amp_done = pd.concat(amp_done, ignore_index = True)

def pre_type_define(row):
    if row['pre_rewarded'] == True:
        return'{}, rewarded'.format(row['pre_side'])
    else:
        return'{}, unrewarded'.format(row['pre_side'])
    
amp_done['pre_category'] = amp_done.apply(pre_type_define, axis=1)

def post_type_define(row):
    if row['post_rewarded'] == True:
        return'{}, rewarded'.format(row['post_side'])
    else:
        return'{}, unrewarded'.format(row['post_side'])

amp_done['post_category'] = amp_done.apply(post_type_define, axis=1)
order = ['contra, rewarded', 'contra, unrewarded', 'ipsi, rewarded', 'ipsi, unrewarded']

amp_done['peak_height_dff'] = amp_done['pre_peak_height'] - amp_done['post_peak_height']

#separate subj-id plots
included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_done
    else:
        plot_mets = amp_done[(amp_done['pre_subj_id'] == subj)]

#plot
    fig, axs = plt.subplots(2, 2, figsize = (8, 8), layout = 'constrained', sharex = True, sharey = True)
    fig.suptitle('Reward Peak Amplitude, Comparing Previous & Current(Post) Trials, {}'.format(subj))
    
    ax = axs[0,0]
    sb.scatterplot(data = plot_mets[plot_mets['post_category'] == 'contra, rewarded'], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_category', hue_order = order, ax = ax, alpha = 0.5)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Post: Contra, Rewarded')
    ax.set_xlabel('Pre Peak Height')
    ax.set_ylabel('Post Peak Height')
    
    ax = axs[0,1]
    sb.scatterplot(data = plot_mets[plot_mets['post_category'] == 'contra, unrewarded'], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_category', hue_order = order, ax = ax, alpha = 0.5)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Post: Contra, Unrewarded')
    ax.set_xlabel('Pre Peak Height')
    ax.set_ylabel('Post Peak Height')
    
    ax = axs[1,0]
    sb.scatterplot(data = plot_mets[plot_mets['post_category'] == 'ipsi, rewarded'], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_category', hue_order = order, ax = ax, alpha = 0.5)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Post: Ipsi, Rewarded')
    ax.set_xlabel('Pre Peak Height')
    ax.set_ylabel('Post Peak Height')
    
    ax = axs[1,1]
    sb.scatterplot(data = plot_mets[plot_mets['post_category'] == 'ipsi, unrewarded'], x = 'pre_peak_height', y = 'post_peak_height', hue = 'pre_category', hue_order = order, ax = ax, alpha = 0.5)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    ax.set_xlim(x_lim[0], x_lim[1])
    ax.set_ylim(y_lim[0], y_lim[1])
    min_lim = min(x_lim[0], y_lim[0])
    max_lim = max(x_lim[1], y_lim[1])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], '--', color = 'black')
    ax.set_title('Post: Ipsi, Unrewarded')
    ax.set_xlabel('Pre Peak Height')
    ax.set_ylabel('Post Peak Height')

# %% value = peak height change trial-by-trial / stripplot & box plot / categories = current[(contra/ipsi)&(rew/unrew)] / color = pre[(contra/ipsi)&(rew/unrew)]

#construct dataframe
rew_amp = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'L_or_R', 'side', 'peak_time']]
rew_amp = rew_amp[(rew_amp['region'] == 'DMS') & (rew_amp['align'] == 'reward') & (rew_amp['signal_type'] == 'z_dff_iso')]

#build up categories
amp_done = []

for item in np.unique(rew_amp['sess_id']):
    
    new_amp = rew_amp[(rew_amp['sess_id'] == item)]
    new_table = [new_amp[:-1].reset_index(), new_amp[1:].reset_index()]
    
    prefixes = ['pre_', 'post_']
    for i, df in enumerate(new_table):
        df.columns = [prefixes[i] + col for col in df.columns]
        
    amp_raw_table = pd.concat(new_table, axis = 1)
    
    amp_done.append(amp_raw_table)

amp_done = pd.concat(amp_done, ignore_index = True)

def pre_type_define(row):
    if row['pre_rewarded'] == True:
        return'{}, rewarded'.format(row['pre_side'])
    else:
        return'{}, unrewarded'.format(row['pre_side'])
    
amp_done['pre_catagory'] = amp_done.apply(pre_type_define, axis=1)

def post_type_define(row):
    if row['post_rewarded'] == True:
        return'{}, rewarded'.format(row['post_side'])
    else:
        return'{}, unrewarded'.format(row['post_side'])

amp_done['post_catagory'] = amp_done.apply(post_type_define, axis=1)
order = ['contra, rewarded', 'contra, unrewarded', 'ipsi, rewarded', 'ipsi, unrewarded']

amp_done['peak_height_dff'] = amp_done['post_peak_height'] - amp_done['pre_peak_height']

included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

#seperate subj-id plots
for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_done
    else:
        plot_mets = amp_done[(amp_done['pre_subj_id'] == subj)]
        
    #stirpplot
    fig, axs = plt.subplots(4, 1, figsize = (8, 8), layout = 'constrained', sharey = True)
    fig.suptitle('Peak Height Trail-by-trial Change, {}'.format(subj))
        
    ax = axs[0]
    sb.swarmplot(data = plot_mets[plot_mets['post_catagory'] == 'contra, rewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = axs[0], alpha = 0.5, legend = False, order = order, size = 1.5)
    axs[0].set_title('Contra, Post-Rewarded')
    axs[0].set_xlabel('Pre Catagory')
    axs[0].set_ylabel('Peak Height Difference')
    axs[0].axhline(y = 0, linestyle = '--', color = 'black')
        
    ax = axs[1]
    sb.swarmplot(data = plot_mets[plot_mets['post_catagory'] == 'contra, unrewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = axs[1], alpha = 0.5, legend = False, order = order, size = 1.5)
    axs[1].set_title('Contra, Post-Unrewarded')
    axs[1].set_xlabel('Pre Catagory')
    axs[1].set_ylabel('Peak Height Difference')
    axs[1].axhline(y = 0, linestyle = '--', color = 'black')
        
    ax = axs[2]
    sb.swarmplot(data = plot_mets[plot_mets['post_catagory'] == 'ipsi, rewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = axs[2], alpha = 0.5, legend = False, order = order, size = 1.5)
    axs[2].set_title('Ipsi, Post-Rewarded')
    axs[2].set_xlabel('Pre Catagory')
    axs[2].set_ylabel('Peak Height Difference')
    axs[2].axhline(y = 0, linestyle = '--', color = 'black')
        
    ax = axs[3]
    sb.swarmplot(data = plot_mets[plot_mets['post_catagory'] == 'ipsi, unrewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = axs[3], alpha = 0.5, legend = False, order = order, size = 1.5)
    axs[3].set_title('Ipsi, Post-Unrewarded')
    axs[3].set_xlabel('Pre Catagory')
    axs[3].set_ylabel('Peak Height Difference')
    axs[3].axhline(y = 0, linestyle = '--', color = 'black')
    
    #box plot
    fig, axs = plt.subplots(2, 2, figsize = (12, 12), layout = 'constrained', sharey = True)
    fig.suptitle('Peak Height Trail-by-trial Change, {}'.format(subj))
    
    ax = axs[0,0]
    sb.boxplot(data = plot_mets[plot_mets['post_catagory'] == 'contra, rewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = ax, order = order)
    ax.set_title('Contra, Post-Rewarded')
    ax.set_xlabel('Pre Catagory')
    ax.set_ylabel('Peak Height Difference')
    ax.axhline(y = 0, linestyle = '--', color = 'black')
    ax.legend_.remove()
    
    ax = axs[0,1]
    sb.boxplot(data = plot_mets[plot_mets['post_catagory'] == 'contra, unrewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = ax, order = order)
    ax.set_title('Contra, Post-Unrewarded')
    ax.set_xlabel('Pre Catagory')
    ax.set_ylabel('Peak Height Difference')
    ax.axhline(y = 0, linestyle = '--', color = 'black')
    ax.legend_.remove()
    
    ax = axs[1,0]
    sb.boxplot(data = plot_mets[plot_mets['post_catagory'] == 'ipsi, rewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = ax, order = order)
    ax.set_title('Ipsi, Post-Rewarded')
    ax.set_xlabel('Pre Catagory')
    ax.set_ylabel('Peak Height Difference')
    ax.axhline(y = 0, linestyle = '--', color = 'black')
    ax.legend_.remove()
    
    ax = axs[1,1]
    sb.boxplot(data = plot_mets[plot_mets['post_catagory'] == 'ipsi, unrewarded'], x = 'pre_catagory', y = 'peak_height_dff', hue = 'pre_catagory', hue_order = order, ax = ax, order = order)
    ax.set_title('Ipsi, Post-Unrewarded')
    ax.set_xlabel('Pre Catagory')
    ax.set_ylabel('Peak Height Difference')
    ax.axhline(y = 0, linestyle = '--', color = 'black')
    ax.legend_.remove()

# %% value = peak amplitude / x_axis:y_axis = cue:reward / filtering condition = cpoke_in_time 12s

#making rough dataframe
amp_comparing = filt_peak_metrics[['subj_id', 'signal_type', 'sess_id', 'region', 'peak_height', 'align', 'trial', 'rewarded', 'RT', 'cpoke_in_time', 'side']] # 'cpoke_in_time', 'RT', 'side'

#refining: pick out DMS & rewarded
amp_refined = amp_comparing[(amp_comparing['region'] == 'DMS') & (amp_comparing['rewarded']) & (amp_comparing['signal_type'] == 'z_dff_iso')]

#use pivot to separate cue and reward
amp_pivoted = amp_refined.pivot(index = ['subj_id', 'sess_id', 'trial', 'RT', 'cpoke_in_time', 'side'], 
                            columns = 'align', values = 'peak_height'
                            ).rename(columns = {'cue':'cue_amp', 'reward':'reward_amp'}
                                     ).reset_index() 
    
def time_set(row):
    if row['cpoke_in_time'] <= 12:
        return 'below'
    else:
        return'above'

amp_pivoted['cpoke_threshold'] = amp_pivoted.apply(time_set, axis = 1)

order = ['below', 'above']

included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']
    
for subj in plot_subjs:
        
        if subj == 'all':
            plot_mets = amp_pivoted
            
        else:
            plot_mets = amp_pivoted[amp_pivoted['subj_id'] == subj]
        
        fig, ax = plt.subplots(1, 1, figsize = (4, 4), layout = 'constrained')
        
        sb.scatterplot(data = plot_mets, x = 'cue_amp', y = 'reward_amp', hue = 'cpoke_threshold', hue_order = order, ax = ax, alpha = 0.5, s = 10)
        
        x_lim = ax.get_xlim()
        y_lim = ax.get_ylim()
        new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
        ax.set_xlim(new_lims)
        ax.set_ylim(new_lims)
        ax.plot(new_lims, new_lims, '--', color = 'black')
        
        ax.set_title('Cue vs Reward Amplitude, cpoke_in_time 12s, {}'.format(subj))
        ax.set_xlabel('Cue Amplitude')
        ax.set_ylabel('Reward Amplitude')
                
# %% value = reward peak amplitude / x_axis:y_axis = PL:DMS (current) / categories = L/R side switch / color = rewarded/unrewarded switch

amp_comp = filt_peak_metrics[['subj_id', 'sess_id', 'trial', 'signal_type', 'align', 'region', 'peak_height', 'rewarded', 'L_or_R']]
amp_comp = amp_comp[(amp_comp['signal_type'] == 'z_dff_iso') & (amp_comp['align'] == "reward")]

amp_PL = amp_comp[(amp_comp['region'] == 'PL')].rename(columns = {'peak_height' : 'PL_amp'})
amp_DMS = amp_comp[(amp_comp['region'] == 'DMS')].rename(columns = {'peak_height' : 'DMS_amp'})

amp_done = amp_PL.merge(amp_DMS, on=['subj_id', 'sess_id', 'trial', 'align'], how = 'inner')
amp_done = amp_done.drop(columns = ['signal_type_y', 'region_x', 'region_y', 'rewarded_y'])
amp_done.rename(columns = {'signal_type_x': 'signal_type', 'rewarded_x': 'rewarded', 'L_or_R_x': 'L_or_R'}, inplace = True)
                                   
def rew(row):
    if row['rewarded'] == True:
        return"Rewarded"
    else:
        return"Unrewarded"
amp_done['rew'] = amp_done.apply(rew, axis = 1)

amp_all = []
for item in np.unique(amp_done['sess_id']):
    
    new_amp = amp_done[(amp_done['sess_id'] == item)]
    new_comp = [new_amp[:-1].reset_index(), new_amp[1:].reset_index()]
    prefix = ['pre_', 'post_']
    for i, df in enumerate(new_comp):
        df.columns = [prefixes[i] + col for col in df.columns]
    amp_raw = pd.concat(new_comp, axis = 1)
    amp_all.append(amp_raw)
amp_fin = pd.concat(amp_all, ignore_index = True)

amp_fin['pre_post'] = amp_fin.apply(lambda row: f'{row["pre_rew"]}->{row["post_rew"]}', axis = 1)
amp_fin['side_change'] = amp_fin.apply(lambda row: f'{row["pre_L_or_R"]}->{row["post_L_or_R"]}', axis = 1)

order = ['Rewarded->Rewarded', 'Rewarded->Unrewarded', 'Unrewarded->Rewarded', 'Unrewarded->Unrewarded']#['left->left', 'left->right', 'right->right', 'right->left']#['Rewarded->Rewarded', 'Rewarded->Unrewarded', 'Unrewarded->Rewarded', 'Unrewarded->Unrewarded']

included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_fin
    else:
        plot_mets = amp_fin[(amp_fin['pre_subj_id'] == subj)]
    
    fig, axs = plt.subplots(2, 2, figsize = (8, 8), layout = 'constrained', sharex = True, sharey = True)
    fig.suptitle('PL vs. DMS amplitude, Comparing Side-choice and Rewarded-situation Switch, {}'.format(subj))
    
    ax = axs[0,0]
    sb.scatterplot(data = plot_mets[plot_mets['side_change'] == 'left->left'], x = 'post_DMS_amp', y = 'post_PL_amp', hue = 'pre_post', hue_order = order, ax = ax, alpha = 0.5, s = 10)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    ax.set_title("Left -> Left")
    ax.set_xlabel('post_DMS_amp')
    ax.set_ylabel('post_PL_amp')
    
    ax = axs[0,1]
    sb.scatterplot(data = plot_mets[plot_mets['side_change'] == 'left->right'], x = 'post_DMS_amp', y = 'post_PL_amp', hue = 'pre_post', hue_order = order, ax = ax, alpha = 0.5, s = 10)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    ax.set_title("Left -> Right")
    ax.set_xlabel('post_DMS_amp')
    ax.set_ylabel('post_PL_amp')
    
    ax = axs[1,0]
    sb.scatterplot(data = plot_mets[plot_mets['side_change'] == 'right->right'], x = 'post_DMS_amp', y = 'post_PL_amp', hue = 'pre_post', hue_order = order, ax = ax, alpha = 0.5, s = 10)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    ax.set_title("Right -> Right")
    ax.set_xlabel('post_DMS_amp')
    ax.set_ylabel('post_PL_amp')
    
    ax = axs[1,1]
    sb.scatterplot(data = plot_mets[plot_mets['side_change'] == 'right->left'], x = 'post_DMS_amp', y = 'post_PL_amp', hue = 'pre_post', hue_order = order, ax = ax, alpha = 0.5, s = 10)
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    new_lims = [min(x_lim[0], y_lim[0]), max(x_lim[1], y_lim[1])]
    ax.set_xlim(new_lims)
    ax.set_ylim(new_lims)
    ax.plot(new_lims, new_lims, '--', color = 'black')
    ax.set_title("Right -> Left")
    ax.set_xlabel('post_DMS_amp')
    ax.set_ylabel('post_PL_amp')
    
# %% value = reward peak amplitude / x_axis:y_axis = trail_gap_time:peak height / scatterplot & stripplot / categories = current side choice

amp_time = filt_peak_metrics[['align', 'region', 'signal_type', 'subj_id', 'sess_id', 'trial', 'rewarded', 'side', 'cpoke_in_time', 'RT', 'reward_time', 'peak_height']]
amp_time = amp_time[(amp_time['align'] == 'reward') & (amp_time['region'] =='PL') & (amp_time['signal_type'] == 'z_dff_iso')]

amp_done = []

for item in np.unique(amp_time['sess_id']):
    amp_seg = amp_time[(amp_time['sess_id'] == item)]
    amp_seg = [amp_seg[:-1].reset_index(), amp_seg[1:].reset_index()]
    prefix = ['pre_', 'post_']
    
    for i, df in enumerate(amp_seg):
        df.columns = [prefix[i] + col for col in df.columns]
    amp_set = pd.concat(amp_seg, axis = 1)
    amp_done.append(amp_set)

amp_fin = pd.concat(amp_done, ignore_index = True)

def pre_combine(row):
    if row['pre_rewarded'] == True:
        return'{}, rewarded'.format(row['pre_side'])
    else:
        return'{}, unrewarded'.format(row['pre_side'])

def post_combine(row):
    if row['post_rewarded'] == True:
        return'Post_trial: {}, rewarded'.format(row['post_side'])
    else:
        return'Post_trial: {}, unrewarded'.format(row['post_side'])

amp_fin['Pre_trial_info'] = amp_fin.apply(pre_combine, axis = 1)
amp_fin['Post_trial_info'] = amp_fin.apply(post_combine, axis = 1)

# order = ['contra, rewarded', 'contra, unrewarded', 'ipsi, rewarded', 'ipsi, unrewarded']
order = ['contra', 'ipsi']

amp_fin['true_block'] = amp_fin['post_rewarded'].cumsum()
unrewarded_term = amp_fin[~amp_fin['post_rewarded']].groupby('true_block')['post_reward_time'].sum()
amp_rewarded = amp_fin[amp_fin['post_rewarded'] == True]
amp_rewarded['rewarded_gap'] = amp_rewarded['true_block'].map(unrewarded_term)
amp_rewarded['rewarded_gap'] = amp_rewarded['rewarded_gap'].fillna(0) + amp_rewarded['post_reward_time']

included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]
plot_subjs = included_subjs.tolist() + ['all']

plot_data_all = amp_rewarded[amp_rewarded['post_subj_id'].isin(included_subjs)]
y_min = plot_data_all['post_peak_height'].min()
y_max = plot_data_all['post_peak_height'].max()
x_min = plot_data_all['rewarded_gap'].min()
x_max = plot_data_all['rewarded_gap'].max()
# x_max = 20
x_lims = [x_min, x_max]
y_lims = [y_min, y_max]

for subj in plot_subjs:
    if subj == 'all':
        plot_mets = amp_rewarded
    else:
        plot_mets = amp_rewarded[(amp_rewarded['post_subj_id'] == subj)]
    
    #scatterplot
    fig, ax = plt.subplots(1, 1, figsize = (4, 4), layout = 'constrained', sharex = True, sharey = True)
    
    sb.scatterplot(data = plot_mets, x = 'rewarded_gap', y = 'post_peak_height', hue = 'post_side', hue_order = order, ax = ax, alpha = 0.5, s = 10)
    
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.plot(x_lims, y_lims, '--', color = 'black')
    
    ax.set_title('Rewarded Trial Gap vs. Peak Height, {}'.format(subj))
    ax.set_xlabel('Rewarded Trail Gap Time')
    ax.set_ylabel('Reward Amplitude')
    
    #stripplot
    fig, axs = plt.subplots(4, 1, figsize = (8, 8), layout = 'constrained', sharey = True)
    fig.suptitle('Peak Height, DMS, Rewarded Trial Gap Time distribution {}'.format(subj))
    
    ax = axs[0]
    sb.swarmplot(data = plot_mets[plot_mets['Pre_trial_info'] == 'contra, rewarded'], x = 'post_side', y = 'post_peak_height', hue = 'rewarded_gap', palette = 'viridis', ax = axs[0], alpha = 0.5, legend = True, order = order, size = 1.5)
    axs[0].set_title('Pre_trial: Contra, Rewarded')
    axs[0].set_xlabel('Side Choice')
    axs[0].set_ylabel('Peak Height')
    axs[0].axhline(y = 0, linestyle = '--', color = 'black')
    
    ax = axs[1]
    sb.swarmplot(data = plot_mets[plot_mets['Pre_trial_info'] == 'contra, unrewarded'], x = 'post_side', y = 'post_peak_height', hue = 'rewarded_gap', palette = 'viridis', ax = axs[1], alpha = 0.5, legend = True, order = order, size = 1.5)
    axs[1].set_title('Pre_trial: Contra, Unrewarded')
    axs[1].set_xlabel('Side Choice')
    axs[1].set_ylabel('Peak Height')
    axs[1].axhline(y = 0, linestyle = '--', color = 'black')
    
    ax = axs[2]
    sb.swarmplot(data = plot_mets[plot_mets['Pre_trial_info'] == 'ipsi, rewarded'], x = 'post_side', y = 'post_peak_height', hue = 'rewarded_gap', palette = 'viridis', ax = axs[2], alpha = 0.5, legend = True, order = order, size = 1.5)
    axs[2].set_title('Pre_trial: Ipsi, Rewarded')
    axs[2].set_xlabel('Side Choice')
    axs[2].set_ylabel('Peak Height')
    axs[2].axhline(y = 0, linestyle = '--', color = 'black')
    
    ax = axs[3]
    sb.swarmplot(data = plot_mets[plot_mets['Pre_trial_info'] == 'ipsi, unrewarded'], x = 'post_side', y = 'post_peak_height', hue = 'rewarded_gap', palette = 'viridis', ax = axs[3], alpha = 0.5, legend = True, order = order, size = 1.5)
    axs[3].set_title('Pre_trial: Ipsi, Unrewarded')
    axs[3].set_xlabel('Side Choice')
    axs[3].set_ylabel('Peak Height')
    axs[3].axhline(y = 0, linestyle = '--', color = 'black')

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

# %% make peak property comparison figures comparing regions at cue and reward

parameters = ['peak_height', 'peak_width'] # 'peak_time', 'decay_tau']

regions = ['DMS', 'PL']
region_colors = ['#53C43B', '#BB6ED8']
plot_aligns = ['Response Cue', 'Reward Delivery']
subj_ids = np.unique(filt_peak_metrics['subj_id'])
plot_signals = ['dff_iso']

# have same jitter for each subject
noise = 0.075
n_neg = int(np.floor(len(subj_ids)/2))
n_pos = int(np.ceil(len(subj_ids)/2))
jitters = np.concatenate([np.random.uniform(-1, -0.1, n_neg), np.random.uniform(0.1, 1, n_pos)]) * noise

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    peak_sel = filt_peak_metrics['signal_type'] == signal_type
    # only show rewarded peaks for reward delivery
    peak_sel = peak_sel & ((filt_peak_metrics['align'] == Align.cue) | ((filt_peak_metrics['align'] == Align.reward) & filt_peak_metrics['rewarded']))
    
    sub_peak_metrics = filt_peak_metrics[peak_sel]
    
    # Compare responses across regions grouped by alignment
    fig, axs = plt.subplots(1, len(parameters), figsize=(4*len(parameters), 4), layout='constrained')
    axs = np.array(axs).reshape((len(parameters)))
    
    for i, param in enumerate(parameters):
        ax = axs[i]
        ax.set_title(parameter_titles[param])

        subj_avgs = sub_peak_metrics.groupby(['subj_id', 'region', 'align_label']).agg({param: np.nanmean}).reset_index()
    
        # plot sensor averages in boxplots
        sb.boxplot(data=sub_peak_metrics, x='align_label', y=param, hue='region', palette=region_colors,
                   order=plot_aligns, hue_order=regions, ax=ax, showfliers=False)
        if param == 'peak_height':
            ax.set_ylabel(parameter_labels[param].format(y_label))
        else:
            ax.set_ylabel(parameter_labels[param])
        ax.set_xlabel('')
        ax.set_yscale('log')
    
        # add subject averages for each alignment with lines connecting them
        dodge = 0.2

        for j, align in enumerate(plot_aligns):
            x = np.array([j - dodge, j + dodge])
    
            for subj_id, jitter in zip(subj_ids, jitters):
                subj_avg = subj_avgs[(subj_avgs['subj_id'] == subj_id) & (subj_avgs['align_label'] == align)]
    
                y = [subj_avg.loc[subj_avg['region'] == r, param] for r in regions]
    
                ax.plot(x+jitter, y, color='black', marker='o', linestyle='dashed', alpha=0.75)
                
        # set y max to be a multiple of 10
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, utils.convert_to_multiple(y_max, 10, direction='up'))
                
    plot_name = 'cue_reward_peak_comp_{}_{}'.format('_'.join(parameters), signal_type)
    fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')


# %% make reward history comparison figures per peak property

parameters = ['peak_height'] # 'peak_time', 'peak_height', 'peak_width', 'decay_tau'

rew_hist_rew_colors = plt.cm.seismic(np.linspace(0.6,1,len(rew_hist_bins)))
rew_hist_palette = sb.color_palette(rew_hist_rew_colors) 

plot_signals = ['z_dff_iso']
plot_aligns = ['reward'] #'cue', 
plot_regions = ['DMS', 'PL']

split_by_side = False
sides = ['contra', 'ipsi']
n_regions = len(regions)

n_aligns = len(plot_aligns)
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}
hatch_order = ['//\\\\', '']
line_order = ['dashed', 'solid']

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    for param in parameters:
        
        plot_name = '{}_reward_hist_{}_{}_back_{}'.format('_'.join(plot_aligns), param, rew_rate_n_back, signal_type)
    
        # Compare responses across alignments grouped by region
        fig, axs = plt.subplots(n_regions, n_aligns, figsize=(4*n_aligns, 4*n_regions), layout='constrained', sharey='row')
        fig.suptitle('{}, {}'.format(parameter_titles[param], signal_label))
        
        axs = np.array(axs).reshape((n_regions, n_aligns))
    
        for i, region in enumerate(plot_regions):
            for j, align in enumerate(plot_aligns):
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
aligns = [Align.reward] #Align.cue, 
signals = ['dff_iso']
include_side = False
use_bins_as_cats = True # whether to use the bin labels as categories or regressors
print_fit_results = False

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
                
                #NOTE: mixed effects models fit a common variance for a zero-mean gaussian distribution
                # for all random effects where the individual random effect estimates are drawn from that distribution
                # thus adding both a subject and side term to the variance means that separate variances will be fit for those random effects across all groups
                if include_side:
                    # if subj_id is not specified, will fit two means per subject, one for each side which does slightly worse than fitting the subject average in addition to any variability based on side
                    vc_form={'subj_id': '1', 'side': '0 + C(side)'} 
                else:
                    vc_form={'subj_id': '1'} #same as 0 + C(subj_id)
        
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
                param_vals = region_metrics[param].to_numpy()
                subj_id_vals = region_metrics['subj_id'].to_numpy()
                side_vals = region_metrics['side'].to_numpy()
                
                if use_bins_as_cats:
                    p_vals = []
                    for i, first_group in enumerate(rew_hist_groups):
                        group_mapping = {first_group: '0'}
                        for other_group in rew_hist_groups[~np.isin(rew_hist_groups, first_group)]:
                            group_mapping.update({other_group: 'bin_'+other_group})
                            
                        rew_hist_vals = region_metrics['rew_hist_bin_label'].apply(lambda x: group_mapping[x])
                        
                        model_data = pd.DataFrame.from_dict({param: param_vals, 'subj_id': subj_id_vals, 'group': rew_hist_vals, 'side': side_vals})
    
                        mem = sm.MixedLM.from_formula(param+' ~ C(group)', groups='subj_id', data=model_data, missing='drop', vc_formula=vc_form).fit() #vc_formula=vc_form, 
                        
                        # get the group comparison p-values
                        p_val_sel = mem.pvalues.index.str.contains('C(group)', regex=False)
                        group_p_vals = mem.pvalues[p_val_sel]
                        group_p_vals.index = first_group+' -> '+group_p_vals.index
                        # only save the unique comparisons
                        p_vals.append(group_p_vals.iloc[i:])

                        if print_fit_results:
                            print('{}: {} {}, {}-aligned Mixed-effects Model, \'{}\' group compared against other groups:\n {}\n'.format(
                                   param, signal_type, region, align, first_group, mem.summary()))
                            print('Random Effects:\n{}\n'.format(mem.random_effects))
                            
                    p_vals = pd.concat(p_vals)
                    reject, _, _, corrected_alpha = smm.multipletests(p_vals, alpha=0.05, method='bonferroni')
                    p_vals = pd.DataFrame(p_vals).rename(columns={0: 'p_val'})
                    p_vals['reject null'] = reject
                    p_vals['corrected alpha'] = corrected_alpha
                    
                    print('{}: {} {}, {}-aligned pairwise group comparison p-values:\n {}\n'.format(
                           param, signal_type, region, align, p_vals))
                else:
                    group_mapping = {g: i for i, g in enumerate(rew_hist_groups)}
                    rew_hist_vals = region_metrics['rew_hist_bin_label'].apply(lambda x: group_mapping[x])
                    model_data = pd.DataFrame.from_dict({param: param_vals, 'subj_id': subj_id_vals, 'group': rew_hist_vals, 'side': side_vals})

                    mem = sm.MixedLM.from_formula(param+' ~ group', groups='subj_id', data=model_data, missing='drop', vc_formula=vc_form).fit() #vc_formula=vc_form, 
                    
                    if print_fit_results:
                        print('{}: {} {}, {}-aligned Mixed-effects Model, linear regression on rew history bin #:\n {}\n'.format(
                               param, signal_type, region, align, mem.summary()))
                        print('Random Effects:\n{}\n'.format(mem.random_effects))
                        
                    print('{}: {} {}, {}-aligned slope regression p-values:\n {}\n'.format(
                           param, signal_type, region, align, mem.pvalues))
                    
                
# %% perform n-back regression on peak height

# look at each region and property separately
parameters = ['peak_height'] # ['peak_time', 'peak_height', 'peak_width', 'decay_tau'] #
regions = ['DMS', 'PL']
aligns = [Align.reward] #Align.cue, 
signals = ['dff_iso', 'z_dff_iso'] # 'dff_iso', 'z_dff_iso'

include_current_side = False
include_side_reward_interaction = False
include_stay_switch = False
fit_subj_separate = False
plot_trial_0 = False
normalize_coeffs = True
print_fit_results = False
plot_fit_results = True
plot_regions_same_axis = True
use_ci_errors = False

lim_n_back_fits = [6] #np.arange(0,11) # rew_hist_n_back
lim_n_back_plot = 4

# define reusable plotting routine
fmt = {'fmt': '-'}
def plot_coeffs(coeffs, errors, ax, include_label=True, label_prefix='', fmt=fmt):
    
    x = np.arange(lim_n_back_plot+1)
    
    if not plot_trial_0:
        x = x[x != 0]
    
    if include_side_reward_interaction:

        contra_labels = []
        ipsi_labels = []
        if plot_trial_0:
            if include_current_side:
                contra_labels.append('contra choice')
                ipsi_labels.append('ipsi choice')
            else:
                contra_labels.append('all choices')
                ipsi_labels.append('all choices')
        
        contra_labels.extend(['contra, reward ({})'.format(i) for i in range(-1, -lim_n_back_plot-1, -1)])
        ipsi_labels.extend(['ipsi, reward ({})'.format(i) for i in range(-1, -lim_n_back_plot-1, -1)])
        
        contra_params = params[contra_labels].to_numpy()
        contra_errors = errors[contra_labels].to_numpy()
        ipsi_params = params[ipsi_labels].to_numpy()
        ipsi_errors = errors[ipsi_labels].to_numpy()
        
        if 'color' in fmt:
            c = fmt.pop('color')
            first_color = c
            second_color = c
        else:
            first_color = 'C0'
            second_color = 'C1'
            
        if include_label:
            ax.errorbar(x, contra_params, yerr=contra_errors, label='{}Reward History before Contra Choices'.format(label_prefix), color=first_color, **fmt)
            ax.errorbar(x, ipsi_params, yerr=ipsi_errors, label='{}Reward History before Ipsi Choices'.format(label_prefix), color=second_color, **fmt)
        else:
            ax.errorbar(x, contra_params, yerr=contra_errors, label='_', color=first_color, **fmt)
            ax.errorbar(x, ipsi_params, yerr=ipsi_errors, label='_', color=second_color, **fmt)
        
    else:
        labels = []
        
        if plot_trial_0:
            if include_current_side:
                
                if 'color' in fmt:
                    c = fmt.pop('color')
                    first_color = c
                    second_color = c
                    next_color = c
                else:
                    first_color = 'C0'
                    second_color = 'C1'
                    next_color = 'C2'
                
                contra_params = params['contra choice']
                contra_errors = errors['contra choice'].to_numpy()
                ipsi_params = params['ipsi choice']
                ipsi_errors = errors['ipsi choice'].to_numpy()
                
                if include_label:
                    ax.errorbar(0, contra_params, yerr=contra_errors, label='{}Contra Choice'.format(label_prefix), color=first_color, **fmt)
                    ax.errorbar(0, ipsi_params, yerr=ipsi_errors, label='{}Ipsi Choice'.format(label_prefix), color=second_color, **fmt)
                else:
                    ax.errorbar(0, contra_params, yerr=contra_errors, label='_', color=first_color, **fmt)
                    ax.errorbar(0, ipsi_params, yerr=ipsi_errors, label='_', color=second_color, **fmt)

                x = x[x != 0]
            else:
                labels.append('all choices')
                if 'color' in fmt:
                    c = fmt.pop('color')
                    next_color = c
                else:
                    next_color = 'C0'
        else:
            if 'color' in fmt:
                c = fmt.pop('color')
                next_color = c
            else:
                next_color = 'C0'
        
        labels.extend(['reward ({})'.format(i) for i in range(-1, -lim_n_back_plot-1, -1)])
        plot_params = params[labels].to_numpy()
        plot_errors = errors[labels].to_numpy()
        
        if include_label:
            ax.errorbar(x, plot_params, yerr=plot_errors, label='{}Reward History'.format(label_prefix), color=next_color, **fmt)
        else:
            ax.errorbar(x, plot_params, yerr=plot_errors, label='_', color=next_color, **fmt)
            
    ax.set_xticks(x)

# define method to normalize coefficients based on maximum value to compute percent change from maximum
def norm_coeffs(coeffs, errors):
    max_coeff = coeffs.max()
    coeffs = coeffs/max_coeff * 100
    errors = errors/max_coeff * 100
    
    return coeffs, errors

peak_metric_subjs = np.unique(filt_peak_metrics['subj_id'])
all_loglike = {s: {p: {r: {a: [] for a in aligns} for r in regions} for p in parameters} for s in signals}
subj_loglike = {s: {p: {r: {a: [[] for subj in peak_metric_subjs] for a in aligns} for r in regions} for p in parameters} for s in signals}

for lim_n_back_fit in lim_n_back_fits:        
    # build predictor matrix labels
    if include_current_side:
        predictor_labels = ['contra choice', 'ipsi choice']
    else:
        predictor_labels = ['all choices']
    
    if include_side_reward_interaction:
        predictor_labels.extend(['contra, reward ({})'.format(i) for i in range(-1, -lim_n_back_fit-1, -1)])
        predictor_labels.extend(['ipsi, reward ({})'.format(i) for i in range(-1, -lim_n_back_fit-1, -1)])
    else:
        predictor_labels.extend(['reward ({})'.format(i) for i in range(-1, -lim_n_back_fit-1, -1)])
    
    for signal_type in signals:
        signal_label, y_label = fpah.get_signal_type_labels(signal_type)
        
        for param in parameters:
            
            all_params = {r: {a: {'params': [], 'cis': [], 'ses': []} for a in aligns} for r in regions}
            subj_params = {r: {a: {'params': [], 'cis': [], 'ses': []} for a in aligns} for r in regions}
            
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
    
                    param_vals = region_metrics[param].reset_index(drop=True)
                    
                    # build predictor matrix
                    if include_current_side:
                        predictor_vals = np.hstack([region_metrics['side'].to_numpy()[:,None] == 'contra', region_metrics['side'].to_numpy()[:,None] == 'ipsi'])
                    else:
                        predictor_vals = np.full((len(param_vals), 1), 1)
                    
                    if include_side_reward_interaction:
                        predictor_vals = np.hstack([predictor_vals, np.vstack(region_metrics.apply(lambda r: (r['side'] == 'contra') * np.array(r['rew_hist'][:lim_n_back_fit]), axis=1))])
                        predictor_vals = np.hstack([predictor_vals, np.vstack(region_metrics.apply(lambda r: (r['side'] == 'ipsi') * np.array(r['rew_hist'][:lim_n_back_fit]), axis=1))])
                    else:
                        predictor_vals = np.hstack([predictor_vals, np.vstack(region_metrics['rew_hist'].apply(lambda x: x[:lim_n_back_fit]))])
                    
                    predictors = pd.DataFrame(predictor_vals.astype(int), columns=predictor_labels)
    
                    mem = sm.MixedLM(param_vals, predictors, groups=region_metrics['subj_id']).fit()
                    if print_fit_results:
                        print('{}: {} {}, {}-aligned All Subjects Mixed-effects Regression Results:\n{}\n'.format(param, signal_type, region, align, mem.summary()))
                        print('Random Effects:\n{}\n'.format(mem.random_effects))
                        
                    print('{}: {} {}, {}-aligned All Subjects p-values:\n {}\n'.format(
                           param, signal_type, region, align, mem.pvalues))
                    
                    all_params[region][align]['params'] = mem.params
                    all_params[region][align]['cis'] = mem.conf_int(0.05)
                    all_params[region][align]['se'] = mem.bse
                    
                    all_loglike[signal_type][param][region][align].append(mem.llf)
                    
                    if fit_subj_separate:
                        for k, subj_id in enumerate(peak_metric_subjs):
                            subj_sel = region_metrics['subj_id'].to_numpy() == subj_id
        
                            mem = sm.OLS(param_vals[subj_sel], predictors.loc[subj_sel,:]).fit()
                            if print_fit_results:
                                print('{}: {} {}, {}-aligned Subject {} OLS Regression Results:\n{}\n'.format(param, signal_type, region, align, subj_id, mem.summary()))
                                
                            print('{}: {} {}, {}-aligned Subject {} p-values:\n {}\n'.format(
                                   param, signal_type, region, align, subj_id, mem.pvalues))
                            
                            subj_params[region][align]['params'].append(mem.params)
                            subj_params[region][align]['cis'].append(mem.conf_int(0.05))
                            subj_params[region][align]['se'].append(mem.bse)
                            
                            subj_loglike[signal_type][param][region][align][k].append(mem.llf)
                            
            if plot_fit_results:
                if plot_regions_same_axis:
                    fig, axs = plt.subplots(1, len(aligns), layout='constrained', figsize=(4*len(aligns), 3), sharey='row')
                    axs = np.array(axs).reshape(len(aligns))
                else:
                    fig, axs = plt.subplots(len(regions), len(aligns), layout='constrained', figsize=(4*len(aligns), 3*len(regions)), sharey='row')
                    axs = axs.reshape((len(regions), len(aligns)))
                    
                plot_name = '{}_{}_{}_{}_back_regression_{}'.format('_'.join(regions), '_'.join(aligns), param, lim_n_back_plot, signal_type)
                
                for i, region in enumerate(regions):
                    for j, align in enumerate(aligns):
                        if plot_regions_same_axis:
                            ax = axs[j]
                            ax.set_title('{}-aligned'.format(align))
                            region_colors = ['C0', 'C1']
                        else:
                            ax = axs[i, j]
                            ax.set_title('{}, {}-aligned'.format(region, align))
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        
                        if fit_subj_separate:
                            for k in range(len(peak_metric_subjs)):
                                params = subj_params[region][align]['params'][k]
                                if use_ci_errors:
                                    # cis are symmetric, so only find the difference to the upper bound
                                    errors = subj_params[region][align]['cis'][k][1] - params
                                else:
                                    errors = subj_params[region][align]['se'][k]
                                
                                if normalize_coeffs:
                                    params, errors = norm_coeffs(params, errors)
                                
                                if plot_regions_same_axis:
                                    plot_coeffs(params, errors, ax, include_label=False, label_prefix=region+' ', fmt={**fmt, 'alpha': 0.4, 'capsize': 0, 'color': region_colors[i]})
                                else:
                                    plot_coeffs(params, errors, ax, include_label=False, fmt={**fmt, 'alpha': 0.4, 'capsize': 0})
        
                        params = all_params[region][align]['params']
                        if use_ci_errors:
                            # cis are symmetric, so only find the difference to the upper bound
                            errors = all_params[region][align]['cis'][1] - params
                        else:
                            errors = all_params[region][align]['se']
                        
                        if normalize_coeffs:
                            params, errors = norm_coeffs(params, errors)
                            
                        if plot_regions_same_axis:
                            plot_coeffs(params, errors, ax, label_prefix=region+' ', fmt={**fmt, 'alpha': 1, 'capsize': 3, 'capthick': 2, 'linewidth': 2, 'color': region_colors[i]})
                        else:
                            plot_coeffs(params, errors, ax, fmt={**fmt, 'alpha': 1, 'capsize': 3, 'capthick': 2, 'linewidth': 2})
                        
                        plot_utils.show_axis_labels(ax)
                        if normalize_coeffs:
                            ax.set_ylabel('% Max Coefficient ({})'.format(y_label))
                        else:
                            ax.set_ylabel('Regression Coefficient ({})'.format(y_label))
                        ax.set_xlabel('Trials Back')
                        ax.legend(loc='best')
                        
                fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

# plot log likelihood values based on the number of trials back
if len(lim_n_back_fits) > 1:
    for signal_type in signals:
        for param in parameters:
            
            fig, axs = plt.subplots(len(regions), len(aligns), layout='constrained', figsize=(4*len(aligns), 3*len(regions)))
            axs = axs.reshape((len(regions), len(aligns)))

            fig.suptitle('Log Likelihood By Trials Back.\nCurrent Choice Side: {}. Side/History Interaction: {}'.format(include_current_side, include_side_reward_interaction))
            
            for i, region in enumerate(regions):
                for j, align in enumerate(aligns):
                    ax = axs[i, j]
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if fit_subj_separate:
                        for k in range(len(peak_metric_subjs)):
                            llhs = subj_loglike[signal_type][param][region][align][k]
                            llhs = llhs/-np.max(llhs)
                            ax.plot(lim_n_back_fits, llhs, alpha=0.5, color='gray')
                            
                    llhs = all_loglike[signal_type][param][region][align]
                    if fit_subj_separate:
                        llhs = llhs/-np.max(llhs)
                        ax.set_ylabel('Normalized Log Likelihood')
                    else:
                        ax.set_ylabel('Log Likelihood')
                        
                    ax.plot(lim_n_back_fits, llhs, color='C0')
                    ax.set_xlabel('Regression Trials Back')
    
                
# %% Perform reward history regression over time

regions = ['DMS', 'PL']
aligns = [Align.cue, Align.reward] #
signals = ['z_dff_iso']
included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]

hist_n_back = 5

include_current_side = False
include_stay_switch = False
include_side_reward_interaction = False
analyze_meta_subj = True

# first build predictor and response matrices
t = aligned_signals['t']
    
subj_stacked_signals = {subj_id: {s: {a: {r: np.zeros((0, len(t[r]))) for r in regions} 
                                      for a in aligns} 
                                  for s in signals}
                        for subj_id in included_subjs}

subj_predictors = {subj_id: [] for subj_id in included_subjs}

for subj_id in included_subjs:
    for sess_id in sess_ids[subj_id]:
        
        # build predictor matrix by trial
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        responded = ~np.isnan(trial_data['response_time']).to_numpy()
        rewarded = trial_data['rewarded'].to_numpy()[responded].astype(int)
        choice = trial_data['choice'].to_numpy()[responded]
        switches = np.concatenate(([False], choice[:-1] != choice[1:])).astype(int)
        rew_hist = pd.cut(trial_data['rew_rate_hist_all'], rew_hist_bins)

        # make buffered predictors to be able to go n back
        buff_reward = np.concatenate((np.full((hist_n_back), 0), rewarded, [0]))
        
        # stack buffered choices and rewards back in time
        preds = {'reward ({})'.format(i-hist_n_back): buff_reward[i:-hist_n_back+i-1] for i in range(hist_n_back, -1, -1)}
        if include_current_side:
            preds.update({'choice': choice})
        
        if include_stay_switch:
            preds.update({'switch': switches})
            
        subj_predictors[subj_id].append(pd.DataFrame(preds))
            
        for region in regions:
            t_r = t[region]
            
            if sess_id in ignored_signals[region]:
                continue

            for signal_type in signals:
                if not signal_type in aligned_signals[subj_id][sess_id]:
                    continue
                for align in aligns:
                    if not align in aligned_signals[subj_id][sess_id][signal_type]:
                        continue
            
                    mat = aligned_signals[subj_id][sess_id][signal_type][align][region]
                    
                    # normalize all grouped matrices to the pre-event signal of the lowest reward rate
                    baseline_mat = mat[(rew_hist == rew_hist_bins[0]) & responded,:]
                    if baseline_mat.shape[0] > 0:
                        baseline_sel = (t_r >= -0.1) & (t_r < 0)
                        baseline = np.nanmean(baseline_mat[:,baseline_sel])
                    else:
                        baseline = 0
                        
                    mat = mat - baseline
                    
                    subj_stacked_signals[subj_id][signal_type][align][region] = np.vstack((subj_stacked_signals[subj_id][signal_type][align][region], mat[responded,:]))
            
    subj_predictors[subj_id] = pd.concat(subj_predictors[subj_id])
    
# if analyzing meta subject, create new subject entry with everything stacked
if analyze_meta_subj:
    subj_predictors['all'] = {r: [] for r in regions}
    subj_stacked_signals['all'] = {s: {a: {r: [] for r in regions} 
                                       for a in aligns} 
                                   for s in signals}
    
    for signal_type in signals:
        for align in aligns:
            for region in regions:
                subj_stacked_signals['all'][signal_type][align][region] = np.vstack([subj_stacked_signals[subj_id][signal_type][align][region] for subj_id in included_subjs])

# perform regression over time

# define method to perform the regression
def regress_over_time(signals, predictors):
    params = []
    ci_lower = []
    ci_upper = []
    se = []
    p_vals = []
    for i in range(signals.shape[1]):
        t_sig = signals[:,i]
        # remove trials with nans at this time step
        rem_nan_sel = ~np.isnan(t_sig)
        mem = sm.OLS(t_sig[rem_nan_sel], predictors[rem_nan_sel]).fit()
        params.append(mem.params.to_dict())
        cis = mem.conf_int(0.05)
        ci_lower.append(cis[0].to_dict())
        ci_upper.append(cis[1].to_dict())
        se.append(mem.bse.to_dict())
        p_vals.append(mem.pvalues.to_dict())
        
    return {'params': pd.DataFrame(params), 
            'ci_lower': pd.DataFrame(ci_lower), 
            'ci_upper': pd.DataFrame(ci_upper),
            'se': pd.DataFrame(se),
            'p_vals': pd.DataFrame(p_vals)}
    
reg_params = {subj_id: {s: {r: {a: {} for a in aligns} 
                            for r in regions} 
                        for s in signals}
              for subj_id in subj_predictors.keys()}

for subj_id in included_subjs:
    for region in regions:
        pred_mat = subj_predictors[subj_id].copy()
        
        if include_current_side:
            region_side = implant_info[subj_id][region]['side']
            choice_side = pred_mat['choice'].apply(lambda x: fpah.get_implant_side_type(x, region_side))
            contra_choice = choice_side == 'contra'
            pred_mat['contra choice'] = contra_choice.astype(int)
            pred_mat['ipsi choice'] = (~contra_choice).astype(int)
            pred_mat = pred_mat.drop('choice', axis=1)
        else:
            pred_mat['intercept'] = 1
        
        # need to do this here since animals' implants are on different sides
        if analyze_meta_subj:
            subj_predictors['all'][region].append(pred_mat)

        for signal_type in signals:
            for align in aligns:
                
                signal_mat = subj_stacked_signals[subj_id][signal_type][align][region]
                 
                reg_params[subj_id][signal_type][region][align] = regress_over_time(signal_mat, pred_mat)
                
if analyze_meta_subj:
    subj_id = 'all'
    for region in regions:
        pred_mat = pd.concat(subj_predictors[subj_id][region])

        for signal_type in signals:
            for align in aligns:
                
                signal_mat = subj_stacked_signals[subj_id][signal_type][align][region]
                 
                reg_params[subj_id][signal_type][region][align] = regress_over_time(signal_mat, pred_mat)
                
# %% plot regression coefficients over time

# Create the colormap
#cmap = LinearSegmentedColormap.from_list('red_to_blue', [plt.cm.Reds(0.7), plt.cm.Blues(0.7)])
cmap = LinearSegmentedColormap.from_list('red_to_blue', [plt.cm.Reds(0.7), plt.cm.Blues(0.7)])

plot_signals = ['z_dff_iso']
plot_regions = ['DMS', 'PL']
plot_aligns = [Align.reward] #Align.cue, 

plot_ind_subj = False
plot_subj_average = False
plot_meta_subj = True
include_current_reward = True
use_ci_errors = False
plot_sig = True
sig_lvl = 0.05

lim_n_back = 4
plot_group_labels = ['Reward History']
if include_current_reward:
    plot_groups = [['reward ({})'.format(i-lim_n_back) for i in range(lim_n_back, -1, -1)]]
    #rew_colors = plt.cm.Reds(np.linspace(1,0.4,lim_n_back+1))
    rew_colors = cmap(np.linspace(0,1,lim_n_back+1))
else:
    plot_groups = [['reward ({})'.format(i-lim_n_back) for i in range(lim_n_back-1, -1, -1)]]
    #rew_colors = plt.cm.Reds(np.linspace(1,0.4,lim_n_back))
    rew_colors = cmap(np.linspace(0,1,lim_n_back))
    
group_colors = [[c for c in rew_colors]]

if include_current_side:
    plot_group_labels.append('Choice Side')
    plot_groups.append(['contra choice', 'ipsi choice'])
    group_colors.append(['C0', 'C1'])

if include_stay_switch:
    plot_group_labels.append('Side Switch')
    plot_groups.append(['switch'])
    group_colors.append(['C0'])
    
plot_lims = {Align.cue: {'DMS': [-0.1,0.8], 'PL': [-1,8]},
             Align.reward: {'DMS': [-0.2,1], 'PL': [-2,10]}}

width_ratios = [np.diff(plot_lims[align]['DMS'])[0] for align in plot_aligns]
    
# define common plotting routine
def plot_regress_over_time(params, t, plot_cols, ax, ci_lower=None, ci_upper=None, error=None, p_vals=None, t_sel=None, colors=None):
    sig_y_dist = 0.03
    if t_sel is None:
        t_sel = np.full_like(t, True)
    
    line_colors = []
    for i, col in enumerate(plot_cols):
        vals = params[col].to_numpy()
        if colors is None:
            color = None
        else:
            color = colors[i]
            
        if not ci_lower is None and not ci_upper is None:
            error = np.abs(np.vstack((ci_lower[col], ci_upper[col])) - vals[None,:])
            line, _ = plot_utils.plot_psth(t[t_sel], vals[t_sel], error[:,t_sel], ax=ax, label=col, plot_x0=False, color=color)
        elif not error is None:
            line, _ = plot_utils.plot_psth(t[t_sel], vals[t_sel], error[col][t_sel], ax=ax, label=col, plot_x0=False, color=color)
        else:
            line, _ = plot_utils.plot_psth(t[t_sel], vals[t_sel], ax=ax, label=col, plot_x0=False, color=color)
        
        line_colors.append(line.get_color())
        
            
    plot_utils.plot_dashlines(0, dir='v', ax=ax)
    plot_utils.plot_dashlines(0, dir='h', ax=ax)
    
    # plot significance from 0    
    if plot_sig:
        y_min, y_max = ax.get_ylim()
        y_offset = (y_max-y_min)*sig_y_dist

        for i, col in enumerate(plot_cols):
            # perform correction
            reject, corrected_pvals, _, _  = smm.multipletests(p_vals[col][t_sel], alpha=0.05, method='fdr_bh')
            sig_t = t[t_sel][reject]
            ax.scatter(sig_t, np.full_like(sig_t, y_max+i*y_offset), color=line_colors[i], marker='.')

    ax.set_xlabel('Time (s)')
    ax.legend()

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
for plot_group, group_label, colors in zip(plot_groups, plot_group_labels, group_colors):
    for signal_type in plot_signals:
        signal_label, y_label = fpah.get_signal_type_labels(signal_type)
        
        if plot_ind_subj:
            for subj_id in included_subjs:
                fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), width_ratios=width_ratios, sharey='row')
                axs = np.array(axs).reshape((n_rows, n_cols))
                
                fig.suptitle('{} Regression, Subj {}, {}'.format(group_label, subj_id, signal_label))
                
                for i, region in enumerate(plot_regions):
                    t_r = t[region]
                    for j, align in enumerate(plot_aligns):
                        ax = axs[i,j]
                        
                        subj_params = reg_params[subj_id][signal_type][region][align]
                        t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                        
                        if use_ci_errors:
                            plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, 
                                                   ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], 
                                                   p_vals=subj_params['p_vals'], t_sel=t_sel, colors=colors)
                        else:
                            plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, 
                                                   error=subj_params['se'], p_vals=subj_params['p_vals'], 
                                                   t_sel=t_sel, colors=colors)   
                        
                        if j == 0:
                            ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
        if plot_subj_average:
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            fig.suptitle('{} Regression, Subject Avg, {}'.format(group_label, signal_label))
            
            for i, region in enumerate(plot_regions):
                t_r = t[region]
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    
                    # average coefficients across subjects
                    all_params = pd.concat([reg_params[subj_id][signal_type][region][align]['params'] for subj_id in included_subjs])
                    param_avg = all_params.groupby(level=0).mean()
                    param_se = all_params.groupby(level=0).std() / np.sqrt(len(included_subjs))
                    
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
            
                    plot_regress_over_time(param_avg, t_r, plot_group, ax, 
                                           error=param_se, t_sel=t_sel, colors=colors)
                    
                    if j == 0:
                        ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
        if plot_meta_subj:
            subj_id = 'all'
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            
            fig.suptitle('{} Regression, Subj {}, {}'.format(group_label, subj_id, signal_label))
            
            for i, region in enumerate(plot_regions):
                t_r = t[region]
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    
                    subj_params = reg_params[subj_id][signal_type][region][align]
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    
                    if use_ci_errors:
                        plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, 
                                               ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], 
                                               p_vals=subj_params['p_vals'], t_sel=t_sel, colors=colors)
                    else:
                        plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, 
                                               error=subj_params['se'], p_vals=subj_params['p_vals'], 
                                               t_sel=t_sel, colors=colors)
                    
                    if j == 0:
                        ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
            plot_name = '{}_{}_time_regression_{}_{}'.format('_'.join(plot_regions), '_'.join(plot_aligns), group_label, signal_type)
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
            
            