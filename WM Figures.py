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
from matplotlib.ticker import MultipleLocator
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

implant_info = db_access.get_fp_implant_info(subj_ids)

# %% Set up variables
signal_types = ['z_dff_iso']
alignments = [Align.cpoke_in, Align.tone, Align.cue, Align.reward]
xlims = {Align.cpoke_in: {'DMS': [-1,1], 'PL': [-1,1]},
         Align.tone: {'DMS': [-1,2], 'PL': [-3,5]},
         Align.cue: {'DMS': [-1,2], 'PL': [-3,5]},
         Align.reward: {'DMS': [-1,2], 'PL': [-3,15]}}

regions = ['PL', 'DMS'] 
recalculate = False

filename = 'sel_wm_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        save_data = pickle.load(f)
        
    aligned_signals = save_data['aligned_signals']
else:
    aligned_signals = {subjid: {sessid: {sig_type: {align: {region: [] for region in regions} 
                                                    for align in alignments}
                                         for sig_type in signal_types}
                                for sessid in sess_ids[subjid]} 
                       for subjid in subj_ids}
                       
rew_rate_n_back = 3
bah.calc_rew_rate_hist(sess_data, n_back=rew_rate_n_back, kernel='uniform')

# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
n_rew_hist_bins = 4
rew_hist_bin_edges = np.linspace(-0.001, 1.001, n_rew_hist_bins+1)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{:.0f}-{:.0f}%'.format(abs(b.left)*100, b.right*100) for b in rew_hist_bins}

side_labels = {'ipsi': 'Ipsi', 'contra': 'Contra'}
rew_labels = {'rewarded': 'rew', 'unrewarded': 'unrew'}
bin_labels = {b:'{:.0f}'.format(np.mean([np.abs(np.ceil(b.left*rew_rate_n_back)), np.floor(b.right*rew_rate_n_back)])) for b in rew_hist_bins}

# %% Build signal matrices aligned to alignment points

# choose 405 over 420 when there are sessions with both for 3.6
isos = {182: ['405', '420'], 202: ['405', '420'], 179: ['420', '405'],
        180: ['420', '405'], 188: ['420', '405'], 191: ['420', '405'], 207: ['420', '405']}

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:
        if sess_id in fpah.__sess_ignore:
            continue

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, fit_baseline=False)
        fp_data = fp_data[subj_id][sess_id]

        trial_data = sess_data[sess_data['sessid'] == sess_id]

        ts = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        reward_ts = trial_start_ts + trial_data['reward_time']
        first_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[0] if utils.is_list(x) else x)
        second_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[1] if utils.is_list(x) else np.nan)

        for signal_type in signal_types:
            for align in alignments:
                match align:
                    case Align.cpoke_in:
                        align_ts = cpoke_in_ts
                        # no mask
                        mask_lims = None
                        
                    case Align.tone:
                        align_ts = [first_tone_ts, second_tone_ts]
                        # no mask
                        mask_lims = None
                        
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

                        lims = xlims[align][region]
                        
                        if align == Align.tone:
                            for tone_ts in align_ts:
                                mat, t = fp_utils.build_signal_matrix(signal, ts, tone_ts, -lims[0], lims[1], mask_lims=mask_lims)
                                aligned_signals[subj_id][sess_id][signal_type][align][region].append(mat)
                        else:
                            mat, t = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)
                            aligned_signals[subj_id][sess_id][signal_type][align][region] = mat

aligned_signals['t'] = {align: {region: [] for region in regions} for align in alignments}
dt = fp_data['dec_info']['decimated_dt']
for align in alignments:
    for region in regions:
        aligned_signals['t'][align][region] = np.arange(xlims[align][region][0], xlims[align][region][1]+dt, dt)

with open(save_path, 'wb') as f:
    pickle.dump({'aligned_signals': aligned_signals,
                 'metadata': {'signal_types': signal_types,
                             'alignments': alignments,
                             'regions': regions,
                             'xlims': xlims}}, f)

# %% Stack aligned signals

ignored_sessions = [] #sess_info.loc[sess_info['proto_stage'] == 'ToneCatDelayResp2_7', 'sessid'].to_numpy()

ignored_signals = {'PL': [],
                   'DMS': []}

ignored_subjects = []

plot_regions = ['DMS', 'PL']
alignments = [Align.tone, Align.cue, Align.reward]
signal_types = ['z_dff_iso'] # 'dff_iso',
norm_types = ['raw', 'norm']
sides = ['contra', 'ipsi']
baseline_lims = [-0.1, 0]

t = aligned_signals['t']
stacked_signals = {s: {a: {r: {n: {} for n in norm_types} 
                           for r in plot_regions} 
                       for a in alignments} 
                   for s in signal_types}

def stack_mat(stacked_mats, key, mat):
    if not key in stacked_mats:
        stacked_mats[key] = np.zeros((0, mat.shape[1]))
    else:
        stacked_mats[key] = np.vstack((stacked_mats[key], mat))
        
# group trials together and stack across sessions

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
        n_tones = trial_data['n_tones']
        variant = trial_data['task_variant']
        incongruent = trial_data['incongruent']
        one_tone = n_tones == 1
        two_tone = n_tones == 2
        
        cpoke_out_ts = trial_data['cpoke_out_time']
        cue_ts = trial_data['response_cue_time']
        
        resp_rewarded = rewarded[responded]
        
        for region in regions:
            if sess_id in ignored_signals[region]:
                continue

            region_side = implant_info[subj_id][region]['side']
            choice_side = choice.apply(lambda x: fpah.get_implant_rel_side(x, region_side) if not x == 'none' else 'none').to_numpy()
            
            for signal_type in signal_types:
                if not signal_type in aligned_signals[subj_id][sess_id]:
                    continue
                for align in alignments:
                    if not align in aligned_signals[subj_id][sess_id][signal_type]:
                        continue

                    if len(aligned_signals[subj_id][sess_id][signal_type][align][region]) == 0:
                        continue
                    
                    t_r = t[align][region]
                    mat = aligned_signals[subj_id][sess_id][signal_type][align][region]
                    
                    for norm_type in norm_types:
                        
                        match align:
                            case Align.tone:
                                first_mat = mat[0]
                                second_mat = mat[1]
                                if norm_type == 'raw':
                                    first_mat_norm = first_mat
                                    second_mat_norm = second_mat
                                else:
                                    # normalize to the pre-poke baseline so normalization is same for both tones
                                    norm_t = t[Align.cpoke_in][region]
                                    poke_mat = aligned_signals[subj_id][sess_id][signal_type][Align.cpoke_in][region]
                                    baseline_sel = (norm_t >= baseline_lims[0]) & (norm_t <= baseline_lims[1])
                                    baseline = np.nanmean(poke_mat[:, baseline_sel], axis=1)[:,None]
                                    first_mat_norm = first_mat - baseline
                                    second_mat_norm = second_mat - baseline
                                
                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'one_tone_rewarded', first_mat_norm[responded & one_tone & rewarded,:])
                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'one_tone_unrewarded', first_mat_norm[responded & one_tone & ~rewarded,:])
                                
                                for side in sides:
                                    side_sel = choice_side == side
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'one_tone_hit_'+side, first_mat_norm[responded & one_tone & rewarded & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'one_tone_miss_'+side, first_mat_norm[responded & one_tone & ~rewarded & side_sel,:])
                                
                                for v in variants:
                                    v_sel = variant == v
                                    
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'first_tone_var_'+v, first_mat_norm[responded & two_tone & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'second_tone_var_'+v, second_mat_norm[responded & two_tone & v_sel,:])
                                    

                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'first_tone_hit_cong_var_'+v, first_mat_norm[responded & two_tone & ~incongruent & rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'first_tone_hit_incong_var_'+v, first_mat_norm[responded & two_tone & incongruent & rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'first_tone_miss_cong_var_'+v, first_mat_norm[responded & two_tone & ~incongruent & ~rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'first_tone_miss_incong_var_'+v, first_mat_norm[responded & two_tone & incongruent & ~rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'second_tone_hit_cong_var_'+v, second_mat_norm[responded & two_tone & ~incongruent & rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'second_tone_hit_incong_var_'+v, second_mat_norm[responded & two_tone & incongruent & rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'second_tone_miss_cong_var_'+v, second_mat_norm[responded & two_tone & ~incongruent & ~rewarded & v_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'second_tone_miss_incong_var_'+v, second_mat_norm[responded & two_tone & incongruent & ~rewarded & v_sel,:])
    
                            case Align.cue:
                                if norm_type == 'raw':
                                    norm_mat = mat
                                else:
                                    baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                                    norm_mat = mat - np.nanmean(mat[:, baseline_sel], axis=1)[:,None]
                                    
                                # only look at trials where the cue happened before poking out

                                poke_out_after_cue_sel = cpoke_out_ts > cue_ts
                                trial_sel = responded & poke_out_after_cue_sel
                                # trial_sel = responded

                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'all', norm_mat[trial_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rewarded', norm_mat[trial_sel & rewarded,:])
                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'unrewarded', norm_mat[trial_sel & ~rewarded,:])
                                
                                for side in sides:
                                    side_sel = choice_side == side
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], side, norm_mat[trial_sel & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rewarded_'+side, norm_mat[trial_sel & rewarded & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'unrewarded_'+side, norm_mat[trial_sel & ~rewarded & side_sel,:])
                                
                                for rew_bin in rew_hist_bins:
                                    rew_sel = rew_hist == rew_bin
                                    bin_str = rew_hist_bin_strs[rew_bin]
                                    
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rew_hist_'+bin_str, norm_mat[rew_sel & responded,:])

                                    for side in sides:
                                        side_sel = choice_side == side
                                        stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rew_hist_'+bin_str+'_'+side, norm_mat[rew_sel & responded & side_sel,:])

                                
                            case Align.reward:
                                if norm_type == 'raw':
                                    norm_mat = mat
                                else:
                                    baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                                    norm_mat = mat - np.nanmean(mat[:, baseline_sel], axis=1)[:,None]
                                
                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rewarded', norm_mat[responded & rewarded,:])
                                stack_mat(stacked_signals[signal_type][align][region][norm_type], 'unrewarded', norm_mat[responded & ~rewarded,:])
                                
                                for side in sides:
                                    side_sel = choice_side == side
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rewarded_'+side, norm_mat[responded & rewarded & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'unrewarded_'+side, norm_mat[responded & ~rewarded & side_sel,:])
                                
                                for rew_bin in rew_hist_bins:
                                    rew_sel = rew_hist == rew_bin
                                    bin_str = rew_hist_bin_strs[rew_bin]
                                    
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rew_hist_'+bin_str+'_rewarded', norm_mat[rew_sel & responded & rewarded,:])
                                    stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rew_hist_'+bin_str+'_unrewarded', norm_mat[rew_sel & responded & ~rewarded,:])
    
                                    for side in sides:
                                        side_sel = choice_side == side
                                        stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rew_hist_'+bin_str+'_rewarded_'+side, norm_mat[rew_sel & responded & rewarded & side_sel,:])
                                        stack_mat(stacked_signals[signal_type][align][region][norm_type], 'rew_hist_'+bin_str+'_unrewarded_'+side, norm_mat[rew_sel & responded & ~rewarded & side_sel,:])

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

align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}   

all_color = '#08AB36'
rew_color = '#BC141A'
unrew_color = '#1764AB'                                     

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
    
# %% Plot single tone by side & outcome

plot_regions = ['DMS']
signal_type = 'z_dff_iso'
norm_type = 'norm' # 'raw' # 
norm_to_zero = False
baseline_lims = [-0.1, 0]

align = Align.tone
groups = ['one_tone_hit_contra', 'one_tone_miss_ipsi', 'one_tone_hit_ipsi', 'one_tone_miss_contra']
group_labels = {'one_tone_hit_contra': 'Contra Hit', 'one_tone_miss_ipsi': 'Contra Miss',
                'one_tone_hit_ipsi': 'Ipsi Hit', 'one_tone_miss_contra': 'Ipsi Miss'}
#colors = {'one_tone_rewarded': rew_color, 'one_tone_unrewarded': unrew_color}

plot_lims = {'DMS': [-0.1,0.6], 'PL': [-0.5,1.25]}

n_rows = len(plot_regions)
n_cols = 1
t = aligned_signals['t']

x_label = 'Time (s)'
title = 'Single Tone Side & Outcome'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5, 4*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_one_tone_side_outcome_{}_{}'.format(align, signal_type, '_'.join(plot_regions))

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
            
            plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[group], plot_x0=False) #color=colors[group], 

        ax.set_title('{} {}'.format(region, title))
        plot_utils.plot_dashlines([0, 0.3], ax=ax)

        ax.set_ylabel(y_label)
        ax.set_xlabel(x_label)
        ax.legend()
        
    fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')
    

# %% Plot response cue by side & outcome

plot_regions = ['DMS', 'PL']
signal_type = 'z_dff_iso'
norm_type = 'norm' # 'raw' # 
norm_to_zero = False
baseline_lims = [-0.1, 0]

align = Align.cue
groups = [['contra', 'ipsi'], ['rewarded_contra', 'unrewarded_contra', 'rewarded_ipsi', 'unrewarded_ipsi']]
group_labels = {**side_labels, 'rewarded_contra': 'Contra Hit', 'unrewarded_contra': 'Contra Miss',
                'rewarded_ipsi': 'Ipsi Hit', 'unrewarded_ipsi': 'Ipsi Miss'}
#colors = {'one_tone_rewarded': rew_color, 'one_tone_unrewarded': unrew_color}

plot_lims = {'DMS': [-0.1,0.6], 'PL': [-1,2]}

n_rows = len(plot_regions)
n_cols = len(groups)
t = aligned_signals['t']

x_label = 'Time (s)'
title = 'Response Cue Choice Side & Outcome'

for signal_type in signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 3.5*n_rows))
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array([axs])
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_side_outcome_{}_{}'.format(align, signal_type, '_'.join(plot_regions))

    for i, region in enumerate(plot_regions):

        region_signals = stacked_signals[signal_type][align][region][norm_type]
        t_r = t[align][region]
        t_sel = (t_r > plot_lims[region][0]) & (t_r < plot_lims[region][1])
        
        for j, sub_groups in enumerate(groups):     
            ax = axs[i,j]

            for group in sub_groups:
                act = region_signals[group]
                if norm_to_zero:
                    baseline_sel = (t_r >= baseline_lims[0]) & (t_r <= baseline_lims[1])
                    act = act - np.nanmean(act[:, baseline_sel], axis=1)[:,None]
                
                error = calc_error(act, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[group], plot_x0=False) #color=colors[group], 
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
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


# %% Plot average traces by reward history

plot_regions = ['DMS'] # 'DMS', 'PL'
plot_aligns = [Align.cue, Align.reward] # 
plot_signal_types = ['z_dff_iso']
norm_type = 'raw' # 'raw' # 

# plot formatting
plot_dec = {'DMS': 1, 'PL': 2}
x_inc = {'DMS': 0.3, 'PL': 3}
y_inc = {'DMS': 0.3, 'PL': 0.3}

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
gen_groups = {Align.cue: ['rew_hist_{}'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
groups = {a: [group.format(rew_hist_bin_strs[rew_bin]) for group in gen_groups[a] for rew_bin in rew_hist_bins] for a in plot_aligns}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,1], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.1,1], 'PL': [-1,10]}}

#width_ratios = [1,2]
#width_ratios = [0.7,0.9]
if len(plot_aligns) == 1:
    width_ratios = [1]
else:
    width_ratios = [np.diff(plot_lims[a][plot_regions[0]])[0] for a in plot_aligns]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

plot_data = {}

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    plot_data[signal_type] = {}
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(4.5*n_cols, 3.5*n_rows), sharey='row', width_ratios=width_ratios)
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs)

    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = 'reward_hist_{}_back_{}_{}_{}'.format(rew_rate_n_back, signal_type, '_'.join(plot_aligns), '_'.join(plot_regions))

    for i, region in enumerate(plot_regions):
        plot_data[signal_type][region] = {}
        
        for j, align in enumerate(plot_aligns):
            match align:
                case Align.cue:
                    title = 'Response Cue'
                    legend_cols = 1
                    
                case Align.reward:
                    title = 'Reward Delivery'
                    legend_cols = 2

            ax = axs[i,j]
            
            plot_data[signal_type][region][align] = {}
            
            region_signals = stacked_signals[signal_type][align][region][norm_type]
            
            t_r = t[align][region]
            t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
            t_r = t_r[t_sel][::plot_dec[region]]
            
            plot_data[signal_type][region][align]['time'] = t_r
    
            for group, color in zip(groups[align], colors[align]):
                act = region_signals[group]
                error = calc_error(act, True)
                
                avg_act = np.nanmean(act, axis=0)[t_sel][::plot_dec[region]]
                avg_err = error[t_sel][::plot_dec[region]]
                
                plot_data[signal_type][region][align][group_labels_dict[group]+'_avg'] = avg_act
                plot_data[signal_type][region][align][group_labels_dict[group]+'_err'] = avg_err
                
                plot_utils.plot_psth(t_r, avg_act, avg_err, ax, label=group_labels_dict[group], color=color, plot_x0=False)
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
            #else:
                # ax.yaxis.set_tick_params(which='both', labelleft=True)
            ax.legend(ncols=legend_cols, loc='upper right', title='# Rewards in last {} Trials'.format(rew_rate_n_back))

            ax.set_xlabel(x_label)
            
            ax.xaxis.set_major_locator(MultipleLocator(x_inc[region]))
            ax.yaxis.set_major_locator(MultipleLocator(y_inc[region]))
            
            plot_data[signal_type][region][align] = pd.DataFrame.from_dict(plot_data[signal_type][region][align])
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')
        
# %% Plot average traces for reward history by side
plot_regions = ['PL'] # 'DMS, 'PL'
plot_aligns = [Align.cue, Align.reward] #  
plot_signal_types = ['z_dff_iso']
norm_type = 'raw'

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}
gen_groups = {Align.cue: ['rew_hist_{}_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,1], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.1,1], 'PL': [-1,10]}}

width_ratios = [np.diff(plot_lims[a][plot_regions[0]])[0] for a in plot_aligns]
#width_ratios = [2,10.5]
#width_ratios = [0.7,0.9]

n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

# plot each side on its own row and alignment in its own column. Each region gets its own figure

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    for region in plot_regions:
        
        plot_data[signal_type][region] = {}

        fig, axs = plt.subplots(2, n_cols, layout='constrained', figsize=(4.5*n_cols, 7), sharey=True, width_ratios=width_ratios)

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
                
                region_signals = stacked_signals[signal_type][align][region][norm_type]
        
                for group, color in zip(groups[align], colors[align]):
                    act = region_signals[group]
                    t_r = t[align][region]
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
                
            fpah.save_fig(fig, fpah.get_figure_save_path('Sel WM', '', plot_name), format='pdf')
            