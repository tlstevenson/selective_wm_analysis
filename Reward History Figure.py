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
from matplotlib.ticker import MultipleLocator
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

# get aggregated session information
sess_info = sess_data.groupby('subjid').agg(n_sess=('sessid', 'nunique'), n_trials=('sessid', 'size')).reset_index()
sess_info = pd.concat([sess_info, 
                       pd.DataFrame({'subjid': ['Avg'],
                                     'n_sess': [sess_info['n_sess'].mean()],
                                     'n_trials': [sess_info['n_trials'].mean()]})], ignore_index=True)

# make sure RT is filled in
sess_data['RT'] = sess_data['response_time'] - sess_data['response_cue_time']

# %% Get Count Information
ignored_subjects = [182]

count_data = sess_data[~sess_data['subjid'].isin(ignored_subjects)][['subjid', 'sessid', 'trial']]

n_subj = len(count_data['subjid'].unique())

# get total and average number of sessions/trials
trial_counts = count_data.groupby(['subjid', 'sessid']).agg('count').reset_index()
sess_counts = trial_counts.groupby('subjid').agg('count').reset_index()

total_sess = len(trial_counts)
avg_sess_mean = np.mean(sess_counts['sessid'])
avg_sess_std = np.std(sess_counts['sessid'])

total_trials = np.sum(trial_counts['trial'])
avg_trial_mean = np.mean(trial_counts['trial'])
avg_trial_std = np.std(trial_counts['trial'])


# %% Set up variables
signal_types = ['dff_iso_baseline', 'z_dff_iso_baseline', 'dff_iso_baseline_fband', 'z_dff_iso_baseline_fband']
alignments = [Align.cue, Align.cpoke_out, Align.resp, Align.reward]
regions = ['DMS', 'DLS', 'PL']
xlims = {Align.cue: {'DMS': [-1,2], 'DLS': [-1,2], 'PL': [-3,5]},
         Align.cpoke_out: {'DMS': [-1,2], 'DLS': [-1,2], 'PL': [-3,5]},
         Align.resp: {'DMS': [-1,2], 'DLS': [-1,2], 'PL': [-3,5]},
         Align.reward: {'DMS': [-1,2], 'DLS': [-1,2], 'PL': [-3,20]}}

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
                       
ignored_signals = {'PL': [],
                   'DMS': [],
                   'DLS': []}

ignored_subjects = [182] # [179]

# %% Build signal matrices aligned to alignment points

tilt_t = False
band_iso_fit = False
baseline_correction = True
baseline_band_iso_fit = True
save_results = True

for subj_id in subj_ids:
    for sess_id in sess_ids[subj_id]:
        if sess_id in fpah.__sess_ignore:
            continue

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, iso_lpf=10, tilt_t=tilt_t, band_iso_fit=band_iso_fit, 
                                       baseline_correction=baseline_correction, baseline_band_iso_fit=baseline_band_iso_fit)
        fp_data = fp_data[subj_id][sess_id]

        trial_data = sess_data[sess_data['sessid'] == sess_id]

        ts = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        resp_ts = trial_start_ts + trial_data['response_time']
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
                        
                    case Align.cpoke_out:
                        align_ts = cpoke_out_ts
                        # no mask
                        mask_lims = None
                        
                    case Align.resp:
                        align_ts = resp_ts
                        # no mask
                        mask_lims = None
                        
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
                        
                        mat, t = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)
                        aligned_signals[subj_id][sess_id][signal_type][align][region] = mat

aligned_signals['t'] = {align: {region: [] for region in regions} for align in alignments}
dt = fp_data['dec_info']['decimated_dt']
for align in alignments:
    for region in regions:
        aligned_signals['t'][align][region] = np.arange(xlims[align][region][0], xlims[align][region][1]+dt, dt)

if save_results:
    with open(save_path, 'wb') as f:
        pickle.dump({'aligned_signals': aligned_signals,
                     'metadata': {'signal_types': signal_types,
                                 'alignments': alignments,
                                 'regions': regions,
                                 'xlims': xlims}}, f)

# %% look at cpoke in latencies by prior reward
reward_trials = sess_data[sess_data['rewarded'] == True]
unreward_trials = sess_data[sess_data['rewarded'] == False]

# plot cpoke in latency histograms
# get bins
bin_width = 2
x_max = 50 # np.max(sess_data['next_cpoke_in_latency'])
bins = np.arange(0, x_max+bin_width, bin_width)
_, ax = plt.subplots(1,1)
ax.hist(reward_trials['next_cpoke_in_latency'], bins=bins, density=True, label='Rewarded', alpha=0.5)
ax.hist(unreward_trials['next_cpoke_in_latency'], bins=bins, density=True, label='Unrewarded', alpha=0.5)
ax.set_title('Cpoke In Latencies By Previous Outcome - All')
ax.legend()

for subj_id in subj_ids:
    subj_rew_trials = reward_trials[reward_trials['subjid'] == subj_id]
    subj_unrew_trials = unreward_trials[unreward_trials['subjid'] == subj_id]
    
    _, ax = plt.subplots(1,1)
    ax.hist(subj_rew_trials['next_cpoke_in_latency'], bins=bins, density=True, label='Rewarded', alpha=0.5)
    ax.hist(subj_unrew_trials['next_cpoke_in_latency'], bins=bins, density=True, label='Unrewarded', alpha=0.5)
    ax.set_title('Cpoke In Latencies By Previous Outcome - {}'.format(subj_id))
    ax.legend()


# %% Analyze aligned signals

rew_n_back = 10
rew_rate_n_back = 3
bah.calc_rew_rate_hist(sess_data, n_back=rew_rate_n_back, kernel='uniform')

# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
n_rew_hist_bins = 4
rew_hist_bin_edges = np.linspace(-0.001, 1.001, n_rew_hist_bins+1)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{:.0f}-{:.0f}%'.format(abs(b.left)*100, b.right*100) for b in rew_hist_bins}

alignments = [Align.cue, Align.reward] #  
signal_types = ['dff_iso_baseline', 'z_dff_iso_baseline', 'dff_iso_baseline_fband', 'z_dff_iso_baseline_fband'] # 

norm_baseline = True
analyze_peaks = True

filter_props = {Align.cue: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'DLS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}},
                Align.reward: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                               'DLS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                               'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}}}

peak_find_props = {Align.cue: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                               'DLS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                               'PL': {'min_dist': 0.2, 'peak_tmax': 1.5, 'peak_edge_buffer': 0.2, 'lim_peak_width_to_edges': True}},
                   Align.reward: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                                  'DLS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                                  'PL': {'min_dist': 0.5, 'peak_tmax': 3.5, 'peak_edge_buffer': 0.2, 'lim_peak_width_to_edges': False}}}

sides = ['contra', 'ipsi']

t = aligned_signals['t']
stacked_signals = {s: {a: {r: {} for r in regions} 
                       for a in alignments} 
                   for s in signal_types}

if analyze_peaks:
    peak_metrics = []

reward_times = {r: {} for r in regions}

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
        reward_time = trial_data['reward_time'].to_numpy()[:,None]
        stays = choice[:-1].to_numpy() == choice[1:].to_numpy()
        switches = np.insert(~stays, 0, False)
        stays = np.insert(stays, 0, False)
        next_switch = trial_data['next_switch'][:-1].fillna(False).astype(bool).to_numpy()
        next_stay = trial_data['next_switch'][:-1].apply(lambda x: not x if isinstance(x, bool) else x).fillna(False).astype(bool).to_numpy()
        next_switch = np.append(next_switch, False)
        next_stay = np.append(next_stay, False)
        prev_rewarded = np.insert(rewarded[:-1], 0, False)
        prev_unrewarded = np.insert(~rewarded[:-1], 0, False)
        
        resp_rewarded = rewarded[responded]
        
        for region in regions:
            if sess_id in ignored_signals[region] or region not in implant_info[subj_id]:
                continue

            region_side = implant_info[subj_id][region]['side']
            choice_side = choice.apply(lambda x: fpah.get_implant_rel_side(x, region_side) if not x == 'none' else 'none').to_numpy()
            
            # save reward times
            for rew_bin in rew_hist_bins:
                rew_sel = rew_hist == rew_bin
                bin_str = rew_hist_bin_strs[rew_bin]
                
                stack_mat(reward_times[region], 'rew_hist_'+bin_str, reward_time[rew_sel & responded])
                stack_mat(reward_times[region], 'rew_hist_'+bin_str+'_rewarded', reward_time[rew_sel & responded & rewarded])
                stack_mat(reward_times[region], 'rew_hist_'+bin_str+'_unrewarded', reward_time[rew_sel & responded & ~rewarded])
                
                for side in sides:
                    side_sel = choice_side == side
                    stack_mat(reward_times[region], 'rew_hist_'+bin_str+'_'+side, reward_time[rew_sel & responded & side_sel,:])
                    stack_mat(reward_times[region], 'rew_hist_'+bin_str+'_rewarded_'+side, reward_time[rew_sel & responded & rewarded & side_sel,:])
                    stack_mat(reward_times[region], 'rew_hist_'+bin_str+'_unrewarded_'+side, reward_time[rew_sel & responded & ~rewarded & side_sel,:])
            

            for signal_type in signal_types:
                if not signal_type in aligned_signals[subj_id][sess_id]:
                    continue
                for align in alignments:
                    if not align in aligned_signals[subj_id][sess_id][signal_type]:
                        continue

                    t_r = t[align][region]
                    mat = aligned_signals[subj_id][sess_id][signal_type][align][region]

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
                                
                                if resp_idx < rew_n_back:
                                    buffer = np.full(rew_n_back-resp_idx, False)
                                    rew_hist_vec = np.flip(np.concatenate((buffer, resp_rewarded[:resp_idx])))
                                    contra_hist_vec = np.flip(np.concatenate((buffer, contra_choices[:resp_idx])))
                                    ipsi_hist_vec = np.flip(np.concatenate((buffer, ~contra_choices[:resp_idx])))
                                else:
                                    rew_hist_vec = np.flip(resp_rewarded[resp_idx-rew_n_back:resp_idx])
                                    contra_hist_vec = np.flip(contra_choices[resp_idx-rew_n_back:resp_idx])
                                    ipsi_hist_vec = np.flip(~contra_choices[resp_idx-rew_n_back:resp_idx])
    
                                peak_metrics.append(dict([('subj_id', subj_id), ('sess_id', sess_id), ('signal_type', signal_type), 
                                                         ('align', align.name), ('region', region), ('trial', i),
                                                         ('rewarded', rewarded[i]), ('rew_hist_bin', rew_hist.iloc[i]), 
                                                         ('side', choice_side[i]), ('rew_hist', rew_hist_vec),
                                                         ('contra_hist', contra_hist_vec), ('ipsi_hist', ipsi_hist_vec),
                                                         ('reward_time', reward_time[i]), ('RT', trial_data['RT'].iloc[i]),
                                                         ('cpoke_out_latency', trial_data['cpoke_out_latency'].iloc[i]), *metrics.items()]))
                                
                                resp_idx += 1
                    
                    if norm_baseline:
                        # normalize all grouped matrices to the pre-event signal of the lowest reward rate
                        baseline_mat = mat[(rew_hist == rew_hist_bins[0]) & responded,:]
                        if baseline_mat.shape[0] > 0:
                            baseline_sel = (t_r >= -0.1) & (t_r < 0)
                            baseline = np.nanmean(baseline_mat[:,baseline_sel])
                        else:
                            baseline = 0
                            
                        mat = mat - baseline
                    
                    # group trials together and stack across sessions
                    match align:
                        case Align.cue:
                            for side in sides:
                                side_sel = choice_side == side
                                stack_mat(stacked_signals[signal_type][align][region], 'stay_prev_reward_'+side, mat[responded & stays & prev_rewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'stay_prev_unreward_'+side, mat[responded & stays & prev_unrewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'switch_prev_reward_'+side, mat[responded & switches & prev_rewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'switch_prev_unreward_'+side, mat[responded & switches & prev_unrewarded & side_sel,:])
                        case Align.reward:
                            
                            stack_mat(stacked_signals[signal_type][align][region], 'rewarded_next_stay', mat[responded & rewarded & next_stay,:])
                            stack_mat(stacked_signals[signal_type][align][region], 'rewarded_next_switch', mat[responded & rewarded & next_switch,:])
                            stack_mat(stacked_signals[signal_type][align][region], 'unrewarded_next_stay', mat[responded & ~rewarded & next_stay,:])
                            stack_mat(stacked_signals[signal_type][align][region], 'unrewarded_next_switch', mat[responded & ~rewarded & next_switch,:])
                            
                            for side in sides:
                                side_sel = choice_side == side
                                stack_mat(stacked_signals[signal_type][align][region], 'stay_rewarded_prev_reward_'+side, mat[responded & stays & rewarded & prev_rewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'stay_rewarded_prev_unreward_'+side, mat[responded & stays & rewarded & prev_unrewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'switch_rewarded_prev_reward_'+side, mat[responded & switches & rewarded & prev_rewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'switch_rewarded_prev_unreward_'+side, mat[responded & switches & rewarded & prev_unrewarded & side_sel,:])
                                
                                stack_mat(stacked_signals[signal_type][align][region], 'stay_unrewarded_prev_reward_'+side, mat[responded & stays & ~rewarded & prev_rewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'stay_unrewarded_prev_unreward_'+side, mat[responded & stays & ~rewarded & prev_unrewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'switch_unrewarded_prev_reward_'+side, mat[responded & switches & ~rewarded & prev_rewarded & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'switch_unrewarded_prev_unreward_'+side, mat[responded & switches & ~rewarded & prev_unrewarded & side_sel,:])
                                
                                stack_mat(stacked_signals[signal_type][align][region], 'rewarded_next_stay_'+side, mat[responded & rewarded & next_stay & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'rewarded_next_switch_'+side, mat[responded & rewarded & next_switch & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'unrewarded_next_stay_'+side, mat[responded & ~rewarded & next_stay & side_sel,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'unrewarded_next_switch_'+side, mat[responded & ~rewarded & next_switch & side_sel,:])
                    
                    for rew_bin in rew_hist_bins:
                        rew_sel = rew_hist == rew_bin
                        bin_str = rew_hist_bin_strs[rew_bin]

                        match align:
                            case Align.cue:

                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str, mat[rew_sel & responded,:])
                                
                                # for baseline subtraction
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded', mat[rew_sel & responded & rewarded,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded', mat[rew_sel & responded & ~rewarded,:])

                                for side in sides:
                                    side_sel = choice_side == side
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_'+side, mat[rew_sel & responded & side_sel,:])

                            case Align.reward:
                                
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded', mat[rew_sel & responded & rewarded,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded', mat[rew_sel & responded & ~rewarded,:])
                                
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded_next_stay', mat[rew_sel & responded & rewarded & next_stay,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded_next_switch', mat[rew_sel & responded & rewarded & next_switch,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded_next_stay', mat[rew_sel & responded & ~rewarded & next_stay,:])
                                stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded_next_switch', mat[rew_sel & responded & ~rewarded & next_switch,:])
                                
                                for side in sides:
                                    side_sel = choice_side == side
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded_'+side, mat[rew_sel & responded & rewarded & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded_'+side, mat[rew_sel & responded & ~rewarded & side_sel,:])
                                    
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded_next_stay_'+side, mat[rew_sel & responded & rewarded & next_stay & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_rewarded_next_switch_'+side, mat[rew_sel & responded & rewarded & next_switch & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded_next_stay_'+side, mat[rew_sel & responded & ~rewarded & next_stay & side_sel,:])
                                    stack_mat(stacked_signals[signal_type][align][region], 'rew_hist_'+bin_str+'_unrewarded_next_switch_'+side, mat[rew_sel & responded & ~rewarded & next_switch & side_sel,:])

if analyze_peaks:
    peak_metrics = pd.DataFrame(peak_metrics)
    # drop unused columns
    peak_metrics.drop(['decay_tau', 'decay_params', 'decay_form'], axis=1, inplace=True)

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

parameter_titles = {'peak_time': 'Time to Peak', 'peak_height': 'Peak Amplitude',
                    'peak_width': 'Peak Width', 'decay_tau': 'Decay τ'}
parameter_labels = {'peak_time': 'Time to Peak (s)', 'peak_height': 'Peak Amplitude ({})',
                    'peak_width': 'Peak FWHM (s)', 'decay_tau': 'Decay τ (s)'}
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}

dms_color = '#53C43B'
dls_color = '#E87A2C'
pl_color = '#BB6ED8'

# %% Prepare peak metrics

# make subject ids categories
peak_metrics['subj_id'] = peak_metrics['subj_id'].astype('category')
peak_metrics['rew_hist_bin_label'] = peak_metrics['rew_hist_bin'].apply(lambda x: bin_labels[x])
peak_metrics['align_label'] = peak_metrics['align'].apply(lambda x: align_labels[x])

ignore_outliers = True
ignore_any_outliers = True
outlier_thresh = 10

t_min = 0.02
t_max = {a: {r: peak_find_props[a][r]['peak_tmax'] - t_min for r in regions} for a in alignments} 

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


# %% Plot average traces across all groups (Fig S9-E)

plot_regions = ['PL', 'DMS', 'DLS'] # 'DMS', 'PL'
plot_aligns = [Align.cue, Align.reward]
plot_signals = ['z_dff_iso_baseline_fband']

# plot formatting
plot_dec = {'DMS': 1, 'DLS': 1, 'PL': 2}
x_inc = {'DMS': 0.3, 'DLS': 0.3, 'PL': 3}
y_inc = {'DMS': 0.2, 'DLS': 0.8, 'PL': 0.2}

all_color = '#08AB36'
rew_color = '#BC141A'
unrew_color = '#1764AB'

separate_outcome = False

if separate_outcome:
    gen_groups = {Align.cue: {'rewarded': 'rew_hist_{}_rewarded', 'unrewarded': 'rew_hist_{}_unrewarded'},
                  Align.reward: {'rewarded': 'rew_hist_{}_rewarded', 'unrewarded': 'rew_hist_{}_unrewarded'}}
    group_labels = {'rewarded': 'Rewarded', 'unrewarded': 'Unrewarded'}

    region_colors = {'DMS': {'rewarded': rew_color, 'unrewarded': unrew_color},
                     'DLS': {'rewarded': rew_color, 'unrewarded': unrew_color},
                     'PL': {'rewarded': rew_color, 'unrewarded': unrew_color}}
else:
    gen_groups = {Align.cue: {'all': 'rew_hist_{}'},
                  Align.reward: {'all': 'rew_hist_{}_rewarded'}}
    group_labels = {'all': '_'}

    region_colors = {'DMS': {'all': dms_color}, 'DLS': {'all': dls_color}, 'PL': {'all': pl_color}}
    
groups = {a: {k: [v.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for k, v in gen_groups[a].items()} for a in plot_aligns}

cue_to_reward = np.nanmedian(sess_data['reward_time'] - sess_data['response_cue_time'])

plot_lims = {Align.cue: {'DMS': [-0.1,0.8], 'DLS': [-0.1,0.8], 'PL': [-1,3]},
             Align.reward: {'DMS': [-0.1,1.2], 'DLS': [-0.1,1.2], 'PL': [-1,12]}}

width_ratios = [np.diff(plot_lims[align]['DMS'])[0] for align in plot_aligns]
#width_ratios = [2,10.5]
#width_ratios = [0.7,0.9]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

plot_data = {}

for signal_type in plot_signals:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5.5, 3*n_rows+0.1), sharey='row', width_ratios=width_ratios)
    
    if n_rows == 1 and n_cols == 1:
        axs = np.array(axs)

    axs = axs.reshape((n_rows, n_cols))
    
    fig.suptitle(signal_title)
    
    plot_name = '{}_all_trials_outcome_{}_{}'.format('_'.join(plot_aligns), signal_type, '_'.join(plot_regions))

    plot_data[signal_type] = {}

    for i, region in enumerate(plot_regions):
        colors = region_colors[region]
        plot_data[signal_type][region] = {}
        
        for j, align in enumerate(plot_aligns):
            match align:
                case Align.cue:
                    title = 'Response Cue'
                    
                case Align.reward:
                    title = 'Reward Delivery'

            plot_data[signal_type][region][align] = {}

            ax = axs[i,j]
            
            region_signals = stacked_signals[signal_type][align][region]
            t_r = t[align][region]
            
            # normalize to pre-cue levels on a trial-by-trial basis
            cue_signals = stacked_signals[signal_type][Align.cue][region]
            t_cue = t[Align.cue][region]
            baseline_sel = (t_cue >= -0.1) & (t_cue < 0)
            
            for key, stack_groups in groups[align].items():
                stacked_signal = np.zeros((0,len(t_r)))
                
                for group in stack_groups:
                    baseline = np.nanmean(cue_signals[group][:,baseline_sel], axis=1)[:,None]
                    stacked_signal = np.vstack((stacked_signal, region_signals[group] - baseline))

                t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                error = calc_error(stacked_signal, True)
                
                plot_t = t_r[t_sel][::plot_dec[region]]
                plot_vals = np.nanmean(stacked_signal, axis=0)[t_sel][::plot_dec[region]]
                plot_err = error[t_sel][::plot_dec[region]]
                
                plot_utils.plot_psth(plot_t, plot_vals, plot_err, 
                                     ax, label=group_labels[key], color=colors[key], plot_x0=False)
                
                plot_data[signal_type][region][align][key] = pd.DataFrame.from_dict({'time': plot_t, 'avg': plot_vals, 'err': plot_err})
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
            #else:
                # ax.yaxis.set_tick_params(which='both', labelleft=True)
            if separate_outcome:
                ax.legend(loc='upper right')

            ax.set_xlabel(x_label)

            ax.xaxis.set_major_locator(MultipleLocator(x_inc[region]))
            ax.yaxis.set_major_locator(MultipleLocator(y_inc[region]))
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

# %% Plot average traces by reward history (Fig 5J)

plot_regions = ['PL', 'DMS', 'DLS'] # 
plot_aligns = [Align.reward] # Align.cue, 
plot_signal_types = ['z_dff_iso_baseline_fband']

# plot formatting
plot_dec = {'DMS': 1, 'DLS': 1, 'PL': 2}
x_inc = {'DMS': 0.3, 'DLS': 0.3, 'PL': 3}
y_inc = {'DMS': 0.3, 'DLS': 0.8, 'PL': 0.3}

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded'], Align.reward: ['rew_hist_{}_rewarded', 'rew_hist_{}_unrewarded']}
groups = {Align.cue: ['rew_hist_{}'.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins], 
          Align.reward: ['rew']+['rew_hist_{}_rewarded'.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins]+
                        ['unrew']+['rew_hist_{}_unrewarded'.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins]}

new_rew_labels = {'rewarded': '', 'unrewarded': ''}
new_group_labels_dict = {'rew_hist_{}_{}'.format(rew_hist_bin_strs[rew_bin], rk): '{} {}'.format(bin_labels[rew_bin], rv)
                for rew_bin in rew_hist_bins
                for rk, rv in new_rew_labels.items()}
new_group_labels_dict.update({'rew': 'Rew', 'unrew': 'Unrew'})

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((np.zeros((1,4)), rew_hist_rew_colors, np.zeros((1,4)), rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,0.5], 'DLS': [-0.1,0.5], 'PL': [-1,1]},
             Align.reward: {'DMS': [-0.2,1.2], 'DLS': [-0.2,1.2], 'PL': [-2,12]}}

#width_ratios = [1,2]
#width_ratios = [0.7,0.9]
if len(plot_aligns) == 1:
    width_ratios = [1]
else:
    width_ratios = [np.diff(plot_lims[Align.cue][plot_regions[0]])[0], np.diff(plot_lims[Align.reward][plot_regions[0]])[0]]

n_rows = len(plot_regions)
n_cols = len(plot_aligns)
t = aligned_signals['t']
x_label = 'Time (s)'

plot_data = {}

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    plot_data[signal_type] = {}
    
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 3*n_rows+0.1), sharey='row', width_ratios=width_ratios)
    
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
            
            region_signals = stacked_signals[signal_type][align][region]
            
            t_r = t[align][region]
            t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
            t_r = t_r[t_sel][::plot_dec[region]]
            
            plot_data[signal_type][region][align]['time'] = t_r
    
            for group, color in zip(groups[align], colors[align]):
                if group in region_signals:
                    act = region_signals[group]
                    error = calc_error(act, True)
                    
                    avg_act = np.nanmean(act, axis=0)[t_sel][::plot_dec[region]]
                    avg_err = error[t_sel][::plot_dec[region]]
                    
                    plot_data[signal_type][region][align][group_labels_dict[group]+'_avg'] = avg_act
                    plot_data[signal_type][region][align][group_labels_dict[group]+'_err'] = avg_err
                    
                    plot_utils.plot_psth(t_r, avg_act, avg_err, ax, label=new_group_labels_dict[group], color=color, plot_x0=False)
                else:
                    ax.plot(0, 0, linestyle='none', marker='None', color=color, label=new_group_labels_dict[group])
    
            ax.set_title('{} {}'.format(region, title))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
            #else:
                # ax.yaxis.set_tick_params(which='both', labelleft=True)
            
            ax.legend()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()

            ax.set_xlabel(x_label)
            
            ax.xaxis.set_major_locator(MultipleLocator(x_inc[region]))
            ax.yaxis.set_major_locator(MultipleLocator(y_inc[region]))
            
            plot_data[signal_type][region][align] = pd.DataFrame.from_dict(plot_data[signal_type][region][align])
            
        fig.legend(handles, labels, loc='outside right center', ncols=legend_cols, title='# Rewards in last {} trials'.format(rew_rate_n_back))
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
        
# %% Plot average traces for the groups by side
plot_regions = ['DMS', 'DLS'] #, 'PL'
plot_aligns = [Align.cue, Align.reward]
plot_signal_types = ['z_dff_iso_baseline_fband']

#gen_groups = {Align.cue: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}
gen_groups = {Align.cue: ['rew_hist_{}_{}'], Align.reward: ['rew_hist_{}_rewarded_{}', 'rew_hist_{}_unrewarded_{}']}

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

#colors = {Align.cue: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}
colors = {Align.cue: rew_hist_all_colors, Align.reward: np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))}

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'DLS': [-0.1,0.6], 'PL': [-1,8]},
             Align.reward: {'DMS': [-0.1,1], 'DLS': [-0.1,1], 'PL': [-1,12]}}

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
        
        plot_data[signal_type][region] = {}

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
                
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
            
# %% Plot average traces for stays/switches by side and prior outcome

plot_regions = ['DMS', 'DLS', 'PL'] #
plot_aligns = [Align.cue, Align.reward]
plot_signal_types = ['z_dff_iso_baseline_fband']

gen_groups = {Align.cue: ['{}_prev_{}_contra', '{}_prev_{}_ipsi'], 
              Align.reward: ['{}_rewarded_prev_{}_contra', '{}_rewarded_prev_{}_ipsi', '{}_unrewarded_prev_{}_contra', '{}_unrewarded_prev_{}_ipsi']}

plot_groups = {a: [[group.format(s, r) for s in ['stay', 'switch'] for r in ['reward', 'unreward']] for group in gen_groups[a]] for a in plot_aligns}

group_options_dict = {'contra': {'stay': 'Contra', 'switch': 'Ipsi'}, 'ipsi': {'stay': 'Ipsi', 'switch': 'Contra'},
                     'reward': 'Rewarded', 'unreward': 'Unrewarded'}
group_labels = [{'{}_prev_{}_contra'.format(s,r): 'Prev {} {}'.format(group_options_dict[r], group_options_dict['contra'][s]),
                 '{}_prev_{}_ipsi'.format(s,r): 'Prev {} {}'.format(group_options_dict[r], group_options_dict['ipsi'][s]),
                 '{}_rewarded_prev_{}_contra'.format(s,r): 'Prev {} {}'.format(group_options_dict[r], group_options_dict['contra'][s]),
                 '{}_rewarded_prev_{}_ipsi'.format(s,r): 'Prev {} {}'.format(group_options_dict[r], group_options_dict['ipsi'][s]),
                 '{}_unrewarded_prev_{}_contra'.format(s,r): 'Prev {} {}'.format(group_options_dict[r], group_options_dict['contra'][s]),
                 '{}_unrewarded_prev_{}_ipsi'.format(s,r): 'Prev {} {}'.format(group_options_dict[r], group_options_dict['ipsi'][s])}
                for s in ['stay', 'switch'] for r in ['reward', 'unreward']]
group_labels = {k: v for d in group_labels for k, v in d.items()}

plot_titles = {Align.cue: ['Contra Choices', 'Ipsi Choices'],
               Align.reward: ['Rewarded Contra Choices', 'Rewarded Ipsi Choices', 'Unrewarded Contra Choices', 'Unrewarded Ipsi Choices']}
fig_titles = {Align.cue: 'Cue Responses By Side and Previous Outcome',
              Align.reward: 'Reward Responses By Side and Previous Outcome'}

plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'DLS': [-0.1,0.6], 'PL': [-1,6]},
             Align.reward: {'DMS': [-0.1,1.2], 'DLS': [-0.1,1.2], 'PL': [-1,12]}}

n_rows = len(plot_regions)
t = aligned_signals['t']
x_label = 'Time (s)'

# plot each alignment separately

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    for align in plot_aligns:
        
        groups = plot_groups[align]
        
        n_cols = len(groups)
        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), sharey='row')
        axs = np.array(axs).reshape((n_rows, n_cols))
        
        fig.suptitle(fig_titles[align])
        plot_name = 'stay_switch_side_prev_outcome_{}'.format(align)
    
        for i, region in enumerate(plot_regions):
            t_r = t[align][region]
            
            for j, group in enumerate(groups):
                ax = axs[i,j]
                
                for signal_name in group:
                    act = stacked_signals[signal_type][align][region][signal_name]
                    
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    error = calc_error(act, True)
                    
                    plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[signal_name], plot_x0=False)
        
                ax.set_title('{} {}'.format(region, plot_titles[align][j]))
                plot_utils.plot_dashlines(0, ax=ax)
        
                if j == 0:
                    ax.set_ylabel(y_label)
                else:
                    plot_utils.show_axis_labels(ax)
                
                ax.set_xlabel(x_label)
                ax.legend()
                
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
            


# %% Plot average traces for future stays/switches by side and reward history

# plot over all reward histories

plot_regions = ['DMS', 'DLS', 'PL'] #
align = Align.reward
plot_signal_types = ['z_dff_iso_baseline_fband']

gen_groups = ['{}_next_{}', '{}_next_{}_contra', '{}_next_{}_ipsi']

plot_groups = [[group.format(r,s) for r in ['rewarded', 'unrewarded'] for s in ['stay', 'switch']] for group in gen_groups]

group_options_dict = {'stay': 'Stay', 'switch': 'Switch', 'rewarded': 'Rew', 'unrewarded': 'Unrew'}
group_labels = [{'{}_next_{}'.format(r,s): '{} Next {}'.format(group_options_dict[r], group_options_dict[s]),
                 '{}_next_{}_contra'.format(r,s): '{} Next {}'.format(group_options_dict[r], group_options_dict[s]),
                 '{}_next_{}_ipsi'.format(r,s): '{} Next {}'.format(group_options_dict[r], group_options_dict[s])}
                for s in ['stay', 'switch'] for r in ['rewarded', 'unrewarded']]
group_labels = {k: v for d in group_labels for k, v in d.items()}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
fig_title = 'Reward Responses By Outcome, Choice Side, and Next Choice'

plot_lims = {#Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,6]},
             Align.reward: {'DMS': [-0.1,1], 'DLS': [-0.1,1], 'PL': [-1,10]}}

n_rows = len(plot_regions)
t = aligned_signals['t']
x_label = 'Time (s)'

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)

    n_cols = len(plot_groups)
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), sharey='row')
    axs = np.array(axs).reshape((n_rows, n_cols))
    
    fig.suptitle(fig_title)
    plot_name = 'future_stay_switch_side_outcome_{}'.format(align)

    for i, region in enumerate(plot_regions):
        t_r = t[align][region]
        
        for j, group in enumerate(plot_groups):
            ax = axs[i,j]
            
            for signal_name in group:
                act = stacked_signals[signal_type][align][region][signal_name]
                
                t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                error = calc_error(act, True)
                
                plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[signal_name], plot_x0=False)
    
            ax.set_title('{} {}'.format(region, plot_titles[j]))
            plot_utils.plot_dashlines(0, ax=ax)
    
            if j == 0:
                ax.set_ylabel(y_label)
            else:
                plot_utils.show_axis_labels(ax)
            
            ax.set_xlabel(x_label)
            ax.legend()
            
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
        
# %% plot separately for each reward history 

gen_groups = ['rew_hist_{}_{}_next_{}', 'rew_hist_{}_{}_next_{}_contra', 'rew_hist_{}_{}_next_{}_ipsi']
plot_groups = [[[group.format(rew_hist_bin_strs[rew_bin],r,s) for r in ['rewarded', 'unrewarded'] for s in ['stay', 'switch']] for rew_bin in rew_hist_bins] for group in gen_groups]

group_labels = [{'rew_hist_{}_{}_next_{}'.format(rew_hist_bin_strs[rew_bin],r,s): '{} Next {}'.format(group_options_dict[r], group_options_dict[s]),
                 'rew_hist_{}_{}_next_{}_contra'.format(rew_hist_bin_strs[rew_bin],r,s): '{} Next {}'.format(group_options_dict[r], group_options_dict[s]),
                 'rew_hist_{}_{}_next_{}_ipsi'.format(rew_hist_bin_strs[rew_bin],r,s): '{} Next {}'.format(group_options_dict[r], group_options_dict[s])}
                for s in ['stay', 'switch'] for r in ['rewarded', 'unrewarded'] for rew_bin in rew_hist_bins]
group_labels = {k: v for d in group_labels for k, v in d.items()}

plot_titles = ['{} Rewards'.format(bin_labels[rew_bin]) for rew_bin in rew_hist_bins]
fig_titles = ['Reward Responses By Outcome, Reward History, and Next Choice', 
              'Contra Choice Reward Responses By Outcome, Reward History, and Next Choice',
              'Ipsi Choice Reward Responses By Outcome, Reward History, and Next Choice']
fig_ids = ['all', 'contra', 'ipsi']

for signal_type in plot_signal_types:
    signal_title, y_label = fpah.get_signal_type_labels(signal_type)
    
    for sub_group, title, fig_id in zip(plot_groups, fig_titles, fig_ids):

        n_cols = len(sub_group)
        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows), sharey='row')
        axs = np.array(axs).reshape((n_rows, n_cols))
        
        fig.suptitle(title)
        plot_name = 'future_stay_switch_side_outcome_{}_{}'.format(align, fig_id)
    
        for i, region in enumerate(plot_regions):
            t_r = t[align][region]
            
            for j, group in enumerate(sub_group):
                ax = axs[i,j]
                
                for signal_name in group:
                    act = stacked_signals[signal_type][align][region][signal_name]
                    
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    error = calc_error(act, True)
                    
                    plot_utils.plot_psth(t_r[t_sel], np.nanmean(act, axis=0)[t_sel], error[t_sel], ax, label=group_labels[signal_name], plot_x0=False)
        
                ax.set_title('{} {}'.format(region, plot_titles[j]))
                plot_utils.plot_dashlines(0, ax=ax)
        
                if j == 0:
                    ax.set_ylabel(y_label)
                else:
                    plot_utils.show_axis_labels(ax)
                
                ax.set_xlabel(x_label)
                ax.legend()
                
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

            

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

# %% make peak property comparison figures comparing regions at cue and reward (Fig S9-F)

parameters = ['peak_height', 'peak_width'] # 'peak_time', 'decay_tau']

regions = ['PL', 'DMS', 'DLS']
region_colors = [pl_color, dms_color, dls_color]
plot_aligns = ['Response Cue', 'Reward Delivery']
subj_ids = np.unique(filt_peak_metrics['subj_id'])
plot_signals = ['dff_iso_baseline_fband', 'z_dff_iso_baseline_fband']

# have same jitter for each subject
noise = 0.075
n_neg = int(np.floor(len(subj_ids)/2))
n_pos = int(np.ceil(len(subj_ids)/2))
jitters = np.concatenate([np.random.uniform(-1, -0.1, n_neg), np.random.uniform(0.1, 1, n_pos)]) * noise

plot_data = {}

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    peak_sel = filt_peak_metrics['signal_type'] == signal_type
    # only show rewarded peaks for reward delivery
    peak_sel = peak_sel & ((filt_peak_metrics['align'] == Align.cue) | ((filt_peak_metrics['align'] == Align.reward) & filt_peak_metrics['rewarded']))
    
    sub_peak_metrics = filt_peak_metrics[peak_sel]
    
    plot_data[signal_type] = {'all': sub_peak_metrics[['subj_id', 'region', 'align_label']+parameters]}

    # Compare responses across regions grouped by alignment
    fig, axs = plt.subplots(len(parameters), 1, figsize=(4, 4*len(parameters)), layout='constrained')
    axs = np.array(axs).reshape((len(parameters)))
    
    for i, param in enumerate(parameters):

        ax = axs[i]
        ax.set_title(parameter_titles[param])

        subj_avgs = sub_peak_metrics.groupby(['subj_id', 'region', 'align_label']).agg({param: np.nanmean}).reset_index()
    
        plot_data[signal_type][param] = subj_avgs
    
        # plot averages in boxplots
        sb.boxplot(data=sub_peak_metrics, x='align_label', y=param, hue='region', palette=region_colors,
                   order=plot_aligns, hue_order=regions, ax=ax, showfliers=False, whis=(5,95), log_scale=True)
        if param == 'peak_height':
            ax.set_ylabel(parameter_labels[param].format(y_label))
        else:
            ax.set_ylabel(parameter_labels[param])
        ax.set_xlabel('')
    
        # add subject averages for each alignment with lines connecting them
        if len(regions) == 2:
            dodge = 0.2
            x = np.array([-dodge, dodge])
        elif len(regions) == 3:
            dodge = 0.25
            x = np.array([-dodge, 0, dodge])
        

        for j, align in enumerate(plot_aligns):
            plot_x = x + j
    
            for subj_id, jitter in zip(subj_ids, jitters):
                subj_avg = subj_avgs[(subj_avgs['subj_id'] == subj_id) & (subj_avgs['align_label'] == align)]
    
                y = np.array([subj_avg.loc[subj_avg['region'] == r, param].iloc[0] for r in regions])
                nan_sel = np.isnan(y)
    
                ax.plot((plot_x+jitter)[~nan_sel], y[~nan_sel], color='black', marker='o', linestyle='solid', alpha=0.75, linewidth=1, markersize=5)
                
        # set y max to be a multiple of 10
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(y_min, utils.convert_to_multiple(y_max, 10, direction='up'))
                
    plot_name = 'cue_reward_peak_comp_{}_{}'.format('_'.join(parameters), signal_type)
    fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

# %% make reward history comparison figures per peak property (Fig 5K)

parameters = ['peak_height'] # 'peak_time', 'peak_height', 'peak_width', 'decay_tau'

rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_palette = sb.color_palette(rew_hist_rew_colors) 

plot_signals = ['dff_iso_baseline_fband']
plot_aligns = ['reward'] #'cue', 
plot_regions = ['PL', 'DMS', 'DLS']

split_by_side = False
sides = ['contra', 'ipsi']
n_regions = len(regions)

n_aligns = len(plot_aligns)
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}
hatch_order = ['//\\\\', '']
line_order = ['dashed', 'solid']

plot_data = {}

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    plot_data[signal_type] = {}
    
    for param in parameters:
        
        plot_data[signal_type][param] = {}
        
        plot_name = '{}_reward_hist_{}_{}_back_{}'.format('_'.join(plot_aligns), param, rew_rate_n_back, signal_type)
    
        # Compare responses across alignments grouped by region
        fig, axs = plt.subplots(n_regions, n_aligns, figsize=(4*n_aligns, 4*n_regions), layout='constrained', sharey='row')
        fig.suptitle('{}, {}'.format(parameter_titles[param], signal_label))
        
        axs = np.array(axs).reshape((n_regions, n_aligns))
    
        for i, region in enumerate(plot_regions):
            
            plot_data[signal_type][param][region] = {}
            
            for j, align in enumerate(plot_aligns):
                peak_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['signal_type'] == signal_type) & (filt_peak_metrics['region'] == region)
    
                match align:
                    #case Align.cue:

                    case Align.reward:
                        peak_sel = peak_sel & filt_peak_metrics['rewarded']

                region_metrics = filt_peak_metrics[peak_sel]
                plot_data[signal_type][param][region][align] = {'raw': region_metrics}
                
                ax = axs[i,j]
                ax.set_title('{} {}'.format(region, align_labels[align]))
                
                if split_by_side:
                    # plot reward history group averages in boxplots
                    sb.boxplot(data=region_metrics, x='rew_hist_bin_label', y=param, hue='side', hue_order=sides,
                               whis=(5,95), ax=ax, showfliers=False, legend=False)
                    
                    # update colors and fills of boxes
                    for k, patch in enumerate(ax.patches):
                        # Left boxes first, then right boxes
                        if k < len(rew_hist_bins):
                            # add hatch to cues
                            patch.set_hatch(hatch_order[int(k/len(rew_hist_bins))])

                        patch.set_facecolor(rew_hist_palette[k % len(rew_hist_bins)])

                    # add subject averages for each alignment with lines connecting them
                    subj_avgs = region_metrics.groupby(['subj_id', 'rew_hist_bin_label', 'side']).agg({param: np.nanmean}).reset_index()
                    plot_data[signal_type][param][region][align]['avg'] = subj_avgs
                    
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
                               hue='rew_hist_bin_label', palette=rew_hist_palette, whis=(5,95),
                               ax=ax, showfliers=False, legend=False, saturation=0.7)
    
                    # add subject averages for each alignment with lines connecting them
                    subj_avgs = region_metrics.groupby(['subj_id', 'rew_hist_bin_label']).agg({param: np.nanmean}).reset_index()
                    plot_data[signal_type][param][region][align]['avg'] = subj_avgs
            
                    group_labels = np.unique(region_metrics['rew_hist_bin_label'])
                    region_subj_ids = np.unique(region_metrics['subj_id'])
                    for subj_id in region_subj_ids:
                        subj_avg = subj_avgs[subj_avgs['subj_id'] == subj_id]
                        y = [subj_avg.loc[subj_avg['rew_hist_bin_label'] == g, param] for g in group_labels]
        
                        ax.plot(np.arange(len(group_labels)), y, color='black', marker='o', linestyle='solid', alpha=0.7, linewidth=1, markersize=5)
                        
                if param == 'peak_height':
                    ax.set_ylabel(parameter_labels[param].format(y_label))
                else:
                    ax.set_ylabel(parameter_labels[param])
                ax.set_xlabel('# Rewards in last {} Trials'.format(rew_rate_n_back))
                
        fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')

# %% Get mean and SE for peak properties
# by region/align
signal_type = 'dff_iso'
parameters = ['peak_height', 'peak_width'] # 'peak_time', 'decay_tau']
plot_regions = ['DMS', 'PL']
plot_aligns = ['cue', 'reward'] #

for param in parameters:
    for region in plot_regions:
        for align in plot_aligns:
            peak_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['signal_type'] == signal_type) & (filt_peak_metrics['region'] == region)
    
            if align == Align.reward:
                    peak_sel = peak_sel & filt_peak_metrics['rewarded']
    
            region_metrics = filt_peak_metrics[peak_sel]
            
            print('{} {}, {}-aligned {}: {:.3f} +/- {:.3f}\n'.format(
                   signal_type, region, align, param, np.nanmean(region_metrics[param]), utils.stderr(region_metrics[param])))
            

# %% Compare peak properties based on reward time

parameters = ['peak_height'] # 'peak_height', 'peak_time', 'peak_width', 'decay_tau'

rew_hist_rew_colors = plt.cm.seismic(np.linspace(0.6,1,len(rew_hist_bins)))
rew_hist_palette = sb.color_palette(rew_hist_rew_colors) 

plot_signals = ['z_dff_iso']
plot_aligns = ['cue', 'reward'] #'cue', 
plot_regions = ['DMS', 'PL']

split_by_side = True

# define reward time bin ranges
thresh = 10 #np.nanmedian(sess_data['reward_time'])
reward_time_bins = [[0, thresh], [thresh, np.inf]]
reward_time_bin_labels = ['<{}s'.format(thresh), '>{}s'.format(thresh)]

n_aligns = len(plot_aligns)
align_labels = {'cue': 'Response Cue', 'reward': 'Reward Delivery'}
hatch_order = ['//\\\\', '']
line_order = ['dashed', 'solid']

n_rows = len(reward_time_bins)
n_cols = len(plot_aligns)

group_labels = np.unique(filt_peak_metrics['rew_hist_bin_label'])

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    for param in parameters:
        
        #plot_name = '{}_reward_hist_{}_{}_back_{}'.format('_'.join(plot_aligns), param, rew_rate_n_back, signal_type)
    
        # Compare responses across alignments by reward time, region by region
        for region in plot_regions:
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(4*n_rows, 4*n_cols), layout='constrained', sharey='row')
            fig.suptitle('{} - {}, {}'.format(region, parameter_titles[param], signal_label))
            
            axs = np.array(axs).reshape((n_rows, n_cols))
        
            for i, (rew_range, range_label) in enumerate(zip(reward_time_bins, reward_time_bin_labels)):
                for j, align in enumerate(plot_aligns):
                    peak_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['signal_type'] == signal_type) & (filt_peak_metrics['region'] == region)
                    peak_sel = peak_sel & (filt_peak_metrics['reward_time'] > rew_range[0]) & (filt_peak_metrics['reward_time'] <= rew_range[1])
        
                    match align:
                        #case Align.cue:
    
                        case Align.reward:
                            peak_sel = peak_sel & filt_peak_metrics['rewarded']
    
                    region_metrics = filt_peak_metrics[peak_sel]
                    
                    ax = axs[i,j]
                    ax.set_title('{}, reward time {}'.format(align_labels[align], range_label))
                    
                    if split_by_side:
                        # plot reward history group averages in boxplots
                        sb.boxplot(data=region_metrics, x='rew_hist_bin_label', y=param, order=group_labels,
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
                                   order=group_labels, hue='rew_hist_bin_label', hue_order=group_labels,
                                   palette=rew_hist_palette, ax=ax, showfliers=False, legend=False)
        
                        # add subject averages for each alignment with lines connecting them
                        subj_avgs = region_metrics.groupby(['subj_id', 'rew_hist_bin_label']).agg({param: np.nanmean}).reset_index()
                
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
                
        #fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')


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

lim_n_back_fits = [6] #np.arange(0,11) # rew_n_back
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
    
                
# %% declare common methods for performing regression over time

# define method to perform the regression
def regress_over_time(signals, predictors):
    params = []
    ci_lower = []
    ci_upper = []
    se = []
    p_vals = []
    rmse = []
    for i in range(signals.shape[1]):
        t_sig = signals[:,i]
        # remove trials with nans at this time step
        rem_nan_sel = ~np.isnan(t_sig)
        
        if np.sum(rem_nan_sel) == 0:
            nan_vals = {col: np.nan for col in predictors.columns}
            params.append(nan_vals)
            ci_lower.append(nan_vals)
            ci_upper.append(nan_vals)
            se.append(nan_vals)
            p_vals.append(nan_vals)
            rmse.append(np.nan)
        else:
            lm = sm.OLS(t_sig[rem_nan_sel], predictors[rem_nan_sel])
            lm = lm.fit()
            #lm = lm.fit_regularized(alpha=0.1, L1_wt=0.5)
            params.append(lm.params.to_dict())
            cis = lm.conf_int(0.05)
            ci_lower.append(cis[0].to_dict())
            ci_upper.append(cis[1].to_dict())
            se.append(lm.bse.to_dict())
            p_vals.append(lm.pvalues.to_dict())
            rmse.append(np.sqrt(np.mean(lm.resid**2)))
        
    return {'params': pd.DataFrame(params), 
            'ci_lower': pd.DataFrame(ci_lower), 
            'ci_upper': pd.DataFrame(ci_upper),
            'se': pd.DataFrame(se),
            'p_vals': pd.DataFrame(p_vals),
            'rmse': np.array(rmse)}

# define common plotting routine
def plot_regress_over_time(params, t, plot_cols, ax, ci_lower=None, ci_upper=None, error=None, sig=None, p_vals=None, group_labels=None,
                           t_sel=None, colors=None, plot_y0=True, plot_x0=True, dec=1, x_inc=None, y_inc=None, filt_outliers=False):
    if len(plot_cols) == 0:
        return
    
    if group_labels is None:
        group_labels = {}
    
    plot_data = {}
    
    sig_y_dist = 0.03
    if t_sel is None:
        t_sel = np.full_like(t, True)
    else:
        t_sel = t_sel.copy()
        
    plot_t = t[t_sel][::dec]
    plot_data['time'] = plot_t
    
    line_colors = []
    for i, col in enumerate(plot_cols):
        vals = params[col].to_numpy()
        plot_vals = vals[t_sel][::dec]
        
        if filt_outliers:
            t_sel = t_sel & (np.abs(vals) < 1e3)
            
        if colors is None:
            color = None
        else:
            color = colors[i]
            
        col_label = group_labels.get(col, col)

        if not ci_lower is None and not ci_upper is None:
            error = np.abs(np.vstack((ci_lower[col], ci_upper[col])) - vals[None,:])
            plot_err = error[:,t_sel][::dec]
            line, _ = plot_utils.plot_psth(plot_t, plot_vals, plot_err, ax=ax, label=col_label, plot_x0=False, color=color)
        elif not error is None:
            plot_err = error[col][t_sel][::dec]
            line, _ = plot_utils.plot_psth(plot_t, plot_vals, plot_err, ax=ax, label=col_label, plot_x0=False, color=color)
        else:
            plot_err = None
            line, _ = plot_utils.plot_psth(plot_t, plot_vals, ax=ax, label=col_label, plot_x0=False, color=color)
            
        plot_data[col_label+'_coeffs'] = plot_vals
        plot_data[col_label+'_err'] = plot_err
        
        line_colors.append(line.get_color())
            
    if plot_x0:
        plot_utils.plot_dashlines(0, dir='v', ax=ax)
    if plot_y0:
        plot_utils.plot_dashlines(0, dir='h', ax=ax)
    
    # plot significance from 0    
    if plot_sig and not sig is None:
        y_min, y_max = ax.get_ylim()
        y_offset = (y_max-y_min)*sig_y_dist

        for i, col in enumerate(plot_cols):
            # # perform correction
            # reject, corrected_pvals, _, _  = smm.multipletests(p_vals[col][t_sel], alpha=0.05, method='fdr_bh')
            sig_t = t[sig[col] & t_sel]
            ax.scatter(sig_t, np.full_like(sig_t, y_max+i*y_offset), color=line_colors[i], marker='.', s=10)
            
            col_label = group_labels.get(col, col)
            plot_data[col_label+'_sig'] = sig[col][t_sel][::dec]
            if not p_vals is None:
                plot_data[col_label+'_p'] = p_vals[col][t_sel][::dec]

    ax.set_xlabel('Time (s)')
    ax.legend(loc='best')
    
    if not x_inc is None:
        ax.xaxis.set_major_locator(MultipleLocator(x_inc))
    if not y_inc is None:
        ax.yaxis.set_major_locator(MultipleLocator(y_inc))
        
    return plot_data
    
    
# %% Prep for reward history regression over time

regions = ['PL', 'DMS', 'DLS'] # 'DMS', 'PL'
aligns = [Align.cue, Align.reward] # Align.cue, Align.reward
signals = ['z_dff_iso_baseline', 'z_dff_iso_baseline_fband']
included_subjs = np.array(subj_ids)[~np.isin(subj_ids, ignored_subjects)]

n_back = 3
normalize = True
exclude_n_back = True # whether to exclude trials less than the n_back history

#bah.calc_rew_rate_hist(sess_data, n_back=n_back, kernel='uniform')

# first build predictor and response matrices
t = aligned_signals['t']
    
subj_stacked_signals = {subj_id: {s: {a: {r: np.zeros((0, len(t[a][r]))) for r in regions} 
                                      for a in aligns} 
                                  for s in signals}
                        for subj_id in included_subjs}

subj_predictors = {subj_id: {r: [] for r in regions} for subj_id in included_subjs}

for subj_id in included_subjs:
    for sess_id in sess_ids[subj_id]:
        
        # build predictor matrix by trial
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        responded = ~np.isnan(trial_data['response_time']).to_numpy()
        rewarded = trial_data['rewarded'].to_numpy()[responded].astype(int)
        unrewarded = (~trial_data['rewarded'].to_numpy())[responded].astype(int)
        choice = trial_data['choice'].to_numpy()[responded]
        left_choice = (choice == 'left').astype(int)
        right_choice = (choice == 'right').astype(int)
        switches = np.concatenate(([False], choice[:-1] != choice[1:])).astype(int)
        stays = np.concatenate(([False], choice[:-1] == choice[1:])).astype(int)
        next_stays = np.concatenate((choice[:-1] == choice[1:], [False])).astype(int)
        rew_hist = pd.cut(trial_data['rew_rate_hist_all'], rew_hist_bins)
        rew_time = trial_data['reward_time'].to_numpy()[responded]

        # make buffered predictors to be able to go n back
        # Note: need to add 0 at the end for generic indexing logic (i:-n_back+i-1) - have to index to at least -1, can't do 0
        if not exclude_n_back:
            buff_reward = np.concatenate((np.full((n_back), 0), rewarded, [0]))
            buff_unreward = np.concatenate((np.full((n_back), 0), unrewarded, [0]))
            buff_left_choice = np.concatenate((np.full((n_back), 0), left_choice, [0]))
            buff_right_choice = np.concatenate((np.full((n_back), 0), right_choice, [0]))
        else:
            buff_reward = np.concatenate((rewarded, [0]))
            buff_unreward = np.concatenate((unrewarded, [0]))
            buff_left_choice = np.concatenate((left_choice, [0]))
            buff_right_choice = np.concatenate((right_choice, [0]))
            choice = choice[n_back:]
            switches = switches[n_back:]
            stays = stays[n_back:]
            next_stays = next_stays[n_back:]
            rew_time = rew_time[n_back:]
            
        for region in regions:
            if sess_id in ignored_signals[region] or not region in implant_info[subj_id]:
                continue
            
            # build predictors by region
            region_side = implant_info[subj_id][region]['side']
            choice_side = [fpah.get_implant_rel_side(x, region_side) for x in choice]
            
            preds = {'reward ({})'.format(i-n_back): buff_reward[i:-n_back+i-1] for i in range(n_back, -1, -1)}
            preds.update({'unreward ({})'.format(i-n_back): buff_unreward[i:-n_back+i-1] for i in range(n_back, -1, -1)})
            if region_side == 'left':
                preds.update({'ipsi choice ({})'.format(i-n_back): buff_left_choice[i:-n_back+i-1] for i in range(n_back-1, -1, -1)})
                preds.update({'contra choice ({})'.format(i-n_back): buff_right_choice[i:-n_back+i-1] for i in range(n_back-1, -1, -1)})
            else:
                preds.update({'ipsi choice ({})'.format(i-n_back): buff_right_choice[i:-n_back+i-1] for i in range(n_back-1, -1, -1)})
                preds.update({'contra choice ({})'.format(i-n_back): buff_left_choice[i:-n_back+i-1] for i in range(n_back-1, -1, -1)})
            preds.update({'choice': choice_side})
            preds.update({'switch': switches})
            preds.update({'stay': switches})
            preds.update({'next_stay': switches})
            preds.update({'reward_time': rew_time})
                
            subj_predictors[subj_id][region].append(pd.DataFrame(preds))

            for signal_type in signals:
                if not signal_type in aligned_signals[subj_id][sess_id]:
                    continue
                for align in aligns:
                    if not align in aligned_signals[subj_id][sess_id][signal_type]:
                        continue
            
                    t_r = t[align][region]
                    mat = aligned_signals[subj_id][sess_id][signal_type][align][region]
                    
                    # normalize all grouped matrices to the average pre-event signal of the lowest reward rate
                    if normalize:
                        baseline_mat = mat[(rew_hist == rew_hist_bins[0]) & responded,:]
                        #baseline_mat = mat[responded,:]
                        if baseline_mat.shape[0] > 0:
                            baseline_sel = (t_r >= -0.1) & (t_r < 0)
                            baseline = np.nanmean(baseline_mat[:,baseline_sel])
                        else:
                            baseline = 0
                            
                        mat = mat - baseline
                        
                    mat = mat[responded,:]
                    if exclude_n_back:
                        mat = mat[n_back:,:]
                    
                    subj_stacked_signals[subj_id][signal_type][align][region] = np.vstack((subj_stacked_signals[subj_id][signal_type][align][region], mat))
    
    for region in regions:
        if not region in implant_info[subj_id]:
            continue
        subj_predictors[subj_id][region] = pd.concat(subj_predictors[subj_id][region])
    
# for analyzing meta subject, create new subject entry with everything stacked
subj_predictors['all'] = {r: [] for r in regions}
subj_stacked_signals['all'] = {s: {a: {r: [] for r in regions} 
                                   for a in aligns} 
                               for s in signals}
for region in regions:
    subj_predictors['all'][region] = pd.concat([subj_predictors[subj_id][region] for subj_id in included_subjs if region in implant_info[subj_id]])
    
    for signal_type in signals:
        for align in aligns:
            subj_stacked_signals['all'][signal_type][align][region] = np.vstack([subj_stacked_signals[subj_id][signal_type][align][region] for subj_id in included_subjs])



# %% perform regression with various options

limit_rewarded_trials = False
include_current_side = False
include_stay_switch = False
include_stay_side_interaction = False
include_current_reward = False # only relevant if not including outcome interaction, mostly for cue-related alignments
include_side_reward_interaction = False
include_outcome_reward_interaction = True # Separate regressions based on outcome
fit_ind_subj = False
fit_meta_subj = True

analyzed_subjs = []
if fit_ind_subj:
    analyzed_subjs.extend(included_subjs.tolist())

if fit_meta_subj:
    analyzed_subjs.append('all')
    
reg_params = {subj_id: {s: {r: {a: {} for a in aligns} 
                            for r in regions} 
                        for s in signals}
              for subj_id in analyzed_subjs}

for subj_id in analyzed_subjs:
    for region in regions:
        # build the predictor matrix based on the current options
        region_preds = subj_predictors[subj_id][region]
        pred_mat = {}

        contra_choice = region_preds['choice'] == 'contra'
        ipsi_choice = region_preds['choice'] == 'ipsi'
        full_rewarded = region_preds['reward (0)'].astype(bool)
        rewarded = full_rewarded
        unrewarded = region_preds['unreward (0)'].astype(bool)
        
        if limit_rewarded_trials:
            region_preds = region_preds[rewarded]
            contra_choice = contra_choice[rewarded]
            rewarded = rewarded[rewarded]

        # determine how to model the intercept and current reward
        if include_outcome_reward_interaction and include_side_reward_interaction:
            pred_mat['contra, rewarded'] = (contra_choice & rewarded).astype(int)
            pred_mat['ipsi, rewarded'] = (ipsi_choice & rewarded).astype(int)
            if not limit_rewarded_trials:
                pred_mat['contra, unrewarded'] = (contra_choice & unrewarded).astype(int)
                pred_mat['ipsi, unrewarded'] = (ipsi_choice & unrewarded).astype(int) 
        elif include_current_side or include_side_reward_interaction:
            pred_mat['contra choice'] = contra_choice.astype(int)
            pred_mat['ipsi choice'] = ipsi_choice.astype(int)
            if include_current_reward:
                if include_side_reward_interaction:
                    pred_mat['contra, rewarded'] = (contra_choice & rewarded).astype(int)
                    pred_mat['ipsi, rewarded'] = (ipsi_choice & rewarded).astype(int)
                else:
                    pred_mat['rewarded'] = rewarded.astype(int)
        elif include_outcome_reward_interaction:
            pred_mat['rewarded'] = rewarded.astype(int)
            if not limit_rewarded_trials:
                pred_mat['unrewarded'] = unrewarded.astype(int)
        else:
            pred_mat['intercept'] = 1
            if include_current_reward and not limit_rewarded_trials:
                pred_mat['rewarded'] = rewarded.astype(int)

        # add in reward history, don't go all the way to the current trial
        for i in range(n_back-1, -1, -1):
            rew_str = 'reward ({})'.format(i-n_back)
            rew_preds = region_preds[rew_str]
            
            if include_outcome_reward_interaction and include_side_reward_interaction:
                pred_mat['contra, rewarded, '+rew_str] = rew_preds * (contra_choice & rewarded)
                pred_mat['ipsi, rewarded, '+rew_str] = rew_preds * (ipsi_choice & rewarded)
                if not limit_rewarded_trials:
                    pred_mat['contra, unrewarded, '+rew_str] = rew_preds * (contra_choice & unrewarded)
                    pred_mat['ipsi, unrewarded, '+rew_str] = rew_preds * (ipsi_choice & unrewarded)
            elif include_side_reward_interaction:
                pred_mat['contra, '+rew_str] = rew_preds * contra_choice
                pred_mat['ipsi, '+rew_str] = rew_preds * ipsi_choice
            elif include_outcome_reward_interaction:
                pred_mat['rewarded, '+rew_str] = rew_preds * rewarded
                if not limit_rewarded_trials:
                    pred_mat['unrewarded, '+rew_str] = rew_preds * unrewarded
            else:
                pred_mat[rew_str] = rew_preds

        # add in switches
        if include_stay_switch:
            if include_stay_side_interaction:
                pred_mat['contra, switch'] = region_preds['switch'] * contra_choice
                pred_mat['ipsi, switch'] = region_preds['switch'] * ipsi_choice
            else:
                pred_mat['switch'] = region_preds['switch']
            
        pred_mat = pd.DataFrame(pred_mat)

        for signal_type in signals:
            for align in aligns:
                
                signal_mat = subj_stacked_signals[subj_id][signal_type][align][region]
                if limit_rewarded_trials:
                    signal_mat = signal_mat[full_rewarded,:]
                 
                reg_params[subj_id][signal_type][region][align] = regress_over_time(signal_mat, pred_mat)
                
                
# %% plot regression coefficients over time (Fig 5L)

# Create the reward history colormap
cmap = LinearSegmentedColormap.from_list('red_to_blue', [plt.cm.Reds(0.7), plt.cm.Blues(0.7)])

plot_signals = ['z_dff_iso_baseline_fband']
plot_regions = ['PL', 'DMS', 'DLS'] # 'DMS', 'PL'
plot_aligns = [Align.reward] #  Align.cue, Align.reward

# plot formatting
plot_dec = {'DMS': 1, 'DLS': 1, 'PL': 2}
x_inc = {'DMS': 0.3, 'DLS': 0.3, 'PL': 3}
y_inc = {'DMS': 0.3, 'DLS': 0.8, 'PL': 0.3}
plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'DLS': [-0.1,0.6], 'PL': [-1,2]},
             Align.reward: {'DMS': [-0.2,1.2], 'DLS': [-0.2,1.2], 'PL': [-2,12]}}

plot_n_back = n_back # 

plot_ind_subj = False
plot_subj_average = False
plot_meta_subj = True
plot_current_reward_separate = False
plot_rmse = False
use_ci_errors = False
plot_sig = True

# Process p-values for significance
sig_lvl = 0.01
method = 'bonferroni' # 'bonferroni' 'holm'

if plot_sig:
    for subj_id in analyzed_subjs:
        for region in plot_regions:
            for signal_type in plot_signals:
                for align in plot_aligns:
    
                    p_vals = reg_params[subj_id][signal_type][region][align]['p_vals']
                    sig = p_vals.apply(lambda x: smm.multipletests(x, alpha=sig_lvl, method=method)[0], axis=1, result_type='broadcast').astype(bool)
                    reg_params[subj_id][signal_type][region][align]['sig'] = sig

# Build plot groups based on regression options

# Simplify side/reward interaction labels only if not plotting them separately
if not plot_current_reward_separate:
    group_labels = {'contra, rewarded': 'rewarded', 'contra, unrewarded': 'unrewarded', 'ipsi, rewarded': 'rewarded', 'ipsi, unrewarded': 'unrewarded'}
else:
    group_labels = {}
    
rew_hist_label = 'reward ({})'
group_labels.update({g.format(rew_hist_label.format(i-plot_n_back)): rew_hist_label.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1) for g in 
                     ['{}', 'contra, {}', 'ipsi, {}', 'rewarded, {}', 'unrewarded, {}', 'contra, rewarded, {}', 'contra, unrewarded, {}', 'ipsi, rewarded, {}', 'ipsi, unrewarded, {}']})

# reward history groups
    
if include_outcome_reward_interaction and include_side_reward_interaction:
    plot_group_labels = ['Reward History before Rewarded Contra Choices', 'Reward History before Rewarded Ipsi Choices']
    hist_groups = ['contra, rewarded, reward ({})', 'ipsi, rewarded, reward ({})']
    if not limit_rewarded_trials:
        plot_group_labels.extend(['Reward History before Unrewarded Contra Choices', 'Reward History before Unrewarded Ipsi Choices'])
        hist_groups.extend(['contra, unrewarded, reward ({})', 'ipsi, unrewarded, reward ({})'])
        
    if plot_current_reward_separate:
        plot_groups = [[g.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)] for g in hist_groups]
    else:
        rew_groups = [['contra, rewarded'], ['ipsi, rewarded']]
        if not limit_rewarded_trials:
            rew_groups.extend([['contra, unrewarded'], ['ipsi, unrewarded']])
        
        plot_groups = [r+[g.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)] for r, g in zip(rew_groups, hist_groups)]
            
elif include_side_reward_interaction:
    plot_group_labels = ['Reward History before Contra Choices', 'Reward History before Ipsi Choices']
    
    hist_groups = ['contra, reward ({})', 'ipsi, reward ({})']
    if plot_current_reward_separate:
        plot_groups = [[g.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)] for g in hist_groups]
    elif include_current_reward:
        rew_groups = [['contra, rewarded'], ['ipsi, rewarded']]
        plot_groups = [r+[g.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)] for r, g in zip(rew_groups, hist_groups)]
    
elif include_outcome_reward_interaction:
    plot_group_labels = ['Reward History before Rewarded Choices']
    hist_groups = ['rewarded, reward ({})']
    
    if not limit_rewarded_trials:
        plot_group_labels.append('Reward History before Unrewarded Choices')
        hist_groups.append('unrewarded, reward ({})')
    
    if plot_current_reward_separate:
        plot_groups = [[g.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)] for g in hist_groups]
    else:
        rew_groups = [['rewarded']]
        if not limit_rewarded_trials:
            rew_groups.append(['unrewarded'])
        plot_groups = [r+[g.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)] for r, g in zip(rew_groups, hist_groups)]
else:
    plot_group_labels = ['Reward History']
    if plot_current_reward_separate:
        plot_groups = [['reward ({})'.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)]]
    else:
        if include_current_reward:
            plot_groups = [['rewarded']+['reward ({})'.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)]]
        else:
            plot_groups = [['reward ({})'.format(i-plot_n_back) for i in range(plot_n_back-1, -1, -1)]]
    
group_colors = [cmap(np.linspace(0,1,len(g))) for g in plot_groups]

# group all other parameters into one plot    
plot_group_labels.append('Other Parameters')
other_group = []

# intercept and current reward
if include_outcome_reward_interaction and include_side_reward_interaction:
    if plot_current_reward_separate:
        other_group.extend(['contra, rewarded', 'ipsi, rewarded'])
        if not limit_rewarded_trials:
            other_group.extend(['contra, unrewarded', 'ipsi, unrewarded'])
        
elif include_current_side or include_side_reward_interaction:
    other_group.extend(['contra choice', 'ipsi choice'])
    if plot_current_reward_separate and include_current_reward:
        if include_side_reward_interaction:
            other_group.extend(['contra, rewarded', 'ipsi, rewarded'])
        else:
            other_group.append('rewarded')
            
elif include_outcome_reward_interaction:
    if plot_current_reward_separate:
        other_group.append('rewarded')
        if not limit_rewarded_trials:
            other_group.append('unrewarded')
else:
    other_group.append('intercept')
    if plot_current_reward_separate and include_current_reward:
        other_group.append('rewarded')

if include_stay_switch:
    if include_stay_side_interaction:
        other_group.extend(['contra, switch', 'ipsi, switch'])
    else:
        other_group.append('switch')
    
if any(other_group):    
    plot_groups.append(other_group)
    group_colors.append(['C{}'.format(i) for i, _ in enumerate(other_group)])

width_ratios = [np.diff(plot_lims[align]['DMS'])[0] for align in plot_aligns]
    
n_rows = len(plot_regions)
n_cols = len(plot_aligns)

plot_data = {}

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    plot_data[signal_type] = {}
    
    if plot_ind_subj:
        for plot_group, group_label, colors in zip(plot_groups, plot_group_labels, group_colors):
            for subj_id in included_subjs:
                fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5.5, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
                axs = np.array(axs).reshape((n_rows, n_cols))
                
                fig.suptitle('{} Regression, Subj {}'.format(group_label, subj_id))
                
                for i, region in enumerate(plot_regions):
                    for j, align in enumerate(plot_aligns):
                        ax = axs[i,j]
                        
                        t_r = t[align][region]
                        subj_params = reg_params[subj_id][signal_type][region][align]
                        t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                        
                        if use_ci_errors:
                            plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                                   ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], 
                                                   sig=subj_params['sig'], t_sel=t_sel, colors=colors,
                                                   dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                        else:
                            plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                                   error=subj_params['se'], sig=subj_params['sig'], 
                                                   t_sel=t_sel, colors=colors, dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])   
                        
                        ax.set_title('{}, {}-aligned'.format(region, align))
                        
                        if j == 0:
                            ax.set_ylabel('Coefficient ({})'.format(y_label))
                            
        if plot_rmse:
            for subj_id in included_subjs:
                fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5.5, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
                axs = np.array(axs).reshape((n_rows, n_cols))
                fig.suptitle('Regression RMSE, Subj {}'.format(subj_id))
                
                for i, region in enumerate(plot_regions):
                    for j, align in enumerate(plot_aligns):
                        ax = axs[i,j]
                        t_r = t[align][region]
                        
                        rmse = reg_params[subj_id][signal_type][region][align]['rmse']
                        t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                        plot_regress_over_time(pd.DataFrame({'rmse': rmse}), t_r, ['rmse'], ax, region, t_sel=t_sel, plot_y0=False)
                        
                        ax.set_title('{}, {}-aligned'.format(region, align))
                        
                        if j == 0:
                            ax.set_ylabel('RMSE ({})'.format(y_label))
                        
    if plot_subj_average:
        for plot_group, group_label, colors in zip(plot_groups, plot_group_labels, group_colors):
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            fig.suptitle('{} Regression, Subject Avg'.format(group_label))
            
            for i, region in enumerate(plot_regions):
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    t_r = t[align][region]
                    
                    # average coefficients across subjects
                    all_params = pd.concat([reg_params[subj_id][signal_type][region][align]['params'] for subj_id in included_subjs])
                    param_avg = all_params.groupby(level=0).mean()
                    param_se = all_params.groupby(level=0).std() / np.sqrt(len(included_subjs))
                    
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
            
                    plot_regress_over_time(param_avg, t_r, plot_group, ax, region,
                                           error=param_se, t_sel=t_sel, colors=colors,
                                           dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if j == 0:
                        ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()
            
            fig.legend(handles, labels, loc='outside right center')
                    
                        
    if plot_meta_subj:
        for plot_group, group_label, colors in zip(plot_groups, plot_group_labels, group_colors):
            subj_id = 'all'
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(7, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            
            fig.suptitle('{} Regression, Subj {}'.format(group_label, subj_id))
            
            for i, region in enumerate(plot_regions):
                if not region in plot_data[signal_type]:
                    plot_data[signal_type][region] = {}
                    
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    t_r = t[align][region]
                    
                    subj_params = reg_params[subj_id][signal_type][region][align]
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    
                    if not align in plot_data[signal_type][region]:
                        plot_data[signal_type][region][align] = {}
                    
                    if use_ci_errors:
                        group_plot_data = plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                               ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], group_labels=group_labels,
                                               sig=subj_params['sig'], p_vals=subj_params['p_vals'], t_sel=t_sel, colors=colors,
                                               dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    else:
                        group_plot_data = plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                               error=subj_params['se'], sig=subj_params['sig'], p_vals=subj_params['p_vals'], group_labels=group_labels,
                                               t_sel=t_sel, colors=colors, dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    
                    plot_data[signal_type][region][align].update(group_plot_data)
                    
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if j == 0:
                        ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()
            
            fig.legend(handles, labels, loc='outside right center')
                        
            plot_name = '{}_{}_time_regression_{}_{}'.format('_'.join(plot_regions), '_'.join(plot_aligns), group_label, signal_type)
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
        
        for region in plot_regions:
            for align in plot_aligns:
                plot_data[signal_type][region][align] = pd.DataFrame.from_dict(plot_data[signal_type][region][align])
        
        if plot_rmse:
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5.5, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            
            fig.suptitle('Regression RMSE, Subj {}'.format(subj_id))
            
            for i, region in enumerate(plot_regions):
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    t_r = t[align][region]
                    
                    rmse = reg_params[subj_id][signal_type][region][align]['rmse']
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    plot_regress_over_time(pd.DataFrame({'rmse': rmse}), t_r, ['rmse'], ax, region, t_sel=t_sel, plot_y0=False)
                    
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if j == 0:
                        ax.set_ylabel('RMSE ({})'.format(y_label))
            
            
# %% Perform Full Side/Outcome Interaction Regression

fit_ind_subj = False
fit_meta_subj = True
include_rew_hist = False
use_unreward_hist = False # whether to use rewarded trials (False) or unrewarded trials (True) as the reward regressors
include_side_hist = False
use_same_diff_side_hist = True
use_diff_hist = True  
use_ipsi_hist = False # whether to use contra trials (False) or ipsi trials (True) as the choice regressors
include_interaction = True
use_full_interaction = True
interaction = 's/u' #'s/r', 's/u', 'd/r', 'd/u', 'c/r', 'c/u', 'i/r', 'i/u'

analyzed_subjs = []
if fit_ind_subj:
    analyzed_subjs.extend(included_subjs.tolist())

if fit_meta_subj:
    analyzed_subjs.append('all')
    
reg_params = {subj_id: {s: {r: {a: {} for a in aligns} 
                            for r in regions} 
                        for s in signals}
              for subj_id in analyzed_subjs}

for subj_id in analyzed_subjs:
    for region in regions:
        # build the predictor matrix based on the current options
        region_preds = subj_predictors[subj_id][region]
        
        for align in aligns:
            pred_mat = {}
            
            if align == Align.cue:
                poss_outcomes = ['all']
            elif align == Align.reward:
                poss_outcomes = ['rewarded', 'unrewarded']
            
            for side in ['contra', 'ipsi']:
                choice_sel = (region_preds['choice'] == side).astype(int)
                for outcome in poss_outcomes:
                    if outcome == 'rewarded':
                        outcome_sel = region_preds['reward (0)']
                    elif outcome == 'unrewarded':
                        outcome_sel = region_preds['unreward (0)']
                    else:
                        outcome_sel = np.full_like(choice_sel, 1)
                        
                    prefix = '{}, {}'.format(side, outcome)
                    
                    sub_sel = choice_sel * outcome_sel

                    pred_mat[prefix] = sub_sel
                    
                    for i in range(n_back-1, -1, -1):
                        rew_str = 'reward ({})'.format(i-n_back)
                        unrew_str = 'unreward ({})'.format(i-n_back)
                        contra_str = 'contra choice ({})'.format(i-n_back)
                        ipsi_str = 'ipsi choice ({})'.format(i-n_back)
                        rew_preds = region_preds[rew_str]
                        unrew_preds = region_preds[unrew_str]
                        contra_preds = region_preds[contra_str]
                        ipsi_preds = region_preds[ipsi_str]
                        
                        if side == 'contra':
                            same_side_preds = contra_preds
                            diff_side_preds = ipsi_preds
                        else:
                            same_side_preds = ipsi_preds
                            diff_side_preds = contra_preds
                        
                        if include_rew_hist:
                            if use_unreward_hist:
                                pred_mat[prefix+', '+unrew_str] = unrew_preds * sub_sel
                            else:
                                pred_mat[prefix+', '+rew_str] = rew_preds * sub_sel
                            
                        if include_side_hist:
                            if use_same_diff_side_hist:
                                if use_diff_hist:
                                    pred_mat[prefix+', diff choice ({})'.format(i-n_back)] = diff_side_preds * sub_sel
                                else:
                                    pred_mat[prefix+', same choice ({})'.format(i-n_back)] = same_side_preds * sub_sel
                            else:
                                if use_ipsi_hist:
                                    pred_mat[prefix+', '+ipsi_str] = ipsi_preds * sub_sel
                                else:
                                    pred_mat[prefix+', '+contra_str] = contra_preds * sub_sel
                        
                        if include_interaction:
                            if use_full_interaction:
                                if use_same_diff_side_hist:
                                    match interaction:
                                        case 's/r':
                                            pred_mat[prefix+', same unrewarded ({})'.format(i-n_back)] = unrew_preds * same_side_preds * sub_sel
                                            pred_mat[prefix+', diff unrewarded ({})'.format(i-n_back)] = unrew_preds * diff_side_preds * sub_sel
                                            pred_mat[prefix+', diff rewarded ({})'.format(i-n_back)] = rew_preds * diff_side_preds * sub_sel
                                        case 's/u':
                                            pred_mat[prefix+', same rewarded ({})'.format(i-n_back)] = rew_preds * same_side_preds * sub_sel
                                            pred_mat[prefix+', diff unrewarded ({})'.format(i-n_back)] = unrew_preds * diff_side_preds * sub_sel
                                            pred_mat[prefix+', diff rewarded ({})'.format(i-n_back)] = rew_preds * diff_side_preds * sub_sel
                                        case 'd/r':
                                            pred_mat[prefix+', same rewarded ({})'.format(i-n_back)] = rew_preds * same_side_preds * sub_sel
                                            pred_mat[prefix+', same unrewarded ({})'.format(i-n_back)] = unrew_preds * same_side_preds * sub_sel
                                            pred_mat[prefix+', diff unrewarded ({})'.format(i-n_back)] = unrew_preds * diff_side_preds * sub_sel
                                        case 'd/u':
                                            pred_mat[prefix+', same rewarded ({})'.format(i-n_back)] = rew_preds * same_side_preds * sub_sel
                                            pred_mat[prefix+', same unrewarded ({})'.format(i-n_back)] = unrew_preds * same_side_preds * sub_sel
                                            pred_mat[prefix+', diff rewarded ({})'.format(i-n_back)] = rew_preds * diff_side_preds * sub_sel
                                        
                                else:
                                    match interaction:
                                        case 'c/r':
                                            pred_mat[prefix+', contra unrewarded ({})'.format(i-n_back)] = unrew_preds * contra_preds * sub_sel
                                            pred_mat[prefix+', ipsi unrewarded ({})'.format(i-n_back)] = unrew_preds * ipsi_preds * sub_sel
                                            pred_mat[prefix+', ipsi rewarded ({})'.format(i-n_back)] = rew_preds * ipsi_preds * sub_sel
                                        case 'c/u':
                                            pred_mat[prefix+', contra rewarded ({})'.format(i-n_back)] = rew_preds * contra_preds * sub_sel
                                            pred_mat[prefix+', ipsi unrewarded ({})'.format(i-n_back)] = unrew_preds * ipsi_preds * sub_sel
                                            pred_mat[prefix+', ipsi rewarded ({})'.format(i-n_back)] = rew_preds * ipsi_preds * sub_sel
                                        case 'i/r':
                                            pred_mat[prefix+', contra rewarded ({})'.format(i-n_back)] = rew_preds * contra_preds * sub_sel
                                            pred_mat[prefix+', contra unrewarded ({})'.format(i-n_back)] = unrew_preds * contra_preds * sub_sel
                                            pred_mat[prefix+', ipsi unrewarded ({})'.format(i-n_back)] = unrew_preds * ipsi_preds * sub_sel
                                        case 'i/u':
                                            pred_mat[prefix+', contra rewarded ({})'.format(i-n_back)] = rew_preds * contra_preds * sub_sel
                                            pred_mat[prefix+', contra unrewarded ({})'.format(i-n_back)] = unrew_preds * contra_preds * sub_sel
                                            pred_mat[prefix+', ipsi rewarded ({})'.format(i-n_back)] = rew_preds * ipsi_preds * sub_sel
                            else:
                                if use_same_diff_side_hist:
                                    match interaction:
                                        case 's/r':
                                            pred_mat[prefix+', same rewarded ({})'.format(i-n_back)] = rew_preds * same_side_preds * sub_sel
                                        case 's/u':
                                            pred_mat[prefix+', same unrewarded ({})'.format(i-n_back)] = unrew_preds * same_side_preds * sub_sel
                                        case 'd/r':
                                            pred_mat[prefix+', diff rewarded ({})'.format(i-n_back)] = rew_preds * diff_side_preds * sub_sel
                                        case 'd/u':
                                            pred_mat[prefix+', diff unrewarded ({})'.format(i-n_back)] = unrew_preds * diff_side_preds * sub_sel
                                else:
                                    match interaction:
                                        case 'c/r':
                                            pred_mat[prefix+', contra rewarded ({})'.format(i-n_back)] = rew_preds * contra_preds * sub_sel
                                        case 'c/u':
                                            pred_mat[prefix+', contra unrewarded ({})'.format(i-n_back)] = unrew_preds * contra_preds * sub_sel
                                        case 'i/r':
                                            pred_mat[prefix+', ipsi rewarded ({})'.format(i-n_back)] = rew_preds * ipsi_preds * sub_sel
                                        case 'i/u':
                                            pred_mat[prefix+', ipsi unrewarded ({})'.format(i-n_back)] = unrew_preds * ipsi_preds * sub_sel
                                #     if use_unreward_hist:
                                #         pred_mat[prefix+', same unrewarded ({})'.format(i-n_back)] = unrew_preds * same_side_preds * sub_sel
                                #         #pred_mat[prefix+', diff unrewarded ({})'.format(i-n_back)] = unrew_preds * diff_side_preds * sub_sel
                                #     else:
                                #         pred_mat[prefix+', same rewarded ({})'.format(i-n_back)] = rew_preds * same_side_preds * sub_sel
                                #         #pred_mat[prefix+', diff rewarded ({})'.format(i-n_back)] = rew_preds * diff_side_preds * sub_sel
                                # else:
                                #     if use_unreward_hist:
                                #         pred_mat[prefix+', contra unrewarded ({})'.format(i-n_back)] = unrew_preds * contra_preds * sub_sel
                                #         #pred_mat[prefix+', ipsi unrewarded ({})'.format(i-n_back)] = unrew_preds * ipsi_preds * sub_sel
                                #     else:
                                #         pred_mat[prefix+', contra rewarded ({})'.format(i-n_back)] = rew_preds * contra_preds * sub_sel
                                #         #pred_mat[prefix+', ipsi rewarded ({})'.format(i-n_back)] = rew_preds * ipsi_preds * sub_sel
                    
            pred_mat = pd.DataFrame(pred_mat)

            for signal_type in signals:
                
                signal_mat = subj_stacked_signals[subj_id][signal_type][align][region]
                 
                reg_params[subj_id][signal_type][region][align] = regress_over_time(signal_mat, pred_mat)
                
                
# %% plot regression coefficients over time

plot_n_back = n_back 

# Create the colormaps
intercept_color = ['black']
contra_hist_cmap = LinearSegmentedColormap.from_list('green_light_to_dark', [np.array([95,214,81])/256, np.array([9,79,1])/256])(np.linspace(1,0,plot_n_back)).tolist()
ipsi_hist_cmap = LinearSegmentedColormap.from_list('pink_light_to_dark', [np.array([227,98,208])/256, np.array([94,0,80])/256])(np.linspace(1,0,plot_n_back)).tolist()
side_rew_hist_cmap = LinearSegmentedColormap.from_list('orange_light_to_dark', [np.array([252,151,78])/256, np.array([145,61,0])/256])(np.linspace(1,0,plot_n_back)).tolist()
side_unrew_hist_cmap = LinearSegmentedColormap.from_list('turqoise_light_to_dark', [np.array([84,230,235])/256, np.array([0,89,92])/256])(np.linspace(1,0,plot_n_back)).tolist()
rew_hist_cmap = plt.cm.Reds(np.linspace(1,0.4,plot_n_back)).tolist()
unrew_hist_cmap = plt.cm.Blues(np.linspace(1,0.4,plot_n_back)).tolist()

plot_signals = ['z_dff_iso']
plot_regions = ['DMS', 'PL'] #'DMS', 'PL'
plot_aligns = [Align.cue, Align.reward] #  Align.cue, Align.reward

# plot formatting
plot_dec = {'DMS': 1, 'PL': 2}
x_inc = {'DMS': None, 'PL': None}
y_inc = {'DMS': None, 'PL': None}
plot_lims = {Align.cue: {'DMS': [-0.1,0.6], 'PL': [-1,2]},
             Align.reward: {'DMS': [-0.2,1.2], 'PL': [-2,12]}}


plot_ind_subj = False
plot_subj_average = False
plot_meta_subj = True
plot_rmse = False
use_ci_errors = False
plot_sig = True
filt_outliers = False

# Process p-values for significance
sig_lvl = 0.05
method = 'bonferroni' # 'bonferroni' 'holm'

if plot_sig:
    for subj_id in analyzed_subjs:
        for region in plot_regions:
            for signal_type in plot_signals:
                for align in plot_aligns:
    
                    p_vals = reg_params[subj_id][signal_type][region][align]['p_vals']
                    sig = p_vals.apply(lambda x: smm.multipletests(x, alpha=sig_lvl, method=method)[0], axis=1, result_type='broadcast').astype(bool)
                    reg_params[subj_id][signal_type][region][align]['sig'] = sig
                    
# Build plot groups per column
rew_groups = ['reward ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
unrew_groups = ['unreward ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
contra_groups = ['contra choice ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
ipsi_groups = ['ipsi choice ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
same_groups = ['same choice ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
diff_groups = ['diff choice ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]

contra_rew_groups = ['contra rewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
contra_unrew_groups = ['contra unrewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
ipsi_rew_groups = ['ipsi rewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
ipsi_unrew_groups = ['ipsi unrewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]

same_rew_groups = ['same rewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
same_unrew_groups = ['same unrewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
diff_rew_groups = ['diff rewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]
diff_unrew_groups = ['diff unrewarded ({})'.format(i-n_back) for i in range(n_back-1,-1,-1)]

group_labels = {}
for choice in ['contra', 'ipsi']:
    for outcome in ['rewarded', 'unrewarded', 'all']:
        prefix = '{}, {}'.format(choice, outcome)
        group_labels[prefix] = prefix
        group_labels.update({'{}, {}'.format(prefix, g): g for g in rew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in unrew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g.replace('choice ','') for g in contra_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g.replace('choice ','') for g in ipsi_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in same_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in diff_groups})

        group_labels.update({'{}, {}'.format(prefix, g): g for g in contra_rew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in contra_unrew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in ipsi_rew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in ipsi_unrew_groups})
        
        group_labels.update({'{}, {}'.format(prefix, g): g for g in same_rew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in same_unrew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in diff_rew_groups})
        group_labels.update({'{}, {}'.format(prefix, g): g for g in diff_unrew_groups})
        
def generic_plot(subj_params, t_r, plot_group, ax, t_sel, colors):
    if use_ci_errors:
        plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, group_labels=group_labels,
                               ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], filt_outliers=filt_outliers,
                               sig=subj_params['sig'], t_sel=t_sel, colors=colors, plot_y0=False, plot_x0=False,
                               dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
    else:
        plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, group_labels=group_labels,
                               error=subj_params['se'], sig=subj_params['sig'], plot_y0=False, plot_x0=False, filt_outliers=filt_outliers,
                               t_sel=t_sel, colors=colors, dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
n_cols = 0
if include_rew_hist:
    n_cols += 1
if include_side_hist:
    n_cols += 1
if include_interaction:
    if not include_rew_hist and not include_side_hist and use_full_interaction:
        n_cols = 3
    else:
        n_cols += 1
        if use_full_interaction:
            n_cols += 1
        
plot_subjs = []
if plot_ind_subj:
    plot_subjs.extend(included_subjs.tolist())

if plot_meta_subj:
    plot_subjs.append('all')

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    for region in plot_regions:
        for align in plot_aligns:
            
            if align == Align.cue:
                n_rows = 2
                poss_outcomes = ['all']
            elif align == Align.reward:
                n_rows = 4
                poss_outcomes = ['rewarded', 'unrewarded']
            
            for subj_id in plot_subjs:
                fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 3*n_rows+0.1))
                axs = axs.reshape((n_rows, n_cols))
    
                fig.suptitle('Full Reward/Choice History Regression, {}, {}-aligned, Subj {}'.format(region, align, subj_id))
                
                subj_params = reg_params[subj_id][signal_type][region][align]
                t_r = t[align][region]
                t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                
                for i, outcome in enumerate(poss_outcomes):
                    for j, choice in enumerate(['contra', 'ipsi']):
                            
                        prefix = '{}, {}'.format(choice, outcome)
                        if i == 0:
                            row = j
                        else:
                            row = 2+j

                        col_idx = 0
                        
                        if include_rew_hist:
                            # reward history coefficients
                            ax = axs[row, col_idx]
                            # plot trial group intercept
                            if use_unreward_hist:
                                plot_groups = [prefix]+['{}, {}'.format(prefix, g) for g in unrew_groups]
                                plot_colors = intercept_color + unrew_hist_cmap
                            else:
                                plot_groups = [prefix]+['{}, {}'.format(prefix, g) for g in rew_groups]
                                plot_colors = intercept_color + rew_hist_cmap
                            
                            generic_plot(subj_params, t_r, plot_groups, ax, t_sel, plot_colors)
                            
                            plot_utils.plot_dashlines(0, dir='v', ax=ax)
                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            ax.set_title('{}, Reward History'.format(prefix))
                            ax.set_ylabel('Coefficient ({})'.format(y_label))
                            ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
                            plot_utils.show_axis_labels(ax, axis='y')
                            
                            col_idx += 1
                        
                        # side history coefficients
                        if include_side_hist:
                            ax = axs[row, col_idx]
                            # plot trial group intercept
                            if use_same_diff_side_hist:
                                if use_diff_hist:
                                    plot_groups = [prefix]+['{}, {}'.format(prefix, g) for g in diff_groups]
                                    plot_colors = intercept_color + ipsi_hist_cmap
                                else:
                                    plot_groups = [prefix]+['{}, {}'.format(prefix, g) for g in same_groups]
                                    plot_colors = intercept_color + contra_hist_cmap
                            else:
                                if use_ipsi_hist:
                                    plot_groups = [prefix]+['{}, {}'.format(prefix, g) for g in ipsi_groups]
                                    plot_colors = intercept_color + ipsi_hist_cmap
                                else:
                                    plot_groups = [prefix]+['{}, {}'.format(prefix, g) for g in contra_groups]
                                    plot_colors = intercept_color + contra_hist_cmap
                            
                            generic_plot(subj_params, t_r, plot_groups, ax, t_sel, plot_colors)
                            
                            plot_utils.plot_dashlines(0, dir='v', ax=ax)
                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            ax.set_title('{}, Choice Side History'.format(prefix))
                            ax.set_ylabel('Coefficient ({})'.format(y_label))
                            ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
                            plot_utils.show_axis_labels(ax, axis='y')
                            
                            col_idx += 1
                        
                        if include_interaction:
                            if not include_rew_hist and not include_side_hist and use_full_interaction:
                                if use_same_diff_side_hist:
                                    intxs = ['s/r', 'd/r', 's/u', 'd/u']
                                else:
                                    intxs = ['c/r', 'i/r', 'c/u', 'i/u']
                                
                                intxs.remove(interaction)

                                for col_idx, intx in enumerate(intxs):
                                    match intx:
                                        case 's/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Same Side Reward History'.format(prefix)
                                        case 's/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Same Side Unreward History'.format(prefix)
                                        case 'd/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Diff Side Reward History'.format(prefix)
                                        case 'd/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Diff Side Unreward History'.format(prefix)
                                        case 'c/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Contra Reward History'.format(prefix)
                                        case 'c/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Contra Unreward History'.format(prefix)
                                        case 'i/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Ipsi Reward History'.format(prefix)
                                        case 'i/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Ipsi Unreward History'.format(prefix)
                                    
                                    ax = axs[row, col_idx]
                                    generic_plot(subj_params, t_r, plot_groups, ax, t_sel, plot_colors)
                                    
                                    plot_utils.plot_dashlines(0, dir='v', ax=ax)
                                    plot_utils.plot_dashlines(0, dir='h', ax=ax)
                                    ax.set_title(title)
                                    ax.set_ylabel('Coefficient ({})'.format(y_label))
                                    ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
                                    plot_utils.show_axis_labels(ax, axis='y')
                            else:
                                # contra/same outcome history coefficients
                                ax = axs[row, col_idx]
                                # plot trial group intercept
                                if use_full_interaction:
                                    if use_same_diff_side_hist:
                                        plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_rew_groups + same_unrew_groups]
                                        plot_colors = intercept_color + side_rew_hist_cmap + side_unrew_hist_cmap
                                        title = '{}, Same Side Reward History'.format(prefix)
                                    else:
                                        plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_rew_groups + contra_unrew_groups]
                                        plot_colors = intercept_color + side_rew_hist_cmap + side_unrew_hist_cmap
                                        title = '{}, Contra Reward History'.format(prefix)                              
                                else:
                                    match interaction:
                                        case 's/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Same Side Reward History'.format(prefix)
                                        case 's/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Same Side Unreward History'.format(prefix)
                                        case 'd/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Diff Side Reward History'.format(prefix)
                                        case 'd/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Diff Side Unreward History'.format(prefix)
                                        case 'c/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_rew_groups]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Contra Reward History'.format(prefix)
                                        case 'c/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Contra Unreward History'.format(prefix)
                                        case 'i/r':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_rew_groups[:-1]]
                                            plot_colors = intercept_color + side_rew_hist_cmap
                                            title = '{}, Ipsi Reward History'.format(prefix)
                                        case 'i/u':
                                            plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_unrew_groups]
                                            plot_colors = intercept_color + side_unrew_hist_cmap
                                            title = '{}, Ipsi Unreward History'.format(prefix)
                                            
                                    # if use_same_diff_side_hist:
                                    #     if use_unreward_hist:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_unrew_groups]
                                    #         plot_colors = intercept_color + side_unrew_hist_cmap
                                    #     else:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in same_rew_groups]
                                    #         plot_colors = intercept_color + side_rew_hist_cmap
                                    # else:
                                    #     if use_unreward_hist:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_unrew_groups]
                                    #         plot_colors = intercept_color + side_unrew_hist_cmap
                                    #     else:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in contra_rew_groups]
                                    #         plot_colors = intercept_color + side_rew_hist_cmap
                                            
                                
                                
                                generic_plot(subj_params, t_r, plot_groups, ax, t_sel, plot_colors)
                                
                                plot_utils.plot_dashlines(0, dir='v', ax=ax)
                                plot_utils.plot_dashlines(0, dir='h', ax=ax)
                                ax.set_title(title)
                                ax.set_ylabel('Coefficient ({})'.format(y_label))
                                ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
                                plot_utils.show_axis_labels(ax, axis='y')
                                
                                # ipsi/outcome history coefficients
                                
                                # plot trial group intercept
                                if use_full_interaction:
                                    ax = axs[row, col_idx+1]
                                    if use_same_diff_side_hist:
                                        plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_rew_groups + diff_unrew_groups]
                                        plot_colors = intercept_color + side_rew_hist_cmap + side_unrew_hist_cmap
                                        title = '{}, Same Side Reward History'.format(prefix)
                                    else:
                                        plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_rew_groups + ipsi_unrew_groups]
                                        plot_colors = intercept_color + side_rew_hist_cmap + side_unrew_hist_cmap
                                        title = '{}, Contra Reward History'.format(prefix)                 
                                    # if use_same_diff_side_hist:
                                    #     if use_unreward_hist:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_unrew_groups]
                                    #         plot_colors = intercept_color + side_unrew_hist_cmap
                                    #     else:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_rew_groups]
                                    #         plot_colors = intercept_color + side_rew_hist_cmap
                                    # else:
                                    #     if use_unreward_hist:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_unrew_groups]
                                    #         plot_colors = intercept_color + side_unrew_hist_cmap
                                    #     else:
                                    #         plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_rew_groups]
                                    #         plot_colors = intercept_color + side_rew_hist_cmap
                                # else:
                                #     if use_same_diff_side_hist:
                                #         if use_unreward_hist:
                                #             plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_unrew_groups]
                                #             plot_colors = intercept_color + side_unrew_hist_cmap
                                #         else:
                                #             plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in diff_rew_groups]
                                #             plot_colors = intercept_color + side_rew_hist_cmap
                                #     else:
                                #         if use_unreward_hist:
                                #             plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_unrew_groups]
                                #             plot_colors = intercept_color + side_unrew_hist_cmap
                                #         else:
                                #             plot_groups = [prefix] + ['{}, {}'.format(prefix, g) for g in ipsi_rew_groups]
                                #             plot_colors = intercept_color + side_rew_hist_cmap
                                        
                                    if use_same_diff_side_hist:
                                        title = '{}, Diff Side Reward History'.format(prefix)
                                    else:
                                        title = '{}, Ipsi Reward History'.format(prefix)
                                        
                                    generic_plot(subj_params, t_r, plot_groups, ax, t_sel, plot_colors)
                                    
                                    plot_utils.plot_dashlines(0, dir='v', ax=ax)
                                    plot_utils.plot_dashlines(0, dir='h', ax=ax)
                                    ax.set_title(title)
                                    ax.set_ylabel('Coefficient ({})'.format(y_label))
                                    ax.legend(loc='upper right', fontsize=8, framealpha=0.5)
                                    plot_utils.show_axis_labels(ax, axis='y')
                            
        # if plot_rmse:
        #     for subj_id in included_subjs:
        #         fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5.5, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
        #         axs = np.array(axs).reshape((n_rows, n_cols))
        #         fig.suptitle('Regression RMSE, Subj {}'.format(subj_id))
                
        #         for i, region in enumerate(plot_regions):
        #             for j, align in enumerate(plot_aligns):
        #                 ax = axs[i,j]
        #                 t_r = t[align][region]
                        
        #                 rmse = reg_params[subj_id][signal_type][region][align]['rmse']
        #                 t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
        #                 plot_regress_over_time(pd.DataFrame({'rmse': rmse}), t_r, ['rmse'], ax, region, t_sel=t_sel, plot_y0=False)
                        
        #                 ax.set_title('{}, {}-aligned'.format(region, align))
                        
        #                 if j == 0:
        #                     ax.set_ylabel('RMSE ({})'.format(y_label))
                        
    # if plot_subj_average:
    #     for plot_group, group_label, colors in zip(plot_groups, plot_group_labels, group_colors):
    #         fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5.5, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
    #         axs = np.array(axs).reshape((n_rows, n_cols))
    #         fig.suptitle('{} Regression, Subject Avg'.format(group_label))
            
    #         for i, region in enumerate(plot_regions):
    #             for j, align in enumerate(plot_aligns):
    #                 ax = axs[i,j]
    #                 t_r = t[align][region]
                    
    #                 # average coefficients across subjects
    #                 all_params = pd.concat([reg_params[subj_id][signal_type][region][align]['params'] for subj_id in included_subjs])
    #                 param_avg = all_params.groupby(level=0).mean()
    #                 param_se = all_params.groupby(level=0).std() / np.sqrt(len(included_subjs))
                    
    #                 t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
            
    #                 plot_regress_over_time(param_avg, t_r, plot_group, ax, region,
    #                                        error=param_se, t_sel=t_sel, colors=colors,
    #                                        dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    
    #                 ax.set_title('{}, {}-aligned'.format(region, align))
                    
    #                 if j == 0:
    #                     ax.set_ylabel('Coefficient ({})'.format(y_label))


# %% perform regression to investigate signatures of perseveration

separate_sides = False
include_reward = False
trial_filter = 'rewarded'
fit_ind_subj = False
fit_meta_subj = True

analyzed_subjs = []
if fit_ind_subj:
    analyzed_subjs.extend(included_subjs.tolist())

if fit_meta_subj:
    analyzed_subjs.append('all')
    
reg_params = {subj_id: {s: {r: {a: {} for a in aligns} 
                            for r in regions} 
                        for s in signals}
              for subj_id in analyzed_subjs}

for subj_id in analyzed_subjs:
    for region in regions:
        # build the predictor matrix based on the current options
        region_preds = subj_predictors[subj_id][region]
        pred_mat = {}
        
        # limit to trials
        match trial_filter:
            case 'rewarded':
                sel = region_preds['reward (-1)'].astype(bool)
            case 'unrewarded':
                sel = region_preds['unreward (-1)'].astype(bool)
            case 'all':
                sel = pd.Series(True, index=region_preds.index)
        
        region_preds = region_preds[sel]

        contra_choice = (region_preds['choice'] == 'contra').astype(int)
        ipsi_choice = (region_preds['choice'] == 'ipsi').astype(int)
        rewarded = region_preds['reward (0)']
        
        if separate_sides:
            for side in ['contra', 'ipsi']:
                side_sel = contra_choice if side == 'contra' else ipsi_choice
                
                pred_mat[side] = side_sel
                pred_mat[side + ' stay'] = side_sel * region_preds['stay']
                
                if include_reward:
                    pred_mat[side + ' rewarded'] = side_sel * rewarded
                    pred_mat[side + ' rewarded stay'] = side_sel * rewarded * region_preds['stay']
        else:
            pred_mat['intercept'] = 1
            pred_mat['stay'] = region_preds['stay']
            
            if include_reward:
                pred_mat['rewarded'] = rewarded
                pred_mat['rewarded stay'] = rewarded * region_preds['stay']
            
        pred_mat = pd.DataFrame(pred_mat)

        for signal_type in signals:
            for align in aligns:
                
                signal_mat = subj_stacked_signals[subj_id][signal_type][align][region][sel,:]
                 
                reg_params[subj_id][signal_type][region][align] = regress_over_time(signal_mat, pred_mat)
                
                
# %% plot regression coefficients over time

plot_signals = ['z_dff_iso']
plot_regions = ['DMS'] # 'DMS', 'PL'
plot_aligns = [Align.cue] #  Align.cue, Align.reward

# plot formatting
plot_dec = {'DMS': 1, 'PL': 2}
x_inc = {'DMS': 0.3, 'PL': 3}
y_inc = {'DMS': 0.3, 'PL': 0.3}
plot_lims = {Align.cue: {'DMS': [-0.1,1], 'PL': [-1,2]},
             Align.reward: {'DMS': [-0.2,1.2], 'PL': [-2,12]}}

fig_width = 5.5 if len(plot_regions) == 2 else 4

plot_ind_subj = False
plot_subj_average = False
plot_meta_subj = True
plot_rmse = False
use_ci_errors = False
plot_sig = True

# Process p-values for significance
sig_lvl = 0.05
method = 'bonferroni' # 'bonferroni' 'holm'

if plot_sig:
    for subj_id in analyzed_subjs:
        for region in plot_regions:
            for signal_type in plot_signals:
                for align in plot_aligns:
    
                    p_vals = reg_params[subj_id][signal_type][region][align]['p_vals']
                    sig = p_vals.apply(lambda x: smm.multipletests(x, alpha=sig_lvl, method=method)[0], axis=1, result_type='broadcast').astype(bool)
                    reg_params[subj_id][signal_type][region][align]['sig'] = sig

# Build plot groups based on regression options

if separate_sides:
    plot_group_titles = ['Perseverative Effects Contra Choices', 'Perseverative Effects Ipsi Choices'] if include_reward else ['Perseverative Effects by Choice Side']
    plot_groups = []
    group_labels = {}
    for side in ['contra', 'ipsi']:
        if include_reward:
            plot_groups.append([side, side+' stay', side+' rewarded', side+' rewarded stay'])
            group_labels.update({side: 'avg', side+' stay': 'stay', side+' rewarded': 'rewarded', side+' rewarded stay': 'rewarded stay'})
        else:
            plot_groups.extend([side, side+' stay'])
else:
    plot_group_titles = ['Perseverative Effects']
    plot_groups = ['intercept', 'stay']
    group_labels = {'intercept': 'avg'}
    if include_reward:
        plot_groups.extend(['rewarded', 'rewarded stay'])
        
if not utils.is_list(plot_groups[0]):
    plot_groups = [plot_groups]

width_ratios = [np.diff(plot_lims[align]['DMS'])[0] for align in plot_aligns]
    
n_rows = len(plot_regions)
n_cols = len(plot_aligns)

for signal_type in plot_signals:
    signal_label, y_label = fpah.get_signal_type_labels(signal_type)
    
    if plot_ind_subj:
        for plot_group, group_label in zip(plot_groups, plot_group_titles):
            for subj_id in included_subjs:
                fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(fig_width, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
                axs = np.array(axs).reshape((n_rows, n_cols))
                
                fig.suptitle('{} Regression, Subj {}'.format(group_label, subj_id))
                
                for i, region in enumerate(plot_regions):
                    for j, align in enumerate(plot_aligns):
                        ax = axs[i,j]
                        
                        t_r = t[align][region]
                        subj_params = reg_params[subj_id][signal_type][region][align]
                        t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                        
                        if use_ci_errors:
                            plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                                   ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], 
                                                   sig=subj_params['sig'], t_sel=t_sel, group_labels=group_labels,
                                                   dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                        else:
                            plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                                   error=subj_params['se'], sig=subj_params['sig'], 
                                                   t_sel=t_sel, group_labels=group_labels, dec=plot_dec[region], 
                                                   x_inc=x_inc[region], y_inc=y_inc[region])   
                        
                        ax.set_title('{}, {}-aligned'.format(region, align))
                        
                        if j == 0:
                            ax.set_ylabel('Coefficient ({})'.format(y_label))
                            
        if plot_rmse:
            for subj_id in included_subjs:
                fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(fig_width, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
                axs = np.array(axs).reshape((n_rows, n_cols))
                fig.suptitle('Regression RMSE, Subj {}'.format(subj_id))
                
                for i, region in enumerate(plot_regions):
                    for j, align in enumerate(plot_aligns):
                        ax = axs[i,j]
                        t_r = t[align][region]
                        
                        rmse = reg_params[subj_id][signal_type][region][align]['rmse']
                        t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                        plot_regress_over_time(pd.DataFrame({'rmse': rmse}), t_r, ['rmse'], ax, region, t_sel=t_sel, plot_y0=False)
                        
                        ax.set_title('{}, {}-aligned'.format(region, align))
                        
                        if j == 0:
                            ax.set_ylabel('RMSE ({})'.format(y_label))
                        
    if plot_subj_average:
        for plot_group, group_label in zip(plot_groups, plot_group_titles):
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(fig_width, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            fig.suptitle('{} Regression, Subject Avg'.format(group_label))
            
            for i, region in enumerate(plot_regions):
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    t_r = t[align][region]
                    
                    # average coefficients across subjects
                    all_params = pd.concat([reg_params[subj_id][signal_type][region][align]['params'] for subj_id in included_subjs])
                    param_avg = all_params.groupby(level=0).mean()
                    param_se = all_params.groupby(level=0).std() / np.sqrt(len(included_subjs))
                    
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
            
                    plot_regress_over_time(param_avg, t_r, plot_group, ax, region,
                                           error=param_se, t_sel=t_sel, group_labels=group_labels,
                                           dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if j == 0:
                        ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
    if plot_meta_subj:
        for plot_group, group_label in zip(plot_groups, plot_group_titles):
            subj_id = 'all'
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(fig_width, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            
            fig.suptitle('{} Regression, Subj {}'.format(group_label, subj_id))
            
            for i, region in enumerate(plot_regions):
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    t_r = t[align][region]
                    
                    subj_params = reg_params[subj_id][signal_type][region][align]
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    
                    if use_ci_errors:
                        plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                               ci_lower=subj_params['ci_lower'], ci_upper=subj_params['ci_upper'], 
                                               sig=subj_params['sig'], t_sel=t_sel, group_labels=group_labels,
                                               dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    else:
                        plot_regress_over_time(subj_params['params'], t_r, plot_group, ax, region,
                                               error=subj_params['se'], sig=subj_params['sig'], 
                                               t_sel=t_sel, group_labels=group_labels, dec=plot_dec[region], x_inc=x_inc[region], y_inc=y_inc[region])
                    
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if j == 0:
                        ax.set_ylabel('Coefficient ({})'.format(y_label))
                        
            plot_name = '{}_{}_persev_regression_{}_{}'.format('_'.join(plot_regions), '_'.join(plot_aligns), group_label, signal_type)
            fpah.save_fig(fig, fpah.get_figure_save_path('Two-armed Bandit', 'Reward History', plot_name), format='pdf')
            
        if plot_rmse:
            fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(fig_width, 3*n_rows+0.1), width_ratios=width_ratios, sharey='row')
            axs = np.array(axs).reshape((n_rows, n_cols))
            
            fig.suptitle('Regression RMSE, Subj {}'.format(subj_id))
            
            for i, region in enumerate(plot_regions):
                for j, align in enumerate(plot_aligns):
                    ax = axs[i,j]
                    t_r = t[align][region]
                    
                    rmse = reg_params[subj_id][signal_type][region][align]['rmse']
                    t_sel = (t_r > plot_lims[align][region][0]) & (t_r < plot_lims[align][region][1])
                    plot_regress_over_time(pd.DataFrame({'rmse': rmse}), t_r, ['rmse'], ax, region, t_sel=t_sel, plot_y0=False)
                    
                    ax.set_title('{}, {}-aligned'.format(region, align))
                    
                    if j == 0:
                        ax.set_ylabel('RMSE ({})'.format(y_label))


# %% Look at correlations between response latencies and cue peaks
signal_type = 'z_dff_iso'
dms_cue_peaks = filt_peak_metrics[(filt_peak_metrics['region'] == 'DMS') & (filt_peak_metrics['align'] == Align.cue) 
                                  & (filt_peak_metrics['signal_type'] == signal_type) & (filt_peak_metrics['cpoke_out_latency'] > 0)]

plot_ind_subj = True
plot_meta_subj = True

plot_subjs = []
if plot_ind_subj:
    plot_subjs.extend(included_subjs.tolist())

if plot_meta_subj:
    plot_subjs.append('all')
    
# plot peak height by response latency, broken out by choice side
for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_peaks = dms_cue_peaks
    else:
        subj_peaks = dms_cue_peaks[dms_cue_peaks['subj_id'] == subj_id]
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained')
    fig.suptitle('Cue Peak Amplitudes by Response Times - {}'.format(subj_id))
    for ax, side in zip(axs, sides):
        side_peaks = subj_peaks[subj_peaks['side'] == side]
        sb.scatterplot(side_peaks, x='peak_height', y='RT', ax=ax)
        ax.set_title(side)
        ax.set_xlabel('Peak Amplitude (Z-dF/F)')
        ax.set_ylabel('Response Latency (s)')
    
# plot peak time by response latency, broken out by choice side
for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_peaks = dms_cue_peaks
    else:
        subj_peaks = dms_cue_peaks[dms_cue_peaks['subj_id'] == subj_id]
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained')
    fig.suptitle('Cue Peak Times by Response Times - {}'.format(subj_id))
    for ax, side in zip(axs, sides):
        side_peaks = subj_peaks[subj_peaks['side'] == side]
        sb.scatterplot(side_peaks, x='peak_time', y='RT', ax=ax)
        ax.set_title(side)
        ax.set_xlabel('Peak Time (s)')
        ax.set_ylabel('Response Latency (s)')

# plot response latency by cpoke out latency, broken out by choice side
for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_peaks = dms_cue_peaks
    else:
        subj_peaks = dms_cue_peaks[dms_cue_peaks['subj_id'] == subj_id]
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained')
    fig.suptitle('Poke Out Times by Response Times - {}'.format(subj_id))
    for ax, side in zip(axs, sides):
        side_peaks = subj_peaks[subj_peaks['side'] == side]
        sb.scatterplot(side_peaks, x='cpoke_out_latency', y='RT', ax=ax)
        ax.set_title(side)
        ax.set_xlabel('Cpoke Out Latency (s)')
        ax.set_ylabel('Response Latency (s)')
        
# plot cpoke out latency by peak height/time, broken out by choice side
for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_peaks = dms_cue_peaks
    else:
        subj_peaks = dms_cue_peaks[dms_cue_peaks['subj_id'] == subj_id]
    
    fig, axs = plt.subplots(2, 2, sharex=True, sharey='row', layout='constrained')
    fig.suptitle('Poke Out Time by Peak Height/Time - {}'.format(subj_id))
    for i, (col, label) in enumerate(zip(['peak_time', 'peak_height'], ['Peak Time (s)', 'Peak Amplitude (Z-dF/F)'])):
        for j, side in enumerate(sides):
            ax = axs[i,j]
            side_peaks = subj_peaks[subj_peaks['side'] == side]
            sb.scatterplot(side_peaks, x='cpoke_out_latency', y=col, ax=ax)
            ax.set_title(side)
            ax.set_xlabel('Cpoke Out Latency (s)')
            ax.set_ylabel(label)
            plot_utils.show_axis_labels(ax, axis='x')

# Look at differences in cue peak time and poke out time
bins = np.arange(-1,2.2,0.2)
for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_peaks = dms_cue_peaks
    else:
        subj_peaks = dms_cue_peaks[dms_cue_peaks['subj_id'] == subj_id]
        
    peak_poke_out_diff = subj_peaks['cpoke_out_latency'] - subj_peaks['peak_time']
    
    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True, layout='constrained')
    fig.suptitle('Poke Out Latency & Peak Time Difference - {}'.format(subj_id))
    for ax, side in zip(axs, sides):
        side_sel = subj_peaks['side'] == side
        ax.hist(peak_poke_out_diff[side_sel], bins=bins, density=True)
        ax.set_title(side)
        ax.set_xlabel('Cpoke Out Latency - Peak Time (s)')
        ax.set_ylabel('Density')
        plot_utils.plot_x0line(ax=ax)

# %% Look at dLight values at the time of poke out for contra/ipsi
poke_out_vals = []
align = Align.cpoke_out
region = 'DMS'
signal_type = 'z_dff_iso'
t = aligned_signals['t']

for subj_id in subj_ids:
    if subj_id in ignored_subjects:
        continue
    for sess_id in sess_ids[subj_id]:
        if sess_id in ignored_signals[region] or not signal_type in aligned_signals[subj_id][sess_id]:
            continue
        
        t_r = t[align][region]
        region_side = implant_info[subj_id][region]['side']
        
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        responded = ~np.isnan(trial_data['response_time']).to_numpy()
        cpoke_out_latency = trial_data['cpoke_out_latency'].to_numpy()
        choice = trial_data['choice']
        choice_side = choice.apply(lambda x: fpah.get_implant_rel_side(x, region_side) if not x == 'none' else 'none').to_numpy()

        mat = aligned_signals[subj_id][sess_id][signal_type][align][region]
        zero_idx = np.argmin(np.abs(t_r))
        
        for i in range(mat.shape[0]):
            if responded[i] and cpoke_out_latency[i] > 0:
                poke_out_vals.append(dict([('subj_id', subj_id), ('sess_id', sess_id), ('trial', i),
                                           ('side', choice_side[i]), ('val', mat[i,zero_idx])]))
                
poke_out_vals = pd.DataFrame(poke_out_vals)

plot_ind_subj = True
plot_meta_subj = True

plot_subjs = []
if plot_ind_subj:
    plot_subjs.extend(included_subjs.tolist())

if plot_meta_subj:
    plot_subjs.append('all')
    
# plot peak height by response latency, broken out by choice side
for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_vals = poke_out_vals
    else:
        subj_vals = poke_out_vals[poke_out_vals['subj_id'] == subj_id]
    
    fig, ax = plt.subplots(1, 1, layout='constrained')
    fig.suptitle('Cpoke Out Values by Choice Side - {}'.format(subj_id))
    sb.boxplot(subj_vals, x='side', y='val', hue='side', ax=ax)
    ax.set_xlabel('Choice Side')
    ax.set_ylabel('Signal at Poke Out (Z-dF/F)')
