# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:33:24 2023

@author: tanne
"""

# %% imports

import init
import pandas as pd
import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
from hankslab_db import db_access
from hankslab_db import tonecatdelayresp_db as wm_db, basicRLtasks_db as bandit_db
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
import numpy as np
import matplotlib.pyplot as plt
import copy
import os.path as path
import pickle
import time

# %% Load behavior data

reload = False

# used for saving plots
wm_beh_name = 'Single Tone WM'
wm_sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=7)

bandit_beh_name = 'Two-armed Bandit'
bandit_sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2)

# optionally limit sessions based on subject ids
subj_ids = [198, 199]
wm_sess_ids = {k: [s for s in v if not s in fpah.__sess_ignore] for k, v in wm_sess_ids.items() if k in subj_ids}
bandit_sess_ids = {k: [s for s in v if not s in fpah.__sess_ignore] for k, v in bandit_sess_ids.items() if k in subj_ids}

subj_ids = np.unique(list(wm_sess_ids.keys())+list(bandit_sess_ids.keys()))

wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

wm_sess_data = wm_loc_db.get_behavior_data(utils.flatten(wm_sess_ids), reload=reload)
bandit_sess_data = bandit_loc_db.get_behavior_data(utils.flatten(bandit_sess_ids), reload=reload)

implant_info = db_access.get_fp_implant_info(subj_ids)

wm_sess_data['cpoke_out_latency'] = wm_sess_data['cpoke_out_time'] - wm_sess_data['response_cue_time']
# bandit_sess_data['cpoke_out_latency'] = bandit_sess_data['cpoke_out_time'] - bandit_sess_data['response_cue_time']

# calculate center poke duration bins
bin_size = 1

dur_bin_max = np.ceil(np.max(wm_sess_data['stim_dur'])/bin_size)
dur_bin_min = np.floor(np.min(wm_sess_data['stim_dur'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
wm_dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

wm_sess_data['cpoke_dur_bin'] = wm_sess_data['stim_dur'].apply(
    lambda x: wm_dur_bin_labels[np.where(x >= dur_bins)[0][-1]])
# make sure they are always sorted appropriately using categories
wm_sess_data['cpoke_dur_bin'] = pd.Categorical(wm_sess_data['cpoke_dur_bin'], categories=wm_dur_bin_labels)

bin_size = 0.5

dur_bin_max = np.ceil(np.max(bandit_sess_data['trial_length'])/bin_size)
dur_bin_min = np.floor(np.min(bandit_sess_data['trial_length'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
bandit_dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

bandit_sess_data['cpoke_dur_bin'] = bandit_sess_data['trial_length'].apply(
    lambda x: bandit_dur_bin_labels[np.where(x >= dur_bins)[0][-1]])
# make sure they are always sorted appropriately using categories
bandit_sess_data['cpoke_dur_bin'] = pd.Categorical(bandit_sess_data['cpoke_dur_bin'], categories=bandit_dur_bin_labels)

# calculate WM response delay bins
wm_sess_data['resp_delay'] = wm_sess_data['stim_dur'] - wm_sess_data['rel_tone_end_times']

n_delay_bins = 4
quantiles = np.quantile(wm_sess_data['resp_delay'], np.linspace(1/n_delay_bins, 1, n_delay_bins-1, endpoint=False))
delay_bins = np.r_[np.floor(np.min(wm_sess_data['resp_delay'])), quantiles, np.ceil(np.max(wm_sess_data['resp_delay']))]
delay_bin_labels = ['{:.1f}-{:.1f}s'.format(delay_bins[i], delay_bins[i+1]) for i in range(len(delay_bins)-1)]

wm_sess_data['resp_delay_bin'] = wm_sess_data['resp_delay'].apply(
    lambda x: delay_bin_labels[np.where(x >= delay_bins)[0][-1]])
# make sure they are always sorted appropriately using categories
wm_sess_data['resp_delay_bin'] = pd.Categorical(wm_sess_data['resp_delay_bin'], categories=delay_bin_labels)

# calculate reward history

rew_rate_n_back = 3
bah.calc_trial_hist(wm_sess_data, n_back=rew_rate_n_back)
wm_sess_data['n_rew_hist'] = wm_sess_data['rew_hist'].apply(sum)
bah.calc_trial_hist(bandit_sess_data, n_back=rew_rate_n_back)
bandit_sess_data['n_rew_hist'] = bandit_sess_data['rew_hist'].apply(sum)

# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
rew_hist_bin_edges = np.arange(-0.5, rew_rate_n_back+1.5, 1)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{}'.format(i) for i,b in enumerate(rew_hist_bins)}


# %% Get and process photometry data

recalculate = False
recalculate_norm_tasks = [] # 'wm', 'bandit'
tilt_t = False
baseline_correction = True
save_process_plots = False
show_process_plots = False
filter_dropout_outliers = False

# ignored_signals = {'PL': [],
#                    'DMS': [],
#                    'DLS': [],
#                    'TS': []}

# ignored_subjects = [] #[182] [179]

reprocess_sess_ids = []

signal_types = ['z_dff_iso_baseline'] # 'dff_iso_baseline', 

bandit_alignments = [Align.cport_on, Align.cpoke_in, Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_reward, Align.cport_on_cpoke_in]
wm_alignments = bandit_alignments.copy()
wm_alignments.append(Align.tone)

regions = ['PL', 'NAc', 'DMS', 'DLS', 'TS']
default_xlims = {'PL': [-3,5], 'NAc': [-1,2], 'DMS': [-1,2], 'DLS': [-1,2], 'TS': [-1,2]}
xlims = {align: default_xlims.copy() for align in wm_alignments}
xlims[Align.reward]['PL'] = [-3,15]

# minimum number of bins to have time invariant intervals between task events 
min_norm_bins = 20
norm_dt = 0.005
pre_cue_dt = 0.5
pre_cport_on_dt = 0.5
post_reward_dt = 1.5
post_cpoke_in_dt = 0.5
max_cport_on_cpoke_in = 10

tasks = ['wm', 'bandit']
beh_names = {'wm': 'Single Tone WM', 'bandit': 'Probabilistic Bandit'}

filename = 'wm_bandit_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        aligned_signals = saved_data['aligned_signals']
        aligned_metadata = saved_data['metadata']
        
        recalculate = aligned_metadata['xlims'] != xlims

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    aligned_signals = {task: {subjid: {} for subjid in subj_ids}
                       for task in tasks}

# Build signal matrices aligned to alignment points
for task in tasks:
    if task == 'wm':
        sess_ids = wm_sess_ids 
        loc_db = wm_loc_db 
        sess_data = wm_sess_data
        alignments = wm_alignments
    else:
        sess_ids = bandit_sess_ids
        loc_db = bandit_loc_db
        sess_data = bandit_sess_data
        alignments = bandit_alignments
        
    # median time intervals across all subjects to determine size of time invariant signals
    cue_cpoke_out_dt = sess_data['cpoke_out_time'] - sess_data['response_cue_time']
    cpoke_out_resp_dt = sess_data['response_time'] - sess_data['cpoke_out_time']
    resp_reward_dt = sess_data['reward_time'] - sess_data['response_time']

    cport_on_cpoke_dt = sess_data['cpoke_in_time'] - sess_data['cport_on_time']

    med_poke_out_dt = np.median(cue_cpoke_out_dt[cue_cpoke_out_dt > 0])
    med_resp_dt = np.median(cpoke_out_resp_dt[cpoke_out_resp_dt > 0])
    med_rew_dt = np.median(resp_reward_dt[resp_reward_dt > 0])
    med_cpoke_dt = np.median(cport_on_cpoke_dt[(cport_on_cpoke_dt > 0) & (cport_on_cpoke_dt < max_cport_on_cpoke_in)])

    n_pre_cue_bins = max(np.ceil(pre_cue_dt/norm_dt), min_norm_bins)
    n_cue_poke_out_bins = max(np.ceil(med_poke_out_dt/norm_dt), min_norm_bins)
    n_poke_out_resp_bins = max(np.ceil(med_resp_dt/norm_dt), min_norm_bins)
    n_resp_rew_bins = max(np.ceil(med_rew_dt/norm_dt), min_norm_bins)
    n_post_rew_bins = max(np.ceil(post_reward_dt/norm_dt), min_norm_bins)
    
    n_pre_cport_on_bins = max(np.ceil(pre_cport_on_dt/norm_dt), min_norm_bins)
    n_cport_poke_in_bins = max(np.ceil(med_cpoke_dt/norm_dt), min_norm_bins)
    n_post_poke_in_bins = max(np.ceil(post_cpoke_in_dt/norm_dt), min_norm_bins)
                                    
    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue
        
        for sess_id in sess_ids[subj_id]:
            if sess_id in fpah.__sess_ignore:
                continue

            if sess_id in aligned_signals[task][subj_id] and not sess_id in reprocess_sess_ids and not task in recalculate_norm_tasks:
                continue
            else:
                aligned_signals[task][subj_id][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, filter_dropout_outliers=filter_dropout_outliers)
            fp_data = fp_data[subj_id][sess_id]
            dt = fp_data['dec_info']['decimated_dt']
            
            if show_process_plots or save_process_plots:
                fig = fpah.view_processed_signals(fp_data['processed_signals'], fp_data['time'], plot_baseline_corr=baseline_correction,
                                            title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id))

                if save_process_plots:
                    fpah.save_fig(fig, fpah.get_figure_save_path(beh_names[task], subj_id, 'sess_{}'.format(sess_id)))

                if show_process_plots:
                    plt.show()
                else:
                    plt.close(fig)
    
            start = time.perf_counter()
            
            trial_data = sess_data[sess_data['sessid'] == sess_id]
    
            ts = fp_data['time']
            trial_start_ts = fp_data['trial_start_ts'][:-1]
            cport_on_ts = trial_start_ts + trial_data['cport_on_time']
            cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
            cue_ts = trial_start_ts + trial_data['response_cue_time']
            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            resp_ts = trial_start_ts + trial_data['response_time']
            reward_ts = trial_start_ts + trial_data['reward_time']

            if task == 'wm':
                tone_start_ts = trial_start_ts + trial_data['abs_tone_start_times']
    
            for signal_type in signal_types:
                
                if signal_type in aligned_signals[task][subj_id][sess_id]:
                    continue
                else:
                    aligned_signals[task][subj_id][sess_id][signal_type] = {}
                    
                for align in alignments:
                    
                    norm_align = align in [Align.cport_on_cpoke_in, Align.cue_reward]
                    
                    if align in aligned_signals[task][subj_id][sess_id][signal_type] and not (task in recalculate_norm_tasks and norm_align):
                        continue
                    else:
                        aligned_signals[task][subj_id][sess_id][signal_type][align] = {}
                    
                    if norm_align:
                        
                        for region in fp_data['processed_signals'].keys():
                            if region in regions:
                                
                                if region in aligned_signals[task][subj_id][sess_id][signal_type][align]:
                                    continue
                                
                                signal = fp_utils.to_cupy(fp_data['processed_signals'][region][signal_type])
                                
                                match align:
                                    case Align.cport_on_cpoke_in:

                                        pre_cport_on = fp_utils.build_time_norm_signal_matrix(signal, ts, cport_on_ts-pre_cport_on_dt, cport_on_ts, n_pre_cport_on_bins)
                                        cport_on_cpoke_in = fp_utils.build_time_norm_signal_matrix(signal, ts, cport_on_ts, cpoke_in_ts, n_cport_poke_in_bins)
                                        post_cpoke_in = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_in_ts, cpoke_in_ts+post_cpoke_in_dt, n_post_poke_in_bins)
                                        
                                        mat = np.hstack([pre_cport_on, cport_on_cpoke_in, post_cpoke_in])

                                    case Align.cue_reward:
                                        pre_cue = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts-pre_cue_dt, cue_ts, n_pre_cue_bins)
                                        cue_poke_out = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, cpoke_out_ts, n_cue_poke_out_bins)
                                        poke_out_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, resp_ts, n_poke_out_resp_bins)
                                        resp_rew = fp_utils.build_time_norm_signal_matrix(signal, ts, resp_ts, reward_ts, n_resp_rew_bins)
                                        post_rew = fp_utils.build_time_norm_signal_matrix(signal, ts, reward_ts, reward_ts+post_reward_dt, n_post_rew_bins)
        
                                        mat = np.hstack([pre_cue, cue_poke_out, poke_out_resp, resp_rew, post_rew])

                                aligned_signals[task][subj_id][sess_id][signal_type][align][region] = mat
                        
                    else:
                        match align:
                            case Align.cport_on:
                                align_ts = cport_on_ts
                                mask_lims = None
                                
                            case Align.cpoke_in:
                                align_ts = cpoke_in_ts
                                mask_lims = None
                                
                            case Align.tone:
                                align_ts = tone_start_ts
                                mask_lims = None
                                
                            case Align.cue:
                                align_ts = cue_ts
                                mask_lims = None
                                # # mask out reward
                                # reward_ts_mask = reward_ts.to_numpy(copy=True)
                                # reward_ts_mask[~trial_data['rewarded']] = np.nan
                                # if np.isnan(reward_ts_mask[-1]):
                                #     reward_ts_mask[-1] = np.inf
                                # reward_ts_mask = pd.Series(reward_ts_mask).bfill().to_numpy()
                                
                                # mask_lims = np.hstack((np.full_like(align_ts, 0)[:, None], reward_ts_mask[:, None]))
                                
                            case Align.cpoke_out:
                                align_ts = cpoke_out_ts
                                mask_lims = None
                                
                            case Align.resp:
                                align_ts = resp_ts
                                mask_lims = None
                                
                            case Align.reward:
                                align_ts = reward_ts
                                mask_lims = None
                                # # mask out next reward
                                # next_reward_ts = reward_ts[1:].to_numpy(copy=True)
                                # next_reward_ts[~trial_data['rewarded'][1:]] = np.nan
                                # next_reward_ts = pd.Series(np.append(next_reward_ts, np.inf))
                                # next_reward_ts = next_reward_ts.bfill().to_numpy()
        
                                # mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], next_reward_ts[:, None]))
                        
                        for region in fp_data['processed_signals'].keys():
                            if region in regions:
                                
                                if region in aligned_signals[task][subj_id][sess_id][signal_type][align]:
                                    continue
                                
                                signal = fp_data['processed_signals'][region][signal_type]
        
                                lims = xlims[align][region]
                                
                                mat, t = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)
                                aligned_signals[task][subj_id][sess_id][signal_type][align][region] = mat
                                
            print('Stacked FP data for subject {} session {} in {:.1f} s'.format(subj_id, sess_id, time.perf_counter()-start))
    
    if not 't' in aligned_signals[task] or task in recalculate_norm_tasks:
        aligned_signals[task]['t'] = {align: {region: [] for region in regions} for align in alignments}
        for align in alignments:
            for region in regions:
                if align == Align.cport_on_cpoke_in:
                    aligned_signals[task]['t'][align][region] = np.arange(0, n_pre_cport_on_bins+n_cport_poke_in_bins+n_post_poke_in_bins, 1)
                elif align == Align.cue_reward:
                    aligned_signals[task]['t'][align][region] = np.arange(0, n_pre_cue_bins+n_cue_poke_out_bins+
                                                                          n_poke_out_resp_bins+n_resp_rew_bins+n_post_rew_bins, 1)
                else:
                    aligned_signals[task]['t'][align][region] = np.arange(xlims[align][region][0], xlims[align][region][1]+dt, dt)
    
    # persist median bin numbers for plotting
    if not Align.cport_on_cpoke_in in aligned_signals[task] or task in recalculate_norm_tasks:
        aligned_signals[task][Align.cport_on_cpoke_in] = {'cport_on': n_pre_cport_on_bins, 
                                                          'cpoke_in': n_pre_cport_on_bins+n_cport_poke_in_bins}
        
    if not Align.cue_reward in aligned_signals[task] or task in recalculate_norm_tasks:
        aligned_signals[task][Align.cue_reward] = {'cue': n_pre_cue_bins, 
                                                   'poke_out': n_pre_cue_bins+n_cue_poke_out_bins,
                                                   'response': n_pre_cue_bins+n_cue_poke_out_bins+n_poke_out_resp_bins,
                                                   'reward': n_pre_cue_bins+n_cue_poke_out_bins+n_poke_out_resp_bins+n_resp_rew_bins}

with open(save_path, 'wb') as f:
    pickle.dump({'aligned_signals': aligned_signals,
                 'metadata': {'signal_types': signal_types,
                             'alignments': alignments,
                             'regions': regions,
                             'xlims': xlims}}, f)

# %% Construct selection vectors for various trial groupings

align_trial_selections = {task: {subjid: {} for subjid in subj_ids} for task in tasks}

rel_sides = ['contra', 'ipsi']
abs_sides = ['left', 'right']

for task in tasks:
    if task == 'wm':
        sess_ids = wm_sess_ids 
        sess_data = wm_sess_data
        alignments = wm_alignments.copy()
        
        resp_delays = np.unique(sess_data['resp_delay_bin'])
        tone_types = np.unique(sess_data['relevant_tone_info'])
        
        # get tone side mapping
        tone_ports = sess_data[['subjid', 'correct_port', 'relevant_tone_info']].drop_duplicates()
        tone_ports = {i: tone_ports[tone_ports['subjid'] == i].set_index('relevant_tone_info')['correct_port'].to_dict() for i in sess_ids.keys()}
        
    else:
        sess_ids = bandit_sess_ids
        sess_data = bandit_sess_data
        alignments = bandit_alignments.copy()
        
    #alignments.append(Align.early_cpoke_in)
    
    if len(sess_data) == 0:
        continue
    
    cpoke_durs = np.unique(sess_data['cpoke_dur_bin'])
    
    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue
        
        subj_regions = list(implant_info[subj_id].keys())
        
        for sess_id in sess_ids[subj_id]:
            if not sess_id in align_trial_selections[task][subj_id]:
                align_trial_selections[task][subj_id][sess_id] = {}
                
            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_started = trial_data['trial_started']

            # create various trial selection criteria
            response = ~np.isnan(trial_data['response_time']).to_numpy()
            reward = (trial_data['rewarded'] == True).to_numpy()
            unreward = (trial_data['rewarded'] == False).to_numpy() & response

            prev_resp = np.insert(response[:-1], 0, False)
            
            prev_reward = np.insert(reward[:-1], 0, False)
            prev_unreward = np.insert(unreward[:-1], 0, False)
            
            next_resp = np.append(response[1:], False)

            rew_hist = pd.cut(trial_data['n_rew_hist'], rew_hist_bins)

            choices = trial_data['choice']
            prev_choices = trial_data['prev_choice']
            
            switch = trial_data['switch'].to_numpy() & prev_resp & response
            stay = trial_data['stay'].to_numpy() & prev_resp & response
            next_switch = trial_data['next_switch'].to_numpy() & response & next_resp
            next_stay = trial_data['next_stay'].to_numpy() & response & next_resp
            
            if task == 'wm':
                bail = (trial_data['bail'] == True).to_numpy()
                
                tones = trial_data['relevant_tone_info'].to_numpy()
                prev_tones = trial_data['prev_tone_info'].apply(lambda x: x[0] if utils.is_list(x) else x).to_numpy()
                next_tones = trial_data['next_tone_info'].apply(lambda x: x[0] if utils.is_list(x) else x).to_numpy()
                
                delays = trial_data['resp_delay_bin'].to_numpy()

                prev_trial_same = tones == prev_tones
                prev_trial_diff = (tones != prev_tones) & (prev_tones != None)
                
                next_trial_same = tones == next_tones
                next_trial_diff = (tones != next_tones) & (next_tones != None)
                
                tone_heard_sel = ~np.isnan(trial_data['abs_tone_start_times']).to_numpy()
                prev_tone_heard_sel = np.insert(tone_heard_sel[:-1], 0, False)
            
            if task == 'bandit':
                bail = np.full_like(response, False)

            prev_bail = np.insert(bail[:-1], 0, False)
            next_bail = np.append(bail[1:], False)

            # ignore cport on trials where they were poked before cport turned on
            cport_on_sel = (trial_data['cpoke_in_latency'] > 0.1).to_numpy()
            early_cpoke_in_sel = (trial_data['cpoke_in_latency'] < 0).to_numpy()
            norm_cpoke_in_sel = (trial_data['cpoke_in_latency'] > 0).to_numpy()
            engage_cpoke_in_sel = (trial_data['cpoke_in_latency'] < max_cport_on_cpoke_in).to_numpy()
            early_cpoke_out_sel = (trial_data['cpoke_out_latency'] < 0).to_numpy()
            norm_cpoke_out_sel = (trial_data['cpoke_out_latency'] > 0).to_numpy()
                
            for region in subj_regions:
                if not region in align_trial_selections[task][subj_id][sess_id]:
                    align_trial_selections[task][subj_id][sess_id][region] = {}
                    
                region_side = implant_info[subj_id][region]['side']
                rel_choice_side = choices.apply(lambda x: fpah.get_implant_rel_side(x, region_side)).to_numpy().copy()
                rel_prev_choice_side = prev_choices.apply(lambda x: fpah.get_implant_rel_side(x, region_side)).to_numpy().copy()
                
                # get all trial selection vectors for different trial groupings
                align_trial_selections[task][subj_id][sess_id][region]['prev_resp'] = prev_resp
                align_trial_selections[task][subj_id][sess_id][region]['prev_bail'] = prev_bail
                
                align_trial_selections[task][subj_id][sess_id][region]['bail'] = bail
                align_trial_selections[task][subj_id][sess_id][region]['response'] = response
                
                align_trial_selections[task][subj_id][sess_id][region]['stay'] = stay 
                align_trial_selections[task][subj_id][sess_id][region]['switch'] = switch
                align_trial_selections[task][subj_id][sess_id][region]['next_stay'] = next_stay 
                align_trial_selections[task][subj_id][sess_id][region]['next_switch'] = next_switch 
                
                align_trial_selections[task][subj_id][sess_id][region]['prev_reward'] = prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['prev_unreward'] = prev_unreward
                # need to have different name for reward so it doesn't get overwritten by Align.reward
                align_trial_selections[task][subj_id][sess_id][region]['rewarded'] = reward
                align_trial_selections[task][subj_id][sess_id][region]['unrewarded'] = unreward
                
                align_trial_selections[task][subj_id][sess_id][region]['reward_prev_reward'] = reward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['reward_prev_unreward'] = reward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['unreward_prev_reward'] = unreward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['unreward_prev_unreward'] = unreward & prev_unreward
                
                align_trial_selections[task][subj_id][sess_id][region]['stay_prev_reward'] = stay & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_prev_reward'] = switch & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_prev_unreward'] = stay & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_prev_unreward'] = switch & prev_unreward 
                
                align_trial_selections[task][subj_id][sess_id][region]['stay_reward'] = stay & reward 
                align_trial_selections[task][subj_id][sess_id][region]['switch_reward'] = switch & reward 
                align_trial_selections[task][subj_id][sess_id][region]['stay_unreward'] = stay & unreward 
                align_trial_selections[task][subj_id][sess_id][region]['switch_unreward'] = switch & unreward 
                
                align_trial_selections[task][subj_id][sess_id][region]['stay_reward_prev_reward'] = stay & reward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_reward_prev_reward'] = switch & reward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_unreward_prev_reward'] = stay & unreward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_unreward_prev_reward'] = switch & unreward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_reward_prev_unreward'] = stay & reward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_reward_prev_unreward'] = switch & reward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['stay_unreward_prev_unreward'] = stay & unreward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_unreward_prev_unreward'] = switch & unreward & prev_unreward

                align_trial_selections[task][subj_id][sess_id][region]['reward_next_stay'] = next_stay & reward 
                align_trial_selections[task][subj_id][sess_id][region]['reward_next_switch'] = next_switch & reward 
                align_trial_selections[task][subj_id][sess_id][region]['unreward_next_stay'] = next_stay & unreward 
                align_trial_selections[task][subj_id][sess_id][region]['unreward_next_switch'] = next_switch & unreward 

                for rel_side in rel_sides:
                    side_sel = rel_choice_side == rel_side
                    prev_side_sel = rel_prev_choice_side == rel_side
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side] = side_sel 
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side] = prev_side_sel & prev_resp 
                    
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_prev_reward'] = prev_side_sel & prev_reward 
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_prev_unreward'] = prev_side_sel & prev_unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_prev_bail'] = prev_side_sel & prev_bail 
                    
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_bail'] = prev_side_sel & bail 
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_response'] = prev_side_sel & response 
                    
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_stay'] = prev_side_sel & stay 
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_switch'] = prev_side_sel & switch 
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_prev_reward'] = side_sel & prev_reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_prev_unreward'] = side_sel & prev_unreward 
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward'] = side_sel & reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward'] = side_sel & unreward 
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay'] = side_sel & stay 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch'] = side_sel & switch 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_next_stay'] = side_sel & next_stay 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_next_switch'] = side_sel & next_switch 
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_prev_reward'] = side_sel & stay & prev_reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_prev_reward'] = side_sel & switch & prev_reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_prev_unreward'] = side_sel & stay & prev_unreward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_prev_unreward'] = side_sel & switch & prev_unreward 
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_prev_reward'] = side_sel & reward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_prev_unreward'] = side_sel & reward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_prev_reward'] = side_sel & unreward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_prev_unreward'] = side_sel & unreward & prev_unreward
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_reward'] = side_sel & stay & reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_reward'] = side_sel & switch & reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_unreward'] = side_sel & stay & unreward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_unreward'] = side_sel & switch & unreward 
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_reward_prev_reward'] = side_sel & stay & reward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_reward_prev_reward'] = side_sel & switch & reward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_unreward_prev_reward'] = side_sel & stay & unreward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_unreward_prev_reward'] = side_sel & switch & unreward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_reward_prev_unreward'] = side_sel & stay & reward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_reward_prev_unreward'] = side_sel & switch & reward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_unreward_prev_unreward'] = side_sel & stay & unreward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_unreward_prev_unreward'] = side_sel & switch & unreward & prev_unreward
                    
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_next_stay'] = side_sel & next_stay & reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_next_switch'] = side_sel & next_switch & reward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_next_stay'] = side_sel & next_stay & unreward 
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_next_switch'] = side_sel & next_switch & unreward 
                    
                for rew_bin in rew_hist_bins:
                    rew_sel = rew_hist == rew_bin
                    bin_str = rew_hist_bin_strs[rew_bin]
                    
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str] = rew_sel 
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_reward'] = rew_sel & reward
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_unreward'] = rew_sel & unreward
                    
                    for rel_side in rel_sides:
                        side_sel = rel_choice_side == rel_side
                        prev_side_sel = rel_prev_choice_side == rel_side
                        
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_'+rel_side] = rew_sel & side_sel 
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_prev_'+rel_side] = rew_sel & prev_side_sel 
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_'+rel_side+'_reward'] = rew_sel & side_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_'+rel_side+'_unreward'] = rew_sel & side_sel & unreward

                if task == 'wm':
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_resp'] = prev_trial_same & response & prev_resp
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_resp'] = prev_trial_diff & response & prev_resp
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_resp_prev_reward'] = prev_trial_same & response & prev_resp & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_resp_prev_unreward'] = prev_trial_same & response & prev_resp & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_resp_prev_reward'] = prev_trial_diff & response & prev_resp & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_resp_prev_unreward'] = prev_trial_diff & response & prev_resp & prev_unreward
                    
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_prev_reward'] = prev_trial_same & stay & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_prev_reward'] = prev_trial_same & switch & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_prev_unreward'] = prev_trial_same & stay & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_prev_unreward'] = prev_trial_same & switch & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_prev_reward'] = prev_trial_diff & stay & prev_reward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_prev_reward'] = prev_trial_diff & switch & prev_reward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_prev_unreward'] = prev_trial_diff & stay & prev_unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_prev_unreward'] = prev_trial_diff & switch & prev_unreward 
                    
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_reward'] = prev_trial_same & stay & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_reward'] = prev_trial_same & switch & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_unreward'] = prev_trial_same & stay & unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_unreward'] = prev_trial_same & switch & unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_reward'] = prev_trial_diff & stay & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_reward'] = prev_trial_diff & switch & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_unreward'] = prev_trial_diff & stay & unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_unreward'] = prev_trial_diff & switch & unreward
                    
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_stay_reward'] = next_trial_same & next_stay & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_switch_reward'] = next_trial_same & next_switch & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_stay_unreward'] = next_trial_same & next_stay & unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_switch_unreward'] = next_trial_same & next_switch & unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_stay_reward'] = next_trial_diff & next_stay & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_switch_reward'] = next_trial_diff & next_switch & reward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_stay_unreward'] = next_trial_diff & next_stay & unreward 
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_switch_unreward'] = next_trial_diff & next_switch & unreward
                    
                    for resp_delay in resp_delays:
                        delay_sel = delays == resp_delay
                        align_trial_selections[task][subj_id][sess_id][region]['delay_'+resp_delay] = delay_sel 
                        align_trial_selections[task][subj_id][sess_id][region]['delay_'+resp_delay+'_reward'] = delay_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region]['delay_'+resp_delay+'_unreward'] = delay_sel & unreward
                        
                        for rel_side in rel_sides:
                            side_sel = rel_choice_side == rel_side
                            align_trial_selections[task][subj_id][sess_id][region][rel_side+'_delay_'+resp_delay] = side_sel & delay_sel 
                            align_trial_selections[task][subj_id][sess_id][region][rel_side+'_delay_'+resp_delay+'_reward'] = side_sel & delay_sel & reward
                            align_trial_selections[task][subj_id][sess_id][region][rel_side+'_delay_'+resp_delay+'_unreward'] = side_sel & delay_sel & unreward 
                    
                    for tone in tone_types:
                        tone_abs_side = tone_ports[subj_id][tone]
                        tone_rel_side = fpah.get_implant_rel_side(tone_ports[subj_id][tone], region_side)
                        tone_sel = tones == tone
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_response'] = tone_sel & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_response'] = tone_sel & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_response'] = tone_sel & response
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_bail'] = tone_sel & bail
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_bail'] = tone_sel & bail
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_bail'] = tone_sel & bail
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_reward'] = tone_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_reward'] = tone_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward'] = tone_sel & reward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_unreward'] = tone_sel & unreward & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_unreward'] = tone_sel & unreward & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward'] = tone_sel & unreward & response
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_reward'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_unreward'] = tone_sel & prev_unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward_prev_reward'] = tone_sel & prev_reward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward_prev_unreward'] = tone_sel & prev_unreward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward_prev_reward'] = tone_sel & prev_reward & unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward_prev_unreward'] = tone_sel & prev_unreward & unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp'] = tone_sel & prev_trial_same & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp'] = tone_sel & prev_trial_diff & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp_prev_reward'] = tone_sel & prev_trial_same & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp_prev_unreward'] = tone_sel & prev_trial_same & response & prev_resp & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp_prev_reward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp_prev_unreward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_unreward
                        

                # get alignment specific selection vectors
                for align in alignments:
                    match align:
                        case Align.cport_on:
                            align_trial_selections[task][subj_id][sess_id][region][align] = cport_on_sel & trial_started

                        # case Align.early_cpoke_in:
                        #     align_trial_selections[task][subj_id][sess_id][region][align] = early_cpoke_in_sel
                            
                        case Align.cpoke_in:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_in_sel | early_cpoke_in_sel
                           
                        case Align.tone:
                            align_trial_selections[task][subj_id][sess_id][region][align] = tone_heard_sel
                            
                        case Align.cue | Align.cpoke_out:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_out_sel
                            
                        case Align.resp | Align.reward:
                            align_trial_selections[task][subj_id][sess_id][region][align] = response
                            
                        case Align.cue_reward:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_out_sel & response
                            
                        case Align.cport_on_cpoke_in:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_in_sel & engage_cpoke_in_sel


# %% Set up average plot options

# modify these options to change what will be used in the average signal plots
plot_signals = ['z_dff_iso_baseline'] # 'z_dff_iso', 
plot_regions = ['PL', 'DMS', 'DLS', 'TS']
plot_tasks = ['wm', 'bandit'] #
plot_meta_subj = False
    
use_se = True
ph = 3.5;
pw = 5;
n_reg = len(plot_regions)

rew_xlims = {'DMS': [-1,2], 'DLS': [-1,2], 'TS': [-1,2], 'PL': [-3,10]}
gen_xlims = {'DMS': [-1,1.5], 'DLS': [-1,1.5], 'TS': [-1,1.5], 'PL': [-3,3]}

all_xlims = {Align.cport_on: gen_xlims, Align.early_cpoke_in: gen_xlims, Align.cpoke_in: gen_xlims, Align.tone: gen_xlims,
            Align.cue: gen_xlims, Align.cpoke_out: gen_xlims, Align.resp: gen_xlims, Align.reward: rew_xlims,
            Align.cue_reward: None, Align.cport_on_cpoke_in: None}

tone_dashlines = [0.4]
norm_keys = {Align.cport_on_cpoke_in: ['cport_on', 'cpoke_in'], Align.cue_reward: ['cue', 'poke_out', 'response', 'reward']}
norm_labels = {Align.cport_on_cpoke_in: ['Cport On', 'Poke In'], Align.cue_reward: ['Cue', 'Out', 'Resp', 'Rew']}
norm_dashlines = {task: {align: [aligned_signals[task][align][a] for a in norm_keys[align]] for align in [Align.cport_on_cpoke_in, Align.cue_reward]} for task in plot_tasks}

save_plots = True
show_plots = True
tone_end = 0.4

# declare method to stack data matrices for the given task, alignment, and selection grouping
def stack_mats(task, align, trial_groups, signal_types=plot_signals):

    if not utils.is_list(signal_types):
        signal_types = signal_types
        
    stacked_mat_dict = {sig: {region: {group: {} for group in trial_groups} for region in plot_regions} for sig in signal_types} 

    for signal_type in signal_types:
            for region in plot_regions:
                for group in trial_groups:
                    for subj_id in subj_ids:
                        
                        stacked_mats = []
                        
                        sess_ids = list(align_trial_selections[task][subj_id])
                        for sess_id in sess_ids:
                            
                            if align == Align.early_cpoke_in:
                                signal_align = Align.cpoke_in
                            else:
                                signal_align = align
                            
                            # some sessions have different regions
                            if not region in aligned_signals[task][subj_id][sess_id][signal_type][signal_align]:
                                continue
                        
                            # get trial selections
                            trial_sel = align_trial_selections[task][subj_id][sess_id][region][group] & align_trial_selections[task][subj_id][sess_id][region][align]
                            
                            # get aligned data matrix
                            aligned_mat = aligned_signals[task][subj_id][sess_id][signal_type][signal_align][region]

                            stacked_mats.append(aligned_mat[trial_sel, :])
                            
                        if len(stacked_mats) > 0:
                            stacked_mat_dict[signal_type][region][group][subj_id] = np.vstack(stacked_mats)

                    if plot_meta_subj:
                        stacked_mat_dict[signal_type][region][group]['all'] = np.vstack([stacked_mat_dict[signal_type][region][group][subj_id] for subj_id in subj_ids])

    return stacked_mat_dict

# declare method to save plots
def save_plot(fig, task, subjects, plot_name):
    if save_plots and not plot_name is None:
        fpah.save_fig(fig, fpah.get_figure_save_path(beh_names[task], subjects, plot_name))

    if not show_plots:
        plt.close(fig)
        
def reformat_stacked_signals(stacked_signals, subj):
    reformat_signals = {}
    for region, groups in stacked_signals.items():
        for group, subjects in groups.items():
            if subj in subjects:
                reformat_signals.setdefault(region, {})[group] = subjects[subj]
                
    return reformat_signals

# declare method to plot avg signals for the given plot_groups for one or more tasks, signal types, and alignments
def plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, subjects=subj_ids, tasks=plot_tasks, signal_types=plot_signals, regions=plot_regions,
                     xlims_dict=None, dashlines=None, legend_params=None, group_colors=None, gen_plot_name=None, include_norm_reward=True):

    if not utils.is_list(aligns):
        aligns = [aligns]
        
    if not utils.is_list(tasks):
        tasks = [tasks]
        
    if not utils.is_list(signal_types):
        signal_types = [signal_types]
        
    if xlims_dict is None:
        xlims_dict = all_xlims
        
    subjects = list(subjects)
        
    if plot_meta_subj:
        subjects.append('all')
        
    all_groups = np.unique(utils.flatten(plot_groups))
        
    for task in tasks:
        for align in aligns:
            align_title = fpah.get_align_title(align)
            x_label = fpah.get_align_xlabel(align)
            
            align_xlims = all_xlims[align]
            
            if align == Align.early_cpoke_in:
                t = aligned_signals[task]['t'][Align.cpoke_in]
            else:
                t = aligned_signals[task]['t'][align]
                
            stacked_signals = stack_mats(task, align, all_groups, signal_types=signal_types)

            x_ticks = None
            plot_x0 = True
            dashlines = None
            
            if align == Align.tone:
                dashlines = tone_dashlines
            elif align in [Align.cport_on_cpoke_in, Align.cue_reward]:
                plot_x0 = False
                dashlines = norm_dashlines[task][align]
                x_ticks = {'ticks': dashlines, 'labels': norm_labels[align]}
                
                if align == Align.cue_reward and not include_norm_reward:
                    align_xlims = {r: [0, dashlines[-1]] for r in plot_regions}
            
            for signal_type in signal_types:
                _, y_label = fpah.get_signal_type_labels(signal_type)
                
                for subj in subjects:
                    
                    # reformat the stacked signals for each subject
                    plot_stacked_signals = reformat_stacked_signals(stacked_signals[signal_type], subj)

                    title = '{} Aligned to {} - Subj {}, {}'.format(gen_title, align_title, subj, beh_names[task])
                    if subj == 'all':
                        implant_side_info = None
                    else:
                        implant_side_info = implant_info[subj]
                        
                    fig, plotted = fpah.plot_avg_signals(plot_groups, group_labels, plot_stacked_signals, regions, t, 
                                                         title, plot_titles, x_label, y_label, 
                                                         xlims_dict=align_xlims, implant_info=implant_side_info,
                                                         dashlines=dashlines, legend_params=legend_params, group_colors=group_colors, 
                                                         use_se=use_se, ph=ph, pw=pw, x_ticks=x_ticks, plot_x0=plot_x0)
                
                    if plotted and not gen_plot_name is None:
                        save_plot(fig, task, subj, '{}_{}_{}'.format(align, gen_plot_name, signal_type))
                
                    if not plotted:
                        plt.close(fig)
                    else:
                        plt.show()


# %% Choice, side, and prior reward groupings for multiple alignment points

plot_groups = [['contra', 'ipsi'], ['stay', 'switch'], ['contra_stay', 'contra_switch', 'ipsi_stay', 'ipsi_switch']]
group_labels = {'stay': 'Stay', 'switch': 'Switch',
                'ipsi': 'Ipsi', 'contra': 'Contra',
                'contra_stay': 'Contra Stay', 'contra_switch': 'Contra Switch',
                'ipsi_stay': 'Ipsi Stay', 'ipsi_switch': 'Ipsi Switch'}

plot_titles = ['Choice Side', 'Stay/Switch', 'Stay/Switch & Side']
gen_title = 'Choice Side & Stay/Switch Groupings'
gen_plot_name = 'stay_switch_side'

aligns = [Align.cport_on_cpoke_in, Align.cue_reward] # Align.cue, Align.cpoke_out, Align.resp, 

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, include_norm_reward=False)


# %% choice side and outcome
plot_groups = [['contra', 'ipsi'],
               ['rewarded', 'unrewarded'],
               ['contra_reward', 'contra_unreward', 'ipsi_reward', 'ipsi_unreward']]
group_labels = {'contra': 'Contra', 'ipsi': 'Ipsi',
                'rewarded': 'Rewarded', 'unrewarded': 'Unrewarded',
                'contra_reward': 'Contra Rewarded', 'ipsi_reward': 'Ipsi Rewarded',
                'contra_unreward': 'Contra Unrewarded', 'ipsi_unreward': 'Ipsi Unrewarded'}

plot_titles = ['Choice Side', 'Outcome', 'Choice Side/Outcome']
gen_title = 'Choice Side by Outcome'
gen_plot_name = 'side_outcome'

aligns = [Align.reward, Align.cue_reward] # Align.cue, Align.cpoke_out, Align.resp, 

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)


# # %% prior & future outcome
# plot_groups = [['prev_reward', 'prev_unreward', 'prev_bail'], ['reward', 'unreward', 'bail']]
# group_labels = {'prev_reward': 'Prev Reward', 'prev_unreward': 'Prev Unreward', 'prev_bail': 'Prev Bail',
#                 'reward': 'Rewarded', 'unreward': 'Unrewarded', 'bail': 'Bail'}

# plot_titles = ['Prior Trial Outcome', 'Current Trial Outcome']
# gen_title = 'Previous and Current Outcome'
# gen_plot_name = 'prev_future_outcome'

# aligns = [Align.cport_on, Align.early_cpoke_in, Align.cpoke_in, Align.cue, Align.cpoke_out]

# plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)

# %% prior choice side/outcome

plot_groups = [['prev_reward', 'prev_unreward'], ['prev_contra', 'prev_ipsi'], ['prev_contra_prev_reward', 'prev_contra_prev_unreward', 'prev_ipsi_prev_reward', 'prev_ipsi_prev_unreward']]
group_labels = {'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi',
                'prev_reward': 'Prev Reward', 'prev_unreward': 'Prev Unreward',
                'prev_contra_prev_reward': 'Prev Contra Reward', 'prev_contra_prev_unreward': 'Prev Contra Unreward',
                'prev_ipsi_prev_reward': 'Prev Ipsi Reward', 'prev_ipsi_prev_unreward': 'Prev Ipsi Unreward'}

plot_titles = ['Prior Outcome', 'Prior Choice Side', 'Prior Choice Side & Outcome']
gen_title = 'Previous Outcome and Choice Side'
gen_plot_name = 'prev_side_prev_outcome'

aligns = [Align.cport_on_cpoke_in]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)


# %% prior reward/choice/side

plot_groups = [['stay_prev_reward', 'switch_prev_reward', 'stay_prev_unreward', 'switch_prev_unreward'],
               ['contra_prev_reward', 'ipsi_prev_reward', 'contra_prev_unreward', 'ipsi_prev_unreward'],
               ['contra_stay_prev_reward', 'contra_switch_prev_reward', 'ipsi_stay_prev_reward', 'ipsi_switch_prev_reward'],
               ['contra_stay_prev_unreward', 'contra_switch_prev_unreward', 'ipsi_stay_prev_unreward', 'ipsi_switch_prev_unreward']]

group_labels = {'stay_prev_reward': 'Stay | Reward', 'switch_prev_reward': 'Switch | Reward',
                'stay_prev_unreward': 'Stay | Unreward', 'switch_prev_unreward': 'Switch | Unreward',
                'contra_prev_reward': 'Contra | Reward', 'ipsi_prev_reward': 'Ipsi | Reward',
                'contra_prev_unreward': 'Contra | Unreward', 'ipsi_prev_unreward': 'Ipsi | Unreward',
                'contra_stay_prev_reward': 'Contra Stay', 'contra_switch_prev_reward': 'Contra Switch',
                'ipsi_stay_prev_reward': 'Ipsi Stay', 'ipsi_switch_prev_reward': 'Ipsi Switch',
                'contra_stay_prev_unreward': 'Contra Stay', 'contra_switch_prev_unreward': 'Contra Switch',
                'ipsi_stay_prev_unreward': 'Ipsi Stay', 'ipsi_switch_prev_unreward': 'Ipsi Switch'}

plot_titles = ['Stay/Switch by Prior Outcome', 'Choice Side by Prior Outcome', 'Prior Reward', 'Prior Unreward']
gen_title = 'Previous Outcome by Stay/Switch and Choice Side'
gen_plot_name = 'prev_outcome_stay_switch_side'

aligns = [Align.cport_on_cpoke_in, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, include_norm_reward=False)

# %% Outcome, future stay/switch

plot_groups = [['reward_next_stay', 'reward_next_switch', 'unreward_next_stay', 'unreward_next_switch'],
               ['contra_reward_next_stay', 'contra_reward_next_switch', 'contra_unreward_next_stay', 'contra_unreward_next_switch'],
               ['ipsi_reward_next_stay', 'ipsi_reward_next_switch', 'ipsi_unreward_next_stay', 'ipsi_unreward_next_switch']]

group_labels = {'reward_next_stay': 'Reward Next Stay', 'reward_next_switch': 'Reward Next Switch',
                'unreward_next_stay': 'Unreward Next Stay', 'unreward_next_switch': 'Unreward Next Switch',
                'contra_reward_next_stay': 'Reward Next Stay', 'contra_reward_next_switch': 'Reward Next Switch',
                'contra_unreward_next_stay': 'Unreward Next Stay', 'contra_unreward_next_switch': 'Unreward Next Switch',
                'ipsi_reward_next_stay': 'Reward Next Stay', 'ipsi_reward_next_switch': 'Reward Next Switch',
                'ipsi_unreward_next_stay': 'Unreward Next Stay', 'ipsi_unreward_next_switch': 'Unreward Next Switch'}

plot_titles = ['Current Outcome and Future Choice', 'Contra Choice Outcome and Future Choice', 'Ipsi Choice Outcome and Future Choice']
gen_title = 'Current Outcome & Choice by Next Choice'
gen_plot_name = 'outcome_side_next_stay_switch'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)

# %% Reward History

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

gen_group_labels = {'rew_hist_{}': '{}', 'rew_hist_{}_reward': '{} Rew', 'rew_hist_{}_unreward': '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin]) 
                for k,v in gen_group_labels.items() 
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_{}_{}': '{}', 'rew_hist_{}_prev_{}': '{}', 'rew_hist_{}_{}_reward': '{} Rew', 'rew_hist_{}_{}_unreward': '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin]) 
                for k,v in gen_group_labels.items() 
                for rew_bin in rew_hist_bins
                for s in rel_sides})

# Reward History by Choice Side

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

# previous choice side
gen_plot_group = ['rew_hist_{}', 'rew_hist_{}_prev_contra', 'rew_hist_{}_prev_ipsi']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Prev Choices', 'Previous Contra Choices', 'Previous Ipsi Choices']
gen_title = 'Reward History by Previous Choice Side'
gen_plot_name = 'rew_hist_prev_side'

aligns = [Align.cport_on_cpoke_in]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=rew_hist_all_colors)

# current choice side
gen_plot_group = ['rew_hist_{}', 'rew_hist_{}_contra', 'rew_hist_{}_ipsi']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History by Choice Side'
gen_plot_name = 'rew_hist_side'

aligns = [Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=rew_hist_all_colors, include_norm_reward=False)

# choice side & outcome

gen_plot_group = ['rew_hist_{}_reward', 'rew_hist_{}_unreward']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins]]

gen_plot_group = ['rew_hist_{}_{}_reward', 'rew_hist_{}_{}_unreward']
plot_groups.extend([[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides])

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History By Choice Side & Outcome'
gen_plot_name = 'rew_hist_side_outcome'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), legend_params={'ncol': 2})


# %% WM Response Delay

task = 'wm'
delay_colors = plt.cm.Oranges(np.linspace(0.4,1,len(resp_delays)))
delay_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(resp_delays)))
delay_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(resp_delays)))

# choice sides

gen_group_labels = {'delay_{}': '{}', 'contra_delay_{}': '{}', 'ipsi_delay_{}': '{}', 
                    'delay_{}_reward': '{} Rew', 'delay_{}_unreward': '{} Unrew'}

group_labels = {k.format(delay): v.format(delay) 
                for k,v in gen_group_labels.items() 
                for delay in resp_delays}

gen_group_labels = {'{}_delay_{}_reward': '{} Rew', '{}_delay_{}_unreward': '{} Unrew'}

group_labels.update({k.format(s, delay): v.format(delay) 
                for k,v in gen_group_labels.items() 
                for delay in resp_delays
                for s in rel_sides})

gen_plot_group = ['delay_{}', 'contra_delay_{}', 'ipsi_delay_{}']
plot_groups = [[g.format(delay) for delay in resp_delays] for g in gen_plot_group]

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Response Delays by Choice Side'
gen_plot_name = 'resp_delay_side'

aligns = [Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=delay_colors, tasks=task, include_norm_reward=False)


# outcomes and choices sides

gen_plot_group = ['delay_{}_reward', 'delay_{}_unreward']
plot_groups = [[g.format(delay) for g in gen_plot_group for delay in resp_delays]]

gen_plot_group = ['{}_delay_{}_reward', '{}_delay_{}_unreward']
plot_groups.extend([[g.format(s, delay) for g in gen_plot_group for delay in resp_delays] for s in rel_sides])

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Response Delays By Choice Side & Outcome'
gen_plot_name = 'resp_delay_side_outcome'

aligns = [Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=np.vstack((delay_rew_colors, delay_unrew_colors)), tasks=task, legend_params={'ncol': 2})

# plot again without the reward
gen_plot_name = 'resp_delay_side_outcome_no_rew'
plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=np.vstack((delay_rew_colors, delay_unrew_colors)), tasks=task, include_norm_reward=False, legend_params={'ncol': 2})


# %% WM Tones

task = 'wm'

gen_group_labels = {'{}_tone_response': '{} Resp', '{}_tone_bail': '{} Bail', '{}_tone_reward': '{} Rew', '{}_tone_unreward': '{} Unrew'}

gen_plot_groups = [['{}_tone_response', '{}_tone_bail'], ['{}_tone_reward', '{}_tone_unreward']]

# tone types
group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in tone_types}

plot_groups = [[g.format(tone) for tone in tone_types for g in gs] for gs in gen_plot_groups]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Tone Types by Response and Outcome'
gen_plot_name = 'tone_type_resp_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# tone correct relative side
group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in rel_sides}

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Tone Correct Relative Sides by Response and Outcome'
gen_plot_name = 'tone_rel_side_resp_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# tone correct relative side
group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in abs_sides}

plot_groups = [[g.format(tone) for tone in abs_sides for g in gs] for gs in gen_plot_groups]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Tone Correct Absolute Sides by Response and Outcome'
gen_plot_name = 'tone_abs_side_resp_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# %% Prev/Next stimulus by prev/next choice and current outcome

task = 'wm'

# consecutive stimuli by prior outcome, all side choices
group_labels = {'same_tone_resp': 'Same', 'diff_tone_resp': 'Diff', 
                'prev_reward': 'Prev Rew', 'prev_unreward': 'Prev Unrew',
                'same_tone_resp_prev_reward': 'Same Prev Rew', 'same_tone_resp_prev_unreward': 'Same Prev Unrew',
                'diff_tone_resp_prev_reward': 'Diff Prev Rew', 'diff_tone_resp_prev_unreward': 'Diff Prev Unrew'}

plot_groups = [['same_tone_resp', 'diff_tone_resp'], 
               ['prev_reward', 'prev_unreward'],
               ['same_tone_resp_prev_reward', 'same_tone_resp_prev_unreward', 'diff_tone_resp_prev_reward', 'diff_tone_resp_prev_unreward']]

plot_titles = ['Prior & Current Tone', 'Prior Outcome', 'Prior & Current Tone by Prev Outcome']
gen_title = 'Consecutive Trial Tones by Previous Outcome'
gen_plot_name = 'consec_tones_prev_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# previous and current outcome
group_labels = {'prev_reward': 'Prev Rew', 'prev_unreward': 'Prev Unrew',
                'reward_prev_reward': 'Rew Prev Rew', 'reward_prev_unreward': 'Rew Prev Unrew', 
                'unreward_prev_reward': 'Unrew Prev Rew', 'unreward_prev_unreward': 'Unrew Prev Unrew'}

plot_groups = [['prev_reward', 'prev_unreward'],
               ['reward_prev_reward', 'unreward_prev_reward', 'reward_prev_unreward', 'unreward_prev_unreward']]

plot_titles = ['Prior Outcome', 'Prior & Current Outcome']
gen_title = 'Previous & Future Outcome'
gen_plot_name = 'prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# consecutive stimuli by prior outcome & tone side
gen_group_labels = {'{}_same_tone_resp': '{} Same', '{}_diff_tone_resp': '{} Diff', 
                    '{}_tone_prev_reward': '{} Prev Rew', '{}_tone_prev_unreward': '{} Prev Unrew',
                    '{}_same_tone_resp_prev_reward': '{} Same Prev Rew', '{}_same_tone_resp_prev_unreward': '{} Same Prev Unrew',
                    '{}_diff_tone_resp_prev_reward': '{} Diff Prev Rew', '{}_diff_tone_resp_prev_unreward': '{} Diff Prev Unrew'}

group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in rel_sides}

gen_plot_groups = [['{}_same_tone_resp', '{}_diff_tone_resp'], ['{}_tone_prev_reward', '{}_tone_prev_unreward']]

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_same_tone_resp_prev_reward', '{}_same_tone_resp_prev_unreward', '{}_diff_tone_resp_prev_reward', '{}_diff_tone_resp_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in rel_sides])

plot_titles = ['Prior & Current Tone', 'Prior Outcome', 'Prior & Current Contra Tone by Prev Outcome', 'Prior & Current Ipsi Tone by Prev Outcome']
gen_title = 'Consecutive Trial Tones by Relative Side and Previous Outcome'
gen_plot_name = 'consec_tones_side_prev_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# previous and current outcome by tone side
gen_group_labels = {'{}_tone_prev_reward': '{} Prev Rew', '{}_tone_prev_unreward': '{} Prev Unrew',
                    '{}_tone_reward': '{} Rew', '{}_tone_unreward': '{} Unrew',
                    '{}_tone_reward_prev_reward': '{} Rew | Rew', '{}_tone_reward_prev_unreward': '{} Rew | Unrew',
                    '{}_tone_unreward_prev_reward': '{} Unrew | Rew', '{}_tone_unreward_prev_unreward': '{} Unrew | Unrew'}

group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in rel_sides}

gen_plot_groups = [['{}_tone_prev_reward', '{}_tone_prev_unreward'], ['{}_tone_reward', '{}_tone_unreward']]

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_tone_reward_prev_reward', '{}_tone_reward_prev_unreward', '{}_tone_unreward_prev_reward', '{}_tone_unreward_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in rel_sides])

plot_titles = ['Prior Outcome', 'Future Outcome', 'Contra Tones by Prev & Future Outcome', 'Ipsi Tones by Prev & Future Outcome']
gen_title = 'Tones by Relative Side, Previous & Future Outcomes'
gen_plot_name = 'tones_side_prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# next trial choice by consecutive stimuli and outcome
group_labels = {'next_same_tone_stay_reward': 'Next Same Stay', 'next_diff_tone_stay_reward': 'Next Diff Stay',
                'next_same_tone_switch_reward': 'Next Same Switch', 'next_diff_tone_switch_reward': 'Next Diff Switch', 
                'next_same_tone_stay_unreward': 'Next Same Stay', 'next_diff_tone_stay_unreward': 'Next Diff Stay',
                'next_same_tone_switch_unreward': 'Next Same Switch', 'next_diff_tone_switch_unreward': 'Next Diff Switch'}

plot_groups = [['next_same_tone_stay_reward', 'next_diff_tone_stay_reward', 'next_same_tone_switch_reward', 'next_diff_tone_switch_reward'],
               ['next_same_tone_stay_unreward', 'next_diff_tone_stay_unreward', 'next_same_tone_switch_unreward', 'next_diff_tone_switch_unreward']]

plot_titles = ['Rewarded', 'Unrewarded']
gen_title = 'Future Tone & Choice by Current Outcome'
gen_plot_name = 'prev_future_outcome'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task)


# %% Regional Comparisons - Cross-correlation



