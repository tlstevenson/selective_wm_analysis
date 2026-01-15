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
import numpy as np
import matplotlib.pyplot as plt
import copy
import os.path as path
import pickle
import time
from scipy.integrate import cumulative_trapezoid, trapezoid
from collections import Counter

Align = fpah.Alignment

# %% Load behavior data

reload = False

# used for saving plots
wm_beh_name = 'Single Tone WM'
wm_sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=7)

bandit_beh_name = 'Two-armed Bandit'
bandit_sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2)

tasks = ['wm', 'bandit']
beh_names = {'wm': 'Single Tone WM', 'bandit': 'Probabilistic Bandit'}

# optionally limit sessions based on subject ids
subj_ids = [198, 199, 274, 400, 402]
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
bah.calc_trial_hist(wm_sess_data, n_back=rew_rate_n_back, exclude_bails=False, col_suffix='bail')
wm_sess_data['n_rew_hist_bail'] = wm_sess_data['rew_hist_bail'].apply(sum)

bah.calc_trial_hist(bandit_sess_data, n_back=rew_rate_n_back)
bandit_sess_data['n_rew_hist'] = bandit_sess_data['rew_hist'].apply(sum)

# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
rew_hist_bin_edges = np.arange(-0.5, rew_rate_n_back+1.5, 1)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{}'.format(i) for i,b in enumerate(rew_hist_bins)}

all_regions = ['PL', 'NAc', 'DMS', 'DLS', 'TS']

# %% Get and process photometry data

recalculate = False
recalculate_norm_tasks = [] # 'wm', 'bandit'
tilt_t = False
baseline_correction = True
baseline_band_iso_fit = True
band_iso_fit = False
save_process_plots = True
show_process_plots = True
filter_dropout_outliers = False

# ignored_signals = {'PL': [],
#                    'DMS': [],
#                    'DLS': [],
#                    'TS': []}

# ignored_subjects = [] #[182] [179]

reprocess_sess_ids = []
reprocess_subj_ids = []

signal_types = ['z_dff_iso_baseline', 'z_dff_iso_baseline_fband'] # 'dff_iso_baseline', 

bandit_alignments = [Align.cport_on, Align.cpoke_in, Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_reward, Align.cport_on_cpoke_in]
wm_alignments = bandit_alignments.copy()
wm_alignments.append(Align.tone)

default_xlims = {'PL': [-3,5], 'NAc': [-2,3], 'DMS': [-1,2], 'DLS': [-1,2], 'TS': [-1,2]}
xlims = {align: default_xlims.copy() for align in wm_alignments}
xlims[Align.reward]['PL'] = [-3,15]

# minimum number of bins to have time invariant intervals between task events 
min_norm_bins = 20
pre_cue_dt = 0.5
pre_cport_on_dt = 0.5
post_reward_dt = 1.5
post_cpoke_in_dt = 0.5
max_cport_on_cpoke_in = 10

filename = 'wm_bandit_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        aligned_signals = saved_data['aligned_signals']
        aligned_time = saved_data['aligned_time']
        aligned_metadata = saved_data['metadata']
        
        recalculate_regions = aligned_metadata['xlims'] != xlims

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    aligned_signals = {task: {subjid: {} for subjid in subj_ids}
                       for task in tasks}
    aligned_time = {task: {'events': {}, 't': {}} for task in tasks}

# get dt
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']

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

    n_pre_cue_bins = max(np.ceil(pre_cue_dt/dt), min_norm_bins)
    n_cue_poke_out_bins = max(np.ceil(med_poke_out_dt/dt), min_norm_bins)
    n_poke_out_resp_bins = max(np.ceil(med_resp_dt/dt), min_norm_bins)
    n_resp_rew_bins = max(np.ceil(med_rew_dt/dt), min_norm_bins)
    n_post_rew_bins = max(np.ceil(post_reward_dt/dt), min_norm_bins)
    
    n_pre_cport_on_bins = max(np.ceil(pre_cport_on_dt/dt), min_norm_bins)
    n_cport_poke_in_bins = max(np.ceil(med_cpoke_dt/dt), min_norm_bins)
    n_post_poke_in_bins = max(np.ceil(post_cpoke_in_dt/dt), min_norm_bins)
    
    # persist median bin numbers for plotting
    aligned_time[task]['events'][Align.cport_on_cpoke_in] = {'cport_on': n_pre_cport_on_bins, 
                                                             'cpoke_in': n_pre_cport_on_bins+n_cport_poke_in_bins}
        
    aligned_time[task]['events'][Align.cue_reward] = {'cue': n_pre_cue_bins, 
                                                      'poke_out': n_pre_cue_bins+n_cue_poke_out_bins,
                                                      'response': n_pre_cue_bins+n_cue_poke_out_bins+n_poke_out_resp_bins,
                                                      'reward': n_pre_cue_bins+n_cue_poke_out_bins+n_poke_out_resp_bins+n_resp_rew_bins}
        
    
    aligned_time[task]['t'] = {align: {} for align in alignments}
    for align in alignments:
        for region in default_xlims.keys():
            if align == Align.cport_on_cpoke_in:
                aligned_time[task]['t'][align][region] = np.arange(0, n_pre_cport_on_bins+n_cport_poke_in_bins+n_post_poke_in_bins, 1)
            elif align == Align.cue_reward:
                aligned_time[task]['t'][align][region] = np.arange(0, n_pre_cue_bins+n_cue_poke_out_bins+
                                                                      n_poke_out_resp_bins+n_resp_rew_bins+n_post_rew_bins, 1)
            else:
                aligned_time[task]['t'][align][region] = np.arange(xlims[align][region][0], xlims[align][region][1]+dt, dt)
    
                                    
    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue
        
        if not subj_id in aligned_signals[task]:
            aligned_signals[task][subj_id] = {}
        
        for sess_id in sess_ids[subj_id]:
            if sess_id in fpah.__sess_ignore:
                continue

            if (sess_id in aligned_signals[task][subj_id] and not sess_id in reprocess_sess_ids and 
                not task in recalculate_norm_tasks and not subj_id in reprocess_subj_ids):
                
                continue
            else:
                aligned_signals[task][subj_id][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, 
                                           band_iso_fit=band_iso_fit, filter_dropout_outliers=filter_dropout_outliers)
            fp_data = fp_data[subj_id][sess_id]
            dt = fp_data['dec_info']['decimated_dt']
            
            if show_process_plots or save_process_plots:
                fig = fpah.view_processed_signals(fp_data['processed_signals'], fp_data['time'], 
                                                  plot_baseline_corr=baseline_correction, plot_fband=band_iso_fit, plot_baseline_fband=baseline_band_iso_fit,
                                                  title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id))

                if save_process_plots:
                    fpah.save_fig(fig, fpah.get_figure_save_path(beh_names[task], subj_id, 'sess_{}'.format(sess_id)))

                if show_process_plots:
                    plt.show()
                
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
                            
                            if region in aligned_signals[task][subj_id][sess_id][signal_type][align]:
                                continue
                            
                            signal = fp_data['processed_signals'][region][signal_type]
                            
                            match align:
                                case Align.cport_on_cpoke_in:

                                    poke_in_sel = ((cpoke_in_ts - cport_on_ts) < max_cport_on_cpoke_in).to_numpy()
                                    pre_cport_on = fp_utils.build_time_norm_signal_matrix(signal, ts, cport_on_ts-pre_cport_on_dt, cport_on_ts, n_pre_cport_on_bins, align_sel=poke_in_sel)
                                    cport_on_cpoke_in = fp_utils.build_time_norm_signal_matrix(signal, ts, cport_on_ts, cpoke_in_ts, n_cport_poke_in_bins, align_sel=poke_in_sel)
                                    post_cpoke_in = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_in_ts, cpoke_in_ts+post_cpoke_in_dt, n_post_poke_in_bins, align_sel=poke_in_sel)
                                    
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

                            if region in aligned_signals[task][subj_id][sess_id][signal_type][align]:
                                continue
                            
                            signal = fp_data['processed_signals'][region][signal_type]
    
                            reg_key = [k for k in xlims[align].keys() if k in region][0]
                            lims = xlims[align][reg_key]
                            
                            mat, _ = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)
                            aligned_signals[task][subj_id][sess_id][signal_type][align][region] = mat
                               
            print('Stacked FP data for subject {} session {} in {:.1f} s'.format(subj_id, sess_id, time.perf_counter()-start))
    
            with open(save_path, 'wb') as f:
                pickle.dump({'aligned_signals': aligned_signals,
                             'aligned_time': aligned_time,
                             'metadata': {'signal_types': signal_types,
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
                
                rew_hist_bail = pd.cut(trial_data['n_rew_hist_bail'], rew_hist_bins)
            
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
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_resp'] = rew_sel & response
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_prev_resp'] = rew_sel & prev_resp
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_prev_bail'] = rew_sel & prev_bail
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
                            
                    for rew_bin in rew_hist_bins:
                        rew_sel = rew_hist_bail == rew_bin
                        bin_str = rew_hist_bin_strs[rew_bin]
                        
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str] = rew_sel 
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_resp'] = rew_sel & response
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_prev_resp'] = rew_sel & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_prev_bail'] = rew_sel & prev_bail
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_reward'] = rew_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_unreward'] = rew_sel & unreward
                        
                        for rel_side in rel_sides:
                            side_sel = rel_choice_side == rel_side
                            prev_side_sel = rel_prev_choice_side == rel_side
                            
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_'+rel_side] = rew_sel & side_sel 
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_prev_'+rel_side] = rew_sel & prev_side_sel 
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_'+rel_side+'_reward'] = rew_sel & side_sel & reward
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_'+rel_side+'_unreward'] = rew_sel & side_sel & unreward
                    
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
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_resp'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_bail'] = tone_sel & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_reward'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_unreward'] = tone_sel & prev_unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_resp'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_bail'] = tone_sel & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_reward'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_unreward'] = tone_sel & prev_unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward_prev_reward'] = tone_sel & prev_reward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward_prev_unreward'] = tone_sel & prev_unreward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward_prev_reward'] = tone_sel & prev_reward & unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward_prev_unreward'] = tone_sel & prev_unreward & unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_reward_prev_reward'] = tone_sel & prev_reward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_reward_prev_unreward'] = tone_sel & prev_unreward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_unreward_prev_reward'] = tone_sel & prev_reward & unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_unreward_prev_unreward'] = tone_sel & prev_unreward & unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp'] = tone_sel & prev_trial_same & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp'] = tone_sel & prev_trial_diff & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp_prev_reward'] = tone_sel & prev_trial_same & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp_prev_unreward'] = tone_sel & prev_trial_same & response & prev_resp & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp_prev_reward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp_prev_unreward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_unreward
                        
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_same_tone_resp'] = tone_sel & prev_trial_same & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_diff_tone_resp'] = tone_sel & prev_trial_diff & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_same_tone_resp_prev_reward'] = tone_sel & prev_trial_same & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_same_tone_resp_prev_unreward'] = tone_sel & prev_trial_same & response & prev_resp & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_diff_tone_resp_prev_reward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_diff_tone_resp_prev_unreward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_unreward
                        

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
plot_signals = ['z_dff_iso_baseline_fband'] # 'z_dff_iso',
plot_tasks = ['wm', 'bandit'] #
plot_meta_subj = True
    
use_se = True
ph = 3.5;
pw = 5;

gen_xlims = {'DMS': [-1,1.5], 'DLS': [-1,1.5], 'TS': [-1,1.5], 'NAc': [-1.5,2], 'PL': [-3,3]}
rew_xlims = {'DMS': [-1,2], 'DLS': [-1,2], 'TS': [-1,2], 'NAc': [-2,4], 'PL': [-3,10]}

all_xlims = {Align.cport_on: gen_xlims, Align.early_cpoke_in: gen_xlims, Align.cpoke_in: gen_xlims, Align.tone: gen_xlims,
            Align.cue: gen_xlims, Align.cpoke_out: gen_xlims, Align.resp: gen_xlims, Align.reward: rew_xlims,
            Align.cue_reward: None, Align.cport_on_cpoke_in: None}

tone_dashlines = [0.4]
norm_keys = {Align.cport_on_cpoke_in: ['cport_on', 'cpoke_in'], Align.cue_reward: ['cue', 'poke_out', 'response', 'reward']}
norm_labels = {Align.cport_on_cpoke_in: ['Cport On', 'Poke In'], Align.cue_reward: ['Cue', 'Out', 'Resp', 'Rew']}
norm_dashlines = {task: {align: [aligned_time[task]['events'][align][a] for a in norm_keys[align]] for align in [Align.cport_on_cpoke_in, Align.cue_reward]} for task in plot_tasks}

save_plots = True
show_plots = True

# declare method to stack data matrices for the given task, alignment, and selection grouping
def stack_mats(task, align, trial_groups, signal_types=plot_signals):

    if not utils.is_list(signal_types):
        signal_types = signal_types
        
    stacked_mat_dict = {sig: {subj: {} for subj in subj_ids} for sig in signal_types} 

    for signal_type in signal_types:
        for subj_id in subj_ids:
            subj_regions = list(implant_info[subj_id].keys())
            sess_ids = list(align_trial_selections[task][subj_id].keys())
            
            for region in subj_regions:
                stacked_mat_dict[signal_type][subj_id][region] = {}
                
                for group in trial_groups:

                    stacked_mats = []

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
                        stacked_mat_dict[signal_type][subj_id][region][group] = np.vstack(stacked_mats)
                    else:
                        stacked_mat_dict[signal_type][subj_id][region][group] = np.full_like(aligned_time[task]['t'][align][region][None,:], np.nan)

        if plot_meta_subj:
            # build region mapping since some animals have bilateral implants in the same region
            reg_mapping = {s: {r: [k for k in all_regions if k in r][0] for r in implant_info[s].keys()} for s in subj_ids}
            
            stacked_mat_dict[signal_type]['all'] = {r: {g: np.full_like(aligned_time[task]['t'][align][r], np.nan) for g in trial_groups} for r in all_regions}
            
            for s in subj_ids:
                subj_regions = list(implant_info[s].keys())
                
                for r in subj_regions:
                    for group in trial_groups:
                        stacked_mat_dict[signal_type]['all'][reg_mapping[s][r]][group] = np.vstack([stacked_mat_dict[signal_type]['all'][reg_mapping[s][r]][group],
                                                                                                    stacked_mat_dict[signal_type][s][r][group]])

    return stacked_mat_dict

# declare method to save plots
def save_plot(fig, task, subjects, plot_name):
    if save_plots and not plot_name is None:
        fpah.save_fig(fig, fpah.get_figure_save_path(beh_names[task], subjects, plot_name))

    if not show_plots:
        plt.close(fig)
        
def sort_subj_regions(subj_regions):
    plot_order = {r: i for i, r in enumerate(all_regions)}
    return sorted(subj_regions, key=lambda x: next((plot_order[r] for r in all_regions if r in x), np.inf))

# declare method to plot avg signals for the given plot_groups for one or more tasks, signal types, and alignments
def plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, subjects=subj_ids, tasks=plot_tasks, signal_types=plot_signals,
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
            
            if task != 'wm' and align == Align.tone:
                continue
            
            align_title = fpah.get_align_title(align)
            x_label = fpah.get_align_xlabel(align)
            
            align_xlims = xlims_dict[align]
            
            if align == Align.early_cpoke_in:
                t = aligned_time[task]['t'][Align.cpoke_in]
            else:
                t = aligned_time[task]['t'][align]
                
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
                    align_xlims = {r: [0, dashlines[-1]] for r in default_xlims.keys()}
            
            for signal_type in signal_types:
                _, y_label = fpah.get_signal_type_labels(signal_type)
                
                for subj in subjects:

                    plot_stacked_signals = stacked_signals[signal_type][subj]
                    # preserves ordering of regions
                    plot_regions = sort_subj_regions(plot_stacked_signals.keys())
                    
                    title = '{} Aligned to {} - Subj {}, {}'.format(gen_title, align_title, subj, beh_names[task])
                    if subj == 'all':
                        implant_side_info = None
                    else:
                        implant_side_info = implant_info[subj]

                    fig, plotted = fpah.plot_avg_signals(plot_groups, group_labels, plot_stacked_signals, plot_regions, t, 
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
                        plt.close(fig)


# %% Choice and side

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


plot_groups = [['stay_reward', 'switch_reward', 'stay_unreward', 'switch_unreward'],
               ['contra_stay_reward', 'contra_switch_reward', 'contra_stay_unreward', 'contra_switch_unreward'],
               ['ipsi_stay_reward', 'ipsi_switch_reward', 'ipsi_stay_unreward', 'ipsi_switch_unreward']]
group_labels = {'stay_reward': 'Rewarded Stay', 'switch_reward': 'Rewarded Switch',
                'stay_unreward': 'Unrewarded Stay', 'switch_unreward': 'Unrewarded Switch',
                'contra_stay_reward': 'Rewarded Stay', 'contra_switch_reward': 'Rewarded Switch',
                'contra_stay_unreward': 'Unrewarded Stay', 'contra_switch_unreward': 'Unrewarded Switch',
                'ipsi_stay_reward': 'Rewarded Stay', 'ipsi_switch_reward': 'Rewarded Switch',
                'ipsi_stay_unreward': 'Unrewarded Stay', 'ipsi_switch_unreward': 'Unrewarded Switch'}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Stay/Switch Choice by Outcome'
gen_plot_name = 'stay_switch_side_outcome'

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

gen_group_labels = {'rew_hist_{}_resp': '{}', 'rew_hist_{}_prev_resp': '{}', 'rew_hist_{}_reward': '{} Rew', 'rew_hist_{}_unreward': '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin]) 
                for k,v in gen_group_labels.items() 
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_{}_{}': '{}', 'rew_hist_{}_prev_{}': '{}', 'rew_hist_{}_{}_reward': '{} Rew', 'rew_hist_{}_{}_unreward': '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin]) 
                for k,v in gen_group_labels.items() 
                for rew_bin in rew_hist_bins
                for s in rel_sides})

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

# Reward History by previous choice side
gen_plot_group = ['rew_hist_{}_prev_resp', 'rew_hist_{}_prev_contra', 'rew_hist_{}_prev_ipsi']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Prev Choices', 'Previous Contra Choices', 'Previous Ipsi Choices']
gen_title = 'Reward History by Previous Choice Side'
gen_plot_name = 'rew_hist_prev_side'

aligns = [Align.cport_on_cpoke_in, Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=rew_hist_all_colors)

# Reward History by current choice side
gen_plot_group = ['rew_hist_{}_resp', 'rew_hist_{}_contra', 'rew_hist_{}_ipsi']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History by Choice Side'
gen_plot_name = 'rew_hist_side'

aligns = [Align.cue_reward, Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=rew_hist_all_colors, include_norm_reward=False)

# Reward History by current choice side & outcome

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

# %% WM Reward history w/ bails

task = 'wm'
rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

gen_group_labels = {'rew_hist_bail_{}_resp': '{}', 'rew_hist_bail_{}_prev_resp': '{}', 'rew_hist_bail_{}_reward': '{} Rew', 'rew_hist_bail_{}_unreward': '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin]) 
                for k,v in gen_group_labels.items() 
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_bail_{}_{}': '{}', 'rew_hist_bail_{}_prev_{}': '{}', 'rew_hist_bail_{}_{}_reward': '{} Rew', 'rew_hist_bail_{}_{}_unreward': '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin]) 
                for k,v in gen_group_labels.items() 
                for rew_bin in rew_hist_bins
                for s in rel_sides})

# Reward History by Choice Side

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

# previous choice side
gen_plot_group = ['rew_hist_bail_{}_prev_resp', 'rew_hist_bail_{}_prev_contra', 'rew_hist_bail_{}_prev_ipsi']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Prev Choices', 'Previous Contra Choices', 'Previous Ipsi Choices']
gen_title = 'Reward History with Bails by Previous Choice Side'
gen_plot_name = 'rew_hist_bail_prev_side'

aligns = [Align.cport_on_cpoke_in, Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=rew_hist_all_colors, tasks=task)

# current choice side
gen_plot_group = ['rew_hist_bail_{}_resp', 'rew_hist_bail_{}_contra', 'rew_hist_bail_{}_ipsi']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History with Bails by Choice Side'
gen_plot_name = 'rew_hist_bail_side'

aligns = [Align.cue_reward, Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=rew_hist_all_colors, tasks=task, include_norm_reward=False)

# choice side & outcome

gen_plot_group = ['rew_hist_bail_{}_reward', 'rew_hist_bail_{}_unreward']
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins]]

gen_plot_group = ['rew_hist_bail_{}_{}_reward', 'rew_hist_bail_{}_{}_unreward']
plot_groups.extend([[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides])

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History with Bails By Choice Side & Outcome'
gen_plot_name = 'rew_hist_bail_side_outcome'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), tasks=task, legend_params={'ncol': 2})

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

aligns = [Align.cue_reward, Align.cue]

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

aligns = [Align.cue_reward, Align.cue]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=np.vstack((delay_rew_colors, delay_unrew_colors)), tasks=task, legend_params={'ncol': 2})

# plot again without the reward
gen_plot_name = 'resp_delay_side_outcome_no_rew'
plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, 
                 group_colors=np.vstack((delay_rew_colors, delay_unrew_colors)), tasks=task, include_norm_reward=False, legend_params={'ncol': 2})


# %% WM Tones

task = 'wm'

group_labels = {'response': 'Resp', 'bail': 'Bail', 'rewarded': 'Rew', 'unrewarded': 'Unrew'}

plot_groups = [['response', 'bail'], ['rewarded', 'unrewarded']]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Response and Outcome for All Tones'
gen_plot_name = 'all_tones_resp_outcome'

aligns = [Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task)


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


# tone correct absolute side
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

# previous and current outcome
group_labels = {'prev_reward': 'Prev Rew', 'prev_unreward': 'Prev Unrew', 'prev_bail': 'Prev Bail',
                'reward_prev_reward': 'Rew Prev Rew', 'reward_prev_unreward': 'Rew Prev Unrew', 
                'unreward_prev_reward': 'Unrew Prev Rew', 'unreward_prev_unreward': 'Unrew Prev Unrew'}

plot_groups = [['prev_reward', 'prev_unreward', 'prev_bail'],
               ['reward_prev_reward', 'unreward_prev_reward', 'reward_prev_unreward', 'unreward_prev_unreward']]

plot_titles = ['Prior Outcome', 'Prior & Current Outcome']
gen_title = 'Previous & Future Outcome'
gen_plot_name = 'prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# previous and current outcome by tone type and side
gen_group_labels = {'{}_tone_prev_reward': '{} Prev Rew', '{}_tone_prev_unreward': '{} Prev Unrew', '{}_tone_prev_bail': '{} Prev Bail',
                    '{}_tone_reward': '{} Rew', '{}_tone_unreward': '{} Unrew', '{}_tone_bail': '{} Bail',
                    '{}_tone_reward_prev_reward': '{} Rew | Rew', '{}_tone_reward_prev_unreward': '{} Rew | Unrew',
                    '{}_tone_unreward_prev_reward': '{} Unrew | Rew', '{}_tone_unreward_prev_unreward': '{} Unrew | Unrew'}

group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in rel_sides}

gen_plot_groups = [['{}_tone_prev_reward', '{}_tone_prev_unreward', '{}_tone_prev_bail'], ['{}_tone_reward', '{}_tone_unreward', '{}_tone_bail']]

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_tone_reward_prev_reward', '{}_tone_reward_prev_unreward', '{}_tone_unreward_prev_reward', '{}_tone_unreward_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in rel_sides])

plot_titles = ['Prior Outcome', 'Future Outcome', 'Contra Tones by Prev & Future Outcome', 'Ipsi Tones by Prev & Future Outcome']
gen_title = 'Tones by Relative Side, Previous & Future Outcomes'
gen_plot_name = 'tone_side_prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


group_labels = {k.format(tone): v.format(tone.capitalize()) 
                for k,v in gen_group_labels.items() 
                for tone in tone_types}

gen_plot_groups = [['{}_tone_prev_reward', '{}_tone_prev_unreward', '{}_tone_prev_bail'], ['{}_tone_reward', '{}_tone_unreward', '{}_tone_bail']]

plot_groups = [[g.format(tone) for tone in tone_types for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_tone_reward_prev_reward', '{}_tone_reward_prev_unreward', '{}_tone_unreward_prev_reward', '{}_tone_unreward_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in tone_types])

plot_titles = ['Prior Outcome', 'Future Outcome', 'Contra Tones by Prev & Future Outcome', 'Ipsi Tones by Prev & Future Outcome']
gen_title = 'Tones by Relative Side, Previous & Future Outcomes'
gen_plot_name = 'tone_type_prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


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

# next trial choice by consecutive stimuli and outcome
group_labels = {'next_same_tone_stay_reward': 'Next Same Stay', 'next_diff_tone_stay_reward': 'Next Diff Stay',
                'next_same_tone_switch_reward': 'Next Same Switch', 'next_diff_tone_switch_reward': 'Next Diff Switch', 
                'next_same_tone_stay_unreward': 'Next Same Stay', 'next_diff_tone_stay_unreward': 'Next Diff Stay',
                'next_same_tone_switch_unreward': 'Next Same Switch', 'next_diff_tone_switch_unreward': 'Next Diff Switch'}

plot_groups = [['next_same_tone_stay_reward', 'next_diff_tone_stay_reward', 'next_same_tone_switch_reward', 'next_diff_tone_switch_reward'],
               ['next_same_tone_stay_unreward', 'next_diff_tone_stay_unreward', 'next_same_tone_switch_unreward', 'next_diff_tone_switch_unreward']]

plot_titles = ['Rewarded', 'Unrewarded']
gen_title = 'Future Tone & Choice by Current Outcome'
gen_plot_name = 'consec_tones_prev_future_outcome'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task)


# %% Calculate engaged states

view_states = False
recalculate = False

engage_next_poke_thresh = 15 # time cutoff between center port on and next poke in to separate between task engaged state and not
engage_states = ['engaged', 'disengaged', 'all']

filename = 'wm_bandit_engage_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        engaged_t_sel = saved_data['engaged_t_sel']
        engage_metadata = saved_data['metadata']
        
        recalculate = engage_metadata['engage_next_poke_thresh'] != engage_next_poke_thresh

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    engaged_t_sel = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:
    
    if not subj_id in engaged_t_sel:
        engaged_t_sel[subj_id] = {task: {} for task in tasks}
    
    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids 
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
                                    
        if not subj_id in sess_ids:
            continue
        
        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:
            
            if sess_id in engaged_t_sel[subj_id][task]:
                continue

            fp_data = loc_db.get_sess_fp_data(sess_id)['fp_data'][subj_id][sess_id]
            t = fp_data['time']
            
            # find engaged and disengaged states
            trial_data = sess_data[sess_data['sessid'] == sess_id]
            
            trial_start_ts = fp_data['trial_start_ts']

            cport_on_ts = trial_start_ts[:-1] + trial_data['cport_on_time'].to_numpy()
            cpoke_in_ts = trial_start_ts[:-1] + trial_data['cpoke_in_time'].to_numpy()
            
            port_on_poke_in = cpoke_in_ts - cport_on_ts
            
            disengaged_trial_sel = (port_on_poke_in > engage_next_poke_thresh) | np.isnan(port_on_poke_in)
            
            sess_engaged_t_sel = np.full_like(t, True, dtype=bool)

            for i, disengaged in enumerate(disengaged_trial_sel):
                if disengaged:
                    
                    if i == 0:
                        t_start_idx = 0
                    else:
                        t_start_idx = np.argmin(np.abs(t - cport_on_ts[i]))
                    
                    disengaged_end = cpoke_in_ts[i]
                    if np.isnan(disengaged_end):
                        if i < len(cport_on_ts)-1:
                            disengaged_end = cport_on_ts[i+1]
                        else:
                            disengaged_end = t[-1]
                        
                    t_end_idx = np.argmin(np.abs(t - disengaged_end))
                    sess_engaged_t_sel[t_start_idx:t_end_idx+1] = False
                    
            # fill in after last trial started to end of recording
            if (t[-1] - trial_start_ts[-1]) > engage_next_poke_thresh:
                t_start_idx = np.argmin(np.abs(t - trial_start_ts[-1]))
                sess_engaged_t_sel[t_start_idx:] = False
                
            engaged_t_sel[subj_id][task][sess_id] = sess_engaged_t_sel
            
            if view_states:
        
                lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts}
            
                fig, ax = plt.subplots(1, 1, sharex=True, layout='constrained', figsize=(12,4))
                
                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))
                
                ax.fill_between(t, 0, 1, where=sess_engaged_t_sel,
                                color='grey', alpha=0.4, transform=ax.get_xaxis_transform())
                
                for j, (name, lines) in enumerate(lines_dict.items()):
                    ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j), linestyles='dashed', 
                              transform=ax.get_xaxis_transform())
                    
                ax.set_xlabel('Time (s)')

                ax.legend()
                    
                plt.show()
                
with open(save_path, 'wb') as f:
    pickle.dump({'engaged_t_sel': engaged_t_sel,
                 'metadata': {'engage_next_poke_thresh': engage_next_poke_thresh}},
                f)

# %% Regional Comparisons - Compute cross-correlations

tilt_t = False
baseline_correction = True
baseline_band_iso_fit = True
band_iso_fit = False
filter_dropout_outliers = False

recalculate = False
reprocess_sess_ids = []

signal_types = ['dff_iso_baseline_fband'] # 'z_dff_iso_baseline' 

max_lag = 10
# get dt information to calculate the number of timesteps in the xcorr results
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']
n_lags = int(utils.convert_to_multiple(max_lag, dt)/dt)*2 + 1

tasks = ['wm', 'bandit']

filename = 'wm_bandit_xcorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        x_corrs = saved_data['x_corrs']
        x_corr_metadata = saved_data['metadata']
        x_corr_regions = x_corr_metadata['regions']
        corr_lags = x_corr_metadata['corr_lags']
        
        recalculate = (x_corr_metadata['max_lag'] != max_lag) or (x_corr_metadata['engage_next_poke_thresh'] != engage_next_poke_thresh)

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    x_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    x_corr_regions = {}

for subj_id in subj_ids:
    
    subj_regions = list(implant_info[subj_id].keys())
    
    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]
        
    x_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)
    
    if not subj_id in x_corrs:
        x_corrs[subj_id] = {task: {} for task in tasks}
    
    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids 
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
                                    
        if not subj_id in sess_ids:
            continue
        
        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in x_corrs[subj_id][task] and not sess_id in reprocess_sess_ids and list(x_corrs[subj_id][task][sess_id].keys()) == signal_types:
                continue
            
            if sess_id not in x_corrs[subj_id][task]:
                x_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, 
                                           band_iso_fit=band_iso_fit, baseline_band_iso_fit=baseline_band_iso_fit,
                                           filter_dropout_outliers=filter_dropout_outliers)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            
            sess_engaged_t_sel = engaged_t_sel[subj_id][task][sess_id]
            
            print('Calculating cross-correlation for subject {} session {}:'.format(subj_id, sess_id))
            
            for signal_type in signal_types:
                if signal_type in x_corrs[subj_id][task][sess_id]:
                    continue
                else:
                    x_corrs[subj_id][task][sess_id][signal_type] = {e: np.full((n_regions, n_regions, n_lags), np.nan, dtype=float) for e in engage_states}
                    
                for reg_i in range(n_regions):
                    if not subj_regions[reg_i] in fp_data['processed_signals']:
                        continue
                    signal_i = fp_data['processed_signals'][subj_regions[reg_i]][signal_type]
                    
                    for reg_j in np.arange(reg_i, n_regions):
                        if not subj_regions[reg_j] in fp_data['processed_signals']:
                            continue
                        signal_j = fp_data['processed_signals'][subj_regions[reg_j]][signal_type]
                        
                        start = time.perf_counter()                    
                        
                        for e in engage_states:
                            
                            if e == 'engaged':
                                t_sel = sess_engaged_t_sel
                            elif e == 'disengaged':
                                t_sel = ~sess_engaged_t_sel
                            else:
                                t_sel = None
                            
                            xcorr, corr_lags = fp_utils.correlate(signal_i, signal_j, dt, max_lag=max_lag, t_sel=t_sel)
    
                            x_corrs[subj_id][task][sess_id][signal_type][e][reg_i, reg_j, :] = xcorr
                            if reg_i != reg_j:
                                x_corrs[subj_id][task][sess_id][signal_type][e][reg_j, reg_i, :] = np.flip(xcorr)
                                
                        print('  Between {} and {} in {:.1f} s'.format(
                            subj_regions[reg_i], subj_regions[reg_j], time.perf_counter()-start))
    
                with open(save_path, 'wb') as f:
                    pickle.dump({'x_corrs': x_corrs,
                                 'metadata': {'regions': x_corr_regions,
                                              'max_lag': max_lag,
                                              'corr_lags': corr_lags,
                                              'engage_next_poke_thresh': engage_next_poke_thresh}},
                                f)


# %% Plot cross-correlations

plot_lag = 1.5
t_sel = np.abs(corr_lags) < plot_lag

plot_sep_tasks = True
plot_sep_states = False
plot_ind_subj = True
plot_comb_subj = False
ax_size = 3

plot_tasks = []
if plot_sep_tasks:
    plot_tasks.extend(tasks.copy())

plot_tasks.append('all')
    
if plot_sep_states:
    plot_states = engage_states.copy()
else:
    plot_states = ['all']
    
plot_beh_names = {'wm': 'WM Task', 'bandit': 'Bandit Task', 'all': 'All Tasks'}
engage_labels = {'engaged': 'engaged', 'disengaged': 'disengaged', 'all': 'all'}

if plot_comb_subj:
    comb_xcorr = {t: {s: {e: [] for e in plot_states} for s in signal_types} for t in tasks}
    comb_xcorr_weights = {t: {s: {e: [] for e in plot_states} for s in signal_types} for t in tasks}

for subj_id in subj_ids:
    
    subj_regions = x_corr_regions[subj_id]
    sorted_regions = sort_subj_regions(subj_regions)
    sorted_region_idxs = [subj_regions.index(r) for r in sorted_regions]
    
    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]
    
    # stack across subjects
    if plot_comb_subj:
        for task in tasks:
            
            if len(x_corrs[subj_id][task]) == 0:
                continue
            
            for signal_type in signal_types:
    
                # stack by session across regions
                if plot_comb_subj and task != 'all':
    
                    for s in x_corrs[subj_id][task].keys():
                        for e in plot_states:
                            sess_xcorr = x_corrs[subj_id][task][s][signal_type][e]
                            
                            # get mapping of subject regions to all regions order
                            all_region_idx_mapping = fpah.get_region_idx_mapping(subj_regions, all_regions)
                            dup_counts = fpah.count_corr_pairs(subj_regions, all_regions)
                            
                            if all([v == 1 for v in dup_counts.values()]):
                                # if no duplicates, can simply use indexing
                                sess_comb_xcorr = np.full((len(all_regions), len(all_regions), len(corr_lags)), np.nan)
                                sess_comb_xcorr[np.ix_(all_region_idx_mapping, all_region_idx_mapping, np.arange(len(corr_lags)))] = sess_xcorr
                                
                                comb_xcorr[task][signal_type][e].append(sess_comb_xcorr)
                                
                                weights = np.ones_like(sess_comb_xcorr)
                                weights[np.isnan(sess_comb_xcorr)] = np.nan
                                comb_xcorr_weights[task][signal_type][e].append(weights)
                            else:
                                # add each duplicate pair separately
                                for subj_i, map_i in enumerate(all_region_idx_mapping):
                                    for subj_j, map_j in enumerate(all_region_idx_mapping):
                                        # only do each cross-regional comparison once
                                        if subj_j < subj_i:
                                            continue
                                        # don't add bilateral cross-correlations of the same region as auto-correlations
                                        if map_i == map_j and subj_i != subj_j:
                                            continue
                                        
                                        sess_comb_xcorr = np.full((len(all_regions), len(all_regions), len(corr_lags)), np.nan)
                                        sess_comb_xcorr[map_i,map_j,:] = sess_xcorr[subj_i,subj_j,:]
                                        if map_i != map_j:
                                            sess_comb_xcorr[map_j,map_i,:] = sess_xcorr[subj_j,subj_i,:]
                                        
                                        comb_xcorr[task][signal_type][e].append(sess_comb_xcorr)
                                        
                                        dup_key = tuple(sorted([map_i,map_j]))
                                        weights = np.full_like(sess_comb_xcorr, 1/dup_counts[dup_key])
                                        weights[np.isnan(sess_comb_xcorr)] = np.nan
                                        comb_xcorr_weights[task][signal_type][e].append(weights)

    # plot individual subject averages
    if plot_ind_subj:
        if plot_sep_states:
            for task in plot_tasks:
                for signal_type in signal_types:
                    
                    signal_title, _ = fpah.get_signal_type_labels(signal_type)
            
                    fig, axs = plt.subplots(n_regions, n_regions, sharex=True, sharey=True, figsize=(ax_size*n_regions, ax_size*n_regions), layout='constrained')
                    fig.suptitle('Regional Cross-correlation for Subject {} in {}. (- first leading, + first lagging)\n{}'.format(subj_id, plot_beh_names[task], signal_title))
                    
                    for reg_i in range(n_regions):   
                        sorted_i = sorted_region_idxs[reg_i]
                        for reg_j in range(n_regions):
                            sorted_j = sorted_region_idxs[reg_j]
            
                            ax = axs[reg_i, reg_j]
                            
                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            
                            for e in plot_states:
                                if task == 'all':
                                    sess_corrs = np.hstack([np.stack([x_corrs[subj_id][t][s][signal_type][e][sorted_i, sorted_j, :] for s in x_corrs[subj_id][t].keys()], axis=1) for t in tasks if len(x_corrs[subj_id][t]) > 0])
                                else:
                                    sess_corrs = np.stack([x_corrs[subj_id][task][s][signal_type][e][sorted_i, sorted_j, :] for s in x_corrs[subj_id][task].keys()], axis=1)
                                
                                plot_utils.plot_psth(corr_lags[t_sel], np.nanmean(sess_corrs, axis=1)[t_sel], utils.stderr(sess_corrs, axis=1)[t_sel], ax, plot_x0=True, label=engage_labels[e])
            
                            ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'], 
                                                                     sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
                            ax.legend()
                            
                            if reg_i == n_regions-1:
                                ax.set_xlabel('Time lag (s)')
                            if reg_j == 0:
                                ax.set_ylabel('Pearson r')
    
        else:
            for signal_type in signal_types:
                
                signal_title, _ = fpah.get_signal_type_labels(signal_type)
        
                fig, axs = plt.subplots(n_regions, n_regions, sharex=True, sharey=True, figsize=(ax_size*n_regions, ax_size*n_regions), layout='constrained')
                fig.suptitle('Regional Cross-correlation for Subject {}. (- first leading, + first lagging)\n{}'.format(subj_id, signal_title))
                
                for reg_i in range(n_regions):   
                    sorted_i = sorted_region_idxs[reg_i]
                    for reg_j in range(n_regions):
                        sorted_j = sorted_region_idxs[reg_j]
        
                        ax = axs[reg_i, reg_j]
                        
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        
                        for task in plot_tasks:
                            if task == 'all':
                                sess_corrs = np.hstack([np.stack([x_corrs[subj_id][t][s][signal_type]['all'][sorted_i, sorted_j, :] for s in x_corrs[subj_id][t].keys()], axis=1) for t in tasks if len(x_corrs[subj_id][t]) > 0])
                            else:
                                sess_corrs = np.stack([x_corrs[subj_id][task][s][signal_type]['all'][sorted_i, sorted_j, :] for s in x_corrs[subj_id][task].keys()], axis=1)
                            
                            plot_utils.plot_psth(corr_lags[t_sel], np.nanmean(sess_corrs, axis=1)[t_sel], utils.stderr(sess_corrs, axis=1)[t_sel], ax, plot_x0=True, label=plot_beh_names[task])
        
                        ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'], 
                                                                 sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
                        ax.legend()
                        
                        if reg_i == n_regions-1:
                            ax.set_xlabel('Time lag (s)')
                        if reg_j == 0:
                            ax.set_ylabel('Pearson r')
                    
                            
# plot combined subject averages
if plot_comb_subj:
    # stack all matrices together
    comb_xcorr = {t: {s: {e: np.stack(comb_xcorr[t][s][e], axis=-1) for e in plot_states} for s in signal_types} for t in tasks}
    comb_xcorr_weights = {t: {s: {e: np.stack(comb_xcorr_weights[t][s][e], axis=-1) for e in plot_states} for s in signal_types} for t in tasks}
    
    if plot_sep_states:
        for task in plot_tasks:
            for signal_type in signal_types:
                
                signal_title, _ = fpah.get_signal_type_labels(signal_type)
            
                fig, axs = plt.subplots(len(all_regions), len(all_regions), sharex=True, sharey=True, 
                                        figsize=(ax_size*len(all_regions), ax_size*len(all_regions)), layout='constrained')
                fig.suptitle('Regional Cross-correlation for All Subjects in {}. (- first leading, + first lagging)\n{}'.format(plot_beh_names[task], signal_title))
                
                for reg_i in range(len(all_regions)):   
                    for reg_j in range(len(all_regions)):
        
                        ax = axs[reg_i, reg_j]
                        
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        
                        for e in plot_states:
                            if task == 'all':
                                comb_corrs = np.hstack([comb_xcorr[t][signal_type][e][reg_i, reg_j, :, :] for t in tasks])
                                comb_weights = np.hstack([comb_xcorr_weights[t][signal_type][e][reg_i, reg_j, :, :] for t in tasks])
                            else:
                                comb_corrs = comb_xcorr[task][signal_type][e][reg_i, reg_j, :, :]
                                comb_weights = comb_xcorr_weights[task][signal_type][e][reg_i, reg_j, :, :]

                            plot_utils.plot_psth(corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel], 
                                                 utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=True, label=engage_labels[e])
    
                        ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
                        if plot_sep_states:
                            ax.legend()
                        
                        if reg_i == len(all_regions)-1:
                            ax.set_xlabel('Time lag (s)')
                        if reg_j == 0:
                            ax.set_ylabel('Pearson r')
    else:
        for signal_type in signal_types:
            
            signal_title, _ = fpah.get_signal_type_labels(signal_type)
        
            fig, axs = plt.subplots(len(all_regions), len(all_regions), sharex=True, sharey=True, 
                                    figsize=(ax_size*len(all_regions), ax_size*len(all_regions)), layout='constrained')
            fig.suptitle('Regional Cross-correlation for All Subjects. (- first leading, + first lagging)\n{}'.format(signal_title))
            
            for reg_i in range(len(all_regions)):   
                for reg_j in range(len(all_regions)):
    
                    ax = axs[reg_i, reg_j]
                    
                    plot_utils.plot_dashlines(0, dir='h', ax=ax)
                    
                    for task in plot_tasks:
                        if task == 'all':
                            comb_corrs = np.hstack([comb_xcorr[t][signal_type]['all'][reg_i, reg_j, :, :] for t in tasks])
                            comb_weights = np.hstack([comb_xcorr_weights[t][signal_type]['all'][reg_i, reg_j, :, :] for t in tasks])
                        else:
                            comb_corrs = comb_xcorr[task][signal_type]['all'][reg_i, reg_j, :, :]
                            comb_weights = comb_xcorr_weights[task][signal_type]['all'][reg_i, reg_j, :, :]

                        plot_utils.plot_psth(corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel], 
                                             utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=True, label=plot_beh_names[task])

                    ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
                    ax.legend()
                    
                    if reg_i == len(all_regions)-1:
                        ax.set_xlabel('Time lag (s)')
                    if reg_j == 0:
                        ax.set_ylabel('Pearson r')
            
    

# %% Calculate power spectra

tilt_t = False
baseline_correction = True
band_iso_fit = False
baseline_band_iso_fit = True
filter_dropout_outliers = False
f_min = 0.005
f_max = 20

recalculate = False
reprocess_sess_ids = []

signal_types = ['z_dff_iso_baseline_fband'] # 'z_dff_iso_baseline' 

tasks = ['wm', 'bandit']

filename = 'wm_bandit_spectral_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        ps_data = saved_data['ps_data']
        freqs = saved_data['freqs']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    ps_data = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:
    
    subj_regions = list(implant_info[subj_id].keys())
    
    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]

    if not subj_id in ps_data:
        ps_data[subj_id] = {task: {} for task in tasks}
    
    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids 
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
                                    
        if not subj_id in sess_ids:
            continue
        
        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in ps_data[subj_id][task] and not sess_id in reprocess_sess_ids and all([t in list(ps_data[subj_id][task][sess_id].keys()) for t in signal_types]):
                continue
            
            if sess_id not in ps_data[subj_id][task]:
                ps_data[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, 
                                           band_iso_fit=band_iso_fit, baseline_band_iso_fit=baseline_band_iso_fit, filter_dropout_outliers=filter_dropout_outliers)
            fp_data = fp_data[subj_id][sess_id]
            dt = fp_data['dec_info']['decimated_dt']
            
            for signal_type in signal_types:
                if signal_type in ps_data[subj_id][task][sess_id]:
                    continue
                else:
                    ps_data[subj_id][task][sess_id][signal_type] = {r: [] for r in subj_regions}

                for region in subj_regions:
                    if not region in fp_data['processed_signals']:
                        continue
                    
                    signal = fp_data['processed_signals'][region][signal_type]
                    
                    freqs, ps = fp_utils.calc_power_spectra(signal, dt, f_min=f_min, f_max=f_max)
                    
                    ps_data[subj_id][task][sess_id][signal_type][region] = ps
                    
                    
                with open(save_path, 'wb') as f:
                    pickle.dump({'ps_data': ps_data,
                                 'freqs': freqs},
                                f)

#%% Plot spectra
# plot session spectra

save_plots = False
show_plots = True

plot_ind_sess = False
plot_subj_avg = False
plot_meta_subj = True
plot_band_prop = True
plot_comb_task = True
x_rot = 90

plot_tasks = tasks.copy()

if plot_comb_task:
    plot_tasks.append('all')
    
plot_beh_names = {'wm': 'WM Task', 'bandit': 'Bandit Task', 'all': 'All Tasks'}
plot_signals = ['z_dff_iso_baseline_fband']

freq_range = [0, 10]

freq_vals = [0] + list(np.logspace(np.log10(0.01), np.log10(10), 20))
freq_vals = np.round(freq_vals, 3)
prop_bands = list(zip(freq_vals[:-1], freq_vals[1:]))
prop_bands = [[0,0.02], [0.02,0.1], [0.1,0.3], [0.3,1], [1,1.8], [1.8,4], [4,10]]
#prop_bands = [[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,1], [1,2], [2,6], [6,10]] #[[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,2], [2,6], [6,10]]
band_labels = ['{}-{}'.format(b[0], b[1]) for b in prop_bands]

# define plotting method
def plot_spectra(ps_dict, freqs, err_dict=None, title='', fname=None, x_lims=freq_range, ax=None, logy=True, ylabel=None):

    freq_sel = (freqs >= x_lims[0]) & (freqs <= x_lims[1])
    
    if ax is None:
        fig, ax = plt.subplots(1)
    
    for key in ps_dict.keys():
        if len(ps_dict[key]) == 0:
            continue
        
        if err_dict is None or not key in err_dict:
            plot_utils.plot_shaded_error(freqs[freq_sel], ps_dict[key][freq_sel], ax=ax, label=key)
        else:
            plot_utils.plot_shaded_error(freqs[freq_sel], ps_dict[key][freq_sel], y_err=err_dict[key][freq_sel], ax=ax, label=key)        

    ax.set_xscale('log')
    ax.set_xlabel('Frequency (Hz)')
    if logy:
        ax.set_yscale('log')
    if ylabel is None:
        ax.set_ylabel('Power Spectral Density (V^2/Hz)')
    else:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    
    if ax is None:
        if save_plots and not fname is None:
            fpah.save_fig(fig, fpah.get_figure_save_path('Power Spectra', subj_id, fname))
            
        if show_plots:
            plt.show()
        
        plt.close(fig)

# plot each session individually
if plot_ind_sess:
    for subj_id in subj_ids:
        for task in tasks:
            for sess_id in ps_data[subj_id][task].keys():
                for signal_type in plot_signals:
    
                    plot_spectra(ps_data[subj_id][task][sess_id][signal_type], freqs, 
                                 title='Subject {}, Session {}, {}\n{}'.format(subj_id, sess_id, plot_beh_names[task], fpah.get_signal_type_labels(signal_type)[0]), 
                                 fname='Subject {} Session {} {} {}'.format(subj_id, sess_id, plot_beh_names[task], signal_type))
                    
# plot average of each subject per task
if plot_subj_avg:
    for subj_id in subj_ids:
        
        subj_regions = list(implant_info[subj_id].keys())
        
        if subj_id in fpah.__region_ignore:
            subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]
            
        for signal_type in plot_signals:
            
            fig_ps, axs_ps = plt.subplots(1, len(plot_tasks), sharey=True, sharex=True, layout='constrained', figsize=(len(plot_tasks)*5, 4))
            fig_ps.suptitle('Subject {} Average Power Spectra, {}'.format(subj_id, fpah.get_signal_type_labels(signal_type)[0]))
            
            if plot_band_prop:
                fig_prop, axs_prop = plt.subplots(1, len(plot_tasks), sharey=True, sharex=True, layout='constrained', figsize=(len(plot_tasks)*5, 4))
                fig_prop.suptitle('Subject {} Average Power Proportion, {}'.format(subj_id, fpah.get_signal_type_labels(signal_type)[0]))
                
                fig_prop_reg, axs_prop_reg = plt.subplots(len(subj_regions), 1, layout='constrained', figsize=(6, 4*len(subj_regions)))
                fig_prop_reg.suptitle('Subject {} Average Power Proportion, {}'.format(subj_id, fpah.get_signal_type_labels(signal_type)[0]))
                
                prop_avg_task = {}
                prop_err_task = {}
                
            for i, task in enumerate(plot_tasks):
                
                if (task != 'all' and len(ps_data[subj_id][task]) == 0) or (task == 'all' and all([len(ps_data[subj_id][t]) == 0 for t in tasks])):
                    continue
                
                ax_ps = axs_ps[i]
                if plot_band_prop:
                    ax_prop = axs_prop[i]
                
                ps_avg = {}
                ps_err = {}
                
                prop_avg = {}
                prop_err = {}
                
                for region in subj_regions:
                    if task == 'all':
                        reg_ps = np.vstack([np.stack([ps_data[subj_id][t][s][signal_type][region] for s in ps_data[subj_id][t].keys()
                                                      if len(ps_data[subj_id][t][s][signal_type][region])> 0], axis=0) for t in tasks if len(ps_data[subj_id][t]) > 0])
                    else:
                        reg_ps = np.stack([ps_data[subj_id][task][s][signal_type][region] for s in ps_data[subj_id][task].keys() 
                                           if len(ps_data[subj_id][task][s][signal_type][region]) > 0], axis=0)
                    
                    ps_avg[region] = np.nanmean(reg_ps, axis=0)
                    ps_err[region] = utils.stderr(reg_ps, axis=0)
                    
                    prop_avg[region] = []
                    prop_err[region] = []
                    freq_sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                    ps_tot = trapezoid(reg_ps[:, freq_sel], freqs[freq_sel], axis=1)
                    for band in prop_bands:
                        freq_sel = (freqs >= band[0]) & (freqs <= band[1])
                        ps_band = trapezoid(reg_ps[:, freq_sel], freqs[freq_sel], axis=1)
                        prop_band = ps_band/ps_tot * 100
                        prop_avg[region].append(np.nanmean(prop_band, axis=0))
                        prop_err[region].append(utils.stderr(prop_band, axis=0))
                        
                    prop_avg_task[task] = prop_avg
                    prop_err_task[task] = prop_err
                
                plot_spectra(ps_avg, freqs, err_dict=ps_err, ax=ax_ps, title=plot_beh_names[task])
                
                if plot_band_prop:
                    plot_vals = [prop_avg[r] for r in subj_regions]
                    plot_err = [prop_err[r] for r in subj_regions]
                    plot_utils.plot_stacked_bar(plot_vals, value_labels=subj_regions, x_labels=band_labels, orientation='h', ax=ax_prop, err=plot_err,
                                                x_label_rot=x_rot)
                    ax_prop.set_title(plot_beh_names[task])
                    ax_prop.set_ylabel('Relative Power (%)')
                    ax_prop.set_xlabel('Frequency Band')
                    
            # compare task bands by region
            if plot_band_prop:
                for i, region in enumerate(subj_regions):
                    ax_prop_reg = axs_prop_reg[i]
                    plot_vals = [prop_avg_task[t][region] for t in plot_tasks]
                    plot_err = [prop_err_task[t][region] for t in plot_tasks]
                    plot_utils.plot_stacked_bar(plot_vals, value_labels=[plot_beh_names[t] for t in plot_tasks], x_labels=band_labels, orientation='h', ax=ax_prop_reg, err=plot_err,
                                                x_label_rot=x_rot)
                    ax_prop_reg.set_title(region)
                    ax_prop_reg.set_ylabel('Relative Power (%)')
                    ax_prop_reg.set_xlabel('Frequency Band')
                
            if save_plots:
                fpah.save_fig(fig_ps, fpah.get_figure_save_path('Power Spectra', subj_id, 'Subject {} Session Avg {}'.format(subj_id, signal_type)))
                if plot_band_prop:
                    fpah.save_fig(fig_prop, fpah.get_figure_save_path('Power Spectra', subj_id, 'Subject {} Session Avg {} power proportion'.format(subj_id, signal_type)))
                
            if show_plots:
                plt.show()
            
            plt.close(fig_ps)
            if plot_band_prop:
                plt.close(fig_prop)
        
if plot_meta_subj:
    for signal_type in plot_signals:
        fig_ps, axs_ps = plt.subplots(1, len(plot_tasks), sharey=True, sharex=True, layout='constrained', figsize=(len(plot_tasks)*5, 4))
        fig_ps.suptitle('Average Regional Power Spectra, {}'.format(fpah.get_signal_type_labels(signal_type)[0]))
        
        if plot_band_prop:
            fig_prop, axs_prop = plt.subplots(1, len(plot_tasks), sharey=True, sharex=True, layout='constrained', figsize=(len(plot_tasks)*5, 4))
            fig_prop.suptitle('Average Regional Power Proportion, {}'.format(fpah.get_signal_type_labels(signal_type)[0]))
            
            fig_prop_reg, axs_prop_reg = plt.subplots(len(all_regions), 1, layout='constrained', figsize=(6, 4*len(all_regions)))
            fig_prop_reg.suptitle('Average Regional Power Proportion\n{}'.format(fpah.get_signal_type_labels(signal_type)[0]))
            
            prop_avg_task = {}
            prop_err_task = {}
                   
        for i, task in enumerate(plot_tasks):
            ax_ps = axs_ps[i]
            if plot_band_prop:
                ax_prop = axs_prop[i]

            ps_avg = {}
            ps_err = {}
            
            prop_avg = {}
            prop_err = {}
        
            for region in all_regions:
                stacked_ps = []
                
                for subj_id in subj_ids:
                    
                    if (task != 'all' and len(ps_data[subj_id][task]) == 0) or (task == 'all' and all([len(ps_data[subj_id][t]) == 0 for t in tasks])):
                        continue
                    
                    subj_regions = list(implant_info[subj_id].keys())
                    
                    if subj_id in fpah.__region_ignore:
                        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]
                       
                    # build region mapping since some animals have bilateral implants
                    reg_mapping = [r for r in subj_regions if region in r]
                    
                    for sub_reg in reg_mapping:
                        
                        if task == 'all':
                            reg_ps = np.vstack([np.stack([ps_data[subj_id][t][s][signal_type][sub_reg] for s in ps_data[subj_id][t].keys()
                                                          if len(ps_data[subj_id][t][s][signal_type][sub_reg])> 0], axis=0) for t in tasks if len(ps_data[subj_id][t]) > 0])
                        else:
                            reg_ps = np.stack([ps_data[subj_id][task][s][signal_type][sub_reg] for s in ps_data[subj_id][task].keys() 
                                               if len(ps_data[subj_id][task][s][signal_type][sub_reg]) > 0], axis=0)
                        
                        stacked_ps.append(reg_ps)
                        
                stacked_ps = np.vstack(stacked_ps)
                ps_avg[region] = np.nanmean(stacked_ps, axis=0)
                ps_err[region] = utils.stderr(stacked_ps, axis=0)
                
                # calculate relative frequency band power
                prop_avg[region] = []
                prop_err[region] = []
                freq_sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                ps_tot = trapezoid(stacked_ps[:, freq_sel], freqs[freq_sel], axis=1)
                for band in prop_bands:
                    freq_sel = (freqs >= band[0]) & (freqs <= band[1])
                    ps_band = trapezoid(stacked_ps[:, freq_sel], freqs[freq_sel], axis=1)
                    prop_band = ps_band/ps_tot * 100
                    prop_avg[region].append(np.nanmean(prop_band, axis=0))
                    prop_err[region].append(utils.stderr(prop_band, axis=0))
                    
                prop_avg_task[task] = prop_avg
                prop_err_task[task] = prop_err
                        
            plot_spectra(ps_avg, freqs, err_dict=ps_err, ax=ax_ps, title=plot_beh_names[task])
            
            if plot_band_prop:
                plot_vals = [prop_avg[r] for r in all_regions]
                plot_err = [prop_err[r] for r in all_regions]
                plot_utils.plot_stacked_bar(plot_vals, value_labels=all_regions, x_labels=band_labels, orientation='h', ax=ax_prop, err=plot_err,
                                            x_label_rot=x_rot)
                ax_prop.set_title(plot_beh_names[task])
                ax_prop.set_ylabel('Relative Power (%)')
                ax_prop.set_xlabel('Frequency Band')

        # compare task bands by region
        if plot_band_prop:
            for i, region in enumerate(all_regions):
                ax_prop_reg = axs_prop_reg[i]
                plot_vals = [prop_avg_task[t][region] for t in plot_tasks]
                plot_err = [prop_err_task[t][region] for t in plot_tasks]
                plot_utils.plot_stacked_bar(plot_vals, value_labels=[plot_beh_names[t] for t in plot_tasks], x_labels=band_labels, orientation='h', ax=ax_prop_reg, err=plot_err,
                                            x_label_rot=x_rot)
                ax_prop_reg.set_title(region)
                ax_prop_reg.set_ylabel('Relative Power (%)')
                ax_prop_reg.set_xlabel('Frequency Band')
            
        if save_plots:
            fpah.save_fig(fig_ps, fpah.get_figure_save_path('Power Spectra', subj_id, 'Meta Subject Session Avg {}'.format(signal_type)))
            if plot_band_prop:
                fpah.save_fig(fig_prop, fpah.get_figure_save_path('Power Spectra', subj_id, 'Meta Subject Session Avg {} power proportion'.format(signal_type)))
            
        if show_plots:
            plt.show()
        
        plt.close(fig_ps)
        if plot_band_prop:
            plt.close(fig_prop)
            
                            
# %% compute correlations by frequency band

tilt_t = False
baseline_correction = True
baseline_band_iso_fit = True
band_iso_fit = False
filter_dropout_outliers = False

freq_vals = [0] + list(np.logspace(np.log10(0.01), np.log10(10), 20))
freq_vals = np.round(freq_vals, 3)
freq_bands = list(zip(freq_vals[:-1], freq_vals[1:]))
#freq_bands = [[0,0.02], [0.02,0.1], [0.1,0.3], [0.3,1], [1,1.8], [1.8,4], [4,10]]
#freq_bands = [[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,1], [1,2], [2,6], [6,10]]

recalculate = False
reprocess_sess_ids = []

signal_types = ['dff_iso_baseline_fband'] # 'z_dff_iso_baseline' 

# get dt information
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']

tasks = ['wm', 'bandit']

filename = 'wm_bandit_fband_xcorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        fband_corrs = saved_data['fband_corrs']
        fband_corr_metadata = saved_data['metadata']
        fband_corr_regions = fband_corr_metadata['regions']
        
        recalculate = fband_corr_metadata['freq_bands'] != freq_bands or (fband_corr_metadata['engage_next_poke_thresh'] != engage_next_poke_thresh)

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    fband_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    fband_corr_regions = {}

for subj_id in subj_ids:
    
    subj_regions = list(implant_info[subj_id].keys())
    
    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]
        
    fband_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)
    
    if not subj_id in fband_corrs:
        fband_corrs[subj_id] = {task: {} for task in tasks}
    
    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids 
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
                                    
        if not subj_id in sess_ids:
            continue
        
        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in fband_corrs[subj_id][task] and not sess_id in reprocess_sess_ids and list(fband_corrs[subj_id][task][sess_id].keys()) == signal_types:
                continue
            
            if sess_id not in fband_corrs[subj_id][task]:
                fband_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, 
                                           band_iso_fit=band_iso_fit, baseline_band_iso_fit=baseline_band_iso_fit,
                                           filter_dropout_outliers=filter_dropout_outliers)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            
            sess_engaged_t_sel = engaged_t_sel[subj_id][task][sess_id]

            print('Calculating frequency band cross-correlation for subject {} session {}:'.format(subj_id, sess_id))
            
            for signal_type in signal_types:
                if signal_type in fband_corrs[subj_id][task][sess_id]:
                    continue
                else:
                    fband_corrs[subj_id][task][sess_id][signal_type] = {e: np.full((n_regions, n_regions, len(freq_bands)+1), np.nan, dtype=float) for e in engage_states}
                    
                for reg_i in range(n_regions):
                    if not subj_regions[reg_i] in fp_data['processed_signals']:
                        continue
                    signal_i = fp_data['processed_signals'][subj_regions[reg_i]][signal_type]
                    signal_i_fbands = fp_utils.decompose_signal_fbands(signal_i, freq_bands, 1/dt)
                    
                    for reg_j in np.arange(reg_i+1, n_regions):
                        if not subj_regions[reg_j] in fp_data['processed_signals']:
                            continue
                        signal_j = fp_data['processed_signals'][subj_regions[reg_j]][signal_type]
                        signal_j_fbands = fp_utils.decompose_signal_fbands(signal_j, freq_bands, 1/dt)
                        
                        start = time.perf_counter()
                        
                        for e in engage_states:
                            
                            if e == 'engaged':
                                t_sel = sess_engaged_t_sel
                            elif e == 'disengaged':
                                t_sel = ~sess_engaged_t_sel
                            else:
                                t_sel = None
                                
                            tot_xcorr, _ = fp_utils.correlate(signal_i, signal_j, dt, max_lag=0, t_sel=t_sel)
                            
                            fband_corrs[subj_id][task][sess_id][signal_type][e][reg_i, reg_j, -1] = tot_xcorr[0]
                            fband_corrs[subj_id][task][sess_id][signal_type][e][reg_j, reg_i, -1] = tot_xcorr[0]
                            
                            for k in range(len(freq_bands)):
                                band_xcorr, _ = fp_utils.correlate(signal_i_fbands[k,:], signal_j_fbands[k,:], dt, max_lag=0, t_sel=t_sel)
    
                                fband_corrs[subj_id][task][sess_id][signal_type][e][reg_i, reg_j, k] = band_xcorr[0]
                                fband_corrs[subj_id][task][sess_id][signal_type][e][reg_j, reg_i, k] = band_xcorr[0]
                                
                        print('  Between {} and {} in {:.1f} s'.format(
                            subj_regions[reg_i], subj_regions[reg_j], time.perf_counter()-start))
    
                with open(save_path, 'wb') as f:
                    pickle.dump({'fband_corrs': fband_corrs,
                                 'metadata': {'regions': fband_corr_regions,
                                              'freq_bands': freq_bands}},
                                f)

# %% Plot frequency band correlations

plot_sep_tasks = False
plot_sep_states = True
plot_ind_subj = True
plot_comb_subj = True
ax_height = 3
ax_width = 5
x_label_rot = 90

plot_tasks = []
if plot_sep_tasks:
    plot_tasks.extend(tasks.copy())
    
plot_tasks.append('all')

if plot_sep_states:
    plot_states = engage_states.copy()
else:
    plot_states = ['all']

plot_beh_names = {'wm': 'WM', 'bandit': 'Bandit', 'all': 'All'}
engage_labels = {'engaged': 'engaged', 'disengaged': 'disengaged', 'all': 'all'}
band_labels = ['{}-{}'.format(b[0], b[1]) for b in freq_bands]

task_names = [plot_beh_names[t] for t in plot_tasks]
engage_names = [engage_labels[e] for e in plot_states]

if plot_comb_subj:
    comb_fband_corr = {t: {s: {e: [] for e in plot_states} for s in signal_types} for t in tasks}
    comb_fband_weights = {t: {s: {e: [] for e in plot_states} for s in signal_types} for t in tasks}

for subj_id in subj_ids:
    subj_regions = fband_corr_regions[subj_id]
    sorted_regions = sort_subj_regions(subj_regions)
    sorted_region_idxs = [subj_regions.index(r) for r in sorted_regions]
    
    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]
    
    # stack across subjects
    if plot_comb_subj:
        for task in tasks:
            
            if len(fband_corrs[subj_id][task]) == 0:
                continue
            
            for signal_type in signal_types:
    
                # stack by session across regions
                for s in fband_corrs[subj_id][task].keys():
                    for e in plot_states:

                        sess_fband_corr = fband_corrs[subj_id][task][s][signal_type][e]
                        
                        # get mapping of subject regions to all regions order
                        all_region_idx_mapping = fpah.get_region_idx_mapping(subj_regions, all_regions)
                        dup_counts = fpah.count_corr_pairs(subj_regions, all_regions)
    
                        if all([v == 1 for v in dup_counts.values()]):
                            # if no duplicates, can simply use indexing
                            sess_comb_fband_corr = np.full((len(all_regions), len(all_regions), len(freq_bands)+1), np.nan, dtype=float)
                            sess_comb_fband_corr[np.ix_(all_region_idx_mapping, all_region_idx_mapping, np.arange(len(freq_bands)+1))] = sess_fband_corr
                            
                            comb_fband_corr[task][signal_type][e].append(sess_comb_fband_corr)
                            
                            weights = np.ones_like(sess_comb_fband_corr)
                            weights[np.isnan(sess_comb_fband_corr)] = np.nan
                            comb_fband_weights[task][signal_type][e].append(weights)
                        else:
                            # add each duplicate pair separately
                            for subj_i, map_i in enumerate(all_region_idx_mapping):
                                for subj_j, map_j in enumerate(all_region_idx_mapping):
                                    # only do each cross-regional comparison once and don't do autocorrelations
                                    if subj_j <= subj_i:
                                        continue
                                    # don't add bilateral cross-correlations of the same region as auto-correlations
                                    if map_i == map_j and subj_i != subj_j:
                                        continue
                                    
                                    sess_comb_fband_corr = np.full((len(all_regions), len(all_regions), len(freq_bands)+1), np.nan, dtype=float)
                                    sess_comb_fband_corr[map_i,map_j,:] = sess_fband_corr[subj_i,subj_j,:]
                                    if map_i != map_j:
                                        sess_comb_fband_corr[map_j,map_i,:] = sess_fband_corr[subj_j,subj_i,:]
    
                                    comb_fband_corr[task][signal_type][e].append(sess_comb_fband_corr)
                                    
                                    dup_key = tuple(sorted([map_i,map_j]))
                                    weights = np.full_like(sess_comb_fband_corr, 1/dup_counts[dup_key])
                                    weights[np.isnan(sess_comb_fband_corr)] = np.nan
                                    comb_fband_weights[task][signal_type][e].append(weights)
                            
    if plot_ind_subj:
        if plot_sep_states:
            for task in plot_tasks:
                for signal_type in signal_types:
                    
                    signal_title, _ = fpah.get_signal_type_labels(signal_type)
            
                    fig, axs = plt.subplots(n_regions-1, n_regions-1, figsize=(ax_width*(n_regions-1)+1, ax_height*(n_regions-1)), layout='constrained')
                    fig.suptitle('Regional Frequency Band Correlations for Subject {} in {}.\n{}'.format(subj_id, plot_beh_names[task], signal_title))
                    
                    if task == 'all':
                        stacked_subj_corrs = {e: np.concatenate([np.stack([fband_corrs[subj_id][t][s][signal_type][e] for s in fband_corrs[subj_id][t].keys()], axis=-1) for t in tasks], axis=3) for e in plot_states}
                    else:
                        stacked_subj_corrs = {e: np.stack([fband_corrs[subj_id][task][s][signal_type][e] for s in fband_corrs[subj_id][task].keys()], axis=-1) for e in plot_states}

                    subj_corr_avg = {e: np.nanmean(stacked_subj_corrs[e], axis=3) for e in plot_states}
                    subj_corr_err = {e: utils.stderr(stacked_subj_corrs[e], axis=3) for e in plot_states}
                    
                    colors = ['C{}'.format(i) for i in range(len(plot_states))]
                    
                    for reg_i in range(n_regions-1):   
                        sorted_i = sorted_region_idxs[reg_i]
                        for reg_j in np.arange(1,n_regions):
                            sorted_j = sorted_region_idxs[reg_j]
        
                            ax = axs[reg_i, reg_j-1]
                            
                            if reg_j <= reg_i:
                                ax.axis('off')
                                continue
                            
                            fband_vals = [subj_corr_avg[e][sorted_i, sorted_j, :-1] for e in plot_states]
                            fband_err = [subj_corr_err[e][sorted_i, sorted_j, :-1] for e in plot_states]
                            
                            tot_corr_vals = [subj_corr_avg[e][sorted_i, sorted_j, -1] for e in plot_states]
                            tot_corr_err = [subj_corr_err[e][sorted_i, sorted_j, -1] for e in plot_states]
                            
                            plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=engage_names, x_labels=band_labels, 
                                                        orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)
                            
                            for i, e in enumerate(plot_states):
                                plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax, 
                                                             color=colors[i], label='{} Avg'.format(engage_labels[e]), linestyle='--')
                            
                            ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'], 
                                                                     sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
        
                            ax.set_xlabel('Frequency Band')
                            ax.set_ylabel('Pearson r')
                            
                            ax.legend()
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend().remove()
                        
                    fig.legend(handles, labels, loc='outside right center')
        else:
            for signal_type in signal_types:
                
                signal_title, _ = fpah.get_signal_type_labels(signal_type)
        
                fig, axs = plt.subplots(n_regions-1, n_regions-1, figsize=(ax_width*(n_regions-1)+1, ax_height*(n_regions-1)), layout='constrained')
                fig.suptitle('Regional Frequency Band Correlations for Subject {}.\n{}'.format(subj_id, signal_title))
                
                stacked_subj_corrs = {t: np.stack([fband_corrs[subj_id][t][s][signal_type]['all'] for s in fband_corrs[subj_id][t].keys()], axis=-1) for t in tasks}
                stacked_subj_corrs['all'] = np.concatenate([stacked_subj_corrs[t] for t in tasks], axis=3)
                
                subj_corr_avg = {t: np.nanmean(stacked_subj_corrs[t], axis=3) for t in plot_tasks}
                subj_corr_err = {t: utils.stderr(stacked_subj_corrs[t], axis=3) for t in plot_tasks}
                
                colors = ['C{}'.format(i) for i in range(len(plot_tasks))]
                
                for reg_i in range(n_regions-1):   
                    sorted_i = sorted_region_idxs[reg_i]
                    for reg_j in np.arange(1,n_regions):
                        sorted_j = sorted_region_idxs[reg_j]
    
                        ax = axs[reg_i, reg_j-1]
                        
                        if reg_j <= reg_i:
                            ax.axis('off')
                            continue
                        
                        fband_vals = [subj_corr_avg[t][sorted_i, sorted_j, :-1] for t in plot_tasks]
                        fband_err = [subj_corr_err[t][sorted_i, sorted_j, :-1] for t in plot_tasks]
                        
                        tot_corr_vals = [subj_corr_avg[t][sorted_i, sorted_j, -1] for t in plot_tasks]
                        tot_corr_err = [subj_corr_err[t][sorted_i, sorted_j, -1] for t in plot_tasks]
                        
                        plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=task_names, x_labels=band_labels, 
                                                    orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)
                        
                        for i, t in enumerate(plot_tasks):
                            plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax, 
                                                         color=colors[i], label='{} Avg'.format(plot_beh_names[t]), linestyle='--')
                        
                        ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'], 
                                                                 sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
    
                        ax.set_xlabel('Frequency Band')
                        ax.set_ylabel('Pearson r')
                        
                        ax.legend()
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend().remove()
                    
                fig.legend(handles, labels, loc='outside right center')
                   
# plot combined subject averages
if plot_comb_subj:
    # stack all matrices together
    comb_fband_corr = {t: {s: {e: np.stack(comb_fband_corr[t][s][e], axis=-1) for e in plot_states} for s in signal_types} for t in tasks}
    comb_fband_weights = {t: {s: {e: np.stack(comb_fband_weights[t][s][e], axis=-1) for e in plot_states} for s in signal_types} for t in tasks}

    if plot_sep_states:
        for task in plot_tasks:
            for signal_type in signal_types:
                signal_title, _ = fpah.get_signal_type_labels(signal_type)
            
                fig, axs = plt.subplots(len(all_regions)-1, len(all_regions)-1, 
                                        figsize=(ax_width*(len(all_regions)-1)+1, ax_height*(len(all_regions)-1)), layout='constrained')
                fig.suptitle('Regional Frequency Band Correlations for All Subjects in {}.\n{}'.format(plot_beh_names[task], signal_title))
                
                if task == 'all':
                    stacked_subj_corrs = {e: np.concatenate([comb_fband_corr[t][signal_type][e] for t in tasks], axis=3) for e in plot_states}
                    stacked_subj_weights = {e: np.concatenate([comb_fband_weights[t][signal_type][e] for t in tasks], axis=3) for e in plot_states}
                else:
                    stacked_subj_corrs = {e: comb_fband_corr[task][signal_type][e] for e in plot_states}
                    stacked_subj_weights = {e: comb_fband_weights[task][signal_type][e] for e in plot_states}
                
                subj_corr_avg = {e: utils.weighted_mean(stacked_subj_corrs[e], stacked_subj_weights[e], axis=3) for e in plot_states}
                subj_corr_err = {e: utils.weighted_se(stacked_subj_corrs[e], stacked_subj_weights[e], axis=3) for e in plot_states}
                
                colors = ['C{}'.format(i) for i in range(len(plot_states))]
                
                for reg_i in range(len(all_regions)-1):   
                    for reg_j in np.arange(1, len(all_regions)):
        
                        ax = axs[reg_i, reg_j-1]
                        
                        if reg_j <= reg_i:
                            ax.axis('off')
                            continue
                        
                        fband_vals = [subj_corr_avg[e][reg_i, reg_j, :-1] for e in plot_states]
                        fband_err = [subj_corr_err[e][reg_i, reg_j, :-1] for e in plot_states]
                        
                        tot_corr_vals = [subj_corr_avg[e][reg_i, reg_j, -1] for e in plot_states]
                        tot_corr_err = [subj_corr_err[e][reg_i, reg_j, -1] for e in plot_states]
                        
                        plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=engage_names, x_labels=band_labels, 
                                                    orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)
                        
                        for i, e in enumerate(plot_states):
                            plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax, 
                                                         color=colors[i], label='{} Avg'.format(engage_labels[e]), linestyle='--')
                        
                        ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
        
                        ax.set_xlabel('Frequency Band')
                        ax.set_ylabel('Pearson r')
                        
                        ax.legend()
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend().remove()
                    
                fig.legend(handles, labels, loc='outside right center')    
                
    else:
        for signal_type in signal_types:
            signal_title, _ = fpah.get_signal_type_labels(signal_type)
        
            fig, axs = plt.subplots(len(all_regions)-1, len(all_regions)-1, 
                                    figsize=(ax_width*(len(all_regions)-1)+1, ax_height*(len(all_regions)-1)), layout='constrained')
            fig.suptitle('Regional Frequency Band Correlations for All Subjects.\n{}'.format(signal_title))
            
            stacked_subj_corrs = {t: comb_fband_corr[t][signal_type]['all'] for t in tasks}
            stacked_subj_corrs['all'] = np.concatenate([stacked_subj_corrs[t] for t in tasks], axis=3)
            
            stacked_subj_weights = {t: comb_fband_weights[t][signal_type]['all'] for t in tasks}
            stacked_subj_weights['all'] = np.concatenate([stacked_subj_weights[t] for t in tasks], axis=3)
            
            subj_corr_avg = {t: utils.weighted_mean(stacked_subj_corrs[t], stacked_subj_weights[t], axis=3) for t in plot_tasks}
            subj_corr_err = {t: utils.weighted_se(stacked_subj_corrs[t], stacked_subj_weights[t], axis=3) for t in plot_tasks}
            
            colors = ['C{}'.format(i) for i in range(len(plot_tasks))]
            
            for reg_i in range(len(all_regions)-1):   
                for reg_j in np.arange(1, len(all_regions)):
    
                    ax = axs[reg_i, reg_j-1]
                    
                    if reg_j <= reg_i:
                        ax.axis('off')
                        continue
                    
                    fband_vals = [subj_corr_avg[t][reg_i, reg_j, :-1] for t in plot_tasks]
                    fband_err = [subj_corr_err[t][reg_i, reg_j, :-1] for t in plot_tasks]
                    
                    tot_corr_vals = [subj_corr_avg[t][reg_i, reg_j, -1] for t in plot_tasks]
                    tot_corr_err = [subj_corr_err[t][reg_i, reg_j, -1] for t in plot_tasks]
                    
                    plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=task_names, x_labels=band_labels, 
                                                orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)
                    
                    for i, t in enumerate(plot_tasks):
                        plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax, 
                                                     color=colors[i], label='{} Avg'.format(plot_beh_names[t]), linestyle='--')
                    
                    ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
    
                    ax.set_xlabel('Frequency Band')
                    ax.set_ylabel('Pearson r')
                    
                    ax.legend()
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()
                
            fig.legend(handles, labels, loc='outside right center')    

# %% compute correlations over time
t_width = 0.5
recalculate = False

filename = 'wm_bandit_tcorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        t_corrs = saved_data['t_corrs']
        t_corr_metadata = saved_data['metadata']
        t_corr_regions = t_corr_metadata['regions']
        
        recalculate = t_corr_metadata['t_width'] != t_width

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    t_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    t_corr_regions = {subjid: [] for subjid in subj_ids}

for subj_id in subj_ids:
    
    subj_regions = list(implant_info[subj_id].keys())
    
    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]
        
    if not subj_id in t_corrs:
        t_corrs[subj_id] = {task: {} for task in tasks}
        
    t_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)
    
    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids 
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
                                    
        if not subj_id in sess_ids:
            continue
        
        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in t_corrs[subj_id][task] and not sess_id in reprocess_sess_ids:
                continue
            else:
                t_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, filter_dropout_outliers=filter_dropout_outliers)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            
            t_corrs[subj_id][task][sess_id]['t'] = t
            
            print('Calculating correlation over time for subject {} session {}'.format(subj_id, sess_id))
            
            for reg_i in range(n_regions):
                signal_i = fp_data['processed_signals'][subj_regions[reg_i]][signal_type]
                
                for reg_j in np.arange(reg_i+1, n_regions):
                    signal_j = fp_data['processed_signals'][subj_regions[reg_j]][signal_type]
                    
                    start = time.perf_counter()                    
                    
                    t_corr = fp_utils.correlate_over_time(signal_i, signal_j, dt, t_width=t_width)

                    t_corrs[subj_id][task][sess_id]['{} & {}'.format(subj_regions[reg_i], subj_regions[reg_j])] = t_corr
                    
            with open(save_path, 'wb') as f:
                pickle.dump({'t_corrs': t_corrs,
                             'metadata': {'regions': t_corr_regions,
                                         't_width': t_width}},
                            f)

# %% Investigate engaged vs disengaged

signal_types = ['z_dff_iso_baseline', 'z_dff_iso_fband']
plot_tasks = ['wm']
dec=2
baseline_correction = True
tilt_t = True

for subj_id in subj_ids:
    
    subj_regions = x_corr_regions[subj_id]
    n_regions = len(subj_regions)

    for task in plot_tasks:
        if task == 'wm':
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
            
        sess_ids = list(x_corrs[subj_id][task].keys())

        for sess_id in sess_ids:

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, filter_dropout_outliers=False)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
                
            sess_engaged_t_sel = engaged_t_sel[subj_id][task][sess_id] 
            
            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_start_ts = fp_data['trial_start_ts'][:-1]

            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            
            response = ~np.isnan(trial_data['response_time']).to_numpy()
            reward = (trial_data['rewarded'] == True).to_numpy()
            unreward = (trial_data['rewarded'] == False).to_numpy() & response
            
            lines_dict = {'Rewarded': cpoke_out_ts[reward], 'Unrewarded': cpoke_out_ts[unreward]}
            
            if task == 'wm':
                bail = (trial_data['bail'] == True).to_numpy()
                lines_dict['Bail'] = cpoke_out_ts[bail]

            for signal_type in signal_types:
                fig, axs = plt.subplots(len(subj_regions), 1, sharex=True, layout='constrained', figsize=(12,3*len(subj_regions)))
                
                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))
                
                for i, region in enumerate(subj_regions):
                    ax = axs[i]
                    
                    if not region in fp_data['processed_signals']:
                        continue
                    signal = fp_data['processed_signals'][region][signal_type]
                    
                    ax.plot(t[::dec], signal[::dec])
                    ax.fill_between(t, 0, 1, where=sess_engaged_t_sel,
                                    color='grey', alpha=0.4, transform=ax.get_xaxis_transform())
                    
                    for j, (name, lines) in enumerate(lines_dict.items()):
                        ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j), linestyles='dashed', 
                                  transform=ax.get_xaxis_transform())
                        
                    ax.set_title(region)
                    ax.legend()
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()
                    
                fig.legend(handles, labels, loc='outside right upper')
                    
            plt.show()


# %% Plot helper

def plot_signal_details(signal_mat, t, title, col_names, lines_dict={}, dec=10):

    t = t[::dec].copy()

    fig, axs = plt.subplots(len(col_names), 1, layout='constrained', figsize=[18,6*len(col_names)], sharex=True)
    if len(col_names) == 1:
        axs = [axs]
        
    fig.suptitle(title)

    for i, col_name in enumerate(col_names):

        ax = axs[i]
        ax.plot(t, signal_mat[::dec,i], label='_')

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j), linestyles='dashed', 
                      transform=ax.get_xaxis_transform())

        ax.set_title(col_name)
        ax.set_xlabel('Time (s)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend()

# %% Plot correlations over time with behavioral events

plot_tasks = tasks.copy()

beh_names = {'wm': 'WM Task', 'bandit': 'Bandit Task'}

for subj_id in subj_ids:
    
    subj_regions = t_corr_regions[subj_id]
    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]
    
    for task in tasks:
        
        if task == 'wm':
            sess_ids = wm_sess_ids 
            loc_db = wm_loc_db 
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data
                                    
        if not subj_id in sess_ids:
            continue
        
        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            trial_data = sess_data[sess_data['sessid'] == sess_id]
            
            trial_start_ts = loc_db.get_sess_fp_data([sess_id])['fp_data'][subj_id][sess_id]['trial_start_ts'][:-1]

            cport_on_ts = trial_start_ts + trial_data['cport_on_time']
            cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
            cue_ts = trial_start_ts + trial_data['response_cue_time']
            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            response_ts = trial_start_ts + trial_data['response_time']
            outcome_ts = trial_start_ts + trial_data['reward_time']
            reward_ts = outcome_ts[trial_data['reward'] > 0]
            unreward_ts = outcome_ts[trial_data['reward'] == 0]

            lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                          'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                          'Reward': reward_ts, 'Unreward': unreward_ts}
            
            if task == 'wm':
                tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
                lines_dict.update({'Tone Start': tone_ts})
            
            corr_keys = list(t_corrs[subj_id][task][sess_id].keys())
            corr_keys = [k for k in corr_keys if k != 't']

            t = t_corrs[subj_id][task][sess_id]['t']
            corr_mat = np.stack([t_corrs[subj_id][task][sess_id][k] for k in corr_keys], axis=1)
            
            plot_signal_details(corr_mat, t, 'Regional correlations over time for Subject {} in {}'.format(subj_id, beh_names[task]), 
                                corr_keys, lines_dict=lines_dict)
                
                    
        
# %% Regional Comparisons - ICA

from sklearn.decomposition import FastICA, PCA

tilt_t = False
baseline_correction = True
filter_dropout_outliers = False

recalculate = True
reprocess_sess_ids = []

signal_type = 'z_dff_iso_baseline' # 'z_dff_iso_baseline' 

#tasks = ['wm', 'bandit']

# filename = 'wm_bandit_xcorr_data'
# save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

# if path.exists(save_path) and not recalculate:
#     with open(save_path, 'rb') as f:
#         saved_data = pickle.load(f)
#         x_corrs = saved_data['x_corrs']
#         x_corr_metadata = saved_data['metadata']
#         x_corr_regions = x_corr_metadata['regions']
#         corr_lags = x_corr_metadata['corr_lags']
        
#         recalculate = x_corr_metadata['max_lag'] != max_lag

# elif not path.exists(save_path):
#     recalculate = True

# if recalculate:
#     x_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
#     x_corr_regions = {subjid: [] for subjid in subj_ids}

subj_ids = [199]
sess_ids = [117251, 117317, 117450]
loc_db = bandit_loc_db

pca_components = {}
ica_mix_mats = {}

for subj_id in subj_ids:
    
    subj_regions = list(implant_info[subj_id].keys())
    
    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]
        
    n_regions = len(subj_regions)
    
    pca_components[subj_id] = {}
    ica_mix_mats[subj_id] = {}

    for sess_id in sess_ids:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, baseline_correction=baseline_correction, tilt_t=tilt_t, filter_dropout_outliers=filter_dropout_outliers)
        fp_data = fp_data[subj_id][sess_id]

        # build signal matrix
        signal_mat = np.stack([fp_data['processed_signals'][r][signal_type] for r in subj_regions], axis=1)
        
        iso_mat = np.stack([fp_data['processed_signals'][r]['raw_iso'] for r in subj_regions], axis=1)
        
        # remove nans
        nans = np.any(np.isnan(signal_mat), axis=1)
        signal_mat = signal_mat[~nans,:]

        # do ICA
        ica = FastICA()
        signal_ica = ica.fit_transform(signal_mat)
        ica_mix_mat = ica.mixing_
        
        # rearrange columns of mixing matrix so that the diagonal has all the largest values
        col_sel = []
        col_rect = []
        all_cols = np.arange(n_regions)
        for i in range(n_regions):
            max_idx = np.argmax(np.abs(ica_mix_mat[i,all_cols]))
            max_idx = all_cols[max_idx]
            all_cols = all_cols[all_cols != max_idx]
            col_sel.append(max_idx)
            max_col_val_idx = np.argmax(np.abs(ica_mix_mat[:,max_idx]))
            if ica_mix_mat[max_col_val_idx,max_idx] < 0:
                col_rect.append(-1)
            else:
                col_rect.append(1)
                
        ica_mix_mat = ica_mix_mat[:,col_sel]
        signal_ica = signal_ica[:, col_sel]
        rect_mult = np.array(col_rect)[None,:]
        ica_mix_mat *= rect_mult
        signal_ica *= rect_mult
        
        # fig, axs = plt.subplots(n_regions, 1, sharex=True, layout='constrained')
        # for i in range(n_regions):
        #     ax = axs[i]
        #     ax.set_title(subj_regions[i])
        #     ax.plot(fp_data['time'][~nans], signal_mat[:,i], label='Signal', alpha=0.5)
        #     ax.plot(fp_data['time'][~nans], signal_ica[:,i], label='ICA', alpha=0.5)
        #     ax.legend()
        
        # do PCA
        pca = PCA()
        signal_pca = pca.fit_transform(signal_mat)
        explained_var = pca.explained_variance_ratio_
        components = pca.components_
        
        pca_components[subj_id][sess_id] = components
        ica_mix_mats[subj_id][sess_id] = ica_mix_mat
        
        # ica = FastICA()
        # signal_pca_ica = ica.fit_transform(signal_pca)
        # pca_ica_mix_mat = ica.mixing_
        
        # # rectify columns of mixing matrix so that the diagonal has all the largest values
        # col_sel = []
        # col_rect = []
        # all_cols = np.arange(n_regions)
        # for i in range(n_regions):
        #     max_idx = np.argmax(np.abs(pca_ica_mix_mat[i,all_cols]))
        #     max_idx = all_cols[max_idx]
        #     all_cols = all_cols[all_cols != max_idx]
        #     col_sel.append(max_idx)
        #     max_col_val_idx = np.argmax(np.abs(pca_ica_mix_mat[:,max_idx]))
        #     if pca_ica_mix_mat[max_col_val_idx,max_idx] < 0:
        #         col_rect.append(-1)
        #     else:
        #         col_rect.append(1)
                
        # pca_ica_mix_mat = pca_ica_mix_mat[:,col_sel]
        # signal_pca_ica = signal_pca_ica[:, col_sel]
        # rect_mult = np.array(col_rect)[None,:]
        # pca_ica_mix_mat *= rect_mult
        # signal_pca_ica *= rect_mult
        
        
        # A_full = components.T @ pca_ica_mix_mat

        trial_data = loc_db.get_behavior_data(sess_id)

        ts = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        title = 'Subject {}, Session {}'.format(subj_id, sess_id)
        # plot_signal_details(signal_mat, ts[~nans], title, subj_regions, lines_dict)

        # plot_signal_details(signal_pca, ts[~nans], title, 
        #                     ['PC {}'.format(i) for i in range(n_regions)], lines_dict)
        
        # plot_signal_details(iso_mat, ts, title, subj_regions, lines_dict)
        
        # plot_signal_details(signal_pca_ica, ts[~nans], 
        #                     ['PC IC {}'.format(i) for i in range(n_regions)], lines_dict)
        
        
