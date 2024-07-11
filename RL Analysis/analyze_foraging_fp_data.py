# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:33:24 2023

@author: tanne
"""

# %% imports

import init
import pandas as pd
from pyutils import utils
from sys_neuro_tools import plot_utils, fp_utils
from hankslab_db import db_access
import hankslab_db.basicRLtasks_db as db
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
import beh_analysis_helpers as bah
import numpy as np
import matplotlib.pyplot as plt
import copy

# %% Load behavior data

# used for saving plots
behavior_name = 'Foraging'

# get all session ids for given protocol
sess_ids = db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 4)

# optionally limit sessions based on subject ids
subj_ids = [179]
sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

loc_db = db.LocalDB_BasicRLTasks('foraging')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids)) # reload=True

# fix for bug with persisting response and reward time on older versions of protocol
sess_data.loc[~sess_data['hit'], ['response_time', 'reward_time']] = np.nan

# %% Get and process photometry data

# get fiber photometry data
fp_data = loc_db.get_sess_fp_data(utils.flatten(sess_ids)) # , reload=True
# separate into different dictionaries
implant_info = fp_data['implant_info']
fp_data = fp_data['fp_data']

iso = '420'
lig = '490'

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        raw_signals = fp_data[subj_id][sess_id]['raw_signals']

        fp_data[subj_id][sess_id]['processed_signals'] = {}

        for region in raw_signals.keys():
            raw_lig = raw_signals[region][lig]
            raw_iso = raw_signals[region][iso]

            fp_data[subj_id][sess_id]['processed_signals'][region] = fpah.get_all_processed_signals(raw_lig, raw_iso)

# %% Observe the full signals

sub_signal = [] # sub signal time limits in seconds
filter_outliers = True
save_plots = True
show_plots = True

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        # Get the block transition trial start times
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        block_start_times = trial_start_ts[trial_data['block_trial'] == 1]
        block_rewards = trial_data['initial_reward'][trial_data['block_trial'] == 1]

        if len(sub_signal) > 0:
            fig = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
                                        title='Full Signals - Session {}'.format(sess_id),
                                        vert_marks=block_start_times, filter_outliers=filter_outliers,
                                        t_min=sub_signal[0], t_max=sub_signal[1], dec=1)
        else:
            fig = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
                                        title='Full Signals - Session {}. Initial Rewards: {}'.format(sess_id, ', '.join([str(r) for r in block_rewards])),
                                        vert_marks=block_start_times, filter_outliers=filter_outliers)

        if save_plots:
            fpah.save_fig(fig, fpah.get_figure_save_path(behavior_name, subj_id, 'sess_{}'.format(sess_id)))

        if not show_plots:
            plt.close(fig)

# %% Get all aligned/sorted stacked signals

signal_types = ['z_dff_iso'] # 'baseline_corr_lig','baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}
cue = copy.deepcopy(data_dict)
resp = copy.deepcopy(data_dict)
cue_resp = copy.deepcopy(data_dict)

initial_rewards = np.unique(sess_data['initial_reward'])
decay_rates = np.unique(sess_data['depletion_rate'])
switch_delays = np.unique(sess_data['patch_switch_delay'])
sides = ['left', 'right']

# get reward bins output by pandas for indexing
rew_bin_edges = [0, 10, 20, *initial_rewards]
rew_bins = pd.IntervalIndex.from_breaks(rew_bin_edges)
rew_bin_strs = {b:'{:.0f}-{:.0f} μL'.format(b.left, b.right) for b in rew_bins}

# declare settings for normalized cue to response intervals
norm_cue_resp_bins = 200

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        # get alignment trial filters
        resp_sel = trial_data['hit'] == True
        right_choice = trial_data['chose_right'].to_numpy()
        left_choice = trial_data['chose_left'].to_numpy()
        center_choice = trial_data['chose_center'].to_numpy()
        side_choice = right_choice | left_choice

        prev_no_resp = np.insert(~resp_sel[:-1], 0, False)
        prev_side_choice = np.insert(side_choice[:-1], 0, False)
        prev_center_choice = np.insert(center_choice[:-1], 0, False)

        future_no_resp = np.append(~resp_sel[1:], False)
        future_side_choice = np.append(side_choice[1:], False)
        future_center_choice = np.append(center_choice[1:], False)

        reward_port = trial_data['reward_port'].to_numpy()

        initial_reward = trial_data['initial_reward'].to_numpy()
        decay_rate = trial_data['depletion_rate'].to_numpy()
        switch_delay = trial_data['patch_switch_delay'].to_numpy()

        rew_bin = pd.cut(trial_data['reward'], rew_bins)
        prev_rew_bin = pd.cut(trial_data['prev_reward'], rew_bins)

        # get alignment times
        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        resp_ts = trial_start_ts + trial_data['response_time']

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # aligned to response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cue_ts, pre, post)
                align_dict = cue

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['no_resp'] = mat[~resp_sel,:]
                align_dict[sess_id][signal_type][region]['center_choice'] = mat[center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice'] = mat[side_choice,:]

                align_dict[sess_id][signal_type][region]['prev_no_resp'] = mat[prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['prev_center_choice'] = mat[prev_center_choice,:]
                align_dict[sess_id][signal_type][region]['prev_side_choice'] = mat[prev_side_choice,:]

                align_dict[sess_id][signal_type][region]['side_choice_prev_no_resp'] = mat[side_choice & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['side_choice_prev_center_choice'] = mat[side_choice & prev_center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice_prev_side_choice'] = mat[side_choice & prev_side_choice,:]

                align_dict[sess_id][signal_type][region]['center_choice_prev_no_resp'] = mat[center_choice & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['center_choice_prev_side_choice'] = mat[center_choice & prev_side_choice,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = reward_port == side

                    align_dict[sess_id][signal_type][region][side_type+'_no_resp'] = mat[~resp_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_center_choice'] = mat[center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice'] = mat[side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_no_resp'] = mat[prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_center_choice'] = mat[prev_center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_side_choice'] = mat[prev_side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_no_resp'] = mat[side_choice & prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_center_choice'] = mat[side_choice & prev_center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_side_choice'] = mat[side_choice & prev_side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_center_choice_prev_no_resp'] = mat[center_choice & prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_center_choice_prev_side_choice'] = mat[center_choice & prev_side_choice & side_sel,:]

                for rb in rew_bins:
                    rew_sel = rew_bin == rb
                    prev_rew_sel = prev_rew_bin == rb
                    bin_str = rew_bin_strs[rb]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice'] = mat[side_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice'] = mat[side_choice & prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice'] = mat[center_choice & prev_rew_sel,:]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_prev_center_choice'] = mat[prev_center_choice & rew_sel,:]

                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_prev_side_choice'] = mat[side_choice & prev_rew_sel & prev_side_choice,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_prev_no_resp'] = mat[side_choice & prev_rew_sel & prev_no_resp,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_prev_side_choice'] = mat[center_choice & prev_rew_sel & prev_side_choice,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_prev_no_resp'] = mat[center_choice & prev_rew_sel & prev_no_resp,:]

                    for ir in initial_rewards:
                        ir_sel = initial_reward == ir

                        align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_init_'+str(ir)] = mat[side_choice & prev_side_choice & prev_rew_sel & ir_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_init_'+str(ir)] = mat[center_choice & prev_side_choice & prev_rew_sel & ir_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = reward_port == side

                            align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_init_'+str(ir)] = mat[side_choice & prev_side_choice & prev_rew_sel & ir_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_init_'+str(ir)] = mat[center_choice & prev_side_choice & prev_rew_sel & ir_sel & side_sel,:]


                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = reward_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice'] = mat[side_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice'] = mat[side_choice & prev_rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice'] = mat[center_choice & prev_rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & prev_rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_prev_center_choice'] = mat[prev_center_choice & rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_prev_side_choice'] = mat[side_choice & prev_rew_sel & prev_side_choice & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_prev_no_resp'] = mat[side_choice & prev_rew_sel & prev_no_resp & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_prev_side_choice'] = mat[center_choice & prev_rew_sel & prev_side_choice & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_prev_no_resp'] = mat[center_choice & prev_rew_sel & prev_no_resp & side_sel,:]


                # aligned to response poke
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, resp_ts, pre, post)
                align_dict = resp

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['center_choice'] = mat[center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice'] = mat[side_choice,:]

                align_dict[sess_id][signal_type][region]['prev_no_resp'] = mat[prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['prev_center_choice'] = mat[prev_center_choice,:]
                align_dict[sess_id][signal_type][region]['prev_side_choice'] = mat[prev_side_choice,:]

                align_dict[sess_id][signal_type][region]['side_choice_prev_no_resp'] = mat[side_choice & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['side_choice_prev_center_choice'] = mat[side_choice & prev_center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice_prev_side_choice'] = mat[side_choice & prev_side_choice,:]

                align_dict[sess_id][signal_type][region]['center_choice_prev_no_resp'] = mat[center_choice & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['center_choice_prev_side_choice'] = mat[center_choice & prev_side_choice,:]

                align_dict[sess_id][signal_type][region]['side_choice_future_no_resp'] = mat[side_choice & future_no_resp,:]
                align_dict[sess_id][signal_type][region]['side_choice_future_center_choice'] = mat[side_choice & future_center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice_future_side_choice'] = mat[side_choice & future_side_choice,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = reward_port == side

                    align_dict[sess_id][signal_type][region][side_type+'_center_choice'] = mat[center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice'] = mat[side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_no_resp'] = mat[prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_center_choice'] = mat[prev_center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_side_choice'] = mat[prev_side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_no_resp'] = mat[side_choice & prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_center_choice'] = mat[side_choice & prev_center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_side_choice'] = mat[side_choice & prev_side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_center_choice_prev_no_resp'] = mat[center_choice & prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_center_choice_prev_side_choice'] = mat[center_choice & prev_side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_future_no_resp'] = mat[side_choice & future_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_future_center_choice'] = mat[side_choice & future_center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_future_side_choice'] = mat[side_choice & future_side_choice & side_sel,:]


                for rb in rew_bins:
                    rew_sel = rew_bin == rb
                    prev_rew_sel = prev_rew_bin == rb
                    bin_str = rew_bin_strs[rb]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice'] = mat[side_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice'] = mat[side_choice & prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice'] = mat[center_choice & prev_rew_sel,:]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_prev_center_choice'] = mat[prev_center_choice & rew_sel,:]

                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_prev_side_choice'] = mat[side_choice & prev_rew_sel & prev_side_choice,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_prev_no_resp'] = mat[side_choice & prev_rew_sel & prev_no_resp,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_prev_side_choice'] = mat[center_choice & prev_rew_sel & prev_side_choice,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_prev_no_resp'] = mat[center_choice & prev_rew_sel & prev_no_resp,:]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice_future_no_resp'] = mat[side_choice & future_no_resp & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice_future_center_choice'] = mat[side_choice & future_center_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice_future_side_choice'] = mat[side_choice & future_side_choice & rew_sel,:]

                    for ir in initial_rewards:
                        ir_sel = initial_reward == ir

                        align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice_init_'+str(ir)] = mat[side_choice & prev_side_choice & rew_sel & ir_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_init_'+str(ir)] = mat[center_choice & prev_side_choice & prev_rew_sel & ir_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = reward_port == side

                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice_init_'+str(ir)] = mat[side_choice & prev_side_choice & rew_sel & ir_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_init_'+str(ir)] = mat[center_choice & prev_side_choice & prev_rew_sel & ir_sel & side_sel,:]

                        for dr in decay_rates:
                            dr_sel = decay_rate == dr

                            align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice_init_'+str(ir)+'_decay_'+str(dr)] = mat[side_choice & prev_side_choice & rew_sel & ir_sel & dr_sel,:]

                            for side in sides:
                                side_type = fpah.get_implant_side_type(side, region_side)
                                side_sel = reward_port == side

                                align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice_init_'+str(ir)+'_decay_'+str(dr)] = mat[side_choice & prev_side_choice & rew_sel & ir_sel & dr_sel & side_sel,:]


                    for dr in decay_rates:
                        dr_sel = decay_rate == dr

                        align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice_decay_'+str(dr)] = mat[side_choice & prev_side_choice & rew_sel & dr_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = reward_port == side

                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice_decay_'+str(dr)] = mat[side_choice & prev_side_choice & rew_sel & dr_sel & side_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = reward_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice'] = mat[side_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice'] = mat[side_choice & prev_rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice'] = mat[center_choice & prev_rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & prev_rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_prev_center_choice'] = mat[prev_center_choice & rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_prev_side_choice'] = mat[side_choice & prev_rew_sel & prev_side_choice & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_prev_no_resp'] = mat[side_choice & prev_rew_sel & prev_no_resp & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_prev_side_choice'] = mat[center_choice & prev_rew_sel & prev_side_choice & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_prev_no_resp'] = mat[center_choice & prev_rew_sel & prev_no_resp & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice_future_no_resp'] = mat[side_choice & future_no_resp & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice_future_center_choice'] = mat[side_choice & future_center_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice_future_side_choice'] = mat[side_choice & future_side_choice & rew_sel & side_sel,:]

                # time normalized signal matrices
                mat = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, resp_ts, norm_cue_resp_bins)
                align_dict = cue_resp

                align_dict['t'] = np.linspace(0, 1, norm_cue_resp_bins)
                align_dict[sess_id][signal_type][region]['center_choice'] = mat[center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice'] = mat[side_choice,:]

                align_dict[sess_id][signal_type][region]['side_choice_prev_no_resp'] = mat[side_choice & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['side_choice_prev_center_choice'] = mat[side_choice & prev_center_choice,:]
                align_dict[sess_id][signal_type][region]['side_choice_prev_side_choice'] = mat[side_choice & prev_side_choice,:]

                align_dict[sess_id][signal_type][region]['center_choice_prev_no_resp'] = mat[center_choice & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['center_choice_prev_side_choice'] = mat[center_choice & prev_side_choice,:]


                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = reward_port == side

                    align_dict[sess_id][signal_type][region][side_type+'_center_choice'] = mat[center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice'] = mat[side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_no_resp'] = mat[side_choice & prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_center_choice'] = mat[side_choice & prev_center_choice & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_side_choice_prev_side_choice'] = mat[side_choice & prev_side_choice & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_center_choice_prev_no_resp'] = mat[center_choice & prev_no_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_center_choice_prev_side_choice'] = mat[center_choice & prev_side_choice & side_sel,:]


                for rb in rew_bins:
                    rew_sel = rew_bin == rb
                    prev_rew_sel = prev_rew_bin == rb
                    bin_str = rew_bin_strs[rb]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_side_choice'] = mat[side_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice'] = mat[side_choice & prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice'] = mat[center_choice & prev_rew_sel,:]

                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+bin_str+'_prev_center_choice'] = mat[prev_center_choice & rew_sel,:]

                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_prev_side_choice'] = mat[side_choice & prev_rew_sel & prev_side_choice,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_prev_no_resp'] = mat[side_choice & prev_rew_sel & prev_no_resp,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_prev_side_choice'] = mat[center_choice & prev_rew_sel & prev_side_choice,:]
                    align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_prev_no_resp'] = mat[center_choice & prev_rew_sel & prev_no_resp,:]

                    for ir in initial_rewards:
                        ir_sel = initial_reward == ir

                        align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_side_choice_init_'+str(ir)] = mat[side_choice & prev_side_choice & prev_rew_sel & ir_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_rew_'+bin_str+'_center_choice_init_'+str(ir)] = mat[center_choice & prev_side_choice & prev_rew_sel & ir_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = reward_port == side

                            align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_init_'+str(ir)] = mat[side_choice & prev_side_choice & prev_rew_sel & ir_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_init_'+str(ir)] = mat[center_choice & prev_side_choice & prev_rew_sel & ir_sel & side_sel,:]


                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = reward_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_side_choice'] = mat[side_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice'] = mat[side_choice & prev_rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice'] = mat[center_choice & prev_rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_prev_side_choice'] = mat[prev_side_choice & prev_rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+bin_str+'_prev_center_choice'] = mat[prev_center_choice & rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_prev_side_choice'] = mat[side_choice & prev_rew_sel & prev_side_choice & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_side_choice_prev_no_resp'] = mat[side_choice & prev_rew_sel & prev_no_resp & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_prev_side_choice'] = mat[center_choice & prev_rew_sel & prev_side_choice & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rew_'+bin_str+'_center_choice_prev_no_resp'] = mat[center_choice & prev_rew_sel & prev_no_resp & side_sel,:]


# %% Set up average plot options

# modify these options to change what will be used in the average signal plots
signal_type = 'z_dff_iso' # 'dff_iso', 'df_baseline_iso', 'raw_lig'
signal_label = 'Z-scored ΔF/F'
regions = ['DMS', 'PL']
subjects = list(sess_ids.keys())
filter_outliers = True
outlier_thresh = 20
use_se = True
ph = 3.5;
pw = 5;
n_reg = len(regions)
resp_xlims = {'DMS': [-1.5,2], 'PL': [-3,10]}
gen_xlims = {'DMS': [-1.5,1.5], 'PL': [-3,3]}

save_plots = True
show_plots = False
reward_time = 0.5

# make this wrapper to simplify the stack command by not having to include the options declared above
def stack_mats(mat_dict, groups=None):
    return fpah.stack_fp_mats(mat_dict, regions, sess_ids, subjects, signal_type, filter_outliers, outlier_thresh, groups)

cue_mats = stack_mats(cue)
resp_mats = stack_mats(resp)
cue_resp_mats = stack_mats(cue_resp)

all_mats = {Align.cue: cue_mats, Align.resp: resp_mats, Align.cue_resp: cue_resp_mats}

all_ts = {Align.cue: cue['t'], Align.resp: resp['t'], Align.cue_resp: cue_resp['t']}

all_xlims = {Align.cue: gen_xlims, Align.resp: resp_xlims, Align.cue_resp: None}

all_dashlines = {Align.cue: None, Align.resp: reward_time, Align.cue_resp: None}

left_left = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper left'}}
left_right = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}
right_left = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper left'}}
right_right = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper right'}}
all_legend_params = {Align.cue: left_left, Align.resp: right_right, Align.cue_resp: right_left}

def save_plot(fig, plot_name):
    if save_plots and not plot_name is None:
        fpah.save_fig(fig, fpah.get_figure_save_path(behavior_name, subjects, plot_name))

    if not show_plots:
        plt.close(fig)

def plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, xlims_dict=None,
                     regions=regions, dashlines=None, legend_params=None, group_colors=None, gen_plot_name=None):

    mat = all_mats[align]
    t = all_ts[align]
    align_title = fpah.get_align_title(align)
    x_label = fpah.get_align_xlabel(align)

    if xlims_dict is None:
        xlims_dict = all_xlims[align]

    if dashlines is None:
        dashlines = all_dashlines[align]

    if legend_params is None:
        legend_params = all_legend_params[align]

    fig = fpah.plot_avg_signals(plot_groups, group_labels, mat, regions, t, gen_title.format(align_title), plot_titles, x_label, signal_label, xlims_dict,
                                dashlines=dashlines, legend_params=legend_params, group_colors=group_colors, use_se=use_se, ph=ph, pw=pw)

    if not gen_plot_name is None:
        save_plot(fig, gen_plot_name.format(align))

# %% Choice, prior choice and side groupings

# choice & previous choice
plot_groups = [['side_choice', 'center_choice', 'no_resp'],
               ['side_choice_prev_side_choice', 'side_choice_prev_center_choice', 'side_choice_prev_no_resp'],
               ['center_choice_prev_side_choice', 'center_choice_prev_no_resp']]
group_labels = {'side_choice': 'Harvest', 'center_choice': 'Switch Patch', 'no_resp': 'No Response',
                'side_choice_prev_side_choice': 'Prev Harvest', 'side_choice_prev_center_choice': 'Prev Switch Patch',
                'side_choice_prev_no_resp': 'Prev No Response', 'center_choice_prev_side_choice': 'Prev Harvest',
                'center_choice_prev_no_resp': 'Prev No Response'}
plot_titles = ['Choices', 'Harvest Choices by Previous Choice', 'Switch Patch Choices by Previous Choice']
gen_title = 'Choice by Prior Choice Aligned to {}'
gen_plot_name = '{}_choice_prev_choice'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# side & choice
plot_groups = [['contra_side_choice', 'ipsi_side_choice'],
               ['contra_center_choice', 'ipsi_center_choice'],
               ['contra_no_resp', 'ipsi_no_resp']]
group_labels = {'contra_side_choice': 'Contra', 'ipsi_side_choice': 'Ipsi',
                'contra_center_choice': 'Contra', 'ipsi_center_choice': 'Ipsi',
                'contra_no_resp': 'Contra', 'ipsi_no_resp': 'Ipsi'}
plot_titles = ['Harvest Choice', 'Switch Patch Choice', 'No Response']
gen_title = 'Choice by Harvest Port Side Aligned to {}'
gen_plot_name = '{}_choice_side'

align = Align.cue

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


plot_groups = [['contra_side_choice', 'ipsi_side_choice'],
               ['contra_center_choice', 'ipsi_center_choice']]

plot_titles = ['Harvest Choices', 'Switch Patch Choices']

aligns = [Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# side, choice, & previous choice
plot_groups = [['contra_side_choice_prev_side_choice', 'contra_side_choice_prev_center_choice', 'contra_side_choice_prev_no_resp'],
               ['ipsi_side_choice_prev_side_choice', 'ipsi_side_choice_prev_center_choice', 'ipsi_side_choice_prev_no_resp'],
               ['contra_center_choice_prev_side_choice', 'contra_center_choice_prev_no_resp'],
               ['ipsi_center_choice_prev_side_choice', 'ipsi_center_choice_prev_no_resp']]
group_labels = {'contra_side_choice_prev_side_choice': 'Prev Harvest', 'contra_side_choice_prev_center_choice': 'Prev Switch Patch', 'contra_side_choice_prev_no_resp': 'Prev No Response',
                'ipsi_side_choice_prev_side_choice': 'Prev Harvest', 'ipsi_side_choice_prev_center_choice': 'Prev Switch Patch', 'ipsi_side_choice_prev_no_resp': 'Prev No Response',
                'contra_center_choice_prev_side_choice': 'Prev Harvest', 'contra_center_choice_prev_no_resp': 'Prev No Response',
                'ipsi_center_choice_prev_side_choice': 'Prev Harvest', 'ipsi_center_choice_prev_no_resp': 'Prev No Response'}

plot_titles = ['Contra Harvest Choices', 'Ipsi Harvest Choices', 'Contra Switch Patch Choices', 'Ipsi Switch Patch Choices']
gen_title = 'Choice by Harvest Port Side & Prior Choice Aligned to {}'
gen_plot_name = '{}_choice_side_prev_choice'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# future choice
plot_groups = [['side_choice_future_side_choice', 'side_choice_future_center_choice', 'side_choice_future_no_resp'],
               ['contra_side_choice_future_side_choice', 'contra_side_choice_future_center_choice', 'contra_side_choice_future_no_resp'],
               ['ipsi_side_choice_future_side_choice', 'ipsi_side_choice_future_center_choice', 'ipsi_side_choice_future_no_resp']]
group_labels = {'side_choice_future_side_choice': 'Future Harvest', 'side_choice_future_center_choice': 'Future Switch Patch', 'side_choice_future_no_resp': 'Future No Response',
                'contra_side_choice_future_side_choice': 'Future Harvest', 'contra_side_choice_future_center_choice': 'Future Switch Patch', 'contra_side_choice_future_no_resp': 'Future No Response',
                'ipsi_side_choice_future_side_choice': 'Future Harvest', 'ipsi_side_choice_future_center_choice': 'Future Switch Patch', 'ipsi_side_choice_future_no_resp': 'Future No Response'}

plot_titles = ['All Harvest Choices', 'Contra Harvest Choices', 'Ipsi Harvest Choices']
gen_title = 'Future Choice by Harvest Port Side Aligned to {}'
gen_plot_name = '{}_choice_side_future_choice'

align = Align.resp

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Reward volume, choice, & side

# reward volume and choice side
gen_plot_groups = ['rew_{}_side_choice', 'contra_rew_{}_side_choice', 'ipsi_rew_{}_side_choice']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['All Harvest Choices', 'Contra Harvest Choices', 'Ipsi Harvest Choices']
gen_title = 'Reward Volume by Harvest Choice Side Aligned to {}'
gen_plot_name = '{}_rew_side'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# previous reward volume and choice side
gen_plot_groups = ['prev_rew_{}_side_choice', 'contra_prev_rew_{}_side_choice', 'ipsi_prev_rew_{}_side_choice']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['All Harvest Choices', 'Contra Harvest Choices', 'Ipsi Harvest Choices']
gen_title = 'Previous Reward Volume by Harvest Choice Side Aligned to {}'
gen_plot_name = '{}_prev_rew_side'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# previous reward volume and side choice
gen_plot_groups = ['prev_rew_{}_side_choice', 'prev_rew_{}_side_choice_prev_side_choice', 'prev_rew_{}_side_choice_prev_no_resp']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['All Harvest Choices', 'Harvest Choices After Harvests', 'Harvest Choices After No Response']
gen_title = 'Previous Reward Volume for Harvest Choices by Prior Choice Aligned to {}'
gen_plot_name = '{}_prev_rew_harvest_prev_choice'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# previous reward volume and center choice
gen_plot_groups = ['prev_rew_{}_center_choice', 'prev_rew_{}_center_choice_prev_side_choice', 'prev_rew_{}_center_choice_prev_no_resp']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['All Switch Patch Choices', 'Switch Patch Choices After Harvests', 'Switch Patch Choices After No Response']
gen_title = 'Previous Reward Volume for Switch Patch Choices by Prior Choice Aligned to {}'
gen_plot_name = '{}_prev_rew_switch_patch_prev_choice'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward volume and previous choice
gen_plot_groups = ['rew_{}_prev_side_choice', 'rew_{}_prev_center_choice']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['Prev Harvest Choice', 'Prev Switch Patch Choice']
gen_title = 'Reward Volume by Prior Choice Aligned to {}'
gen_plot_name = '{}_rew_prev_choice'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward volume and future choice
align = Align.resp

gen_plot_groups = ['rew_{}_side_choice_future_side_choice', 'rew_{}_side_choice_future_center_choice', 'rew_{}_side_choice_future_no_resp']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['Future Harvest Choice', 'Future Switch Patch Choice', 'Future No Response']
gen_title = 'Future Choice after Harvest Choices by Reward Volume Aligned to {}'
gen_plot_name = '{}_rew_future_choice'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

gen_plot_groups = ['contra_rew_{}_side_choice_future_side_choice', 'contra_rew_{}_side_choice_future_center_choice', 'contra_rew_{}_side_choice_future_no_resp']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['Future Harvest Choice', 'Future Switch Patch Choice', 'Future No Response']
gen_title = 'Future Choice after Contra Harvest Choices by Reward Volume Aligned to {}'
gen_plot_name = '{}_rew_future_choice_contra'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

gen_plot_groups = ['ipsi_rew_{}_side_choice_future_side_choice', 'ipsi_rew_{}_side_choice_future_center_choice', 'ipsi_rew_{}_side_choice_future_no_resp']
plot_groups = [[group.format(rew_bin_strs[rb]) for rb in rew_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_bin_strs[rb]): rew_bin_strs[rb] for group in gen_plot_groups for rb in rew_bins}

plot_titles = ['Future Harvest Choice', 'Future Switch Patch Choice', 'Future No Response']
gen_title = 'Future Choice after Contra Harvest Choices by Reward Volume Aligned to {}'
gen_plot_name = '{}_rew_future_choice_ipsi'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Reward volume by initial volume, decay rate, & side

# previous reward and initial volume for center choices
group = 'prev_rew_{}_center_choice_init_{}'
plot_groups = [[group.format(rew_bin_strs[rb], ir) for rb in rew_bins] for ir in initial_rewards]
group_labels = {group.format(rew_bin_strs[rb], ir): rew_bin_strs[rb] for rb in rew_bins for ir in initial_rewards}

plot_titles = ['Initial Volume - {} μL'.format(ir) for ir in initial_rewards]
gen_title = 'Previous Reward Volume for Switch Patch Choices by Initial Volume Aligned to {}'
gen_plot_name = '{}_prev_rew_switch_patch_init_vol'

aligns = [Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# previous reward and initial volume for side choices
group = 'prev_rew_{}_side_choice_init_{}'
plot_groups = [[group.format(rew_bin_strs[rb], ir) for rb in rew_bins] for ir in initial_rewards]
group_labels = {group.format(rew_bin_strs[rb], ir): rew_bin_strs[rb] for rb in rew_bins for ir in initial_rewards}

plot_titles = ['Initial Volume - {} μL'.format(ir) for ir in initial_rewards]
gen_title = 'Previous Reward Volume for Harvest Choices by Initial Volume Aligned to {}'
gen_plot_name = '{}_prev_rew_harvest_init_vol'

aligns = [Align.cue, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# same for contra choices only
group = 'contra_prev_rew_{}_side_choice_init_{}'
plot_groups = [[group.format(rew_bin_strs[rb], ir) for rb in rew_bins] for ir in initial_rewards]
group_labels = {group.format(rew_bin_strs[rb], ir): rew_bin_strs[rb] for rb in rew_bins for ir in initial_rewards}

plot_titles = ['Initial Volume - {} μL'.format(ir) for ir in initial_rewards]
gen_title = 'Previous Reward Volume for Contra Harvest Choices by Initial Volume Aligned to {}'
gen_plot_name = '{}_prev_rew_harvest_init_vol_contra'

aligns = [Align.cue, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# same for ipsi choices only
group = 'ipsi_prev_rew_{}_side_choice_init_{}'
plot_groups = [[group.format(rew_bin_strs[rb], ir) for rb in rew_bins] for ir in initial_rewards]
group_labels = {group.format(rew_bin_strs[rb], ir): rew_bin_strs[rb] for rb in rew_bins for ir in initial_rewards}

plot_titles = ['Initial Volume - {} μL'.format(ir) for ir in initial_rewards]
gen_title = 'Previous Reward Volume for Ipsi Harvest Choices by Initial Volume Aligned to {}'
gen_plot_name = '{}_prev_rew_harvest_init_vol_ipsi'

aligns = [Align.cue, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# current reward and initial volume for side choices
align = Align.resp

group = 'rew_{}_side_choice_init_{}'
plot_groups = [[group.format(rew_bin_strs[rb], ir) for ir in initial_rewards] for rb in rew_bins[1:]]
group_labels = {group.format(rew_bin_strs[rb], ir): '{} μL'.format(ir) for ir in initial_rewards for rb in rew_bins[1:]}

plot_titles = [rew_bin_strs[rb] for rb in rew_bins[1:]]
gen_title = 'Reward Volume for Harvest Choices by Initial Volume Aligned to {}'
gen_plot_name = '{}_rew_harvest_init_vol'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# current reward and decay rate for side choices
group = 'rew_{}_side_choice_decay_{}'
plot_groups = [[group.format(rew_bin_strs[rb], dr) for dr in decay_rates] for rb in rew_bins[1:]]
group_labels = {group.format(rew_bin_strs[rb], dr): str(dr) for dr in decay_rates for rb in rew_bins[1:]}

plot_titles = [rew_bin_strs[rb] for rb in rew_bins[1:]]
gen_title = 'Reward Volume for Harvest Choices by Decay Rate Aligned to {}'
gen_plot_name = '{}_rew_harvest_decay_rate'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)
