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
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
import numpy as np
import matplotlib.pyplot as plt
import copy

# %% Load behavior data

# used for saving plots
behavior_name = 'Intertemporal Choice'

# get all session ids for given protocol
sess_ids = db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 3)
# optionally limit sessions based on subject ids
subj_ids = [179]
sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}
sess_ids = bah.limit_sess_ids(sess_ids, last_idx=-1)

loc_db = db.LocalDB_BasicRLTasks('temporalChoice')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids)) # , reload=True

# make slow delay a string for better plot formatting
sess_data['slow_delay'] = sess_data['slow_delay'].apply(lambda x: '{:.0f}'.format(x))
# set any missing cport outs to nan
sess_data['cpoke_out_time'] = sess_data['cpoke_out_time'].apply(lambda x: x if utils.is_scalar(x) else np.nan)
sess_data['cpoke_out_latency'] = sess_data['cpoke_out_time'] - sess_data['response_cue_time']

# %% Get and process photometry data

# get fiber photometry data
reload = False
fp_data, implant_info = fpah.load_fp_data(loc_db, sess_ids, reload=reload)

# %% Observe the full signals

filter_outliers = True
save_plots = False
show_plots = False

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        # Get the block transition trial start times
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        block_start_times = trial_start_ts[trial_data['block_trial'] == 1]

        fig = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
                                    title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id),
                                    vert_marks=block_start_times, filter_outliers=filter_outliers)

        if save_plots:
            fpah.save_fig(fig, fpah.get_figure_save_path(behavior_name, subj_id, 'sess_{}'.format(sess_id)))

        if not show_plots:
            plt.close(fig)


# %% Observe any sub-signals
# tmp_sess_id = {182: [101717]}
# tmp_fp_data, tmp_implant_info = fpah.load_fp_data(loc_db, tmp_sess_id)
# sub_signal = [0, np.inf]
# filter_outliers = True

# subj_id = list(tmp_sess_id.keys())[0]
# sess_id = tmp_sess_id[subj_id][0]
# #sess_fp = fp_data[subj_id][sess_id]
# sess_fp = tmp_fp_data[subj_id][sess_id]
# _ = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
#                             title='Sub Signal - Subject {}, Session {}'.format(subj_id, sess_id),
#                             filter_outliers=filter_outliers,
#                             t_min=sub_signal[0], t_max=sub_signal[1], dec=1)


# %% Get all aligned/sorted stacked signals

signal_types = ['z_dff_iso'] # 'baseline_corr_lig','baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}
cport_on = copy.deepcopy(data_dict)
cpoke_in = copy.deepcopy(data_dict)
early_cpoke_in = copy.deepcopy(data_dict)
cpoke_out = copy.deepcopy(data_dict)
early_cpoke_out = copy.deepcopy(data_dict)
cue = copy.deepcopy(data_dict)
resp = copy.deepcopy(data_dict)
reward = copy.deepcopy(data_dict)
cue_poke_out_resp = copy.deepcopy(data_dict)
poke_out_cue_resp = copy.deepcopy(data_dict)
resp_reward = copy.deepcopy(data_dict)

block_rates = np.unique(sess_data['block_rates'])
block_rewards = np.unique(sess_data['block_rewards'])
slow_delays = np.unique(sess_data['slow_delay'])
rates = np.unique(sess_data['choice_rate'])
rates = rates[~np.isnan(rates)]
rewards = np.unique(sess_data['reward'])
rewards = rewards[rewards != 0]
delays = np.unique(sess_data['choice_delay'])
delays = delays[~np.isnan(delays)]
sides = ['left', 'right']

# declare settings for normalized cue to response intervals
norm_cue_resp_bins = 200
norm_cue_poke_out_pct = 0.2 # % of bins for cue to poke out or poke out to cue, depending on which comes first

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        trial_data = sess_data[sess_data['sessid'] == sess_id]
        resp_sel = (trial_data['hit'] == True)
        # remove no responses
        trial_data = trial_data[resp_sel]
        sess_fp = fp_data[subj_id][sess_id]

        # get alignment trial filters
        choices = trial_data['choice'].to_numpy()
        instruct_trial = trial_data['instruct_trial'].to_numpy()
        stays = (choices[:-1] == choices[1:]) & ~instruct_trial[:-1]
        switches = (choices[:-1] != choices[1:]) & ~instruct_trial[:-1]
        future_switches = np.append(switches, False)
        future_stays = np.append(stays, False)
        switches = np.insert(switches, 0, False)
        stays = np.insert(stays, 0, False)

        choice_delay = trial_data['choice_delay'].to_numpy()
        trial_block_rate = trial_data['block_rates'].to_numpy()
        choice_reward = trial_data['reward'].to_numpy()
        fast_choice = trial_data['chose_fast_port'].to_numpy()
        slow_choice = trial_data['chose_slow_port'].to_numpy()
        right_choice = trial_data['chose_right'].to_numpy()
        left_choice = trial_data['chose_left'].to_numpy()

        prev_right_choice = np.insert(right_choice[:-1], 0, False)
        prev_left_choice = np.insert(left_choice[:-1], 0, False)
        prev_reward = np.insert(choice_reward[:-1], 0, 0)

        # ignore cport on trials where it came on when they were already poked in
        cport_on_sel = trial_data['cpoke_in_latency'] > 0.1
        early_cpoke_in_sel = trial_data['cpoke_in_latency'] < 0
        norm_cpoke_in_sel = trial_data['cpoke_in_latency'] > 0
        early_cpoke_out_sel = trial_data['cpoke_out_latency'] < 0
        norm_cpoke_out_sel = trial_data['cpoke_out_latency'] > 0

        # get alignment times
        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1][resp_sel]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        resp_ts = trial_start_ts + trial_data['response_time']
        reward_ts = trial_start_ts + trial_data['reward_time']

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # center port on
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cport_on_ts, pre, post)
                align_dict = cport_on
                sel = cport_on_sel

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice & sel,:]
                align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice & sel,:]
                align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice & stays & sel,:]
                align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice & switches & sel,:]
                align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice & stays & sel,:]
                align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice & switches & sel,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = (left_choice if side == 'left' else right_choice) & sel
                    prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice & switches,:]

                    align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]

                for rew in rewards:
                    rew_sel = (prev_reward == rew) & sel
                    align_dict[sess_id][signal_type][region]['prev_rew_'+str(rew)] = mat[rew_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rew_'+str(rew)] = mat[prev_side_sel & rew_sel,:]

                # center poke in
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_in_ts, pre, post)
                align_dicts = [early_cpoke_in, cpoke_in]
                sels = [early_cpoke_in_sel, norm_cpoke_in_sel]

                for align_dict, sel in zip(align_dicts, sels):

                    align_dict['t'] = t
                    align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice & switches & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice & switches & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice & stays,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice & switches,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice & stays,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice & switches,:]

                        align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]

                    for rew in rewards:
                        rew_sel = (choice_reward == rew) & sel
                        prev_rew_sel = (prev_reward == rew) & sel
                        align_dict[sess_id][signal_type][region]['prev_rew_'+str(rew)] = mat[prev_rew_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(rew)] = mat[rew_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice
                            align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rew_'+str(rew)] = mat[prev_side_sel & prev_rew_sel,:]


                # aligned to response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cue_ts, pre, post)
                align_dict = cue
                # only look at response cues before cpoke outs
                sel = norm_cpoke_out_sel

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice & sel,:]
                align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice & sel,:]
                align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice & stays & sel,:]
                align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice & switches & sel,:]
                align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice & stays & sel,:]
                align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice & switches & sel,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = (left_choice if side == 'left' else right_choice) & sel
                    prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice & switches,:]

                    align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]

                for rew in rewards:
                    rew_sel = (choice_reward == rew) & sel
                    prev_rew_sel = (prev_reward == rew) & sel
                    align_dict[sess_id][signal_type][region]['prev_rew_'+str(rew)] = mat[prev_rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(rew)] = mat[rew_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rew_'+str(rew)] = mat[prev_side_sel & prev_rew_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)] = mat[side_sel & rew_sel,:]

                    for delay in delays:
                        delay_sel = (choice_delay == delay) & sel
                        align_dict[sess_id][signal_type][region]['rew_'+str(rew)+'_delay_'+str(delay)] = mat[delay_sel & rew_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rew_sel,:]

                for rate in block_rates:
                    rate_sel = (trial_block_rate == rate) & sel
                    align_dict[sess_id][signal_type][region]['rate_'+rate] = mat[rate_sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_rate_'+rate] = mat[rate_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_rate_'+rate] = mat[rate_sel & slow_choice,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate] = mat[side_sel & rate_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_rate_'+rate] = mat[side_sel & rate_sel & fast_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_rate_'+rate] = mat[side_sel & rate_sel & slow_choice,:]

                    for delay in delays:
                        delay_sel = choice_delay == delay
                        align_dict[sess_id][signal_type][region]['rate_'+rate+'_delay_'+str(delay)] = mat[delay_sel & rate_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rate_sel,:]

                for delay in delays:
                    delay_sel = (choice_delay == delay) & sel
                    align_dict[sess_id][signal_type][region]['delay_'+str(delay)] = mat[delay_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+str(delay)] = mat[side_sel & delay_sel,:]


                # aligned to center poke out
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_out_ts, pre, post)
                align_dicts = [early_cpoke_out, cpoke_out]
                sels = [early_cpoke_out_sel, norm_cpoke_out_sel]

                for align_dict, sel in zip(align_dicts, sels):

                    align_dict['t'] = t
                    align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice & switches & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice & switches & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice & stays,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice & switches,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice & stays,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice & switches,:]

                    for rew in rewards:
                        rew_sel = (choice_reward == rew) & sel
                        align_dict[sess_id][signal_type][region]['rew_'+str(rew)] = mat[rew_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)] = mat[side_sel & rew_sel,:]

                        for delay in delays:
                            delay_sel = choice_delay == delay
                            align_dict[sess_id][signal_type][region]['rew_'+str(rew)+'_delay_'+str(delay)] = mat[delay_sel & rew_sel,:]

                            for side in sides:
                                side_type = fpah.get_implant_side_type(side, region_side)
                                side_sel = left_choice if side == 'left' else right_choice
                                align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rew_sel,:]

                    for rate in block_rates:
                        rate_sel = (trial_block_rate == rate) & sel
                        align_dict[sess_id][signal_type][region]['rate_'+rate] = mat[rate_sel,:]
                        align_dict[sess_id][signal_type][region]['chose_fast_rate_'+rate] = mat[rate_sel & fast_choice,:]
                        align_dict[sess_id][signal_type][region]['chose_slow_rate_'+rate] = mat[rate_sel & slow_choice,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate] = mat[side_sel & rate_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_chose_fast_rate_'+rate] = mat[side_sel & rate_sel & fast_choice,:]
                            align_dict[sess_id][signal_type][region][side_type+'_chose_slow_rate_'+rate] = mat[side_sel & rate_sel & slow_choice,:]

                        for delay in delays:
                            delay_sel = choice_delay == delay
                            align_dict[sess_id][signal_type][region]['rate_'+rate+'_delay_'+str(delay)] = mat[delay_sel & rate_sel,:]

                            for side in sides:
                                side_type = fpah.get_implant_side_type(side, region_side)
                                side_sel = left_choice if side == 'left' else right_choice
                                align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rate_sel,:]

                    for delay in delays:
                        delay_sel = (choice_delay == delay) & sel
                        align_dict[sess_id][signal_type][region]['delay_'+str(delay)] = mat[delay_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_delay_'+str(delay)] = mat[side_sel & delay_sel,:]


                # aligned to response poke
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, resp_ts, pre, post)
                align_dict = resp

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches,:]
                align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice,:]
                align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice,:]
                align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice & stays,:]
                align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice & switches,:]
                align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice & stays,:]
                align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice & switches,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = left_choice if side == 'left' else right_choice

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice & switches,:]

                for rew in rewards:
                    rew_sel = choice_reward == rew
                    prev_rew_sel = prev_reward == rew
                    align_dict[sess_id][signal_type][region]['rew_'+str(rew)] = mat[rew_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)] = mat[side_sel & rew_sel,:]

                    for delay in delays:
                        delay_sel = choice_delay == delay
                        align_dict[sess_id][signal_type][region]['rew_'+str(rew)+'_delay_'+str(delay)] = mat[delay_sel & rew_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rew_sel,:]

                for rate in block_rates:
                    rate_sel = trial_block_rate == rate
                    align_dict[sess_id][signal_type][region]['rate_'+rate] = mat[rate_sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_rate_'+rate] = mat[rate_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_rate_'+rate] = mat[rate_sel & slow_choice,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate] = mat[side_sel & rate_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_rate_'+rate] = mat[side_sel & rate_sel & fast_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_rate_'+rate] = mat[side_sel & rate_sel & slow_choice,:]

                    for delay in delays:
                        delay_sel = choice_delay == delay
                        align_dict[sess_id][signal_type][region]['rate_'+rate+'_delay_'+str(delay)] = mat[delay_sel & rate_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rate_sel,:]

                for delay in delays:
                    delay_sel = choice_delay == delay
                    align_dict[sess_id][signal_type][region]['delay_'+str(delay)] = mat[delay_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+str(delay)] = mat[side_sel & delay_sel,:]


                # aligned to reward delivery
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, reward_ts, pre, post)
                align_dict = reward

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches,:]
                align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice,:]
                align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice,:]
                align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice & stays,:]
                align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice & switches,:]
                align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice & stays,:]
                align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice & switches,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = left_choice if side == 'left' else right_choice

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice & switches,:]

                for rew in rewards:
                    rew_sel = choice_reward == rew
                    prev_rew_sel = prev_reward == rew
                    align_dict[sess_id][signal_type][region]['rew_'+str(rew)] = mat[rew_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)] = mat[side_sel & rew_sel,:]

                    for delay in delays:
                        delay_sel = choice_delay == delay
                        align_dict[sess_id][signal_type][region]['rew_'+str(rew)+'_delay_'+str(delay)] = mat[delay_sel & rew_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rew_sel,:]

                for rate in block_rates:
                    rate_sel = trial_block_rate == rate
                    align_dict[sess_id][signal_type][region]['rate_'+rate] = mat[rate_sel,:]
                    align_dict[sess_id][signal_type][region]['chose_fast_rate_'+rate] = mat[rate_sel & fast_choice,:]
                    align_dict[sess_id][signal_type][region]['chose_slow_rate_'+rate] = mat[rate_sel & slow_choice,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate] = mat[side_sel & rate_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_rate_'+rate] = mat[side_sel & rate_sel & fast_choice,:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_rate_'+rate] = mat[side_sel & rate_sel & slow_choice,:]

                    for delay in delays:
                        delay_sel = choice_delay == delay
                        align_dict[sess_id][signal_type][region]['rate_'+rate+'_delay_'+str(delay)] = mat[delay_sel & rate_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region][side_type+'_rate_'+rate+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rate_sel,:]

                for delay in delays:
                    delay_sel = choice_delay == delay
                    align_dict[sess_id][signal_type][region]['delay_'+str(delay)] = mat[delay_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+str(delay)] = mat[side_sel & delay_sel,:]

                # time normalized signal matrices
                cue_poke_out = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, cpoke_out_ts, norm_cue_resp_bins*norm_cue_poke_out_pct,
                                                                      align_sel = norm_cpoke_out_sel)
                poke_out_cue = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, cue_ts, norm_cue_resp_bins*norm_cue_poke_out_pct,
                                                                      align_sel = early_cpoke_out_sel)
                poke_out_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, resp_ts, norm_cue_resp_bins*(1-norm_cue_poke_out_pct),
                                                                       align_sel = norm_cpoke_out_sel)
                cue_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, resp_ts, norm_cue_resp_bins*(1-norm_cue_poke_out_pct),
                                                                  align_sel = early_cpoke_out_sel)

                resp_rew = fp_utils.build_time_norm_signal_matrix(signal, ts, resp_ts, reward_ts, norm_cue_resp_bins)

                mats = [np.hstack((cue_poke_out, poke_out_resp)), np.hstack((poke_out_cue, cue_resp)), resp_rew]
                align_dicts = [cue_poke_out_resp, poke_out_cue_resp, resp_reward]
                sels = [norm_cpoke_out_sel, early_cpoke_out_sel, np.full(resp_ts.shape, True)]

                for mat, align_dict, sel in zip(mats, align_dicts, sels):
                    align_dict['t'] = np.linspace(0, 1, norm_cue_resp_bins)
                    align_dict[sess_id][signal_type][region]['stay'] = mat[stays[sel],:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[switches[sel],:]
                    align_dict[sess_id][signal_type][region]['chose_fast'] = mat[fast_choice[sel],:]
                    align_dict[sess_id][signal_type][region]['chose_slow'] = mat[slow_choice[sel],:]
                    align_dict[sess_id][signal_type][region]['chose_fast_stay'] = mat[fast_choice[sel] & stays[sel],:]
                    align_dict[sess_id][signal_type][region]['chose_fast_switch'] = mat[fast_choice[sel] & switches[sel],:]
                    align_dict[sess_id][signal_type][region]['chose_slow_stay'] = mat[slow_choice[sel] & stays[sel],:]
                    align_dict[sess_id][signal_type][region]['chose_slow_switch'] = mat[slow_choice[sel] & switches[sel],:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice[sel] if side == 'left' else right_choice[sel]

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast'] = mat[side_sel & fast_choice[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow'] = mat[side_sel & slow_choice[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_stay'] = mat[side_sel & fast_choice[sel] & stays[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_fast_switch'] = mat[side_sel & fast_choice[sel] & switches[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_stay'] = mat[side_sel & slow_choice[sel] & stays[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_chose_slow_switch'] = mat[side_sel & slow_choice[sel] & switches[sel],:]


                # do more with the normal cue, poke out, response sequence, and response to reward
                mats = [np.hstack((cue_poke_out, poke_out_resp)), resp_rew]
                align_dicts = [cue_poke_out_resp, resp_reward]
                sels = [norm_cpoke_out_sel, np.full(resp_ts.shape, True)]

                for mat, align_dict, sel in zip(mats, align_dicts, sels):
                    for rew in rewards:
                        rew_sel = choice_reward[sel] == rew
                        prev_rew_sel = prev_reward[sel] == rew
                        align_dict[sess_id][signal_type][region]['rew_'+str(rew)] = mat[rew_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice[sel] if side == 'left' else right_choice[sel]
                            align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)] = mat[side_sel & rew_sel,:]

                        for delay in delays:
                            delay_sel = choice_delay[sel] == delay
                            align_dict[sess_id][signal_type][region]['rew_'+str(rew)+'_delay_'+str(delay)] = mat[delay_sel & rew_sel,:]

                            for side in sides:
                                side_type = fpah.get_implant_side_type(side, region_side)
                                side_sel = left_choice[sel] if side == 'left' else right_choice[sel]
                                align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(rew)+'_delay_'+str(delay)] = mat[side_sel & delay_sel & rew_sel,:]

                    for delay in delays:
                        delay_sel = choice_delay[sel] == delay
                        align_dict[sess_id][signal_type][region]['delay_'+str(delay)] = mat[delay_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice[sel] if side == 'left' else right_choice[sel]
                            align_dict[sess_id][signal_type][region][side_type+'_delay_'+str(delay)] = mat[side_sel & delay_sel,:]


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
rew_xlims = {'DMS': [-1.5,2], 'PL': [-3,10]}
gen_xlims = {'DMS': [-1.5,1.5], 'PL': [-3,3]}

save_plots = True
show_plots = False

# make this wrapper to simplify the stack command by not having to include the options declared above
def stack_mats(mat_dict, groups=None):
    return fpah.stack_fp_mats(mat_dict, regions, sess_ids, subjects, signal_type, filter_outliers, outlier_thresh, groups)

cport_on_mats = stack_mats(cport_on)
cpoke_in_mats = stack_mats(cpoke_in)
early_cpoke_in_mats = stack_mats(early_cpoke_in)
cpoke_out_mats = stack_mats(cpoke_out)
early_cpoke_out_mats = stack_mats(early_cpoke_out)
cue_mats = stack_mats(cue)
resp_mats = stack_mats(resp)
rew_mats = stack_mats(reward)
cue_poke_resp_mats = stack_mats(cue_poke_out_resp)
poke_cue_resp_mats = stack_mats(poke_out_cue_resp)
resp_rew_mats = stack_mats(resp_reward)

all_mats = {Align.cport_on: cport_on_mats, Align.cpoke_in: cpoke_in_mats, Align.early_cpoke_in: early_cpoke_in_mats,
            Align.cue: cue_mats, Align.cpoke_out: cpoke_out_mats, Align.early_cpoke_out: early_cpoke_out_mats,
            Align.resp: resp_mats, Align.reward: rew_mats, Align.cue_poke_resp: cue_poke_resp_mats,
            Align.poke_cue_resp: poke_cue_resp_mats, Align.resp_reward: resp_rew_mats}

all_ts = {Align.cport_on: cport_on['t'], Align.cpoke_in: cpoke_in['t'], Align.early_cpoke_in: early_cpoke_in['t'],
          Align.cue: cue['t'], Align.cpoke_out: cpoke_out['t'], Align.early_cpoke_out: early_cpoke_out['t'],
          Align.resp: resp['t'], Align.reward: reward['t'], Align.cue_poke_resp: cue_poke_out_resp['t'],
          Align.poke_cue_resp: poke_out_cue_resp['t'], Align.resp_reward: resp_reward['t']}

all_xlims = {Align.cport_on: gen_xlims, Align.cpoke_in: gen_xlims, Align.early_cpoke_in: gen_xlims,
            Align.cue: gen_xlims, Align.cpoke_out: gen_xlims, Align.early_cpoke_out: gen_xlims,
            Align.resp: gen_xlims, Align.reward: rew_xlims, Align.cue_poke_resp: None,
            Align.poke_cue_resp: None, Align.resp_reward: None}

all_dashlines = {Align.cport_on: None, Align.cpoke_in: None, Align.early_cpoke_in: None, Align.cue: None,
                Align.cpoke_out: None, Align.early_cpoke_out: None, Align.resp: None, Align.reward: None,
                Align.cue_poke_resp: norm_cue_poke_out_pct, Align.poke_cue_resp: norm_cue_poke_out_pct, Align.resp_reward: None}

left_left = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper left'}}
left_right = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}
right_left = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper left'}}
right_right = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper right'}}
all_legend_params = {Align.cport_on: {'DMS': {'loc': 'upper left'}, 'PL': None}, Align.cpoke_in: left_right,
                     Align.early_cpoke_in: right_right, Align.cue: left_left, Align.cpoke_out: left_left, Align.early_cpoke_out: left_left,
                     Align.resp: right_left, Align.reward: right_right, Align.cue_poke_resp: right_left, Align.poke_cue_resp: right_left,
                     Align.resp_reward: right_left}

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

    fig, plotted = fpah.plot_avg_signals(plot_groups, group_labels, mat, regions, t, gen_title.format(align_title), plot_titles, x_label, signal_label, xlims_dict,
                                dashlines=dashlines, legend_params=legend_params, group_colors=group_colors, use_se=use_se, ph=ph, pw=pw)

    if plotted and not gen_plot_name is None:
        save_plot(fig, gen_plot_name.format(align))

    if not plotted:
        plt.close(fig)

# %% Choice, side, and prior reward groupings for multiple alignment points

# choice, side, & side/choice

plot_groups = [['stay', 'switch'], ['contra', 'ipsi'], ['contra_stay', 'contra_switch', 'ipsi_stay', 'ipsi_switch']]
group_labels = {'stay': 'Stay', 'switch': 'Switch',
                'ipsi': 'Ipsi', 'contra': 'Contra',
                'contra_stay': 'Contra Stay', 'contra_switch': 'Contra Switch',
                'ipsi_stay': 'Ipsi Stay', 'ipsi_switch': 'Ipsi Switch'}
plot_titles = ['Choice', 'Side', 'Choice/Side']
gen_title = 'Choice/Side Groupings Aligned to {}'
gen_plot_name = '{}_stay_switch_side'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out,
          Align.resp, Align.reward, Align.cue_poke_resp, Align.poke_cue_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# chose fast/slow by side or stay/switch
plot_groups = [['chose_fast', 'chose_slow'],
               ['chose_fast_stay', 'chose_fast_switch', 'chose_slow_stay', 'chose_slow_switch'],
               ['contra_chose_fast', 'ipsi_chose_fast', 'contra_chose_slow', 'ipsi_chose_slow']]
group_labels = {'chose_fast': 'Fast', 'chose_slow': 'Slow',
                'chose_fast_stay': 'Fast Stay', 'chose_fast_switch': 'Fast Switch',
                'chose_slow_stay': 'Slow Stay', 'chose_slow_switch': 'Slow Switch',
                'contra_chose_fast': 'Fast Contra', 'ipsi_chose_fast': 'Fast Ipsi',
                'contra_chose_slow': 'Slow Contra', 'ipsi_chose_slow': 'Slow Ipsi'}

plot_titles = ['Trial Length', 'Trial Length & Stay/Switch', 'Trial Length/Side']
gen_title = 'Choice Trial Length by Stay/switch or Side Groupings Aligned to {}'
gen_plot_name = '{}_trial_length_stay_switch_or_side'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out,
          Align.resp, Align.reward, Align.cue_poke_resp, Align.poke_cue_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# chose fast/slow by side and stay/switch
plot_groups = [['contra_chose_fast_stay', 'contra_chose_fast_switch', 'contra_chose_slow_stay', 'contra_chose_slow_switch'],
               ['ipsi_chose_fast_stay', 'ipsi_chose_fast_switch', 'ipsi_chose_slow_stay', 'ipsi_chose_slow_switch']]

group_labels = {'contra_chose_fast_stay': 'Fast Stay', 'contra_chose_fast_switch': 'Fast Switch',
                'contra_chose_slow_stay': 'Slow Stay', 'contra_chose_slow_switch': 'Slow Switch',
                'ipsi_chose_fast_stay': 'Fast Stay', 'ipsi_chose_fast_switch': 'Fast Switch',
                'ipsi_chose_slow_stay': 'Slow Stay', 'ipsi_chose_slow_switch': 'Slow Switch',}

plot_titles = ['Contra Choices', 'Ipsi Choices']
gen_title = 'Choice Trial Length by Stay/switch and Side Aligned to {}'
gen_plot_name = '{}_trial_length_stay_switch_and_side'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out,
          Align.resp, Align.reward, Align.cue_poke_resp, Align.poke_cue_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Groupings by previous side choice and previous reward

gen_plot_groups = ['prev_rew_{}', 'prev_contra_prev_rew_{}', 'prev_ipsi_prev_rew_{}']
plot_groups = [[group.format(r) for r in rewards] for group in gen_plot_groups]
plot_groups.insert(0, ['prev_contra', 'prev_ipsi'])
group_labels = {group.format(r): r for group in gen_plot_groups for r in rewards}
group_labels.update({'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi'})

plot_titles = ['Previous Side', 'Previous Reward Volume', 'Prev Contra by Prev Reward Volume', 'Prev Ipsi by Prev Reward Volume']
gen_title = 'Previous Side and Reward Volume Aligned to {}'
gen_plot_name = '{}_prev_side_prev_reward'

aligns = [Align.cport_on, Align.cpoke_in, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Choice reward amounts or rates

# reward volumes and choice side
gen_plot_groups = ['rew_{}', 'contra_rew_{}', 'ipsi_rew_{}']
plot_groups = [[group.format(r) for r in rewards] for group in gen_plot_groups]
group_labels = {group.format(r): r for group in gen_plot_groups for r in rewards}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward Volumes by Choice Side Aligned to {}'
gen_plot_name = '{}_reward_vol_side'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_poke_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward rates and choice trial length
gen_plot_groups = ['rate_{}', 'chose_fast_rate_{}', 'chose_slow_rate_{}']
plot_groups = [[group.format(r) for r in block_rates] for group in gen_plot_groups]
group_labels = {group.format(r): r for group in gen_plot_groups for r in block_rates}

plot_titles = ['All Choices', 'Fast Choices', 'Slow Choices']
gen_title = 'Reward Rates (Fast/Slow) by Trial Length Aligned to {}'
gen_plot_name = '{}_reward_rate_trial_length'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward rates and sides
gen_plot_groups = ['contra_rate_{}', 'ipsi_rate_{}']
plot_groups = [[group.format(r) for r in block_rates] for group in gen_plot_groups]
group_labels = {group.format(r): r for group in gen_plot_groups for r in block_rates}

plot_titles = ['Contra Choices', 'Ipsi Choices']
gen_title = 'Reward Rates (Fast/Slow) by Choice Side Aligned to {}'
gen_plot_name = '{}_reward_rate_side'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward rates, side, and trial length
gen_plot_groups = ['contra_chose_fast_rate_{}', 'contra_chose_slow_rate_{}', 'ipsi_chose_fast_rate_{}', 'ipsi_chose_slow_rate_{}']
plot_groups = [[group.format(r) for r in block_rates] for group in gen_plot_groups]
group_labels = {group.format(r): r for group in gen_plot_groups for r in block_rates}

plot_titles = ['Contra Fast Choices', 'Contra Slow Choices', 'Ipsi Fast Choices', 'Ipsi Slow Choices']
gen_title = 'Reward Rates (Fast/Slow) by Trial Length and Choice Side Aligned to {}'
gen_plot_name = '{}_reward_rate_trial_length_side'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Investigate signal modulation by delay

# delay length and choice side
gen_plot_groups = ['delay_{}', 'contra_delay_{}', 'ipsi_delay_{}']
plot_groups = [[group.format(d) for d in delays] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in delays}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward Delays by Choice Side Aligned to {}'
gen_plot_name = '{}_reward_delay_side'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_poke_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward volumes and delays
plot_groups = [['rew_{}_delay_{}'.format(r,d) for d in delays] for r in rewards]
group_labels = {'rew_{}_delay_{}'.format(r,d): d for d in delays for r in rewards}

plot_titles = ['{} μl'.format(r) for r in rewards]
gen_title = 'Reward Volume by Reward Delay Aligned to {}'
gen_plot_name = '{}_reward_volume_delay'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_poke_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward volumes, delays, and sides
plot_groups = [['contra_rew_{}_delay_{}'.format(r,d) for d in delays] for r in rewards]
group_labels = {'contra_rew_{}_delay_{}'.format(r,d): d for d in delays for r in rewards}

plot_titles = ['{} μl'.format(r) for r in rewards]
gen_title = 'Reward Volume by Reward Delay for Contra Choices Aligned to {}'
gen_plot_name = '{}_reward_volume_delay_contra'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_poke_resp, Align.resp_reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

plot_groups = [['ipsi_rew_{}_delay_{}'.format(r,d) for d in delays] for r in rewards]
group_labels = {'ipsi_rew_{}_delay_{}'.format(r,d): d for d in delays for r in rewards}

gen_title = 'Reward Volume by Reward Delay for Ipsi Choices Aligned to {}'
gen_plot_name = '{}_reward_volume_delay_ipsi'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# block rates and delays
plot_groups = [['rate_{}_delay_{}'.format(r,d) for d in delays] for r in block_rates]
group_labels = {'rate_{}_delay_{}'.format(r,d): d for d in delays for r in block_rates}

plot_titles = ['{} μl/s (Fast/Slow)'.format(r) for r in block_rates]
gen_title = 'Reward Rates (Fast/Slow) by Reward Delay Aligned to {}'
gen_plot_name = '{}_reward_rate_delay'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# rates, delays, and sides
plot_groups = [['contra_rate_{}_delay_{}'.format(r,d) for d in delays] for r in block_rates]
group_labels = {'contra_rate_{}_delay_{}'.format(r,d): d for d in delays for r in block_rates}

plot_titles = ['{} μl/s (Fast/Slow)'.format(r) for r in block_rates]
gen_title = 'Reward Rates (Fast/Slow) by Reward Delay for Contra Choices Aligned to {}'
gen_plot_name = '{}_reward_rate_delay_contra'

aligns = [Align.cue, Align.cpoke_out, Align.resp, Align.reward]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

plot_groups = [['ipsi_rate_{}_delay_{}'.format(r,d) for d in delays] for r in block_rates]
group_labels = {'ipsi_rate_{}_delay_{}'.format(r,d): d for d in delays for r in block_rates}

gen_title = 'Reward Rates (Fast/Slow) by Reward Delay for Ipsi Choices Aligned to {}'
gen_plot_name = '{}_reward_rate_delay_ipsi'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)
