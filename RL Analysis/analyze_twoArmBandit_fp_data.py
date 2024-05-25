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
import numpy as np
import matplotlib.pyplot as plt
import copy

# %% Load fiber photometry and behavior data

# get all session ids for given protocol
sess_ids = db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 2)

loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids)) # reload=True

# set any missing cpoke outs to nans
sess_data['cpoke_out_time'] = sess_data['cpoke_out_time'].apply(lambda x: x if utils.is_scalar(x) else np.nan)
# add in missing center port on time
sess_data['cport_on_time'] = sess_data['parsed_events'].apply(lambda x: x['States']['WaitForCenterPoke'][0])

# get fiber photometry data
fp_data = loc_db.get_sess_fp_data(utils.flatten(sess_ids), reload=True) # , reload=True
# separate into different dictionaries
implant_info = fp_data['implant_info']
fp_data = fp_data['fp_data']

# %% Process photometry data in different ways

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

for subj_id in sess_ids.keysS():
    for sess_id in sess_ids[subj_id]:
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        # Get the block transition trial start times
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        block_start_times = trial_start_ts[trial_data['block_trial'] == 1]
        #block_rewards = trial_data['reward_volume'][trial_data['block_trial'] == 1]

        if len(sub_signal) > 0:
            fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
                                        title='Full Signals - Session {}'.format(sess_id),
                                        vert_marks=block_start_times, filter_outliers=filter_outliers,
                                        t_min=sub_signal[0], t_max=sub_signal[1], dec=1)
        else:
            fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
                                        title='Full Signals - Session {}'.format(sess_id), #. Block Rewards: {}'.format(sess_id, ', '.join([str(r) for r in block_rewards]
                                        vert_marks=block_start_times, filter_outliers=filter_outliers)

# %% Get all aligned/sorted stacked signals

signal_types = ['z_dff_iso'] # 'baseline_corr_lig','baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}
cport_on = copy.deepcopy(data_dict)
cpoke_in = copy.deepcopy(data_dict)
cpoke_out = copy.deepcopy(data_dict)
cue = copy.deepcopy(data_dict)
resp = copy.deepcopy(data_dict)
resp_pchoice_rewarded = copy.deepcopy(data_dict)
resp_pchoice_unrewarded = copy.deepcopy(data_dict)
cue_br = copy.deepcopy(data_dict)
resp_br_rewarded = copy.deepcopy(data_dict)
resp_br_unrewarded = copy.deepcopy(data_dict)

block_rates = np.unique(sess_data['block_prob'])
choice_probs = np.unique(np.round(sess_data['choice_prob'], 2)*100)
choice_probs = choice_probs[~np.isnan(choice_probs)]
sides = ['left', 'right']

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        trial_data = sess_data[sess_data['sessid'] == sess_id]
        resp_sel = (trial_data['hit'] == True)
        # remove no responses
        trial_data = trial_data[resp_sel]
        sess_fp = fp_data[subj_id][sess_id]

        # get alignment trial filters
        choices = trial_data['choice'].to_numpy()
        stays = choices[:-1] == choices[1:]
        future_switches = np.append(~stays, False)
        future_stays = np.append(stays, False)
        switches = np.insert(~stays, 0, False)
        stays = np.insert(stays, 0, False)
        rewarded = trial_data['rewarded'].to_numpy()
        choice_prob = np.round(trial_data['choice_prob'].to_numpy(), 2)*100
        choice_prev_prob = np.round(trial_data['choice_prev_prob'].to_numpy(), 2)*100
        block_rate = trial_data['block_prob'].to_numpy()
        high_choice = trial_data['chose_high'].to_numpy()
        right_choice = trial_data['chose_right'].to_numpy()
        left_choice = trial_data['chose_left'].to_numpy()

        prev_right_choice = np.insert(right_choice[:-1], 0, False)
        prev_left_choice = np.insert(left_choice[:-1], 0, False)
        prev_rewarded = np.insert(rewarded[:-1], 0, False)
        prev_unrewarded = np.insert(~rewarded[:-1], 0, False)

        # get alignment times
        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1][resp_sel]
        abs_cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        abs_cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        abs_cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        abs_cue_ts = trial_start_ts + trial_data['response_cue_time']
        abs_resp_ts = trial_start_ts + trial_data['response_time']

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # aligned to center port on
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cport_on_ts, pre, post)

                cport_on['t'] = t
                cport_on[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded,:]
                cport_on[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded,:]
                cport_on[sess_id][signal_type][region]['stay'] = mat[stays,:]
                cport_on[sess_id][signal_type][region]['switch'] = mat[switches,:]
                cport_on[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays,:]
                cport_on[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches,:]
                cport_on[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays,:]
                cport_on[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_choice if side == 'left' else right_choice
                    prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice

                    cport_on[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                    cport_on[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                    cport_on[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]

                    cport_on[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    cport_on[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_rewarded & prev_side_sel,:]
                    cport_on[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_unrewarded & prev_side_sel,:]


                # aligned to center poke in
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cpoke_in_ts, pre, post)

                cpoke_in['t'] = t
                cpoke_in[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded,:]
                cpoke_in[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded,:]
                cpoke_in[sess_id][signal_type][region]['stay'] = mat[stays,:]
                cpoke_in[sess_id][signal_type][region]['switch'] = mat[switches,:]
                cpoke_in[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays,:]
                cpoke_in[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches,:]
                cpoke_in[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays,:]
                cpoke_in[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_choice if side == 'left' else right_choice

                    cpoke_in[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]


                # aligned to response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cue_ts, pre, post)

                cue['t'] = t
                cue[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded,:]
                cue[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded,:]
                cue[sess_id][signal_type][region]['stay'] = mat[stays,:]
                cue[sess_id][signal_type][region]['switch'] = mat[switches,:]
                cue[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays,:]
                cue[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches,:]
                cue[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays,:]
                cue[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_choice if side == 'left' else right_choice
                    prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice

                    cue[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    cue[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]
                    cue[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    cue[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_side_sel & prev_rewarded,:]
                    cue[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded,:]


                # aligned to center poke out
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cpoke_out_ts, pre, post)

                cpoke_out['t'] = t
                cpoke_out[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded,:]
                cpoke_out[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded,:]
                cpoke_out[sess_id][signal_type][region]['stay'] = mat[stays,:]
                cpoke_out[sess_id][signal_type][region]['switch'] = mat[switches,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_choice if side == 'left' else right_choice

                    cpoke_out[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]


                # aligned to response poke
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, pre, post)

                resp['t'] = t
                resp[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded,:]
                resp[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded,:]
                resp[sess_id][signal_type][region]['rewarded'] = mat[rewarded,:]
                resp[sess_id][signal_type][region]['unrewarded'] = mat[~rewarded,:]
                resp[sess_id][signal_type][region]['rewarded_prev_rewarded'] = mat[rewarded & prev_rewarded,:]
                resp[sess_id][signal_type][region]['rewarded_prev_unrewarded'] = mat[rewarded & prev_unrewarded,:]
                resp[sess_id][signal_type][region]['unrewarded_prev_rewarded'] = mat[~rewarded & prev_rewarded,:]
                resp[sess_id][signal_type][region]['unrewarded_prev_unrewarded'] = mat[~rewarded & prev_unrewarded,:]
                resp[sess_id][signal_type][region]['stay'] = mat[stays,:]
                resp[sess_id][signal_type][region]['switch'] = mat[switches,:]
                resp[sess_id][signal_type][region]['stay_rewarded'] = mat[rewarded & stays,:]
                resp[sess_id][signal_type][region]['switch_rewarded'] = mat[rewarded & switches,:]
                resp[sess_id][signal_type][region]['stay_unrewarded'] = mat[~rewarded & stays,:]
                resp[sess_id][signal_type][region]['switch_unrewarded'] = mat[~rewarded & switches,:]
                resp[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays,:]
                resp[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches,:]
                resp[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays,:]
                resp[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches,:]
                resp[sess_id][signal_type][region]['stay_rewarded_prev_rewarded'] = mat[rewarded & stays & prev_rewarded,:]
                resp[sess_id][signal_type][region]['stay_rewarded_prev_unrewarded'] = mat[rewarded & stays & prev_unrewarded,:]
                resp[sess_id][signal_type][region]['stay_unrewarded_prev_rewarded'] = mat[~rewarded & stays & prev_rewarded,:]
                resp[sess_id][signal_type][region]['stay_unrewarded_prev_unrewarded'] = mat[~rewarded & stays & prev_unrewarded,:]
                resp[sess_id][signal_type][region]['switch_rewarded_prev_rewarded'] = mat[rewarded & switches & prev_rewarded,:]
                resp[sess_id][signal_type][region]['switch_rewarded_prev_unrewarded'] = mat[rewarded & switches & prev_unrewarded,:]
                resp[sess_id][signal_type][region]['switch_unrewarded_prev_rewarded'] = mat[~rewarded & switches & prev_rewarded,:]
                resp[sess_id][signal_type][region]['switch_unrewarded_prev_unrewarded'] = mat[~rewarded & switches & prev_unrewarded,:]

                resp[sess_id][signal_type][region]['rewarded_future_stay'] = mat[rewarded & future_stays,:]
                resp[sess_id][signal_type][region]['rewarded_future_switch'] = mat[rewarded & future_switches,:]
                resp[sess_id][signal_type][region]['unrewarded_future_stay'] = mat[~rewarded & future_stays,:]
                resp[sess_id][signal_type][region]['unrewarded_future_switch'] = mat[~rewarded & future_switches,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_choice if side == 'left' else right_choice
                    prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice

                    resp[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    resp[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                    resp[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]

                    resp[sess_id][signal_type][region][side_type+'_rewarded'] = mat[side_sel & rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_unrewarded'] = mat[side_sel & ~rewarded,:]

                    resp[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    resp[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]

                    resp[sess_id][signal_type][region][side_type+'_rewarded_prev_rewarded'] = mat[side_sel & prev_rewarded & rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_rewarded_prev_unrewarded'] = mat[side_sel & prev_unrewarded & rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_unrewarded_prev_rewarded'] = mat[side_sel & prev_rewarded & ~rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_unrewarded_prev_unrewarded'] = mat[side_sel & prev_unrewarded & ~rewarded,:]

                    resp[sess_id][signal_type][region][side_type+'_stay_rewarded'] = mat[side_sel & stays & rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_rewarded'] = mat[side_sel & switches & rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_stay_unrewarded'] = mat[side_sel & stays & ~rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_unrewarded'] = mat[side_sel & switches & ~rewarded,:]

                    resp[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    resp[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]

                    resp[sess_id][signal_type][region][side_type+'_stay_rewarded_prev_rewarded'] = mat[side_sel & stays & rewarded & prev_rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_stay_rewarded_prev_unrewarded'] = mat[side_sel & stays & rewarded & prev_unrewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_stay_unrewarded_prev_rewarded'] = mat[side_sel & stays & ~rewarded & prev_rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_stay_unrewarded_prev_unrewarded'] = mat[side_sel & stays & ~rewarded & prev_unrewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_rewarded_prev_rewarded'] = mat[side_sel & switches & rewarded & prev_rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_rewarded_prev_unrewarded'] = mat[side_sel & switches & rewarded & prev_unrewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_unrewarded_prev_rewarded'] = mat[side_sel & switches & ~rewarded & prev_rewarded,:]
                    resp[sess_id][signal_type][region][side_type+'_switch_unrewarded_prev_unrewarded'] = mat[side_sel & switches & ~rewarded & prev_unrewarded,:]

                    resp[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_side_sel & prev_rewarded,:]
                    resp[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded,:]
                    resp[sess_id][signal_type][region]['prev_'+side_type+'_rewarded_prev_rewarded'] = mat[prev_side_sel & prev_rewarded & rewarded,:]
                    resp[sess_id][signal_type][region]['prev_'+side_type+'_rewarded_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded & rewarded,:]
                    resp[sess_id][signal_type][region]['prev_'+side_type+'_unrewarded_prev_rewarded'] = mat[prev_side_sel & prev_rewarded & ~rewarded,:]
                    resp[sess_id][signal_type][region]['prev_'+side_type+'_unrewarded_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded & ~rewarded,:]

                    resp[sess_id][signal_type][region][side_type+'_rewarded_future_stay'] = mat[side_sel & rewarded & future_stays,:]
                    resp[sess_id][signal_type][region][side_type+'_rewarded_future_switch'] = mat[side_sel & rewarded & future_switches,:]
                    resp[sess_id][signal_type][region][side_type+'_unrewarded_future_stay'] = mat[side_sel & ~rewarded & future_stays,:]
                    resp[sess_id][signal_type][region][side_type+'_unrewarded_future_switch'] = mat[side_sel & ~rewarded & future_switches,:]


                # aligned to response and reward by trial outcome and perceived port reward probability (p reward for port from prior trial)
                pre = 2
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, pre, post)

                resp_pchoice_rewarded['t'] = t
                resp_pchoice_unrewarded['t'] = t

                for p in choice_probs:
                    p_sel = choice_prev_prob == p
                    resp_pchoice_rewarded[sess_id][signal_type][region][p] = mat[rewarded & p_sel,:]
                    resp_pchoice_unrewarded[sess_id][signal_type][region][p] = mat[~rewarded & p_sel,:]


                # aligned to cue/response sorted by block reward probability
                cue_mat, cue_t = fp_utils.build_signal_matrix(signal, ts, abs_cue_ts, 3, 3)
                resp_mat, resp_t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, 2, 10)

                cue_br['t'] = cue_t
                resp_br_rewarded['t'] = resp_t
                resp_br_unrewarded['t'] = resp_t

                for br in block_rates:
                    br_sel = block_rate == br
                    cue_br[sess_id][signal_type][region][br] = cue_mat[br_sel,:]
                    resp_br_rewarded[sess_id][signal_type][region][br] = resp_mat[br_sel & rewarded,:]
                    resp_br_unrewarded[sess_id][signal_type][region][br] = resp_mat[br_sel & ~rewarded,:]



#%% Plot Alignment Results for single sessions

# title_suffix = '' #'420 Iso'
# outlier_thresh = 8 # z-score threshold

# for sess_id in sess_ids:
#     for signal_type in signal_types:

#         # get appropriate labels
#         signal_type_title, signal_type_label = fpah.get_signal_type_labels(signal_type)

#         if title_suffix != '':
#             signal_type_title += ' - ' + title_suffix

#         all_sub_titles = {'reward': 'Rewarded', 'noreward': 'Unrewarded', 'reward_stay': 'Stay | Reward',
#                           'noreward_stay': 'Stay | No reward', 'reward_switch': 'Switch | Reward', 'noreward_switch': 'Switch | No reward',
#                           'stay_rewarded': 'Rewarded Stay', 'stay_unrewarded': 'Unrewarded Stay',
#                           'switch_rewarded': 'Rewarded Switch', 'switch_unrewarded': 'Unrewarded Switch',
#                           'left': 'Left Choice', 'right': 'Right Choice', 'left_stay': 'Left Stay', 'left_switch': 'Left Switch',
#                           'right_stay': 'Right Stay', 'right_switch': 'Right Switch',
#                           'left_stay_rewarded': 'Rewarded Left Stay', 'left_stay_unrewarded': 'Unrewarded Left Stay',
#                           'left_switch_rewarded': 'Rewarded Left Switch', 'left_switch_unrewarded': 'Unrewarded Left Switch',
#                           'right_stay_rewarded': 'Rewarded Right Stay', 'right_stay_unrewarded': 'Unrewarded Right Stay',
#                           'right_switch_rewarded': 'Rewarded Right Switch', 'right_switch_unrewarded': 'Unrewarded Right Switch'}

#         for p in choice_probs:
#             all_sub_titles.update({p: '{:.0f}% Choice'.format(p)})

#         for br in block_rates:
#             all_sub_titles.update({br: 'Block Reward Rate: {}%'.format(br)})

#         fpah.plot_aligned_signals(resp_outcome[sess_id][signal_type], resp_outcome['t'], 'Response Aligned - {} (session {})'.format(signal_type_title, sess_id),
#                              all_sub_titles, 'Time from response poke (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                              trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(reward_outcome[sess_id][signal_type], reward_outcome['t'], 'Reward Aligned - {} (session {})'.format(signal_type_title, sess_id),
#                              all_sub_titles, 'Time from reward (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                              trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_reward_pchoice[sess_id][signal_type], resp_reward_pchoice['t'], 'Rewarded Response by Choice Reward Probability - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from reponse poke (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_noreward_pchoice[sess_id][signal_type], resp_noreward_pchoice['t'], 'Unrewarded Response by Choice Reward Probability - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from reponse poke (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cpoke_in_br[sess_id][signal_type], cpoke_in_br['t'], 'Center Poke In by Block Rate - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from poke in (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cpoke_out_br[sess_id][signal_type], cpoke_out_br['t'], 'Center Poke Out by Block Rate - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from poke out (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cue_br[sess_id][signal_type], cue_br['t'], 'Response Cue by Block Rate - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_choice_prev_outcome[sess_id][signal_type], resp_choice_prev_outcome['t'], 'Response by Choice after Outcome - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cpoke_in_choice_prev_outcome[sess_id][signal_type], cpoke_in_choice_prev_outcome['t'], 'Center Poke In by Future Choice after Outcome - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from poke in (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cpoke_out_choice_prev_outcome[sess_id][signal_type], cpoke_out_choice_prev_outcome['t'], 'Center Poke Out by Future Choice after Outcome - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from poke out (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cue_choice[sess_id][signal_type], cue_choice['t'], 'Response Cue by Choice - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_choice[sess_id][signal_type], resp_choice['t'], 'Response by Choice - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_choice_curr_outcome[sess_id][signal_type], resp_choice_curr_outcome['t'], 'Response by Choice and Outcome - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cue_side[sess_id][signal_type], cue_side['t'], 'Response Cue by Side - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_side[sess_id][signal_type], resp_side['t'], 'Response by Side - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cue_side_choice[sess_id][signal_type], cue_side_choice['t'], 'Response Cue by Side & Choice - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_side_choice[sess_id][signal_type], resp_side_choice['t'], 'Response by Side & Choice - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(cue_side_choice_outcome[sess_id][signal_type], cue_side_choice_outcome['t'], 'Response Cue by Side, Choice & Outcome - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)

#         fpah.plot_aligned_signals(resp_side_choice_outcome[sess_id][signal_type], resp_side_choice_outcome['t'], 'Response by Side, Choice & Outcome - {} (session {})'.format(signal_type_title, sess_id),
#                               all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh,
#                               trial_markers=block_trans_idxs)


# %% Plot average signals from multiple sessions on the same axes

# modify these options to change what will be used in the average signal plots
signal_type = 'z_dff_iso' # 'dff_iso', 'df_baseline_iso', 'raw_lig'
regions = ['DMS', 'PL']
subjects = list(sess_ids.keys())
filter_outliers = False
outlier_thresh = 20
use_se = True
ph = 3.5;
pw = 5;
n_reg = len(regions)
resp_reg_xlims = {'DMS': [-1.5,2], 'PL': [-2,10]}

# make this wrapper to simplify the stack command by not having to include the options declared above
def stack_mats(mat_dict, groups=None):
    return fpah.stack_fp_mats(mat_dict, regions, sess_ids, subjects, signal_type, filter_outliers, outlier_thresh, groups)

def calc_error(mat):
    if use_se:
        return utils.stderr(mat, axis=0)
    else:
        return np.std(mat, axis=0)


cport_on_mats = stack_mats(cport_on)
cpoke_in_mats = stack_mats(cpoke_in)
cpoke_out_mats = stack_mats(cpoke_out)
cue_mats = stack_mats(cue)
resp_mats = stack_mats(resp)
resp_pchoice_rewarded_mats = stack_mats(resp_pchoice_rewarded)
resp_pchoice_unrewarded_mats = stack_mats(resp_pchoice_unrewarded)
cue_br_mats = stack_mats(cue_br)
resp_br_rewarded_mats = stack_mats(resp_br_rewarded)
resp_br_unrewarded_mats = stack_mats(resp_br_unrewarded)

# %% Choice, side, and prior reward groupings for multiple alignment points

# choice, side, & side/choice

choice_groups = ['stay', 'switch']
side_groups = ['contra', 'ipsi']
choice_side_groups = ['contra_stay', 'contra_switch', 'ipsi_stay', 'ipsi_switch']
group_labels = {'stay': 'Stay', 'switch': 'Switch',
                'ipsi': 'Ipsi', 'contra': 'Contra',
                'contra_stay': 'Contra Stay', 'contra_switch': 'Contra Switch',
                'ipsi_stay': 'Ipsi Stay', 'ipsi_switch': 'Ipsi Switch'}

mats = [cport_on_mats, cpoke_in_mats, cue_mats, cpoke_out_mats, resp_mats]
ts = [cport_on['t'], cpoke_in['t'], cue['t'], cpoke_out['t'], resp['t']]
titles = ['Center Port On', 'Center Poke In', 'Response Cue', 'Center Poke Out', 'Response']
x_labels = ['port on', 'poke in', 'response cue', 'poke out', 'response poke']
n_cols = 3

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Choice/Side Groupings Aligned to ' + title)

    legend_locs = ['upper right', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Choice - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in choice_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Side - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in side_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Choice/Side - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in choice_side_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])

        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# prior reward, choice/prior reward, side/choice/prior reward
rew_groups = ['prev_rewarded', 'prev_unrewarded']
choice_rew_groups = ['stay_prev_rewarded', 'switch_prev_rewarded', 'stay_prev_unrewarded', 'switch_prev_unrewarded']
side_rew_groups = ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded']
group_labels = {'prev_rewarded': 'Reward', 'prev_unrewarded': 'No Reward',
                'stay_prev_rewarded': 'Stay | Reward', 'switch_prev_rewarded': 'Switch | Reward',
                'stay_prev_unrewarded': 'Stay | No Reward', 'switch_prev_unrewarded': 'Switch | No Reward',
                'contra_prev_rewarded': 'Contra | Reward', 'ipsi_prev_rewarded': 'Ipsi | Reward',
                'contra_prev_unrewarded': 'Contra | No Reward', 'ipsi_prev_unrewarded': 'Ipsi | No Reward'}

mats = [cport_on_mats, cue_mats, cpoke_out_mats, resp_mats]
ts = [cport_on['t'], cue['t'], cpoke_out['t'], resp['t']]
titles = ['Center Port On', 'Response Cue', 'Center Poke Out', 'Response']
x_labels = ['port on', 'response cue', 'poke out', 'response poke']

n_cols = 3

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Prior Outcome by Choice/Side Groupings Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Prior Outcome - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Prior Outcome/Choice - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in choice_rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Prior Outcome/Side - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in side_rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])

        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# prior reward/choice/side
rew_groups = ['contra_stay_prev_rewarded', 'contra_switch_prev_rewarded', 'ipsi_stay_prev_rewarded', 'ipsi_switch_prev_rewarded']
unrew_groups = ['contra_stay_prev_unrewarded', 'contra_switch_prev_unrewarded', 'ipsi_stay_prev_unrewarded', 'ipsi_switch_prev_unrewarded']
# group_labels = {'contra_stay_prev_rewarded': 'Contra Stay', 'contra_switch_prev_rewarded': 'Contra Switch',
#                 'ipsi_stay_prev_rewarded': 'Ipsi Stay', 'ipsi_switch_prev_rewarded': 'Ipsi Switch',
#                 'contra_stay_prev_unrewarded': 'Contra Stay', 'contra_switch_prev_unrewarded': 'Contra Switch',
#                 'ipsi_stay_prev_unrewarded': 'Ipsi Stay', 'ipsi_switch_prev_unrewarded': 'Ipsi Switch'}
group_labels = {'contra_stay_prev_rewarded': 'Contra -> Contra', 'contra_switch_prev_rewarded': 'Ipsi -> Contra',
                'ipsi_stay_prev_rewarded': 'Ipsi -> Ipsi', 'ipsi_switch_prev_rewarded': 'Contra -> Ipsi',
                'contra_stay_prev_unrewarded': 'Contra -> Contra', 'contra_switch_prev_unrewarded': 'Ipsi -> Contra',
                'ipsi_stay_prev_unrewarded': 'Ipsi -> Ipsi', 'ipsi_switch_prev_unrewarded': 'Contra -> Ipsi'}

mats = [cport_on_mats, cue_mats, cpoke_out_mats, resp_mats]
ts = [cport_on['t'], cue['t'], cpoke_out['t'], resp['t']]
titles = ['Center Port On', 'Response Cue', 'Center Poke Out', 'Response']
x_labels = ['port on', 'response cue', 'poke out', 'response poke']

n_cols = 2

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Prior Outcome by Choice/Side Groupings Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Prior Reward - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} No Prior Reward - {}'.format(region, title))
        if title == 'Response':
            plot_utils.plot_dashlines(0.5, ax=ax)
        for group in unrew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)



# %% Groupings by previous side choice and previous reward

# cport on
xlims = [-1.5,1.5]
n_cols = 4
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Prior Outcome by Prior or Next Choice - Center Port On')

t = cport_on['t']

side_groups = ['contra', 'ipsi']
prev_side_groups = ['prev_contra', 'prev_ipsi']
side_rew_groups = ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded']
prev_side_rew_groups = ['prev_contra_prev_rewarded', 'prev_ipsi_prev_rewarded', 'prev_contra_prev_unrewarded', 'prev_ipsi_prev_unrewarded']
group_labels = {'contra': 'Contra', 'ipsi': 'Ipsi', 'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi',
                'contra_prev_rewarded': 'Contra | Reward', 'ipsi_prev_rewarded': 'Ipsi | Reward',
                'contra_prev_unrewarded': 'Contra | No Reward', 'ipsi_prev_unrewarded': 'Ipsi | No Reward',
                'prev_contra_prev_rewarded': 'Prev Rewarded Contra', 'prev_ipsi_prev_rewarded': 'Prev Rewarded Ipsi',
                'prev_contra_prev_unrewarded': 'Prev Unrewarded Contra', 'prev_ipsi_prev_unrewarded': 'Prev Unrewarded Ipsi'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    ax = axs[i,0]
    ax.set_title(region+' Previous Choice')
    for group in prev_side_groups:
        act = cport_on_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from center port on (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Future Choice')
    for group in side_groups:
        act = cport_on_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from center port on (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Prev Outcome and Previous Choice')
    for group in prev_side_rew_groups:
        act = cport_on_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from center port on (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title(region+' Prev Outcome and Future Choice')
    for group in side_rew_groups:
        act = cport_on_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from center port on (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# response cue
xlims = [-1.5,1.5]
n_cols = 4
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Prior or Next Choice by Prior Outcome - Response Cue')

t = cue['t']

side_groups = ['contra', 'ipsi']
prev_side_groups = ['prev_contra', 'prev_ipsi']
side_rew_groups = ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded']
prev_side_rew_groups = ['prev_contra_prev_rewarded', 'prev_ipsi_prev_rewarded', 'prev_contra_prev_unrewarded', 'prev_ipsi_prev_unrewarded']
group_labels = {'contra': 'Contra', 'ipsi': 'Ipsi', 'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi',
                'contra_prev_rewarded': 'Contra | Reward', 'ipsi_prev_rewarded': 'Ipsi | Reward',
                'contra_prev_unrewarded': 'Contra | No Reward', 'ipsi_prev_unrewarded': 'Ipsi | No Reward',
                'prev_contra_prev_rewarded': 'Prev Rewarded Contra', 'prev_ipsi_prev_rewarded': 'Prev Rewarded Ipsi',
                'prev_contra_prev_unrewarded': 'Prev Unrewarded Contra', 'prev_ipsi_prev_unrewarded': 'Prev Unrewarded Ipsi'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    ax = axs[i,0]
    ax.set_title(region+' Previous Choice')
    for group in prev_side_groups:
        act = cue_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response cue (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Future Choice')
    for group in side_groups:
        act = cue_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response cue (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Prev Outcome and Previous Choice')
    for group in prev_side_rew_groups:
        act = cue_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response cue (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title(region+' Prev Outcome and Future Choice')
    for group in side_rew_groups:
        act = cue_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response cue (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# response, prior outcome and prior/current choice
n_cols = 2
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Prior Outcome by Prior or Current Choice - Response')

t = resp['t']

side_rew_groups = ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded']
prev_side_rew_groups = ['prev_contra_prev_rewarded', 'prev_ipsi_prev_rewarded', 'prev_contra_prev_unrewarded', 'prev_ipsi_prev_unrewarded']
group_labels = {'contra_prev_rewarded': 'Contra | Reward', 'ipsi_prev_rewarded': 'Ipsi | Reward',
                'contra_prev_unrewarded': 'Contra | No Reward', 'ipsi_prev_unrewarded': 'Ipsi | No Reward',
                'prev_contra_prev_rewarded': 'Prev Rewarded Contra', 'prev_ipsi_prev_rewarded': 'Prev Rewarded Ipsi',
                'prev_contra_prev_unrewarded': 'Prev Unrewarded Contra', 'prev_ipsi_prev_unrewarded': 'Prev Unrewarded Ipsi'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Prev Outcome and Previous Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in prev_side_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Prev Outcome and Current Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

# response, prior & current outcome by prior/current choice
xlims = [-1.5,1.5]
n_cols = 4
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Prior & Current Outcome by Prior or Next Choice - Response')

t = resp['t']

side_rew_groups = ['contra_rewarded_prev_rewarded', 'ipsi_rewarded_prev_rewarded', 'contra_rewarded_prev_unrewarded', 'ipsi_rewarded_prev_unrewarded']
side_unrew_groups = ['contra_unrewarded_prev_rewarded', 'ipsi_unrewarded_prev_rewarded', 'contra_unrewarded_prev_unrewarded', 'ipsi_unrewarded_prev_unrewarded']
prev_side_rew_groups = ['prev_contra_rewarded_prev_rewarded', 'prev_ipsi_rewarded_prev_rewarded', 'prev_contra_rewarded_prev_unrewarded', 'prev_ipsi_rewarded_prev_unrewarded']
prev_side_unrew_groups = ['prev_contra_unrewarded_prev_rewarded', 'prev_ipsi_unrewarded_prev_rewarded', 'prev_contra_unrewarded_prev_unrewarded', 'prev_ipsi_unrewarded_prev_unrewarded']
group_labels = {'contra_rewarded_prev_rewarded': 'Contra | Reward', 'ipsi_rewarded_prev_rewarded': 'Ipsi | Reward',
                'contra_rewarded_prev_unrewarded': 'Contra | No Reward', 'ipsi_rewarded_prev_unrewarded': 'Ipsi | No Reward',
                'contra_unrewarded_prev_rewarded': 'Contra | Reward', 'ipsi_unrewarded_prev_rewarded': 'Ipsi | Reward',
                'contra_unrewarded_prev_unrewarded': 'Contra | No Reward', 'ipsi_unrewarded_prev_unrewarded': 'Ipsi | No Reward',
                'prev_contra_rewarded_prev_rewarded': 'Prev Rewarded Contra', 'prev_ipsi_rewarded_prev_rewarded': 'Prev Rewarded Ipsi',
                'prev_contra_rewarded_prev_unrewarded': 'Prev Unrewarded Contra', 'prev_ipsi_rewarded_prev_unrewarded': 'Prev Unrewarded Ipsi',
                'prev_contra_unrewarded_prev_rewarded': 'Prev Rewarded Contra', 'prev_ipsi_unrewarded_prev_rewarded': 'Prev Rewarded Ipsi',
                'prev_contra_unrewarded_prev_unrewarded': 'Prev Unrewarded Contra', 'prev_ipsi_unrewarded_prev_unrewarded': 'Prev Unrewarded Ipsi'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Rewarded by Previous Outcome & Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in prev_side_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Unrewarded by Previous Outcome & Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in prev_side_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Rewarded Choice by Previous Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title(region+' Unrewarded Choice by Previous Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# %% Current outcome at time of response, for multiple different groupings

# response outcome
n_cols = 2
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Current & Prior Outcome - Response')
t = resp['t']

rew_groups = ['rewarded', 'unrewarded']
prev_rew_groups = ['rewarded_prev_rewarded', 'rewarded_prev_unrewarded', 'unrewarded_prev_rewarded', 'unrewarded_prev_unrewarded']
group_labels = {'rewarded': 'Rewarded', 'unrewarded': 'Unrewarded',
                'rewarded_prev_rewarded': 'Rewarded | Reward', 'rewarded_prev_unrewarded': 'Rewarded | No Reward',
                'unrewarded_prev_rewarded': 'Unrewarded | Reward', 'unrewarded_prev_unrewarded': 'Unrewarded | No Reward'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Current Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Current & Prior Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in prev_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# response side/outcome
n_cols = 3
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Current & Prior Outcome by Choice Side - Response')

side_groups = ['contra_rewarded', 'ipsi_rewarded', 'contra_unrewarded', 'ipsi_unrewarded']
side_rew_groups = ['contra_rewarded_prev_rewarded', 'ipsi_rewarded_prev_rewarded', 'contra_rewarded_prev_unrewarded', 'ipsi_rewarded_prev_unrewarded']
side_unrew_groups = ['contra_unrewarded_prev_rewarded', 'ipsi_unrewarded_prev_rewarded', 'contra_unrewarded_prev_unrewarded', 'ipsi_unrewarded_prev_unrewarded']
group_labels = {'contra_rewarded': 'Rewarded Contra', 'ipsi_rewarded': 'Rewarded Ipsi',
                'contra_unrewarded': 'Unrewarded Contra', 'ipsi_unrewarded': 'Unrewarded Ipsi',
                'contra_rewarded_prev_rewarded': 'Contra | Reward', 'ipsi_rewarded_prev_rewarded': 'Ipsi | Reward',
                'contra_rewarded_prev_unrewarded': 'Contra | No Reward', 'ipsi_rewarded_prev_unrewarded': 'Ipsi | No Reward',
                'contra_unrewarded_prev_rewarded': 'Contra | Reward', 'ipsi_unrewarded_prev_rewarded': 'Ipsi | Reward',
                'contra_unrewarded_prev_unrewarded': 'Contra | No Reward', 'ipsi_unrewarded_prev_unrewarded': 'Ipsi | No Reward'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Side/Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Rewarded Side by Prior Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Unrewarded Side by Prior Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

# response choice/outcome
n_cols = 3
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Current & Prior Outcome by Choice - Response')

choice_groups = ['stay_rewarded', 'switch_rewarded', 'stay_unrewarded', 'switch_unrewarded']
choice_rew_groups = ['stay_rewarded_prev_rewarded', 'switch_rewarded_prev_rewarded', 'stay_rewarded_prev_unrewarded', 'switch_rewarded_prev_unrewarded']
choice_unrew_groups = ['stay_unrewarded_prev_rewarded', 'switch_unrewarded_prev_rewarded', 'stay_unrewarded_prev_unrewarded', 'switch_unrewarded_prev_unrewarded']
group_labels = {'stay_rewarded': 'Rewarded Stay', 'switch_rewarded': 'Rewarded Switch',
                'stay_unrewarded': 'Unrewarded Stay', 'switch_unrewarded': 'Unrewarded Switch',
                'stay_rewarded_prev_rewarded': 'Stay | Reward', 'switch_rewarded_prev_rewarded': 'Switch | Reward',
                'stay_rewarded_prev_unrewarded': 'Stay | No Reward', 'switch_rewarded_prev_unrewarded': 'Switch | No Reward',
                'stay_unrewarded_prev_rewarded': 'Stay | Reward', 'switch_unrewarded_prev_rewarded': 'Switch | Reward',
                'stay_unrewarded_prev_unrewarded': 'Stay | No Reward', 'switch_unrewarded_prev_unrewarded': 'Switch | No Reward'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Choice/Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in choice_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Rewarded Choice by Prior Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in choice_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Unrewarded Choice by Prior Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in choice_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

# response choice/side/outcome
n_cols = 4
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Side Stay/Switch Choices By Outcome - Response')

side_groups = ['contra_rewarded', 'ipsi_rewarded', 'contra_unrewarded', 'ipsi_unrewarded']
choice_groups = ['stay_rewarded', 'switch_rewarded', 'stay_unrewarded', 'switch_unrewarded']
rew_groups = ['contra_stay_rewarded', 'contra_switch_rewarded', 'ipsi_stay_rewarded', 'ipsi_switch_rewarded']
unrew_groups = ['contra_stay_unrewarded', 'contra_switch_unrewarded', 'ipsi_stay_unrewarded', 'ipsi_switch_unrewarded']
group_labels = {'contra_rewarded': 'Rewarded Contra', 'ipsi_rewarded': 'Rewarded Ipsi',
                'contra_unrewarded': 'Unrewarded Contra', 'ipsi_unrewarded': 'Unrewarded Ipsi',
                'stay_rewarded': 'Rewarded Stay', 'switch_rewarded': 'Rewarded Switch',
                'stay_unrewarded': 'Unrewarded Stay', 'switch_unrewarded': 'Unrewarded Switch',
                'contra_stay_rewarded': 'Contra Stay', 'contra_stay_unrewarded': 'Contra Stay',
                'contra_switch_rewarded': 'Contra Switch', 'contra_switch_unrewarded': 'Contra Switch',
                'ipsi_stay_rewarded': 'Ipsi Stay', 'ipsi_stay_unrewarded': 'Ipsi Stay',
                'ipsi_switch_rewarded': 'Ipsi Switch', 'ipsi_switch_unrewarded': 'Ipsi Switch'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Side/Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Choice/Outcome')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in choice_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Rewarded Side/Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title(region+' Unrewarded Side/Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# response choice/side/outcome/prior outcome
n_cols = 4
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Side Stay/Switch Choices By Prior and Current Outcome - Response')

rew_prev_rew_groups = ['contra_stay_rewarded_prev_rewarded', 'contra_switch_rewarded_prev_rewarded', 'ipsi_stay_rewarded_prev_rewarded', 'ipsi_switch_rewarded_prev_rewarded']
rew_prev_unrew_groups = ['contra_stay_rewarded_prev_unrewarded', 'contra_switch_rewarded_prev_unrewarded', 'ipsi_stay_rewarded_prev_unrewarded', 'ipsi_switch_rewarded_prev_unrewarded']
unrew_prev_rew_groups = ['contra_stay_unrewarded_prev_rewarded', 'contra_switch_unrewarded_prev_rewarded', 'ipsi_stay_unrewarded_prev_rewarded', 'ipsi_switch_unrewarded_prev_rewarded']
unrew_prev_unrew_groups = ['contra_stay_unrewarded_prev_unrewarded', 'contra_switch_unrewarded_prev_unrewarded', 'ipsi_stay_unrewarded_prev_unrewarded', 'ipsi_switch_unrewarded_prev_unrewarded']
group_labels = {'contra_stay_rewarded_prev_rewarded': 'Contra Stay', 'contra_switch_rewarded_prev_rewarded': 'Contra Switch',
                'ipsi_stay_rewarded_prev_rewarded': 'Ipsi Stay', 'ipsi_switch_rewarded_prev_rewarded': 'Ipsi Switch',
                'contra_stay_rewarded_prev_unrewarded': 'Contra Stay', 'contra_switch_rewarded_prev_unrewarded': 'Contra Switch',
                'ipsi_stay_rewarded_prev_unrewarded': 'Ipsi Stay', 'ipsi_switch_rewarded_prev_unrewarded': 'Ipsi Switch',
                'contra_stay_unrewarded_prev_rewarded': 'Contra Stay', 'contra_switch_unrewarded_prev_rewarded': 'Contra Switch',
                'ipsi_stay_unrewarded_prev_rewarded': 'Ipsi Stay', 'ipsi_switch_unrewarded_prev_rewarded': 'Ipsi Switch',
                'contra_stay_unrewarded_prev_unrewarded': 'Contra Stay', 'contra_switch_unrewarded_prev_unrewarded': 'Contra Switch',
                'ipsi_stay_unrewarded_prev_unrewarded': 'Ipsi Stay', 'ipsi_switch_unrewarded_prev_unrewarded': 'Ipsi Switch'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Rewarded | Reward')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_prev_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Rewarded | No Reward')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_prev_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Unrewarded | Reward')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_prev_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title(region+' Unrewarded | No Reward')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_prev_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

# %% Response aligned grouped by side/reward/future response

t = resp['t']

n_cols = 3
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Current Outcome by Future Choice - Response')

choice_groups = ['rewarded_future_stay', 'rewarded_future_switch', 'unrewarded_future_stay', 'unrewarded_future_switch']
side_rew_groups = ['contra_rewarded_future_stay', 'contra_rewarded_future_switch', 'ipsi_rewarded_future_stay', 'ipsi_rewarded_future_switch']
side_unrew_groups = ['contra_unrewarded_future_stay', 'contra_unrewarded_future_switch', 'ipsi_unrewarded_future_stay', 'ipsi_unrewarded_future_switch']
group_labels = {'rewarded_future_stay': 'Stay | Reward', 'rewarded_future_switch': 'Switch | Reward',
                'unrewarded_future_stay': 'Stay | No Reward', 'unrewarded_future_switch': 'Switch | No Reward',
                'contra_rewarded_future_stay': 'Stay | Rewarded Contra', 'contra_rewarded_future_switch': 'Switch | Rewarded Contra',
                'ipsi_rewarded_future_stay': 'Stay | Rewarded Ipsi', 'ipsi_rewarded_future_switch': 'Switch | Rewarded Ipsi',
                'contra_unrewarded_future_stay': 'Stay | Unrewarded Contra', 'contra_unrewarded_future_switch': 'Switch | Unrewarded Contra',
                'ipsi_unrewarded_future_stay': 'Stay | Unrewarded Ipsi', 'ipsi_unrewarded_future_switch': 'Switch | Unrewarded Ipsi'}

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    xlims = resp_reg_xlims[region]

    ax = axs[i,0]
    ax.set_title(region+' Outcome/Future Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in choice_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Rewarded Side by Future Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_rew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Unrewarded Side by Future Choice')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for group in side_unrew_groups:
        act = resp_mats[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# %% Group by block rates and port probability

# block rates at cue and response

n_cols = 3
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Response Cue and Response by Block Rate')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    ax = axs[i,0]
    ax.set_title(region+' Response Cue')
    for br in block_rates:
        act = cue_br_mats[region][br]
        plot_utils.plot_psth(np.nanmean(act, axis=0), cue_br['t'], calc_error(act), ax, label='{}%'.format(br))
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response cue (s)')
    ax.set_xlim([-1.5, 1.5])
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title(region+' Rewarded Response')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for br in block_rates:
        act = resp_br_rewarded_mats[region][br]
        plot_utils.plot_psth(np.nanmean(act, axis=0), resp_br_rewarded['t'], calc_error(act), ax, label='{}%'.format(br))
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(resp_reg_xlims[region])
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title(region+' Unrewarded Response')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for br in block_rates:
        act = resp_br_unrewarded_mats[region][br]
        plot_utils.plot_psth(np.nanmean(act, axis=0), resp_br_unrewarded['t'], calc_error(act), ax, label='{}%'.format(br))
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(resp_reg_xlims[region])
    ax.legend(loc=legend_loc)

# choice reward probability at response
n_cols = 2
fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Response by Choice Reward Rate')
t = resp_pchoice_rewarded['t']

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    ax = axs[i,0]
    ax.set_title(region+' Rewarded')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for p in choice_probs:
        act = resp_pchoice_rewarded_mats[region][p]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{:.0f}%'.format(p))
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(resp_reg_xlims[region])
    ax.legend(loc=legend_loc, ncols=2)

    ax = axs[i,1]
    ax.set_title(region+' Unrewarded')
    plot_utils.plot_dashlines(0.5, ax=ax)
    for p in choice_probs:
        act = resp_pchoice_unrewarded_mats[region][p]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{:.0f}%'.format(p))
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from response poke (s)')
    ax.set_xlim(resp_reg_xlims[region])
    ax.legend(loc=legend_loc, ncols=2)

# rewarded choice probability normalized to the activity at the time of the response poke
fig, ax = plt.subplots(1, 1, figsize=(pw, ph*1.25), layout='constrained', sharey='row')
fig.suptitle('Rewarded Responses Normalized to Activity at t=0')
t = resp_pchoice_rewarded['t']
center_idx = np.argmin(np.abs(t))
region = 'PFC'

plot_utils.plot_dashlines(0.5, ax=ax)
for p in choice_probs:
    act = resp_pchoice_rewarded_mats[region][p]
    plot_utils.plot_psth(np.nanmean(act, axis=0) - np.nanmean(act[:,center_idx]), t, calc_error(act), ax, label='{:.0f}%'.format(p))
ax.set_ylabel('Shifted Z-scored ΔF/F')
ax.set_xlabel('Time from response poke (s)')
ax.set_xlim(resp_reg_xlims[region])
ax.legend(loc=legend_loc, ncols=2)
