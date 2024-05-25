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
import hankslab_db.tonecatdelayresp_db as db
import fp_analysis_helpers as fpah
import numpy as np
import matplotlib.pyplot as plt
import pickle
import copy

# %% Load behavior data

# get all session ids for given protocol
sess_ids = db_access.get_fp_protocol_subj_sess_ids('ToneCatDelayResp', 8)

loc_db = db.LocalDB_ToneCatDelayResp()
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids)) # reload=True

# old behavior didn't have cport on
if not 'cport_on_time' in sess_data.columns:
    sess_data['cport_on_time'] = 0.016

sess_data['tone_info_str'] = sess_data['tone_info'].apply(lambda x: ', '.join(x) if utils.is_list(x) else x)
sess_data['prev_tone_info_str'] = sess_data['prev_choice_tone_info'].apply(lambda x: ', '.join(x) if utils.is_list(x) else x)

# get fiber photometry data
fp_data = loc_db.get_sess_fp_data(utils.flatten(sess_ids), reload=True) # , reload=True
# separate into different dictionaries
implant_info = fp_data['implant_info']
fp_data = fp_data['fp_data']


# %% Process photometry data in different ways

isos = np.array(['420', '405'])
ligs = np.array(['490', '465'])

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        raw_signals = fp_data[subj_id][sess_id]['raw_signals']

        fp_data[subj_id][sess_id]['processed_signals'] = {}

        for region in raw_signals.keys():
            lig_sel = np.array([k in raw_signals[region].keys() for k in ligs])
            iso_sel = np.array([k in raw_signals[region].keys() for k in isos])
            if sum(lig_sel) > 1:
                lig = ligs[0]
            elif sum(lig_sel) == 1:
                lig = ligs[lig_sel][0]
            else:
                raise Exception('No ligand wavelength found')

            if sum(iso_sel) > 1:
                iso = isos[0]
            elif sum(iso_sel) == 1:
                iso = isos[iso_sel][0]
            else:
                raise Exception('No isosbestic wavelength found')

            raw_lig = raw_signals[region][lig]
            raw_iso = raw_signals[region][iso]

            fp_data[subj_id][sess_id]['processed_signals'][region] = fpah.get_all_processed_signals(raw_lig, raw_iso)

# %% Observe the full signals

sub_signal = [] # sub signal time limits in seconds
filter_outliers = True

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        if len(sub_signal) > 0:
            fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'], title='Full Signals - Session {}'.format(sess_id),
                                        filter_outliers=filter_outliers, outlier_zthresh=7,
                                        t_min=sub_signal[0], t_max=sub_signal[1], dec=1)

        else:
            fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'], title='Full Signals - Session {}'.format(sess_id),
                                        filter_outliers=filter_outliers, outlier_zthresh=7)


# %% Construct aligned signal matrices grouped by various factors

signal_types = ['z_dff_iso'] # 'baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}
cport_on = copy.deepcopy(data_dict)
cpoke_in = copy.deepcopy(data_dict)
tones = copy.deepcopy(data_dict)
response_cue = copy.deepcopy(data_dict)
cpoke_out = copy.deepcopy(data_dict)
response = copy.deepcopy(data_dict)

stim_types = np.array(sorted(sess_data['tone_info_str'].unique().tolist(), key=lambda x: (len(x), x)))
tone_types = sess_data['response_tone'].unique().tolist()

sides = ['left', 'right']

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        # ignore trials that didn't start
        trial_start_sel = trial_data['trial_started']
        trial_data = trial_data[trial_start_sel]

        # create various trial selection criteria
        hit_sel = trial_data['hit'] == True
        miss_sel = trial_data['hit'] == False
        bail_sel = trial_data['bail'] == True

        prev_hit_sel = np.insert(hit_sel[:-1].to_numpy(), 0, False)
        prev_miss_sel = np.insert(miss_sel[:-1].to_numpy(), 0, False)
        prev_bail_sel = np.insert(bail_sel[:-1].to_numpy(), 0, False)

        next_resp_sel = np.append(~bail_sel[1:].to_numpy(), False)
        next_bail_sel = np.append(bail_sel[1:].to_numpy(), False)

        single_tone_sel = trial_data['n_tones'] == 1
        two_tone_sel = trial_data['n_tones'] == 2

        choices = trial_data['choice'].to_numpy()

        left_sel = choices == 'left'
        right_sel = choices == 'right'
        prev_left_sel = np.insert(left_sel[:-1], 0, False)
        prev_right_sel = np.insert(right_sel[:-1], 0, False)

        prev_choice_same = (choices == trial_data['prev_choice_side']) & ~bail_sel
        prev_choice_diff = (choices != trial_data['prev_choice_side']) & ~bail_sel & ~trial_data['prev_choice_side'].isnull()

        next_choice_same = np.append(choices[:-1] == choices[1:], False) & ~bail_sel
        next_choice_diff = np.append(choices[:-1] != choices[1:], False) & ~bail_sel & ~next_bail_sel

        prev_trial_same = (trial_data['tone_info_str'] == trial_data['prev_tone_info_str']) & ~bail_sel
        prev_trial_diff = (trial_data['tone_info_str'] != trial_data['prev_tone_info_str']) & ~bail_sel & ~trial_data['prev_tone_info_str'].isnull()

        prev_correct_same = (trial_data['correct_port'] == trial_data['prev_choice_correct_port']) & ~bail_sel
        prev_correct_diff = (trial_data['correct_port'] != trial_data['prev_choice_correct_port']) & ~bail_sel & ~trial_data['prev_choice_correct_port'].isnull()

        tone_infos = trial_data['tone_info']
        trial_stims = trial_data['tone_info_str']
        prev_trial_stims = trial_data['prev_tone_info_str']
        correct_sides = trial_data['correct_port']

        # create the alignment points
        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1][trial_start_sel]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        response_cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']

        # abs_pre_poke_windows = cpoke_in_ts.to_numpy()[:, None] + pre_poke_norm_window[None, :]
        # cpoke_in_windows = abs_pre_poke_windows - cpoke_in_ts.to_numpy()[:, None]
        # cpoke_out_windows = abs_pre_poke_windows - cpoke_out_ts.to_numpy()[:, None]
        # response_windows = abs_pre_poke_windows - response_ts.to_numpy()[:, None]
        # response_cue_windows = abs_pre_poke_windows - response_cue_ts.to_numpy()[:, None]
        # cport_on_windows = abs_pre_poke_windows - cport_on_ts[:, None]

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # Center port on
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cport_on_ts, pre, post)

                cport_on['t'] = t
                cport_on[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel,:]
                cport_on[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel,:]
                cport_on[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel,:]
                cport_on[sess_id][signal_type][region]['hit'] = mat[hit_sel,:]
                cport_on[sess_id][signal_type][region]['miss'] = mat[miss_sel,:]
                cport_on[sess_id][signal_type][region]['bail'] = mat[bail_sel,:]
                cport_on[sess_id][signal_type][region]['response'] = mat[~bail_sel,:]
                cport_on[sess_id][signal_type][region]['stay'] = mat[prev_choice_same,:]
                cport_on[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff,:]
                cport_on[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same,:]
                cport_on[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff,:]
                cport_on[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same,:]
                cport_on[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_sel if side == 'left' else right_sel
                    prev_side_sel = prev_left_sel if side == 'left' else prev_right_sel

                    cport_on[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_stay'] = mat[prev_choice_same & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_switch'] = mat[prev_choice_diff & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_prev_bail'] = mat[prev_bail_sel & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    cport_on[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    cport_on[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    cport_on[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                    cport_on[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]

                for stim_type in stim_types:
                    prev_stim_sel = prev_trial_stims == stim_type
                    cport_on[sess_id][signal_type][region]['prev_stim_'+stim_type] = mat[prev_stim_sel & ~prev_bail_sel,:]
                    cport_on[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_hit'] = mat[prev_stim_sel & prev_hit_sel,:]
                    cport_on[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_miss'] = mat[prev_stim_sel & prev_miss_sel,:]

                # Center poke in
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_in_ts, pre, post)

                cpoke_in['t'] = t
                cpoke_in[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel,:]
                cpoke_in[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel,:]
                cpoke_in[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel,:]
                cpoke_in[sess_id][signal_type][region]['hit'] = mat[hit_sel,:]
                cpoke_in[sess_id][signal_type][region]['miss'] = mat[miss_sel,:]
                cpoke_in[sess_id][signal_type][region]['bail'] = mat[bail_sel,:]
                cpoke_in[sess_id][signal_type][region]['response'] = mat[~bail_sel,:]
                cpoke_in[sess_id][signal_type][region]['stay'] = mat[prev_choice_same,:]
                cpoke_in[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff,:]
                cpoke_in[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same,:]
                cpoke_in[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff,:]
                cpoke_in[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same,:]
                cpoke_in[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_sel if side == 'left' else right_sel
                    prev_side_sel = prev_left_sel if side == 'left' else prev_right_sel

                    cpoke_in[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_prev_bail'] = mat[prev_bail_sel & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    cpoke_in[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    cpoke_in[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    cpoke_in[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                    cpoke_in[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]


                for stim_type in stim_types:
                    prev_stim_sel = prev_trial_stims == stim_type
                    cpoke_in[sess_id][signal_type][region]['prev_stim_'+stim_type] = mat[prev_stim_sel & ~prev_bail_sel,:]
                    cpoke_in[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_hit'] = mat[prev_stim_sel & prev_hit_sel,:]
                    cpoke_in[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_miss'] = mat[prev_stim_sel & prev_miss_sel,:]


                # Tones
                pre = 1
                post = 1

                first_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[0] if utils.is_list(x) else x)
                second_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[1] if utils.is_list(x) else np.nan)
                first_mat, t = fp_utils.build_signal_matrix(signal, ts, first_tone_ts, pre, post)
                second_mat, t = fp_utils.build_signal_matrix(signal, ts, second_tone_ts, pre, post)

                tones['t'] = t
                tones[sess_id][signal_type][region]['first_all'] = first_mat[~bail_sel, :]
                tones[sess_id][signal_type][region]['first_hit_all'] = first_mat[hit_sel & ~bail_sel,:]
                tones[sess_id][signal_type][region]['first_miss_all'] = first_mat[miss_sel & ~bail_sel,:]
                tones[sess_id][signal_type][region]['first_hit_one_tone'] = first_mat[hit_sel & single_tone_sel,:]
                tones[sess_id][signal_type][region]['first_miss_one_tone'] = first_mat[miss_sel & single_tone_sel,:]
                tones[sess_id][signal_type][region]['first_hit_two_tones'] = first_mat[hit_sel & two_tone_sel,:]
                tones[sess_id][signal_type][region]['first_miss_two_tones'] = first_mat[miss_sel & two_tone_sel,:]

                tones[sess_id][signal_type][region]['second_all'] = second_mat[two_tone_sel & ~bail_sel,:]
                tones[sess_id][signal_type][region]['second_hit'] = second_mat[hit_sel & two_tone_sel,:]
                tones[sess_id][signal_type][region]['second_miss'] = second_mat[miss_sel & two_tone_sel,:]

                for tone_type in tone_types:
                    stim_sel_first = tone_infos.apply(lambda x: x[0] == tone_type if utils.is_list(x) else x == tone_type).to_numpy() & ~bail_sel
                    stim_sel_second = tone_infos.apply(lambda x: x[1] == tone_type if (utils.is_list(x) and len(x) > 1) else False).to_numpy() & ~bail_sel

                    tones[sess_id][signal_type][region]['first_'+tone_type+'_all'] = first_mat[stim_sel_first,:]
                    tones[sess_id][signal_type][region]['first_'+tone_type+'_hit_all'] = first_mat[stim_sel_first & hit_sel,:]
                    tones[sess_id][signal_type][region]['first_'+tone_type+'_miss_all'] = first_mat[stim_sel_first & miss_sel,:]
                    tones[sess_id][signal_type][region]['first_'+tone_type+'_hit_one_tone'] = first_mat[stim_sel_first & hit_sel & single_tone_sel,:]
                    tones[sess_id][signal_type][region]['first_'+tone_type+'_miss_one_tone'] = first_mat[stim_sel_first & miss_sel & single_tone_sel,:]
                    tones[sess_id][signal_type][region]['first_'+tone_type+'_hit_two_tones'] = first_mat[stim_sel_first & hit_sel & two_tone_sel,:]
                    tones[sess_id][signal_type][region]['first_'+tone_type+'_miss_two_tones'] = first_mat[stim_sel_first & miss_sel & two_tone_sel,:]

                    tones[sess_id][signal_type][region]['second_'+tone_type+'_all'] = second_mat[stim_sel_second,:]
                    tones[sess_id][signal_type][region]['second_'+tone_type+'_hit'] = second_mat[stim_sel_second & hit_sel,:]
                    tones[sess_id][signal_type][region]['second_'+tone_type+'_miss'] = second_mat[stim_sel_second & miss_sel,:]

                    # TODO: Look at tone type by stimulus type

                # response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, response_cue_ts, pre, post)

                response_cue['t'] = t
                response_cue[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel,:]
                response_cue[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel,:]
                response_cue[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel,:]
                response_cue[sess_id][signal_type][region]['hit'] = mat[hit_sel,:]
                response_cue[sess_id][signal_type][region]['miss'] = mat[miss_sel,:]
                response_cue[sess_id][signal_type][region]['stay'] = mat[prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff,:]

                response_cue[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff,:]
                response_cue[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff,:]

                response_cue[sess_id][signal_type][region]['stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff,:]
                response_cue[sess_id][signal_type][region]['stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff,:]

                response_cue[sess_id][signal_type][region]['stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same,:]
                response_cue[sess_id][signal_type][region]['stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same,:]
                response_cue[sess_id][signal_type][region]['stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff,:]
                response_cue[sess_id][signal_type][region]['switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff,:]
                response_cue[sess_id][signal_type][region]['stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff,:]
                response_cue[sess_id][signal_type][region]['switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff,:]

                response_cue[sess_id][signal_type][region]['stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff,:]
                response_cue[sess_id][signal_type][region]['stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same,:]
                response_cue[sess_id][signal_type][region]['switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff,:]

                response_cue[sess_id][signal_type][region]['rewarded_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & hit_sel,:]
                response_cue[sess_id][signal_type][region]['rewarded_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & hit_sel,:]
                response_cue[sess_id][signal_type][region]['rewarded_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & hit_sel,:]
                response_cue[sess_id][signal_type][region]['rewarded_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & hit_sel,:]
                response_cue[sess_id][signal_type][region]['unrewarded_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & miss_sel,:]
                response_cue[sess_id][signal_type][region]['unrewarded_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & miss_sel,:]
                response_cue[sess_id][signal_type][region]['unrewarded_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & miss_sel,:]
                response_cue[sess_id][signal_type][region]['unrewarded_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & miss_sel,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_sel if side == 'left' else right_sel
                    prev_side_sel = prev_left_sel if side == 'left' else prev_right_sel

                    response_cue[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                    response_cue[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]

                    response_cue[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    response_cue[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    response_cue[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                    response_cue[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]

                    correct_sel = correct_sides == side
                    response_cue[sess_id][signal_type][region][side_type+'_correct'] = mat[correct_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_correct_hit'] = mat[hit_sel & correct_sel,:]
                    response_cue[sess_id][signal_type][region][side_type+'_correct_miss'] = mat[miss_sel & correct_sel,:]

                for stim_type in stim_types:
                    stim_sel = trial_stims == stim_type
                    response_cue[sess_id][signal_type][region]['stim_'+stim_type] = mat[stim_sel,:]
                    response_cue[sess_id][signal_type][region]['stim_'+stim_type+'_prev_hit'] = mat[stim_sel & prev_hit_sel,:]
                    response_cue[sess_id][signal_type][region]['stim_'+stim_type+'_prev_miss'] = mat[stim_sel & prev_miss_sel,:]
                    response_cue[sess_id][signal_type][region]['stim_'+stim_type+'_hit'] = mat[stim_sel & hit_sel,:]
                    response_cue[sess_id][signal_type][region]['stim_'+stim_type+'_miss'] = mat[stim_sel & miss_sel,:]


                # poke out
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_out_ts, pre, post)

                cpoke_out['t'] = t
                cpoke_out[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & ~bail_sel,:]
                cpoke_out[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & ~bail_sel,:]
                cpoke_out[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & ~bail_sel,:]
                cpoke_out[sess_id][signal_type][region]['hit'] = mat[hit_sel,:]
                cpoke_out[sess_id][signal_type][region]['miss'] = mat[miss_sel,:]
                cpoke_out[sess_id][signal_type][region]['bail'] = mat[bail_sel,:]
                cpoke_out[sess_id][signal_type][region]['stay'] = mat[prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff,:]

                cpoke_out[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff,:]

                cpoke_out[sess_id][signal_type][region]['stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff,:]

                cpoke_out[sess_id][signal_type][region]['stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff,:]

                cpoke_out[sess_id][signal_type][region]['stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff,:]
                cpoke_out[sess_id][signal_type][region]['stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same,:]
                cpoke_out[sess_id][signal_type][region]['switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_sel if side == 'left' else right_sel
                    prev_side_sel = (prev_left_sel if side == 'left' else prev_right_sel) & ~bail_sel

                    cpoke_out[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]

                    cpoke_out[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    cpoke_out[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    cpoke_out[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                    cpoke_out[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]

                    correct_sel = (correct_sides == side) & ~bail_sel
                    cpoke_out[sess_id][signal_type][region][side_type+'_correct'] = mat[correct_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_correct_hit'] = mat[hit_sel & correct_sel,:]
                    cpoke_out[sess_id][signal_type][region][side_type+'_correct_miss'] = mat[miss_sel & correct_sel,:]

                for stim_type in stim_types:
                    stim_sel = (trial_stims == stim_type) & ~bail_sel
                    cpoke_out[sess_id][signal_type][region]['stim_'+stim_type] = mat[stim_sel,:]
                    cpoke_out[sess_id][signal_type][region]['stim_'+stim_type+'_prev_hit'] = mat[stim_sel & prev_hit_sel,:]
                    cpoke_out[sess_id][signal_type][region]['stim_'+stim_type+'_prev_miss'] = mat[stim_sel & prev_miss_sel,:]
                    cpoke_out[sess_id][signal_type][region]['stim_'+stim_type+'_hit'] = mat[stim_sel & hit_sel,:]
                    cpoke_out[sess_id][signal_type][region]['stim_'+stim_type+'_miss'] = mat[stim_sel & miss_sel,:]

                # response
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, response_ts, pre, post)

                response['t'] = t
                response[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & ~bail_sel,:]
                response[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & ~bail_sel,:]
                response[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & ~bail_sel,:]
                response[sess_id][signal_type][region]['hit'] = mat[hit_sel,:]
                response[sess_id][signal_type][region]['miss'] = mat[miss_sel,:]
                response[sess_id][signal_type][region]['bail'] = mat[bail_sel,:]
                response[sess_id][signal_type][region]['stay'] = mat[prev_choice_same,:]
                response[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff,:]

                response[sess_id][signal_type][region]['rewarded_stay'] = mat[prev_choice_same & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_switch'] = mat[prev_choice_diff & hit_sel,:]
                response[sess_id][signal_type][region]['unrewarded_stay'] = mat[prev_choice_same & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_switch'] = mat[prev_choice_diff & miss_sel,:]

                response[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same,:]
                response[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff,:]
                response[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same,:]
                response[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff,:]

                response[sess_id][signal_type][region]['rewarded_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & hit_sel,:]
                response[sess_id][signal_type][region]['unrewarded_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & miss_sel,:]

                response[sess_id][signal_type][region]['stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same,:]
                response[sess_id][signal_type][region]['switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff,:]
                response[sess_id][signal_type][region]['stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same,:]
                response[sess_id][signal_type][region]['switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff,:]

                response[sess_id][signal_type][region]['stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same,:]
                response[sess_id][signal_type][region]['switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same,:]
                response[sess_id][signal_type][region]['stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same,:]
                response[sess_id][signal_type][region]['switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same,:]
                response[sess_id][signal_type][region]['stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff,:]
                response[sess_id][signal_type][region]['switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff,:]
                response[sess_id][signal_type][region]['stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff,:]
                response[sess_id][signal_type][region]['switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff,:]

                response[sess_id][signal_type][region]['stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same,:]
                response[sess_id][signal_type][region]['switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff,:]
                response[sess_id][signal_type][region]['stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same,:]
                response[sess_id][signal_type][region]['switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff,:]

                response[sess_id][signal_type][region]['rewarded_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & hit_sel,:]
                response[sess_id][signal_type][region]['unrewarded_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & miss_sel,:]

                response[sess_id][signal_type][region]['rewarded_future_resp'] = mat[next_resp_sel & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_future_bail'] = mat[next_bail_sel & hit_sel,:]
                response[sess_id][signal_type][region]['unrewarded_future_resp'] = mat[next_resp_sel & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_future_bail'] = mat[next_bail_sel & miss_sel,:]

                response[sess_id][signal_type][region]['rewarded_future_stay'] = mat[next_choice_same & hit_sel,:]
                response[sess_id][signal_type][region]['rewarded_future_switch'] = mat[next_choice_diff & hit_sel,:]
                response[sess_id][signal_type][region]['unrewarded_future_stay'] = mat[next_choice_same & miss_sel,:]
                response[sess_id][signal_type][region]['unrewarded_future_switch'] = mat[next_choice_diff & miss_sel,:]

                for side in sides:
                    side_type = 'ipsi' if region_side == side else 'contra'
                    side_sel = left_sel if side == 'left' else right_sel
                    prev_side_sel = (prev_left_sel if side == 'left' else prev_right_sel) & ~bail_sel

                    response[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_rewarded'] = mat[side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded'] = mat[side_sel & miss_sel,:]
                    response[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_rewarded_prev_hit'] = mat[prev_hit_sel & side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_rewarded_prev_miss'] = mat[prev_miss_sel & side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded_prev_hit'] = mat[prev_hit_sel & side_sel & miss_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded_prev_miss'] = mat[prev_miss_sel & side_sel & miss_sel,:]

                    response[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                    response[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]
                    response[sess_id][signal_type][region][side_type+'_stay_rewarded'] = mat[side_sel & prev_choice_same & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_rewarded'] = mat[side_sel & prev_choice_diff & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_unrewarded'] = mat[side_sel & prev_choice_same & miss_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_unrewarded'] = mat[side_sel & prev_choice_diff & miss_sel,:]

                    response[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    response[sess_id][signal_type][region][side_type+'_stay_rewarded_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_rewarded_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_rewarded_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_rewarded_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel & hit_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_unrewarded_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel & miss_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_unrewarded_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel & miss_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_unrewarded_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel & miss_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_unrewarded_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel & miss_sel,:]

                    response[sess_id][signal_type][region][side_type+'_stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff & side_sel,:]

                    response[sess_id][signal_type][region][side_type+'_stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff & side_sel,:]

                    response[sess_id][signal_type][region][side_type+'_rewarded_future_resp'] = mat[next_resp_sel & hit_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_rewarded_future_bail'] = mat[next_bail_sel & hit_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded_future_resp'] = mat[next_resp_sel & miss_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded_future_bail'] = mat[next_bail_sel & miss_sel & side_sel,:]

                    response[sess_id][signal_type][region][side_type+'_rewarded_future_stay'] = mat[next_choice_same & hit_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_rewarded_future_switch'] = mat[next_choice_diff & hit_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded_future_stay'] = mat[next_choice_same & miss_sel & side_sel,:]
                    response[sess_id][signal_type][region][side_type+'_unrewarded_future_switch'] = mat[next_choice_diff & miss_sel & side_sel,:]

                    correct_sel = (correct_sides == side) & ~bail_sel
                    response[sess_id][signal_type][region][side_type+'_correct'] = mat[correct_sel,:]
                    response[sess_id][signal_type][region][side_type+'_correct_hit'] = mat[hit_sel & correct_sel,:]
                    response[sess_id][signal_type][region][side_type+'_correct_miss'] = mat[miss_sel & correct_sel,:]

                for stim_type in stim_types:
                    stim_sel = (trial_stims == stim_type) & ~bail_sel
                    response[sess_id][signal_type][region]['stim_'+stim_type] = mat[stim_sel,:]
                    response[sess_id][signal_type][region]['stim_'+stim_type+'_prev_hit'] = mat[stim_sel & prev_hit_sel,:]
                    response[sess_id][signal_type][region]['stim_'+stim_type+'_prev_miss'] = mat[stim_sel & prev_miss_sel,:]
                    response[sess_id][signal_type][region]['stim_'+stim_type+'_hit'] = mat[stim_sel & hit_sel,:]
                    response[sess_id][signal_type][region]['stim_'+stim_type+'_miss'] = mat[stim_sel & miss_sel,:]


# %% plot alignment results

# title_suffix = ''
# outlier_thresh = 8 # z-score threshold

# for sess_id in sess_ids:
#     for signal_type in signal_types:

#         # get appropriate labels
#         signal_type_title, signal_type_label = fpah.get_signal_type_labels(signal_type)

        # if trial_normalize:
        #     signal_type_title += ' - Trial Normalized'

        # if title_suffix != '':
        #     signal_type_title += ' - ' + title_suffix

        # all_sub_titles = {'hit': 'Hits', 'miss': 'Misses', 'bail': 'Bails',
        #                   'first hit': 'First Tone Hits', 'first miss': 'First Tone Misses',
        #                   'last hit': 'Last Tone Hits', 'last miss': 'Last Tone Misses',
        #                   'same stay': 'Same Tone, Same Choice', 'same switch': 'Same Tone, Different Choice',
        #                   'diff stay': 'Different Tone, Same Choice', 'diff switch': 'Different Tone, Different Choice',
        #                   'stay | same hit': 'Same Response, Previous Hit, Same Tone', 'stay | diff hit': 'Same Response, Previous Hit, Different Tone',
        #                   'stay | same miss': 'Same Response, Previous Miss, Same Tone', 'stay | diff miss': 'Same Response, Previous Miss, Different Tone',
        #                   'switch | same hit': 'Different Response, Previous Hit, Same Tone', 'switch | diff hit': 'Different Response, Previous Hit, Different Tone',
        #                   'switch | same miss': 'Different Response, Previous Miss, Same Tone', 'switch | diff miss': 'Different Response, Previous Miss, Different Tone'}

        # fpah.plot_aligned_signals(one_tone_outcome, 'Single Tone Trials - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(two_tone_outcome, 'Two Tone Trials - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(response_cue, 'Response Cue Aligned - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from response cue (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(cpoke_out, 'Poke Out - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from poke out (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(response, 'Response Aligned - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(cpoke_in_post_outcome, 'Poke In, Future Outcome - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from poke in (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(cpoke_in_prev_outcome, 'Poke In, Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from poke in (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(cport_light_on_prev_outcome, 'Light On, Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from center light on (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # fpah.plot_aligned_signals(delay_first_tone, 'Delay After First Tone - {} (session {})'.format(signal_type_title, sess_id),
        #                      all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)

        # if not main_alignments:

        #     fpah.plot_aligned_signals(miss_by_next_resp, 'Misses Grouped By Next Tone/Response - {} (session {})'.format(signal_type_title, sess_id),
        #                          all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

        #     fpah.plot_aligned_signals(hits_by_next_resp, 'Hits Grouped By Next Tone/Response - {} (session {})'.format(signal_type_title, sess_id),
        #                          all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

        #     fpah.plot_aligned_signals(miss_by_prev_resp, 'Misses Grouped By Previous Tone/Response - {} (session {})'.format(signal_type_title, sess_id),
        #                          all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

        #     fpah.plot_aligned_signals(miss_by_prev_outcome, 'Misses Grouped By Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
        #                          all_sub_titles, 'Time from response (s)', signal_type_label, outlier_thresh=outlier_thresh)

        #     fpah.plot_aligned_signals(first_tone_prev_outcome, 'First Tone By Choice & Previous Outcome - {} (session {})'.format(signal_type_title, sess_id),
        #                          all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh)



# %% Plot Averaged Signals over multiple sessions

signal_type = 'z_dff_iso'
regions = ['DMS', 'PL']
subjects = list(sess_ids.keys())
filter_outliers = True
outlier_thresh = 20
use_se = True
ph = 3.5;
pw = 5;
n_reg = len(regions)
resp_reg_xlims = {'DMS': [-1.5,1.5], 'PL': [-2,10]}

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
tone_mats = stack_mats(tones)
cue_mats = stack_mats(response_cue)
resp_mats = stack_mats(response)


# %% Choice, side, and prior reward groupings for multiple alignment points

choice_groups = ['stay', 'switch']
side_groups = ['contra', 'ipsi']
choice_side_groups = ['contra_stay', 'contra_switch', 'ipsi_stay', 'ipsi_switch']
group_labels = {'stay': 'Stay', 'switch': 'Switch',
                'ipsi': 'Ipsi', 'contra': 'Contra',
                'contra_stay': 'Contra Stay', 'contra_switch': 'Contra Switch',
                'ipsi_stay': 'Ipsi Stay', 'ipsi_switch': 'Ipsi Switch'}

mats = [cport_on_mats, cpoke_in_mats, cue_mats, cpoke_out_mats, resp_mats]
ts = [cport_on['t'], cpoke_in['t'], response_cue['t'], cpoke_out['t'], response['t']]
titles = ['Center Port On', 'Center Poke In', 'Response Cue', 'Center Poke Out', 'Response']
x_labels = ['port on', 'poke in', 'response cue', 'poke out', 'response poke']
n_cols = 3

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Choice Side & Stay/Switch Groupings Aligned to ' + title)

    legend_locs = ['upper right', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Stay/Switch - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in choice_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Choice Side - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in side_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Stay/Switch & Side - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in choice_side_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])

        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# prior & future outcome
prev_outcome_groups = ['prev_hit', 'prev_miss', 'prev_bail']
outcome_groups = ['hit', 'miss'] # , 'bail'
group_labels = {'prev_hit': 'Prev Hit', 'prev_miss': 'Prev Miss', 'prev_bail': 'Prev Bail',
                'hit': 'Hit', 'miss': 'Miss', 'bail': 'Bail'}

mats = [cport_on_mats, cue_mats, cpoke_out_mats, resp_mats]
ts = [cport_on['t'], response_cue['t'], cpoke_out['t'], response['t']]
titles = ['Center Port On', 'Response Cue', 'Center Poke Out', 'Response']
x_labels = ['port on', 'response cue', 'poke out', 'response poke']

n_cols = 2

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Prior Outcome by Stay/Switch & Side Groupings Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Prior Outcome - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in prev_outcome_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Outcome - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in outcome_groups:
            if group not in mat[region]:
                continue
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# prior reward/choice/side
choice_groups = ['stay_prev_hit', 'switch_prev_hit', 'stay_prev_miss', 'switch_prev_miss']
side_groups = ['contra_prev_hit', 'ipsi_prev_hit', 'contra_prev_miss', 'ipsi_prev_miss']
rew_groups = ['contra_stay_prev_hit', 'contra_switch_prev_hit', 'ipsi_stay_prev_hit', 'ipsi_switch_prev_hit']
unrew_groups = ['contra_stay_prev_miss', 'contra_switch_prev_miss', 'ipsi_stay_prev_miss', 'ipsi_switch_prev_miss']
group_labels = {'stay_prev_hit': 'Stay | Hit', 'switch_prev_hit': 'Switch | Hit',
                'stay_prev_miss': 'Stay | Miss', 'switch_prev_miss': 'Switch | Miss',
                'contra_prev_hit': 'Contra | Hit', 'ipsi_prev_hit': 'Ipsi | Hit',
                'contra_prev_miss': 'Contra | Miss', 'ipsi_prev_miss': 'Ipsi | Miss',
                'contra_stay_prev_hit': 'Contra Stay', 'contra_switch_prev_hit': 'Contra Switch',
                'ipsi_stay_prev_hit': 'Ipsi Stay', 'ipsi_switch_prev_hit': 'Ipsi Switch',
                'contra_stay_prev_miss': 'Contra Stay', 'contra_switch_prev_miss': 'Contra Switch',
                'ipsi_stay_prev_miss': 'Ipsi Stay', 'ipsi_switch_prev_miss': 'Ipsi Switch'}

n_cols = 4

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Prior Outcome by Stay/Switch and Side Groupings Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Stay/Switch by Prior Outcome - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in choice_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Choice Side by Prior Outcome - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in side_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])

        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Prior Hit - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,3]
        ax.set_title('{} Prior Miss - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in unrew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# %% prior/current stimuli/outcome

# prior stimuli & outcome
mats = [cport_on_mats, cpoke_in_mats]
ts = [cport_on['t'], cpoke_in['t']]
titles = ['Center Port On', 'Center Poke In']
x_labels = ['port on', 'poke in']
n_cols = 3

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Prior Stimulus and Outcome Groupings Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Prior Stimulus - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for stim_type in stim_types:
            act = mat[region]['prev_stim_'+stim_type]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=stim_type)
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Prior Stimulus & Hit - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for stim_type in stim_types:
            act = mat[region]['prev_stim_'+stim_type+'_prev_hit']
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=stim_type)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Prior Stimulus & Miss - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for stim_type in stim_types:
            act = mat[region]['prev_stim_'+stim_type+'_prev_miss']
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=stim_type)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# current stimuli/prior outcome
mats = [cue_mats, cpoke_out_mats, resp_mats]
ts = [response_cue['t'], cpoke_out['t'], response['t']]
titles = ['Response Cue', 'Center Poke Out', 'Response']
x_labels = ['response cue', 'poke out', 'response poke']
n_cols = 3

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Prior Outcome and Current Stimulus Groupings Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Current Stimulus - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for stim_type in stim_types:
            act = mat[region]['stim_'+stim_type]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=stim_type)
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Current Stimulus Hits - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for stim_type in stim_types:
            act = mat[region]['stim_'+stim_type+'_hit']
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=stim_type)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Current Stimulus Misses - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for stim_type in stim_types:
            act = mat[region]['stim_'+stim_type+'_miss']
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=stim_type)
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# side of correct choice
mats = [cue_mats, cpoke_out_mats, resp_mats]
ts = [response_cue['t'], cpoke_out['t'], response['t']]
titles = ['Response Cue', 'Center Poke Out', 'Response']
x_labels = ['response cue', 'poke out', 'response poke']
n_cols = 2

side_groups = ['contra_correct', 'ipsi_correct']
side_outcome_groups = ['contra_correct_hit', 'ipsi_correct_hit', 'contra_correct_miss', 'ipsi_correct_miss']
group_labels = {'contra_correct': 'Contra', 'ipsi_correct': 'Ipsi',
                'contra_correct_hit': 'Contra Hit', 'ipsi_correct_hit': 'Ipsi Hit',
                'contra_correct_miss': 'Contra Miss', 'ipsi_correct_miss': 'Ipsi Miss'}

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Correct Side by Outcome Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Correct Side - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in side_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Correct Side/Outcome - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in side_outcome_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# %% Same/Different stimuli and stay/switch

# side of correct choice
mats = [cue_mats, resp_mats]
ts = [response_cue['t'], response['t']]
titles = ['Response Cue', 'Response']
x_labels = ['response cue', 'response poke']
n_cols = 3

stim_groups = ['stay_prev_same_correct', 'switch_prev_same_correct', 'stay_prev_diff_correct', 'switch_prev_diff_correct']
stim_rew_groups = ['stay_prev_hit_same_correct', 'switch_prev_miss_same_correct', 'stay_prev_miss_diff_correct', 'switch_prev_hit_diff_correct']
stim_unrew_groups = ['stay_prev_miss_same_correct', 'switch_prev_hit_same_correct', 'stay_prev_hit_diff_correct', 'switch_prev_miss_diff_correct']
group_labels = {'stay_prev_same_correct': 'Stay | Same', 'switch_prev_same_correct': 'Switch | Same',
                'stay_prev_diff_correct': 'Stay | Diff', 'switch_prev_diff_correct': 'Switch | Diff',
                'stay_prev_hit_same_correct': 'Stay | Same', 'switch_prev_miss_same_correct': 'Switch | Same',
                'stay_prev_miss_diff_correct': 'Stay | Diff', 'switch_prev_hit_diff_correct': 'Switch | Diff',
                'stay_prev_miss_same_correct': 'Stay | Same', 'switch_prev_hit_same_correct': 'Switch | Same',
                'stay_prev_hit_diff_correct': 'Stay | Diff', 'switch_prev_miss_diff_correct': 'Switch | Diff'}

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Correct Side Repeat by Outcome Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Correct Side Repeat and Choice - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in stim_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Rewarded Correct Side Repeat and Choice - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in stim_rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Unrewarded Correct Side Repeat and Choice - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in stim_unrew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# trial repeat
stim_groups = ['stay_prev_same_trial', 'switch_prev_same_trial', 'stay_prev_diff_trial', 'switch_prev_diff_trial']
stim_rew_groups = ['rewarded_stay_prev_same_trial', 'rewarded_switch_prev_same_trial', 'rewarded_stay_prev_diff_trial', 'rewarded_switch_prev_diff_trial']
stim_unrew_groups = ['unrewarded_stay_prev_same_trial', 'unrewarded_switch_prev_same_trial', 'unrewarded_stay_prev_diff_trial', 'unrewarded_switch_prev_diff_trial']
group_labels = {'stay_prev_same_trial': 'Stay | Same', 'switch_prev_same_trial': 'Switch | Same',
                'stay_prev_diff_trial': 'Stay | Diff', 'switch_prev_diff_trial': 'Switch | Diff',
                'rewarded_stay_prev_same_trial': 'Stay | Same', 'rewarded_switch_prev_same_trial': 'Switch | Same',
                'rewarded_stay_prev_diff_trial': 'Stay | Diff', 'rewarded_switch_prev_diff_trial': 'Switch | Diff',
                'unrewarded_stay_prev_same_trial': 'Stay | Same', 'unrewarded_switch_prev_same_trial': 'Switch | Same',
                'unrewarded_stay_prev_diff_trial': 'Stay | Diff', 'unrewarded_switch_prev_diff_trial': 'Switch | Diff'}

for mat, t, title, x_label in zip(mats, ts, titles, x_labels):

    fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
    fig.suptitle('Trial Repeat by Outcome Aligned to ' + title)

    legend_locs = ['upper left', 'upper right']
    for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

        if title == 'Response':
            xlims = resp_reg_xlims[region]
        else:
            xlims = [-1.5,1.5]

        ax = axs[i,0]
        ax.set_title('{} Trial Repeat and Choice - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in stim_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.set_ylabel('Z-scored ΔF/F')
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,1]
        ax.set_title('{} Rewarded Trial Repeat and Choice - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in stim_rew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)

        ax = axs[i,2]
        ax.set_title('{} Unrewarded Trial Repeat and Choice - {}'.format(region, title))
        # if title == 'Response':
        #     plot_utils.plot_dashlines(0.5, ax=ax)
        for group in stim_unrew_groups:
            act = mat[region][group]
            plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
        ax.yaxis.set_tick_params(which='both', labelleft=True)
        ax.set_xlabel('Time from {} (s)'.format(x_label))
        ax.set_xlim(xlims)
        ax.legend(loc=legend_loc)


# side of correct choice and choice side

n_cols = 2

contra_stim_groups = ['contra_stay_prev_same_correct', 'contra_switch_prev_same_correct', 'contra_stay_prev_diff_correct', 'contra_switch_prev_diff_correct']
ipsi_stim_groups = ['ipsi_stay_prev_same_correct', 'ipsi_switch_prev_same_correct', 'ipsi_stay_prev_diff_correct', 'ipsi_switch_prev_diff_correct']
group_labels = {'contra_stay_prev_same_correct': 'Stay | Same', 'contra_switch_prev_same_correct': 'Switch | Same',
                'contra_stay_prev_diff_correct': 'Stay | Diff', 'contra_switch_prev_diff_correct': 'Switch | Diff',
                'ipsi_stay_prev_same_correct': 'Stay | Same', 'ipsi_switch_prev_same_correct': 'Switch | Same',
                'ipsi_stay_prev_diff_correct': 'Stay | Diff', 'ipsi_switch_prev_diff_correct': 'Switch | Diff'}

mat = resp_mats
t = response['t']
x_label = 'response poke'

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Correct Side Repeat by Choice Side Aligned to Response')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    if title == 'Response':
        xlims = resp_reg_xlims[region]
    else:
        xlims = [-1.5,1.5]

    ax = axs[i,0]
    ax.set_title('{} Contra Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in contra_stim_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Ipsi Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in ipsi_stim_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


n_cols = 4

contra_rew_groups = ['contra_stay_prev_hit_same_correct', 'contra_switch_prev_miss_same_correct', 'contra_stay_prev_miss_diff_correct', 'contra_switch_prev_hit_diff_correct']
contra_unrew_groups = ['contra_stay_prev_miss_same_correct', 'contra_switch_prev_hit_same_correct', 'contra_stay_prev_hit_diff_correct', 'contra_switch_prev_miss_diff_correct']
ipsi_rew_groups = ['ipsi_stay_prev_hit_same_correct', 'ipsi_switch_prev_miss_same_correct', 'ipsi_stay_prev_miss_diff_correct', 'ipsi_switch_prev_hit_diff_correct']
ipsi_unrew_groups = ['ipsi_stay_prev_miss_same_correct', 'ipsi_switch_prev_hit_same_correct', 'ipsi_stay_prev_hit_diff_correct', 'ipsi_switch_prev_miss_diff_correct']
group_labels = {'contra_stay_prev_hit_same_correct': 'Stay | Same', 'contra_switch_prev_miss_same_correct': 'Switch | Same',
                'contra_stay_prev_miss_diff_correct': 'Stay | Diff', 'contra_switch_prev_hit_diff_correct': 'Switch | Diff',
                'contra_stay_prev_miss_same_correct': 'Stay | Same', 'contra_switch_prev_hit_same_correct': 'Switch | Same',
                'contra_stay_prev_hit_diff_correct': 'Stay | Diff', 'contra_switch_prev_miss_diff_correct': 'Switch | Diff',
                'ipsi_stay_prev_hit_same_correct': 'Stay | Same', 'ipsi_switch_prev_miss_same_correct': 'Switch | Same',
                'ipsi_stay_prev_miss_diff_correct': 'Stay | Diff', 'ipsi_switch_prev_hit_diff_correct': 'Switch | Diff',
                'ipsi_stay_prev_miss_same_correct': 'Stay | Same', 'ipsi_switch_prev_hit_same_correct': 'Switch | Same',
                'ipsi_stay_prev_hit_diff_correct': 'Stay | Diff', 'ipsi_switch_prev_miss_diff_correct': 'Switch | Diff'}

mat = resp_mats
t = response['t']
x_label = 'response poke'

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Correct Side Repeat by Choice Side Aligned to Response')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    if title == 'Response':
        xlims = resp_reg_xlims[region]
    else:
        xlims = [-1.5,1.5]

    ax = axs[i,0]
    ax.set_title('{} Rewarded Contra Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in contra_rew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Rewarded Ipsi Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in ipsi_rew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title('{} Unrewarded Contra Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in contra_unrew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title('{} Unrewarded Ipsi Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in ipsi_unrew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

# %% Various Response Outcome groupings

# stay/switch by prior & current outcome
n_cols = 3

choice_groups = ['rewarded_stay', 'rewarded_switch', 'unrewarded_stay', 'unrewarded_switch']
rew_groups = ['rewarded_stay_prev_hit', 'rewarded_switch_prev_hit', 'rewarded_stay_prev_miss', 'rewarded_switch_prev_miss']
unrew_groups = ['unrewarded_stay_prev_hit', 'unrewarded_switch_prev_hit', 'unrewarded_stay_prev_miss', 'unrewarded_switch_prev_miss']
group_labels = {'rewarded_stay': 'Rewarded Stay', 'rewarded_switch': 'Rewarded Switch',
                'unrewarded_stay': 'Unrewarded Stay', 'unrewarded_switch': 'Unrewarded Switch',
                'rewarded_stay_prev_hit': 'Stay | Hit', 'rewarded_switch_prev_hit': 'Switch | Hit',
                'rewarded_stay_prev_miss': 'Stay | Miss', 'rewarded_switch_prev_miss': 'Switch | Miss',
                'unrewarded_stay_prev_hit': 'Stay | Hit', 'unrewarded_switch_prev_hit': 'Switch | Hit',
                'unrewarded_stay_prev_miss': 'Stay | Miss', 'unrewarded_switch_prev_miss': 'Switch | Miss'}

mat = resp_mats
t = response['t']
x_label = 'response poke'

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Stay/Switch By Current and Prior Outcome Aligned to Response')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    if title == 'Response':
        xlims = resp_reg_xlims[region]
    else:
        xlims = [-1.5,1.5]

    ax = axs[i,0]
    ax.set_title('{} Stay/Switch by Outcome - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in choice_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Rewarded Stay/Switch by Prior Outcome - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title('{} Unrewarded Stay/Switch by Prior Outcome - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

n_cols = 3

outcome_groups = ['contra_rewarded', 'ipsi_rewarded', 'contra_unrewarded', 'ipsi_unrewarded']
rew_groups = ['contra_stay_rewarded', 'ipsi_stay_rewarded', 'contra_switch_rewarded', 'ipsi_switch_rewarded']
unrew_groups = ['contra_stay_unrewarded', 'ipsi_stay_unrewarded', 'contra_switch_unrewarded', 'ipsi_switch_unrewarded']
group_labels = {'contra_rewarded': 'Contra Hit', 'ipsi_rewarded': 'Ipsi Hit',
                'contra_unrewarded': 'Contra Miss', 'ipsi_unrewarded': 'Ipsi Miss',
                'contra_stay_rewarded': 'Contra Stay', 'ipsi_stay_rewarded': 'Ipsi Stay',
                'contra_switch_rewarded': 'Contra Switch', 'ipsi_switch_rewarded': 'Ipsi Switch',
                'contra_stay_unrewarded': 'Contra Stay', 'ipsi_stay_unrewarded': 'Ipsi Stay',
                'contra_switch_unrewarded': 'Contra Switch', 'ipsi_switch_unrewarded': 'Ipsi Switch'}

mat = resp_mats
t = response['t']
x_label = 'response poke'

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Choice Side & Stay/Switch By Outcome Aligned to Response')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    if title == 'Response':
        xlims = resp_reg_xlims[region]
    else:
        xlims = [-1.5,1.5]

    ax = axs[i,0]
    ax.set_title('{} Side/Outcome - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in outcome_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Side & Stay/Switch Hits - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title('{} Side & Stay/Switch Misses - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


n_cols = 2

contra_groups = ['contra_rewarded_prev_hit', 'contra_rewarded_prev_miss', 'contra_unrewarded_prev_hit', 'contra_unrewarded_prev_miss']
ipsi_groups = ['ipsi_rewarded_prev_hit', 'ipsi_rewarded_prev_miss', 'ipsi_unrewarded_prev_hit', 'ipsi_unrewarded_prev_miss']
group_labels = {'contra_rewarded_prev_hit': 'Hit | Hit', 'contra_rewarded_prev_miss': 'Hit | Miss',
                'contra_unrewarded_prev_hit': 'Miss | Hit', 'contra_unrewarded_prev_miss': 'Miss | Miss',
                'ipsi_rewarded_prev_hit': 'Hit | Hit', 'ipsi_rewarded_prev_miss': 'Hit | Miss',
                'ipsi_unrewarded_prev_hit': 'Miss | Hit', 'ipsi_unrewarded_prev_miss': 'Miss | Miss'}

mat = resp_mats
t = response['t']
x_label = 'response poke'

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Choice Side By Current & Prior Outcome Aligned to Response')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    if title == 'Response':
        xlims = resp_reg_xlims[region]
    else:
        xlims = [-1.5,1.5]

    ax = axs[i,0]
    ax.set_title('{} Contra Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in contra_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Ipsi Choice - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in ipsi_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


n_cols = 4

rew_rew_groups = ['contra_stay_rewarded_prev_hit', 'ipsi_stay_rewarded_prev_hit', 'contra_switch_rewarded_prev_hit', 'ipsi_switch_rewarded_prev_hit']
rew_unrew_groups = ['contra_stay_rewarded_prev_miss', 'ipsi_stay_rewarded_prev_miss', 'contra_switch_rewarded_prev_miss', 'ipsi_switch_rewarded_prev_miss']
unrew_rew_groups = ['contra_stay_unrewarded_prev_hit', 'ipsi_stay_unrewarded_prev_hit', 'contra_switch_unrewarded_prev_hit', 'ipsi_switch_unrewarded_prev_hit']
unrew_unrew_groups = ['contra_stay_unrewarded_prev_miss', 'ipsi_stay_unrewarded_prev_miss', 'contra_switch_unrewarded_prev_miss', 'ipsi_switch_unrewarded_prev_miss']
group_labels = {'contra_stay_rewarded_prev_hit': 'Contra Stay', 'ipsi_stay_rewarded_prev_hit': 'Ipsi Stay',
                'contra_switch_rewarded_prev_hit': 'Contra Switch', 'ipsi_switch_rewarded_prev_hit': 'Ipsi Switch',
                'contra_stay_rewarded_prev_miss': 'Contra Stay', 'ipsi_stay_rewarded_prev_miss': 'Ipsi Stay',
                'contra_switch_rewarded_prev_miss': 'Contra Switch', 'ipsi_switch_rewarded_prev_miss': 'Ipsi Switch',
                'contra_stay_unrewarded_prev_hit': 'Contra Stay', 'ipsi_stay_unrewarded_prev_hit': 'Ipsi Stay',
                'contra_switch_unrewarded_prev_hit': 'Contra Switch', 'ipsi_switch_unrewarded_prev_hit': 'Ipsi Switch',
                'contra_stay_unrewarded_prev_miss': 'Contra Stay', 'ipsi_stay_unrewarded_prev_miss': 'Ipsi Stay',
                'contra_switch_unrewarded_prev_miss': 'Contra Switch', 'ipsi_switch_unrewarded_prev_miss': 'Ipsi Switch'}

mat = resp_mats
t = response['t']
x_label = 'response poke'

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Choice Side & Stay/Switch By Current & Prior Outcome Aligned to Response')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    if title == 'Response':
        xlims = resp_reg_xlims[region]
    else:
        xlims = [-1.5,1.5]

    ax = axs[i,0]
    ax.set_title('{} Hit | Hit - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_rew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Hit | Miss - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in rew_unrew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title('{} Miss | Hit - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_rew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,3]
    ax.set_title('{} Miss | Miss - {}'.format(region, title))
    # if title == 'Response':
    #     plot_utils.plot_dashlines(0.5, ax=ax)
    for group in unrew_unrew_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# %% Tone Alignments

# all tone types
n_cols = 2

all_groups = ['first_all', 'second_all']
outcome_groups = ['first_hit_all', 'first_miss_all', 'second_hit', 'second_miss']
group_labels = {'first_all': 'First Tone', 'second_all': 'Second Tone',
                'first_hit_all': 'First Hit', 'first_miss_all': 'First Miss',
                'second_hit': 'Second Hit', 'second_miss': 'Second Miss'}

mat = tone_mats
t = tones['t']
x_label = 'tone start'
xlims = [-1,1]
#ylims = [-0.5,0.3]

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Tone Position Aligned to Tone Start')

legend_locs = ['upper left', 'upper right']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    ax = axs[i,0]
    ax.set_title('{} Tone Start'.format(region))
    plot_utils.plot_dashlines(0.25, ax=ax)
    for group in all_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    #ax.set_ylim(ylims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Tone Start By Outcome'.format(region))
    plot_utils.plot_dashlines(0.25, ax=ax)
    for group in outcome_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    #ax.set_ylim(ylims)
    ax.legend(loc=legend_loc)


# by tone type
n_cols = 3

tone_type_labels = {'high': 'High', 'low': 'Low'}
all_groups = ['first_{}_all', 'second_{}_all']
hit_groups = ['first_{}_hit_all', 'second_{}_hit']
miss_groups = ['first_{}_miss_all', 'second_{}_miss']
group_labels = {'first_{}_all': 'First {}', 'second_{}_all': 'Second {}',
                'first_{}_hit_all': 'First {} Hit', 'first_{}_miss_all': 'First {} Miss',
                'second_{}_hit': 'Second {} Hit', 'second_{}_miss': 'Second {} Miss'}

all_groups = [g.format(t) for g in all_groups for t in tone_types]
hit_groups = [g.format(t) for g in hit_groups for t in tone_types]
miss_groups = [g.format(t) for g in miss_groups for t in tone_types]
group_labels = {k.format(t): v.format(tone_type_labels[t]) for k, v in group_labels.items() for t in tone_types}

mat = tone_mats
t = tones['t']
title = 'Tone Start'
x_label = 'tone start'
xlims = [-1,1]

fig, axs = plt.subplots(n_reg, n_cols, figsize=(pw*n_cols, ph*n_reg), layout='constrained', sharey='row')
fig.suptitle('Tone Position and Tone Type Aligned to Tone Start')

legend_locs = ['upper left', 'upper left']
for i, (region, legend_loc) in enumerate(zip(regions, legend_locs)):

    ax = axs[i,0]
    ax.set_title('{} All Trials - {}'.format(region, title))
    plot_utils.plot_dashlines(0.25, ax=ax)
    for group in all_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.set_ylabel('Z-scored ΔF/F')
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,1]
    ax.set_title('{} Hits - {}'.format(region, title))
    plot_utils.plot_dashlines(0.25, ax=ax)
    for group in hit_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)

    ax = axs[i,2]
    ax.set_title('{} Misses - {}'.format(region, title))
    plot_utils.plot_dashlines(0.25, ax=ax)
    for group in miss_groups:
        act = mat[region][group]
        plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label=group_labels[group])
    ax.yaxis.set_tick_params(which='both', labelleft=True)
    ax.set_xlabel('Time from {} (s)'.format(x_label))
    ax.set_xlim(xlims)
    ax.legend(loc=legend_loc)


# %% Look at some behavioral metrics

sess_metrics = {}

for sess_id in sess_ids:
    sess_metrics[sess_id] = {}

    trial_data = sess_data[sess_data['sessid'] == sess_id]
    hit_sel = trial_data['hit'] == True
    miss_sel = trial_data['hit'] == False
    bail_sel = trial_data['bail'] == True

    prev_hit_sel = np.insert(hit_sel[:-1].to_numpy(), 0, False)
    prev_miss_sel = np.insert(miss_sel[:-1].to_numpy(), 0, False)
    prev_bail_sel = np.insert(bail_sel[:-1].to_numpy(), 0, False)

    prev_choice_same = np.insert(trial_data['choice'][:-1].to_numpy() == trial_data['choice'][1:].to_numpy(),
                                 0, False)
    prev_tone_same = np.insert(trial_data['correct_port'][:-1].to_numpy() == trial_data['correct_port'][1:].to_numpy(),
                                 0, False)

    # probability of outcome given previous outcome
    sess_metrics[sess_id]['p(hit|prev hit)'] = np.sum(hit_sel & prev_hit_sel)/np.sum(prev_hit_sel)
    sess_metrics[sess_id]['p(hit|prev miss)'] = np.sum(hit_sel & prev_miss_sel)/np.sum(prev_miss_sel)
    sess_metrics[sess_id]['p(hit|prev bail)'] = np.sum(hit_sel & prev_bail_sel)/np.sum(prev_bail_sel)

    sess_metrics[sess_id]['p(miss|prev hit)'] = np.sum(miss_sel & prev_hit_sel)/np.sum(prev_hit_sel)
    sess_metrics[sess_id]['p(miss|prev miss)'] = np.sum(miss_sel & prev_miss_sel)/np.sum(prev_miss_sel)
    sess_metrics[sess_id]['p(miss|prev bail)'] = np.sum(miss_sel & prev_bail_sel)/np.sum(prev_bail_sel)

    sess_metrics[sess_id]['p(bail|prev hit)'] = np.sum(bail_sel & prev_hit_sel)/np.sum(prev_hit_sel)
    sess_metrics[sess_id]['p(bail|prev miss)'] = np.sum(bail_sel & prev_miss_sel)/np.sum(prev_miss_sel)
    sess_metrics[sess_id]['p(bail|prev bail)'] = np.sum(bail_sel & prev_bail_sel)/np.sum(prev_bail_sel)

    # stay and switch require animals to make responses on consecutive trials, so they cant bail
    sess_metrics[sess_id]['p(stay|prev hit)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(stay|prev miss)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

    sess_metrics[sess_id]['p(switch|prev hit)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(switch|prev miss)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

    sess_metrics[sess_id]['p(stay|prev hit & same tone)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel & prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(stay|prev hit & diff tone)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & ~prev_tone_same)
    sess_metrics[sess_id]['p(stay|prev miss & same tone)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel & prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(stay|prev miss & diff tone)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & ~prev_tone_same)

    sess_metrics[sess_id]['p(switch|prev hit & same tone)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel & prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(switch|prev hit & diff tone)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & ~prev_tone_same)
    sess_metrics[sess_id]['p(switch|prev miss & same tone)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel & prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & prev_tone_same)
    sess_metrics[sess_id]['p(switch|prev miss & diff tone)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & ~prev_tone_same)

    # probability of hit/miss given previous outcome when responding multiple times in a row
    sess_metrics[sess_id]['p(hit|prev hit & no bail)'] = np.sum(hit_sel & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(hit|prev miss & no bail)'] = np.sum(hit_sel & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

    sess_metrics[sess_id]['p(miss|prev hit & no bail)'] = np.sum(miss_sel & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
    sess_metrics[sess_id]['p(miss|prev miss & no bail)'] = np.sum(miss_sel & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)
