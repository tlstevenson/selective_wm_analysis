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
from fp_analysis_helpers import Alignment as Align
import numpy as np
import matplotlib.pyplot as plt
import copy

# %% Load behavior data

# used for saving plots
behavior_name = 'Single Tone WM'
sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=7)

# behavior_name = 'SelWM - Two Tones'
# sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=8)

# behavior_name = 'SelWM - Grow Nosepoke'
# sess_ids = db_access.get_fp_sess_ids(protocol='ToneCatDelayResp2', stage_num=7)

# behavior_name = 'SelWM - Grow Delay'
# sess_ids = db_access.get_fp_sess_ids(protocol='ToneCatDelayResp2', stage_num=9)

# behavior_name = 'SelWM - Two Tones'
# sess_ids = db_access.get_fp_sess_ids(protocol='ToneCatDelayResp2', stage_num=10)

# optionally limit sessions based on subject ids
# subj_ids = [179]
# sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

loc_db = db.LocalDB_ToneCatDelayResp()
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids)) # reload=True

sess_data['incongruent'] = sess_data['tone_info'].apply(lambda x: x[0] != x[-1] if utils.is_list(x) and len(x) == 2 else False)
sess_data['tone_info_str'] = sess_data['tone_info'].apply(lambda x: ', '.join(x) if utils.is_list(x) else x)
sess_data['prev_tone_info_str'] = sess_data['prev_choice_tone_info'].apply(lambda x: ', '.join(x) if utils.is_list(x) else x)
sess_data['cpoke_out_latency'] = sess_data['cpoke_out_time'] - sess_data['response_cue_time']

if 'irrelevant_tone_db_offset' in sess_data.columns:
    sess_data.rename(columns={'irrelevant_tone_db_offset': 'tone_db_offsets'}, inplace=True)

# calculate stimulus duration bins
# TODO: see if pd.cut makes more sense
bin_size = 1

dur_bin_max = np.ceil(np.max(sess_data['stim_dur'])/bin_size)
dur_bin_min = np.floor(np.min(sess_data['stim_dur'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

sess_data['stim_dur_bin'] = sess_data['stim_dur'].apply(
    lambda x: dur_bin_labels[np.where(x >= dur_bins)[0][-1]])
# make sure they are always sorted appropriately using categories
sess_data['stim_dur_bin'] = pd.Categorical(sess_data['stim_dur_bin'], categories=dur_bin_labels)

# calculate response delay length bins
# first get end times of last tone
if 'rel_tone_end_times' in sess_data.columns:
    last_tone_ends = sess_data['rel_tone_end_times'].apply(lambda x: x[-1] if utils.is_list(x) else x )
else:
    last_tone_ends = sess_data['rel_tone_start_times'].apply(lambda x: x[-1] if utils.is_list(x) else x ) + 0.3

sess_data['resp_delay'] = sess_data['stim_dur'] - last_tone_ends

bin_size = 1

dur_bin_max = np.ceil(np.max(sess_data['resp_delay'])/bin_size)
dur_bin_min = np.floor(np.min(sess_data['resp_delay'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

sess_data['resp_delay_bin'] = sess_data['resp_delay'].apply(
    lambda x: dur_bin_labels[np.where(x >= dur_bins)[0][-1]])
# make sure they are always sorted appropriately using categories
sess_data['resp_delay_bin'] = pd.Categorical(sess_data['resp_delay_bin'], categories=dur_bin_labels)

if 'task_variant' not in sess_data.columns:
    sess_data['task_variant'] = 'none'


# %% Get and process photometry data

# get fiber photometry data
reload = False
fp_data, implant_info = fpah.load_fp_data(loc_db, sess_ids, reload=reload, tilt_t=False)

# %% Observe the full signals

filter_outliers = True
save_plots = False
show_plots = False

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        fig = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
                                    title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id),
                                    filter_outliers=filter_outliers)

        if save_plots:
            fpah.save_fig(fig, fpah.get_figure_save_path(behavior_name, subj_id, 'sess_{}'.format(sess_id)))

        if not show_plots:
            plt.close(fig)


# %% Observe any sub-signals
tmp_sess_id = {180: [101447]}
tmp_fp_data, tmp_implant_info = fpah.load_fp_data(loc_db, tmp_sess_id)
sub_signal = [950, 1050] # [0, np.inf] #
filter_outliers = True

subj_id = list(tmp_sess_id.keys())[0]
sess_id = tmp_sess_id[subj_id][0]
#sess_fp = fp_data[subj_id][sess_id]
sess_fp = tmp_fp_data[subj_id][sess_id]
# _ = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
#                             title='Sub Signal - Subject {}, Session {}'.format(subj_id, sess_id),
#                             filter_outliers=filter_outliers,
#                             t_min=sub_signal[0], t_max=sub_signal[1], dec=1)

fpah.view_signal(sess_fp['processed_signals'], sess_fp['time'], 'dff_iso', title=None, dec=10,
            t_min=sub_signal[0], t_max=sub_signal[1], figsize=(9,6), ylabel='% ΔF/F')

# %% Construct aligned signal matrices grouped by various factors

signal_types = ['dff_iso'] # 'baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}
cport_on = copy.deepcopy(data_dict)
early_cpoke_in = copy.deepcopy(data_dict)
cpoke_in = copy.deepcopy(data_dict)
tones = copy.deepcopy(data_dict)
cue = copy.deepcopy(data_dict)
early_cpoke_out = copy.deepcopy(data_dict)
cpoke_out = copy.deepcopy(data_dict)
resp = copy.deepcopy(data_dict)
cue_poke_out_resp = copy.deepcopy(data_dict)
poke_out_cue_resp = copy.deepcopy(data_dict)

stim_types = np.array(sorted(sess_data['tone_info_str'].unique().tolist(), key=lambda x: (len(x), x)))
#tone_types = np.unique(sess_data['response_tone'])
tone_types = np.unique(sess_data['relevant_tone_info'])
stim_durs = np.unique(sess_data['stim_dur_bin'])
resp_delays = np.unique(sess_data['resp_delay_bin'])
variants = np.unique(sess_data['task_variant'])

# get tone side mapping
# tone_port = sess_data[['subjid','correct_port', 'response_tone']].drop_duplicates()
# tone_port = {i: tone_port[tone_port['subjid'] == i].set_index('response_tone')['correct_port'].to_dict() for i in sess_ids.keys()}

tone_port = sess_data[['subjid','correct_port', 'relevant_tone_info']].drop_duplicates()
tone_port = {i: tone_port[tone_port['subjid'] == i].set_index('relevant_tone_info')['correct_port'].to_dict() for i in sess_ids.keys()}

# declare settings for normalized cue to response intervals
norm_cue_resp_bins = 200
norm_cue_poke_out_pct = 0.2 # % of bins for cue to poke out or poke out to cue, depending on which comes first

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

        one_tone_sel = trial_data['n_tones'] == 1
        two_tone_sel = trial_data['n_tones'] == 2
        no_tone_offset = trial_data['tone_db_offsets'].apply(lambda x: all(np.array(utils.flatten([x])) == 0)).to_numpy()

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

        # ignore cport on trials where they were poked before cport turned on
        cport_on_sel = trial_data['cpoke_in_latency'] > 0.1
        early_cpoke_in_sel = trial_data['cpoke_in_latency'] < 0
        norm_cpoke_in_sel = trial_data['cpoke_in_latency'] > 0
        early_cpoke_out_sel = trial_data['cpoke_out_latency'] < 0
        norm_cpoke_out_sel = trial_data['cpoke_out_latency'] > 0

        tone_infos = trial_data['tone_info']
        trial_stims = trial_data['tone_info_str']
        prev_trial_stims = trial_data['prev_tone_info_str']
        correct_sides = trial_data['correct_port']
        incongruent = trial_data['incongruent']
        variant = trial_data['task_variant']

        trial_durs = trial_data['stim_dur_bin']
        trial_delays = trial_data['resp_delay_bin']

        # create the alignment points
        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1][trial_start_sel]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']

        # abs_pre_poke_windows = cpoke_in_ts.to_numpy()[:, None] + pre_poke_norm_window[None, :]
        # cpoke_in_windows = abs_pre_poke_windows - cpoke_in_ts.to_numpy()[:, None]
        # cpoke_out_windows = abs_pre_poke_windows - cpoke_out_ts.to_numpy()[:, None]
        # response_windows = abs_pre_poke_windows - response_ts.to_numpy()[:, None]
        # cue_windows = abs_pre_poke_windows - cue_ts.to_numpy()[:, None]
        # cport_on_windows = abs_pre_poke_windows - cport_on_ts[:, None]

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # Center port on
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cport_on_ts, pre, post)
                align_dict = cport_on
                sel = cport_on_sel

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & sel,:]
                align_dict[sess_id][signal_type][region]['hit'] = mat[hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['miss'] = mat[miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['bail'] = mat[bail_sel & sel,:]
                align_dict[sess_id][signal_type][region]['response'] = mat[~bail_sel & sel,:]
                align_dict[sess_id][signal_type][region]['stay'] = mat[prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & sel,:]

                for side in sides:
                    side_type = fpah.get_implant_rel_side(side, region_side)
                    side_sel = (left_sel if side == 'left' else right_sel) & sel
                    prev_side_sel = (prev_left_sel if side == 'left' else prev_right_sel) & sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[prev_choice_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_bail'] = mat[prev_bail_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]

                for stim_type in stim_types:
                    prev_stim_sel = (prev_trial_stims == stim_type) & sel
                    align_dict[sess_id][signal_type][region]['prev_stim_'+stim_type] = mat[prev_stim_sel & ~prev_bail_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_hit'] = mat[prev_stim_sel & prev_hit_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_miss'] = mat[prev_stim_sel & prev_miss_sel,:]

                # Center poke in
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_in_ts, pre, post)
                align_dicts = [early_cpoke_in, cpoke_in]
                sels = [early_cpoke_in_sel, norm_cpoke_in_sel]

                for align_dict, sel in zip(align_dicts, sels):

                    align_dict['t'] = t
                    align_dict[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['hit'] = mat[hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['miss'] = mat[miss_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['bail'] = mat[bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['response'] = mat[~bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['stay'] = mat[prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel & sel if side == 'left' else right_sel & sel
                        prev_side_sel = prev_left_sel & sel if side == 'left' else prev_right_sel & sel

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_bail'] = mat[prev_bail_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                        align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]

                    for stim_type in stim_types:
                        prev_stim_sel = (prev_trial_stims == stim_type) & sel
                        align_dict[sess_id][signal_type][region]['prev_stim_'+stim_type] = mat[prev_stim_sel & ~prev_bail_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_hit'] = mat[prev_stim_sel & prev_hit_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_stim_'+stim_type+'_prev_miss'] = mat[prev_stim_sel & prev_miss_sel,:]


                # Tones
                pre = 2
                post = 2

                first_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[0] if utils.is_list(x) else x)
                second_tone_ts = trial_start_ts + trial_data['abs_tone_start_times'].apply(lambda x: x[1] if utils.is_list(x) else np.nan)
                first_mat, t = fp_utils.build_signal_matrix(signal, ts, first_tone_ts, pre, post)
                second_mat, t = fp_utils.build_signal_matrix(signal, ts, second_tone_ts, pre, post)
                align_dict = tones

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['first'] = first_mat[~bail_sel, :]
                align_dict[sess_id][signal_type][region]['first_one_tone'] = first_mat[~bail_sel & one_tone_sel, :]
                align_dict[sess_id][signal_type][region]['first_two_tone'] = first_mat[~bail_sel & two_tone_sel, :]
                align_dict[sess_id][signal_type][region]['first_hit'] = first_mat[hit_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['first_miss'] = first_mat[miss_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['first_hit_one_tone'] = first_mat[hit_sel & one_tone_sel,:]
                align_dict[sess_id][signal_type][region]['first_miss_one_tone'] = first_mat[miss_sel & one_tone_sel,:]
                align_dict[sess_id][signal_type][region]['first_hit_two_tone'] = first_mat[hit_sel & two_tone_sel,:]
                align_dict[sess_id][signal_type][region]['first_miss_two_tone'] = first_mat[miss_sel & two_tone_sel,:]

                align_dict[sess_id][signal_type][region]['second'] = second_mat[two_tone_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['second_hit'] = second_mat[hit_sel & two_tone_sel,:]
                align_dict[sess_id][signal_type][region]['second_miss'] = second_mat[miss_sel & two_tone_sel,:]

                align_dict[sess_id][signal_type][region]['second_cong'] = second_mat[two_tone_sel & ~bail_sel & ~incongruent,:]
                align_dict[sess_id][signal_type][region]['second_incong'] = second_mat[two_tone_sel & ~bail_sel & incongruent,:]
                align_dict[sess_id][signal_type][region]['second_hit_cong'] = second_mat[hit_sel & two_tone_sel & ~incongruent,:]
                align_dict[sess_id][signal_type][region]['second_miss_cong'] = second_mat[miss_sel & two_tone_sel & ~incongruent,:]
                align_dict[sess_id][signal_type][region]['second_hit_incong'] = second_mat[hit_sel & two_tone_sel & incongruent,:]
                align_dict[sess_id][signal_type][region]['second_miss_incong'] = second_mat[miss_sel & two_tone_sel & incongruent,:]

                # TODO: Look at tones heard before bails

                for v in variants:
                    v_sel = variant == v

                    align_dict[sess_id][signal_type][region]['first_var_'+v] = first_mat[~bail_sel & v_sel, :]
                    align_dict[sess_id][signal_type][region]['first_one_tone_var_'+v] = first_mat[~bail_sel & one_tone_sel & v_sel, :]
                    align_dict[sess_id][signal_type][region]['first_two_tone_var_'+v] = first_mat[~bail_sel & two_tone_sel & v_sel, :]
                    align_dict[sess_id][signal_type][region]['first_hit_var_'+v] = first_mat[hit_sel & ~bail_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['first_miss_var_'+v] = first_mat[miss_sel & ~bail_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['first_hit_one_tone_var_'+v] = first_mat[hit_sel & one_tone_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['first_miss_one_tone_var_'+v] = first_mat[miss_sel & one_tone_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['first_hit_two_tone_var_'+v] = first_mat[hit_sel & two_tone_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['first_miss_two_tone_var_'+v] = first_mat[miss_sel & two_tone_sel & v_sel,:]

                    align_dict[sess_id][signal_type][region]['second_var_'+v] = second_mat[two_tone_sel & ~bail_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_hit_var_'+v] = second_mat[hit_sel & two_tone_sel & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_miss_var_'+v] = second_mat[miss_sel & two_tone_sel & v_sel,:]

                    align_dict[sess_id][signal_type][region]['second_cong_var_'+v] = second_mat[two_tone_sel & ~bail_sel & ~incongruent & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_incong_var_'+v] = second_mat[two_tone_sel & ~bail_sel & incongruent & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_hit_cong_var_'+v] = second_mat[hit_sel & two_tone_sel & ~incongruent & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_miss_cong_var_'+v] = second_mat[miss_sel & two_tone_sel & ~incongruent & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_hit_incong_var_'+v] = second_mat[hit_sel & two_tone_sel & incongruent & v_sel,:]
                    align_dict[sess_id][signal_type][region]['second_miss_incong_var_'+v] = second_mat[miss_sel & two_tone_sel & incongruent & v_sel,:]

                    align_dict[sess_id][signal_type][region]['first_var_'+v+'_db_offset'] = first_mat[two_tone_sel & ~bail_sel & v_sel & ~no_tone_offset, :]
                    align_dict[sess_id][signal_type][region]['first_hit_var_'+v+'_db_offset'] = first_mat[two_tone_sel & hit_sel & ~bail_sel & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['first_miss_var_'+v+'_db_offset'] = first_mat[two_tone_sel & miss_sel & ~bail_sel & v_sel & ~no_tone_offset,:]

                    align_dict[sess_id][signal_type][region]['first_var_'+v+'_no_db_offset'] = first_mat[two_tone_sel & ~bail_sel & v_sel & no_tone_offset, :]
                    align_dict[sess_id][signal_type][region]['first_hit_var_'+v+'_no_db_offset'] = first_mat[two_tone_sel & hit_sel & ~bail_sel & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['first_miss_var_'+v+'_no_db_offset'] = first_mat[two_tone_sel & miss_sel & ~bail_sel & v_sel & no_tone_offset,:]

                    align_dict[sess_id][signal_type][region]['second_var_'+v+'_db_offset'] = second_mat[two_tone_sel & ~bail_sel & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_hit_var_'+v+'_db_offset'] = second_mat[hit_sel & two_tone_sel & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_miss_var_'+v+'_db_offset'] = second_mat[miss_sel & two_tone_sel & v_sel & ~no_tone_offset,:]

                    align_dict[sess_id][signal_type][region]['second_var_'+v+'_no_db_offset'] = second_mat[two_tone_sel & ~bail_sel & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_hit_var_'+v+'_no_db_offset'] = second_mat[hit_sel & two_tone_sel & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_miss_var_'+v+'_no_db_offset'] = second_mat[miss_sel & two_tone_sel & v_sel & no_tone_offset,:]

                    align_dict[sess_id][signal_type][region]['second_cong_var_'+v+'_db_offset'] = second_mat[two_tone_sel & ~bail_sel & ~incongruent & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_incong_var_'+v+'_db_offset'] = second_mat[two_tone_sel & ~bail_sel & incongruent & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_hit_cong_var_'+v+'_db_offset'] = second_mat[hit_sel & two_tone_sel & ~incongruent & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_miss_cong_var_'+v+'_db_offset'] = second_mat[miss_sel & two_tone_sel & ~incongruent & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_hit_incong_var_'+v+'_db_offset'] = second_mat[hit_sel & two_tone_sel & incongruent & v_sel & ~no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_miss_incong_var_'+v+'_db_offset'] = second_mat[miss_sel & two_tone_sel & incongruent & v_sel & ~no_tone_offset,:]

                    align_dict[sess_id][signal_type][region]['second_cong_var_'+v+'_no_db_offset'] = second_mat[two_tone_sel & ~bail_sel & ~incongruent & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_incong_var_'+v+'_no_db_offset'] = second_mat[two_tone_sel & ~bail_sel & incongruent & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_hit_cong_var_'+v+'_no_db_offset'] = second_mat[hit_sel & two_tone_sel & ~incongruent & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_miss_cong_var_'+v+'_no_db_offset'] = second_mat[miss_sel & two_tone_sel & ~incongruent & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_hit_incong_var_'+v+'_no_db_offset'] = second_mat[hit_sel & two_tone_sel & incongruent & v_sel & no_tone_offset,:]
                    align_dict[sess_id][signal_type][region]['second_miss_incong_var_'+v+'_no_db_offset'] = second_mat[miss_sel & two_tone_sel & incongruent & v_sel & no_tone_offset,:]

                    for tone_type in tone_types:
                        side_type = fpah.get_implant_rel_side(tone_port[subj_id][tone_type], region_side)
                        stim_sel_first = tone_infos.apply(lambda x: x[0] == tone_type if utils.is_list(x) else x == tone_type).to_numpy() & ~bail_sel
                        stim_sel_second = tone_infos.apply(lambda x: x[1] == tone_type if (utils.is_list(x) and len(x) > 1) else False).to_numpy() & ~bail_sel

                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_var_'+v] = first_mat[stim_sel_first & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_hit_var_'+v] = first_mat[stim_sel_first & hit_sel & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_miss_var_'+v] = first_mat[stim_sel_first & miss_sel & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_hit_one_tone_var_'+v] = first_mat[stim_sel_first & hit_sel & one_tone_sel & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_miss_one_tone_var_'+v] = first_mat[stim_sel_first & miss_sel & one_tone_sel & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_hit_two_tone_var_'+v] = first_mat[stim_sel_first & hit_sel & two_tone_sel & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_miss_two_tone_var_'+v] = first_mat[stim_sel_first & miss_sel & two_tone_sel & v_sel,:]

                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_var_'+v] = second_mat[stim_sel_second & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_hit_var_'+v] = second_mat[stim_sel_second & hit_sel & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_miss_var_'+v] = second_mat[stim_sel_second & miss_sel & v_sel,:]

                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_cong_var_'+v] = second_mat[stim_sel_second & two_tone_sel & ~bail_sel & ~incongruent & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_incong_var_'+v] = second_mat[stim_sel_second & two_tone_sel & ~bail_sel & incongruent & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_hit_cong_var_'+v] = second_mat[stim_sel_second & hit_sel & two_tone_sel & ~incongruent & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_miss_cong_var_'+v] = second_mat[stim_sel_second & miss_sel & two_tone_sel & ~incongruent & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_hit_incong_var_'+v] = second_mat[stim_sel_second & hit_sel & two_tone_sel & incongruent & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_miss_incong_var_'+v] = second_mat[stim_sel_second & miss_sel & two_tone_sel & incongruent & v_sel,:]

                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_var_'+v+'_db_offset'] = first_mat[two_tone_sel & stim_sel_first & ~no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_hit_var_'+v+'_db_offset'] = first_mat[two_tone_sel & stim_sel_first & hit_sel & ~no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_miss_var_'+v+'_db_offset'] = first_mat[two_tone_sel & stim_sel_first & miss_sel & ~no_tone_offset & v_sel,:]

                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_var_'+v+'_no_db_offset'] = first_mat[two_tone_sel & stim_sel_first & no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_hit_var_'+v+'_no_db_offset'] = first_mat[two_tone_sel & stim_sel_first & hit_sel & no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['first_'+side_type+'_miss_var_'+v+'_no_db_offset'] = first_mat[two_tone_sel & stim_sel_first & miss_sel & no_tone_offset & v_sel,:]

                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_var_'+v+'_db_offset'] = second_mat[stim_sel_second & ~no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_hit_var_'+v+'_db_offset'] = second_mat[stim_sel_second & hit_sel & ~no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_miss_var_'+v+'_db_offset'] = second_mat[stim_sel_second & miss_sel & ~no_tone_offset & v_sel,:]

                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_var_'+v+'_no_db_offset'] = second_mat[stim_sel_second & no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_hit_var_'+v+'_no_db_offset'] = second_mat[stim_sel_second & hit_sel & no_tone_offset & v_sel,:]
                        align_dict[sess_id][signal_type][region]['second_'+side_type+'_miss_var_'+v+'_no_db_offset'] = second_mat[stim_sel_second & miss_sel & no_tone_offset & v_sel,:]


                # response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cue_ts, pre, post)
                align_dict = cue
                # only look at response cues before cpoke outs
                sel = norm_cpoke_out_sel

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & sel,:]
                align_dict[sess_id][signal_type][region]['hit'] = mat[hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['miss'] = mat[miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['stay'] = mat[prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff & sel,:]

                align_dict[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & sel,:]

                align_dict[sess_id][signal_type][region]['stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff & sel,:]

                align_dict[sess_id][signal_type][region]['stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff & sel,:]

                align_dict[sess_id][signal_type][region]['stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & sel,:]

                align_dict[sess_id][signal_type][region]['hit_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['hit_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & hit_sel & sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & miss_sel & sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & miss_sel & sel,:]

                align_dict[sess_id][signal_type][region]['one_tone'] = mat[one_tone_sel & sel,:]
                align_dict[sess_id][signal_type][region]['two_tone'] = mat[two_tone_sel & sel,:]

                for side in sides:
                    side_type = fpah.get_implant_rel_side(side, region_side)
                    side_sel = (left_sel if side == 'left' else right_sel) & sel
                    prev_side_sel = (prev_left_sel if side == 'left' else prev_right_sel) & sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_hit'] = mat[hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss'] = mat[miss_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]


                for stim_type in stim_types:
                    stim_sel = (trial_stims == stim_type) & sel
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type] = mat[stim_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_prev_hit'] = mat[stim_sel & prev_hit_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_prev_miss'] = mat[stim_sel & prev_miss_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_hit'] = mat[stim_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_miss'] = mat[stim_sel & miss_sel,:]

                for dur in stim_durs:
                    dur_sel = (trial_durs == dur) & sel
                    align_dict[sess_id][signal_type][region]['dur_'+dur] = mat[dur_sel,:]
                    align_dict[sess_id][signal_type][region]['one_tone_dur_'+dur] = mat[one_tone_sel & dur_sel,:]
                    align_dict[sess_id][signal_type][region]['two_tone_dur_'+dur] = mat[two_tone_sel & dur_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel if side == 'left' else right_sel

                        align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur] = mat[dur_sel & side_sel,:]

                for delay in resp_delays:
                    delay_sel = (trial_delays == delay) & sel
                    align_dict[sess_id][signal_type][region]['delay_'+delay] = mat[delay_sel,:]
                    align_dict[sess_id][signal_type][region]['one_tone_delay_'+delay] = mat[one_tone_sel & delay_sel,:]
                    align_dict[sess_id][signal_type][region]['two_tone_delay_'+delay] = mat[two_tone_sel & delay_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel if side == 'left' else right_sel

                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay] = mat[delay_sel & side_sel,:]

                # poke out
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_out_ts, pre, post)
                align_dicts = [early_cpoke_out, cpoke_out]
                sels = [early_cpoke_out_sel, norm_cpoke_out_sel]

                for align_dict, sel in zip(align_dicts, sels):

                    align_dict['t'] = t
                    align_dict[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & ~bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & ~bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & ~bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['hit'] = mat[hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['miss'] = mat[miss_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['bail'] = mat[bail_sel,:] # intentionally left out sel since there is no resp cue in bails
                    align_dict[sess_id][signal_type][region]['stay'] = mat[prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff & sel,:]

                    align_dict[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & sel,:]

                    align_dict[sess_id][signal_type][region]['stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff & sel,:]

                    align_dict[sess_id][signal_type][region]['stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff & sel,:]

                    align_dict[sess_id][signal_type][region]['stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & sel,:]

                    align_dict[sess_id][signal_type][region]['hit_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['hit_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['hit_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['hit_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & hit_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['miss_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & miss_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['miss_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & miss_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['miss_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & miss_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['miss_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & miss_sel & sel,:]

                    align_dict[sess_id][signal_type][region]['one_tone'] = mat[one_tone_sel & ~bail_sel & sel,:]
                    align_dict[sess_id][signal_type][region]['two_tone'] = mat[two_tone_sel & ~bail_sel & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel & sel if side == 'left' else right_sel & sel
                        prev_side_sel = (prev_left_sel if side == 'left' else prev_right_sel) & ~bail_sel & sel

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_hit'] = mat[hit_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_miss'] = mat[miss_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]

                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff & side_sel,:]

                        align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_hit'] = mat[prev_hit_sel & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_miss'] = mat[prev_miss_sel & prev_side_sel,:]

                    for stim_type in stim_types:
                        stim_sel = (trial_stims == stim_type) & ~bail_sel & sel
                        align_dict[sess_id][signal_type][region]['stim_'+stim_type] = mat[stim_sel,:]
                        align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_prev_hit'] = mat[stim_sel & prev_hit_sel,:]
                        align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_prev_miss'] = mat[stim_sel & prev_miss_sel,:]
                        align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_hit'] = mat[stim_sel & hit_sel,:]
                        align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_miss'] = mat[stim_sel & miss_sel,:]

                    for dur in stim_durs:
                        dur_sel = (trial_durs == dur) & ~bail_sel & sel
                        align_dict[sess_id][signal_type][region]['dur_'+dur] = mat[dur_sel,:]
                        align_dict[sess_id][signal_type][region]['one_tone_dur_'+dur] = mat[one_tone_sel & dur_sel,:]
                        align_dict[sess_id][signal_type][region]['two_tone_dur_'+dur] = mat[two_tone_sel & dur_sel,:]
                        align_dict[sess_id][signal_type][region]['dur_'+dur+'_hit'] = mat[dur_sel & hit_sel,:]
                        align_dict[sess_id][signal_type][region]['dur_'+dur+'_miss'] = mat[dur_sel & miss_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_rel_side(side, region_side)
                            side_sel = left_sel if side == 'left' else right_sel

                            align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur] = mat[dur_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur+'_hit'] = mat[dur_sel & hit_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur+'_miss'] = mat[dur_sel & miss_sel & side_sel,:]

                    for delay in resp_delays:
                        delay_sel = (trial_delays == delay) & ~bail_sel & sel
                        align_dict[sess_id][signal_type][region]['delay_'+delay] = mat[delay_sel,:]
                        align_dict[sess_id][signal_type][region]['one_tone_delay_'+delay] = mat[one_tone_sel & delay_sel,:]
                        align_dict[sess_id][signal_type][region]['two_tone_delay_'+delay] = mat[two_tone_sel & delay_sel,:]
                        align_dict[sess_id][signal_type][region]['delay_'+delay+'_hit'] = mat[delay_sel & hit_sel,:]
                        align_dict[sess_id][signal_type][region]['delay_'+delay+'_miss'] = mat[delay_sel & miss_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_rel_side(side, region_side)
                            side_sel = left_sel if side == 'left' else right_sel

                            align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay] = mat[delay_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay+'_hit'] = mat[delay_sel & hit_sel & side_sel,:]
                            align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay+'_miss'] = mat[delay_sel & miss_sel & side_sel,:]

                # response
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, response_ts, pre, post)
                align_dict = resp

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_hit'] = mat[prev_hit_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['prev_miss'] = mat[prev_miss_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['prev_bail'] = mat[prev_bail_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['hit'] = mat[hit_sel,:]
                align_dict[sess_id][signal_type][region]['miss'] = mat[miss_sel,:]
                align_dict[sess_id][signal_type][region]['bail'] = mat[bail_sel,:]
                align_dict[sess_id][signal_type][region]['stay'] = mat[prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff,:]

                align_dict[sess_id][signal_type][region]['hit_stay'] = mat[prev_choice_same & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch'] = mat[prev_choice_diff & hit_sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay'] = mat[prev_choice_same & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch'] = mat[prev_choice_diff & miss_sel,:]

                align_dict[sess_id][signal_type][region]['stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff,:]

                align_dict[sess_id][signal_type][region]['hit_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & hit_sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & miss_sel,:]

                align_dict[sess_id][signal_type][region]['stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff,:]
                align_dict[sess_id][signal_type][region]['stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff,:]

                align_dict[sess_id][signal_type][region]['stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same,:]
                align_dict[sess_id][signal_type][region]['stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff,:]
                align_dict[sess_id][signal_type][region]['switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff,:]
                align_dict[sess_id][signal_type][region]['stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff,:]
                align_dict[sess_id][signal_type][region]['switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff,:]

                align_dict[sess_id][signal_type][region]['stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff,:]
                align_dict[sess_id][signal_type][region]['stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same,:]
                align_dict[sess_id][signal_type][region]['switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff,:]

                align_dict[sess_id][signal_type][region]['hit_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & hit_sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay_prev_same_trial'] = mat[prev_trial_same & prev_choice_same & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch_prev_same_trial'] = mat[prev_trial_same & prev_choice_diff & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_stay_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_same & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_switch_prev_diff_trial'] = mat[prev_trial_diff & prev_choice_diff & miss_sel,:]

                align_dict[sess_id][signal_type][region]['hit_future_resp'] = mat[next_resp_sel & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_future_bail'] = mat[next_bail_sel & hit_sel,:]
                align_dict[sess_id][signal_type][region]['miss_future_resp'] = mat[next_resp_sel & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_future_bail'] = mat[next_bail_sel & miss_sel,:]

                align_dict[sess_id][signal_type][region]['hit_future_stay'] = mat[next_choice_same & hit_sel,:]
                align_dict[sess_id][signal_type][region]['hit_future_switch'] = mat[next_choice_diff & hit_sel,:]
                align_dict[sess_id][signal_type][region]['miss_future_stay'] = mat[next_choice_same & miss_sel,:]
                align_dict[sess_id][signal_type][region]['miss_future_switch'] = mat[next_choice_diff & miss_sel,:]

                align_dict[sess_id][signal_type][region]['one_tone'] = mat[one_tone_sel & ~bail_sel,:]
                align_dict[sess_id][signal_type][region]['two_tone'] = mat[two_tone_sel & ~bail_sel,:]

                for side in sides:
                    side_type = fpah.get_implant_rel_side(side, region_side)
                    side_sel = left_sel if side == 'left' else right_sel
                    prev_side_sel = (prev_left_sel if side == 'left' else prev_right_sel) & ~bail_sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_hit'] = mat[side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss'] = mat[side_sel & miss_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_hit'] = mat[prev_hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_miss'] = mat[prev_miss_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_hit_prev_hit'] = mat[prev_hit_sel & side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_hit_prev_miss'] = mat[prev_miss_sel & side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss_prev_hit'] = mat[prev_hit_sel & side_sel & miss_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss_prev_miss'] = mat[prev_miss_sel & side_sel & miss_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_hit'] = mat[side_sel & prev_choice_same & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_hit'] = mat[side_sel & prev_choice_diff & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_miss'] = mat[side_sel & prev_choice_same & miss_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_miss'] = mat[side_sel & prev_choice_diff & miss_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_hit_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_hit_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_hit_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_hit_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_miss_prev_hit'] = mat[prev_hit_sel & prev_choice_same & side_sel & miss_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_miss_prev_hit'] = mat[prev_hit_sel & prev_choice_diff & side_sel & miss_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_miss_prev_miss'] = mat[prev_miss_sel & prev_choice_same & side_sel & miss_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_miss_prev_miss'] = mat[prev_miss_sel & prev_choice_diff & side_sel & miss_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_same_correct'] = mat[prev_correct_same & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_same_correct'] = mat[prev_correct_same & prev_choice_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_diff_correct'] = mat[prev_correct_diff & prev_choice_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit_same_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss_same_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_same & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_hit_diff_correct'] = mat[prev_hit_sel & prev_choice_diff & prev_correct_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_same & prev_correct_diff & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_miss_diff_correct'] = mat[prev_miss_sel & prev_choice_diff & prev_correct_diff & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_hit_future_resp'] = mat[next_resp_sel & hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_hit_future_bail'] = mat[next_bail_sel & hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss_future_resp'] = mat[next_resp_sel & miss_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss_future_bail'] = mat[next_bail_sel & miss_sel & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_hit_future_stay'] = mat[next_choice_same & hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_hit_future_switch'] = mat[next_choice_diff & hit_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss_future_stay'] = mat[next_choice_same & miss_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_miss_future_switch'] = mat[next_choice_diff & miss_sel & side_sel,:]

                    correct_sel = (correct_sides == side) & ~bail_sel
                    align_dict[sess_id][signal_type][region][side_type+'_correct'] = mat[correct_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_correct_hit'] = mat[hit_sel & correct_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_correct_miss'] = mat[miss_sel & correct_sel,:]

                for stim_type in stim_types:
                    stim_sel = (trial_stims == stim_type) & ~bail_sel
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type] = mat[stim_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_prev_hit'] = mat[stim_sel & prev_hit_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_prev_miss'] = mat[stim_sel & prev_miss_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_hit'] = mat[stim_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region]['stim_'+stim_type+'_miss'] = mat[stim_sel & miss_sel,:]

                for dur in stim_durs:
                    dur_sel = (trial_durs == dur) & ~bail_sel
                    align_dict[sess_id][signal_type][region]['dur_'+dur] = mat[dur_sel,:]
                    align_dict[sess_id][signal_type][region]['one_tone_dur_'+dur] = mat[one_tone_sel & dur_sel,:]
                    align_dict[sess_id][signal_type][region]['two_tone_dur_'+dur] = mat[two_tone_sel & dur_sel,:]
                    align_dict[sess_id][signal_type][region]['dur_'+dur+'_hit'] = mat[dur_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region]['dur_'+dur+'_miss'] = mat[dur_sel & miss_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel if side == 'left' else right_sel

                        align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur] = mat[dur_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur+'_hit'] = mat[dur_sel & hit_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_dur_'+dur+'_miss'] = mat[dur_sel & miss_sel & side_sel,:]

                for delay in resp_delays:
                    delay_sel = (trial_delays == delay) & ~bail_sel
                    align_dict[sess_id][signal_type][region]['delay_'+delay] = mat[delay_sel,:]
                    align_dict[sess_id][signal_type][region]['one_tone_delay_'+delay] = mat[one_tone_sel & delay_sel,:]
                    align_dict[sess_id][signal_type][region]['two_tone_delay_'+delay] = mat[two_tone_sel & delay_sel,:]
                    align_dict[sess_id][signal_type][region]['delay_'+delay+'_hit'] = mat[delay_sel & hit_sel,:]
                    align_dict[sess_id][signal_type][region]['delay_'+delay+'_miss'] = mat[delay_sel & miss_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel if side == 'left' else right_sel

                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay] = mat[delay_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay+'_hit'] = mat[delay_sel & hit_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_delay_'+delay+'_miss'] = mat[delay_sel & miss_sel & side_sel,:]


                # time normalized signal matrices
                cue_poke_out = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, cpoke_out_ts, norm_cue_resp_bins*norm_cue_poke_out_pct,
                                                                      align_sel = norm_cpoke_out_sel)
                poke_out_cue = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, cue_ts, norm_cue_resp_bins*norm_cue_poke_out_pct,
                                                                      align_sel = early_cpoke_out_sel)
                poke_out_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, response_ts, norm_cue_resp_bins*(1-norm_cue_poke_out_pct),
                                                                       align_sel = norm_cpoke_out_sel)
                cue_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, response_ts, norm_cue_resp_bins*(1-norm_cue_poke_out_pct),
                                                                  align_sel = early_cpoke_out_sel)

                mats = [np.hstack((cue_poke_out, poke_out_resp)), np.hstack((poke_out_cue, cue_resp))]
                align_dicts = [cue_poke_out_resp, poke_out_cue_resp]
                sels = [norm_cpoke_out_sel, early_cpoke_out_sel]

                for mat, align_dict, sel in zip(mats, align_dicts, sels):
                    align_dict['t'] = np.linspace(0, 1, norm_cue_resp_bins)
                    align_dict[sess_id][signal_type][region]['stay'] = mat[prev_choice_same[sel],:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[prev_choice_diff[sel],:]

                    for side in sides:
                        side_type = fpah.get_implant_rel_side(side, region_side)
                        side_sel = left_sel[sel] if side == 'left' else right_sel[sel]

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_hit'] = mat[hit_sel[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_miss'] = mat[miss_sel[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & prev_choice_same[sel],:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & prev_choice_diff[sel],:]


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

        # fpah.plot_aligned_signals(cue, 'Response Cue Aligned - {} (session {})'.format(signal_type_title, sess_id),
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



# %% Set up average plot options

# modify these options to change what will be used in the average signal plots
signal_type = 'dff_iso'
signal_label = 'ΔF/F'
regions = ['PL', 'DMS', 'DLS', 'TS']
subjects = [199] #sorted(sess_ids.keys())
filter_outliers = False
outlier_thresh = 20
use_se = True
ph = 3.5;
pw = 5;
n_reg = len(regions)
resp_xlims = {'DMS': [-1.5,2], 'DLS': [-1.5,2], 'TS': [-1.5,2], 'PL': [-3,10]}
gen_xlims = {'DMS': [-1,1.5], 'DLS': [-1,1.5], 'TS': [-1,1.5], 'PL': [-3,3]}
tone_xlims = {'DMS': [-1,1.5], 'DLS': [-1,1.5], 'TS': [-1,1.5], 'PL': [-3,3]}

save_plots = True
show_plots = True
reward_time = None # 0.5 #
tone_end = 0.4 # 0.3 #

# make this wrapper to simplify the stack command by not having to include the options declared above
def stack_mats(mat_dict, groups=None):
    return fpah.stack_fp_mats(mat_dict, regions, sess_ids, subjects, signal_type, filter_outliers, outlier_thresh, groups)

cport_on_mats = stack_mats(cport_on)
cpoke_in_mats = stack_mats(cpoke_in)
early_cpoke_in_mats = stack_mats(early_cpoke_in)
cpoke_out_mats = stack_mats(cpoke_out)
early_cpoke_out_mats = stack_mats(early_cpoke_out)
tone_mats = stack_mats(tones)
cue_mats = stack_mats(cue)
resp_mats = stack_mats(resp)
cue_poke_resp_mats = stack_mats(cue_poke_out_resp)
poke_cue_resp_mats = stack_mats(poke_out_cue_resp)

all_mats = {Align.cport_on: cport_on_mats, Align.cpoke_in: cpoke_in_mats, Align.early_cpoke_in: early_cpoke_in_mats, Align.tone: tone_mats,
            Align.cue: cue_mats, Align.cpoke_out: cpoke_out_mats, Align.early_cpoke_out: early_cpoke_out_mats,
            Align.resp: resp_mats, Align.cue_poke_resp: cue_poke_resp_mats, Align.poke_cue_resp: poke_cue_resp_mats}

all_ts = {Align.cport_on: cport_on['t'], Align.cpoke_in: cpoke_in['t'], Align.early_cpoke_in: early_cpoke_in['t'], Align.tone: tones['t'],
          Align.cue: cue['t'], Align.cpoke_out: cpoke_out['t'], Align.early_cpoke_out: early_cpoke_out['t'],
          Align.resp: resp['t'], Align.cue_poke_resp: cue_poke_out_resp['t'], Align.poke_cue_resp: poke_out_cue_resp['t']}

all_xlims = {Align.cport_on: gen_xlims, Align.cpoke_in: gen_xlims, Align.early_cpoke_in: gen_xlims, Align.tone: tone_xlims,
            Align.cue: gen_xlims, Align.cpoke_out: gen_xlims, Align.early_cpoke_out: gen_xlims,
            Align.resp: resp_xlims, Align.cue_poke_resp: None, Align.poke_cue_resp: None}

all_dashlines = {Align.cport_on: None, Align.cpoke_in: None, Align.early_cpoke_in: None, Align.tone: tone_end,
                Align.cue: None, Align.cpoke_out: None, Align.early_cpoke_out: None, Align.resp: reward_time,
                Align.cue_poke_resp: norm_cue_poke_out_pct, Align.poke_cue_resp: norm_cue_poke_out_pct}

# left_left = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper left'}}
# left_right = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}
# right_left = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper left'}}
# right_right = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper right'}}
# all_legend_params = {Align.cport_on: {'DMS': {'loc': 'upper left'}, 'PL': None}, Align.cpoke_in: right_right, Align.early_cpoke_in: right_right,
#                      Align.tone: left_left, Align.cue: left_left, Align.cpoke_out: left_left, Align.early_cpoke_out: left_left,
#                      Align.resp: right_right, Align.cue_poke_resp: right_left, Align.poke_cue_resp: right_left}

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

    # if legend_params is None:
    #     legend_params = all_legend_params[align]

    fig, plotted = fpah.plot_avg_signals(plot_groups, group_labels, mat, regions, t, gen_title.format(align_title), plot_titles, x_label, signal_label, xlims_dict,
                                dashlines=dashlines, group_colors=group_colors, use_se=use_se, ph=ph, pw=pw) # legend_params=legend_params, 

    if plotted and not gen_plot_name is None:
        save_plot(fig, gen_plot_name.format(align))

    if not plotted:
        plt.close(fig)


# %% Choice, side, and prior reward groupings for multiple alignment points

plot_groups = [['stay', 'switch'], ['contra', 'ipsi'], ['contra_stay', 'contra_switch', 'ipsi_stay', 'ipsi_switch']]
group_labels = {'stay': 'Stay', 'switch': 'Switch',
                'ipsi': 'Ipsi', 'contra': 'Contra',
                'contra_stay': 'Contra Stay', 'contra_switch': 'Contra Switch',
                'ipsi_stay': 'Ipsi Stay', 'ipsi_switch': 'Ipsi Switch'}

plot_titles = ['Stay/Switch', 'Choice Side', 'Stay/Switch & Side']
gen_title = 'Choice Side & Stay/Switch Groupings Aligned to {}'
gen_plot_name = '{}_stay_switch_side'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out,
          Align.resp, Align.cue_poke_resp, Align.poke_cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# choice side and outcome
plot_groups = [['contra', 'ipsi'],
               ['contra_hit', 'contra_miss', 'ipsi_hit', 'ipsi_miss']]
group_labels = {'contra': 'Contra', 'ipsi': 'Ipsi',
                'contra_hit': 'Contra Hit', 'ipsi_hit': 'Ipsi Hit',
                'contra_miss': 'Contra Miss', 'ipsi_miss': 'Ipsi Miss'}

plot_titles = ['Choice Side', 'Choice Side/Outcome']
gen_title = 'Choice Side by Outcome Aligned to {}'
gen_plot_name = '{}_side_outcome'

aligns = [Align.cue, Align.cpoke_out, Align.early_cpoke_out, Align.resp, Align.cue_poke_resp, Align.poke_cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# prior & future outcome
plot_groups = [['prev_hit', 'prev_miss', 'prev_bail'], ['hit', 'miss', 'bail']]
group_labels = {'prev_hit': 'Prev Hit', 'prev_miss': 'Prev Miss', 'prev_bail': 'Prev Bail',
                'hit': 'Hit', 'miss': 'Miss', 'bail': 'Bail'}

plot_titles = ['Prior Trial Outcome', 'Current Trial Outcome']
gen_title = 'Previous and Current Outcome Groupings Aligned to {}'
gen_plot_name = '{}_prev_future_outcome'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cpoke_out, Align.early_cpoke_out]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# cue and response without current bails
plot_groups = [['prev_hit', 'prev_miss', 'prev_bail'], ['hit', 'miss']]
aligns = [Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# prior reward/choice/side
plot_groups = [['stay_prev_hit', 'switch_prev_hit', 'stay_prev_miss', 'switch_prev_miss'],
               ['contra_prev_hit', 'ipsi_prev_hit', 'contra_prev_miss', 'ipsi_prev_miss'],
               ['contra_stay_prev_hit', 'contra_switch_prev_hit', 'ipsi_stay_prev_hit', 'ipsi_switch_prev_hit'],
               ['contra_stay_prev_miss', 'contra_switch_prev_miss', 'ipsi_stay_prev_miss', 'ipsi_switch_prev_miss']]
group_labels = {'stay_prev_hit': 'Stay | Hit', 'switch_prev_hit': 'Switch | Hit',
                'stay_prev_miss': 'Stay | Miss', 'switch_prev_miss': 'Switch | Miss',
                'contra_prev_hit': 'Contra | Hit', 'ipsi_prev_hit': 'Ipsi | Hit',
                'contra_prev_miss': 'Contra | Miss', 'ipsi_prev_miss': 'Ipsi | Miss',
                'contra_stay_prev_hit': 'Contra Stay', 'contra_switch_prev_hit': 'Contra Switch',
                'ipsi_stay_prev_hit': 'Ipsi Stay', 'ipsi_switch_prev_hit': 'Ipsi Switch',
                'contra_stay_prev_miss': 'Contra Stay', 'contra_switch_prev_miss': 'Contra Switch',
                'ipsi_stay_prev_miss': 'Ipsi Stay', 'ipsi_switch_prev_miss': 'Ipsi Switch'}

plot_titles = ['Stay/Switch by Prior Outcome', 'Choice Side by Prior Outcome', 'Prior Hit', 'Prior Miss']
gen_title = 'Previous Outcome by Stay/Switch and Side Groupings Aligned to {}'
gen_plot_name = '{}_prev_outcome_stay_switch_side'

aligns = [Align.cport_on, Align.cue, Align.cpoke_out, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# %% prior choice side/outcome

aligns = [Align.cport_on, Align.cpoke_in, Align.cue]

plot_groups = [['prev_contra', 'prev_ipsi'], ['prev_contra_prev_hit', 'prev_contra_prev_miss', 'prev_ipsi_prev_hit', 'prev_ipsi_prev_miss']]
group_labels = {'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi',
                'prev_contra_prev_hit': 'Prev Contra Hit', 'prev_contra_prev_miss': 'Prev Contra Miss',
                'prev_ipsi_prev_hit': 'Prev Ipsi Hit', 'prev_ipsi_prev_miss': 'Prev Ipsi Miss'}

plot_titles = ['Prior Choice Side', 'Prior Choice Side & Outcome']
gen_title = 'Prior Choice Side and Outcome Groupings Aligned to {}'
gen_plot_name = '{}_prev_side_prev_outcome'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% prior/current stimuli/outcome

# prior stimuli & outcome
gen_plot_groups = ['prev_stim_{}', 'prev_stim_{}_prev_hit', 'prev_stim_{}_prev_miss']
plot_groups = [[group.format(s) for s in stim_types] for group in gen_plot_groups]
group_labels = {group.format(s): s for group in gen_plot_groups for s in stim_types}

plot_titles = ['Prior Stimulus', 'Prior Stimulus & Hit', 'Prior Stimulus & Miss']
gen_title = 'Prior Stimulus and Outcome Groupings Aligned to {}'
gen_plot_name = '{}_prev_stim_prev_outcome'

aligns = [Align.cport_on, Align.cpoke_in]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# current stimuli & prior/current outcome
aligns = [Align.cue, Align.cpoke_out, Align.resp]

gen_plot_groups = ['stim_{}', 'stim_{}_prev_hit', 'stim_{}_prev_miss']
plot_groups = [[group.format(s) for s in stim_types] for group in gen_plot_groups]
group_labels = {group.format(s): s for group in gen_plot_groups for s in stim_types}

plot_titles = ['Current Stimulus', 'Current Stimulus & Prior Hit', 'Current Stimulus & Prior Miss']
gen_title = 'Prior Outcome and Current Stimulus Groupings Aligned to {}'
gen_plot_name = '{}_stim_prev_outcome'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

gen_plot_groups = ['stim_{}', 'stim_{}_hit', 'stim_{}_miss']
plot_groups = [[group.format(s) for s in stim_types] for group in gen_plot_groups]
plot_groups.insert(0, ['one_tone', 'two_tone'])
group_labels = {group.format(s): s for group in gen_plot_groups for s in stim_types}
group_labels.update({'one_tone': '1 Tone', 'two_tone': '2 Tones'})

plot_titles = ['Number of Tones', 'Current Stimulus', 'Current Stimulus Hits', 'Current Stimulus Misses']
gen_title = 'Prior Outcome and Current Stimulus Groupings Aligned to {}'
gen_plot_name = '{}_stim_outcome'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Same/Different stimuli and stay/switch

# correct port same/diff by stay/switch and prior outcome
aligns = [Align.cue, Align.cpoke_out, Align.resp]

plot_groups = [['stay_prev_same_correct', 'switch_prev_same_correct', 'stay_prev_diff_correct', 'switch_prev_diff_correct'],
               ['stay_prev_hit_same_correct', 'switch_prev_miss_same_correct', 'stay_prev_miss_diff_correct', 'switch_prev_hit_diff_correct'],
               ['stay_prev_miss_same_correct', 'switch_prev_hit_same_correct', 'stay_prev_hit_diff_correct', 'switch_prev_miss_diff_correct']]
group_labels = {'stay_prev_same_correct': 'Stay | Same', 'switch_prev_same_correct': 'Switch | Same',
                'stay_prev_diff_correct': 'Stay | Diff', 'switch_prev_diff_correct': 'Switch | Diff',
                'stay_prev_hit_same_correct': 'Stay | Same', 'switch_prev_miss_same_correct': 'Switch | Same',
                'stay_prev_miss_diff_correct': 'Stay | Diff', 'switch_prev_hit_diff_correct': 'Switch | Diff',
                'stay_prev_miss_same_correct': 'Stay | Same', 'switch_prev_hit_same_correct': 'Switch | Same',
                'stay_prev_hit_diff_correct': 'Stay | Diff', 'switch_prev_miss_diff_correct': 'Switch | Diff'}

plot_titles = ['Correct Side Repeat and Choice', 'Rewarded Correct Side Repeat and Choice', 'Unrewarded Correct Side Repeat and Choice']
gen_title = 'Correct Side Repeat by Outcome Aligned to {}'
gen_plot_name = '{}_cor_side_repeat_stay_switch_outcome'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# trial repeat
plot_groups = [['stay_prev_same_trial', 'switch_prev_same_trial', 'stay_prev_diff_trial', 'switch_prev_diff_trial'],
               ['hit_stay_prev_same_trial', 'hit_switch_prev_same_trial', 'hit_stay_prev_diff_trial', 'hit_switch_prev_diff_trial'],
               ['miss_stay_prev_same_trial', 'miss_switch_prev_same_trial', 'miss_stay_prev_diff_trial', 'miss_switch_prev_diff_trial']]
group_labels = {'stay_prev_same_trial': 'Stay | Same', 'switch_prev_same_trial': 'Switch | Same',
                'stay_prev_diff_trial': 'Stay | Diff', 'switch_prev_diff_trial': 'Switch | Diff',
                'hit_stay_prev_same_trial': 'Stay | Same', 'hit_switch_prev_same_trial': 'Switch | Same',
                'hit_stay_prev_diff_trial': 'Stay | Diff', 'hit_switch_prev_diff_trial': 'Switch | Diff',
                'miss_stay_prev_same_trial': 'Stay | Same', 'miss_switch_prev_same_trial': 'Switch | Same',
                'miss_stay_prev_diff_trial': 'Stay | Diff', 'miss_switch_prev_diff_trial': 'Switch | Diff'}

plot_titles = ['Stim Repeat and Choice', 'Rewarded Stim Repeat and Choice', 'Unrewarded Stim Repeat and Choice']
gen_title = 'Stimulus Repeat by Outcome Aligned to {}'
gen_plot_name = '{}_stim_repeat_stay_switch_outcome'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# side of correct choice and choice side
plot_groups = [['contra_stay_prev_same_correct', 'contra_switch_prev_same_correct', 'contra_stay_prev_diff_correct', 'contra_switch_prev_diff_correct'],
               ['ipsi_stay_prev_same_correct', 'ipsi_switch_prev_same_correct', 'ipsi_stay_prev_diff_correct', 'ipsi_switch_prev_diff_correct']]
group_labels = {'contra_stay_prev_same_correct': 'Stay | Same', 'contra_switch_prev_same_correct': 'Switch | Same',
                'contra_stay_prev_diff_correct': 'Stay | Diff', 'contra_switch_prev_diff_correct': 'Switch | Diff',
                'ipsi_stay_prev_same_correct': 'Stay | Same', 'ipsi_switch_prev_same_correct': 'Switch | Same',
                'ipsi_stay_prev_diff_correct': 'Stay | Diff', 'ipsi_switch_prev_diff_correct': 'Switch | Diff'}

plot_titles = ['Contra Choice', 'Ipsi Choice']
gen_title = 'Correct Side Repeat by Choice Side Aligned to {}'
gen_plot_name = '{}_cor_side_repeat_stay_switch_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# side of correct choice, choice side, outcome
plot_groups = [['contra_stay_prev_hit_same_correct', 'contra_switch_prev_miss_same_correct', 'contra_stay_prev_miss_diff_correct', 'contra_switch_prev_hit_diff_correct'],
               ['contra_stay_prev_miss_same_correct', 'contra_switch_prev_hit_same_correct', 'contra_stay_prev_hit_diff_correct', 'contra_switch_prev_miss_diff_correct'],
               ['ipsi_stay_prev_hit_same_correct', 'ipsi_switch_prev_miss_same_correct', 'ipsi_stay_prev_miss_diff_correct', 'ipsi_switch_prev_hit_diff_correct'],
               ['ipsi_stay_prev_miss_same_correct', 'ipsi_switch_prev_hit_same_correct', 'ipsi_stay_prev_hit_diff_correct', 'ipsi_switch_prev_miss_diff_correct']]
group_labels = {'contra_stay_prev_hit_same_correct': 'Stay | Same', 'contra_switch_prev_miss_same_correct': 'Switch | Same',
                'contra_stay_prev_miss_diff_correct': 'Stay | Diff', 'contra_switch_prev_hit_diff_correct': 'Switch | Diff',
                'contra_stay_prev_miss_same_correct': 'Stay | Same', 'contra_switch_prev_hit_same_correct': 'Switch | Same',
                'contra_stay_prev_hit_diff_correct': 'Stay | Diff', 'contra_switch_prev_miss_diff_correct': 'Switch | Diff',
                'ipsi_stay_prev_hit_same_correct': 'Stay | Same', 'ipsi_switch_prev_miss_same_correct': 'Switch | Same',
                'ipsi_stay_prev_miss_diff_correct': 'Stay | Diff', 'ipsi_switch_prev_hit_diff_correct': 'Switch | Diff',
                'ipsi_stay_prev_miss_same_correct': 'Stay | Same', 'ipsi_switch_prev_hit_same_correct': 'Switch | Same',
                'ipsi_stay_prev_hit_diff_correct': 'Stay | Diff', 'ipsi_switch_prev_miss_diff_correct': 'Switch | Diff'}

plot_titles = ['Rewarded Contra Choice', 'Unrewarded Contra Choice', 'Rewarded Ipsi Choice',  'Unrewarded Ipsi Choice']
gen_title = 'Correct Side Repeat by Choice Side/Outcome Aligned to {}'
gen_plot_name = '{}_cor_side_repeat_stay_switch_side_outcome'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)



# %% Various Response Outcome groupings
align = Align.resp

# stay/switch by prior & current outcome
plot_groups = [['hit_stay', 'hit_switch', 'miss_stay', 'miss_switch'],
               ['hit_stay_prev_hit', 'hit_switch_prev_hit', 'hit_stay_prev_miss', 'hit_switch_prev_miss'],
               ['miss_stay_prev_hit', 'miss_switch_prev_hit', 'miss_stay_prev_miss', 'miss_switch_prev_miss']]
group_labels = {'hit_stay': 'Rewarded Stay', 'hit_switch': 'Rewarded Switch',
                'miss_stay': 'Unrewarded Stay', 'miss_switch': 'Unrewarded Switch',
                'hit_stay_prev_hit': 'Stay | Hit', 'hit_switch_prev_hit': 'Switch | Hit',
                'hit_stay_prev_miss': 'Stay | Miss', 'hit_switch_prev_miss': 'Switch | Miss',
                'miss_stay_prev_hit': 'Stay | Hit', 'miss_switch_prev_hit': 'Switch | Hit',
                'miss_stay_prev_miss': 'Stay | Miss', 'miss_switch_prev_miss': 'Switch | Miss'}

plot_titles = ['Stay/Switch by Outcome', 'Rewarded Stay/Switch by Prior Outcome', 'Unrewarded Stay/Switch by Prior Outcome']
gen_title = 'Stay/Switch By Prior & Current Outcome Aligned to {}'
gen_plot_name = '{}_stay_switch_prev_current_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# stay/switch by choice side, and prior/current outcome
plot_groups = [['contra_hit', 'ipsi_hit', 'contra_miss', 'ipsi_miss'],
               ['contra_stay_hit', 'ipsi_stay_hit', 'contra_switch_hit', 'ipsi_switch_hit'],
               ['contra_stay_miss', 'ipsi_stay_miss', 'contra_switch_miss', 'ipsi_switch_miss']]
group_labels = {'contra_hit': 'Contra Hit', 'ipsi_hit': 'Ipsi Hit',
                'contra_miss': 'Contra Miss', 'ipsi_miss': 'Ipsi Miss',
                'contra_stay_hit': 'Contra Stay', 'ipsi_stay_hit': 'Ipsi Stay',
                'contra_switch_hit': 'Contra Switch', 'ipsi_switch_hit': 'Ipsi Switch',
                'contra_stay_miss': 'Contra Stay', 'ipsi_stay_miss': 'Ipsi Stay',
                'contra_switch_miss': 'Contra Switch', 'ipsi_switch_miss': 'Ipsi Switch'}

plot_titles = ['Side/Outcome', 'Rewarded Stay/Switch by Side', 'Unrewarded Stay/Switch by Side']
gen_title = 'Choice Side & Stay/Switch By Outcome Aligned to {}'
gen_plot_name = '{}_side_stay_switch_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# choice side by prior/current outcome
plot_groups = [['contra_hit_prev_hit', 'contra_hit_prev_miss', 'contra_miss_prev_hit', 'contra_miss_prev_miss'],
               ['ipsi_hit_prev_hit', 'ipsi_hit_prev_miss', 'ipsi_miss_prev_hit', 'ipsi_miss_prev_miss']]
group_labels = {'contra_hit_prev_hit': 'Hit | Hit', 'contra_hit_prev_miss': 'Hit | Miss',
                'contra_miss_prev_hit': 'Miss | Hit', 'contra_miss_prev_miss': 'Miss | Miss',
                'ipsi_hit_prev_hit': 'Hit | Hit', 'ipsi_hit_prev_miss': 'Hit | Miss',
                'ipsi_miss_prev_hit': 'Miss | Hit', 'ipsi_miss_prev_miss': 'Miss | Miss'}

plot_titles = ['Contra Choice', 'Ipsi Choice']
gen_title = 'Choice Side By Prior & Current Outcome Aligned to {}'
gen_plot_name = '{}_side_prev_current_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# choice side & stay/switch by prior/current outcome
plot_groups = [['contra_stay_hit_prev_hit', 'ipsi_stay_hit_prev_hit', 'contra_switch_hit_prev_hit', 'ipsi_switch_hit_prev_hit'],
               ['contra_stay_hit_prev_miss', 'ipsi_stay_hit_prev_miss', 'contra_switch_hit_prev_miss', 'ipsi_switch_hit_prev_miss'],
               ['contra_stay_miss_prev_hit', 'ipsi_stay_miss_prev_hit', 'contra_switch_miss_prev_hit', 'ipsi_switch_miss_prev_hit'],
               ['contra_stay_miss_prev_miss', 'ipsi_stay_miss_prev_miss', 'contra_switch_miss_prev_miss', 'ipsi_switch_miss_prev_miss']]
group_labels = {'contra_stay_hit_prev_hit': 'Contra Stay', 'ipsi_stay_hit_prev_hit': 'Ipsi Stay',
                'contra_switch_hit_prev_hit': 'Contra Switch', 'ipsi_switch_hit_prev_hit': 'Ipsi Switch',
                'contra_stay_hit_prev_miss': 'Contra Stay', 'ipsi_stay_hit_prev_miss': 'Ipsi Stay',
                'contra_switch_hit_prev_miss': 'Contra Switch', 'ipsi_switch_hit_prev_miss': 'Ipsi Switch',
                'contra_stay_miss_prev_hit': 'Contra Stay', 'ipsi_stay_miss_prev_hit': 'Ipsi Stay',
                'contra_switch_miss_prev_hit': 'Contra Switch', 'ipsi_switch_miss_prev_hit': 'Ipsi Switch',
                'contra_stay_miss_prev_miss': 'Contra Stay', 'ipsi_stay_miss_prev_miss': 'Ipsi Stay',
                'contra_switch_miss_prev_miss': 'Contra Switch', 'ipsi_switch_miss_prev_miss': 'Ipsi Switch'}

plot_titles = ['Hit | Hit', 'Hit | Miss', 'Miss | Hit', 'Miss | Miss']
gen_title = 'Choice Side & Stay/Switch By Prior & Current Outcome Aligned to {}'
gen_plot_name = '{}_side_stay_switch_prev_current_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Future choice by outcome
align = Align.resp

plot_groups = [['hit_future_resp', 'hit_future_bail', 'miss_future_resp', 'miss_future_bail'],
               ['contra_hit_future_resp', 'contra_hit_future_bail', 'contra_miss_future_resp', 'contra_miss_future_bail'],
               ['ipsi_hit_future_resp', 'ipsi_hit_future_bail', 'ipsi_miss_future_resp', 'ipsi_miss_future_bail']]
group_labels = {'hit_future_resp': 'Resp | Hit', 'hit_future_bail': 'Bail | Hit',
                'miss_future_resp': 'Resp | Miss', 'miss_future_bail': 'Bail | Miss',
                'contra_hit_future_resp': 'Resp | Hit', 'contra_hit_future_bail': 'Bail | Hit',
                'contra_miss_future_resp': 'Resp | Miss', 'contra_miss_future_bail': 'Bail | Miss',
                'ipsi_hit_future_resp': 'Resp | Hit', 'ipsi_hit_future_bail': 'Bail | Hit',
                'ipsi_miss_future_resp': 'Resp | Miss', 'ipsi_miss_future_bail': 'Bail | Miss'}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Future Response Type By Choice Side & Outcome Aligned to {}'
gen_plot_name = '{}_side_outcome_future_resp'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


plot_groups = [['hit_future_stay', 'hit_future_switch', 'miss_future_stay', 'miss_future_switch'],
               ['contra_hit_future_stay', 'contra_hit_future_switch', 'contra_miss_future_stay', 'contra_miss_future_switch'],
               ['ipsi_hit_future_stay', 'ipsi_hit_future_switch', 'ipsi_miss_future_stay', 'ipsi_miss_future_switch']]
group_labels = {'hit_future_stay': 'Stay | Hit', 'hit_future_switch': 'Switch | Hit',
                'miss_future_stay': 'Stay | Miss', 'miss_future_switch': 'Switch | Miss',
                'contra_hit_future_stay': 'Stay | Hit', 'contra_hit_future_switch': 'Switch | Hit',
                'contra_miss_future_stay': 'Stay | Miss', 'contra_miss_future_switch': 'Switch | Miss',
                'ipsi_hit_future_stay': 'Stay | Hit', 'ipsi_hit_future_switch': 'Switch | Hit',
                'ipsi_miss_future_stay': 'Stay | Miss', 'ipsi_miss_future_switch': 'Switch | Miss'}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Future Stay/Switch Response By Choice Side & Outcome Aligned to {}'
gen_plot_name = '{}_side_outcome_future_stay_switch'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# %% Tone Alignments
align = Align.tone

for v in variants:
    # all tones together
    plot_groups = [['first_var_{}', 'second_var_{}'],
                   ['first_contra_var_{}', 'first_ipsi_var_{}', 'second_contra_var_{}', 'second_ipsi_var_{}']]

    group_labels = {'first_var_{}': 'First', 'second_var_{}': 'Second',
                    'first_contra_var_{}': 'First Contra', 'first_ipsi_var_{}': 'First Ipsi',
                    'second_contra_var_{}': 'Second Contra', 'second_ipsi_var_{}': 'Second Ipsi'}

    plot_groups = [[g.format(v) for g in pg] for pg in plot_groups]
    group_labels = {k.format(v): l for k, l in group_labels.items()}

    plot_titles = ['Tone Position', 'Tone Position & Port Side']
    gen_title = 'Tone Position & Associated Port Side for \''+v+'\' Variant Aligned to {}'
    gen_plot_name = '{}_position_side_'+v

    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


    # Single tone trials, side and outcome
    plot_groups = [['first_hit_one_tone_var_{}', 'first_miss_one_tone_var_{}'],
                   ['first_contra_hit_one_tone_var_{}', 'first_contra_miss_one_tone_var_{}', 'first_ipsi_hit_one_tone_var_{}', 'first_ipsi_miss_one_tone_var_{}']]
    group_labels = {'first_hit_one_tone_var_{}': 'Hit', 'first_miss_one_tone_var_{}': 'Miss',
                    'first_contra_hit_one_tone_var_{}': 'Contra Hit', 'first_contra_miss_one_tone_var_{}': 'Contra Miss',
                    'first_ipsi_hit_one_tone_var_{}': 'Ipsi Hit', 'first_ipsi_miss_one_tone_var_{}': 'Ipsi Miss'}

    plot_groups = [[g.format(v) for g in pg] for pg in plot_groups]
    group_labels = {k.format(v): l for k, l in group_labels.items()}

    plot_titles = ['All Tones by Outcome', 'Tones by Port Side & Outcome']
    gen_title = 'One Tone Trials by Port Side & Outcome for \''+v+'\' Variant Aligned to {}'
    gen_plot_name = '{}_one_tone_side_outcome_'+v

    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


    # two tone trials, position, side & outcome
    plot_groups = [['first_hit_two_tone_var_{}', 'first_miss_two_tone_var_{}', 'second_hit_var_{}', 'second_miss_var_{}'],
                   ['first_contra_hit_two_tone_var_{}', 'first_contra_miss_two_tone_var_{}', 'first_ipsi_hit_two_tone_var_{}', 'first_ipsi_miss_two_tone_var_{}'],
                   ['second_contra_hit_var_{}', 'second_contra_miss_var_{}', 'second_ipsi_hit_var_{}', 'second_ipsi_miss_var_{}']]
    group_labels = {'first_hit_two_tone_var_{}': 'First Hit', 'first_miss_two_tone_var_{}': 'First Miss',
                    'second_hit_var_{}': 'Second Hit', 'second_miss_var_{}': 'Second Miss',
                    'first_contra_hit_two_tone_var_{}': 'Contra Hit', 'first_contra_miss_two_tone_var_{}': 'Contra Miss',
                    'first_ipsi_hit_two_tone_var_{}': 'Ipsi Hit', 'first_ipsi_miss_two_tone_var_{}': 'Ipsi Miss',
                    'second_contra_hit_var_{}': 'Contra Hit', 'second_contra_miss_var_{}': 'Contra Miss',
                    'second_ipsi_hit_var_{}': 'Ipsi Hit', 'second_ipsi_miss_var_{}': 'Ipsi Miss'}

    plot_groups = [[g.format(v) for g in pg] for pg in plot_groups]
    group_labels = {k.format(v): l for k, l in group_labels.items()}

    plot_titles = ['All Tones by Position & Outcome', 'First Tones by Port Side & Outome', 'Second Tones by Port Side & Outcome']
    gen_title = 'Two Tone Trials by Tone Positions, Port Side, & Outcome for \''+v+'\' Variant Aligned to {}'
    gen_plot_name = '{}_two_tone_side_outcome_'+v

    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


    # two tone trials, position, side & outcome, no volume offset
    plot_groups = [['first_hit_var_{}_no_db_offset', 'first_miss_var_{}_no_db_offset', 'second_hit_var_{}_no_db_offset', 'second_miss_var_{}_no_db_offset'],
                   ['first_contra_hit_var_{}_no_db_offset', 'first_contra_miss_var_{}_no_db_offset', 'first_ipsi_hit_var_{}_no_db_offset', 'first_ipsi_miss_var_{}_no_db_offset'],
                   ['second_contra_hit_var_{}_no_db_offset', 'second_contra_miss_var_{}_no_db_offset', 'second_ipsi_hit_var_{}_no_db_offset', 'second_ipsi_miss_var_{}_no_db_offset']]
    group_labels = {'first_hit_var_{}_no_db_offset': 'First Hit', 'first_miss_var_{}_no_db_offset': 'First Miss',
                    'second_hit_var_{}_no_db_offset': 'Second Hit', 'second_miss_var_{}_no_db_offset': 'Second Miss',
                    'first_contra_hit_var_{}_no_db_offset': 'Contra Hit', 'first_contra_miss_var_{}_no_db_offset': 'Contra Miss',
                    'first_ipsi_hit_var_{}_no_db_offset': 'Ipsi Hit', 'first_ipsi_miss_var_{}_no_db_offset': 'Ipsi Miss',
                    'second_contra_hit_var_{}_no_db_offset': 'Contra Hit', 'second_contra_miss_var_{}_no_db_offset': 'Contra Miss',
                    'second_ipsi_hit_var_{}_no_db_offset': 'Ipsi Hit', 'second_ipsi_miss_var_{}_no_db_offset': 'Ipsi Miss'}

    plot_groups = [[g.format(v) for g in pg] for pg in plot_groups]
    group_labels = {k.format(v): l for k, l in group_labels.items()}

    plot_titles = ['All Tones by Position & Outcome', 'First Tones by Port Side & Outome', 'Second Tones by Port Side & Outcome']
    gen_title = 'Full Volume Two Tone Trials by Tone Positions, Port Side, & Outcome for \''+v+'\' Variant Aligned to {}'
    gen_plot_name = '{}_two_tone_side_outcome_no_offset_'+v

    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Trial duration and response delay duration

aligns = [Align.cue, Align.cpoke_out, Align.resp]

# trial durations by choice side
gen_plot_groups = ['dur_{}', 'contra_dur_{}', 'ipsi_dur_{}']
plot_groups = [[group.format(d) for d in stim_durs] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in stim_durs}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Trial Duration by Choice Side Aligned to {}'
gen_plot_name = '{}_trial_dur_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# trial durations by number of tones
gen_plot_groups = ['one_tone_dur_{}', 'two_tone_dur_{}']
plot_groups = [[group.format(d) for d in stim_durs] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in stim_durs}

plot_titles = ['One Tone', 'Two Tones']
gen_title = 'Trial Duration by Number of Tones Aligned to {}'
gen_plot_name = '{}_trial_dur_num_tones'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# response delays by choice side
gen_plot_groups = ['delay_{}', 'contra_delay_{}', 'ipsi_delay_{}']
plot_groups = [[group.format(d) for d in resp_delays] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in resp_delays}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Response Delays from Last Tone by Choice Side Aligned to {}'
gen_plot_name = '{}_resp_delay_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# response delays by number of tones
gen_plot_groups = ['one_tone_delay_{}', 'two_tone_delay_{}']
plot_groups = [[group.format(d) for d in resp_delays] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in resp_delays}

plot_titles = ['One Tone', 'Two Tones']
gen_title = 'Response Delays from Last Tone by Number of Tones Aligned to {}'
gen_plot_name = '{}_resp_delay_num_tones'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# break out by outcome at the response
align = Align.resp

gen_plot_groups = ['dur_{}_hit', 'dur_{}_miss']
plot_groups = [[group.format(d) for d in stim_durs] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in stim_durs}

plot_titles = ['Hits', 'Misses']
gen_title = 'Trial Duration by Outcome Aligned to {}'
gen_plot_name = '{}_trial_dur_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


gen_plot_groups = ['contra_dur_{}_hit', 'ipsi_dur_{}_hit', 'contra_dur_{}_miss', 'ipsi_dur_{}_miss']
plot_groups = [[group.format(d) for d in stim_durs] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in stim_durs}

plot_titles = ['Contra Hits', 'Ipsi Hits', 'Contra Misses', 'Ipsi Misses']
gen_title = 'Trial Duration by Choice Side & Outcome Aligned to {}'
gen_plot_name = '{}_trial_dur_side_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


gen_plot_groups = ['delay_{}_hit', 'delay_{}_miss']
plot_groups = [[group.format(d) for d in resp_delays] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in resp_delays}

plot_titles = ['Hits', 'Misses']
gen_title = 'Response Delay by Outcome Aligned to {}'
gen_plot_name = '{}_resp_delay_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


gen_plot_groups = ['contra_delay_{}_hit', 'ipsi_delay_{}_hit', 'contra_delay_{}_miss', 'ipsi_delay_{}_miss']
plot_groups = [[group.format(d) for d in resp_delays] for group in gen_plot_groups]
group_labels = {group.format(d): d for group in gen_plot_groups for d in resp_delays}

plot_titles = ['Contra Hits', 'Ipsi Hits', 'Contra Misses', 'Ipsi Misses']
gen_title = 'Response Delay by Choice Side & Outcome Aligned to {}'
gen_plot_name = '{}_resp_delay_side_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Look at some behavioral metrics

# sess_metrics = {}

# for sess_id in sess_ids:
#     sess_metrics[sess_id] = {}

#     trial_data = sess_data[sess_data['sessid'] == sess_id]
#     hit_sel = trial_data['hit'] == True
#     miss_sel = trial_data['hit'] == False
#     bail_sel = trial_data['bail'] == True

#     prev_hit_sel = np.insert(hit_sel[:-1].to_numpy(), 0, False)
#     prev_miss_sel = np.insert(miss_sel[:-1].to_numpy(), 0, False)
#     prev_bail_sel = np.insert(bail_sel[:-1].to_numpy(), 0, False)

#     prev_choice_same = np.insert(trial_data['choice'][:-1].to_numpy() == trial_data['choice'][1:].to_numpy(),
#                                  0, False)
#     prev_tone_same = np.insert(trial_data['correct_port'][:-1].to_numpy() == trial_data['correct_port'][1:].to_numpy(),
#                                  0, False)

#     # probability of outcome given previous outcome
#     sess_metrics[sess_id]['p(hit|prev hit)'] = np.sum(hit_sel & prev_hit_sel)/np.sum(prev_hit_sel)
#     sess_metrics[sess_id]['p(hit|prev miss)'] = np.sum(hit_sel & prev_miss_sel)/np.sum(prev_miss_sel)
#     sess_metrics[sess_id]['p(hit|prev bail)'] = np.sum(hit_sel & prev_bail_sel)/np.sum(prev_bail_sel)

#     sess_metrics[sess_id]['p(miss|prev hit)'] = np.sum(miss_sel & prev_hit_sel)/np.sum(prev_hit_sel)
#     sess_metrics[sess_id]['p(miss|prev miss)'] = np.sum(miss_sel & prev_miss_sel)/np.sum(prev_miss_sel)
#     sess_metrics[sess_id]['p(miss|prev bail)'] = np.sum(miss_sel & prev_bail_sel)/np.sum(prev_bail_sel)

#     sess_metrics[sess_id]['p(bail|prev hit)'] = np.sum(bail_sel & prev_hit_sel)/np.sum(prev_hit_sel)
#     sess_metrics[sess_id]['p(bail|prev miss)'] = np.sum(bail_sel & prev_miss_sel)/np.sum(prev_miss_sel)
#     sess_metrics[sess_id]['p(bail|prev bail)'] = np.sum(bail_sel & prev_bail_sel)/np.sum(prev_bail_sel)

#     # stay and switch require animals to make responses on consecutive trials, so they cant bail
#     sess_metrics[sess_id]['p(stay|prev hit)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
#     sess_metrics[sess_id]['p(stay|prev miss)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

#     sess_metrics[sess_id]['p(switch|prev hit)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
#     sess_metrics[sess_id]['p(switch|prev miss)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

#     sess_metrics[sess_id]['p(stay|prev hit & same tone)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel & prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & prev_tone_same)
#     sess_metrics[sess_id]['p(stay|prev hit & diff tone)'] = np.sum(prev_choice_same & prev_hit_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & ~prev_tone_same)
#     sess_metrics[sess_id]['p(stay|prev miss & same tone)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel & prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & prev_tone_same)
#     sess_metrics[sess_id]['p(stay|prev miss & diff tone)'] = np.sum(prev_choice_same & prev_miss_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & ~prev_tone_same)

#     sess_metrics[sess_id]['p(switch|prev hit & same tone)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel & prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & prev_tone_same)
#     sess_metrics[sess_id]['p(switch|prev hit & diff tone)'] = np.sum(~prev_choice_same & prev_hit_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_hit_sel & ~bail_sel & ~prev_tone_same)
#     sess_metrics[sess_id]['p(switch|prev miss & same tone)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel & prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & prev_tone_same)
#     sess_metrics[sess_id]['p(switch|prev miss & diff tone)'] = np.sum(~prev_choice_same & prev_miss_sel & ~bail_sel & ~prev_tone_same)/np.sum(prev_miss_sel & ~bail_sel & ~prev_tone_same)

#     # probability of hit/miss given previous outcome when responding multiple times in a row
#     sess_metrics[sess_id]['p(hit|prev hit & no bail)'] = np.sum(hit_sel & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
#     sess_metrics[sess_id]['p(hit|prev miss & no bail)'] = np.sum(hit_sel & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

#     sess_metrics[sess_id]['p(miss|prev hit & no bail)'] = np.sum(miss_sel & prev_hit_sel & ~bail_sel)/np.sum(prev_hit_sel & ~bail_sel)
#     sess_metrics[sess_id]['p(miss|prev miss & no bail)'] = np.sum(miss_sel & prev_miss_sel & ~bail_sel)/np.sum(prev_miss_sel & ~bail_sel)

# %% Look at timing information

# sess_data['cpoke_out_latency'] = sess_data['cpoke_out_time'] - sess_data['response_cue_time']
# sess_data['cpoke_out_pct'] = sess_data['cpoke_out_latency']/sess_data['RT']

# plt.hist(sess_data['RT'])
# plt.hist(sess_data['cpoke_out_latency'])
# plt.hist(sess_data['cpoke_out_pct'])
