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
import time

# %% Load behavior data

# used for saving plots
behavior_name = 'Two-armed Bandit'

# get all session ids for given protocol
sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2)

# optionally limit sessions based on subject ids
subj_ids = [188] #[179, 207, 191, 182, 188]
sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

sess_data['cpoke_out_latency'] = sess_data['cpoke_out_time'] - sess_data['response_cue_time']

n_back = 5
bah.get_rew_rate_hist(sess_data, n_back=n_back, kernel='uniform')

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

# sess_id = 101926
# sub_signal = [542, 545]
# filter_outliers = True

# subj_id = [i for i,s in sess_ids.items() if sess_id in s][0]
# sess_fp = fp_data[subj_id][sess_id]
# _ = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
#                             title='Sub Signal - Subject {}, Session {}'.format(subj_id, sess_id),
#                             filter_outliers=filter_outliers,
#                             t_min=sub_signal[0], t_max=sub_signal[1], dec=1)

# %% Get all aligned/sorted stacked signals

signal_types = ['z_dff_iso'] # 'baseline_corr_lig','baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}
cport_on = copy.deepcopy(data_dict)
early_cpoke_in = copy.deepcopy(data_dict)
cpoke_in = copy.deepcopy(data_dict)
cue = copy.deepcopy(data_dict)
early_cpoke_out = copy.deepcopy(data_dict)
cpoke_out = copy.deepcopy(data_dict)
resp = copy.deepcopy(data_dict)
cue_poke_out_resp = copy.deepcopy(data_dict)
poke_out_cue_resp = copy.deepcopy(data_dict)

block_rates = np.unique(sess_data['block_prob'])
side_rates = np.unique(sess_data['side_prob'])
choice_block_probs = np.unique(sess_data['choice_block_prob'])
choice_block_probs = [x for x in choice_block_probs if not 'nan' in x]
choice_probs = np.unique(np.round(sess_data['choice_prob'], 2)*100)
choice_probs = choice_probs[~np.isnan(choice_probs)]
sides = ['left', 'right']

# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
rew_hist_bin_edges = np.linspace(-0.001, 1.001, 4)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{:.0f}:{:.0f}%'.format(abs(b.left)*100, b.right*100) for b in rew_hist_bins}

rew_hist_diffs_bin_edges = np.concatenate((np.linspace(-1.001, -0.331, 3), np.linspace(0.331, 1.001, 3)))
rew_hist_diff_bins = pd.IntervalIndex.from_breaks(rew_hist_diffs_bin_edges)
rew_hist_diff_bin_strs = {b:'{:.0f}:{:.0f}%'.format(b.left*100, b.right*100) for b in rew_hist_diff_bins}

# declare settings for normalized cue to response intervals
norm_cue_resp_bins = 200
norm_cue_poke_out_pct = 0.2 # % of bins for cue to poke out or poke out to cue, depending on which comes first

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:
        print('Processing session {} for subject {}...'.format(sess_id, subj_id))
        start = time.perf_counter()

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
        side_rate = trial_data['side_prob'].to_numpy()
        choice_block_rate = trial_data['choice_block_prob'].to_numpy()
        high_choice = trial_data['chose_high'].to_numpy()
        right_choice = trial_data['chose_right'].to_numpy()
        left_choice = trial_data['chose_left'].to_numpy()

        prev_right_choice = np.insert(right_choice[:-1], 0, False)
        prev_left_choice = np.insert(left_choice[:-1], 0, False)
        prev_rewarded = np.insert(rewarded[:-1], 0, False)
        prev_unrewarded = np.insert(~rewarded[:-1], 0, False)

        rew_hist_bin_all = pd.cut(trial_data['rew_rate_hist_all'], rew_hist_bins)
        rew_hist_bin_left_all = pd.cut(trial_data['rew_rate_hist_left_all'], rew_hist_bins)
        rew_hist_bin_right_all = pd.cut(trial_data['rew_rate_hist_right_all'], rew_hist_bins)
        rew_hist_bin_left_only = pd.cut(trial_data['rew_rate_hist_left_only'], rew_hist_bins)
        rew_hist_bin_right_only = pd.cut(trial_data['rew_rate_hist_right_only'], rew_hist_bins)

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

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # need to make sure reward rate diffs are always contra-ipsi
                if region_side == 'left':
                    rew_hist_bin_diff_all = pd.cut(-trial_data['rew_rate_hist_diff_all'], rew_hist_diff_bins)
                    rew_hist_bin_diff_only = pd.cut(-trial_data['rew_rate_hist_diff_only'], rew_hist_diff_bins)
                else:
                    rew_hist_bin_diff_all = pd.cut(trial_data['rew_rate_hist_diff_all'], rew_hist_diff_bins)
                    rew_hist_bin_diff_only = pd.cut(trial_data['rew_rate_hist_diff_only'], rew_hist_diff_bins)

                # aligned to center port on
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cport_on_ts, pre, post)
                align_dict = cport_on
                sel = cport_on_sel

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded & sel,:]
                align_dict[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded & sel,:]
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches & sel,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = (left_choice if side == 'left' else right_choice) & sel
                    prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[side_sel & stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[side_sel & switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]

                    align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_rewarded & prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_unrewarded & prev_side_sel,:]

                for rew_bin in rew_hist_bins:
                    bin_str = rew_hist_bin_strs[rew_bin]
                    align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str] = mat[(rew_hist_bin_all == rew_bin) & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel
                        rew_side_all_same = rew_hist_bin_left_all if side == 'left' else rew_hist_bin_right_all
                        rew_side_all_diff = rew_hist_bin_right_all if side == 'left' else rew_hist_bin_left_all
                        rew_side_only_same = rew_hist_bin_left_only if side == 'left' else rew_hist_bin_right_only
                        rew_side_only_diff = rew_hist_bin_right_only if side == 'left' else rew_hist_bin_left_only

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & side_sel,:]

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & prev_side_sel,:]

                for rew_bin in rew_hist_diff_bins:
                    bin_str = rew_hist_diff_bin_strs[rew_bin]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str] = mat[(rew_hist_bin_diff_all == rew_bin) & sel,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str] = mat[(rew_hist_bin_diff_only == rew_bin) & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & prev_side_sel,:]


                # aligned to center poke in
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_in_ts, pre, post)
                align_dicts = [early_cpoke_in, cpoke_in]
                sels = [early_cpoke_in_sel, norm_cpoke_in_sel]

                for align_dict, sel in zip(align_dicts, sels):

                    align_dict['t'] = t
                    align_dict[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded & sel,:]
                    align_dict[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded & sel,:]
                    align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_side_sel & prev_rewarded,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded,:]

                    for rew_bin in rew_hist_bins:
                        bin_str = rew_hist_bin_strs[rew_bin]
                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str] = mat[(rew_hist_bin_all == rew_bin) & sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = (left_choice if side == 'left' else right_choice) & sel
                            prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel
                            rew_side_all_same = rew_hist_bin_left_all if side == 'left' else rew_hist_bin_right_all
                            rew_side_all_diff = rew_hist_bin_right_all if side == 'left' else rew_hist_bin_left_all
                            rew_side_only_same = rew_hist_bin_left_only if side == 'left' else rew_hist_bin_right_only
                            rew_side_only_diff = rew_hist_bin_right_only if side == 'left' else rew_hist_bin_left_only

                            align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & side_sel,:]

                            align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & prev_side_sel,:]

                    for rew_bin in rew_hist_diff_bins:
                        bin_str = rew_hist_diff_bin_strs[rew_bin]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str] = mat[(rew_hist_bin_diff_all == rew_bin) & sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str] = mat[(rew_hist_bin_diff_only == rew_bin) & sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = (left_choice if side == 'left' else right_choice) & sel
                            prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                            align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & prev_side_sel,:]


                # aligned to response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cue_ts, pre, post)
                align_dict = cue
                # only look at response cues before cpoke outs
                sel = norm_cpoke_out_sel

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded & sel,:]
                align_dict[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded & sel,:]
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches & sel,:]
                align_dict[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays & sel,:]
                align_dict[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches & sel,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = (left_choice if side == 'left' else right_choice) & sel
                    prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_side_sel & prev_rewarded,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded,:]

                for rew_bin in rew_hist_bins:
                    bin_str = rew_hist_bin_strs[rew_bin]
                    align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str] = mat[(rew_hist_bin_all == rew_bin) & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                        rew_side_all_same = rew_hist_bin_left_all if side == 'left' else rew_hist_bin_right_all
                        rew_side_all_diff = rew_hist_bin_right_all if side == 'left' else rew_hist_bin_left_all
                        rew_side_only_same = rew_hist_bin_left_only if side == 'left' else rew_hist_bin_right_only
                        rew_side_only_diff = rew_hist_bin_right_only if side == 'left' else rew_hist_bin_left_only

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & side_sel,:]

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & prev_side_sel,:]

                for rew_bin in rew_hist_diff_bins:
                    bin_str = rew_hist_diff_bin_strs[rew_bin]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str] = mat[(rew_hist_bin_diff_all == rew_bin) & sel,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str] = mat[(rew_hist_bin_diff_only == rew_bin) & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & prev_side_sel,:]

                for cp in choice_block_probs:
                    cp_sel = (choice_block_rate == cp) & sel
                    align_dict[sess_id][signal_type][region]['choice_block_'+cp] = mat[cp_sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_'+side_type] = mat[cp_sel & side_sel,:]

                # aligned to center poke out
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cpoke_out_ts, pre, post)
                align_dicts = [early_cpoke_out, cpoke_out]
                sels = [early_cpoke_out_sel, norm_cpoke_out_sel]

                for align_dict, sel in zip(align_dicts, sels):

                    align_dict['t'] = t
                    align_dict[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded & sel,:]
                    align_dict[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded & sel,:]
                    align_dict[sess_id][signal_type][region]['stay'] = mat[stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[switches & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches & sel,:]
                    align_dict[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays & sel,:]
                    align_dict[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches & sel,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = (left_choice if side == 'left' else right_choice) & sel
                        prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type] = mat[prev_side_sel,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_side_sel & prev_rewarded,:]
                        align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded,:]

                    for rew_bin in rew_hist_bins:
                        bin_str = rew_hist_bin_strs[rew_bin]
                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str] = mat[(rew_hist_bin_all == rew_bin) & sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = (left_choice if side == 'left' else right_choice) & sel
                            prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                            rew_side_all_same = rew_hist_bin_left_all if side == 'left' else rew_hist_bin_right_all
                            rew_side_all_diff = rew_hist_bin_right_all if side == 'left' else rew_hist_bin_left_all
                            rew_side_only_same = rew_hist_bin_left_only if side == 'left' else rew_hist_bin_right_only
                            rew_side_only_diff = rew_hist_bin_right_only if side == 'left' else rew_hist_bin_left_only

                            align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & side_sel,:]

                            align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_prev_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & prev_side_sel,:]

                    for rew_bin in rew_hist_diff_bins:
                        bin_str = rew_hist_diff_bin_strs[rew_bin]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str] = mat[(rew_hist_bin_diff_all == rew_bin) & sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str] = mat[(rew_hist_bin_diff_only == rew_bin) & sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = (left_choice if side == 'left' else right_choice) & sel
                            prev_side_sel = (prev_left_choice if side == 'left' else prev_right_choice) & sel

                            align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & prev_side_sel,:]
                            align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_prev_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & prev_side_sel,:]

                    for cp in choice_block_probs:
                        cp_sel = (choice_block_rate == cp) & sel
                        align_dict[sess_id][signal_type][region]['choice_block_'+cp] = mat[cp_sel,:]

                        for side in sides:
                            side_type = fpah.get_implant_side_type(side, region_side)
                            side_sel = left_choice if side == 'left' else right_choice
                            align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_'+side_type] = mat[cp_sel & side_sel,:]


                # aligned to response poke
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, resp_ts, pre, post)
                align_dict = resp

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded,:]
                align_dict[sess_id][signal_type][region]['rewarded'] = mat[rewarded,:]
                align_dict[sess_id][signal_type][region]['unrewarded'] = mat[~rewarded,:]
                align_dict[sess_id][signal_type][region]['rewarded_prev_rewarded'] = mat[rewarded & prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['rewarded_prev_unrewarded'] = mat[rewarded & prev_unrewarded,:]
                align_dict[sess_id][signal_type][region]['unrewarded_prev_rewarded'] = mat[~rewarded & prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['unrewarded_prev_unrewarded'] = mat[~rewarded & prev_unrewarded,:]
                align_dict[sess_id][signal_type][region]['stay'] = mat[stays,:]
                align_dict[sess_id][signal_type][region]['switch'] = mat[switches,:]
                align_dict[sess_id][signal_type][region]['stay_rewarded'] = mat[rewarded & stays,:]
                align_dict[sess_id][signal_type][region]['switch_rewarded'] = mat[rewarded & switches,:]
                align_dict[sess_id][signal_type][region]['stay_unrewarded'] = mat[~rewarded & stays,:]
                align_dict[sess_id][signal_type][region]['switch_unrewarded'] = mat[~rewarded & switches,:]
                align_dict[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded & stays,:]
                align_dict[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded & switches,:]
                align_dict[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded & stays,:]
                align_dict[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded & switches,:]
                align_dict[sess_id][signal_type][region]['stay_rewarded_prev_rewarded'] = mat[rewarded & stays & prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['stay_rewarded_prev_unrewarded'] = mat[rewarded & stays & prev_unrewarded,:]
                align_dict[sess_id][signal_type][region]['stay_unrewarded_prev_rewarded'] = mat[~rewarded & stays & prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['stay_unrewarded_prev_unrewarded'] = mat[~rewarded & stays & prev_unrewarded,:]
                align_dict[sess_id][signal_type][region]['switch_rewarded_prev_rewarded'] = mat[rewarded & switches & prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['switch_rewarded_prev_unrewarded'] = mat[rewarded & switches & prev_unrewarded,:]
                align_dict[sess_id][signal_type][region]['switch_unrewarded_prev_rewarded'] = mat[~rewarded & switches & prev_rewarded,:]
                align_dict[sess_id][signal_type][region]['switch_unrewarded_prev_unrewarded'] = mat[~rewarded & switches & prev_unrewarded,:]

                align_dict[sess_id][signal_type][region]['rewarded_future_stay'] = mat[rewarded & future_stays,:]
                align_dict[sess_id][signal_type][region]['rewarded_future_switch'] = mat[rewarded & future_switches,:]
                align_dict[sess_id][signal_type][region]['unrewarded_future_stay'] = mat[~rewarded & future_stays,:]
                align_dict[sess_id][signal_type][region]['unrewarded_future_switch'] = mat[~rewarded & future_switches,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = left_choice if side == 'left' else right_choice
                    prev_side_sel = prev_left_choice if side == 'left' else prev_right_choice

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[switches & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_rewarded'] = mat[side_sel & rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unrewarded'] = mat[side_sel & ~rewarded,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_rewarded_prev_rewarded'] = mat[side_sel & prev_rewarded & rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_rewarded_prev_unrewarded'] = mat[side_sel & prev_unrewarded & rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unrewarded_prev_rewarded'] = mat[side_sel & prev_rewarded & ~rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unrewarded_prev_unrewarded'] = mat[side_sel & prev_unrewarded & ~rewarded,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_rewarded'] = mat[side_sel & stays & rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_rewarded'] = mat[side_sel & switches & rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_unrewarded'] = mat[side_sel & stays & ~rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_unrewarded'] = mat[side_sel & switches & ~rewarded,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded & stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded & switches & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded & stays & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded & switches & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_stay_rewarded_prev_rewarded'] = mat[side_sel & stays & rewarded & prev_rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_rewarded_prev_unrewarded'] = mat[side_sel & stays & rewarded & prev_unrewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_unrewarded_prev_rewarded'] = mat[side_sel & stays & ~rewarded & prev_rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_stay_unrewarded_prev_unrewarded'] = mat[side_sel & stays & ~rewarded & prev_unrewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_rewarded_prev_rewarded'] = mat[side_sel & switches & rewarded & prev_rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_rewarded_prev_unrewarded'] = mat[side_sel & switches & rewarded & prev_unrewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_unrewarded_prev_rewarded'] = mat[side_sel & switches & ~rewarded & prev_rewarded,:]
                    align_dict[sess_id][signal_type][region][side_type+'_switch_unrewarded_prev_unrewarded'] = mat[side_sel & switches & ~rewarded & prev_unrewarded,:]

                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_rewarded'] = mat[prev_side_sel & prev_rewarded,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_rewarded_prev_rewarded'] = mat[prev_side_sel & prev_rewarded & rewarded,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_rewarded_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded & rewarded,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_unrewarded_prev_rewarded'] = mat[prev_side_sel & prev_rewarded & ~rewarded,:]
                    align_dict[sess_id][signal_type][region]['prev_'+side_type+'_unrewarded_prev_unrewarded'] = mat[prev_side_sel & prev_unrewarded & ~rewarded,:]

                    align_dict[sess_id][signal_type][region][side_type+'_rewarded_future_stay'] = mat[side_sel & rewarded & future_stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_rewarded_future_switch'] = mat[side_sel & rewarded & future_switches,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unrewarded_future_stay'] = mat[side_sel & ~rewarded & future_stays,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unrewarded_future_switch'] = mat[side_sel & ~rewarded & future_switches,:]

                for rew_bin in rew_hist_bins:
                    bin_str = rew_hist_bin_strs[rew_bin]
                    align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str] = mat[rew_hist_bin_all == rew_bin,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_rewarded'] = mat[(rew_hist_bin_all == rew_bin) & rewarded,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_unrewarded'] = mat[(rew_hist_bin_all == rew_bin) & ~rewarded,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        rew_side_all_same = rew_hist_bin_left_all if side == 'left' else rew_hist_bin_right_all
                        rew_side_all_diff = rew_hist_bin_right_all if side == 'left' else rew_hist_bin_left_all
                        rew_side_only_same = rew_hist_bin_left_only if side == 'left' else rew_hist_bin_right_only
                        rew_side_only_diff = rew_hist_bin_right_only if side == 'left' else rew_hist_bin_left_only

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_all == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str] = mat[(rew_side_all_same == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str] = mat[(rew_side_all_diff == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str] = mat[(rew_side_only_same == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str] = mat[(rew_side_only_diff == rew_bin) & side_sel,:]

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type+'_rewarded'] = mat[(rew_hist_bin_all == rew_bin) & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str+'_rewarded'] = mat[(rew_side_all_same == rew_bin) & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str+'_rewarded'] = mat[(rew_side_all_diff == rew_bin) & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str+'_rewarded'] = mat[(rew_side_only_same == rew_bin) & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str+'_rewarded'] = mat[(rew_side_only_diff == rew_bin) & side_sel & rewarded,:]

                        align_dict[sess_id][signal_type][region]['rew_hist_all_'+bin_str+'_'+side_type+'_unrewarded'] = mat[(rew_hist_bin_all == rew_bin) & side_sel & ~rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_all_'+bin_str+'_unrewarded'] = mat[(rew_side_all_same == rew_bin) & side_sel & ~rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_all_'+bin_str+'_unrewarded'] = mat[(rew_side_all_diff == rew_bin) & side_sel & ~rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_same_only_'+bin_str+'_unrewarded'] = mat[(rew_side_only_same == rew_bin) & side_sel & ~rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_'+side_type+'_diff_only_'+bin_str+'_unrewarded'] = mat[(rew_side_only_diff == rew_bin) & side_sel & ~rewarded,:]

                for rew_bin in rew_hist_diff_bins:
                    bin_str = rew_hist_diff_bin_strs[rew_bin]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str] = mat[rew_hist_bin_diff_all == rew_bin,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str] = mat[rew_hist_bin_diff_only == rew_bin,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_rewarded'] = mat[(rew_hist_bin_diff_all == rew_bin) & rewarded,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_rewarded'] = mat[(rew_hist_bin_diff_only == rew_bin) & rewarded,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_unrewarded'] = mat[(rew_hist_bin_diff_all == rew_bin) & ~rewarded,:]
                    align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_unrewarded'] = mat[(rew_hist_bin_diff_only == rew_bin) & ~rewarded,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice

                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type+'_rewarded'] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type+'_rewarded'] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_all_'+bin_str+'_'+side_type+'_unrewarded'] = mat[(rew_hist_bin_diff_all == rew_bin) & side_sel & ~rewarded,:]
                        align_dict[sess_id][signal_type][region]['rew_hist_diff_only_'+bin_str+'_'+side_type+'_unrewarded'] = mat[(rew_hist_bin_diff_only == rew_bin) & side_sel & ~rewarded,:]

                # by trial outcome and perceived port reward probability (p reward for port from prior trial)
                for cp in choice_block_probs:
                    cp_sel = choice_block_rate == cp
                    align_dict[sess_id][signal_type][region]['choice_block_'+cp] = mat[cp_sel,:]
                    align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_rewarded'] = mat[cp_sel & rewarded,:]
                    align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_unrewarded'] = mat[cp_sel & ~rewarded,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice if side == 'left' else right_choice
                        align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_'+side_type] = mat[cp_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_'+side_type+'_rewarded'] = mat[cp_sel & side_sel & rewarded,:]
                        align_dict[sess_id][signal_type][region]['choice_block_'+cp+'_'+side_type+'_unrewarded'] = mat[cp_sel & side_sel & ~rewarded,:]


                # time normalized signal matrices
                cue_poke_out = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, cpoke_out_ts, norm_cue_resp_bins*norm_cue_poke_out_pct,
                                                                      align_sel = norm_cpoke_out_sel)
                poke_out_cue = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, cue_ts, norm_cue_resp_bins*norm_cue_poke_out_pct,
                                                                      align_sel = early_cpoke_out_sel)
                poke_out_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, resp_ts, norm_cue_resp_bins*(1-norm_cue_poke_out_pct),
                                                                       align_sel = norm_cpoke_out_sel)
                cue_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, resp_ts, norm_cue_resp_bins*(1-norm_cue_poke_out_pct),
                                                                  align_sel = early_cpoke_out_sel)

                mats = [np.hstack((cue_poke_out, poke_out_resp)), np.hstack((poke_out_cue, cue_resp))]
                align_dicts = [cue_poke_out_resp, poke_out_cue_resp]
                sels = [norm_cpoke_out_sel, early_cpoke_out_sel]

                for mat, align_dict, sel in zip(mats, align_dicts, sels):
                    align_dict['t'] = np.linspace(0, 1, norm_cue_resp_bins)
                    align_dict[sess_id][signal_type][region]['stay'] = mat[stays[sel],:]
                    align_dict[sess_id][signal_type][region]['switch'] = mat[switches[sel],:]
                    align_dict[sess_id][signal_type][region]['prev_rewarded'] = mat[prev_rewarded[sel],:]
                    align_dict[sess_id][signal_type][region]['prev_unrewarded'] = mat[prev_unrewarded[sel],:]
                    align_dict[sess_id][signal_type][region]['stay_prev_rewarded'] = mat[prev_rewarded[sel] & stays[sel],:]
                    align_dict[sess_id][signal_type][region]['switch_prev_rewarded'] = mat[prev_rewarded[sel] & switches[sel],:]
                    align_dict[sess_id][signal_type][region]['stay_prev_unrewarded'] = mat[prev_unrewarded[sel] & stays[sel],:]
                    align_dict[sess_id][signal_type][region]['switch_prev_unrewarded'] = mat[prev_unrewarded[sel] & switches[sel],:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = left_choice[sel] if side == 'left' else right_choice[sel]

                        align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay'] = mat[stays[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch'] = mat[switches[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_rewarded'] = mat[prev_rewarded[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_prev_unrewarded'] = mat[prev_unrewarded[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_rewarded'] = mat[prev_rewarded[sel] & stays[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_rewarded'] = mat[prev_rewarded[sel] & switches[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_stay_prev_unrewarded'] = mat[prev_unrewarded[sel] & stays[sel] & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_switch_prev_unrewarded'] = mat[prev_unrewarded[sel] & switches[sel] & side_sel,:]

        print('  Finished in {:.1f} s'.format(time.perf_counter()-start))

# %% Set up average plot options

# modify these options to change what will be used in the average signal plots
signal_type = 'z_dff_iso' # 'dff_iso', 'df_baseline_iso', 'raw_lig'
signal_label = 'Z-scored ΔF/F'
regions = ['DMS', 'PL']
subjects = list(sess_ids.keys())
filter_outliers = False
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

cport_on_mats = stack_mats(cport_on)
cpoke_in_mats = stack_mats(cpoke_in)
early_cpoke_in_mats = stack_mats(early_cpoke_in)
cpoke_out_mats = stack_mats(cpoke_out)
early_cpoke_out_mats = stack_mats(early_cpoke_out)
cue_mats = stack_mats(cue)
resp_mats = stack_mats(resp)
cue_poke_resp_mats = stack_mats(cue_poke_out_resp)
poke_cue_resp_mats = stack_mats(poke_out_cue_resp)

all_mats = {Align.cport_on: cport_on_mats, Align.cpoke_in: cpoke_in_mats, Align.early_cpoke_in: early_cpoke_in_mats,
            Align.cue: cue_mats, Align.cpoke_out: cpoke_out_mats, Align.early_cpoke_out: early_cpoke_out_mats,
            Align.resp: resp_mats, Align.cue_poke_resp: cue_poke_resp_mats, Align.poke_cue_resp: poke_cue_resp_mats}

all_ts = {Align.cport_on: cport_on['t'], Align.cpoke_in: cpoke_in['t'], Align.early_cpoke_in: early_cpoke_in['t'],
          Align.cue: cue['t'], Align.cpoke_out: cpoke_out['t'], Align.early_cpoke_out: early_cpoke_out['t'],
          Align.resp: resp['t'], Align.cue_poke_resp: cue_poke_out_resp['t'], Align.poke_cue_resp: poke_out_cue_resp['t']}

all_xlims = {Align.cport_on: gen_xlims, Align.cpoke_in: gen_xlims, Align.early_cpoke_in: gen_xlims,
            Align.cue: gen_xlims, Align.cpoke_out: gen_xlims, Align.early_cpoke_out: gen_xlims,
            Align.resp: resp_xlims, Align.cue_poke_resp: None, Align.poke_cue_resp: None}

all_dashlines = {Align.cport_on: None, Align.cpoke_in: None, Align.early_cpoke_in: None, Align.cue: None,
                Align.cpoke_out: None, Align.early_cpoke_out: None, Align.resp: reward_time,
                Align.cue_poke_resp: norm_cue_poke_out_pct, Align.poke_cue_resp: norm_cue_poke_out_pct}

left_left = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper left'}}
left_right = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}
right_left = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper left'}}
right_right = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper right'}}
all_legend_params = {Align.cport_on: {'DMS': {'loc': 'upper left'}, 'PL': None}, Align.cpoke_in: right_right,
                     Align.early_cpoke_in: right_right, Align.cue: left_left, Align.cpoke_out: left_left, Align.early_cpoke_out: left_left,
                     Align.resp: left_left, Align.cue_poke_resp: right_left, Align.poke_cue_resp: right_left}

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
          Align.resp, Align.cue_poke_resp, Align.poke_cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# prior reward, choice/prior reward, side/choice/prior reward
plot_groups = [['prev_rewarded', 'prev_unrewarded'],
               ['stay_prev_rewarded', 'switch_prev_rewarded', 'stay_prev_unrewarded', 'switch_prev_unrewarded'],
               ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded']]
group_labels = {'prev_rewarded': 'Reward', 'prev_unrewarded': 'No Reward',
                'stay_prev_rewarded': 'Stay | Rew', 'switch_prev_rewarded': 'Switch | Rew',
                'stay_prev_unrewarded': 'Stay | No Rew', 'switch_prev_unrewarded': 'Switch | No Rew',
                'contra_prev_rewarded': 'Contra | Rew', 'ipsi_prev_rewarded': 'Ipsi | Rew',
                'contra_prev_unrewarded': 'Contra | No Rew', 'ipsi_prev_unrewarded': 'Ipsi | No Rew'}
plot_titles = ['Prior Outcome', 'Prior Outcome/Choice', 'Prior Outcome/Side']
gen_title = 'Prior Outcome by Choice or Side Aligned to {}'
gen_plot_name = '{}_prev_outcome_choice_side'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out,
          Align.resp, Align.cue_poke_resp, Align.poke_cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# prior reward/choice/side
plot_groups = [['contra_stay_prev_rewarded', 'contra_switch_prev_rewarded', 'ipsi_stay_prev_rewarded', 'ipsi_switch_prev_rewarded'],
               ['contra_stay_prev_unrewarded', 'contra_switch_prev_unrewarded', 'ipsi_stay_prev_unrewarded', 'ipsi_switch_prev_unrewarded']]
# group_labels = {'contra_stay_prev_rewarded': 'Contra Stay', 'contra_switch_prev_rewarded': 'Contra Switch',
#                 'ipsi_stay_prev_rewarded': 'Ipsi Stay', 'ipsi_switch_prev_rewarded': 'Ipsi Switch',
#                 'contra_stay_prev_unrewarded': 'Contra Stay', 'contra_switch_prev_unrewarded': 'Contra Switch',
#                 'ipsi_stay_prev_unrewarded': 'Ipsi Stay', 'ipsi_switch_prev_unrewarded': 'Ipsi Switch'}
group_labels = {'contra_stay_prev_rewarded': 'Contra -> Contra', 'contra_switch_prev_rewarded': 'Ipsi -> Contra',
                'ipsi_stay_prev_rewarded': 'Ipsi -> Ipsi', 'ipsi_switch_prev_rewarded': 'Contra -> Ipsi',
                'contra_stay_prev_unrewarded': 'Contra -> Contra', 'contra_switch_prev_unrewarded': 'Ipsi -> Contra',
                'ipsi_stay_prev_unrewarded': 'Ipsi -> Ipsi', 'ipsi_switch_prev_unrewarded': 'Contra -> Ipsi'}

plot_titles = ['Prev Rewarded', 'Prev Unrewarded']
gen_title = 'Prior Outcome by Prior and Current Side Choice Aligned to {}'
gen_plot_name = '{}_prev_outcome_stay_switch_side'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out,
          Align.resp, Align.cue_poke_resp, Align.poke_cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)



# %% Groupings by previous side choice and previous reward

plot_groups = [['prev_contra', 'prev_ipsi'], ['contra', 'ipsi'],
               ['prev_contra_prev_rewarded', 'prev_ipsi_prev_rewarded', 'prev_contra_prev_unrewarded', 'prev_ipsi_prev_unrewarded'],
               ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded']]
group_labels = {'contra': 'Contra', 'ipsi': 'Ipsi', 'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi',
                'contra_prev_rewarded': 'Contra | Rew', 'ipsi_prev_rewarded': 'Ipsi | Rew',
                'contra_prev_unrewarded': 'Contra | No Rew', 'ipsi_prev_unrewarded': 'Ipsi | No Rew',
                'prev_contra_prev_rewarded': 'Prev Rew Contra', 'prev_ipsi_prev_rewarded': 'Prev Rew Ipsi',
                'prev_contra_prev_unrewarded': 'Prev Unrew Contra', 'prev_ipsi_prev_unrewarded': 'Prev Unrew Ipsi'}

plot_titles = ['Prev Choice', 'Future Choice', 'Prev Outcome and Prev Choice', 'Prev Outcome and Future Choice']
gen_title = 'Prior Outcome by Prior or Next Choice Aligned to {}'
gen_plot_name = '{}_side_prev_side_prev_outcome'

aligns = [Align.cport_on, Align.cpoke_in, Align.early_cpoke_in, Align.cue, Align.cpoke_out, Align.early_cpoke_out]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# response, prior outcome and prior/current choice
align = Align.resp

plot_groups = [['prev_contra_prev_rewarded', 'prev_ipsi_prev_rewarded', 'prev_contra_prev_unrewarded', 'prev_ipsi_prev_unrewarded'],
               ['contra_prev_rewarded', 'ipsi_prev_rewarded', 'contra_prev_unrewarded', 'ipsi_prev_unrewarded'],]
group_labels = {'contra_prev_rewarded': 'Contra | Rew', 'ipsi_prev_rewarded': 'Ipsi | Rew',
                'contra_prev_unrewarded': 'Contra | No Rew', 'ipsi_prev_unrewarded': 'Ipsi | No Rew',
                'prev_contra_prev_rewarded': 'Prev Rew Contra', 'prev_ipsi_prev_rewarded': 'Prev Rew Ipsi',
                'prev_contra_prev_unrewarded': 'Prev Unrew Contra', 'prev_ipsi_prev_unrewarded': 'Prev Unrew Ipsi'}

plot_titles = ['Prev Outcome and Prev Choice', 'Prev Outcome and Current Choice']
gen_title = 'Prior Outcome by Prior or Current Choice Aligned to {}'
gen_plot_name = '{}_side_prev_side_prev_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# response, prior & current outcome by prior/current choice

plot_groups = [['contra_rewarded_prev_rewarded', 'ipsi_rewarded_prev_rewarded', 'contra_rewarded_prev_unrewarded', 'ipsi_rewarded_prev_unrewarded'],
               ['contra_unrewarded_prev_rewarded', 'ipsi_unrewarded_prev_rewarded', 'contra_unrewarded_prev_unrewarded', 'ipsi_unrewarded_prev_unrewarded'],
               ['prev_contra_rewarded_prev_rewarded', 'prev_ipsi_rewarded_prev_rewarded', 'prev_contra_rewarded_prev_unrewarded', 'prev_ipsi_rewarded_prev_unrewarded'],
               ['prev_contra_unrewarded_prev_rewarded', 'prev_ipsi_unrewarded_prev_rewarded', 'prev_contra_unrewarded_prev_unrewarded', 'prev_ipsi_unrewarded_prev_unrewarded']]
group_labels = {'contra_rewarded_prev_rewarded': 'Contra | Rew', 'ipsi_rewarded_prev_rewarded': 'Ipsi | Rew',
                'contra_rewarded_prev_unrewarded': 'Contra | No Rew', 'ipsi_rewarded_prev_unrewarded': 'Ipsi | No Rew',
                'contra_unrewarded_prev_rewarded': 'Contra | Rew', 'ipsi_unrewarded_prev_rewarded': 'Ipsi | Rew',
                'contra_unrewarded_prev_unrewarded': 'Contra | No Rew', 'ipsi_unrewarded_prev_unrewarded': 'Ipsi | No Rew',
                'prev_contra_rewarded_prev_rewarded': 'Prev Rew Contra', 'prev_ipsi_rewarded_prev_rewarded': 'Prev Rew Ipsi',
                'prev_contra_rewarded_prev_unrewarded': 'Prev Unrew Contra', 'prev_ipsi_rewarded_prev_unrewarded': 'Prev Unrew Ipsi',
                'prev_contra_unrewarded_prev_rewarded': 'Prev Rew Contra', 'prev_ipsi_unrewarded_prev_rewarded': 'Prev Rew Ipsi',
                'prev_contra_unrewarded_prev_unrewarded': 'Prev Unrew Contra', 'prev_ipsi_unrewarded_prev_unrewarded': 'Prev Unrew Ipsi'}

plot_titles = ['Rewarded by Previous Outcome/Choice', 'Unrewarded by Previous Outcome/Choice', 'Rewarded Choice by Previous Outcome', 'Unrewarded Choice by Previous Outcome']
gen_title = 'Prev & Current Outcome by Prev or Next Choice Aligned to {}'
gen_plot_name = '{}_side_prev_side_outcome_prev_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Current outcome at time of response, for multiple different groupings

align = Align.resp

# response outcome
plot_groups = [['rewarded', 'unrewarded'],
               ['rewarded_prev_rewarded', 'rewarded_prev_unrewarded', 'unrewarded_prev_rewarded', 'unrewarded_prev_unrewarded']]
group_labels = {'rewarded': 'Rewarded', 'unrewarded': 'Unrewarded',
                'rewarded_prev_rewarded': 'Rew | Rew', 'rewarded_prev_unrewarded': 'Rew | No Rew',
                'unrewarded_prev_rewarded': 'Unrew | Rew', 'unrewarded_prev_unrewarded': 'Unrew | No Rew'}

plot_titles = ['Current Outcome', 'Current & Prior Outcome']
gen_title = 'Current & Prior Outcome Aligned to {}'
gen_plot_name = '{}_outcome_prev_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# response side/outcome
plot_groups = [['contra_rewarded', 'ipsi_rewarded', 'contra_unrewarded', 'ipsi_unrewarded'],
               ['contra_rewarded_prev_rewarded', 'ipsi_rewarded_prev_rewarded', 'contra_rewarded_prev_unrewarded', 'ipsi_rewarded_prev_unrewarded'],
               ['contra_unrewarded_prev_rewarded', 'ipsi_unrewarded_prev_rewarded', 'contra_unrewarded_prev_unrewarded', 'ipsi_unrewarded_prev_unrewarded']]
group_labels = {'contra_rewarded': 'Rew Contra', 'ipsi_rewarded': 'Rew Ipsi',
                'contra_unrewarded': 'Unrew Contra', 'ipsi_unrewarded': 'Unrew Ipsi',
                'contra_rewarded_prev_rewarded': 'Contra | Rew', 'ipsi_rewarded_prev_rewarded': 'Ipsi | Rew',
                'contra_rewarded_prev_unrewarded': 'Contra | No Rew', 'ipsi_rewarded_prev_unrewarded': 'Ipsi | No Rew',
                'contra_unrewarded_prev_rewarded': 'Contra | Rew', 'ipsi_unrewarded_prev_rewarded': 'Ipsi | Rew',
                'contra_unrewarded_prev_unrewarded': 'Contra | No Rew', 'ipsi_unrewarded_prev_unrewarded': 'Ipsi | No Rew'}

plot_titles = ['Side/Outcome', 'Rewarded Side by Prior Outcome', 'Unrewarded Side by Prior Outcome']
gen_title = 'Current & Prior Outcome by Choice Side Aligned to {}'
gen_plot_name = '{}_side_outcome_prev_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# response choice/outcome
plot_groups = [['stay_rewarded', 'switch_rewarded', 'stay_unrewarded', 'switch_unrewarded'],
               ['stay_rewarded_prev_rewarded', 'switch_rewarded_prev_rewarded', 'stay_rewarded_prev_unrewarded', 'switch_rewarded_prev_unrewarded'],
               ['stay_unrewarded_prev_rewarded', 'switch_unrewarded_prev_rewarded', 'stay_unrewarded_prev_unrewarded', 'switch_unrewarded_prev_unrewarded']]
group_labels = {'stay_rewarded': 'Rew Stay', 'switch_rewarded': 'Rew Switch',
                'stay_unrewarded': 'Unrew Stay', 'switch_unrewarded': 'Unrew Switch',
                'stay_rewarded_prev_rewarded': 'Stay | Rew', 'switch_rewarded_prev_rewarded': 'Switch | Rew',
                'stay_rewarded_prev_unrewarded': 'Stay | No Rew', 'switch_rewarded_prev_unrewarded': 'Switch | No Rew',
                'stay_unrewarded_prev_rewarded': 'Stay | Rew', 'switch_unrewarded_prev_rewarded': 'Switch | Rew',
                'stay_unrewarded_prev_unrewarded': 'Stay | No Rew', 'switch_unrewarded_prev_unrewarded': 'Switch | No Rew'}

plot_titles = ['Choice/Outcome', 'Rewarded Choice by Prior Outcome', 'Unrewarded Choice by Prior Outcome']
gen_title = 'Current & Prior Outcome by Choice Aligned to {}'
gen_plot_name = '{}_stay_switch_outcome_prev_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# response choice/side/outcome
plot_groups = [['contra_rewarded', 'ipsi_rewarded', 'contra_unrewarded', 'ipsi_unrewarded'],
               ['stay_rewarded', 'switch_rewarded', 'stay_unrewarded', 'switch_unrewarded'],
               ['contra_stay_rewarded', 'contra_switch_rewarded', 'ipsi_stay_rewarded', 'ipsi_switch_rewarded'],
               ['contra_stay_unrewarded', 'contra_switch_unrewarded', 'ipsi_stay_unrewarded', 'ipsi_switch_unrewarded']]
group_labels = {'contra_rewarded': 'Rew Contra', 'ipsi_rewarded': 'Rew Ipsi',
                'contra_unrewarded': 'Unrew Contra', 'ipsi_unrewarded': 'Unrew Ipsi',
                'stay_rewarded': 'Rew Stay', 'switch_rewarded': 'Rew Switch',
                'stay_unrewarded': 'Unrew Stay', 'switch_unrewarded': 'Unrew Switch',
                'contra_stay_rewarded': 'Contra Stay', 'contra_stay_unrewarded': 'Contra Stay',
                'contra_switch_rewarded': 'Contra Switch', 'contra_switch_unrewarded': 'Contra Switch',
                'ipsi_stay_rewarded': 'Ipsi Stay', 'ipsi_stay_unrewarded': 'Ipsi Stay',
                'ipsi_switch_rewarded': 'Ipsi Switch', 'ipsi_switch_unrewarded': 'Ipsi Switch'}

plot_titles = ['Side/Outcome', 'Choice/Outcome', 'Rewarded Side/Choice', 'Unrewarded Side/Choice']
gen_title = 'Side Stay/Switch Choices By Outcome Aligned to {}'
gen_plot_name = '{}_side_stay_switch_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# response choice/side/outcome/prior outcome
plot_groups = [['contra_stay_rewarded_prev_rewarded', 'contra_switch_rewarded_prev_rewarded', 'ipsi_stay_rewarded_prev_rewarded', 'ipsi_switch_rewarded_prev_rewarded'],
               ['contra_stay_rewarded_prev_unrewarded', 'contra_switch_rewarded_prev_unrewarded', 'ipsi_stay_rewarded_prev_unrewarded', 'ipsi_switch_rewarded_prev_unrewarded'],
               ['contra_stay_unrewarded_prev_rewarded', 'contra_switch_unrewarded_prev_rewarded', 'ipsi_stay_unrewarded_prev_rewarded', 'ipsi_switch_unrewarded_prev_rewarded'],
               ['contra_stay_unrewarded_prev_unrewarded', 'contra_switch_unrewarded_prev_unrewarded', 'ipsi_stay_unrewarded_prev_unrewarded', 'ipsi_switch_unrewarded_prev_unrewarded']]
group_labels = {'contra_stay_rewarded_prev_rewarded': 'Contra Stay', 'contra_switch_rewarded_prev_rewarded': 'Contra Switch',
                'ipsi_stay_rewarded_prev_rewarded': 'Ipsi Stay', 'ipsi_switch_rewarded_prev_rewarded': 'Ipsi Switch',
                'contra_stay_rewarded_prev_unrewarded': 'Contra Stay', 'contra_switch_rewarded_prev_unrewarded': 'Contra Switch',
                'ipsi_stay_rewarded_prev_unrewarded': 'Ipsi Stay', 'ipsi_switch_rewarded_prev_unrewarded': 'Ipsi Switch',
                'contra_stay_unrewarded_prev_rewarded': 'Contra Stay', 'contra_switch_unrewarded_prev_rewarded': 'Contra Switch',
                'ipsi_stay_unrewarded_prev_rewarded': 'Ipsi Stay', 'ipsi_switch_unrewarded_prev_rewarded': 'Ipsi Switch',
                'contra_stay_unrewarded_prev_unrewarded': 'Contra Stay', 'contra_switch_unrewarded_prev_unrewarded': 'Contra Switch',
                'ipsi_stay_unrewarded_prev_unrewarded': 'Ipsi Stay', 'ipsi_switch_unrewarded_prev_unrewarded': 'Ipsi Switch'}

plot_titles = ['Rewarded | Reward', 'Rewarded | No Reward', 'Unrewarded | Reward', 'Unrewarded | No Reward']
gen_title = 'Side Stay/Switch Choices By Prior and Current Outcome Aligned to {}'
gen_plot_name = '{}_side_stay_switch_outcome_prev_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)

# %% Response aligned grouped by side/reward/future response

align = Align.resp

plot_groups = [['rewarded_future_stay', 'rewarded_future_switch', 'unrewarded_future_stay', 'unrewarded_future_switch'],
               ['contra_rewarded_future_stay', 'contra_rewarded_future_switch', 'ipsi_rewarded_future_stay', 'ipsi_rewarded_future_switch'],
               ['contra_unrewarded_future_stay', 'contra_unrewarded_future_switch', 'ipsi_unrewarded_future_stay', 'ipsi_unrewarded_future_switch']]
group_labels = {'rewarded_future_stay': 'Stay | Rew', 'rewarded_future_switch': 'Switch | Rew',
                'unrewarded_future_stay': 'Stay | No Rew', 'unrewarded_future_switch': 'Switch | No Rew',
                'contra_rewarded_future_stay': 'Stay | Rew Contra', 'contra_rewarded_future_switch': 'Switch | Rew Contra',
                'ipsi_rewarded_future_stay': 'Stay | Rew Ipsi', 'ipsi_rewarded_future_switch': 'Switch | Rew Ipsi',
                'contra_unrewarded_future_stay': 'Stay | Unrew Contra', 'contra_unrewarded_future_switch': 'Switch | Unrew Contra',
                'ipsi_unrewarded_future_stay': 'Stay | Unrew Ipsi', 'ipsi_unrewarded_future_switch': 'Switch | Unrew Ipsi'}

plot_titles = ['Outcome/Future Choice', 'Rewarded Side by Future Choice', 'Unrewarded Side by Future Choice']
gen_title = 'Current Outcome by Future Choice Aligned to {}'
gen_plot_name = '{}_side_outcome_future_stay_switch'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Group by reward history

aligns = [Align.cport_on, Align.cue, Align.cpoke_out, Align.resp]
rew_hist_rew_colors = plt.cm.seismic(np.linspace(0.6,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.seismic(np.linspace(0.4,0,len(rew_hist_bins)))
rew_hist_diff_colors = plt.cm.turbo(np.linspace(0.05,0.95,len(rew_hist_diff_bins)))

colors = rew_hist_unrew_colors
legend_params = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}

# all choice reward history
gen_plot_groups = ['rew_hist_all_{}', 'rew_hist_all_{}_contra', 'rew_hist_all_{}_ipsi']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_bin_strs[rew_bin]): rew_hist_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_bins}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History Over Last '+ str(n_back) + ' Choices Grouped by Choice Side Aligned to {}'
gen_plot_name = '{}_rew_hist_all'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history over all choices back
gen_plot_groups = ['rew_hist_contra_same_all_{}', 'rew_hist_contra_diff_all_{}', 'rew_hist_ipsi_same_all_{}', 'rew_hist_ipsi_diff_all_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_bin_strs[rew_bin]): rew_hist_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_bins}

plot_titles = ['Contra Choices, Contra History', 'Contra Choices, Ipsi History', 'Ipsi Choices, Ipsi History', 'Ipsi Choices, Contra History']
gen_title = 'Side Reward History Over Last '+ str(n_back) + ' Choices Grouped by Side Choice/History Aligned to {}'
gen_plot_name = '{}_rew_hist_side_all'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history over choices for only the stated side
gen_plot_groups = ['rew_hist_contra_same_only_{}', 'rew_hist_contra_diff_only_{}', 'rew_hist_ipsi_same_only_{}', 'rew_hist_ipsi_diff_only_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_bin_strs[rew_bin]): rew_hist_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_bins}

plot_titles = ['Contra Choices, Contra History', 'Contra Choices, Ipsi History', 'Ipsi Choices, Ipsi History', 'Ipsi Choices, Contra History']
gen_title = 'Side Reward History Over Last '+ str(n_back) + ' Side Choices Grouped by Side Choice/History Aligned to {}'
gen_plot_name = '{}_rew_hist_side_only'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history differences over all choices back
colors = rew_hist_diff_colors

gen_plot_groups = ['rew_hist_diff_all_{}', 'rew_hist_diff_all_{}_contra', 'rew_hist_diff_all_{}_ipsi']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_diff_bin_strs[rew_bin]): rew_hist_diff_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_diff_bins}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Side Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Choices Grouped by Side Choice/History Aligned to {}'
gen_plot_name = '{}_rew_hist_diff_all'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history differences over side choices back
gen_plot_groups = ['rew_hist_diff_only_{}', 'rew_hist_diff_only_{}_contra', 'rew_hist_diff_only_{}_ipsi']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_diff_bin_strs[rew_bin]): rew_hist_diff_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_diff_bins}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Side Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Side Choices Grouped by Side Choice/History Aligned to {}'
gen_plot_name = '{}_rew_hist_diff_only'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


#%% split response aligned reward history by rewarded/unrewarded
align = Align.resp
colors = np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors))

rew_labels = {'rewarded': 'rew', 'unrewarded': 'unrew'}

group_labels = {'rew_hist_all_{}_{}': '{} {}', 'rew_hist_all_{}_contra_{}': '{} {}', 'rew_hist_all_{}_ipsi_{}': '{} {}',
                'rew_hist_contra_same_all_{}_{}': '{} {}', 'rew_hist_contra_diff_all_{}_{}': '{} {}',
                'rew_hist_ipsi_same_all_{}_{}': '{} {}', 'rew_hist_ipsi_diff_all_{}_{}': '{} {}',
                'rew_hist_contra_same_only_{}_{}': '{} {}', 'rew_hist_contra_diff_only_{}_{}': '{} {}',
                'rew_hist_ipsi_same_only_{}_{}': '{} {}', 'rew_hist_ipsi_diff_only_{}_{}': '{} {}'}
group_labels = {k.format(rew_hist_bin_strs[rew_bin], rk): v.format(rv, rew_hist_bin_strs[rew_bin])
                for k, v in group_labels.items()
                for rew_bin in rew_hist_bins
                for rk, rv in rew_labels.items()}

# all choice history
plot_groups = ['rew_hist_all_{}_{}', 'rew_hist_all_{}_contra_{}', 'rew_hist_all_{}_ipsi_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin], r) for r in rew_labels.keys() for rew_bin in rew_hist_bins] for group in plot_groups]

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History Over Last '+ str(n_back) + ' Choices Grouped by Choice Side/History/Outcome Aligned to {}'
gen_plot_name = '{}_rew_hist_all_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)

# side choice history, all choices back
plot_groups = ['rew_hist_contra_same_all_{}_{}', 'rew_hist_contra_diff_all_{}_{}', 'rew_hist_ipsi_same_all_{}_{}', 'rew_hist_ipsi_diff_all_{}_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin], r) for r in rew_labels.keys() for rew_bin in rew_hist_bins] for group in plot_groups]

plot_titles = ['Contra Choices, Contra History', 'Contra Choices, Ipsi History', 'Ipsi Choices, Ipsi History', 'Ipsi Choices, Contra History']
gen_title = 'Side Reward History Over Last '+ str(n_back) + ' Choices Grouped by Choice Side/History/Outcome Aligned to {}'
gen_plot_name = '{}_rew_hist_side_all_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)

# side choice history, side choices back
plot_groups = ['rew_hist_contra_same_only_{}_{}', 'rew_hist_contra_diff_only_{}_{}', 'rew_hist_ipsi_same_only_{}_{}', 'rew_hist_ipsi_diff_only_{}_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin], r) for r in rew_labels.keys() for rew_bin in rew_hist_bins] for group in plot_groups]

plot_titles = ['Contra Choices, Contra History', 'Contra Choices, Ipsi History', 'Ipsi Choices, Ipsi History', 'Ipsi Choices, Contra History']
gen_title = 'Side Reward History Over Last '+ str(n_back) + ' Side Choices Grouped by Choice Side/History/Outcome Aligned to {}'
gen_plot_name = '{}_rew_hist_side_only_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# reward history differences
colors = rew_hist_diff_colors
legend_params = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}

group_labels = {'rew_hist_diff_all_{}_{}': '{}', 'rew_hist_diff_only_{}_{}': '{}',
                'rew_hist_diff_all_{}_contra_{}': '{}', 'rew_hist_diff_all_{}_ipsi_{}': '{}',
                'rew_hist_diff_only_{}_contra_{}': '{}', 'rew_hist_diff_only_{}_ipsi_{}': '{}'}
group_labels = {k.format(rew_hist_diff_bin_strs[rew_bin], rk): v.format(rew_hist_diff_bin_strs[rew_bin])
                for k, v in group_labels.items()
                for rew_bin in rew_hist_diff_bins
                for rk, rv in rew_labels.items()}

# all choices, compare side back methods
plot_groups = ['rew_hist_diff_all_{}_rewarded', 'rew_hist_diff_all_{}_unrewarded', 'rew_hist_diff_only_{}_rewarded', 'rew_hist_diff_only_{}_unrewarded']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in plot_groups]

plot_titles = ['Rewarded, All Choice History', 'Unrewarded, All Choice History', 'Rewarded, Side Choice History', 'Unrewarded, Side Choice History']
gen_title = 'Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Choices Grouped by Outcome Aligned to {}'
gen_plot_name = '{}_rew_hist_side_all_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choices, all choices back
plot_groups = ['rew_hist_diff_all_{}_contra_rewarded', 'rew_hist_diff_all_{}_contra_unrewarded', 'rew_hist_diff_all_{}_ipsi_rewarded', 'rew_hist_diff_all_{}_ipsi_unrewarded']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in plot_groups]

plot_titles = ['Rewarded Contra Choices', 'Unrewarded Contra Choices', 'Rewarded Ipsi Choices', 'Unrewarded Ipsi Choices']
gen_title = 'Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Choices Grouped by Side/Outcome Aligned to {}'
gen_plot_name = '{}_rew_hist_diff_all_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choices, side choices back
plot_groups = ['rew_hist_diff_only_{}_contra_rewarded', 'rew_hist_diff_only_{}_contra_unrewarded', 'rew_hist_diff_only_{}_ipsi_rewarded', 'rew_hist_diff_only_{}_ipsi_unrewarded']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in plot_groups]

plot_titles = ['Rewarded Contra Choices', 'Unrewarded Contra Choices', 'Rewarded Ipsi Choices', 'Unrewarded Ipsi Choices']
gen_title = 'Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Side Choices Grouped by Side/Outcome Aligned to {}'
gen_plot_name = '{}_rew_hist_diff_only_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


#%% reward history by previous side
aligns = [Align.cport_on, Align.cpoke_in, Align.cue]
colors = rew_hist_unrew_colors

# all choice reward history
gen_plot_groups = ['rew_hist_all_{}_prev_contra', 'rew_hist_all_{}_prev_ipsi']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_bin_strs[rew_bin]): rew_hist_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_bins}

plot_titles = ['Prev Contra Choice', 'Prev Ipsi Choice']
gen_title = 'Reward History Over Last '+ str(n_back) + ' Choices Grouped by Previous Choice Side Aligned to {}'
gen_plot_name = '{}_rew_hist_all_prev_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)

# side choice reward history over all choices back
gen_plot_groups = ['rew_hist_prev_contra_same_all_{}', 'rew_hist_prev_contra_diff_all_{}', 'rew_hist_prev_ipsi_same_all_{}', 'rew_hist_prev_ipsi_diff_all_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_bin_strs[rew_bin]): rew_hist_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_bins}

plot_titles = ['Prev Contra Choice, Contra History', 'Prev Contra Choice, Ipsi History', 'Prev Ipsi Choice, Ipsi History', 'Prev Ipsi Choice, Contra History']
gen_title = 'Side Reward History Over Last '+ str(n_back) + ' Choices Grouped by Previous Side/History Aligned to {}'
gen_plot_name = '{}_rew_hist_side_all_prev_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history over choices for only the stated side
gen_plot_groups = ['rew_hist_prev_contra_same_only_{}', 'rew_hist_prev_contra_diff_only_{}', 'rew_hist_prev_ipsi_same_only_{}', 'rew_hist_prev_ipsi_diff_only_{}']
plot_groups = [[group.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_bin_strs[rew_bin]): rew_hist_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_bins}

plot_titles = ['Prev Contra Choice, Contra History', 'Prev Contra Choice, Ipsi History', 'Prev Ipsi Choice, Ipsi History', 'Prev Ipsi Choice, Contra History']
gen_title = 'Side Reward History Over Last '+ str(n_back) + ' Side Choices Grouped by Previous Side/History Aligned to {}'
gen_plot_name = '{}_rew_hist_side_only_prev_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history differences over all choices back
colors = rew_hist_diff_colors

gen_plot_groups = ['rew_hist_diff_all_{}_prev_contra', 'rew_hist_diff_all_{}_prev_ipsi']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_diff_bin_strs[rew_bin]): rew_hist_diff_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_diff_bins}

plot_titles = ['Prev Contra Choice', 'Prev Ipsi Choice']
gen_title = 'Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Choices Grouped by Previous Choice Side Aligned to {}'
gen_plot_name = '{}_rew_hist_diff_all_prev_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# side choice reward history differences over side choices back
gen_plot_groups = ['rew_hist_diff_only_{}_prev_contra', 'rew_hist_diff_only_{}_prev_ipsi']
plot_groups = [[group.format(rew_hist_diff_bin_strs[rew_bin]) for rew_bin in rew_hist_diff_bins] for group in gen_plot_groups]
group_labels = {group.format(rew_hist_diff_bin_strs[rew_bin]): rew_hist_diff_bin_strs[rew_bin] for group in gen_plot_groups for rew_bin in rew_hist_diff_bins}

plot_titles = ['Prev Contra Choice', 'Prev Ipsi Choice']
gen_title = 'Reward History Differences (Contra-Ipsi) Over Last '+ str(n_back) + ' Side Choices Grouped by Previous Choice Side Aligned to {}'
gen_plot_name = '{}_rew_hist_diff_only_prev_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name, group_colors=colors)


# %% Group by block rates and port probability

# block rates and choice probabilities at cue and response
aligns = [Align.cue, Align.cpoke_out, Align.resp]

gen_plot_groups = ['choice_block_{}', 'choice_block_{}_contra', 'choice_block_{}_ipsi']
plot_groups = [[group.format(cp) for cp in choice_block_probs] for group in gen_plot_groups]
group_labels = {group.format(cp): cp for group in gen_plot_groups for cp in choice_block_probs}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Block and Choice Reward Probabilities Grouped by Choice Side Aligned to {}'
gen_plot_name = '{}_choice_block_prob_side'

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# block rates and choice probabilities at response by outcome
align = Align.resp

group_labels = {'choice_block_{}_rewarded': '{}', 'choice_block_{}_unrewarded': '{}',
                'choice_block_{}_contra_rewarded': '{}', 'choice_block_{}_contra_unrewarded': '{}',
                'choice_block_{}_ipsi_rewarded': '{}', 'choice_block_{}_ipsi_unrewarded': '{}'}
group_labels = {k.format(cp): v.format(cp) for k, v in group_labels.items() for cp in choice_block_probs}

# by outcome
plot_groups = ['choice_block_{}_rewarded', 'choice_block_{}_unrewarded']
plot_groups = [[group.format(cp) for cp in choice_block_probs] for group in plot_groups]

plot_titles = ['Rewarded', 'Unrewarded']
gen_title = 'Block and Choice Reward Probabilities Grouped by Outcome Aligned to {}'
gen_plot_name = '{}_choice_block_prob_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# by outcome and choice side
plot_groups = ['choice_block_{}_contra_rewarded', 'choice_block_{}_contra_unrewarded', 'choice_block_{}_ipsi_rewarded', 'choice_block_{}_ipsi_unrewarded']
plot_groups = [[group.format(cp) for cp in choice_block_probs] for group in plot_groups]

plot_titles = ['Rewarded Contra Choice', 'Unrewarded Contra Choice', 'Rewarded Ipsi Choice', 'Unrewarded Ipsi Choice']
gen_title = 'Block and Choice Reward Probabilities Grouped by Choice Side/Outcome Aligned to {}'
gen_plot_name = '{}_choice_block_prob_side_outcome'

plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# rewarded choice probability normalized to the activity at the time of the response poke
plot_groups = ['choice_block_{}_rewarded', 'choice_block_{}_contra_rewarded', 'choice_block_{}_ipsi_rewarded']
plot_groups = [[group.format(cp) for cp in choice_block_probs] for group in plot_groups]

plot_titles = ['Rewarded All Choices', 'Rewarded Contra Choices', 'Rewarded Ipsi Choices']
title = 'Rewarded PL Responses Normalized to Activity at t=0'
plot_name = '{}_choice_block_prob_side_rewarded_norm'.format(align)
region = 'PL'

# make normalized data dictionary
mat = all_mats[align]
t = all_ts[align]
x_label = fpah.get_align_xlabel(align)
xlims = all_xlims[align]
dashlines = all_dashlines[align]
legend_params = all_legend_params[align]

center_idx = np.argmin(np.abs(t))
norm_mat = {region: {g: [] for g in utils.flatten(plot_groups)}}
for g in utils.flatten(plot_groups):
    act = mat[region][g].copy()
    act = act - act[:,center_idx][:,None]
    norm_mat[region][g] = act

fig, _ = fpah.plot_avg_signals(plot_groups, group_labels, norm_mat, [region], t, title, plot_titles, x_label, 'Shifted Z-scored ΔF/F', xlims,
                            dashlines=dashlines, use_se=use_se, ph=ph*1.25, pw=pw)

save_plot(fig, plot_name)

# %% Clear variables

import gc

del cport_on_mats, cpoke_in_mats, early_cpoke_in_mats, cpoke_out_mats, early_cpoke_out_mats, cue_mats, resp_mats, cue_poke_resp_mats, poke_cue_resp_mats, all_mats
gc.collect()
