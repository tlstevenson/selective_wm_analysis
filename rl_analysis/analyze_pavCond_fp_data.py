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
behavior_name = 'Operant Pavlovian Conditioning'

# get all session ids for given protocol
sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=1)

# optionally limit sessions based on subject ids
subj_ids = [202]
sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

reload = False
loc_db = db.LocalDB_BasicRLTasks('pavlovCond')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)


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

# tmp_sess_id = {207: [102367]}
# tmp_fp_data, tmp_implant_info = fpah.load_fp_data(loc_db, tmp_sess_id)
# sub_signal =  [0, np.inf] # [381, 385] #
# filter_outliers = True
# dec = 20

# subj_id = list(tmp_sess_id.keys())[0]
# sess_id = tmp_sess_id[subj_id][0]
# #sess_fp = fp_data[subj_id][sess_id]
# sess_fp = tmp_fp_data[subj_id][sess_id]
# _ = fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'],
#                             title='Sub Signal - Subject {}, Session {}'.format(subj_id, sess_id),
#                             filter_outliers=filter_outliers,
#                             t_min=sub_signal[0], t_max=sub_signal[1], dec=dec)

# %% Get all aligned/sorted stacked signals

signal_types = ['z_dff_iso'] # 'baseline_corr_lig','baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

all_regions = np.unique([r for s in sess_ids.keys() for r in implant_info[s].keys()])
data_dict = {sess_id: {signal: {region: {} for region in all_regions} for signal in signal_types} for sess_id in utils.flatten(sess_ids)}

tone = copy.deepcopy(data_dict)
cue = copy.deepcopy(data_dict)
resp = copy.deepcopy(data_dict)
cue_resp = copy.deepcopy(data_dict)

rewards = np.unique(sess_data['reward_volume'])
tone_vols = np.flip(np.unique(sess_data['tone_db_offset']))
tone_rew_corrs = np.unique(sess_data['tone_reward_corr'])
tone_rew_corrs = tone_rew_corrs[~np.isnan(tone_rew_corrs)]
sides = ['left', 'right']

# declare settings for normalized cue to response intervals
norm_cue_resp_bins = 200

for subj_id in sess_ids.keys():
    for sess_id in sess_ids[subj_id]:

        trial_data = sess_data[sess_data['sessid'] == sess_id]
        sess_fp = fp_data[subj_id][sess_id]

        # get alignment trial filters
        resp_sel = trial_data['hit'] == True
        no_resp_sel = trial_data['hit'] == False
        prev_resp = np.insert(resp_sel[:-1], 0, False)
        prev_no_resp = np.insert(no_resp_sel[:-1], 0, False)
        future_resp = np.append(resp_sel[1:], False)
        future_no_resp = np.append(no_resp_sel[1:], False)

        resp_port = trial_data['response_port'].to_numpy()
        reward_tone = trial_data['reward_tone'].to_numpy()
        averse_outcome = trial_data['aversive_outcome'].to_numpy()

        # only look at tones after first response for each tone type
        post_rew_resp = np.full_like(resp_sel, False)
        post_unrew_resp = np.full_like(resp_sel, False)

        first_rew_resp = np.where(reward_tone & resp_sel)[0]
        first_unrew_resp = np.where(~reward_tone & resp_sel)[0]
        if len(first_rew_resp) > 0:
            post_rew_resp[first_rew_resp[0]+1:] = True
        if len(first_unrew_resp) > 0:
            post_unrew_resp[first_unrew_resp[0]+1:] = True

        reward = trial_data['reward_volume'].to_numpy()
        tone_vol = trial_data['tone_db_offset'].to_numpy()
        tone_rew_corr = trial_data['tone_reward_corr'].to_numpy()

        # get alignment times
        ts = sess_fp['time']
        trial_start_ts = sess_fp['trial_start_ts'][:-1]
        tone_start_ts = trial_start_ts + trial_data['abs_tone_start_time']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        resp_ts = trial_start_ts + trial_data['response_time']

        for signal_type in signal_types:
            for region in sess_fp['processed_signals'].keys():
                signal = sess_fp['processed_signals'][region][signal_type]
                region_side = implant_info[subj_id][region]['side']

                # aligned to tone
                pre = 3
                post = 5
                mat, t = fp_utils.build_signal_matrix(signal, ts, tone_start_ts, pre, post)
                align_dict = tone

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['resp'] = mat[resp_sel,:]
                align_dict[sess_id][signal_type][region]['no_resp'] = mat[no_resp_sel,:]

                align_dict[sess_id][signal_type][region]['prev_resp'] = mat[prev_resp,:]
                align_dict[sess_id][signal_type][region]['prev_no_resp'] = mat[prev_no_resp,:]

                align_dict[sess_id][signal_type][region]['resp_prev_resp'] = mat[resp_sel & prev_resp,:]
                align_dict[sess_id][signal_type][region]['resp_prev_no_resp'] = mat[resp_sel & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_prev_resp'] = mat[no_resp_sel & prev_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_prev_no_resp'] = mat[no_resp_sel & prev_no_resp,:]

                align_dict[sess_id][signal_type][region]['reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]

                align_dict[sess_id][signal_type][region]['resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]

                align_dict[sess_id][signal_type][region]['resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = resp_port == side

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_resp'] = mat[resp_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp'] = mat[no_resp_sel & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_resp'] = mat[prev_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_no_resp'] = mat[prev_no_resp & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region][side_type+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region][side_type+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]

                for v in tone_vols:
                    vol_sel = tone_vol == v

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)] = mat[vol_sel,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp'] = mat[resp_sel & vol_sel,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp'] = mat[no_resp_sel & vol_sel,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)] = mat[vol_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_resp'] = mat[resp_sel & vol_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_no_resp'] = mat[no_resp_sel & vol_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & side_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & side_sel & post_unrew_resp,:]

                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)] = mat[vol_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp'] = mat[resp_sel & vol_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp'] = mat[no_resp_sel & vol_sel & corr_sel,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]

                for r in rewards:
                    rew_sel = reward == r

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)] = mat[rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_resp'] = mat[resp_sel & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_no_resp'] = mat[no_resp_sel & rew_sel,:]

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)] = mat[rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_resp'] = mat[resp_sel & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_no_resp'] = mat[no_resp_sel & rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & side_sel & post_rew_resp,:]

                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)] = mat[rew_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_resp'] = mat[resp_sel & rew_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_no_resp'] = mat[no_resp_sel & rew_sel & corr_sel,:]

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                    for v in tone_vols:
                        vol_sel = tone_vol == v

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_vol_'+str(v)] = mat[rew_sel & vol_sel & ~averse_outcome & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_vol_'+str(v)+'_averse'] = mat[rew_sel & vol_sel & averse_outcome & post_rew_resp,:]


                # aligned to response cue
                pre = 3
                post = 3
                mat, t = fp_utils.build_signal_matrix(signal, ts, cue_ts, pre, post)
                align_dict = cue

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['resp'] = mat[resp_sel,:]
                align_dict[sess_id][signal_type][region]['no_resp'] = mat[no_resp_sel,:]

                align_dict[sess_id][signal_type][region]['prev_resp'] = mat[prev_resp,:]
                align_dict[sess_id][signal_type][region]['prev_no_resp'] = mat[prev_no_resp,:]

                align_dict[sess_id][signal_type][region]['resp_prev_resp'] = mat[resp_sel & prev_resp,:]
                align_dict[sess_id][signal_type][region]['resp_prev_no_resp'] = mat[resp_sel & prev_no_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_prev_resp'] = mat[no_resp_sel & prev_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_prev_no_resp'] = mat[no_resp_sel & prev_no_resp,:]

                align_dict[sess_id][signal_type][region]['reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]

                align_dict[sess_id][signal_type][region]['resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]

                align_dict[sess_id][signal_type][region]['resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = resp_port == side

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_resp'] = mat[resp_sel & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp'] = mat[no_resp_sel & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_resp'] = mat[prev_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_no_resp'] = mat[prev_no_resp & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region][side_type+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region][side_type+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]

                for v in tone_vols:
                    vol_sel = tone_vol == v

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)] = mat[vol_sel,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp'] = mat[resp_sel & vol_sel,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp'] = mat[no_resp_sel & vol_sel,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)] = mat[vol_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_resp'] = mat[resp_sel & vol_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_no_resp'] = mat[no_resp_sel & vol_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & side_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & side_sel & post_unrew_resp,:]


                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)] = mat[vol_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp'] = mat[resp_sel & vol_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp'] = mat[no_resp_sel & vol_sel & corr_sel,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_unreward_tone'] = mat[no_resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_resp_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_no_resp_unreward_tone_averse'] = mat[no_resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]


                for r in rewards:
                    rew_sel = reward == r

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)] = mat[rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_resp'] = mat[resp_sel & rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_no_resp'] = mat[no_resp_sel & rew_sel,:]

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)] = mat[rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_resp'] = mat[resp_sel & rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_no_resp'] = mat[no_resp_sel & rew_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & side_sel & post_rew_resp,:]


                    #TODO: have all reward tones together in addition to splitting out by averse or not
                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)] = mat[rew_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_resp'] = mat[resp_sel & rew_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_no_resp'] = mat[no_resp_sel & rew_sel & corr_sel,:]

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_resp_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_no_resp_reward_tone'] = mat[no_resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_resp_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_no_resp_reward_tone_averse'] = mat[no_resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                    for v in tone_vols:
                        vol_sel = tone_vol == v

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_vol_'+str(v)] = mat[rew_sel & vol_sel & ~averse_outcome & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_vol_'+str(v)+'_averse'] = mat[rew_sel & vol_sel & averse_outcome & post_rew_resp,:]


                # aligned to response poke
                pre = 3
                post = 10
                mat, t = fp_utils.build_signal_matrix(signal, ts, resp_ts, pre, post)
                align_dict = resp

                align_dict['t'] = t
                align_dict[sess_id][signal_type][region]['prev_resp'] = mat[prev_resp,:]
                align_dict[sess_id][signal_type][region]['prev_no_resp'] = mat[prev_no_resp,:]

                align_dict[sess_id][signal_type][region]['reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = resp_port == side

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_resp'] = mat[prev_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_no_resp'] = mat[prev_no_resp & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]

                for v in tone_vols:
                    vol_sel = tone_vol == v

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)] = mat[vol_sel,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)] = mat[vol_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & side_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & side_sel & post_unrew_resp,:]

                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)] = mat[vol_sel & corr_sel,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]


                for r in rewards:
                    rew_sel = reward == r

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)] = mat[rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)] = mat[rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & side_sel & post_rew_resp,:]

                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)] = mat[rew_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]

                    for v in tone_vols:
                        vol_sel = tone_vol == v

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_vol_'+str(v)] = mat[rew_sel & vol_sel & ~averse_outcome & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_vol_'+str(v)+'_averse'] = mat[rew_sel & vol_sel & averse_outcome & post_rew_resp,:]


                # time normalized signal matrices
                mat = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, resp_ts, norm_cue_resp_bins)
                align_dict = cue_resp

                align_dict['t'] = np.linspace(0, 1, norm_cue_resp_bins)
                align_dict[sess_id][signal_type][region]['prev_resp'] = mat[prev_resp,:]
                align_dict[sess_id][signal_type][region]['prev_no_resp'] = mat[prev_no_resp,:]

                align_dict[sess_id][signal_type][region]['reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & post_unrew_resp,:]
                align_dict[sess_id][signal_type][region]['reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & post_rew_resp,:]
                align_dict[sess_id][signal_type][region]['unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & post_unrew_resp,:]

                for side in sides:
                    side_type = fpah.get_implant_side_type(side, region_side)
                    side_sel = resp_port == side

                    align_dict[sess_id][signal_type][region][side_type] = mat[side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_prev_resp'] = mat[prev_resp & side_sel,:]
                    align_dict[sess_id][signal_type][region][side_type+'_prev_no_resp'] = mat[prev_no_resp & side_sel,:]

                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & side_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & side_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region][side_type+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & side_sel & post_unrew_resp,:]

                for v in tone_vols:
                    vol_sel = tone_vol == v

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)] = mat[vol_sel,:]

                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & post_unrew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & post_unrew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)] = mat[vol_sel & side_sel,:]

                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & side_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_vol_'+str(v)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & side_sel & post_unrew_resp,:]

                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)] = mat[vol_sel & corr_sel,:]

                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone'] = mat[resp_sel & ~reward_tone & ~averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & vol_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['vol_'+str(v)+'_corr_'+str(c)+'_unreward_tone_averse'] = mat[resp_sel & ~reward_tone & averse_outcome & vol_sel & corr_sel & post_unrew_resp,:]


                for r in rewards:
                    rew_sel = reward == r

                    align_dict[sess_id][signal_type][region]['rew_'+str(r)] = mat[rew_sel,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & post_rew_resp,:]
                    align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & post_rew_resp,:]

                    for side in sides:
                        side_type = fpah.get_implant_side_type(side, region_side)
                        side_sel = resp_port == side

                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)] = mat[rew_sel & side_sel,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & side_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region][side_type+'_rew_'+str(r)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & side_sel & post_rew_resp,:]

                    for c in tone_rew_corrs:
                        corr_sel = tone_rew_corr == c

                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)] = mat[rew_sel & corr_sel,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone'] = mat[resp_sel & reward_tone & ~averse_outcome & rew_sel & corr_sel & post_rew_resp,:]
                        align_dict[sess_id][signal_type][region]['rew_'+str(r)+'_corr_'+str(c)+'_reward_tone_averse'] = mat[resp_sel & reward_tone & averse_outcome & rew_sel & corr_sel & post_rew_resp,:]


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
tone_xlims = {'DMS': [-1.5,2], 'PL': [-3,4]}
resp_xlims = {'DMS': [-1.5,2], 'PL': [-3,10]}
gen_xlims = {'DMS': [-1.5,1.5], 'PL': [-3,3]}

tone_end = 1 # 0.5 #
reward_time = 0.5 # None #

save_plots = True
show_plots = False

# make this wrapper to simplify the stack command by not having to include the options declared above
def stack_mats(mat_dict, groups=None):
    return fpah.stack_fp_mats(mat_dict, regions, sess_ids, subjects, signal_type, filter_outliers, outlier_thresh, groups)

tone_mats = stack_mats(tone)
cue_mats = stack_mats(cue)
resp_mats = stack_mats(resp)
cue_resp_mats = stack_mats(cue_resp)

all_mats = {Align.tone: tone_mats, Align.cue: cue_mats, Align.resp: resp_mats, Align.cue_resp: cue_resp_mats}

all_ts = {Align.tone: tone['t'], Align.cue: cue['t'], Align.resp: resp['t'], Align.cue_resp: cue_resp['t']}

all_xlims = {Align.tone: tone_xlims, Align.cue: gen_xlims, Align.resp: resp_xlims, Align.cue_resp: None}

all_dashlines = {Align.tone: tone_end, Align.cue: None, Align.resp: reward_time, Align.cue_resp: None}

left_left = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper left'}}
left_right = {'DMS': {'loc': 'upper left'}, 'PL': {'loc': 'upper right'}}
right_left = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper left'}}
right_right = {'DMS': {'loc': 'upper right'}, 'PL': {'loc': 'upper right'}}
all_legend_params = {Align.tone: right_left, Align.cue: left_left, Align.resp: left_right, Align.cue_resp: right_left}

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

# %% Response, Side, and Tone Types

# response and side
plot_groups = [['resp', 'no_resp'],
               ['contra', 'ipsi'],
               ['contra_resp', 'contra_no_resp', 'ipsi_resp', 'ipsi_no_resp']]
group_labels = {'resp': 'Response', 'no_resp': 'No Response', 'contra': 'Contra',
                'ipsi': 'Ipsi', 'contra_resp': 'Contra Resp', 'contra_no_resp': 'Contra No Resp',
                'ipsi_resp': 'Ipsi Resp', 'ipsi_no_resp': 'Ipsi No Resp',}
plot_titles = ['Choice', 'Response Port Side', 'Response Port Side & Choice']
gen_title = 'Choice by Response Port Side Aligned to {}'
gen_plot_name = '{}_choice_side'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone type and side
plot_groups = [['reward_tone', 'unreward_tone'],
               ['contra_reward_tone', 'contra_unreward_tone', 'ipsi_reward_tone', 'ipsi_unreward_tone']]
group_labels = {'reward_tone': 'Rewarding', 'unreward_tone': 'Unrewarding',
                'contra_reward_tone': 'Contra Rew', 'contra_unreward_tone': 'Contra Unrew',
                'ipsi_reward_tone': 'Ipsi Rew', 'ipsi_unreward_tone': 'Ipsi Unrew'}
plot_titles = ['Tone Type', 'Response Port Side & Tone Type']
gen_title = 'Tone Type by Response Port Side Aligned to {}'
gen_plot_name = '{}_tone_type_side'

aligns = [Align.tone, Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone type, response, and side
plot_groups = [['resp_reward_tone', 'resp_unreward_tone', 'no_resp_reward_tone', 'no_resp_unreward_tone'],
               ['contra_resp_reward_tone', 'contra_resp_unreward_tone', 'contra_no_resp_reward_tone', 'contra_no_resp_unreward_tone'],
               ['ipsi_resp_reward_tone', 'ipsi_resp_unreward_tone', 'ipsi_no_resp_reward_tone', 'ipsi_no_resp_unreward_tone']]
group_labels = {'resp_reward_tone': 'Rew, Resp', 'resp_unreward_tone': 'Unrew, Resp',
                'no_resp_reward_tone': 'Rew, No Resp', 'no_resp_unreward_tone': 'Unrew, No Resp',
                'contra_resp_reward_tone': 'Rew, Resp', 'contra_resp_unreward_tone': 'Unrew, Resp',
                'contra_no_resp_reward_tone': 'Rew, No Resp', 'contra_no_resp_unreward_tone': 'Unrew, No Resp',
                'ipsi_resp_reward_tone': 'Rew, Resp', 'ipsi_resp_unreward_tone': 'Unrew, Resp',
                'ipsi_no_resp_reward_tone': 'Rew, No Resp', 'ipsi_no_resp_unreward_tone': 'Unrew, No Resp'}

plot_titles = ['All Port Sides', 'Contra Response Port', 'Ipsi Response Port']
gen_title = 'Tone Type & Choice by Response Port Side Aligned to {}'
gen_plot_name = '{}_tone_type_choice_side'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Aversive tone types

# tone type and side
plot_groups = [['reward_tone_averse', 'unreward_tone_averse'],
               ['contra_reward_tone_averse', 'contra_unreward_tone_averse', 'ipsi_reward_tone_averse', 'ipsi_unreward_tone_averse']]
group_labels = {'reward_tone_averse': 'Rewarding', 'unreward_tone_averse': 'Unrewarding',
                'contra_reward_tone_averse': 'Contra Rew', 'contra_unreward_tone_averse': 'Contra Unrew',
                'ipsi_reward_tone_averse': 'Ipsi Rew', 'ipsi_unreward_tone_averse': 'Ipsi Unrew'}
plot_titles = ['Tone Type', 'Response Port Side & Tone Type']
gen_title = 'Tone Type by Response Port Side for Aversive Blocks Aligned to {}'
gen_plot_name = '{}_tone_type_side_averse'

aligns = [Align.tone, Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# Averse and non-averse on same axes
plot_groups = [['reward_tone', 'reward_tone_averse', 'unreward_tone', 'unreward_tone_averse'],
               ['contra_reward_tone', 'contra_reward_tone_averse', 'contra_unreward_tone', 'contra_unreward_tone_averse'],
               ['ipsi_reward_tone', 'ipsi_reward_tone_averse', 'ipsi_unreward_tone', 'ipsi_unreward_tone_averse']]
group_labels = {'reward_tone': 'Rewarding', 'unreward_tone': 'Unrewarding',
                'reward_tone_averse': 'Rewarding Averse', 'unreward_tone_averse': 'Unrewarding Averse',
                'contra_reward_tone': 'Rewarding', 'contra_unreward_tone': 'Unrewarding',
                'contra_reward_tone_averse': 'Rewarding Averse', 'contra_unreward_tone_averse': 'Unrewarding Averse',
                'ipsi_reward_tone': 'Rewarding', 'ipsi_unreward_tone': 'Unrewarding',
                'ipsi_reward_tone_averse': 'Rewarding Averse', 'ipsi_unreward_tone_averse': 'Unrewarding Averse'}
plot_titles = ['Tone Type', 'Contra Response Port & Tone Type', 'Ipsi Response Port & Tone Type']
gen_title = 'Tone Type by Response Port Side & Type of Block Aligned to {}'
gen_plot_name = '{}_tone_type_side_both_conditions'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone type, response, and side
plot_groups = [['resp_reward_tone_averse', 'resp_unreward_tone_averse', 'no_resp_reward_tone_averse', 'no_resp_unreward_tone_averse'],
               ['contra_resp_reward_tone_averse', 'contra_resp_unreward_tone_averse', 'contra_no_resp_reward_tone_averse', 'contra_no_resp_unreward_tone_averse'],
               ['ipsi_resp_reward_tone_averse', 'ipsi_resp_unreward_tone_averse', 'ipsi_no_resp_reward_tone_averse', 'ipsi_no_resp_unreward_tone_averse']]
group_labels = {'resp_reward_tone_averse': 'Rew, Resp', 'resp_unreward_tone_averse': 'Unrew, Resp',
                'no_resp_reward_tone_averse': 'Rew, No Resp', 'no_resp_unreward_tone_averse': 'Unrew, No Resp',
                'contra_resp_reward_tone_averse': 'Rew, Resp', 'contra_resp_unreward_tone_averse': 'Unrew, Resp',
                'contra_no_resp_reward_tone_averse': 'Rew, No Resp', 'contra_no_resp_unreward_tone_averse': 'Unrew, No Resp',
                'ipsi_resp_reward_tone_averse': 'Rew, Resp', 'ipsi_resp_unreward_tone_averse': 'Unrew, Resp',
                'ipsi_no_resp_reward_tone_averse': 'Rew, No Resp', 'ipsi_no_resp_unreward_tone_averse': 'Unrew, No Resp'}

plot_titles = ['All Port Sides', 'Contra Response Port', 'Ipsi Response Port']
gen_title = 'Tone Type & Choice by Response Port Side for Aversive Blocks Aligned to {}'
gen_plot_name = '{}_tone_type_choice_side_averse'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Tone Volumes

# tone volume and choice
gen_plot_groups = ['vol_{}', 'vol_{}_resp', 'vol_{}_no_resp']
plot_groups = [[group.format(v) for v in tone_vols] for group in gen_plot_groups]
group_labels = {group.format(v): '{} dB'.format(v) for group in gen_plot_groups for v in tone_vols}

plot_titles = ['All Choices', 'Responses', 'No Responses']
gen_title = 'Tone Volume Offset for All Tones by Choice Aligned to {}'
gen_plot_name = '{}_tone_vol_choice_by_choice'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone volume and choice
gen_plot_groups = ['vol_{}_resp', 'vol_{}_no_resp']
plot_groups = [[group.format(v) for group in gen_plot_groups] for v in tone_vols]
gen_group_labels = {'vol_{}_resp': 'Response', 'vol_{}_no_resp': 'No Response'}
group_labels = {group.format(v): label for group, label in gen_group_labels.items() for v in tone_vols}

plot_titles = ['{} dB Offset'.format(v) for v in tone_vols]
gen_title = 'Choice by Tone Volume Offset Aligned to {}'
gen_plot_name = '{}_tone_vol_choice_by_offset'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone volume & type
gen_plot_groups = ['vol_{}_reward_tone', 'vol_{}_unreward_tone']
plot_groups = [[group.format(v) for group in gen_plot_groups] for v in tone_vols]
gen_group_labels = {'vol_{}_reward_tone': 'Rewarding', 'vol_{}_unreward_tone': 'Unrewarding'}
group_labels = {group.format(v): label for group, label in gen_group_labels.items() for v in tone_vols}

plot_titles = ['{} dB Offset'.format(v) for v in tone_vols]
gen_title = 'Tone Type by Tone Volume Offset Aligned to {}'
gen_plot_name = '{}_tone_vol_tone_type'

aligns = [Align.tone, Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone volume & type and choice
gen_plot_groups = ['vol_{}_resp_reward_tone', 'vol_{}_no_resp_reward_tone', 'vol_{}_resp_unreward_tone', 'vol_{}_no_resp_unreward_tone']
plot_groups = [[group.format(v) for group in gen_plot_groups] for v in tone_vols]
gen_group_labels = {'vol_{}_resp_reward_tone': 'Rew, Resp', 'vol_{}_no_resp_reward_tone': 'Rew, No Resp',
                    'vol_{}_resp_unreward_tone': 'Unrew, Resp', 'vol_{}_no_resp_unreward_tone': 'Unrew, No Resp'}
group_labels = {group.format(v): label for group, label in gen_group_labels.items() for v in tone_vols}

plot_titles = ['{} dB Offset'.format(v) for v in tone_vols]
gen_title = 'Tone Type & Choice by Tone Volume Offset Aligned to {}'
gen_plot_name = '{}_tone_vol_tone_type_choice'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone volume & side
gen_plot_groups = ['contra_vol_{}', 'ipsi_vol_{}']
plot_groups = [[group.format(v) for group in gen_plot_groups] for v in tone_vols]
gen_group_labels = {'contra_vol_{}': 'Contra', 'ipsi_vol_{}': 'Ipsi'}
group_labels = {group.format(v): label for group, label in gen_group_labels.items() for v in tone_vols}

plot_titles = ['{} dB Offset'.format(v) for v in tone_vols]
gen_title = 'Response Port Side by Tone Volume Offset Aligned to {}'
gen_plot_name = '{}_tone_vol_side'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# TODO: all of the below for responses only
# tone volume & tone/reward correlation
gen_plot_group = 'vol_{}_corr_{}'
plot_groups = [[gen_plot_group.format(v,c) for v in tone_vols] for c in tone_rew_corrs]
group_labels = {gen_plot_group.format(v,c): '{} dB'.format(v) for v in tone_vols for c in tone_rew_corrs}

plot_titles = ['{} Correlation'.format(c) for c in tone_rew_corrs]
gen_title = 'Tone Volume Offset by Tone Offset/Reward Volume Correlation Aligned to {}'
gen_plot_name = '{}_tone_vol_corr'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# rewarding tone volume & tone/reward correlation
gen_plot_group = 'vol_{}_corr_{}_reward_tone'
plot_groups = [[gen_plot_group.format(v,c) for v in tone_vols] for c in tone_rew_corrs]
group_labels = {gen_plot_group.format(v,c): '{} dB'.format(v) for v in tone_vols for c in tone_rew_corrs}

plot_titles = ['{} Correlation'.format(c) for c in tone_rew_corrs]
gen_title = 'Rewarding Tone Volume Offset by Tone Offset/Reward Volume Correlation Aligned to {}'
gen_plot_name = '{}_tone_vol_corr_rewarding'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# unrewarding tone volume & tone/reward correlation
gen_plot_group = 'vol_{}_corr_{}_unreward_tone'
plot_groups = [[gen_plot_group.format(v,c) for v in tone_vols] for c in tone_rew_corrs]
group_labels = {gen_plot_group.format(v,c): '{} dB'.format(v) for v in tone_vols for c in tone_rew_corrs}

plot_titles = ['{} Correlation'.format(c) for c in tone_rew_corrs]
gen_title = 'Unrewarding Tone Volume Offset by Tone Offset/Reward Volume Correlation Aligned to {}'
gen_plot_name = '{}_tone_vol_corr_unrewarding'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Aversive Tone Volumes

# tone volume & type
gen_plot_groups = ['vol_{}_reward_tone_averse', 'vol_{}_unreward_tone_averse']
plot_groups = [[group.format(v) for group in gen_plot_groups] for v in tone_vols]
gen_group_labels = {'vol_{}_reward_tone_averse': 'Rewarding', 'vol_{}_unreward_tone_averse': 'Unrewarding'}
group_labels = {group.format(v): label for group, label in gen_group_labels.items() for v in tone_vols}

plot_titles = ['{} dB Offset'.format(v) for v in tone_vols]
gen_title = 'Tone Type by Tone Volume Offset for Aversive Blocks Aligned to {}'
gen_plot_name = '{}_tone_vol_tone_type_averse'

aligns = [Align.tone, Align.cue, Align.resp, Align.cue_resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# tone volume & type and choice
gen_plot_groups = ['vol_{}_resp_reward_tone_averse', 'vol_{}_no_resp_reward_tone_averse', 'vol_{}_resp_unreward_tone_averse', 'vol_{}_no_resp_unreward_tone_averse']
plot_groups = [[group.format(v) for group in gen_plot_groups] for v in tone_vols]
gen_group_labels = {'vol_{}_resp_reward_tone_averse': 'Rew, Resp', 'vol_{}_no_resp_reward_tone_averse': 'Rew, No Resp',
                    'vol_{}_resp_unreward_tone_averse': 'Unrew, Resp', 'vol_{}_no_resp_unreward_tone_averse': 'Unrew, No Resp'}
group_labels = {group.format(v): label for group, label in gen_group_labels.items() for v in tone_vols}

plot_titles = ['{} dB Offset'.format(v) for v in tone_vols]
gen_title = 'Tone Type & Choice by Tone Volume Offset for Aversive Blocks Aligned to {}'
gen_plot_name = '{}_tone_vol_tone_type_choice_averse'

aligns = [Align.tone, Align.cue]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# %% Reward Volume

# reward volume & side
gen_plot_groups = ['rew_{}_reward_tone', 'contra_rew_{}_reward_tone', 'ipsi_rew_{}_reward_tone']
plot_groups = [[group.format(r) for r in rewards] for group in gen_plot_groups]
group_labels = {group.format(r): '{} μL'.format(r) for group in gen_plot_groups for r in rewards}

plot_titles = ['All Port Sides', 'Contra Response Port', 'Ipsi Response Port']
gen_title = 'Reward Volume by Response Port Side Aligned to {}'
gen_plot_name = '{}_rew_side'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


# reward & tone volume
gen_plot_group = 'rew_{}_vol_{}'
plot_groups = [[gen_plot_group.format(r,v) for v in tone_vols] for r in rewards]
group_labels = {gen_plot_group.format(r,v): '{} dB'.format(v) for v in tone_vols for r in rewards}

plot_titles = ['{} μL'.format(r) for r in rewards]
gen_title = 'Tone Volume Offset by Reward Volume Aligned to {}'
gen_plot_name = '{}_rew_tone_vol_by_rew'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)


gen_plot_group = 'rew_{}_vol_{}'
plot_groups = [[gen_plot_group.format(r,v) for r in rewards] for v in tone_vols]
group_labels = {gen_plot_group.format(r,v): '{} μL'.format(r) for v in tone_vols for r in rewards}

plot_titles = ['{} dB'.format(v) for v in tone_vols]
gen_title = 'Reward Volume by Tone Volume Offset Aligned to {}'
gen_plot_name = '{}_rew_tone_vol_by_vol'

aligns = [Align.tone, Align.cue, Align.resp]

for align in aligns:
    plot_avg_signals(align, plot_groups, group_labels, plot_titles, gen_title, gen_plot_name=gen_plot_name)
