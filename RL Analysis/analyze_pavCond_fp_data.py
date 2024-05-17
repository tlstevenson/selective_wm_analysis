# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:33:24 2023

@author: tanne
"""

# %% imports

import init
import os.path as path
import pandas as pd
from pyutils import utils
from sys_neuro_tools import plot_utils, fp_utils
import hankslab_db.basicRLtasks_db as db
import fp_analysis_helpers as fpah
import numpy as np
import matplotlib.pyplot as plt
import peakutils
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import pickle
import copy

# %% Load behavior data

sess_ids = [94104] #93580, 94072, 94104, 95035

loc_db = db.LocalDB_BasicRLTasks('pavlovCond') # 
sess_data = loc_db.get_behavior_data(sess_ids) # reload=True

# %% Load fiber photometry data

data_path = path.join(utils.get_user_home(), 'fp_data')
fp_data = {}
for sess_id in sess_ids:
    load_path = path.join(data_path, 'Session_{}.pkl'.format(sess_id))
    with open(load_path, 'rb') as f:
        fp_data[sess_id] = pickle.load(f)

# %% Process photometry data in different ways

regions = ['DMS', 'PFC'] #

#iso_pfc = 'PFC_405'
#iso_dms = 'DMS_405'
iso_pfc = 'PFC_420'
iso_dms = 'DMS_420'
lig_pfc = 'PFC_490'
lig_dms = 'DMS_490'

signals_by_region = {region: {'iso': iso_pfc if region == 'PFC' else iso_dms,
                              'lig': lig_pfc if region == 'PFC' else lig_dms}
                     for region in regions}

for sess_id in sess_ids:
    signals = fp_data[sess_id]['signals']

    fp_data[sess_id]['processed_signals'] = {}

    for region, signal_names in signals_by_region.items():
        raw_lig = signals[signal_names['lig']]
        raw_iso = signals[signal_names['iso']]

        fp_data[sess_id]['processed_signals'][region] = fpah.get_all_processed_signals(raw_lig, raw_iso)

# %% Observe the full signals

for sess_id in sess_ids:
    trial_data = sess_data[sess_data['sessid'] == sess_id]
    sess_fp = fp_data[sess_id]
    
    # Get the block transition trial start times
    trial_start_ts = sess_fp['trial_start_ts'][:-1]
    block_start_times = trial_start_ts[trial_data['block_trial'] == 1]
    block_rewards = trial_data['reward_volume'][trial_data['block_trial'] == 1]
    
    # fpah.view_processed_signals(sess_fp['processed_signals'], sess_fp['time'], 
    #                             title='Full Signals - Session {}. Block Rewards: {}'.format(sess_id, ', '.join([str(r) for r in block_rewards])), 
    #                             vert_marks=block_start_times, filter_outliers=True)
    
    fpah.view_signal(sess_fp['processed_signals'], sess_fp['time'], 'dff_iso', t_min=1950, t_max=1950+120, 
                     figsize=None, ylabel='% dF/F', dec=20)

# %% Average signal to the tone presentations, and response, all split by trial outcome

signal_types = ['dff_iso'] # 'baseline_corr_lig','baseline_corr_iso',  'dff_iso', 'z_dff_iso', 'dff_baseline', 'z_dff_baseline', 'df_baseline_iso', 'z_df_baseline_iso'

# whether to nomarlize aligned signals on a trial-by-trial basis to the average of the signal in the window before poking in
trial_normalize = False

title_suffix = '' #'420 Iso'

outlier_thresh = 8 # z-score threshold

def create_mat(signal, ts, align_ts, pre, post, windows, sel=[]):
    if trial_normalize:
        return fp_utils.build_trial_dff_signal_matrix(signal, ts, align_ts, pre, post, windows, sel)
    else:
        return fp_utils.build_signal_matrix(signal, ts, align_ts, pre, post, sel)

for sess_id in sess_ids:
    trial_data = sess_data[sess_data['sessid'] == sess_id]
    sess_fp = fp_data[sess_id]

    # get alignment trial filters
    resp_sel = trial_data['hit'] == True
    noresp_sel = trial_data['hit'] == False
    rewarded_sel = trial_data['reward'] > 0
    reward_tone_sel = trial_data['reward_tone']
    resp_reward_sel = resp_sel & rewarded_sel
    resp_noreward_sel = resp_sel & ~rewarded_sel

    # get alignment times
    ts = sess_fp['time']
    trial_start_ts = sess_fp['trial_start_ts'][:-1]
    abs_tone_ts = trial_start_ts + trial_data['abs_tone_start_time']
    abs_cue_ts = trial_start_ts + trial_data['response_cue_time']
    abs_resp_ts = trial_start_ts + trial_data['response_time']
    # for trial signal normalization, choose range 0.5-1.5s after the trial starts
    # there is a 2s ITI between response and trial start and the earliest a tone can be played is 2s after the trial starts
    trial_norm_windows = trial_start_ts[:, None] + np.array([0.5, 1.5])

    # get other important information
    tone_vols = np.unique(trial_data['tone_db_offset'])
    reward_vols = np.unique(trial_data['reward_volume'])

    # compute block transition indices for each trial filter
    first_block_trial_idxs = np.where(trial_data['block_trial'] == 1)[0]
    def find_transition_idxs(trial_sel):
        trial_sel_idxs = np.where(trial_sel)[0]
        return np.unique([np.argmax(t_idx < trial_sel_idxs) for t_idx in first_block_trial_idxs])

    block_trans_idxs = {}
    block_trans_idxs['resp_reward'] = find_transition_idxs(resp_reward_sel)
    block_trans_idxs['resp_noreward'] = find_transition_idxs(resp_noreward_sel)
    block_trans_idxs['noresp'] = find_transition_idxs(noresp_sel)
    for reward_vol in reward_vols:
        reward_vol_sel = trial_data['reward_volume'] == reward_vol
        block_trans_idxs['resp_reward_vol' + str(reward_vol)] = find_transition_idxs(resp_reward_sel & reward_vol_sel)
        block_trans_idxs['resp_noreward_vol' + str(reward_vol)] = find_transition_idxs(resp_noreward_sel & reward_vol_sel)

    for tone_vol in tone_vols:
        tone_vol_sel = trial_data['tone_db_offset'] == tone_vol
        block_trans_idxs['resp_reward_dB' + str(tone_vol)] = find_transition_idxs(resp_reward_sel & tone_vol_sel)
        block_trans_idxs['resp_noreward_dB' + str(tone_vol)] = find_transition_idxs(resp_noreward_sel & tone_vol_sel)

    for signal_type in signal_types:

        data_dict = {region: {} for region in regions}
        tone_outcome = copy.deepcopy(data_dict)
        #tone_prev_outcome = copy.deepcopy(data_dict)
        cue_outcome = copy.deepcopy(data_dict)
        #cue_prev_outcome = copy.deepcopy(data_dict)
        response = copy.deepcopy(data_dict)
        tone_reward_vol = copy.deepcopy(data_dict)
        cue_reward_vol = copy.deepcopy(data_dict)
        resp_reward_vol = copy.deepcopy(data_dict)
        tone_tone_vol = copy.deepcopy(data_dict)

        for region in regions:
            signal = sess_fp['processed_signals'][region][signal_type]

            # aligned to tone by trial outcome
            pre = 1
            post = 3

            norm_windows_tone = trial_norm_windows - abs_tone_ts.to_numpy()[:, None]
            tone_outcome[region]['resp_reward'], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, resp_reward_sel)
            tone_outcome[region]['resp_noreward'], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, resp_noreward_sel)
            tone_outcome[region]['noresp'], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, noresp_sel)
            tone_outcome['t'] = t

            # aligned to cue by trial outcome
            pre = 2
            post = 3

            norm_windows_cue = trial_norm_windows - abs_cue_ts.to_numpy()[:, None]
            cue_outcome[region]['resp_reward'], t = create_mat(signal, ts, abs_cue_ts, pre, post, norm_windows_cue, resp_reward_sel)
            cue_outcome[region]['resp_noreward'], t = create_mat(signal, ts, abs_cue_ts, pre, post, norm_windows_cue, resp_noreward_sel)
            cue_outcome[region]['noresp'], t = create_mat(signal, ts, abs_cue_ts, pre, post, norm_windows_cue, noresp_sel)
            cue_outcome['t'] = t

            # aligned to response by trial outcome
            pre = 2
            post = 3

            norm_windows_resp = trial_norm_windows - abs_resp_ts.to_numpy()[:, None]
            response[region]['resp_reward'], t = create_mat(signal, ts, abs_resp_ts, pre, post, norm_windows_resp, resp_reward_sel)
            response[region]['resp_noreward'], t = create_mat(signal, ts, abs_resp_ts, pre, post, norm_windows_resp, resp_noreward_sel)
            response['t'] = t
            
            # aligned to response by trial outcome
            pre = 2
            post = 3

            norm_windows_resp = trial_norm_windows - abs_resp_ts.to_numpy()[:, None]
            response[region]['resp_reward'], t = create_mat(signal, ts, abs_resp_ts, pre, post, norm_windows_resp, resp_reward_sel)
            response[region]['resp_noreward'], t = create_mat(signal, ts, abs_resp_ts, pre, post, norm_windows_resp, resp_noreward_sel)
            response['t'] = t

            # aligned to tone sorted by reward volumes and outcome
            pre = 1
            post = 3

            for reward_vol in reward_vols:
                reward_vol_sel = trial_data['reward_volume'] == reward_vol
                tone_reward_vol[region]['resp_reward_vol' + str(reward_vol)], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, resp_reward_sel & reward_vol_sel)
                tone_reward_vol[region]['resp_noreward_vol' + str(reward_vol)], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, resp_noreward_sel & reward_vol_sel)

            tone_reward_vol['t'] = t
            
            # aligned to cue sorted by reward volumes and outcome
            pre = 2
            post = 2

            for reward_vol in reward_vols:
                reward_vol_sel = trial_data['reward_volume'] == reward_vol
                cue_reward_vol[region]['resp_reward_vol' + str(reward_vol)], t = create_mat(signal, ts, abs_cue_ts, pre, post, norm_windows_cue, resp_reward_sel & reward_vol_sel)
                cue_reward_vol[region]['resp_noreward_vol' + str(reward_vol)], t = create_mat(signal, ts, abs_cue_ts, pre, post, norm_windows_cue, resp_noreward_sel & reward_vol_sel)

            cue_reward_vol['t'] = t

            # aligned to response sorted by reward volumes and outcome
            pre = 2
            post = 2

            for reward_vol in reward_vols:
                reward_vol_sel = trial_data['reward_volume'] == reward_vol
                resp_reward_vol[region]['resp_reward_vol' + str(reward_vol)], t = create_mat(signal, ts, abs_resp_ts, pre, post, norm_windows_resp, resp_reward_sel & reward_vol_sel)
                resp_reward_vol[region]['resp_noreward_vol' + str(reward_vol)], t = create_mat(signal, ts, abs_resp_ts, pre, post, norm_windows_resp, resp_noreward_sel & reward_vol_sel)

            resp_reward_vol['t'] = t

            # aligned to tone sorted by tone volumes and outcome
            pre = 1
            post = 3

            for tone_vol in tone_vols:
                tone_vol_sel = trial_data['tone_db_offset'] == tone_vol
                tone_tone_vol[region]['resp_reward_dB' + str(tone_vol)], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, resp_reward_sel & tone_vol_sel)
                tone_tone_vol[region]['resp_noreward_dB' + str(tone_vol)], t = create_mat(signal, ts, abs_tone_ts, pre, post, norm_windows_tone, resp_noreward_sel & tone_vol_sel)

            tone_tone_vol['t'] = t

        # plot alignment results

        # get appropriate labels
        match signal_type: # noqa <-- to disable incorrect error message
            case 'dff_iso':
                signal_type_title = 'Isosbestic Normalized'
                signal_type_label = 'dF/F'
            case 'z_dff_iso':
                signal_type_title = 'Isosbestic Normalized'
                signal_type_label = 'z-scored dF/F'
            case 'dff_baseline':
                signal_type_title = 'Baseline Normalized'
                signal_type_label = 'dF/F'
            case 'z_dff_baseline':
                signal_type_title = 'Baseline Normalized'
                signal_type_label = 'z-scored dF/F'
            case 'df_baseline_iso':
                signal_type_title = 'Baseline-Subtracted Isosbestic Residuals'
                signal_type_label = 'dF residuals'
            case 'z_df_baseline_iso':
                signal_type_title = 'Baseline-Subtracted Isosbestic Residuals'
                signal_type_label = 'z-scored dF residuals'
            case 'baseline_corr_lig':
                signal_type_title = 'Baseline-Subtracted Ligand Signal'
                signal_type_label = 'dF'
            case 'baseline_corr_iso':
                signal_type_title = 'Baseline-Subtracted Isosbestic Signal'
                signal_type_label = 'dF'
            case 'raw_lig':
                signal_type_title = 'Raw Ligand Signal'
                signal_type_label = 'dF/F' if trial_normalize else 'V'
            case 'raw_iso':
                signal_type_title = 'Raw Isosbestic Signal'
                signal_type_label = 'dF/F' if trial_normalize else 'V'

        if trial_normalize:
            signal_type_title += ' - Trial Normalized'

        if title_suffix != '':
            signal_type_title += ' - ' + title_suffix

        all_sub_titles = {'resp_reward': 'Rewarded Response', 'resp_noreward': 'Unrewarded Response', 'noresp': 'No Response'}
        for reward_vol in reward_vols:
            all_sub_titles.update({'resp_reward_vol' + str(reward_vol): 'Rewarded Response - {} μL Block'.format(reward_vol),
                                  'resp_noreward_vol' + str(reward_vol): 'Unrewarded Response - {} μL Block'.format(reward_vol)})

        for tone_vol in tone_vols:
            all_sub_titles.update({'resp_reward_dB' + str(tone_vol): 'Rewarded Tone - {} dB Offset'.format(tone_vol),
                                  'resp_noreward_dB' + str(tone_vol): 'Unrewarded Tone - {} dB Offset'.format(tone_vol)})

        fpah.plot_aligned_signals(tone_outcome, 'Tone Aligned - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)

        fpah.plot_aligned_signals(cue_outcome, 'Response Cue Aligned - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)

        fpah.plot_aligned_signals(response, 'Response Aligned - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from reponse poke (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)

        fpah.plot_aligned_signals(tone_reward_vol, 'Tone Aligned, Reward Volume - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)
        
        fpah.plot_aligned_signals(cue_reward_vol, 'Response Cue Aligned, Reward Volume - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from response cue (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)

        fpah.plot_aligned_signals(resp_reward_vol, 'Response Aligned, Reward Volume - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from response poke (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)
        
        fpah.plot_aligned_signals(tone_tone_vol, 'Tone Aligned, Tone Volume - {} (session {})'.format(signal_type_title, sess_id),
                             all_sub_titles, 'Time from tone start (s)', signal_type_label, outlier_thresh=outlier_thresh,
                             trial_markers=block_trans_idxs)


# %% Plot average signals from multiple sessions on the same axes

signal_type = 'z_dff_iso' # 'dff_iso', 'df_baseline_iso', 'raw_lig'
regions = ['DMS', 'PFC']

def stack_mat(old_mat, new_mat):
    if len(old_mat) == 0:
        return new_mat
    else:
        return np.vstack((old_mat, new_mat))
    
tone_vols = np.unique(sess_data['tone_db_offset'])
reward_vols = np.unique(sess_data['reward_volume'])

# build data matrices
response_outcome = {region: {'rewarded': [], 'unrewarded': []} for region in regions}
response_reward_vols = {region: {vol: {'rewarded': [], 'unrewarded': []} for vol in reward_vols} for region in regions}
# reward_vols_block_epoch = {region: {vol: {'early_rewarded': [], 'early_unrewarded': [], 
#                                           'late_rewarded': [], 'late_unrewarded': []} for vol in reward_vols} for region in regions}
tone_outcome = {region: {'rewarded': [], 'unrewarded': [], 'noResp': []} for region in regions}
tone_tone_vols = {region: {vol: {'rewarded': [], 'unrewarded': []} for vol in tone_vols} for region in regions}
# tone_vols_block_epoch = {region: {vol: {'early_rewarded': [], 'early_unrewarded': [], 
#                                           'late_rewarded': [], 'late_unrewarded': []} for vol in tone_vols} for region in regions}

cue_outcome = {region: {'rewarded': [], 'unrewarded': [], 'noResp': []} for region in regions}

# block_epoch_trials = 10

for sess_id in sess_ids:
    trial_data = sess_data[sess_data['sessid'] == sess_id]
    sess_fp = fp_data[sess_id]

    ts = sess_fp['time']
    trial_start_ts = sess_fp['trial_start_ts'][:-1]
    
    resp_sel = trial_data['hit'] == True
    noresp_sel = trial_data['hit'] == False
    rewarded_sel = trial_data['reward'] > 0
    reward_tone_sel = trial_data['reward_tone']
    resp_reward_sel = resp_sel & rewarded_sel
    resp_noreward_sel = resp_sel & ~rewarded_sel
    
    # get alignment times
    ts = sess_fp['time']
    trial_start_ts = sess_fp['trial_start_ts'][:-1]
    abs_tone_ts = trial_start_ts + trial_data['abs_tone_start_time']
    abs_cue_ts = trial_start_ts + trial_data['response_cue_time']
    abs_resp_ts = trial_start_ts + trial_data['response_time']

    for region in regions:
        signal = sess_fp['processed_signals'][region][signal_type]

        # aligned to response by trial outcome
        pre = 1
        post = 2

        reward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, pre, post, resp_reward_sel)
        unreward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, pre, post, resp_noreward_sel)
        response_outcome[region]['rewarded'] = stack_mat(response_outcome[region]['rewarded'], reward_mat)
        response_outcome[region]['unrewarded'] = stack_mat(response_outcome[region]['unrewarded'], unreward_mat)
        response_outcome['t'] = t

        # aligned to response by reward volume
        pre = 1
        post = 2
        for vol in reward_vols:
            reward_vol_sel = trial_data['reward_volume'] == vol
            reward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, pre, post, resp_reward_sel & reward_vol_sel)
            unreward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_resp_ts, pre, post, resp_noreward_sel & reward_vol_sel)
            response_reward_vols[region][vol]['rewarded'] = stack_mat(response_reward_vols[region][vol]['rewarded'], reward_mat)
            response_reward_vols[region][vol]['unrewarded'] = stack_mat(response_reward_vols[region][vol]['unrewarded'], unreward_mat)
        
        response_reward_vols['t'] = t
        
        # aligned to tone by trial outcome
        pre = 1
        post = 3

        reward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_tone_ts, pre, post, resp_reward_sel)
        unreward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_tone_ts, pre, post, resp_noreward_sel)
        noResp_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_tone_ts, pre, post, noresp_sel)
        tone_outcome[region]['rewarded'] = stack_mat(tone_outcome[region]['rewarded'], reward_mat)
        tone_outcome[region]['unrewarded'] = stack_mat(tone_outcome[region]['unrewarded'], unreward_mat)
        tone_outcome[region]['noResp'] = stack_mat(tone_outcome[region]['noResp'], noResp_mat)
        tone_outcome['t'] = t
        
        # aligned to tone by tone volume
        pre = 1
        post = 3
        for vol in tone_vols:
            tone_vol_sel = trial_data['tone_db_offset'] == vol
            reward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_tone_ts, pre, post, resp_reward_sel & tone_vol_sel)
            unreward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_tone_ts, pre, post, resp_noreward_sel & tone_vol_sel)
            tone_tone_vols[region][vol]['rewarded'] = stack_mat(tone_tone_vols[region][vol]['rewarded'], reward_mat)
            tone_tone_vols[region][vol]['unrewarded'] = stack_mat(tone_tone_vols[region][vol]['unrewarded'], unreward_mat)
        
        tone_tone_vols['t'] = t
        
        # aligned to response cue by trial outcome
        pre = 2
        post = 2

        reward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cue_ts, pre, post, resp_reward_sel)
        unreward_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cue_ts, pre, post, resp_noreward_sel)
        noResp_mat, t = fp_utils.build_signal_matrix(signal, ts, abs_cue_ts, pre, post, noresp_sel)
        cue_outcome[region]['rewarded'] = stack_mat(cue_outcome[region]['rewarded'], reward_mat)
        cue_outcome[region]['unrewarded'] = stack_mat(cue_outcome[region]['unrewarded'], unreward_mat)
        cue_outcome[region]['noResp'] = stack_mat(cue_outcome[region]['noResp'], noResp_mat)
        cue_outcome['t'] = t

# %% Plot Averaged Signals

filter_outliers = True
outlier_thresh = 20

def remove_outliers(mat):
    if filter_outliers:
        mat[np.abs(mat) > outlier_thresh] = np.nan
        
    return mat

use_se = True
def calc_error(mat):
    if use_se:
        return utils.stderr(mat, axis=0)
    else:
        return np.std(mat, axis=0)

# trial outcome
fig, axs = plt.subplots(2, 1, figsize=(5, 7), layout='constrained')
fig.suptitle('Response Aligned by Trial Outcome')
t = response_outcome['t']

ax = axs[0]
region = 'DMS'
ax.set_title(region)
reward_act = remove_outliers(response_outcome[region]['rewarded'])
unreward_act = remove_outliers(response_outcome[region]['unrewarded'])
plot_utils.plot_psth(np.nanmean(reward_act, axis=0), t, calc_error(reward_act), ax, label='Rewarded')
plot_utils.plot_psth(np.nanmean(unreward_act, axis=0), t, calc_error(unreward_act), ax, label='Unrewarded')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from response (s)')

ax = axs[1]
region = 'PFC'
ax.set_title(region)
reward_act = remove_outliers(response_outcome[region]['rewarded'])
unreward_act = remove_outliers(response_outcome[region]['unrewarded'])
plot_utils.plot_psth(np.nanmean(reward_act, axis=0), t, calc_error(reward_act), ax, label='Rewarded')
plot_utils.plot_psth(np.nanmean(unreward_act, axis=0), t, calc_error(unreward_act), ax, label='Unrewarded')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from response (s)')
ax.legend(loc='upper left')


# Reward volumes by outcome
fig, axs = plt.subplots(2, 2, figsize=(9, 7), layout='constrained', sharey='row')
fig.suptitle('Response Aligned by Reward Volume')
t = response_reward_vols['t']

ax = axs[0,0]
region = 'DMS'
ax.set_title(region+' - Rewarded')
for vol in reward_vols:
    act = remove_outliers(response_reward_vols[region][vol]['rewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} μL'.format(vol))
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from response (s)')

ax = axs[0,1]
ax.set_title(region+' - Unrewarded')
for vol in reward_vols:
    act = remove_outliers(response_reward_vols[region][vol]['unrewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} μL'.format(vol))
ax.set_xlabel('Time from response (s)')
ax.legend(loc='upper left')

ax = axs[1,0]
region = 'PFC'
ax.set_title(region+' - Rewarded')
for vol in reward_vols:
    act = remove_outliers(response_reward_vols[region][vol]['rewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} μL'.format(vol))
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from response (s)')

ax = axs[1,1]
ax.set_title(region+' - Unrewarded')
for vol in reward_vols:
    act = remove_outliers(response_reward_vols[region][vol]['unrewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} μL'.format(vol))
ax.set_xlabel('Time from response (s)')
ax.legend(loc='upper left')

# Tone outcome
fig, axs = plt.subplots(2, 1, figsize=(5, 7), layout='constrained')
fig.suptitle('Tone Aligned by Trial Outcome')
t = tone_outcome['t']

ax = axs[0]
region = 'DMS'
ax.set_title(region)
reward_act = remove_outliers(tone_outcome[region]['rewarded'])
unreward_act = remove_outliers(tone_outcome[region]['unrewarded'])
noResp_act = remove_outliers(tone_outcome[region]['noResp'])
plot_utils.plot_psth(np.nanmean(reward_act, axis=0), t, calc_error(reward_act), ax, label='Rewarded')
plot_utils.plot_psth(np.nanmean(unreward_act, axis=0), t, calc_error(unreward_act), ax, label='Unrewarded')
plot_utils.plot_psth(np.nanmean(noResp_act, axis=0), t, calc_error(noResp_act), ax, label='No Response')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from tone start (s)')
ax.legend(loc='upper right')

ax = axs[1]
region = 'PFC'
ax.set_title(region)
reward_act = remove_outliers(tone_outcome[region]['rewarded'])
unreward_act = remove_outliers(tone_outcome[region]['unrewarded'])
noResp_act = remove_outliers(tone_outcome[region]['noResp'])
plot_utils.plot_psth(np.nanmean(reward_act, axis=0), t, calc_error(reward_act), ax, label='Rewarded')
plot_utils.plot_psth(np.nanmean(unreward_act, axis=0), t, calc_error(unreward_act), ax, label='Unrewarded')
plot_utils.plot_psth(np.nanmean(noResp_act, axis=0), t, calc_error(noResp_act), ax, label='No Response')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from tone start (s)')



# Tone volumes by outcome
fig, axs = plt.subplots(2, 2, figsize=(9, 7), layout='constrained', sharey='row')
fig.suptitle('Tone Aligned by Tone Volume')
t = tone_tone_vols['t']

ax = axs[0,0]
region = 'DMS'
ax.set_title(region+' - Rewarded')
for vol in tone_vols:
    act = remove_outliers(tone_tone_vols[region][vol]['rewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} dB offset'.format(vol))
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from tone start (s)')
ax.legend(loc='upper right')

ax = axs[0,1]
ax.set_title(region+' - Unrewarded')
for vol in tone_vols:
    act = remove_outliers(tone_tone_vols[region][vol]['unrewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} dB offset'.format(vol))
ax.set_xlabel('Time from tone start (s)')


ax = axs[1,0]
region = 'PFC'
ax.set_title(region+' - Rewarded')
for vol in tone_vols:
    act = remove_outliers(tone_tone_vols[region][vol]['rewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} dB offset'.format(vol))
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from tone start (s)')

ax = axs[1,1]
ax.set_title(region+' - Unrewarded')
for vol in tone_vols:
    act = remove_outliers(tone_tone_vols[region][vol]['unrewarded'])
    plot_utils.plot_psth(np.nanmean(act, axis=0), t, calc_error(act), ax, label='{} dB offset'.format(vol))
ax.set_xlabel('Time from tone start (s)')
ax.legend(loc='upper left')

# Cue outcome
fig, axs = plt.subplots(2, 1, figsize=(5, 7), layout='constrained')
fig.suptitle('Response Cue Aligned by Trial Outcome')
t = cue_outcome['t']

ax = axs[0]
region = 'DMS'
ax.set_title(region)
reward_act = remove_outliers(cue_outcome[region]['rewarded'])
unreward_act = remove_outliers(cue_outcome[region]['unrewarded'])
noResp_act = remove_outliers(cue_outcome[region]['noResp'])
plot_utils.plot_psth(np.nanmean(reward_act, axis=0), t, calc_error(reward_act), ax, label='Rewarded')
plot_utils.plot_psth(np.nanmean(unreward_act, axis=0), t, calc_error(unreward_act), ax, label='Unrewarded')
plot_utils.plot_psth(np.nanmean(noResp_act, axis=0), t, calc_error(noResp_act), ax, label='No Response')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from cue (s)')
ax.legend(loc='upper right')

ax = axs[1]
region = 'PFC'
ax.set_title(region)
reward_act = remove_outliers(cue_outcome[region]['rewarded'])
unreward_act = remove_outliers(cue_outcome[region]['unrewarded'])
noResp_act = remove_outliers(cue_outcome[region]['noResp'])
plot_utils.plot_psth(np.nanmean(reward_act, axis=0), t, calc_error(reward_act), ax, label='Rewarded')
plot_utils.plot_psth(np.nanmean(unreward_act, axis=0), t, calc_error(unreward_act), ax, label='Unrewarded')
plot_utils.plot_psth(np.nanmean(noResp_act, axis=0), t, calc_error(noResp_act), ax, label='No Response')
ax.set_ylabel('Z-scored ΔF/F')
ax.set_xlabel('Time from cue (s)')
