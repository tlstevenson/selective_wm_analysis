# -*- coding: utf-8 -*-
"""
Script to investigate performance on the tone categorization task stage 8 - multi tone

@author: tanner stevenson
"""

import init

import pyutils.utils as utils
from sys_neuro_tools import plot_utils
import hankslab_db.tonecatdelayresp_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import re
import seaborn as sb

plot_timing = False
exclude_instruct_from_hitrate = True
plot_bail = False
# plot_summary = True

# %% LOAD DATA

stage = 9
n_back = 10
active_subjects_only = True

subject_info = db_access.get_active_subj_stage('ToneCatDelayResp2')
if active_subjects_only:
    subject_info = subject_info[subject_info['stage'] == stage]
else:
    subject_info = subject_info[subject_info['stage'] >= stage]

subj_ids = subject_info['subjid']
#subj_ids = [179]

# get session ids
sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_num=stage)
sess_ids = bah.limit_sess_ids(sess_ids, n_back)
# sess_ids = {180: [80585, 80542, 80497, 80452, 80407, 81250, 81205, 81160, 81115, 81070, 81025, 80980, 80933, 80876]}

# get trial information
loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids))
# remove trials where the stimulus didn't start
all_sess = all_sess[all_sess['trial_started']]

# %% Format columns for ease of aggregating and display
# make stimulus duration bins
bin_size = 1
dur_bin_max = np.ceil(np.max(all_sess['stim_dur'])/bin_size)
dur_bin_min = np.floor(np.min(all_sess['stim_dur'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

all_sess['stim_dur_bin'] = all_sess['stim_dur'].apply(
    lambda x: dur_bin_labels[np.where(x >= dur_bins)[0][-1]])

# make tone position bins
bin_size = 1
pos_bin_max = np.ceil(np.max(all_sess['rel_tone_start_times'].apply(lambda x: np.max(x)))/bin_size)
pos_bin_min = np.floor(np.min(all_sess['rel_tone_start_times'].apply(lambda x: np.min(x)))/bin_size)
pos_bins = np.arange(pos_bin_min, pos_bin_max+1)*bin_size
pos_bin_labels = ['{:.0f}-{:.0f}s'.format(pos_bins[i], pos_bins[i+1]) for i in range(len(pos_bins)-1)]

def get_tone_pos_bin_str(tone_starts):
    if utils.is_scalar(tone_starts):
        tone_starts = np.array([tone_starts])
        
    tone_bins = [pos_bin_labels[np.where(ts >= pos_bins)[0][-1]] for ts in tone_starts]
    return ', '.join(tone_bins)
        
all_sess['tone_start_bins_str'] = all_sess['rel_tone_start_times'].apply(lambda x: get_tone_pos_bin_str(x))

# reformat tone info arrays into strings to be hashable for value counting and for display
all_sess['tone_info_str'] = all_sess['tone_info'].apply(
    lambda x: x if utils.is_scalar(x) else ', '.join(x)).apply(
    lambda x: x.replace('low ', 'Lo/').replace('high ', 'Hi/').replace('left', 'L').replace('right', 'R'))
        
# flatten different tone/side combos for ease of comparing across subjects
def get_tone_side_pitch(tone_info, info_type):
    if utils.is_scalar(tone_info):
        tone_info = np.array([tone_info])
    
    match info_type:
        case 'side':
            tone_side_pitch = ['L' if 'left' in ti else 'R' if 'right' in ti else '' for ti in tone_info]
        case 'pitch':
            tone_side_pitch = ['H' if 'high' in ti else 'L' if 'low' in ti else '' for ti in tone_info]
    return ', '.join(tone_side_pitch)

all_sess['tone_info_side'] = all_sess['tone_info'].apply(lambda x: get_tone_side_pitch(x, 'side'))
all_sess['tone_info_pitch'] = all_sess['tone_info'].apply(lambda x: get_tone_side_pitch(x, 'pitch'))
all_sess['trial_type'] = all_sess['tone_info'].apply(
    lambda x: '1 tone' if (utils.is_scalar(x) or len(x) == 1) else
    '{0} tones - same'.format(len(x)) if x[0] == x[-1] else '{0} tones - diff'.format(len(x)))
# combine tone positions with stimulus durations
all_sess['tone_pos_dur'] = '[' + all_sess['tone_start_bins_str'] + '] - ' + all_sess['stim_dur_bin']
all_sess['n_tones_dur'] = all_sess['n_tones'].apply(lambda x: str(x)) + ' (' + all_sess['stim_dur_bin'] + ')'

# get custom sorted values
tone_info_labels = np.array(sorted(all_sess['tone_info_str'].unique().tolist(), key=lambda x: (len(x), x)))
tone_info_side_labels = np.array(sorted(all_sess['tone_info_side'].unique().tolist(), key=lambda x: (len(x), x)))
tone_info_pitch_labels = np.array(sorted(all_sess['tone_info_pitch'].unique().tolist(), key=lambda x: (len(x), x)))
tone_pos_dur_labels = np.array(sorted(all_sess['tone_pos_dur'].unique().tolist(), key=lambda x: (len(x), x)))
n_tones_dur_labels = np.array(sorted(all_sess['n_tones_dur'].unique().tolist()))

# make sure these values are always sorted appropriately using categories
all_sess['tone_info_str'] = pd.Categorical(all_sess['tone_info_str'], categories=tone_info_labels)
all_sess['tone_info_side'] = pd.Categorical(all_sess['tone_info_side'], categories=tone_info_side_labels)
all_sess['tone_info_pitch'] = pd.Categorical(all_sess['tone_info_pitch'], categories=tone_info_pitch_labels)
all_sess['tone_pos_dur'] = pd.Categorical(all_sess['tone_pos_dur'], categories=tone_pos_dur_labels)
all_sess['n_tones_dur'] = pd.Categorical(all_sess['n_tones_dur'], categories=n_tones_dur_labels)

# %% INVESTIGATE TRIAL TYPE COUNTS

# ignore bails because they are repeated
all_sess_no_bails = all_sess[all_sess['bail'] == False]

# aggregate count tables into dictionary
count_columns = ['correct_port', 'n_tones', 'stim_dur_bin', 'tone_info_side', 'tone_info_pitch', 'trial_type']
count_dict = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=True)

# plot bar charts and tables of trial distribution

def plotCounts(count_dict, stack, subj_ids, title, y_label):

    fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                            figsize=(4+0.5*len(subj_ids), 3*len(count_dict.keys())))
    fig.suptitle(title)

    bah.plot_counts(count_dict['correct_port'], axs[0], 'Correct Port', y_label, stack)
    bah.plot_counts(count_dict['n_tones'], axs[1], '# Tones', y_label, stack)
    bah.plot_counts(count_dict['stim_dur_bin'], axs[2], 'Stimulus Duration', y_label, stack)
    bah.plot_counts(count_dict['trial_type'], axs[3], 'Trial Types', y_label, stack)
    bah.plot_counts(count_dict['tone_info_side'], axs[4], 'Tone Sides', y_label, stack)
    bah.plot_counts(count_dict['tone_info_pitch'], axs[5], 'Tone Pitches', y_label, stack)


plotCounts(count_dict_pct, 'v', subj_ids, 'Trial Distribution', '% Trials')
plotCounts(count_dict, 'h', subj_ids, 'Trial Counts', '# Trials')

# %% LOOK AT HIT & BAIL RATES

# calculate per-subject metrics
# hit rates over time
n_back = 10
hit_rate_signal = []
irr_tone_volume = []

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]

    variants = subj_sess['task_variant'].unique()

    # CALCULATE HIT/BAIL METRICS
    # ignore bails and no responses
    rate_columns = ['trial_type', 'stim_dur_bin', ['trial_type', 'stim_dur_bin'],
                    'tone_info_str', ['tone_info_str', 'stim_dur_bin']]
    hit_metrics_dict = {}
    bail_metrics_dict = {}

    for variant in variants:
        var_subj_sess = subj_sess[subj_sess['task_variant'] == variant]
        var_subj_sess_no_bails = var_subj_sess[(var_subj_sess['bail'] == False) & (var_subj_sess['choice'] != 'none')]
        if exclude_instruct_from_hitrate:
            var_subj_sess_no_bails = var_subj_sess_no_bails[var_subj_sess_no_bails['tone_db_offsets'].apply(lambda x: np.all(x == 0))]
        var_subj_sess_tone_heard = var_subj_sess[var_subj_sess['abs_tone_start_times'].str[0] < var_subj_sess['cpoke_out_time']]

        hit_metrics_dict[variant] = bah.get_rate_dict(var_subj_sess_no_bails, 'hit', rate_columns)
        bail_metrics_dict[variant] = bah.get_rate_dict(var_subj_sess_tone_heard, 'bail', rate_columns)

    # COMPUTE METRICS SESSION BY SESSION

    # PROBABILITY OF OUTCOME BASED ON PREVIOUS OUTCOME:
    # p(incorrectly choose high|any high)
    # p(incorrectly choose low|any low)
    # p(choose right|previously chose right)
    # p(choose right|previously chose left)
    # p(stay with previous choice)
    # p(win-stay)
    # p(lose-switch)
    # p(bail|previous bail)
    # p(bail|previously incorrect)
    # p(bail|previously correct)

    n_high_any_high = {'num': 0, 'denom': 0}
    n_low_any_low = {'num': 0, 'denom': 0}
    n_left_any_left = {'num': 0, 'denom': 0}
    n_right_any_right = {'num': 0, 'denom': 0}
    n_right_prev_right = {'num': 0, 'denom': 0}
    n_right_prev_left = {'num': 0, 'denom': 0}
    n_repeat_choice = {'num': 0, 'denom': 0}
    n_win_stay = {'num': 0, 'denom': 0}
    n_lose_switch = {'num': 0, 'denom': 0}
    n_bail_prev_bail = {'num': 0, 'denom': 0}
    n_bail_prev_correct = {'num': 0, 'denom': 0}
    n_bail_prev_incorrect = {'num': 0, 'denom': 0}

    # ROLLING PERFORMANCE OVER TIME
    subj_hit_rate_signal = []
    subj_irr_tone_vol = []

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess[subj_sess['sessid'] == sess_id]
        ind_sess_no_bails = ind_sess[(ind_sess['bail'] == False) & (ind_sess['choice'] != 'none')]

        if len(ind_sess) == 0:
            continue

        # p(incorrect side|tone presence)
        # find which side is the high tone
        high_tone_side = ind_sess[ind_sess['response_tone'].str.contains('high')].iloc[0]['correct_port']
        high_tone_choice = ind_sess_no_bails['choice'].to_numpy() == high_tone_side
        left_tone_choice = ind_sess_no_bails['choice'].to_numpy() == 'left'
        incorrect = ind_sess_no_bails['hit'].to_numpy() == False
        any_tone_high = ind_sess_no_bails['tone_info_pitch'].apply(lambda x: 'H' in x).to_numpy()
        any_tone_low = ind_sess_no_bails['tone_info_pitch'].apply(lambda x: 'L' in x).to_numpy()
        any_tone_left = ind_sess_no_bails['tone_info_side'].apply(lambda x: 'L' in x).to_numpy()
        any_tone_right = ind_sess_no_bails['tone_info_side'].apply(lambda x: 'R' in x).to_numpy()
        n_high_any_high['num'] += sum(high_tone_choice & incorrect & any_tone_high)
        n_high_any_high['denom'] += sum(any_tone_high)
        n_low_any_low['num'] += sum(~high_tone_choice & incorrect & any_tone_low)
        n_low_any_low['denom'] += sum(any_tone_low)
        n_left_any_left['num'] += sum(left_tone_choice & incorrect & any_tone_left)
        n_left_any_left['denom'] += sum(any_tone_left)
        n_right_any_right['num'] += sum(~left_tone_choice & incorrect & any_tone_right)
        n_right_any_right['denom'] += sum(any_tone_right)

        # p(choice|previous choice)
        choices = ind_sess_no_bails['choice'].to_numpy()
        prev_choice_right = choices[:-1] == 'right'
        cur_choice_right = choices[1:] == 'right'
        n_right_prev_right['num'] += sum(cur_choice_right & prev_choice_right)
        n_right_prev_right['denom'] += sum(prev_choice_right)
        n_right_prev_left['num'] += sum(cur_choice_right & ~prev_choice_right)
        n_right_prev_left['denom'] += sum(~prev_choice_right)
        n_repeat_choice['num'] += sum(choices[:-1] == choices[1:])
        n_repeat_choice['denom'] += len(choices)-1

        # p(win-stay/lose-switch)
        stays = choices[:-1] == choices[1:]
        hits = ind_sess_no_bails['hit'].astype(bool).to_numpy()[:-1]
        n_win_stay['num'] += sum(stays & hits)
        n_win_stay['denom'] += sum(hits)
        n_lose_switch['num'] += sum(~stays & ~hits)
        n_lose_switch['denom'] += sum(~hits)

        # p(bail|previous result)
        bails = ind_sess['bail'].to_numpy()
        hits = ind_sess['hit'].to_numpy()
        prev_bail = bails[:-1] == True
        prev_correct = hits[:-1] == True
        prev_incorrect = hits[:-1] == False
        cur_bail = bails[1:] == True
        n_bail_prev_bail['num'] += sum(cur_bail & prev_bail)
        n_bail_prev_bail['denom'] += sum(prev_bail)
        n_bail_prev_correct['num'] += sum(cur_bail & prev_correct)
        n_bail_prev_correct['denom'] += sum(prev_correct)
        n_bail_prev_incorrect['num'] += sum(cur_bail & prev_incorrect)
        n_bail_prev_incorrect['denom'] += sum(prev_incorrect)

        # performance over time
        hits = ind_sess_no_bails['hit'].astype(bool).to_numpy()
        subj_hit_rate_signal.extend(np.convolve(hits, np.ones(n_back)/n_back, 'valid'))
        subj_irr_tone_vol.extend(ind_sess[ind_sess['bail'] == False]['tone_db_offsets'].apply(lambda x: np.min(x)).iloc[n_back:])

    hit_rate_signal.append(subj_hit_rate_signal)
    irr_tone_volume.append(subj_irr_tone_vol)

    # PLOT HIT/BAIL RATES AND RESPONSE PROBABILITIES

    # plot rate metrics

    if plot_bail:
        fig = plt.figure(constrained_layout=True, figsize=(9*len(variants), 10))
        gs = GridSpec(3, 2*len(variants), figure=fig)
    else:
        fig = plt.figure(constrained_layout=True, figsize=(8*len(variants), 7))
        gs = GridSpec(2, 2*len(variants), figure=fig, width_ratios=[2, 1])

    fig.suptitle('Psychometrics (subj {0})'.format(str(subj_id)))

    for i, variant in enumerate(variants):
        ax = fig.add_subplot(gs[0, 2*i])

        ax.set_title('Hit Rates \'{0}\' Variant'.format(variant))

        # plot hit rates for n tones by stimulus duration
        bah.plot_rate_heatmap(hit_metrics_dict[variant], 'stim_dur_bin', 'Stimulus Duration', 'trial_type', 'Trial Type', ax)

        # plot hit rates for unique tones by stimulus layout
        ax = fig.add_subplot(gs[1, 2*i])

        bah.plot_rate_heatmap(hit_metrics_dict[variant], 'stim_dur_bin', 'Stimulus Duration', 'tone_info_str', 'Stimuli', ax)

        if plot_bail:
            # plot bail rates
            ax = fig.add_subplot(gs[0, 2*i+1])

            ax.set_title('Bail Rates \'{0}\' Variant'.format(variant))

            # plot bail rates for n tones by stimulus duration
            bah.plot_rate_heatmap(bail_metrics_dict[variant], 'stim_dur_bin', 'Stimulus Duration', 'trial_type', 'Trial Type', ax)

            # plot bail rates for unique tones by stimulus layout
            ax = fig.add_subplot(gs[1, 2*i+1])

            bah.plot_rate_heatmap(bail_metrics_dict[variant], 'stim_dur_bin', 'Stimulus Duration', 'tone_info_str', 'Stimuli', ax)

        # plot probabilities
        def comp_p(n_dict): return n_dict['num']/n_dict['denom']
        prob_labels = ['p(wrong high|any high)', 'p(wrong low|any low)', 'p(wrong left|any left)', 'p(wrong right|any right)', 'p(right|prev right)',
                       'p(right|prev left)', 'p(repeat choice)', 'p(stay|correct)', 'p(switch|incorrect)', 'p(bail|bail)', 'p(bail|incorrect)', 'p(bail|correct)']
        prob_values = [comp_p(n_high_any_high), comp_p(n_low_any_low), comp_p(n_left_any_left), comp_p(n_right_any_right), comp_p(n_right_prev_right), 
                       comp_p(n_right_prev_left), comp_p(n_repeat_choice), comp_p(n_win_stay), comp_p(n_lose_switch), comp_p(n_bail_prev_bail),
                       comp_p(n_bail_prev_incorrect), comp_p(n_bail_prev_correct)]

        if plot_bail:
            ax = fig.add_subplot(gs[2, 2*i:2*i+2])
            ax.plot(np.arange(len(prob_labels)), prob_values, 'o')
            ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.25, 0.25))
            ax.set_xticks(np.arange(len(prob_labels)), prob_labels, rotation=-45)
            ax.yaxis.grid(True)
        else:
            ax = fig.add_subplot(gs[:, 2*i+1])
            ax.plot(np.flip(prob_values), np.arange(len(prob_labels)), 'o')
            ax.axvline(0.5, dashes=[4, 4], c='k', lw=1)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.25, 0.25))
            ax.set_yticks(np.arange(len(prob_labels)), np.flip(prob_labels), rotation=30)
            ax.xaxis.grid(True)
        ax.set_title('Response Probabilities')


# plot performance over time
n_cols = 1 if len(subj_ids) == 1 else 2
n_rows = int(np.ceil(len(subj_ids)/n_cols))*2
fig, axs = plt.subplots(n_rows, n_cols,
                        figsize=(8*n_cols, n_rows*2), constrained_layout=True)
axs = axs.reshape((-1, n_cols))
fig.suptitle('Rolling task performance (last {0} trials) and instructor cue'.format(n_back))
for i, subj_id in enumerate(subj_ids):
    ax = axs[int(np.floor(i/2)*2), i % 2]
    ax.plot(hit_rate_signal[i])
    ax.set_title('Subject {0}'.format(subj_id))
    ax.set_xticks([])
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True)
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
    if i % 2 == 0:
        ax.set_ylabel('Hit Rate')

    ax = axs[int(np.floor(i/2)*2+1), i % 2]
    ax.plot(irr_tone_volume[i])
    ax.set_xticks([])
    ax.set_ylim(-52, 2)
    ax.yaxis.grid(True)
    if i % 2 == 0:
        ax.set_ylabel('Irrelevant Tone Loudness (dB)')

plt.show(block=False)

# %% LOOK AT HIT REACTION TIMES/BAIL TIMES BY STIMULUS TYPE

# if plot_timing:
#     # format the data appropriately
#     rt_data = all_sess.rename(
#         columns={'subjid': 'Subject', 'RT': 'Reaction Time'})
#     rt_data['Subject'] = rt_data['Subject'].astype(str)
#     rt_data['Outcome'] = rt_data['hit'].apply(lambda x: 'Correct' if x else 'Incorrect')

#     # add bail time
#     rt_data['Bail Time'] = rt_data['cpoke_out_time'] - rt_data['stim_start_time']
#     rt_data.loc[rt_data['bail'] == False, 'Bail Time'] = np.nan

#     # first look at reaction times grouped by tones and stimulus layout accross subjects

#     plot = (ggplot(rt_data[rt_data['bail'] == False], aes(x='Subject', y='Reaction Time', fill='Outcome'))
#             + geom_violin(alpha=0.7)
#             + facet_grid('tone_info_str~tone_pos_dur')
#             + stat_summary(fun_data='mean_se', position=position_dodge(0.9))
#             + labs(title='Reaction Time Distributions'))

#     plot.draw(show=True)

#     # Then look at bail times, filtering out bails that happen before any tone is played
#     # construct dataframe with horizontal lines for each tone
#     tone_times = rt_data.groupby(['tone_info_str', 'tone_pos_dur'])['tone_start_bins_str'].agg(np.unique).apply(
#         lambda x: float(x) if not type(x) is np.ndarray else np.array([float(i) for i in x[0].split(',')])).reset_index()
#     # need to separate out the lines
#     unique_tone_times = [i for i in tone_times['tone_start_bins_str'].to_list() if type(i) is np.ndarray]
#     unique_tone_times = np.unique([i for l in unique_tone_times for i in l])
#     tone_times_dict = {}
#     for i, t in enumerate(unique_tone_times):
#         tone_times_dict[i] = tone_times.copy()
#         tone_times_dict[i]['tone_start_bins_str'] = tone_times_dict[i]['tone_start_bins_str'].apply(
#             lambda x: np.nan if np.isnan(x).any() else x[x == t][0] if any(x == t) else np.nan)

#     plot = (ggplot(rt_data[(rt_data['bail'] == True)], aes(x='Subject', y='Bail Time', fill='Subject'))
#             + geom_violin(alpha=0.7)
#             + geom_hline(data=tone_times_dict[0], mapping=aes(yintercept='tone_start_bins_str'), linetype='dashed')
#             + geom_hline(data=tone_times_dict[1], mapping=aes(yintercept='tone_start_bins_str'), linetype='dashed')
#             + geom_hline(data=tone_times_dict[2], mapping=aes(yintercept='tone_start_bins_str'), linetype='dashed')
#             + facet_grid('tone_info_str~tone_pos_dur')
#             + stat_summary(fun_data='mean_se', position=position_dodge(0.9))
#             + labs(title='Bail Time Distributions'))

#     plot.draw(show=True)

# %% Plot combined performance for multiple subjects

# This is updated code based on the above, just didn't want to rewrite all the prior stuff to make this

# subj_ids = [179, 180, 182, 186, 188, 202]
# sess_ids = db_access.get_subj_sess_ids(subj_ids, stage=8, date_end='2023-09-17')

# # get trial information
# loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True
# all_sess = loc_db.get_behavior_data(utils.flatten_dict_array(sess_ids))
# # remove trials where the stimulus didn't start
# all_sess = all_sess[all_sess['trial_started']]

# variants = all_sess['task_variant'].unique()

# # format columns for ease of aggregating and display
# # round stimulus duration
# all_sess['stim_dur'] = all_sess['stim_dur'].round(2).apply(lambda x: str(x) + 's')
# # reformat tone start times and high tones arrays into strings to be hashable for value counting and for display
# all_sess['trial_type'] = all_sess['tone_info'].apply(
#     lambda x: '1 tone' if not type(x) is list or len(x) == 1 else
#         '{0} tones - same'.format(len(x)) if x[0] == x[-1] else '{0} tones - diff'.format(len(x)))

# rate_columns = ['stim_dur', 'trial_type', ['trial_type', 'stim_dur']]
# hit_metrics_dict = {}
# data_metrics = {}

# for variant in variants:
#     var_sess = all_sess[all_sess['task_variant'] == variant]
#     var_sess_no_bails = var_sess[(var_sess['bail'] == False) & (var_sess['choice'] != 'none')]

#     data_metrics[variant] = {}
#     data_metrics[variant]['n_subjects'] = var_sess['subjid'].nunique()
#     data_metrics[variant]['n_sessions'] = var_sess['sessid'].nunique()
#     data_metrics[variant]['n_trials'] = len(var_sess_no_bails)

#     hit_metrics_dict[variant] = bah.get_rate_dict(var_sess_no_bails, 'hit', rate_columns)

# fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), constrained_layout=True)

# for i, variant in enumerate(variants):
#     ax = axs[i]
#     ax.set_title('\'{0}\' Variant'.format(variant))

#     bah.plot_rate_heatmap(hit_metrics_dict[variant], 'stim_dur', 'Stimulus Duration', 'trial_type', 'Trial Type', ax, fmt='.2f')

# plt.rcParams["svg.fonttype"] = 'none'
# fig.savefig(path.join(utils.get_user_home(), 'downloads', 'beh_performance.svg'), format='svg', dpi=1200)

