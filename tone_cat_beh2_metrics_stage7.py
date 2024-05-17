# -*- coding: utf-8 -*-
"""
Script to investigate performance on the tone categorization task stage 7 - grow nosepoke

@author: tanner stevenson
"""

import init

import pyutils.utils as utils
import hankslab_db.tonecatdelayresp_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

plot_bail = False
# plot_summary = True

# %% LOAD DATA

stage = 7
active_subjects_only = True
n_back = 1

subject_info = db_access.get_active_subj_stage('ToneCatDelayResp2')
if active_subjects_only:
    subject_info = subject_info[subject_info['stage'] == stage]
else:
    subject_info = subject_info[subject_info['stage'] >= stage]

subj_ids = subject_info['subjid']
subj_ids = 182

# get session ids
sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_num=stage)
sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# get trial information
loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids))
# remove trials where the stimulus didn't start
all_sess = all_sess[all_sess['trial_started']]

# %% Format Data

# calculate stimulus duration bins
all_sess['stim_dur'] = all_sess['stim_dur'].round(4)

bin_size = 1

dur_bin_max = np.ceil(np.max(all_sess['stim_dur'])/bin_size)
dur_bin_min = np.floor(np.min(all_sess['stim_dur'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

# format columns for ease of aggregating and display

# Add delay and duration bins
all_sess['stim_dur_bin'] = all_sess['stim_dur'].apply(
    lambda x: dur_bin_labels[np.where(x > dur_bins)[0][-1]])
# make sure they are always sorted appropriately using categories
all_sess['stim_dur_bin'] = pd.Categorical(all_sess['stim_dur_bin'], categories=dur_bin_labels)

# reformat tone info into strings to be hashable for value counting and for display
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

# get custom sorted values
tone_info_labels = np.array(sorted(all_sess['tone_info_str'].unique().tolist(), key=lambda x: (len(x), x)))
tone_info_side_labels = np.array(sorted(all_sess['tone_info_side'].unique().tolist(), key=lambda x: (len(x), x)))
tone_info_pitch_labels = np.array(sorted(all_sess['tone_info_pitch'].unique().tolist(), key=lambda x: (len(x), x)))

# make sure these values are always sorted appropriately using categories
all_sess['tone_info_str'] = pd.Categorical(all_sess['tone_info_str'], categories=tone_info_labels)
all_sess['tone_info_side'] = pd.Categorical(all_sess['tone_info_side'], categories=tone_info_side_labels)
all_sess['tone_info_pitch'] = pd.Categorical(all_sess['tone_info_pitch'], categories=tone_info_pitch_labels)


# %% INVESTIGATE TRIAL TYPE COUNTS

# ignore bails because they are repeated
all_sess_no_bails = all_sess[all_sess['bail'] == False]

# aggregate count tables into dictionary
count_columns = ['correct_port', 'stim_dur_bin', 'tone_info_side', 'tone_info_pitch']
count_dict = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=True)

# plot bar charts and tables of trial distribution

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
bah.plot_counts(count_dict['correct_port'], axs[0], 'Correct Port', '# Trials', 'h')
bah.plot_counts(count_dict['stim_dur_bin'], axs[1], 'Stimulus Duration', '# Trials', 'h')
bah.plot_counts(count_dict['tone_info_side'], axs[2], 'Stimulus Side', '# Trials', 'h')
bah.plot_counts(count_dict['tone_info_pitch'], axs[3], 'Stimulus Pitch', '# Trials', 'h')

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
bah.plot_counts(count_dict_pct['correct_port'], axs[0], 'Correct Port', '% Trials', 'v')
bah.plot_counts(count_dict_pct['stim_dur_bin'], axs[1], 'Trial Duration', '% Trials', 'v')
bah.plot_counts(count_dict_pct['tone_info_side'], axs[2], 'Stimulus Side', '% Trials', 'v')
bah.plot_counts(count_dict_pct['tone_info_pitch'], axs[3], 'Stimulus Pitch', '% Trials', 'v')

# %% LOOK AT HIT & BAIL RATES

# CALCULATE HIT/BAIL METRICS
# ignore bails and no responses
rate_columns = ['tone_info_str', 'stim_dur_bin', ['tone_info_str', 'stim_dur_bin']]

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]

    subj_sess_no_bails = subj_sess[(subj_sess['bail'] == False) & (subj_sess['choice'] != 'none')]
    subj_sess_tone_heard = subj_sess[subj_sess['cpoke_out_time'] > subj_sess['abs_tone_start_times']]

    hit_metrics_dict = bah.get_rate_dict(subj_sess_no_bails, 'hit', rate_columns)
    bail_metrics_dict = bah.get_rate_dict(subj_sess_tone_heard, 'bail', rate_columns)

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

    n_right_prev_right = {'num': 0, 'denom': 0}
    n_right_prev_left = {'num': 0, 'denom': 0}
    n_repeat_choice = {'num': 0, 'denom': 0}
    n_win_stay = {'num': 0, 'denom': 0}
    n_lose_switch = {'num': 0, 'denom': 0}
    n_bail_prev_bail = {'num': 0, 'denom': 0}
    n_bail_prev_correct = {'num': 0, 'denom': 0}
    n_bail_prev_incorrect = {'num': 0, 'denom': 0}

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess[subj_sess['sessid'] == sess_id]
        ind_sess_no_bails = ind_sess[(ind_sess['bail'] == False) & (ind_sess['choice'] != 'none')]

        if len(ind_sess) == 0:
            continue

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

    # PLOT HIT/BAIL RATES AND RESPONSE PROBABILITIES

    # plot hit metrics

    if plot_bail:
        fig = plt.figure(layout='constrained', figsize=(10, 6))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2,1])
    else:
        fig = plt.figure(layout='constrained', figsize=(5, 6))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2,1])

    fig.suptitle('Psychometrics (subj {0})'.format(str(subj_id)))

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Hit Rates')

    bah.plot_rate_heatmap(hit_metrics_dict, 'stim_dur_bin', 'Trial Duration', 'tone_info_str', 'Tone', ax)

    # ax = fig.add_subplot(gs[1, 0])

    # bah.plot_rate_heatmap(hit_metrics_dict, 'delay_bin', 'Response Delay', 'tone_info_str', 'Tone', ax, col_summary=False)

    # ax = fig.add_subplot(gs[0, 1])

    # bah.plot_rate_heatmap(hit_metrics_dict, 'tone_info_str', 'Tone', 'tone_db_offsets', 'Volume Offset', ax, row_summary=False)

    if plot_bail:
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('Bail Rates')

        bah.plot_rate_heatmap(bail_metrics_dict, 'stim_dur_bin', 'Trial Duration', 'tone_info_str', 'Tone', ax)

        # ax = fig.add_subplot(gs[1, 2])

        # bah.plot_rate_heatmap(bail_metrics_dict, 'delay_bin', 'Response Delay', 'tone_info_str', 'Tone', ax, col_summary=False)

        # ax = fig.add_subplot(gs[0, 3])

        # bah.plot_rate_heatmap(bail_metrics_dict, 'tone_info_str', 'Tone', 'tone_db_offsets', 'Volume Offset', ax, row_summary=False)

    # plot probabilities
    def comp_p(n_dict): return n_dict['num']/n_dict['denom']
    prob_labels = ['p(right|prev right)', 'p(right|prev left)', 'p(repeat choice)', 'p(stay|correct)',
                   'p(switch|incorrect)', 'p(bail|bail)', 'p(bail|incorrect)', 'p(bail|correct)']
    prob_values = [comp_p(n_right_prev_right), comp_p(n_right_prev_left), comp_p(n_repeat_choice),
                   comp_p(n_win_stay), comp_p(n_lose_switch), comp_p(n_bail_prev_bail),
                   comp_p(n_bail_prev_incorrect), comp_p(n_bail_prev_correct)]

    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.arange(len(prob_labels)), prob_values, 'o')
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_xticks(np.arange(len(prob_labels)), prob_labels, rotation=-45)
    ax.yaxis.grid(True)
    ax.set_title('Response Probabilities')


# %% plot combined performance

    all_sess_no_bails = all_sess[(all_sess['bail'] == False) & (all_sess['choice'] != 'none')]
    all_sess_tone_heard = all_sess[all_sess['cpoke_out_time'] > all_sess['abs_tone_start_times']]

    hit_metrics_dict = bah.get_rate_dict(all_sess_no_bails, 'hit', rate_columns)
    bail_metrics_dict = bah.get_rate_dict(all_sess_tone_heard, 'bail', rate_columns)

    if plot_bail:
        fig = plt.figure(layout='constrained', figsize=(6,4))
        gs = GridSpec(1, 2, figure=fig)
    else:
        fig = plt.figure(layout='constrained', figsize=(4,4))
        gs = GridSpec(1, 1, figure=fig)

    fig.suptitle('Psychometrics - All Subjects')

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Hit Rates')

    bah.plot_rate_heatmap(hit_metrics_dict, 'stim_dur_bin', 'Trial Duration', 'tone_info_str', 'Tone', ax)

    # ax = fig.add_subplot(gs[1, 0])

    # bah.plot_rate_heatmap(hit_metrics_dict, 'delay_bin', 'Response Delay', 'tone_info_str', 'Tone', ax, col_summary=False)

    # ax = fig.add_subplot(gs[0, 1])

    # bah.plot_rate_heatmap(hit_metrics_dict, 'tone_info_str', 'Tone', 'tone_db_offsets', 'Volume Offset', ax, row_summary=False)

    if plot_bail:
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('Bail Rates')

        bah.plot_rate_heatmap(bail_metrics_dict, 'stim_dur_bin', 'Trial Duration', 'tone_info_str', 'Tone', ax)
        # bah.plot_rate_heatmap(bail_metrics_dict, 'delay_bin', 'Response Delay', 'tone_db_offsets', 'Volume Offset', ax)

        # ax = fig.add_subplot(gs[1, 2])

        # bah.plot_rate_heatmap(bail_metrics_dict, 'delay_bin', 'Response Delay', 'tone_info_str', 'Tone', ax, col_summary=False)

        # ax = fig.add_subplot(gs[0, 3])

        # bah.plot_rate_heatmap(bail_metrics_dict, 'tone_info_str', 'Tone', 'tone_db_offsets', 'Volume Offset', ax, row_summary=False)

