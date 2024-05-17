# -*- coding: utf-8 -*-
"""
Script to investigate performance on the tone categorization task stage 5 - single tone

@author: tanner stevenson
"""

import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..'))

from hankslab_db import db_access
import hankslab_db.tonecatdelayresp_db as db
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from sys_neuro_tools import plot_utils

subject_info = db_access.get_active_subj_stage('ToneCatDelayResp2', stage=5)
subj_ids = subject_info['subjid']

sess_ids = db_access.get_subj_sess_ids(subj_ids, 5)
loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True

n_sessions_back = 3
tone_dur = 0.25
# performance
right_high_subj = []
left_high_subj = []
# counts of trials
trial_counts = {'side': {'left': [], 'right': []}}

# sequence counts for all subjects
# bails in a row
# responses in a row
# correct in a row
# incorrect in a row
# choose right in a row
# choose left in a row
# same tone in a row
sequence_counts = {'bails': [], 'responses': [], 'correct': [], 'incorrect': [],
                   'right': [], 'left': [], 'tones': []}

hit_metrics_dict[variant][key] = var_subj_sess_no_bails.groupby(col).agg(
    n=('hit', 'count'), rate=('hit', 'mean')).astype({'rate': 'float64'})

# hit rates over time
n_back = 10
hit_rate_signal = []

for subj_id in subj_ids:
    subj_sess_ids = sess_ids[subj_id][-n_sessions_back:]
    sess = loc_db.get_behavior_data(subj_sess_ids)

    # remove trials where the stimulus didn't start and where the animals didn't respond to the response cue
    sess = sess[sess['trial_started'] & ~((sess['bail'] == False) & (sess['choice'] == 'none'))]

    # HIT AND BAIL RATES BY SIDE/TONE
    hit_rates = {'left': np.nan, 'right': np.nan}
    bail_rates = {'left': np.nan, 'right': np.nan}
    right_trials = sess[sess['correct_port'] == 'right']
    left_trials = sess[sess['correct_port'] == 'left']
    tone_heard_trials = sess[sess['abs_tone_start_times'] < sess['cpoke_out_time']]
    right_tone_heard_trials = tone_heard_trials[tone_heard_trials['correct_port'] == 'right']
    left_tone_heard_trials = tone_heard_trials[tone_heard_trials['correct_port'] == 'left']

    hit_rates['right'] = sum(right_trials['hit'] == True)/sum(right_trials['bail'] == False)
    hit_rates['left'] = sum(left_trials['hit'] == True)/sum(left_trials['bail'] == False)
    bail_rates['right'] = sum(right_tone_heard_trials['bail'] == True)/len(right_tone_heard_trials)
    bail_rates['left'] = sum(left_tone_heard_trials['bail'] == True)/len(left_tone_heard_trials)

    # TRIAL COUNTS
    sess_no_bail = sess[sess['bail'] == False]
    trial_counts['side']['left'].append((sess_no_bail['correct_port'] == 'left').sum())
    trial_counts['side']['right'].append((sess_no_bail['correct_port'] == 'right').sum())

    # COMPUTE METRICS SESSION BY SESSION

    # PROBABILITY OF OUTCOME BASED ON PREVIOUS OUTCOME:
    # p(choose right|previously chose right)
    # p(choose right|previously chose left)
    # p(stay with previous choice)
    # p(win-stay)
    # p(lose-switch)
    # p(bail|previous bail)
    # p(bail|previously incorrect)
    # p(bail|previously correct)
    # p(same tone repeating)

    n_right_prev_right = {'num': 0, 'denom': 0}
    n_right_prev_left = {'num': 0, 'denom': 0}
    n_repeat_choice = {'num': 0, 'denom': 0}
    n_win_stay = {'num': 0, 'denom': 0}
    n_lose_switch = {'num': 0, 'denom': 0}
    n_bail_prev_bail = {'num': 0, 'denom': 0}
    n_bail_prev_correct = {'num': 0, 'denom': 0}
    n_bail_prev_incorrect = {'num': 0, 'denom': 0}
    n_repeat_tone = {'num': 0, 'denom': 0}

    # FREQUENCIES OF VARIOUS SEQUENCES:
    subj_sequence_counts = {k: [] for k in sequence_counts.keys()}

    # ROLLING PERFORMANCE OVER TIME
    subj_hit_rate_signal = []

    for sess_id in subj_sess_ids:
        ind_sess = sess[sess['sessid'] == sess_id]

        if len(ind_sess) == 0:
            continue

        # p(choice|previous choice)
        choices = ind_sess[ind_sess['bail'] == False]['choice'].to_numpy()
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
        hits = ind_sess[ind_sess['bail'] == False]['hit'].astype(bool).to_numpy()[:-1]
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

        # p(same tone repeating)
        tones = ind_sess[ind_sess['bail'] == False]['response_tone'].to_numpy()
        n_repeat_tone['num'] += sum(tones[:-1] == tones[1:])
        n_repeat_tone['denom'] += len(tones)-1

        # sequence metrics
        # bails and responses in a row
        bails = ind_sess['bail'].astype(bool).to_numpy()
        seq_len = 1
        for i in range(len(bails)-1):
            if bails[i] == bails[i+1]:
                seq_len += 1

            if bails[i] != bails[i+1] or i == len(bails)-2:
                if bails[i]:
                    subj_sequence_counts['bails'].append(seq_len)
                else:
                    subj_sequence_counts['responses'].append(seq_len)
                seq_len = 1

        # correct/incorrect in a row
        hits = ind_sess[ind_sess['bail'] == False]['hit'].astype(bool).to_numpy()
        seq_len = 1
        for i in range(len(hits)-1):
            if hits[i] == hits[i+1]:
                seq_len += 1

            if hits[i] != hits[i+1] or i == len(hits)-2:
                if hits[i]:
                    subj_sequence_counts['correct'].append(seq_len)
                else:
                    subj_sequence_counts['incorrect'].append(seq_len)
                seq_len = 1

        # side choices in a row
        choices = ind_sess[ind_sess['bail'] == False]['choice'].to_numpy()
        seq_len = 1
        for i in range(len(choices)-1):
            if choices[i] == choices[i+1]:
                seq_len += 1

            if choices[i] != choices[i+1] or i == len(choices)-2:
                if choices[i] == 'right':
                    subj_sequence_counts['right'].append(seq_len)
                else:
                    subj_sequence_counts['left'].append(seq_len)
                seq_len = 1

        # stimuli in a row
        tones = ind_sess[ind_sess['bail'] == False]['response_tone'].to_numpy()
        seq_len = 1
        for i in range(len(tones)-1):
            if tones[i] == tones[i+1]:
                seq_len += 1

            if tones[i] != tones[i+1] or i == len(tones)-2:
                subj_sequence_counts['tones'].append(seq_len)
                seq_len = 1

        # performance over time
        hits = ind_sess[ind_sess['bail'] == False]['hit'].astype(bool).to_numpy()
        subj_hit_rate_signal.extend(np.convolve(hits, np.ones(n_back)/n_back, 'valid'))

    # aggregate sequence counts and performance signals
    for k in subj_sequence_counts.keys():
        sequence_counts[k].append(subj_sequence_counts[k])

    hit_rate_signal.append(subj_hit_rate_signal)

    # plot psychometrics per subject

    fig = plt.figure(constrained_layout=True)
    fig.suptitle('Psychometrics (subj {0})'.format(str(subj_id)))

    gs = GridSpec(2, 2, figure=fig)

    # plot rates
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(0, len(hit_rates.keys()))
    ax.plot(x, hit_rates.values(), 'o')
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
    ax.set_title('Rates')
    ax.set_ylabel('Hit rate')
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels(hit_rates.keys())
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.yaxis.grid(True)

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(bail_rates.values(), 'o')
    ax.set_xlabel('Correct Side')
    ax.set_ylabel('Bail rate')
    ax.set_ylim(0, 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(bail_rates.keys())
    ax.yaxis.grid(True)

    # plot probabilities
    def comp_p(n_dict): return n_dict['num']/n_dict['denom']
    prob_labels = ['p(right|prev right)', 'p(right|prev left)', 'p(repeat choice)', 'p(stay|correct)',
                   'p(switch|incorrect)', 'p(bail|bail)', 'p(bail|incorrect)', 'p(bail|correct)', 'p(repeated tone)']
    prob_values = [comp_p(n_right_prev_right), comp_p(n_right_prev_left), comp_p(n_repeat_choice),
                   comp_p(n_win_stay), comp_p(n_lose_switch), comp_p(n_bail_prev_bail),
                   comp_p(n_bail_prev_incorrect), comp_p(n_bail_prev_correct), comp_p(n_repeat_tone)]
    ax = fig.add_subplot(gs[:, 1])
    ax.plot(np.flip(prob_values), np.arange(len(prob_labels)), 'o')
    ax.axvline(0.5, dashes=[4, 4], c='k', lw=1)
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    ax.set_xticks(np.arange(0, 1.25, 0.25))
    ax.set_yticks(np.arange(len(prob_labels)), np.flip(prob_labels), rotation=30)
    ax.xaxis.grid(True)


# plot trial counts

fig, ax = plt.subplots(1, 1, constrained_layout=True)
fig.suptitle('Trial Counts')
x_labels = [str(i) for i in subj_ids]

val_labels = list(trial_counts['side'].keys())
vals = [trial_counts['side'][k] for k in val_labels]
plot_utils.plot_stacked_bar(vals, val_labels, x_labels, ax=ax)
ax.set_ylabel('# Trials')
ax.set_title('Side')

# also plot in percentages
fig, ax = plt.subplots(1, 1, constrained_layout=True)
fig.suptitle('Trial Distribution')

val_labels = list(trial_counts['side'].keys())
vals = [trial_counts['side'][k] for k in val_labels]
val_tot = np.sum(vals, 0)
vals = [vals[i]/val_tot for i in range(len(vals))]
plot_utils.plot_stacked_bar(vals, val_labels, x_labels, 'v', ax)
ax.set_ylabel('% Trials')
ax.set_title('Side')

# plot sequence counts
x = np.arange(len(subj_ids))+1
x_labels = [str(i) for i in subj_ids]

fig = plt.figure(constrained_layout=True, figsize=(10, 9))
fig.suptitle('Sequence Distributions')

gs = GridSpec(3, 2, figure=fig)

# bails & responses
ax = fig.add_subplot(gs[0, 0])
ax.boxplot(sequence_counts['bails'])
ax.set_title('Bails')
ax.set_ylabel('Sequence Length')
ax.set_xticks(x, x_labels)
ax.yaxis.grid(True)

ax = fig.add_subplot(gs[0, 1])
ax.boxplot(sequence_counts['responses'])
ax.set_title('Responses')
ax.set_xticks(x, x_labels)
ax.yaxis.grid(True)

# correct/incorrect responses
ax = fig.add_subplot(gs[1, 0])
ax.boxplot(sequence_counts['correct'])
ax.set_title('Correct')
ax.set_ylabel('Sequence Length')
ax.set_xticks(x, x_labels)
ax.yaxis.grid(True)

ax = fig.add_subplot(gs[1, 1])
ax.boxplot(sequence_counts['incorrect'])
ax.set_title('Incorrect')
ax.set_xticks(x, x_labels)
ax.yaxis.grid(True)

# right/left responses
ax = fig.add_subplot(gs[2, 0])
ax.boxplot(sequence_counts['right'])
ax.set_title('Right Choice')
ax.set_ylabel('Sequence Length')
ax.set_xticks(x, x_labels)
ax.yaxis.grid(True)

ax = fig.add_subplot(gs[2, 1])
ax.boxplot(sequence_counts['left'])
ax.set_title('Left Choice')
ax.set_xticks(x, x_labels)
ax.yaxis.grid(True)

# # tones
# ax = fig.add_subplot(gs[3, :])
# ax.boxplot(sequence_counts['tones'])
# ax.set_title('Tones')
# ax.set_ylabel('Sequence Length')
# ax.set_xticks(x, x_labels)
# ax.yaxis.grid(True)

# plot performance over time
n_cols = 1 if len(subj_ids) == 1 else 2
n_rows = int(np.ceil(len(subj_ids)/n_cols))
fig, axs = plt.subplots(n_rows, n_cols, sharey=True,
                        figsize=(8*n_cols, n_rows*2), constrained_layout=True)
axs = axs.reshape((-1, n_cols))
fig.suptitle('Rolling task performance (last {0} trials)'.format(n_back))
for i, subj_id in enumerate(subj_ids):
    ax = axs[int(np.floor(i/2)), i % 2]
    ax.plot(hit_rate_signal[i])
    ax.set_title('Subject {0}'.format(subj_id))
    ax.set_xticks([])
    ax.yaxis.grid(True)

    if i % 2 == 0:
        ax.set_ylabel('Hit Rate')
