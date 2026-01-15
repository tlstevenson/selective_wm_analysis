# -*- coding: utf-8 -*-
"""
Script to investigate performance on the two-arm bandit task

@author: tanner stevenson
"""

# %% imports

import init
import pandas as pd
from pyutils import utils
from sys_neuro_tools import plot_utils
import hankslab_db.basicRLtasks_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sb
import copy
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
import statsmodels.api as sm

# %% LOAD DATA

# active_subjects_only = False

# subject_info = db_access.get_active_subj_stage('ClassicRLTasks')
# stage = 3
# if active_subjects_only:
#     subject_info = subject_info[subject_info['stage'] == stage]
# else:
#     subject_info = subject_info[subject_info['stage'] >= stage]

# subj_ids = subject_info['subjid']

subj_ids = [191]
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=1)
sess_ids = bah.limit_sess_ids(sess_ids, 15)

# get trial information
loc_db = db.LocalDB_BasicRLTasks('pavlovCond')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids))
#only look at trials that started (i.e. no sync failure)
all_sess = all_sess[all_sess['trial_started']]

# fix data formatting bug
all_sess['response_port'] = all_sess['response_port'].apply(lambda x: x[0] if utils.is_list(x) else x)
# add tone type for grouping/plotting
all_sess['tone_type'] = all_sess['reward_tone'].apply(lambda x: 'rewarded' if x else 'unrewarded')
# add session block for ease of aggregation
all_sess['sess_block'] = all_sess['sessid'].astype(str) + '_' + all_sess['block_num'].astype(str)

# make tone start bins
bin_size = 3
dur_bin_max = np.ceil(np.max(all_sess['rel_tone_start_time'])/bin_size)
dur_bin_min = np.floor(np.min(all_sess['rel_tone_start_time'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

all_sess['tone_start_bin'] = all_sess['rel_tone_start_time'].apply(
    lambda x: dur_bin_labels[np.where(x >= dur_bins)[0][-1]])

# make sure these values are always sorted appropriately using categories
dur_bin_labels = np.array(sorted(dur_bin_labels, key=lambda x: (len(x), x)))
all_sess['tone_start_bin'] = pd.Categorical(all_sess['tone_start_bin'], categories=dur_bin_labels)
all_sess['tone_type'] = pd.Categorical(all_sess['tone_type'], categories=['rewarded', 'unrewarded'])

# %% TRIAL COUNTS

# aggregate count tables into dictionary
count_columns = ['response_port', 'tone_type', 'tone_db_offset', 'reward_volume', 'tone_reward_corr', 'tone_start_bin']
column_titles = ['Response Port', 'Tone Type', 'Tone Volume Offsets (dB)', 'Reward Volumes (μL)', 'Tone Reward Correlation', 'Tone Start Time']
count_dict = bah.get_count_dict(all_sess, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(all_sess, 'subjid', count_columns, normalize=True)

# plot bar charts and tables of trial distribution

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
for i, (col_name, title) in enumerate(zip(count_columns, column_titles)):
    bah.plot_counts(count_dict[col_name], axs[i], title, '# Trials', 'h')

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
for i, (col_name, title) in enumerate(zip(count_columns, column_titles)):
    bah.plot_counts(count_dict_pct[col_name], axs[i], title, '% Trials', 'v')

# %% RESPONSE METRICS

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_rew_tones = subj_sess[subj_sess['reward_tone']]

    resp_probs = bah.get_rate_dict(subj_sess, 'hit', [['response_port', 'tone_type'], ['tone_start_bin', 'tone_type']])
    resp_tone_probs = bah.get_rate_dict(subj_sess_rew_tones, 'hit', [['tone_reward_corr', 'reward_volume'], ['tone_reward_corr', 'tone_db_offset']])

    fig = plt.figure(layout='constrained', figsize=(8, 10))
    fig.suptitle('Response Rates (Rat {})'.format(subj_id))
    gs = GridSpec(3, 2, figure=fig)

    # Response metrics for all tones
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('All Tones by Side')
    bah.plot_rate_heatmap(resp_probs, 'tone_type', 'Tone Type', 'response_port', 'Response Side', ax, col_summary=False)

    ax = fig.add_subplot(gs[1, 0])
    ax.set_title('All Tones by Tone Delay')
    bah.plot_rate_heatmap(resp_probs, 'tone_type', 'Tone Type', 'tone_start_bin', 'Tone Delay', ax)

    # Response metrics for rewarded tones
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title('Rewarded Tones by Reward Volume')
    bah.plot_rate_heatmap(resp_tone_probs, 'tone_reward_corr', 'Tone Volume/Reward Correlation', 'reward_volume', 'Reward Volume',  ax, col_summary=False)

    ax = fig.add_subplot(gs[1, 1])
    ax.set_title('Rewarded Tones by Tone Volume')
    bah.plot_rate_heatmap(resp_tone_probs, 'tone_reward_corr', 'Tone Volume/Reward Correlation', 'tone_db_offset', 'Tone Volume Offset (dB)', ax)

    # Probability of responding after rewarded/unrewarded tone over time
    n_trials = 50
    blocks = subj_sess['sess_block'].unique()
    rew_resp_mat = np.full((len(blocks), n_trials), np.nan)
    unrew_resp_mat = copy.deepcopy(rew_resp_mat)

    # fill in the matrices with the responses for each trial type at the start of each block
    for i, block in enumerate(blocks):
        block_sess = subj_sess[subj_sess['sess_block'] == block]
        rew_resp = block_sess[block_sess['reward_tone'] == True]['hit'].to_numpy()
        unrew_resp = block_sess[block_sess['reward_tone'] == False]['hit'].to_numpy()
        if len(rew_resp) > n_trials:
            rew_resp = rew_resp[:n_trials]
        if len(unrew_resp) > n_trials:
            unrew_resp = unrew_resp[:n_trials]
        rew_resp_mat[i,:len(rew_resp)] = rew_resp
        unrew_resp_mat[i,:len(unrew_resp)] = unrew_resp

    ax = fig.add_subplot(gs[2, :])
    ax.set_title('Responses Over Time')

    x = np.arange(1, n_trials+1)
    avg, err = bah.get_rate_avg_err(rew_resp_mat)
    plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='Rewarded Tone')
    avg, err = bah.get_rate_avg_err(unrew_resp_mat)
    plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='Unrewarded Tone')
    ax.set_xlabel('Trials from block switch')
    ax.set_ylabel('p(Response)')
    ax.legend()

    # RESPONSE TIMES

    fig = plt.figure(layout='constrained', figsize=(8, 9))
    fig.suptitle('Response Latencies (Rat {})'.format(subj_id))
    gs = GridSpec(3, 1, figure=fig)

    # By Rewarded/unrewarded tone and delays
    ax = fig.add_subplot(gs[0])
    sb.boxplot(data=subj_sess, x='tone_start_bin', y='RT', hue='tone_type', ax=ax)
    ax.set_xlabel('Tone Delay')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('All Tones')
    ax.legend(title='Tone Type')

    ax = fig.add_subplot(gs[1])
    sb.boxplot(data=subj_sess_rew_tones, x='tone_reward_corr', y='RT', hue='reward_volume', ax=ax)
    ax.set_xlabel('Tone Volume/Reward Correlation')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Rewarded Tones by Reward Volume')
    ax.legend(title='Reward Volume (μL)')

    ax = fig.add_subplot(gs[2])
    sb.boxplot(data=subj_sess_rew_tones, x='tone_reward_corr', y='RT', hue='tone_db_offset', ax=ax)
    ax.set_xlabel('Tone Volume/Reward Correlation')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Rewarded Tones by Tone Volume Offset')
    ax.legend(title='Tone Volume Offset (dB)')
