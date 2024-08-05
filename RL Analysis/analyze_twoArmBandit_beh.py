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
import re

# %% LOAD DATA

# active_subjects_only = False

# subject_info = db_access.get_active_subj_stage('ClassicRLTasks')
# stage = 3
# if active_subjects_only:
#     subject_info = subject_info[subject_info['stage'] == stage]
# else:
#     subject_info = subject_info[subject_info['stage'] >= stage]

# subj_ids = subject_info['subjid']

#sess_ids = db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 2)

# optionally limit sessions based on subject ids
subj_ids = [179, 188, 191, 207, 182]
# sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=2)
#sess_ids = bah.limit_sess_ids(sess_ids, 10)
#sess_ids = {179: [95201, 95312, 95347]}

# get trial information
reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

# make sure RT column is filled in (it wasn't initially)
all_sess['RT'] = all_sess['response_time'] - all_sess['response_cue_time']
all_sess['cpoke_out_latency'] = all_sess['cpoke_out_time'] - all_sess['response_cue_time']

# %% TRIAL COUNTS

# aggregate count tables into dictionary
count_columns = ['side_prob', 'block_prob', 'high_side']
column_titles = ['Side Probability (L/R)', 'Block Probabilities', 'High Side']
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

# %% Basic response metrics

# ignore no responses
#all_sess_resp = all_sess[all_sess['choice'] != 'none']

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

    choose_left_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_left', ['side_prob', 'high_side'])
    choose_right_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_right', ['high_side'])
    choose_high_trial_probs = bah.get_rate_dict(subj_sess_resp, 'chose_high', ['block_prob'])

    fig = plt.figure(layout='constrained', figsize=(6, 6))
    fig.suptitle('Response Probabilities (Rat {})'.format(subj_id))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2,1])

    # choose high for each trial probability
    ax = fig.add_subplot(gs[0, 0])

    data = choose_high_trial_probs['block_prob']
    x_labels = data['block_prob']
    x_vals = np.arange(len(x_labels))
    prob_vals = np.unique([int(x.split('/')[0]) for x in x_labels])

    ax.errorbar(x_vals, data['rate'], yerr=bah.convert_rate_err_to_mat(data), fmt='o', capsize=4)
    plot_utils.plot_dashlines(prob_vals/100, dir='h', ax=ax)
    ax.set_ylabel('p(Choose High)')
    ax.set_xlabel('Block Reward Probability (High/Low %)')
    ax.set_xticks(x_vals, x_labels)
    ax.set_xlim(-0.5, len(x_vals)-0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Choose High Side')

    # choose side when side is high/low
    ax = fig.add_subplot(gs[0, 1])

    left_data = choose_left_side_probs['high_side']
    right_data = choose_right_side_probs['high_side']
    x_labels = left_data['high_side']
    x_vals = [1,2]

    ax.errorbar(x_vals, left_data['rate'], yerr=bah.convert_rate_err_to_mat(left_data), fmt='o', capsize=4, label='Left Choice')
    ax.errorbar(x_vals, right_data['rate'], yerr=bah.convert_rate_err_to_mat(right_data), fmt='o', capsize=4, label='Right Choice')
    ax.set_ylabel('p(Choose Side)')
    ax.set_xlabel('High Probability Side')
    ax.set_xticks(x_vals, x_labels)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='best')
    ax.set_title('Choose Side')

    # choose left by left probability
    ax = fig.add_subplot(gs[1, :])

    data = choose_left_side_probs['side_prob']
    x_labels = data['side_prob']
    x_vals = np.arange(len(x_labels))
    prob_vals = np.unique([int(x.split('/')[0]) for x in x_labels])

    ax.errorbar(x_vals, data['rate'], yerr=bah.convert_rate_err_to_mat(data), fmt='o', capsize=4)
    plot_utils.plot_dashlines(prob_vals/100, dir='h', ax=ax)
    ax.set_ylabel('p(Choose Left)')
    ax.set_xlabel('Block Reward Probability (Left/Right %)')
    ax.set_xticks(x_vals, x_labels)
    ax.set_xlim(-0.5, len(x_vals)-0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Choose Left')

    # SWITCHING PROBABILITIES
    choice_block_probs = np.sort(subj_sess_resp['choice_block_prob'].unique())
    choice_probs = np.sort(subj_sess_resp['choice_prob'].unique())
    block_rates = np.sort(subj_sess_resp['block_prob'].unique())

    n_stay_reward_choice = {p: {'k': 0, 'n': 0} for p in choice_block_probs}
    n_switch_noreward_choice = copy.deepcopy(n_stay_reward_choice)
    n_stay_reward_block = {br: {'k': 0, 'n': 0} for br in block_rates}
    n_stay_noreward_block = copy.deepcopy(n_stay_reward_block)

    # empirical transition matrices by block reward rates
    # structure: choice x next choice, (high prob, low prob)
    trans_mats = {br: {'k': np.zeros((2,2)), 'n': np.zeros((2,2))} for br in block_rates}

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

        if len(ind_sess) == 0:
            continue

        choices = ind_sess['choice'].to_numpy()
        rewarded = ind_sess['rewarded'].to_numpy()[:-1] # ignore last trial since no trial after
        choice_block_prob = ind_sess['choice_block_prob'].to_numpy()[:-1]
        stays = choices[:-1] == choices[1:]
        block_rate = ind_sess['block_prob'].to_numpy()[:-1]
        high_choice = ind_sess['chose_high'].to_numpy()[:-1]

        for p in choice_block_probs:
            choice_sel = choice_block_prob == p
            n_stay_reward_choice[p]['k'] += sum(rewarded & stays & choice_sel)
            n_stay_reward_choice[p]['n'] += sum(rewarded & choice_sel)
            n_switch_noreward_choice[p]['k'] += sum(~rewarded & ~stays & choice_sel)
            n_switch_noreward_choice[p]['n'] += sum(~rewarded & choice_sel)

        for br in block_rates:
            rate_sel = block_rate == br
            n_stay_reward_block[br]['k'] += sum(rewarded & stays & rate_sel)
            n_stay_reward_block[br]['n'] += sum(rewarded & rate_sel)
            n_stay_noreward_block[br]['k'] += sum(~rewarded & stays & rate_sel)
            n_stay_noreward_block[br]['n'] += sum(~rewarded & rate_sel)

            trans_mats[br]['k'][0,0] += sum(high_choice & stays & rate_sel)
            trans_mats[br]['n'][0,0] += sum(high_choice & rate_sel)
            trans_mats[br]['k'][0,1] += sum(high_choice & ~stays & rate_sel)
            trans_mats[br]['n'][0,1] += sum(high_choice & rate_sel)
            trans_mats[br]['k'][1,0] += sum(~high_choice & ~stays & rate_sel)
            trans_mats[br]['n'][1,0] += sum(~high_choice & rate_sel)
            trans_mats[br]['k'][1,1] += sum(~high_choice & stays & rate_sel)
            trans_mats[br]['n'][1,1] += sum(~high_choice & rate_sel)

    # plot results
    # define reusable helper methods
    def comp_p(n_dict): return n_dict['k']/n_dict['n']
    def comp_err(n_dict): return abs(utils.binom_cis(n_dict['k'], n_dict['n']) - comp_p(n_dict))

    fig = plt.figure(layout='constrained', figsize=(9, 7))
    fig.suptitle('Switching Probabilities (Rat {})'.format(subj_id))
    gs = GridSpec(2, max(len(block_rates), 2), figure=fig, height_ratios=[3,2])

    # first row, left, is the win-stay/lose-switch rates by choice probability
    ax = fig.add_subplot(gs[0, :-1])
    stay_reward_vals = [comp_p(n_stay_reward_choice[p]) for p in choice_block_probs]
    stay_reward_err = np.asarray([comp_err(n_stay_reward_choice[p]) for p in choice_block_probs]).T
    switch_noreward_vals = [comp_p(n_switch_noreward_choice[p]) for p in choice_block_probs]
    switch_noreward_err = np.asarray([comp_err(n_switch_noreward_choice[p]) for p in choice_block_probs]).T

    x_vals = np.arange(len(choice_block_probs))

    ax.errorbar(x_vals, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Win Stay')
    ax.errorbar(x_vals, switch_noreward_vals, yerr=switch_noreward_err, fmt='o', capsize=4, label='Lose Switch')
    ax.set_ylabel('Proportion of Choices')
    ax.set_xlabel('Choice Reward Probability (Block Probs)')
    ax.set_xticks(x_vals, choice_block_probs)
    ax.set_xlim(-0.5, len(x_vals)-0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y')
    ax.legend(loc='best')

    # first row, right, is the stay percentage after reward/no reward by block rate
    ax = fig.add_subplot(gs[0, -1])
    stay_reward_vals = [comp_p(n_stay_reward_block[br]) for br in block_rates]
    stay_reward_err = np.asarray([comp_err(n_stay_reward_block[br]) for br in block_rates]).T
    stay_noreward_vals = [comp_p(n_stay_noreward_block[br]) for br in block_rates]
    stay_noreward_err = np.asarray([comp_err(n_stay_noreward_block[br]) for br in block_rates]).T

    plot_utils.plot_stacked_bar([stay_reward_vals, stay_noreward_vals], value_labels=['Rewarded', 'Unrewarded'],
                                x_labels=block_rates, err=[stay_reward_err, stay_noreward_err], ax=ax)
    ax.set_ylabel('p(Stay)')
    ax.set_xlabel('Block Reward Probability (Left/Right %)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left')

    # second row is empirical transition matrices
    for i, br in enumerate(block_rates):
        ax = fig.add_subplot(gs[1, i])
        p_mat = trans_mats[br]['k']/trans_mats[br]['n']
        plot_utils.plot_value_matrix(p_mat, ax=ax, xticklabels=['high', 'low'], yticklabels=['high', 'low'], cbar=False, cmap='vlag')
        ax.set_ylabel('Choice Reward Rate')
        ax.set_xlabel('Next Choice Reward Rate')
        ax.xaxis.set_label_position('top')
        ax.set_title('Block Rate {}%'.format(br))


    # Simple summary of choice behavior
    fig = plt.figure(layout='constrained', figsize=(6, 3))
    fig.suptitle('Choice Probabilities ({})'.format(subj_id))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[3,4])

    data = choose_high_trial_probs['block_prob']
    x_labels = data['block_prob']
    x_vals = np.arange(len(x_labels))
    prob_vals = np.unique([int(x.split('/')[0]) for x in x_labels])

    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(x_vals, data['rate'], yerr=bah.convert_rate_err_to_mat(data), fmt='o', capsize=4)
    plot_utils.plot_dashlines(prob_vals/100, dir='h', ax=ax)
    ax.set_ylabel('p(Choose High)')
    ax.set_xlabel('Block Reward Probability (High/Low %)')
    ax.set_xticks(x_vals, x_labels)
    ax.set_xlim(-0.5, len(x_vals)-0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Choose High Side')


    stay_reward_vals = [comp_p(n_stay_reward_block[br]) for br in block_rates]
    stay_reward_err = np.asarray([comp_err(n_stay_reward_block[br]) for br in block_rates]).T
    stay_noreward_vals = [comp_p(n_stay_noreward_block[br]) for br in block_rates]
    stay_noreward_err = np.asarray([comp_err(n_stay_noreward_block[br]) for br in block_rates]).T

    ax = fig.add_subplot(gs[0, 1])
    plot_utils.plot_stacked_bar([stay_reward_vals, stay_noreward_vals], value_labels=['Rewarded', 'Unrewarded'],
                                x_labels=block_rates, err=[stay_reward_err, stay_noreward_err], ax=ax)
    ax.set_ylabel('p(Stay)')
    ax.set_xlabel('Block Reward Probability (High/Low %)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower right')
    ax.set_title('Choose Previous Side')


# %% Choice metrics over trials
for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

    block_rates = np.sort(subj_sess_resp['block_prob'].unique())

    # Investigate how animals choose to stay/switch their response
    # Probability of choosing high port/getting reward pre and post block change by block reward
    # probability of getting reward leading up to stay/switch by port probability (high/low)
    # numbers of low/high reward rate choices in a row that are rewarded/unrewarded/any
    n_away = 10
    p_choose_high_blocks = {br: {'pre': [], 'post': []} for br in block_rates}
    p_choose_high_blocks['all'] = []

    reward_choices = ['high', 'low', 'all']
    p_reward_switch = {c: {'stay': [], 'switch': []} for c in reward_choices}
    p_reward_switch_blocks = {br: {c: {'stay': [], 'switch': []} for c in reward_choices} for br in block_rates}

    sequence_counts = {r: {br: [] for br in block_rates} for r in ['high', 'low']}
    get_seq_counts = lambda x: utils.get_sequence_lengths(x)[True] if np.sum(x) > 0 else []

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

        if len(ind_sess) == 0:
            continue

        rewarded = ind_sess['rewarded'].to_numpy()
        choices = ind_sess['choice'].to_numpy()
        stays = choices[:-1] == choices[1:]
        block_rate = ind_sess['block_prob'].to_numpy()
        high_choice = ind_sess['chose_high'].to_numpy()
        block_switch_idxs = np.where(np.diff(ind_sess['block_num']) != 0)[0] + 1
        pre_switch_rates = block_rate[block_switch_idxs-1]
        post_switch_rates = block_rate[block_switch_idxs]

        # block changes by pre and post change block reward
        for i, switch_idx in enumerate(block_switch_idxs):
            pre_br = pre_switch_rates[i]
            post_br = post_switch_rates[i]

            choose_high_switch = np.full(n_away*2, np.nan)
            if i == 0:
                pre_switch_mask_dist = np.minimum(n_away, switch_idx)
            else:
                pre_switch_mask_dist = np.minimum(n_away, switch_idx - block_switch_idxs[i-1])

            if i == len(block_switch_idxs)-1:
                post_switch_mask_dist = np.minimum(n_away, len(ind_sess) - switch_idx)
            else:
                post_switch_mask_dist = np.minimum(n_away, block_switch_idxs[i+1] - switch_idx)

            choose_high_switch[n_away-pre_switch_mask_dist : n_away+post_switch_mask_dist] = high_choice[switch_idx-pre_switch_mask_dist : switch_idx+post_switch_mask_dist]

            p_choose_high_blocks[pre_br]['pre'].append(choose_high_switch)
            p_choose_high_blocks[post_br]['post'].append(choose_high_switch)
            p_choose_high_blocks['all'].append(choose_high_switch)

        # probability of reward before a stay/switch decision
        # first construct reward matrix where center column is the current trial, starting from second trial
        nan_buffer = np.full(n_away-1, np.nan)
        buff_rewarded = np.concatenate((nan_buffer, rewarded, nan_buffer))
        reward_mat = np.hstack([buff_rewarded[i:-n_away*2+i+1, None] if i<n_away*2-1 else buff_rewarded[i:, None] for i in range(n_away*2)])
        # group by choice across all blocks
        for choice_type in reward_choices:
            match choice_type:
                case 'high':
                    sel = high_choice[1:] == True
                case 'low':
                    sel = high_choice[1:] == False
                case 'all':
                    sel = np.full_like(high_choice[1:], True)

            p_reward_switch[choice_type]['stay'].append(reward_mat[sel & stays, :])
            p_reward_switch[choice_type]['switch'].append(reward_mat[sel & ~stays, :])

        # group by block of prior choice, excluding second trials
        rem_second_trials = ind_sess['block_trial'][:-1] != 2
        for br in block_rates:
            rate_sel = (block_rate[:-1] == br) & rem_second_trials
            for choice_type in reward_choices:
                match choice_type:
                    case 'high':
                        sel = high_choice[1:] == True
                    case 'low':
                        sel = high_choice[1:] == False
                    case 'all':
                        sel = np.full_like(high_choice[1:], True)

                p_reward_switch_blocks[br][choice_type]['stay'].append(reward_mat[rate_sel & sel & stays, :])
                p_reward_switch_blocks[br][choice_type]['switch'].append(reward_mat[rate_sel & sel & ~stays, :])

        # sequence counts before a switch
        high_stays = stays & high_choice[:-1]
        low_stays = stays & ~high_choice[:-1]
        for br in block_rates:
            rate_sel = block_rate[:-1] == br
            sequence_counts['high'][br].extend(get_seq_counts(high_stays & rate_sel))
            sequence_counts['low'][br].extend(get_seq_counts(low_stays & rate_sel))

    # plot switching information
    fig = plt.figure(layout='constrained', figsize=(16, 7))
    fig.suptitle('Switching Metrics (Rat {})'.format(subj_id))
    gs = GridSpec(2, 4, figure=fig)

    x = np.arange(-n_away, n_away)

    # p choose high grouped by prior block rate
    ax = fig.add_subplot(gs[0,0])
    plot_utils.plot_x0line(ax=ax)
    for br in block_rates:
        raw_mat = np.asarray(p_choose_high_blocks[br]['pre'])
        avg, err = bah.get_rate_avg_err(raw_mat)
        plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='{0}%'.format(br))

    ax.plot(x, np.nanmean(np.asarray(p_choose_high_blocks['all']), axis=0), dashes=[4,4], c='k', label='all')

    ax.set_xlabel('Trials from block switch')
    ax.set_ylabel('p(Choose High)')
    ax.set_title('Choose High by Pre-switch Reward Rates')
    ax.set_xlim(-5, 8)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', title='Reward Rates')

    # p choose high by next block rate
    ax = fig.add_subplot(gs[1,0])
    plot_utils.plot_x0line(ax=ax)
    for br in block_rates:
        raw_mat = np.asarray(p_choose_high_blocks[br]['post'])
        avg, err = bah.get_rate_avg_err(raw_mat)
        plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='{0}%'.format(br))

    ax.plot(x, np.nanmean(np.asarray(p_choose_high_blocks['all']), axis=0), dashes=[4,4], c='k', label='all')

    ax.set_xlabel('Trials from block switch')
    ax.set_ylabel('p(Choose High)')
    ax.set_title('Choose High by Post-switch Reward Rates')
    ax.set_xlim(-5, 8)
    ax.set_ylim(-0.05, 1.05)

    # p reward for stays
    ax = fig.add_subplot(gs[0,1])
    plot_utils.plot_x0line(ax=ax)
    for choice_type in reward_choices:
        raw_mat = np.vstack(p_reward_switch[choice_type]['stay'])
        avg, err = bah.get_rate_avg_err(raw_mat)
        if choice_type == 'all':
            ax.plot(x, avg, dashes=[4,4], c='k', label=choice_type)
        else:
            plot_utils.plot_shaded_error(x, avg, err, ax=ax, label=choice_type)

    ax.set_xlabel('Trials from decision to stay')
    ax.set_ylabel('p(Reward)')
    ax.set_title('Reward Probability Before Stay Choices')
    ax.set_xlim(-6, 5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', title='Choice Port\nReward Rate')

    # p reward for switches
    ax = fig.add_subplot(gs[1,1])
    plot_utils.plot_x0line(ax=ax)
    for choice_type in reward_choices:
        raw_mat = np.vstack(p_reward_switch[choice_type]['switch'])
        avg, err = bah.get_rate_avg_err(raw_mat)
        if choice_type == 'all':
            ax.plot(x, avg, dashes=[4,4], c='k', label=choice_type)
        else:
            plot_utils.plot_shaded_error(x, avg, err, ax=ax, label=choice_type)

    ax.set_xlabel('Trials from decision to switch')
    ax.set_ylabel('p(Reward)')
    ax.set_title('Reward Probability Before Switch Choices')
    ax.set_xlim(-6, 5)
    ax.set_ylim(-0.05, 1.05)

    # p reward for stays by block rate
    ax = fig.add_subplot(gs[0,2])
    plot_utils.plot_x0line(ax=ax)
    for br in block_rates:
        raw_mat = np.vstack(p_reward_switch_blocks[br]['all']['stay'])
        avg, err = bah.get_rate_avg_err(raw_mat)
        plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='{0}%'.format(br))

    ax.set_xlabel('Trials from decision to stay')
    ax.set_ylabel('p(Reward)')
    ax.set_title('Reward Probability Before Stay Choices')
    ax.set_xlim(-6, 5)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', title='Reward Rates')

    # p reward for switches by block rate
    ax = fig.add_subplot(gs[1,2])
    plot_utils.plot_x0line(ax=ax)
    for br in block_rates:
        raw_mat = np.vstack(p_reward_switch_blocks[br]['all']['switch'])
        avg, err = bah.get_rate_avg_err(raw_mat)
        plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='{0}%'.format(br))

    ax.set_xlabel('Trials from decision to switch')
    ax.set_ylabel('p(Reward)')
    ax.set_title('Reward Probability Before Switch Choices')
    ax.set_xlim(-6, 5)
    ax.set_ylim(-0.05, 1.05)

    # histograms of sequence lengths
    # high choices, same side
    ax = fig.add_subplot(gs[0,3])
    high_max = np.max([np.max(sequence_counts['high'][br]) for br in block_rates])
    low_max = np.max([np.max(sequence_counts['low'][br]) for br in block_rates])
    t_max = np.max([high_max, low_max])
    hist_args = dict(histtype='step', density=True, cumulative=True, bins=t_max, range=(1, t_max))
    for br in block_rates:
        ax.hist(sequence_counts['high'][br], **hist_args, label=br)

    ax.set_xlabel('# Trials')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('High Choice Sequence Lengths')
    ax.set_xlim(0, high_max)
    ax.legend(loc='lower right', title='Reward Rates')

    # low choices, same side
    ax = fig.add_subplot(gs[1,3])
    for br in block_rates:
        ax.hist(sequence_counts['low'][br], **hist_args, label=br)

    ax.set_xlabel('# Trials')
    ax.set_ylabel('Cumulative Proportion')
    ax.set_title('Low Choice Sequence Lengths')
    ax.set_xlim(0, low_max)



    # Simple Summary
    _, ax = plt.subplots(1, 1, figsize=(4,3))
    plot_utils.plot_x0line(ax=ax)
    raw_mat = np.vstack(p_reward_switch['all']['stay'])
    avg, err = bah.get_rate_avg_err(raw_mat)
    plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='Stay')
    raw_mat = np.vstack(p_reward_switch['all']['switch'])
    avg, err = bah.get_rate_avg_err(raw_mat)
    plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='Switch')

    ax.set_xlabel('Trials from current choice')
    ax.set_ylabel('p(Reward)')
    ax.set_title('Reward Probability Before Choice')
    ax.set_xlim(-4, 2)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='lower left', title='Choice')


# %% Logistic regression of choice by past choices and trial outcomes
separate_block_rates = True
n_back = 5
# whether to model the interaction as win-stay/lose switch or not
include_winstay = False
# whether to include reward as a predictor on its own
include_reward = False

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

    block_rates = np.sort(subj_sess_resp['block_prob'].unique())

    reg_groups = ['All']
    if separate_block_rates:
        reg_groups.extend(block_rates)

    label_list = ['choice ({})']
    if include_reward:
        label_list.append('reward ({})')

    if include_winstay:
        label_list.append('win-stay/lose-switch ({})')
    else:
        label_list.append('choice x reward ({})')

    predictor_labels = utils.flatten([[label.format(i) for label in label_list] for i in range(-1, -n_back-1, -1)])

    reg_results = []

    for reg_group in reg_groups:

        predictor_mat = []
        choice_mat = [] # will be 1 for choose left, 0 for choose right
        # choices need to be binary for the outcome in order for the model fitting to work
        # doesn't change outcome of fit if 0/1 or -1/+1

        # build predictor matrix
        for sess_id in sess_ids[subj_id]:
            ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

            if len(ind_sess) == 0:
                continue

            # reshape variables to columns [:,None]
            rewarded = ind_sess['rewarded'].to_numpy()[:,None].astype(int)
            choices = ind_sess['choice'].to_numpy()[:,None] == 'left'
            choices = choices.astype(int)
            # keep copy with choices as 1/0 for outcome matrix
            choice_outcome = choices.copy()
            # reformat choices to be -1/+1 for right/left predictors
            choices[choices == 0] = -1
            # create win-stay/lose_switch predictors
            winstay = rewarded.copy()
            winstay[winstay == 0] = -1
            winstay = winstay * choices

            # make buffered predictor vectors to build predictor matrix
            buffer = np.full((n_back-1,1), 0)
            buff_choices = np.concatenate((buffer, choices))
            buff_reward = np.concatenate((buffer, rewarded))
            buff_interaction = np.concatenate((buffer, rewarded * choices))
            buff_winstay = np.concatenate((buffer, winstay))

            # build predictor and outcome matrices
            if reg_group == 'All':

                choice_mat.append(choice_outcome[1:])

                # construct list of predictors to stack into matrix
                predictor_list = [buff_choices]
                if include_reward:
                    predictor_list.append(buff_reward)

                if include_winstay:
                    predictor_list.append(buff_winstay)
                else:
                    predictor_list.append(buff_interaction)

                # construct n-back matrix of predictors
                sess_mat = np.hstack([np.hstack([pred[i:-n_back+i] for pred in predictor_list])
                                          for i in range(n_back-1, -1, -1)])
                predictor_mat.append(sess_mat)
            else:
                block_rate = ind_sess['block_prob'].to_numpy()
                block_sel = block_rate == reg_group

                if sum(block_sel) == 0:
                    continue

                # find block probability transitions
                block_trans_idxs = np.where(np.diff(block_sel))[0]+1
                block_trans_idxs = np.insert(block_trans_idxs, 0, 0)
                block_trans_idxs = np.append(block_trans_idxs, len(ind_sess))

                for j in range(len(block_trans_idxs)-1):
                    # only build matrices if this is the start of the block
                    if block_sel[block_trans_idxs[j]]:

                        # always ignore the first trial in the block
                        choice_mat.append(choice_outcome[block_trans_idxs[j]+1:block_trans_idxs[j+1]])

                        # construct n-back matrices for choice/outcome pairs
                        block_choices = buff_choices[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back-1]
                        block_reward = buff_reward[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back-1]
                        block_interaction = buff_interaction[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back-1]
                        block_winstay = buff_winstay[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back-1]

                        predictor_list = [block_choices]
                        if include_reward:
                            predictor_list.append(block_reward)

                        if include_winstay:
                            predictor_list.append(block_winstay)
                        else:
                            predictor_list.append(block_interaction)

                        # construct n-back matrix of predictors
                        block_mat = np.hstack([np.hstack([pred[i:-n_back+i] for pred in predictor_list])
                                                  for i in range(n_back-1, -1, -1)])

                        predictor_mat.append(block_mat)

        predictor_mat = np.vstack(predictor_mat)
        choice_mat = np.vstack(choice_mat).reshape(-1)

        print('Subject {} Regression Results for {} trials:'.format(subj_id, reg_group))

        clf = lm.LogisticRegression().fit(predictor_mat, choice_mat)
        print('Regression, L2 penalty')
        print(np.concatenate((clf.intercept_, clf.coef_[0]))[:,None])
        print('Accuracy: {}%'.format(clf.score(predictor_mat, choice_mat)*100))
        print('R2: {}\n'.format(r2_score(choice_mat, clf.predict(predictor_mat))))

        # clf = lm.LogisticRegressionCV(cv=10).fit(predictor_mat, choice_mat)
        # print('Cross-validated regression, L2 penalty')
        # print(np.concatenate((clf.intercept_, clf.coef_[0]))[:,None])
        # print('Accuracy: {}%'.format(clf.score(predictor_mat, choice_mat)*100))
        # print('R2: {}\n'.format(r2_score(choice_mat, clf.predict(predictor_mat))))

        # clf = lm.LogisticRegression(penalty='l1', solver='liblinear').fit(predictor_mat, choice_mat)
        # print('Regression, L1 penalty')
        # print(np.concatenate((clf.intercept_, clf.coef_[0]))[:,None])
        # print('Accuracy: {}%'.format(clf.score(predictor_mat, choice_mat)*100))
        # print('R2: {}\n'.format(r2_score(choice_mat, clf.predict(predictor_mat))))

        # clf = lm.LogisticRegressionCV(cv=10, penalty='l1', solver='liblinear').fit(predictor_mat, choice_mat)
        # print('Cross-validated regression, L1 penalty')
        # print(np.concatenate((clf.intercept_, clf.coef_[0]))[:,None])
        # print('Accuracy: {}%'.format(clf.score(predictor_mat, choice_mat)*100))
        # print('R2: {}\n'.format(r2_score(choice_mat, clf.predict(predictor_mat))))

        predictors = pd.DataFrame(predictor_mat, columns=predictor_labels)
        predictors = sm.add_constant(predictors)
        fit_res = sm.Logit(choice_mat, predictors).fit()
        print(fit_res.summary())
        # mfx = fit_res.get_margeff()
        # print(mfx.summary())
        reg_results.append(fit_res)

    # plot regression coefficients over trials back
    fig, axs = plt.subplots(1, len(reg_groups), layout='constrained', figsize=(4*len(reg_groups), 4), sharey=True)
    fig.suptitle('Choice Regression Coefficients by Block Reward Rate (Rat {})'.format(subj_id))

    x_vals = np.arange(n_back)+1

    for i, group in enumerate(reg_groups):

        fit_res = reg_results[i]
        params = fit_res.params
        cis = fit_res.conf_int(0.05)

        ax = axs[i]
        ax.set_title('Block Rate: {}'.format(group))
        plot_utils.plot_dashlines(0, dir='h', ax=ax)

        # plot constant
        key = 'const'
        ax.errorbar(0, params[key], yerr=np.abs(cis.loc[key,:] - params[key]).to_numpy()[:,None], fmt='o', capsize=4, label='bias')

        row_labels = params.index.to_numpy()
        for j, pred_label in enumerate(label_list):
            pred_label = pred_label.replace(' ({})', '')
            pred_row_labels = [label for label in row_labels if pred_label == re.sub(r' \(.*\)', '', label)]

            pred_params = params[pred_row_labels].to_numpy()
            pred_cis = cis.loc[pred_row_labels,:].to_numpy()

            ax.errorbar(x_vals, pred_params, yerr=np.abs(pred_cis - pred_params[:,None]).T, fmt='o-', capsize=4, label=pred_label)

        if i == 0:
            ax.set_ylabel('Regresssion Coefficient for Choosing Left')
        ax.set_xlabel('Trials Back')
        ax.legend(loc='best')


# %% Response TImes

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]

    # organize time to reward and time to engage center port by prior reward, block probabilities, and whether switched response
    timing_data = pd.DataFrame(columns=['RT', 'cpoke_in_time', 'cpoke_out_latency', 'choice', 'reward_type', 'response_type', 'block_prob', 'all'])

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess[subj_sess['sessid'] == sess_id].reset_index(drop=True)

        if len(ind_sess) == 0:
            continue

        data = ind_sess[['RT', 'cpoke_in_time', 'cpoke_out_latency', 'choice', 'block_prob']].iloc[1:].reset_index(drop=True) # ignore first trial because no trial before
        # add reward and response type labels
        choices = ind_sess['choice'].to_numpy()
        stays = choices[:-1] == choices[1:]

        data['reward_type'] = ind_sess['rewarded'].iloc[:-1].apply(lambda x: 'prev rewarded' if x else 'prev unrewarded').reset_index(drop=True)
        # NEED TO UPDATE FOR STAY/SWITCH
        data['response_type'] = pd.Series(stays).apply(lambda x: 'stay' if x else 'switch')
        # ignore trials where the current response or the past response was none
        data = data[(choices[:-1] != 'none') & (choices[1:] != 'none')]
        data['all'] = 'All'

        timing_data = pd.concat([timing_data, data], ignore_index=True)

    # collapse reward type and response type into one column and add 'all' to block_prob column
    timing_data = pd.lreshape(timing_data, {'type': ['reward_type', 'response_type', 'choice']})
    timing_data = pd.lreshape(timing_data, {'prob': ['block_prob', 'all']})

    # fig = plt.figure(layout='constrained', figsize=(8, 7))
    # fig.suptitle('Response Latencies')
    # gs = GridSpec(2, 1, figure=fig)

    # # first row is response times
    # ax = fig.add_subplot(gs[0])
    # sb.violinplot(data=timing_data, x='prob', y='RT', hue='type', inner='quart', ax=ax) # split=True, gap=.1,
    # ax.set_xlabel('Block Reward Probability (High/Low %)')
    # ax.set_ylabel('Response Latency (s)')
    # ax.set_title('Time from Cue to Response')

    # # first row is cpoke times
    # ax = fig.add_subplot(gs[1])
    # sb.violinplot(data=timing_data[timing_data['cpoke_in_time'] < 60], x='prob', y='cpoke_in_time', hue='type', inner='quart', ax=ax)
    # ax.set_xlabel('Block Reward Probability (High/Low %)')
    # ax.set_ylabel('Next Trial Latency (s)')
    # ax.set_title('Time to Engage Center Port on Next Trial')

    # show box plots
    fig = plt.figure(layout='constrained', figsize=(8, 9))
    fig.suptitle('Response Latencies (Rat {})'.format(subj_id))
    gs = GridSpec(3, 1, figure=fig)

    # first row is response times
    ax = fig.add_subplot(gs[0])
    sb.boxplot(data=timing_data, x='prob', y='RT', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Probability (High/Low %)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Time from Cue to Response')
    ax.legend(ncol=3, title='Trial Type', loc='upper left')

    ax = fig.add_subplot(gs[1])
    sb.boxplot(data=timing_data[timing_data['cpoke_out_latency'] < 1], x='prob', y='cpoke_out_latency', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Probability (High/Low %)')
    ax.set_ylabel('Center Poke Out Latency (s)')
    ax.set_title('Time from Cue to Center Poke Out')
    ax.legend(ncol=3, title='Trial Type', loc='upper left')

    # second row is cpoke times
    ax = fig.add_subplot(gs[2])
    sb.boxplot(data=timing_data[timing_data['cpoke_in_time'] < 60], x='prob', y='cpoke_in_time', hue='type', ax=ax)
    ax.set_xlabel('Block Reward Probability (High/Low %)')
    ax.set_ylabel('Next Trial Latency (s)')
    ax.set_title('Time to Engage Center Port on Next Trial')
    ax.legend(ncol=3, title='Trial Type', loc='upper left')

# %% Effect of inter trial interval on choice behavior

# look at relationship between previous reward and future choice (stay/switch) based on how long it took to start the next trial

t_thresh = 60
# create uneven bins tiling the cpoke in times so bins have roughly the same amount of data points
bin_edges = np.concatenate((np.arange(0, 10, 2), np.arange(10, t_thresh, 20), [np.inf]))
# get bins output by pandas for indexing
bins = pd.IntervalIndex.from_breaks(bin_edges)

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

    # create structures to accumulate event data across sessions
    stay_reward = {'k': np.zeros(len(bins)), 'n': np.zeros(len(bins))}
    stay_noreward = copy.deepcopy(stay_reward)

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

        if len(ind_sess) == 0:
            continue

        cpoke_in_bins = pd.cut(ind_sess['cpoke_in_latency'], bins)[1:] # ignore first trial time since no trial before

        choices = ind_sess['choice'].to_numpy()
        stays = choices[:-1] == choices[1:]
        rewarded = ind_sess['rewarded'].to_numpy()[:-1] # ignore last trial reward since no trial after

        for i, b in enumerate(bins):
            bin_sel = cpoke_in_bins == b
            stay_reward['k'][i] += sum(rewarded & stays & bin_sel)
            stay_reward['n'][i] += sum(rewarded & bin_sel)
            stay_noreward['k'][i] += sum(~rewarded & stays & bin_sel)
            stay_noreward['n'][i] += sum(~rewarded & bin_sel)

    # plot results
    # define reusable helper methods
    def comp_p(n_dict): return n_dict['k']/n_dict['n']
    def comp_err(n_dict): return abs(np.array([utils.binom_cis(n_dict['k'][i], n_dict['n'][i]) for i in range(len(n_dict['k']))]) - comp_p(n_dict)[:,None])

    # first row, left, is the win-stay/lose-switch rates by choice probability
    stay_reward_vals = comp_p(stay_reward)
    stay_reward_err = comp_err(stay_reward).T
    stay_noreward_vals = comp_p(stay_noreward)
    stay_noreward_err = comp_err(stay_noreward).T

    # x = (bin_edges[:-1] + bin_edges[1:])/2
    # x[-1] = t_thresh
    x = np.arange(len(bins))

    _, ax = plt.subplots(1, 1, layout='constrained', figsize=(6, 4))
    ax.set_title('Stay Probabilities by Trial Latency (Rat {})'.format(subj_id))

    ax.errorbar(x, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Rewarded')
    ax.errorbar(x, stay_noreward_vals, yerr=stay_noreward_err, fmt='o', capsize=4, label='Unrewarded')
    ax.set_ylabel('p(stay)')
    ax.set_xlabel('Next Trial Latency (s)')
    ax.set_xticks(x, ['{:.0f} - {:.0f}'.format(b.left, b.right) for b in bins])
    ax.set_ylim(-0.05, 1.05)
    ax.grid(axis='y')
    ax.legend(loc='best')
