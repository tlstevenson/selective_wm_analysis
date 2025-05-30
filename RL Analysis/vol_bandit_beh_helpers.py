# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 12:03:59 2025

@author: tanne
"""

# %% Imports

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

# %% Trial History Labels

def make_trial_hist_labels(sess_data, n_back=3):
    ''' Make labels to represent the prior choice/outcome history.
        Labels go left to right for successive trials back (i.e. t-1, t-2, t-3, ...) and have a capital letter for rewarded and lowercase for unrewarded
        The first label is always 'A' or 'a' representing that we are comparing all choices further back in time to the immediately previous choice (t-1)
        and the subsequent labels are A/a for choices to the same side as t-1 choice or B/b for choices to the different side as the t-1 choice
    '''
    
    if not bah.trial_hist_exists(sess_data):
        bah.calc_trial_hist(sess_data, n_back=n_back)
        
    sess_data['trial_hist_label ({})'.format(n_back)] = sess_data.apply(lambda x: _get_trial_hist_label(x['choice_hist'], x['rew_hist'], n_back), axis=1)
    
    
def _get_trial_hist_label(choice_hist, rew_hist, n_back):
    label = []
    same_choice_hist = choice_hist == choice_hist[0]
    
    for i in range(n_back):
        if np.isnan(rew_hist[i]):
            label.append('_')
        else:
            if same_choice_hist[i]:
                if rew_hist[i] == 1:
                    label.append('A')
                else:
                    label.append('a')
            else:
                if rew_hist[i] == 1:
                    label.append('B')
                else:
                    label.append('b')
    
    return ''.join(label)

def get_unique_hist_labels(sess_data, col, n_back):
    
    #hist_labels = sorted(np.unique(sess_data['{} ({})'.format(col, n_back)]), key=lambda s: (s.lower(), s))
    hist_labels = np.unique(sess_data['{} ({})'.format(col, n_back)])
    # remove history labels with missing history elements denoted by '_'
    hist_labels = hist_labels[np.char.find(hist_labels.astype(str), '_') == -1]
    
    return hist_labels

def make_rew_hist_labels(sess_data, n_back=3):
    ''' Make labels to represent the prior outcome history.
        Labels go left to right for successive trials back (i.e. t-1, t-2, t-3, ...) and have 'R' for rewarded and 'U' for unrewarded
    '''
    
    if not bah.trial_hist_exists(sess_data):
        bah.calc_trial_hist(sess_data, n_back=n_back)
        
    sess_data['rew_hist_label ({})'.format(n_back)] = sess_data['rew_hist'].apply(lambda x: _get_rew_hist_label(x, n_back))
    
    
def _get_rew_hist_label(rew_hist, n_back):
    label = []

    for i in range(n_back):
        if np.isnan(rew_hist[i]):
            label.append('_')
        else:
            if rew_hist[i] == 1:
                label.append('R')
            else:
                label.append('U')
    
    return ''.join(label)
    
# %% Trial Counts

def count_block_lengths(sess_data, meta_subj=True, ind_subj=True, block_bin_width=2, epoch_bin_width=1, 
                        plot_all_block=False, plot_ind_block=True, plot_suffix=''):
    
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
        
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none'] 
        
        block_types = np.unique(subj_sess_resp['epoch_block_label'])
        block_lengths = {bt: [] for bt in block_types}
        block_lengths['all'] = []
        consec_blocks = {bt: [] for bt in block_types}

        for sess_id in subj_sess_ids:
            ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

            if len(ind_sess) == 0:
                continue

            # find last block trial number before switch
            # ignore the last block since it likely ended prematurely
            block_switch_trial_idx = np.nonzero(ind_sess['block_trial'] == 1)[0][1:]-1
            
            block_type = ind_sess['epoch_block_label'].to_numpy()
            prev_block_type = block_type[0]
            consec_block = 0
            
            for i in block_switch_trial_idx:
                length = ind_sess.iloc[i]['block_trial']
                block_lengths[block_type[i]].append(length)
                block_lengths['all'].append(length)
                
                if block_type[i] == prev_block_type:
                    consec_block += 1
                else:
                    consec_blocks[prev_block_type].append(consec_block)
                    consec_block = 1
                    prev_block_type = block_type[i]
            
            # count the last epoch's trials
            consec_blocks[prev_block_type].append(consec_block)
                
        # plot results
        fig, axs = plt.subplots(1,2,layout='constrained', figsize=(8,3))
        
        ax = axs[0]
        min_len = np.min(block_lengths['all'])
        max_len = np.max(block_lengths['all'])
        bins = np.arange(min_len, max_len+block_bin_width, block_bin_width)
        hist_args = dict(histtype='bar', density=True, cumulative=False, bins=bins, alpha=0.5)
        
        if plot_all_block:
            ax.hist(block_lengths['all'], **hist_args, label='All', color='black')
        if plot_ind_block:
            for bt in block_types:
                ax.hist(block_lengths[bt], **hist_args, label=bt)
            
        ax.set_title('Block Trial Lengths ({}){}'.format(subj, plot_suffix))
        ax.set_xlabel('Block Length (trials)')
        ax.set_ylabel('Proportion')
        ax.legend()
        
        ax = axs[1]
        min_len = np.min([np.min(consec_blocks[bt]) for bt in block_types])
        max_len = np.max([np.max(consec_blocks[bt]) for bt in block_types])
        bins = np.arange(min_len, max_len+epoch_bin_width, epoch_bin_width)
        hist_args = dict(histtype='bar', density=True, cumulative=False, bins=bins, alpha=0.5)

        for bt in block_types:
            ax.hist(consec_blocks[bt], **hist_args, label=bt)
            
        ax.set_title('Block Epoch Lengths ({}){}'.format(subj, plot_suffix))
        ax.set_xlabel('Epoch Length (blocks)')
        ax.set_ylabel('Proportion')
        ax.legend()
    
# %% Choice Probability Methods

def analyze_choice_behavior(sess_data, plot_simple_summary=False, meta_subj=True, ind_subj=True, plot_suffix=''):
    
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
    
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]

        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        # TO DO: use previous epoch side label
        choose_left_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_left', ['prev_high_side', 'epoch_side_label'])
        choose_right_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_right', ['prev_high_side'])
        choose_high_trial_probs = bah.get_rate_dict(subj_sess_resp, 'chose_prev_high', ['epoch_block_label'])
        
        # plot simple rates
        fig = plt.figure(layout='constrained', figsize=(6, 6))
        fig.suptitle('Response Probabilities ({}){}'.format(subj, plot_suffix))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[2,1])

        # choose high for each trial probability
        ax = fig.add_subplot(gs[0, 0])

        data = choose_high_trial_probs['epoch_block_label']
        x_labels = data['epoch_block_label']
        x_vals = np.arange(len(x_labels))

        ax.errorbar(x_vals, data['rate'], yerr=bah.convert_rate_err_to_mat(data), fmt='o', capsize=4)
        ax.set_ylabel('p(Choose High)')
        ax.set_xlabel('Block Epoch Label (Stoch/Vol-High/Low Mean)')
        ax.set_xticks(x_vals, x_labels)
        ax.set_xlim(-0.5, len(x_vals)-0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y')
        ax.set_title('Choose High Side')

        # choose side when side is high/low
        ax = fig.add_subplot(gs[0, 1])

        left_data = choose_left_side_probs['prev_high_side']
        right_data = choose_right_side_probs['prev_high_side']
        x_labels = left_data['prev_high_side']
        x_vals = [1,2]

        ax.errorbar(x_vals, left_data['rate'], yerr=bah.convert_rate_err_to_mat(left_data), fmt='o', capsize=4, label='Left Choice')
        ax.errorbar(x_vals, right_data['rate'], yerr=bah.convert_rate_err_to_mat(right_data), fmt='o', capsize=4, label='Right Choice')
        ax.set_ylabel('p(Choose Side)')
        ax.set_xlabel('High Volume Side')
        ax.set_xticks(x_vals, x_labels)
        ax.set_xlim(0.5, 2.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y')
        ax.legend(loc='best')
        ax.set_title('Choose Side')

        # choose left by left probability
        ax = fig.add_subplot(gs[1, :])

        data = choose_left_side_probs['epoch_side_label']
        x_labels = data['epoch_side_label']
        x_vals = np.arange(len(x_labels))

        ax.errorbar(x_vals, data['rate'], yerr=bah.convert_rate_err_to_mat(data), fmt='o', capsize=4)
        ax.set_ylabel('p(Choose Left)')
        ax.set_xlabel('Side Epoch Label (Stoch/Vol-Left/Right Mean)')
        ax.set_xticks(x_vals, x_labels)
        ax.set_xlim(-0.5, len(x_vals)-0.5)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y')
        ax.set_title('Choose Left')


        # SWITCHING PROBABILITIES
        all_switch_trial_probs = bah.get_rate_dict(subj_sess_resp, 'next_switch', [['epoch_block_label', 'reward']])
        switch_trial_probs = bah.get_rate_dict(subj_sess_resp, 'next_switch', [['epoch_block_label', 'chose_prev_high', 'reward']])
        all_switch_trial_probs = all_switch_trial_probs['epoch_block_label x reward']
        switch_trial_probs = switch_trial_probs['epoch_block_label x chose_prev_high x reward']
        block_types = np.unique(switch_trial_probs['epoch_block_label'])
        
        n_rows = len(block_types)
        n_cols = 2

        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(3.5*n_cols, 3.5*n_rows), width_ratios=[2,1])
        axs = axs.reshape(n_rows, n_cols)
        
        fig.suptitle('Switch Psychometrics by Choice and Reward Volume ({}){}'.format(subj, plot_suffix))

        for i, bt in enumerate(block_types):
            all_bt_data = all_switch_trial_probs[all_switch_trial_probs['epoch_block_label'] == bt]
            bt_data = switch_trial_probs[switch_trial_probs['epoch_block_label'] == bt]
            
            # left is the psychometric curve of switch rate by reward volume
            ax = axs[i,0]
            plot_utils.plot_shaded_error(all_bt_data['reward'], all_bt_data['rate'], bah.convert_rate_err_to_mat(all_bt_data), color='k', label='All', ax=ax)
            for chose_high in [True, False]:
                sub_data = bt_data[bt_data['chose_prev_high'] == chose_high]
                label = 'High' if chose_high else 'Low'
                ax.errorbar(sub_data['reward'], sub_data['rate'], yerr=bah.convert_rate_err_to_mat(sub_data), fmt='o', capsize=4, label=label, alpha=0.6)
                
            ax.set_ylabel('p(Switch)')
            ax.set_xlabel('Reward Volume (μL)')
            ax.set_ylim(-0.05, 1.05)
            ax.grid(axis='y')
            ax.set_title('Next Trial Switch Probabilities - {}'.format(bt))
            ax.legend(title='Previous Value of Choice')
            
            # right is the trial count histogram
            ax = axs[i,1]
            for chose_high in [True, False]:
                sub_data = bt_data[bt_data['chose_prev_high'] == chose_high]
                label = 'High' if chose_high else 'Low'
                ax.bar(sub_data['reward'], sub_data['n'], label=label, width=1, alpha=0.6)
                
            ax.set_ylabel('# Trials')
            ax.set_xlabel('Reward Volume (μL)')
            ax.grid(axis='y')
            ax.set_title('Trial Counts')
            ax.legend(title='Previous Value of Choice')
            
            
        fig, ax = plt.subplots(1, 1, layout='constrained', figsize=(4, 3))
        
        fig.suptitle('Switch Psychometrics by Epoch Type ({}){}'.format(subj, plot_suffix))

        for i, bt in enumerate(block_types):
            all_bt_data = all_switch_trial_probs[all_switch_trial_probs['epoch_block_label'] == bt]

            plot_utils.plot_shaded_error(all_bt_data['reward'], all_bt_data['rate'], bah.convert_rate_err_to_mat(all_bt_data), label=bt, ax=ax)
            
        ax.set_ylabel('p(Switch)')
        ax.set_xlabel('Reward Volume (μL)')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y')
        ax.legend(title='Epoch Type')


# %% Choice Probabilities over trials

def analyze_trial_choice_behavior(sess_data, plot_simple_summary=False, meta_subj=True, ind_subj=True, plot_suffix=''):
    
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
    
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        block_types = np.unique(subj_sess_resp['epoch_block_label'])

        # Investigate how animals choose to stay/switch their response
        # Probability of choosing high port/getting reward pre and post block change by block reward
        # probability of getting reward leading up to stay/switch by port probability (high/low)
        # numbers of low/high reward rate choices in a row that are rewarded/unrewarded/any
        n_away = 10
        p_choose_high_blocks = {bt: {'pre': [], 'post': []} for bt in block_types}
        p_choose_high_blocks['all'] = []
        
        n_before_switch = {bt: {'pre': [], 'post': []} for bt in block_types}

        reward_choices = ['high', 'low', 'all']
        reward_switch = {c: {'stay': [], 'switch': []} for c in reward_choices}
        reward_switch_blocks = {bt: {c: {'stay': [], 'switch': []} for c in reward_choices} for bt in block_types}

        for sess_id in subj_sess_ids:
            ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

            if len(ind_sess) == 0:
                continue

            reward = ind_sess['reward'].to_numpy()
            chose_high = ind_sess['chose_high'].fillna(np.nan).to_numpy()
            chose_prev_high = ind_sess['chose_prev_high'].fillna(np.nan).to_numpy()
            next_switch = ind_sess['next_switch'].fillna(np.nan).to_numpy()
            block_type = ind_sess['epoch_block_label'].to_numpy()
            block_switch_idxs = np.where(np.diff(ind_sess['block_num']) > 0)[0] + 1
            pre_switch_rates = block_type[block_switch_idxs-1]
            post_switch_rates = block_type[block_switch_idxs]

            # block switches by epoch type
            for i, switch_idx in enumerate(block_switch_idxs):
                pre_bt = pre_switch_rates[i]
                post_bt = post_switch_rates[i]

                choose_high_switch = np.full(n_away*2+1, np.nan)
                if i == 0:
                    pre_switch_mask_dist = np.minimum(n_away, switch_idx)
                else:
                    pre_switch_mask_dist = np.minimum(n_away, switch_idx - block_switch_idxs[i-1])

                if i == len(block_switch_idxs)-1:
                    post_switch_mask_dist = np.minimum(n_away+1, len(ind_sess) - switch_idx)
                else:
                    post_switch_mask_dist = np.minimum(n_away+1, block_switch_idxs[i+1] - switch_idx)

                choose_high_switch[n_away-pre_switch_mask_dist : n_away+post_switch_mask_dist] = chose_high[switch_idx-pre_switch_mask_dist : switch_idx+post_switch_mask_dist]

                p_choose_high_blocks[pre_bt]['pre'].append(choose_high_switch)
                p_choose_high_blocks[post_bt]['post'].append(choose_high_switch)
                p_choose_high_blocks['all'].append(choose_high_switch)
                
                # count number of trials before switch after block switch
                n_stays_post_switch = 0
                chose_low = ~chose_high[switch_idx]
                while chose_low:
                    n_stays_post_switch += 1
                    if (switch_idx + n_stays_post_switch) == len(chose_high):
                        break
                    chose_low = ~chose_high[switch_idx + n_stays_post_switch]
                    
                n_before_switch[pre_bt]['pre'].append(n_stays_post_switch)
                n_before_switch[post_bt]['post'].append(n_stays_post_switch)

            # volume of reward before a stay/switch decision
            # first construct reward matrix where center column is the current trial, starting from second trial
            nan_buffer = np.full(n_away, np.nan)
            buff_reward = np.concatenate((nan_buffer, reward, nan_buffer))
            reward_mat = np.hstack([buff_reward[i:i+len(reward), None] for i in range(n_away*2+1)])
            # group by choice across all blocks
            for choice_type in reward_choices:
                match choice_type:
                    case 'high':
                        sel = chose_high == True
                    case 'low':
                        sel = chose_high == False
                    case 'all':
                        sel = np.full_like(chose_high, True)

                reward_switch[choice_type]['stay'].append(reward_mat[sel & (next_switch == False), :])
                reward_switch[choice_type]['switch'].append(reward_mat[sel & (next_switch == True), :])

            # group by block of choice
            for bt in block_types:
                block_sel = block_type == bt
                for choice_type in reward_choices:
                    match choice_type:
                        case 'high':
                            sel = chose_high == True
                        case 'low':
                            sel = chose_high == False
                        case 'all':
                            sel = np.full_like(chose_high, True)

                    reward_switch_blocks[bt][choice_type]['stay'].append(reward_mat[block_sel & sel & (next_switch == False), :])
                    reward_switch_blocks[bt][choice_type]['switch'].append(reward_mat[block_sel & sel & (next_switch == True), :])

        # plot switching information
        fig = plt.figure(layout='constrained', figsize=(15, 8))
        fig.suptitle('Switching Metrics ({}){}'.format(subj, plot_suffix))
        gs = GridSpec(2, 3, figure=fig)

        x = np.arange(-n_away, n_away+1)

        # p choose high grouped by prior block rate
        ax = fig.add_subplot(gs[0,0])
        plot_utils.plot_x0line(ax=ax)
        for bt in block_types:
            raw_mat = np.asarray(p_choose_high_blocks[bt]['post'])
            avg, err = bah.get_rate_avg_err(raw_mat)
            plot_utils.plot_shaded_error(x, avg, err, ax=ax, label=bt)

        ax.plot(x, np.nanmean(np.asarray(p_choose_high_blocks['all']), axis=0), dashes=[4,4], c='k', label='all')

        ax.set_xlabel('Trials from block switch')
        ax.set_ylabel('p(Choose High)')
        ax.set_title('Choose High by Epoch Type')
        ax.set_xlim(-6, n_away)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc='lower left', title='Epoch Type')

        # histograms of choices sequence length before switch after block change
        ax = fig.add_subplot(gs[1,0])

        t_max = np.max([np.max(n_before_switch[bt]['post']) for bt in block_types])
        hist_args = {'histtype': 'step', 'density': True, 'cumulative': True, 'bins': t_max, 'range': (0, t_max)}
        for bt in block_types:
            ax.hist(n_before_switch[bt]['post'], **hist_args, label=bt)

        ax.set_xlabel('# Trials after Block Switch')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Trials to Switch by Epoch Type')
        ax.set_xlim(-0.5, t_max)
        ax.legend(loc='lower right', title='Epoch Type')
        
        # avg reward before stays/switches for high choices
        ax = fig.add_subplot(gs[0,1])
        plot_utils.plot_x0line(ax=ax)
        plot_choices = ['stay', 'switch']
        for choice_type in plot_choices:
            raw_mat = np.vstack(reward_switch['high'][choice_type])
            avg = np.nanmean(raw_mat, axis=0)
            se = utils.stderr(raw_mat, axis=0)
            plot_utils.plot_shaded_error(x, avg, se, ax=ax, label=choice_type)

        ax.set_xlabel('Trials before stay/switch decision (next trial)')
        ax.set_ylabel('Reward (μL)')
        ax.set_title('High Option Reward Before Stay/Switch')
        ax.set_xlim(-8, 8)
        ax.legend(loc='lower left', title='Choice')

        # avg reward before switches by choice
        ax = fig.add_subplot(gs[1,1])
        plot_utils.plot_x0line(ax=ax)
        for choice_type in plot_choices:
            raw_mat = np.vstack(reward_switch['low'][choice_type])
            avg = np.nanmean(raw_mat, axis=0)
            se = utils.stderr(raw_mat, axis=0)
            plot_utils.plot_shaded_error(x, avg, se, ax=ax, label=choice_type)

        ax.set_xlabel('Trials before stay/switch decision (next trial)')
        ax.set_ylabel('Reward (μL)')
        ax.set_title('Low Option Reward Before Stay/Switch')
        ax.set_xlim(-8, 8)
        
        # avg reward before stays by block rate
        ax = fig.add_subplot(gs[0,2])
        plot_utils.plot_x0line(ax=ax)
        for bt in block_types:
            raw_mat = np.vstack(reward_switch_blocks[bt]['all']['stay'])
            avg = np.nanmean(raw_mat, axis=0)
            se = utils.stderr(raw_mat, axis=0)
            plot_utils.plot_shaded_error(x, avg, se, ax=ax, label=bt)

        ax.set_xlabel('Trials from decision to stay')
        ax.set_ylabel('Reward (μL)')
        ax.set_title('Reward Before Stay Choices')
        ax.set_xlim(-8, 8)
        ax.legend(loc='lower left', title='Block Type')

        # avg reward before switches by block rate
        ax = fig.add_subplot(gs[1,2])
        plot_utils.plot_x0line(ax=ax)
        for bt in block_types:
            raw_mat = np.vstack(reward_switch_blocks[bt]['all']['switch'])
            avg = np.nanmean(raw_mat, axis=0)
            se = utils.stderr(raw_mat, axis=0)
            plot_utils.plot_shaded_error(x, avg, se, ax=ax, label=bt)

        ax.set_xlabel('Trials from decision to switch')
        ax.set_ylabel('Reward (μL)')
        ax.set_title('Reward Before Switch Choices')
        ax.set_xlim(-8, 8)
        
        # break out reward before stay/switch by port and epoch type
        n_rows = len(block_types)
        n_cols = 2

        fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=(5*n_cols, 4*n_rows))
        axs = axs.reshape(n_rows, n_cols)
        
        fig.suptitle('Reward Volumes Before Stay/Switch by Port, Choice, and Epoch ({}){}'.format(subj, plot_suffix))

        for i, bt in enumerate(block_types):
            epoch_data = reward_switch_blocks[bt]
            
            # avg reward before stays/switches for high choices
            ax = axs[i,0]
            plot_utils.plot_x0line(ax=ax)
            plot_choices = ['stay', 'switch']
            for choice_type in plot_choices:
                raw_mat = np.vstack(epoch_data['high'][choice_type])
                avg = np.nanmean(raw_mat, axis=0)
                se = utils.stderr(raw_mat, axis=0)
                plot_utils.plot_shaded_error(x, avg, se, ax=ax, label=choice_type)

            ax.set_xlabel('Trials before stay/switch decision (next trial)')
            ax.set_ylabel('Reward (μL)')
            ax.set_title('High Option Avg Reward - {}'.format(bt))
            ax.set_xlim(-8, 8)
            ax.legend(loc='lower left', title='Choice')

            # avg reward before switches by choice
            ax = axs[i,1]
            plot_utils.plot_x0line(ax=ax)
            for choice_type in plot_choices:
                raw_mat = np.vstack(epoch_data['low'][choice_type])
                avg = np.nanmean(raw_mat, axis=0)
                se = utils.stderr(raw_mat, axis=0)
                plot_utils.plot_shaded_error(x, avg, se, ax=ax, label=choice_type)

            ax.set_xlabel('Trials before stay/switch decision (next trial)')
            ax.set_ylabel('Reward (μL)')
            ax.set_title('Low Option Avg Reward - {}'.format(bt))
            ax.set_xlim(-8, 8)
            ax.legend(loc='lower left', title='Choice')
        
        # # Simple Summary
        # if plot_simple_summary:
            
        #     fig, axs = plt.subplots(1, 2, figsize=(7,3), layout='constrained')
        #     fig.suptitle('Choice Probabilities Over Trials ({}){}'.format(subj, plot_suffix))
            
        #     # left plot is average p(choose high) aligned to block switches
        #     ax = axs[0]
        #     raw_mat = np.asarray(p_choose_high_blocks['all'])
        #     avg, err = bah.get_rate_avg_err(raw_mat)
        #     plot_utils.plot_x0line(ax=ax)
        #     plot_utils.plot_shaded_error(x, avg, err, ax=ax)

        #     ax.set_xlabel('Trials from block switch')
        #     ax.set_ylabel('p(Choose High)')
        #     ax.set_title('Choose High at Block Switch')
        #     ax.set_xlim(-6, n_away)
        #     ax.set_ylim(-0.05, 1.05)
            
            
        #     # right plot is reward probability aligned to switching choices
        #     ax = axs[1]
        #     plot_utils.plot_x0line(ax=ax)
            
        #     raw_mat = np.vstack(reward_switch['all']['stay'])
        #     avg, err = bah.get_rate_avg_err(raw_mat)
        #     plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='Stay')
            
        #     raw_mat = np.vstack(reward_switch['all']['switch'])
        #     avg, err = bah.get_rate_avg_err(raw_mat)
        #     plot_utils.plot_shaded_error(x, avg, err, ax=ax, label='Switch')
    
        #     ax.set_xlabel('Trials from current choice')
        #     ax.set_ylabel('p(Reward)')
        #     ax.set_title('Reward Probability Before Choice')
        #     ax.set_xlim(-6, 6)
        #     ax.set_ylim(-0.05, 1.05)
        #     ax.legend(loc='lower left', title='Choice')


# %% Choice Regressions

def logit_regress_side_choice(sess_data, n_back=5, separate_block_types=True, include_winstay=False, include_reward=False, 
                              separate_unreward=False, include_full_interaction=False, plot_cis=True, ind_subj=True, meta_subj=True, plot_suffix=''):

    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
    
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        block_types = np.sort(subj_sess_resp['block_prob'].unique())

        reg_groups = ['All']
        if separate_block_types:
            reg_groups.extend(block_types)
        
        label_list = []
        
        if include_full_interaction:
            label_list.append('left/rewarded ({})')
            label_list.append('left/unrewarded ({})')
            #label_list.append('right/rewarded ({})')
            label_list.append('right/unrewarded ({})')
        else:
            label_list.append('choice ({})')
            if include_reward:
                label_list.append('reward ({})')
        
            if separate_unreward:
                label_list.append('reward ({})')
                label_list.append('unreward ({})')
            else:
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
            for sess_id in subj_sess_ids:
                ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

                if len(ind_sess) == 0:
                    continue

                # reshape variables to columns [:,None]
                rewarded = ind_sess['rewarded'].to_numpy()[:,None].astype(int)
                unrewarded = (~ind_sess['rewarded'].to_numpy()[:,None]).astype(int)
                chose_left = (ind_sess['choice'].to_numpy()[:,None] == 'left').astype(int)
                chose_right = (ind_sess['choice'].to_numpy()[:,None] == 'right').astype(int)
                # keep copy with choices as 1/0 for outcome matrix
                choice_outcome = chose_left.copy()
                # reformat choices to be -1/+1 for right/left predictors
                choices = chose_left.copy()
                choices[choices == 0] = -1
                # create win-stay/lose_switch predictors of -1/+1 based on choice and outcome (-1 for rewarded/right or unrewarded/left and vice versa)
                winstay = rewarded.copy()
                winstay[winstay == 0] = -1
                winstay = winstay * choices
                # create rewarded/unrewarded only predictors +1/-1 for left/right for each outcome
                rew_only = rewarded * choices # +1/-1 for rewarded left/right trials, 0 for unrewarded
                unrew_only = unrewarded * choices # +1/-1 for unrewarded left/right trials, 0 for rewarded

                # make buffered predictor vectors to build predictor matrix
                buffer = np.full((n_back-1,1), 0)
                buff_choices = np.concatenate((buffer, choices))
                buff_reward = np.concatenate((buffer, rewarded))
                buff_winstay = np.concatenate((buffer, winstay))
                buff_rew_only = np.concatenate((buffer, rew_only))
                buff_unrew_only = np.concatenate((buffer, unrew_only))
                buff_left_rew = np.concatenate((buffer, chose_left*rewarded))
                buff_right_rew = np.concatenate((buffer, chose_right*rewarded))
                buff_left_unrew = np.concatenate((buffer, chose_left*unrewarded))
                buff_right_unrew = np.concatenate((buffer, chose_right*unrewarded))

                # build predictor and outcome matrices
                if reg_group == 'All':

                    choice_mat.append(choice_outcome[1:])

                    # construct list of predictors to stack into matrix
                    predictor_list = []
                    if include_full_interaction:
                        predictor_list.append(buff_left_rew)
                        predictor_list.append(buff_left_unrew)
                        #predictor_list.append(buff_right_rew)
                        predictor_list.append(buff_right_unrew)
                    else:
                        predictor_list.append(buff_choices)
                        if include_reward:
                            predictor_list.append(buff_reward)
        
                        if separate_unreward:
                            predictor_list.append(buff_rew_only)
                            predictor_list.append(buff_unrew_only)
                        else:
                            if include_winstay:
                                predictor_list.append(buff_winstay)
                            else:
                                predictor_list.append(buff_rew_only)

                    # construct n-back matrix of predictors
                    sess_mat = np.hstack([np.hstack([pred[i:-n_back+i] for pred in predictor_list])
                                              for i in range(n_back-1, -1, -1)])
                    predictor_mat.append(sess_mat)
                else:
                    block_type = ind_sess['block_prob'].to_numpy()
                    block_sel = block_type == reg_group

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
                            choice_mat.append(choice_outcome[block_trans_idxs[j]+1:block_trans_idxs[j+1]+1])

                            # construct n-back matrices for choice/outcome pairs
                            block_choices = buff_choices[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_reward = buff_reward[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_winstay = buff_winstay[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_rew_only = buff_rew_only[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_unrew_only = buff_unrew_only[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_left_rew = buff_left_rew[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_left_unrew = buff_left_unrew[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_right_rew = buff_right_rew[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_right_unrew = buff_right_unrew[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]

                            predictor_list = []
                            if include_full_interaction:
                                predictor_list.append(block_left_rew)
                                predictor_list.append(block_left_unrew)
                                #predictor_list.append(block_right_rew)
                                predictor_list.append(block_right_unrew)
                            else:
                                predictor_list.append(block_choices)
                                if include_reward:
                                    predictor_list.append(block_reward)
        
                                if separate_unreward:
                                    predictor_list.append(block_rew_only)
                                    predictor_list.append(block_unrew_only)
                                else:
                                    if include_winstay:
                                        predictor_list.append(block_winstay)
                                    else:
                                        predictor_list.append(block_rew_only)

                            # construct n-back matrix of predictors
                            block_mat = np.hstack([np.hstack([pred[i:-n_back+i] for pred in predictor_list])
                                                      for i in range(n_back-1, -1, -1)])

                            predictor_mat.append(block_mat)

            predictor_mat = np.vstack(predictor_mat)
            choice_mat = np.vstack(choice_mat).reshape(-1)

            print('Subject {} Regression Results for {} trials:'.format(subj, reg_group))

            # clf = lm.LogisticRegression().fit(predictor_mat, choice_mat)
            # print('Regression, L2 penalty')
            # print(np.concatenate((clf.intercept_, clf.coef_[0]))[:,None])
            # print('Accuracy: {}%'.format(clf.score(predictor_mat, choice_mat)*100))
            # print('R2: {}\n'.format(r2_score(choice_mat, clf.predict(predictor_mat))))

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
        fig.suptitle('Choice Regression Coefficients by Block Reward Rate ({}){}'.format(subj, plot_suffix))

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
            if plot_cis:
                ax.errorbar(0, params[key], yerr=np.abs(cis.loc[key,:] - params[key]).to_numpy()[:,None], fmt='o', capsize=4, label='bias')
            else:
                ax.errorbar(0, params[key], fmt='o', label='bias')

            row_labels = params.index.to_numpy()
            for j, pred_label in enumerate(label_list):
                pred_label = pred_label.replace(' ({})', '')
                pred_row_labels = [label for label in row_labels if pred_label == re.sub(r' \(.*\)', '', label)]

                pred_params = params[pred_row_labels].to_numpy()
                pred_cis = cis.loc[pred_row_labels,:].to_numpy()

                if plot_cis:
                    ax.errorbar(x_vals, pred_params, yerr=np.abs(pred_cis - pred_params[:,None]).T, fmt='o-', capsize=4, label=pred_label)
                else:
                    ax.errorbar(x_vals, pred_params, fmt='o-', label=pred_label)

            if i == 0:
                ax.set_ylabel('Regresssion Coefficient for Choosing Left')
            ax.set_xlabel('Trials Back')
            ax.legend(loc='best')


def logit_regress_stay_choice(sess_data, n_back=5, separate_block_types=True, fit_switches=False, include_unreward=False, include_choice=True, 
                              include_diff_choice=True, include_interaction=True, include_full_interaction=False, plot_cis=True, ind_subj=True, meta_subj=True, plot_suffix=''):
    
    label_order = []
    if include_unreward:
        label_order.append('reward 1/-1')
    else:
        label_order.append('reward 1/0')
        
    if include_full_interaction:
        label_order.extend(['same/rewarded', 'same/unrewarded', 'diff/unrewarded']) #'diff/rewarded'
    else:
        if include_choice:
            if include_diff_choice:
                label_order.append('same choice 1/-1')
            else:
                label_order.append('same choice 1/0')
            
        if include_interaction:
            label_order.append('choice x reward')
            
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
    
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        block_types = np.sort(subj_sess_resp['block_prob'].unique())

        reg_groups = ['All']
        if separate_block_types:
            reg_groups.extend(block_types)
        
        reg_results = []

        for reg_group in reg_groups:

            predictor_mat = []
            choice_mat = [] # will be 1 for stay, 0 for switch
            # choices need to be binary for the outcome in order for the model fitting to work
            # doesn't change outcome of fit if 0/1 or -1/+1

            # build predictor matrix
            for sess_id in subj_sess_ids:
                ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

                if len(ind_sess) == 0:
                    continue

                # reshape variables to columns [:,None]
                rewarded = ind_sess['rewarded'].to_numpy().astype(int)
                unrewarded = (~ind_sess['rewarded'].to_numpy()).astype(int)
                rew_unrew = rewarded.copy()
                rew_unrew[unrewarded == 1] = -1
                choices = ind_sess['choice'].to_numpy()
                stays = (choices[:-1] == choices[1:]).astype(int)
                switches = (choices[:-1] != choices[1:]).astype(int)

                # make buffered predictor vectors to build predictor matrix
                buffer = np.full(n_back-1, 0)
                buff_reward = np.concatenate((buffer, rewarded))
                buff_unreward = np.concatenate((buffer, unrewarded))
                buff_choices = np.concatenate((buffer, choices))
                buff_rew_unrew = np.concatenate((buffer, rew_unrew))
                
                # create same/diff predictors for each n-back step
                same_sides = {}
                diff_sides = {}
                same_diff_sides = {}
                for i in range(n_back-2, -1, -1): 
                    same_sides[i-n_back] = (buff_choices[i:-n_back+i] == buff_choices[n_back-1:-1]).astype(int)
                    diff_sides[i-n_back] = (buff_choices[i:-n_back+i] != buff_choices[n_back-1:-1]).astype(int)
                    same_diff_sides[i-n_back] = same_sides[i-n_back].copy()
                    same_diff_sides[i-n_back][diff_sides[i-n_back] == 1] = -1

                # build predictor and outcome matrices
                if reg_group == 'All':

                    if fit_switches:
                        choice_mat.append(switches[:,None])
                    else:
                        choice_mat.append(stays[:,None])

                    # construct list of predictors to stack into matrix
                    predictors = {}
                    
                    if include_full_interaction:
                        if include_unreward:
                            predictors['reward 1/-1 (-1)'] = buff_rew_unrew[n_back-1:-1]
                        else:
                            predictors['reward 1/0 (-1)'] = buff_reward[n_back-1:-1]
                            
                        predictors.update({'same/rewarded ({})'.format(i-n_back): same_sides[i-n_back]*buff_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                        predictors.update({'same/unrewarded ({})'.format(i-n_back): same_sides[i-n_back]*buff_unreward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                        #predictors.update({'diff/rewarded ({})'.format(i-n_back): diff_sides[i-n_back]*buff_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                        predictors.update({'diff/unrewarded ({})'.format(i-n_back): diff_sides[i-n_back]*buff_unreward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                    else:
                        if include_unreward:
                            predictors.update({'reward 1/-1 ({})'.format(i-n_back): buff_rew_unrew[i:-n_back+i] for i in range(n_back-1, -1, -1)})
                        else:
                            predictors.update({'reward 1/0 ({})'.format(i-n_back): buff_reward[i:-n_back+i] for i in range(n_back-1, -1, -1)})

                        if include_choice:
                            if include_diff_choice:
                                predictors.update({'same choice 1/-1 ({})'.format(i-n_back): same_diff_sides[i-n_back] for i in range(n_back-2, -1, -1)})
                            else:
                                predictors.update({'same choice 1/0 ({})'.format(i-n_back): same_sides[i-n_back] for i in range(n_back-2, -1, -1)})
                            
                        if include_interaction:
                            if include_unreward:
                                if include_diff_choice:
                                    predictors.update({'choice x reward ({})'.format(i-n_back): same_diff_sides[i-n_back]*buff_rew_unrew[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                else:
                                    predictors.update({'choice x reward ({})'.format(i-n_back): same_sides[i-n_back]*buff_rew_unrew[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                            else:
                                if include_diff_choice:
                                    predictors.update({'choice x reward ({})'.format(i-n_back): same_diff_sides[i-n_back]*buff_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                else:
                                    predictors.update({'choice x reward ({})'.format(i-n_back): same_sides[i-n_back]*buff_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})

                    # construct n-back matrix of predictors
                    sess_mat = pd.DataFrame(predictors)
                    predictor_mat.append(sess_mat)
                else:
                    block_type = ind_sess['block_prob'].to_numpy()
                    block_sel = block_type == reg_group

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
                            if fit_switches:
                                choice_mat.append(switches[block_trans_idxs[j]:block_trans_idxs[j+1]][:,None])
                            else:
                                choice_mat.append(stays[block_trans_idxs[j]:block_trans_idxs[j+1]][:,None])

                            # construct n-back matrices for choice/outcome pairs
                            block_reward = buff_reward[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_unreward = buff_unreward[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            block_rew_unrew = buff_rew_unrew[block_trans_idxs[j]:block_trans_idxs[j+1]+n_back]
                            
                            # create same/diff predictors for each n-back step
                            block_same_sides = {k: v[block_trans_idxs[j]:block_trans_idxs[j+1]] for k,v in same_sides.items()}
                            block_diff_sides = {k: v[block_trans_idxs[j]:block_trans_idxs[j+1]] for k,v in diff_sides.items()}
                            block_same_diff_sides = {k: v[block_trans_idxs[j]:block_trans_idxs[j+1]] for k,v in same_diff_sides.items()}

                            predictors = {}
                            
                            if include_full_interaction:
                                if include_unreward:
                                    predictors['reward 1/-1 (-1)'] = block_rew_unrew[n_back-1:-1]
                                else:
                                    predictors['reward 1/0 (-1)'] = block_reward[n_back-1:-1]
                                    
                                predictors.update({'same/rewarded ({})'.format(i-n_back): block_same_sides[i-n_back]*block_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                predictors.update({'same/unrewarded ({})'.format(i-n_back): block_same_sides[i-n_back]*block_unreward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                #predictors.update({'diff/rewarded ({})'.format(i-n_back): block_diff_sides[i-n_back]*block_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                predictors.update({'diff/unrewarded ({})'.format(i-n_back): block_diff_sides[i-n_back]*block_unreward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                            else:
                                if include_unreward:
                                    predictors.update({'reward 1/-1 ({})'.format(i-n_back): block_rew_unrew[i:-n_back+i] for i in range(n_back-1, -1, -1)})
                                else:
                                    predictors.update({'reward 1/0 ({})'.format(i-n_back): block_reward[i:-n_back+i] for i in range(n_back-1, -1, -1)})

                                if include_choice:
                                    if include_diff_choice:
                                        predictors.update({'same choice 1/-1 ({})'.format(i-n_back): block_same_diff_sides[i-n_back] for i in range(n_back-2, -1, -1)})
                                    else:
                                        predictors.update({'same choice 1/0 ({})'.format(i-n_back): block_same_sides[i-n_back] for i in range(n_back-2, -1, -1)})
                                    
                                if include_interaction:
                                    if include_unreward:
                                        if include_diff_choice:
                                            predictors.update({'choice x reward ({})'.format(i-n_back): block_same_diff_sides[i-n_back]*block_rew_unrew[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                        else:
                                            predictors.update({'choice x reward ({})'.format(i-n_back): block_same_sides[i-n_back]*block_rew_unrew[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                    else:
                                        if include_diff_choice:
                                            predictors.update({'choice x reward ({})'.format(i-n_back): block_same_diff_sides[i-n_back]*block_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})
                                        else:
                                            predictors.update({'choice x reward ({})'.format(i-n_back): block_same_sides[i-n_back]*block_reward[i:-n_back+i] for i in range(n_back-2, -1, -1)})

                            block_mat = pd.DataFrame(predictors)
                            predictor_mat.append(block_mat)

            predictor_mat = pd.concat(predictor_mat)
            choice_mat = np.vstack(choice_mat).reshape(-1)

            print('Subject {} Regression Results for {} trials:'.format(subj, reg_group))

            # clf = lm.LogisticRegression().fit(predictor_mat, choice_mat)
            # print('Regression, L2 penalty')
            # print(np.concatenate((clf.intercept_, clf.coef_[0]))[:,None])
            # print('Accuracy: {}%'.format(clf.score(predictor_mat, choice_mat)*100))
            # print('R2: {}\n'.format(r2_score(choice_mat, clf.predict(predictor_mat))))

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

            predictor_mat = sm.add_constant(predictor_mat)
            fit_res = sm.Logit(choice_mat, predictor_mat).fit()
            print(fit_res.summary())
            # mfx = fit_res.get_margeff()
            # print(mfx.summary())
            reg_results.append(fit_res)

        # plot regression coefficients over trials back
        fig, axs = plt.subplots(1, len(reg_groups), layout='constrained', figsize=(4*len(reg_groups), 4), sharey=True)
        fig.suptitle('Choice Regression Coefficients by Block Reward Rate ({}){}'.format(subj, plot_suffix))

        for i, group in enumerate(reg_groups):

            fit_res = reg_results[i]
            params = fit_res.params
            cis = fit_res.conf_int(0.05)

            ax = axs[i]
            ax.set_title('Block Rate: {}'.format(group))
            plot_utils.plot_dashlines(0, dir='h', ax=ax)

            # plot constant
            key = 'const'
            if plot_cis:
                ax.errorbar(0, params[key], yerr=np.abs(cis.loc[key,:] - params[key]).to_numpy()[:,None], fmt='o', capsize=4, label='bias')
            else:
                ax.errorbar(0, params[key], fmt='o', label='bias')

            row_labels = params.index.to_numpy()
            for j, pred_label in enumerate(label_order):
                pred_row_labels = [label for label in row_labels if pred_label == re.sub(r' \(.*\)', '', label)]
                pred_x_vals = [-int(match.group(1)) for label in pred_row_labels if (match := re.search(r'\((-?\d+)\)', label))]

                pred_params = params[pred_row_labels].to_numpy()
                pred_cis = cis.loc[pred_row_labels,:].to_numpy()

                if plot_cis:
                    ax.errorbar(pred_x_vals, pred_params, yerr=np.abs(pred_cis - pred_params[:,None]).T, fmt='o-', capsize=4, label=pred_label)
                else:
                    ax.errorbar(pred_x_vals, pred_params, fmt='o-', label=pred_label)

            if i == 0:
                ax.set_ylabel('Regresssion Coefficient for Staying with Same Choice')
            ax.set_xlabel('Trials Back')
            ax.legend(loc='best')
    
    
    
# %% Time-related Behavioral Effects

def analyze_choice_time_effects(sess_data, time_col, time_bin_edges, ind_subj=True, meta_subj=True, plot_suffix=''):
    # look at relationship between previous reward and future choice (stay/switch) based on how long it took to start the next trial
    
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')

    # get bins output by pandas for indexing
    bins = pd.IntervalIndex.from_bteaks(time_bin_edges)

    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        # create structures to accumulate event data across sessions
        stay_reward = {'k': np.zeros(len(bins)), 'n': np.zeros(len(bins))}
        stay_noreward = copy.deepcopy(stay_reward)

        for sess_id in subj_sess_ids:
            ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

            if len(ind_sess) == 0:
                continue

            time_bins = pd.cut(ind_sess[time_col], bins)[1:] # ignore first trial time since no trial before

            choices = ind_sess['choice'].to_numpy()
            stays = choices[:-1] == choices[1:]
            rewarded = ind_sess['rewarded'].to_numpy()[:-1] # ignore last trial reward since no trial after

            for i, b in enumerate(bins):
                bin_sel = time_bins == b
                stay_reward['k'][i] += sum(rewarded & stays & bin_sel)
                stay_reward['n'][i] += sum(rewarded & bin_sel)
                stay_noreward['k'][i] += sum(~rewarded & stays & bin_sel)
                stay_noreward['n'][i] += sum(~rewarded & bin_sel)

        # plot results
        # define reusable helper methods
        def comp_p(n_dict): return n_dict['k']/n_dict['n']
        def comp_err(n_dict): return abs(np.array([utils.binom_cis(n_dict['k'][i], n_dict['n'][i]) for i in range(len(n_dict['k']))]) - comp_p(n_dict)[:,None])

        # calculate rates
        stay_reward_vals = comp_p(stay_reward)
        stay_reward_err = comp_err(stay_reward).T
        stay_noreward_vals = comp_p(stay_noreward)
        stay_noreward_err = comp_err(stay_noreward).T
        
        # calculate trial distribution
        total_trials = np.sum(stay_reward['n'] + stay_noreward['n'])
        rel_reward = stay_reward['n']/total_trials
        rel_unreward = stay_noreward['n']/total_trials

        # x = (bin_edges[:-1] + bin_edges[1:])/2
        # x[-1] = t_thresh
        x = np.arange(len(bins))
        x_labels = ['{:.0f} - {:.0f}'.format(b.left, b.right) for b in bins]

        fig, axs = plt.subplots(2, 1, layout='constrained', figsize=(7, 6), height_ratios=[3,2])
        
        fig.suptitle('Stay Probabilities by Trial Latency ({}){}'.format(subj, plot_suffix))
        
        ax = axs[0]
        ax.set_title('Stay Probabilities')

        ax.errorbar(x, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Rewarded')
        ax.errorbar(x, stay_noreward_vals, yerr=stay_noreward_err, fmt='o', capsize=4, label='Unrewarded')
        ax.set_ylabel('p(stay)')
        ax.set_xlabel('Next Trial Latency (s)')
        ax.set_xticks(x, x_labels)
        ax.set_ylim(0.35, 1.05)
        ax.grid(axis='y')
        ax.legend(loc='best')
        
        ax = axs[1]
        ax.set_title('Trial Distribution')
        
        plot_utils.plot_stacked_bar([rel_reward, rel_unreward], value_labels=['Rewarded', 'Unrewarded'], x_labels=x_labels, ax=ax)
        ax.set_ylabel('Proportion of Total Trials')
        ax.set_xlabel('Next Trial Latency (s)')
        ax.legend(loc='best')


