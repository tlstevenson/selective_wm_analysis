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

def analyze_trial_hist_counts(sess_data, n_back=3, plot_suffix=''):
    
    if not bah.trial_hist_exists(sess_data):
        bah.calc_trial_hist(sess_data, n_back)
        
    hist_label_col = 'trial_hist_label ({})'.format(n_back)

    if not hist_label_col in sess_data.columns:
        make_trial_hist_labels(sess_data, n_back)
        
    # get trial history frequency
    trial_hist_counts = bah.get_count_dict(sess_data, 'subjid', hist_label_col, normalize=False)[hist_label_col]
    # remove trial hist labels for trials less than n_back
    unique_hist_labels = get_unique_hist_labels(sess_data, 'trial_hist_label', n_back)
    trial_hist_counts = trial_hist_counts[unique_hist_labels]
    # convert counts to percentages
    trial_hist_pct = trial_hist_counts.div(trial_hist_counts.sum(axis=1), axis=0)*100
    
    trial_hist_pct_plot = trial_hist_pct.reset_index().melt(id_vars='subjid', var_name=hist_label_col, value_name='pct')
    
    # get frequency of each choice/outcome at each trial back latency and at the current trial
    current_trial_mets = []
    t_back_hist = []
    
    for subj in np.unique(sess_data['subjid']):
        subj_data = sess_data[sess_data['subjid'] == subj]
        choice_hist_mat = np.vstack(subj_data['choice_hist'])
        rew_hist_mat = np.vstack(subj_data['rew_hist'])
        choices = subj_data['choice'].to_numpy()
        rewarded = subj_data['rewarded'].to_numpy()
        
        for choice in ['Left', 'Right']:
            if choice == 'Left':
                choice_sel = choices == 'left'
            else:
                choice_sel = choices == 'right'
                
            for rew in ['Rewarded', 'Unrewarded']:
                if rew == 'Rewarded':
                    rew_sel = rewarded
                else:
                    rew_sel = ~rewarded
                    
                current_trial_mets.append({'subj': subj, 't_back': 0, 'Trial Type': '{}/{}'.format(choice, rew),
                                           'pct': np.sum(choice_sel & rew_sel)/len(choices)})
        
        for i in range(n_back):
            valid_sel = ~np.isnan(rew_hist_mat[:,i])
            
            for choice in ['Same', 'Diff']:
                if choice == 'Same':
                    choice_sel = choice_hist_mat[:,i] == choices
                else:
                    choice_sel = choice_hist_mat[:,i] != choices
                    
                for rew in ['Rewarded', 'Unrewarded']:
                    if rew == 'Rewarded':
                        rew_sel = rew_hist_mat[:,i] == 1
                    else:
                        rew_sel = rew_hist_mat[:,i] == 0
                        
                    t_back_hist.append({'subj': subj, 't_back': i+1, 'Trial Type': '{}/{}'.format(choice, rew),
                                        'pct': np.sum(valid_sel & choice_sel & rew_sel)/np.sum(valid_sel)})
    
    current_trial_mets = pd.DataFrame(current_trial_mets)
    t_back_hist = pd.DataFrame(t_back_hist)

    fig = plt.figure(layout='constrained', figsize=(12, 6))
    fig.suptitle('Trial History Frequencies{}'.format(plot_suffix))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1,3])
    
    # plot frequency of current choice/outcome
    ax = fig.add_subplot(gs[0, 0])
    sb.barplot(current_trial_mets, x='t_back', y='pct', errorbar='se', hue='Trial Type', palette='colorblind', ax=ax)
    sb.stripplot(current_trial_mets, x='t_back', y='pct', hue='Trial Type', color='black', alpha=0.5, legend=False, dodge=True)
    ax.set_title('Trial Choice/Outcome Frequency')
    ax.set_xlabel('Trial Back')
    ax.set_ylabel('% Occurance')
    
    # plot frequency of choice/outcome history at each trial back latency based on current choice
    ax = fig.add_subplot(gs[0, 1])
    sb.barplot(t_back_hist, x='t_back', y='pct', errorbar='se', hue='Trial Type', palette='colorblind', ax=ax)
    sb.stripplot(t_back_hist, x='t_back', y='pct', hue='Trial Type', color='black', alpha=0.5, legend=False, dodge=True)
    ax.set_title('Relative Choice/Outcome Frequency at Each Trial Back')
    ax.set_xlabel('Trial Back')
    ax.set_ylabel('% Occurance')

    # Plot the frequency of trial histories irrespective of current choice
    ax = fig.add_subplot(gs[1, :])
    sb.barplot(trial_hist_pct_plot, x=hist_label_col, y='pct', order=unique_hist_labels, errorbar='se', color='black', alpha=0.5, ax=ax)
    sb.stripplot(trial_hist_pct_plot, x=hist_label_col, y='pct', order=unique_hist_labels, hue='subjid', ax=ax, palette='colorblind', alpha=0.7)
    ax.set_title('Choice/Outcome History Relative to Previous Trial Choice')
    ax.set_xlabel('Trial History Label (t-1 -> t-{})'.format(n_back))
    ax.set_ylabel('% Occurance')


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
        
        block_rates = np.unique(subj_sess_resp['block_prob'])
        block_lengths = {br: [] for br in block_rates}
        block_lengths['all'] = []
        consec_blocks = {br: [] for br in block_rates}

        for sess_id in subj_sess_ids:
            ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

            if len(ind_sess) == 0:
                continue

            # find last block trial number before switch
            # ignore the last block since it likely ended prematurely
            block_switch_trial_idx = np.nonzero(ind_sess['block_trial'] == 1)[0][1:]-1
            
            block_rate = ind_sess['block_prob'].to_numpy()
            prev_block_rate = block_rate[0]
            consec_block = 0
            
            for i in block_switch_trial_idx:
                length = ind_sess.iloc[i]['block_trial']
                block_lengths[block_rate[i]].append(length)
                block_lengths['all'].append(length)
                
                if block_rate[i] == prev_block_rate:
                    consec_block += 1
                else:
                    consec_blocks[prev_block_rate].append(consec_block)
                    consec_block = 1
                    prev_block_rate = block_rate[i]
            
            # count the last epoch's trials
            consec_blocks[prev_block_rate].append(consec_block)
                
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
            for br in block_rates:
                ax.hist(block_lengths[br], **hist_args, label=br)
            
        ax.set_title('Block Trial Lengths ({}){}'.format(subj, plot_suffix))
        ax.set_xlabel('Block Length (trials)')
        ax.set_ylabel('Proportion')
        ax.legend()
        
        ax = axs[1]
        min_len = np.min([np.min(consec_blocks[br]) for br in block_rates])
        max_len = np.max([np.max(consec_blocks[br]) for br in block_rates])
        bins = np.arange(min_len, max_len+epoch_bin_width, epoch_bin_width)
        hist_args = dict(histtype='bar', density=True, cumulative=False, bins=bins, alpha=0.5)

        for br in block_rates:
            ax.hist(consec_blocks[br], **hist_args, label=br)
            
        ax.set_title('Block Epoch Lengths ({}){}'.format(subj, plot_suffix))
        ax.set_xlabel('Epoch Length (blocks)')
        ax.set_ylabel('Proportion')
        ax.legend()
    
# %% Choice Probability Methods

def analyze_choice_behavior(sess_data, n_back_hist=3, plot_simple_summary=False, meta_subj=True, ind_subj=True, plot_suffix=''):
    
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
        
    figs = {s: [] for s in subj_ids}
    
    if not bah.trial_hist_exists(sess_data):
        bah.calc_trial_hist(sess_data, n_back_hist)

    make_trial_hist_labels(sess_data, n_back_hist)
    make_rew_hist_labels(sess_data, n_back_hist)
    make_rew_hist_labels(sess_data, 2)
    
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        choose_left_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_left', ['side_prob', 'high_side'])
        choose_right_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_right', ['high_side'])
        choose_high_trial_probs = bah.get_rate_dict(subj_sess_resp, 'chose_high', ['block_prob'])

        fig = plt.figure(layout='constrained', figsize=(6, 6))
        fig.suptitle('Response Probabilities ({}){}'.format(subj, plot_suffix))
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
        ax.set_ylim(0.35, 1.05)
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
        
        figs[subj].append(fig)


        # SWITCHING PROBABILITIES
        choice_block_probs = np.sort(subj_sess_resp['choice_block_prob'].unique())
        block_rates = np.sort(subj_sess_resp['block_prob'].unique())
        trial_hist_labels = get_unique_hist_labels(sess_data, 'trial_hist_label', n_back_hist)
        rew_hist_labels_nback = get_unique_hist_labels(sess_data, 'rew_hist_label', n_back_hist)
        rew_hist_labels_2back = get_unique_hist_labels(sess_data, 'rew_hist_label', 2)

        n_stay_reward_choice = {p: {'k': 0, 'n': 0} for p in choice_block_probs}
        n_stay_noreward_choice = copy.deepcopy(n_stay_reward_choice)
        n_stay_reward_block = {br: {'k': 0, 'n': 0} for br in block_rates}
        n_stay_noreward_block = copy.deepcopy(n_stay_reward_block)
        n_switch_trial_hist_labels = {l: {'k': 0, 'n': 0} for l in trial_hist_labels}
        
        n_switch_rew_hist_labels_nback = {br: {l: {'k': 0, 'n': 0} for l in rew_hist_labels_nback} for br in np.append(block_rates, 'all')}
        n_switch_rew_hist_labels_2back = {br: {l: {'k': 0, 'n': 0} for l in rew_hist_labels_2back} for br in np.append(block_rates, 'all')}

        # empirical transition matrices by block reward rates
        # structure: choice x next choice, (high prob, low prob)
        trans_mats = {br: {'k': np.zeros((2,2)), 'n': np.zeros((2,2))} for br in block_rates}

        for sess_id in subj_sess_ids:
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
                n_stay_noreward_choice[p]['k'] += sum(~rewarded & stays & choice_sel)
                n_stay_noreward_choice[p]['n'] += sum(~rewarded & choice_sel)

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
                
            # trial and reward history labels
            # start from trial 2 so that we match the trial where a switch occured with the history for that trial
            sess_trial_hist_labels = ind_sess['trial_hist_label ({})'.format(n_back_hist)].to_numpy()[1:]
            for label in trial_hist_labels:
                label_sel = sess_trial_hist_labels == label
                n_switch_trial_hist_labels[label]['k'] += sum(~stays & label_sel)
                n_switch_trial_hist_labels[label]['n'] += sum(label_sel)
                
            sess_rew_hist_labels_nback = ind_sess['rew_hist_label ({})'.format(n_back_hist)].to_numpy()[1:]
            sess_rew_hist_labels_2back = ind_sess['rew_hist_label (2)'].to_numpy()[1:]
            all_choice_hist_same_nback = ind_sess['choice_hist'].apply(lambda x: np.all(x[:n_back_hist] == x[0])).to_numpy()[1:]
            all_choice_hist_same_2back = ind_sess['choice_hist'].apply(lambda x: np.all(x[:2] == x[0])).to_numpy()[1:]
            
            for br in np.append(block_rates, 'all'):
                if br == 'all':
                    rate_sel = np.full_like(block_rate, True)
                else:
                    rate_sel = block_rate == br
                    
                for label in rew_hist_labels_nback:
                    label_sel = sess_rew_hist_labels_nback == label
                    n_switch_rew_hist_labels_nback[br][label]['k'] += sum(~stays & label_sel & all_choice_hist_same_nback & rate_sel)
                    n_switch_rew_hist_labels_nback[br][label]['n'] += sum(label_sel & all_choice_hist_same_nback & rate_sel)
                
                for label in rew_hist_labels_2back:
                    label_sel = sess_rew_hist_labels_2back == label
                    n_switch_rew_hist_labels_2back[br][label]['k'] += sum(~stays & label_sel & all_choice_hist_same_2back & rate_sel)
                    n_switch_rew_hist_labels_2back[br][label]['n'] += sum(label_sel & all_choice_hist_same_2back & rate_sel)

        # plot results
        # define reusable helper methods
        def comp_p(n_dict): return 0 if n_dict['n'] == 0 else n_dict['k']/n_dict['n']
        def comp_err(n_dict): return abs(utils.binom_cis(n_dict['k'], n_dict['n']) - comp_p(n_dict))
        
        # need to add enough cols to make the first row always have columns with width ratios of [2/3, 1/3]
        n_cols = utils.lcm(len(block_rates), 3)

        fig = plt.figure(layout='constrained', figsize=(3.5*len(block_rates), 7))
        gs = GridSpec(2, n_cols, figure=fig)
        
        fig.suptitle('Stay/Switch Probabilities ({}){}'.format(subj, plot_suffix))

        # first row, left, is the stay rate by choice probability & block rate
        split = int(n_cols * 2 / 3)
        ax = fig.add_subplot(gs[0, :split])
        stay_reward_vals = [comp_p(n_stay_reward_choice[p]) for p in choice_block_probs]
        stay_reward_err = np.asarray([comp_err(n_stay_reward_choice[p]) for p in choice_block_probs]).T
        stay_noreward_vals = [comp_p(n_stay_noreward_choice[p]) for p in choice_block_probs]
        stay_noreward_err = np.asarray([comp_err(n_stay_noreward_choice[p]) for p in choice_block_probs]).T

        x_vals = np.arange(len(choice_block_probs))

        ax.errorbar(x_vals, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Rewarded')
        ax.errorbar(x_vals, stay_noreward_vals, yerr=stay_noreward_err, fmt='o', capsize=4, label='Unrewarded')
        ax.set_ylabel('p(Stay)')
        ax.set_xlabel('Choice Reward Probability (Block Probs)')
        ax.set_xticks(x_vals, choice_block_probs)
        ax.set_xlim(-0.5, len(x_vals)-0.5)
        ax.set_ylim(0.3, 1.05)
        ax.grid(axis='y')
        ax.legend(loc='best')

        # first row, right, is the stay rate after reward/no reward by block rate only
        ax = fig.add_subplot(gs[0, split:])
        stay_reward_vals = [comp_p(n_stay_reward_block[br]) for br in block_rates]
        stay_reward_err = np.asarray([comp_err(n_stay_reward_block[br]) for br in block_rates]).T
        stay_noreward_vals = [comp_p(n_stay_noreward_block[br]) for br in block_rates]
        stay_noreward_err = np.asarray([comp_err(n_stay_noreward_block[br]) for br in block_rates]).T

        # plot_utils.plot_stacked_bar([stay_reward_vals, stay_noreward_vals], value_labels=['Rewarded', 'Unrewarded'],
        #                             x_labels=block_rates, err=[stay_reward_err, stay_noreward_err], ax=ax)
        
        x_vals = np.arange(len(block_rates))
        
        ax.errorbar(x_vals, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Rewarded')
        ax.errorbar(x_vals, stay_noreward_vals, yerr=stay_noreward_err, fmt='o', capsize=4, label='Unrewarded')
        ax.set_ylabel('p(Stay)')
        ax.set_xlabel('Block Reward Probability')
        ax.set_xticks(x_vals, block_rates)
        ax.set_xlim(-0.5, len(x_vals)-0.5)
        ax.set_ylim(0.3, 1.05)
        ax.grid(axis='y')
        ax.legend(loc='best')

        col_width = n_cols // len(block_rates)
        # second row is empirical transition matrices
        for i, br in enumerate(block_rates):
            ax = fig.add_subplot(gs[1, i*col_width:(i+1)*col_width])
            p_mat = trans_mats[br]['k']/trans_mats[br]['n']
            plot_utils.plot_value_matrix(p_mat, ax=ax, xticklabels=['high', 'low'], yticklabels=['high', 'low'], cbar=False, cmap='vlag')
            ax.set_ylabel('Choice Reward Rate')
            ax.set_xlabel('Next Choice Reward Rate')
            ax.xaxis.set_label_position('top')
            ax.set_title('Block Rate {}%'.format(br))
            
        figs[subj].append(fig)


        # Switching probabilities based on choice/reward histories
        
        fig = plt.figure(layout='constrained', figsize=(12, 6))
        gs = GridSpec(2, 2, figure=fig, width_ratios=[1,2])
        
        fig.suptitle('Switch Probabilities by Trial History ({}){}'.format(subj, plot_suffix))
        
        # first row, left is same-side reward history 2 trials back
        ax = fig.add_subplot(gs[0,0])
        stay_rew_hist_vals = [[comp_p(n_switch_rew_hist_labels_2back[br][l]) for l in rew_hist_labels_2back] for br in np.append(block_rates, 'all')]
        stay_rew_hist_err = [np.asarray([comp_err(n_switch_rew_hist_labels_2back[br][l]) for l in rew_hist_labels_2back]).T for br in np.append(block_rates, 'all')]

        plot_utils.plot_stacked_bar(stay_rew_hist_vals, value_labels=np.append(block_rates, 'all'), x_labels=rew_hist_labels_2back, err=stay_rew_hist_err, ax=ax)
        ax.set_title('Same Side Reward History')
        ax.set_xlabel('Reward History (t-1, t-2)')
        ax.set_ylabel('p(Switch)')
        ax.set_ylim(-0.05, 1.05)
        #ax.set_xlim(-0.5, len(rew_hist_labels_2back)-0.5)
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        
        # first row, right is same-side reward history 3 trials back
        ax = fig.add_subplot(gs[0,1])
        stay_rew_hist_vals = [[comp_p(n_switch_rew_hist_labels_nback[br][l]) for l in rew_hist_labels_nback] for br in np.append(block_rates, 'all')]
        stay_rew_hist_err = [np.asarray([comp_err(n_switch_rew_hist_labels_nback[br][l]) for l in rew_hist_labels_nback]).T for br in np.append(block_rates, 'all')]

        plot_utils.plot_stacked_bar(stay_rew_hist_vals, value_labels=np.append(block_rates, 'all'), x_labels=rew_hist_labels_nback, err=stay_rew_hist_err, ax=ax)
        ax.set_title('Same Side Reward History')
        ax.set_xlabel('Reward History (t-1 -> t-{})'.format(n_back_hist))
        ax.set_ylabel('p(Switch)')
        ax.set_ylim(-0.05, 1.05)
        #ax.set_xlim(-0.5, len(rew_hist_labels_nback)-0.5)
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        
        # second row is choice x reward history
        ax = fig.add_subplot(gs[1,:])
        stay_trial_hist_vals = [comp_p(n_switch_trial_hist_labels[l]) for l in trial_hist_labels]
        stay_trial_hist_err = np.asarray([comp_err(n_switch_trial_hist_labels[l]) for l in trial_hist_labels]).T
        
        #comp_err(n_switch_trial_hist_labels['AAA'])

        plot_utils.plot_stacked_bar([stay_trial_hist_vals], x_labels=trial_hist_labels, err=[stay_trial_hist_err], ax=ax)
        ax.set_title('Choice/Outcome History Relative to Previous Trial Choice')
        ax.set_xlabel('Choice/Outcome History (t-1 -> t-{})'.format(n_back_hist))
        ax.set_ylabel('p(Switch)')
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.5, len(trial_hist_labels)-0.5)
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        
        figs[subj].append(fig)

        ## Simple summary of choice behavior
        
        if plot_simple_summary:
            # p(Choose High)
            fig = plt.figure(layout='constrained', figsize=(6, 3))
            fig.suptitle('Choice Probabilities ({}){}'.format(subj, plot_suffix))
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
            ax.set_ylim(0.35, 1.05)
            ax.set_title('Choose High Side')
    
            # p(Stay)
            stay_reward_vals = [comp_p(n_stay_reward_block[br]) for br in block_rates]
            stay_reward_err = np.asarray([comp_err(n_stay_reward_block[br]) for br in block_rates]).T
            stay_noreward_vals = [comp_p(n_stay_noreward_block[br]) for br in block_rates]
            stay_noreward_err = np.asarray([comp_err(n_stay_noreward_block[br]) for br in block_rates]).T
    
            ax = fig.add_subplot(gs[0, 1])
            x_vals = np.arange(len(block_rates))
            
            ax.errorbar(x_vals, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Rewarded')
            ax.errorbar(x_vals, stay_noreward_vals, yerr=stay_noreward_err, fmt='o', capsize=4, label='Unrewarded')
            # plot_utils.plot_stacked_bar([stay_reward_vals, stay_noreward_vals], value_labels=['Rewarded', 'Unrewarded'],
            #                             x_labels=block_rates, err=[stay_reward_err, stay_noreward_err], ax=ax)
            ax.set_ylabel('p(Stay)')
            ax.set_xlabel('Block Reward Probability (High/Low %)')
            ax.set_xticks(x_vals, block_rates)
            ax.set_xlim(-0.5, len(x_vals)-0.5)
            ax.set_ylim(0.35, 1.05)
            ax.legend(loc='lower right')
            ax.set_title('Choose Previous Side')
            
            figs[subj].append(fig)
            
    return figs


# %% Choice Probabilities over trials

def analyze_trial_choice_behavior(sess_data, plot_simple_summary=False, meta_subj=True, ind_subj=True, plot_suffix=''):
    
    subj_ids = []
    if ind_subj:
        subj_ids.extend(np.unique(sess_data['subjid']).tolist())
    if meta_subj:
        subj_ids.append('all')
        
    figs = {s: [] for s in subj_ids}
    
    for subj in subj_ids:
        if subj == 'all':
            subj_sess = sess_data
        else:
            subj_sess = sess_data[sess_data['subjid'] == subj]
            
        subj_sess_ids = np.unique(subj_sess['sessid'])
        subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']

        block_rates = np.sort(subj_sess_resp['block_prob'].unique())

        # Investigate how animals choose to stay/switch their response
        # Probability of choosing high port/getting reward pre and post block change by block reward
        # probability of getting reward leading up to stay/switch by port probability (high/low)
        # numbers of low/high reward rate choices in a row that are rewarded/unrewarded/any
        n_away = 10
        p_choose_high_blocks = {br: {'pre': [], 'post': []} for br in block_rates}
        p_choose_high_blocks['all'] = []
        
        n_before_switch = {br: {'pre': [], 'post': []} for br in block_rates}

        reward_choices = ['high', 'low', 'all']
        p_reward_switch = {c: {'stay': [], 'switch': []} for c in reward_choices}
        p_reward_switch_blocks = {br: {c: {'stay': [], 'switch': []} for c in reward_choices} for br in block_rates}

        sequence_counts = {r: {br: [] for br in block_rates} for r in ['high', 'low']}
        unrew_sequence_counts = {r: {br: [] for br in block_rates} for r in ['high', 'low']}
        get_seq_counts = lambda x: utils.get_sequence_lengths(x)[True] if np.sum(x) > 0 else []

        for sess_id in subj_sess_ids:
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
                
                # count number of trials before switch after block switch
                n_stays_post_switch = 0
                chose_low = ~high_choice[switch_idx]
                while chose_low:
                    n_stays_post_switch += 1
                    if (switch_idx + n_stays_post_switch) == len(high_choice):
                        break
                    chose_low = ~high_choice[switch_idx + n_stays_post_switch]
                    
                n_before_switch[pre_br]['pre'].append(n_stays_post_switch)
                n_before_switch[post_br]['post'].append(n_stays_post_switch)

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
                
            # unrewarded sequence counts before a switch
            high_stays = stays & high_choice[:-1] & ~rewarded[:-1]
            low_stays = stays & ~high_choice[:-1] & ~rewarded[:-1]
            for br in block_rates:
                rate_sel = block_rate[:-1] == br
                unrew_sequence_counts['high'][br].extend(get_seq_counts(high_stays & rate_sel))
                unrew_sequence_counts['low'][br].extend(get_seq_counts(low_stays & rate_sel))

        # plot switching information
        fig = plt.figure(layout='constrained', figsize=(16, 7))
        fig.suptitle('Switching Metrics ({}){}'.format(subj, plot_suffix))
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
        ax.set_xlim(-6, n_away-1)
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
        ax.set_xlim(-6, n_away-1)
        ax.set_ylim(-0.05, 1.05)

        # # p reward for stays
        # ax = fig.add_subplot(gs[0,1])
        # plot_utils.plot_x0line(ax=ax)
        # for choice_type in reward_choices:
        #     raw_mat = np.vstack(p_reward_switch[choice_type]['stay'])
        #     avg, err = bah.get_rate_avg_err(raw_mat)
        #     if choice_type == 'all':
        #         ax.plot(x, avg, dashes=[4,4], c='k', label=choice_type)
        #     else:
        #         plot_utils.plot_shaded_error(x, avg, err, ax=ax, label=choice_type)

        # ax.set_xlabel('Trials from decision to stay')
        # ax.set_ylabel('p(Reward)')
        # ax.set_title('Reward Probability Before Stay Choices')
        # ax.set_xlim(-6, 6)
        # ax.set_ylim(-0.05, 1.05)
        # ax.legend(loc='lower left', title='Choice Port\nReward Rate')

        # # p reward for switches
        # ax = fig.add_subplot(gs[1,1])
        # plot_utils.plot_x0line(ax=ax)
        # for choice_type in reward_choices:
        #     raw_mat = np.vstack(p_reward_switch[choice_type]['switch'])
        #     avg, err = bah.get_rate_avg_err(raw_mat)
        #     if choice_type == 'all':
        #         ax.plot(x, avg, dashes=[4,4], c='k', label=choice_type)
        #     else:
        #         plot_utils.plot_shaded_error(x, avg, err, ax=ax, label=choice_type)

        # ax.set_xlabel('Trials from decision to switch')
        # ax.set_ylabel('p(Reward)')
        # ax.set_title('Reward Probability Before Switch Choices')
        # ax.set_xlim(-6, 6)
        # ax.set_ylim(-0.05, 1.05)

        # histograms of choices sequence length before switch after block change
        ax = fig.add_subplot(gs[0,1])
        pre_max = np.max([np.max(n_before_switch[br]['pre']) for br in block_rates])
        post_max = np.max([np.max(n_before_switch[br]['post']) for br in block_rates])
        t_max = np.max([pre_max, post_max])
        hist_args = {'histtype': 'step', 'density': True, 'cumulative': True, 'bins': t_max, 'range': (0, t_max)}
        for br in block_rates:
            ax.hist(n_before_switch[br]['pre'], **hist_args, label=br)

        ax.set_xlabel('# Trials after Block Switch')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Trials to Switch - Pre-switch Rates')
        ax.set_xlim(-0.5, pre_max)
        ax.legend(loc='lower right', title='Reward Rates')

        # low choices, same side
        ax = fig.add_subplot(gs[1,1])
        for br in block_rates:
            ax.hist(n_before_switch[br]['post'], **hist_args, label=br)

        ax.set_xlabel('# Trials after Block Switch')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Trials to Switch - Post-switch Rates')
        ax.set_xlim(-0.5, post_max)
        

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
        ax.set_xlim(-6, 6)
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
        ax.set_xlim(-6, 6)
        ax.set_ylim(-0.05, 1.05)
        

        # histograms of sequence lengths
        # high choices, same side
        ax = fig.add_subplot(gs[0,3])
        high_max = np.max([np.max(unrew_sequence_counts['high'][br]) for br in block_rates])
        low_max = np.max([np.max(unrew_sequence_counts['low'][br]) for br in block_rates])
        t_max = np.max([high_max, low_max])
        hist_args = {'histtype': 'step', 'density': True, 'cumulative': True, 'bins': t_max, 'range': (1, t_max)}
        for br in block_rates:
            ax.hist(unrew_sequence_counts['high'][br], **hist_args, label=br)

        ax.set_xlabel('# Trials')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Unrewarded High Choice Sequence Lengths')
        ax.set_xlim(0.5, high_max)
        ax.legend(loc='lower right', title='Reward Rates')

        # low choices, same side
        ax = fig.add_subplot(gs[1,3])
        for br in block_rates:
            ax.hist(unrew_sequence_counts['low'][br], **hist_args, label=br)

        ax.set_xlabel('# Trials')
        ax.set_ylabel('Cumulative Proportion')
        ax.set_title('Unrewarded Low Choice Sequence Lengths')
        ax.set_xlim(0.5, low_max)
        
        figs[subj].append(fig)

        # Simple Summary
        if plot_simple_summary:
            
            fig, axs = plt.subplots(1, 2, figsize=(7,3), layout='constrained')
            fig.suptitle('Choice Probabilities Over Trials ({}){}'.format(subj, plot_suffix))
            
            # left plot is average p(choose high) aligned to block switches
            ax = axs[0]
            raw_mat = np.asarray(p_choose_high_blocks['all'])
            avg, err = bah.get_rate_avg_err(raw_mat)
            plot_utils.plot_x0line(ax=ax)
            plot_utils.plot_shaded_error(x, avg, err, ax=ax)

            ax.set_xlabel('Trials from block switch')
            ax.set_ylabel('p(Choose High)')
            ax.set_title('Choose High at Block Switch')
            ax.set_xlim(-6, n_away-1)
            ax.set_ylim(-0.05, 1.05)
            
            
            # right plot is reward probability aligned to switching choices
            ax = axs[1]
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
            ax.set_xlim(-6, 6)
            ax.set_ylim(-0.05, 1.05)
            ax.legend(loc='lower left', title='Choice')
            
            figs[subj].append(fig)
            
    return figs


# %% Choice Regressions

def logit_regress_side_choice(sess_data, n_back=5, separate_block_rates=True, include_winstay=False, include_reward=False, 
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

        block_rates = np.sort(subj_sess_resp['block_prob'].unique())

        reg_groups = ['All']
        if separate_block_rates:
            reg_groups.extend(block_rates)
        
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
        
        if len(reg_groups) == 1:
            axs = np.array([axs])

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


def logit_regress_stay_choice(sess_data, n_back=5, separate_block_rates=True, fit_switches=False, include_unreward=False, include_choice=True, 
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

        block_rates = np.sort(subj_sess_resp['block_prob'].unique())

        reg_groups = ['All']
        if separate_block_rates:
            reg_groups.extend(block_rates)
        
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
    bins = pd.IntervalIndex.from_breaks(time_bin_edges)

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


