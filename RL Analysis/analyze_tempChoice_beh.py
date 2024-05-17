# -*- coding: utf-8 -*-
"""
Script to investigate performance on the intertemporal choice task

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

subj_ids = [179]
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=3)
sess_ids = bah.limit_sess_ids(sess_ids, 12)
#sess_ids = {179: [95201, 95312, 95347]}

# get trial information
loc_db = db.LocalDB_BasicRLTasks('temporalChoice')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids)) #, reload=True
# make slow delay a string for better plot formatting
all_sess['slow_delay'] = all_sess['slow_delay'].apply(lambda x: '{:.0f}'.format(x))

# %% TRIAL COUNTS

# aggregate count tables into dictionary
count_columns = ['block_rates', 'block_rewards', 'slow_delay', 'block_rates_delay', 'block_rewards_delay']
column_titles = ['Block Reward Rates (Fast/Slow)', 'Block Rewards (Fast/Slow)', 'Block Slow Delays', 'Block Rates and Slow Delay', 'Block Rewards and Slow Delay']
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

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_no_instruct = subj_sess[~subj_sess['instruct_trial']]
    subj_sess_resp = subj_sess_no_instruct[subj_sess_no_instruct['choice'] != 'none']

    choose_fast_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_fast_port', [['block_rates', 'slow_delay'], ['block_rewards', 'slow_delay']])
    
    # plot by reward rates
    fig = plt.figure(layout='constrained', figsize=(8, 4))
    fig.suptitle('Choose Fast Port Probabilities (Rat {})'.format(subj_id))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1,1])
    
    # plot choose fast for each reward rate and slow delay in both a heatmap matrix and a lineplot
    ax = fig.add_subplot(gs[0])
    bah.plot_rate_heatmap(choose_fast_side_probs, 'slow_delay', 'Slow Reward Delay', 'block_rates', 'Block Reward Rates (fast/slow, μL/s)', ax=ax)
    
    # line plot
    ax = fig.add_subplot(gs[1])
    data = choose_fast_side_probs['block_rates x slow_delay']
    #x_vals = [float(x) for x in choose_fast_side_probs['slow_delay'].index.to_list()]
    rates = choose_fast_side_probs['block_rates'].index.to_list()
    
    for rate in rates:
        rate_data = data.loc[[rate]]
        rate_x_vals = [float(x) for x in rate_data.index.get_level_values('slow_delay')]
        ax.errorbar(rate_x_vals, rate_data['rate'], yerr=bah.convert_rate_err_to_mat(rate_data), fmt='o-', capsize=4, label=rate)
    
    ax.set_ylabel('p(Choose Fast)')
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylim(0, 1)
    ax.set_title('Choose Fast Side')
    ax.legend(title='Reward Rates (Fast/Slow, μL/s)', ncols=2)
    
    # plot by reward volumes
    fig = plt.figure(layout='constrained', figsize=(8, 4))
    fig.suptitle('Choose Fast Port Probabilities (Rat {})'.format(subj_id))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1,1])
    
    # plot choose fast for each reward rate and slow delay in both a heatmap matrix and a lineplot
    ax = fig.add_subplot(gs[0])
    bah.plot_rate_heatmap(choose_fast_side_probs, 'slow_delay', 'Slow Reward Delay', 'block_rewards', 'Block Reward Volumes (fast/slow, μL)', ax=ax)
    
    # line plot
    ax = fig.add_subplot(gs[1])
    data = choose_fast_side_probs['block_rewards x slow_delay']
    #x_vals = [float(x) for x in choose_fast_side_probs['slow_delay'].index.to_list()]
    rewards = choose_fast_side_probs['block_rewards'].index.to_list()
    
    for reward in rewards:
        reward_data = data.loc[[reward]]
        reward_x_vals = [float(x) for x in reward_data.index.get_level_values('slow_delay')]
        ax.errorbar(reward_x_vals, reward_data['rate'], yerr=bah.convert_rate_err_to_mat(reward_data), fmt='o-', capsize=4, label=reward)
    
    ax.set_ylabel('p(Choose Fast)')
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylim(0, 1)
    ax.set_title('Choose Fast Side')
    ax.legend(title='Reward Volumes (Fast/Slow, μL)', ncols=2)

    
    # SWITCHING PROBABILITIES
    # get switching rates by previous fast/slow choice in each block rate and slow choice delay
    block_rates = np.sort(subj_sess_resp['block_rates'].unique())
    slow_delays = np.sort(subj_sess_resp['slow_delay'].unique())
    
    n_switches = {br: {d: {t: {'k': 0, 'n': 0} for t in ['fast', 'slow']} for d in slow_delays} for br in block_rates}

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id]

        if len(ind_sess) == 0:
            continue

        choices = ind_sess['choice'].to_numpy()
        switches = choices[:-1] != choices[1:]
        block_rate = ind_sess['block_rates'].to_numpy()[:-1]
        slow_delay = ind_sess['slow_delay'].to_numpy()[:-1]
        fast_choice = ind_sess['chose_fast_port'].to_numpy()[:-1]
        
        # ignore any switches between block switches
        block_trans_sel = ind_sess['block_trial'].diff() < 0
        switches = switches[~block_trans_sel]
        block_rate = block_rate[~block_trans_sel]
        slow_delay = slow_delay[~block_trans_sel]
        fast_choice = fast_choice[~block_trans_sel]

        for br in block_rates:
            rate_sel = block_rate == br
            for d in slow_delays:
                delay_sel = (slow_delay == d) & rate_sel
                fast_sel = delay_sel & fast_choice
                slow_sel = delay_sel & ~fast_choice
            
                n_switches[br][d]['fast']['k'] += sum(fast_sel & switches)
                n_switches[br][d]['fast']['n'] += sum(fast_sel)
                n_switches[br][d]['slow']['k'] += sum(slow_sel & switches)
                n_switches[br][d]['slow']['n'] += sum(slow_sel)
                
            
    # # plot results
    # # define reusable helper methods
    # def comp_p(n_dict): return n_dict['k']/n_dict['n']
    # def comp_err(n_dict): return abs(utils.binom_cis(n_dict['k'], n_dict['n']) - comp_p(n_dict))
    
    # fig = plt.figure(layout='constrained', figsize=(9, 7))
    # fig.suptitle('Switching Probabilities (Rat {})'.format(subj_id))
    # gs = GridSpec(2, len(block_rates), figure=fig, height_ratios=[3,2])
    
    # # first row, left, is the win-stay/lose-switch rates by choice probability
    # ax = fig.add_subplot(gs[0, :-1])
    # stay_reward_vals = [comp_p(n_stay_reward_choice[p]) for p in choice_probs]
    # stay_reward_err = np.asarray([comp_err(n_stay_reward_choice[p]) for p in choice_probs]).T
    # switch_noreward_vals = [comp_p(n_switch_noreward_choice[p]) for p in choice_probs]
    # switch_noreward_err = np.asarray([comp_err(n_switch_noreward_choice[p]) for p in choice_probs]).T

    # ax.errorbar(choice_probs, stay_reward_vals, yerr=stay_reward_err, fmt='o', capsize=4, label='Win Stay')
    # ax.errorbar(choice_probs, switch_noreward_vals, yerr=switch_noreward_err, fmt='o', capsize=4, label='Lose Switch')
    # ax.set_ylabel('Proportion of Choices')
    # ax.set_xlabel('Choice Reward Probability (%)')
    # ax.set_xticks(choice_probs, ['{:.0f}'.format(p) for p in choice_probs*100])
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-0.05, 1.05)
    # ax.grid(axis='y')
    # ax.legend(loc='best')
    
    # # first row, right, is the stay percentage after reward/no reward by block rate
    # ax = fig.add_subplot(gs[0, -1])
    # stay_reward_vals = [comp_p(n_stay_reward_block[br]) for br in block_rates]
    # stay_reward_err = np.asarray([comp_err(n_stay_reward_block[br]) for br in block_rates]).T
    # stay_noreward_vals = [comp_p(n_stay_noreward_block[br]) for br in block_rates]
    # stay_noreward_err = np.asarray([comp_err(n_stay_noreward_block[br]) for br in block_rates]).T

    # plot_utils.plot_stacked_bar([stay_reward_vals, stay_noreward_vals], value_labels=['Rewarded', 'Unrewarded'],
    #                             x_labels=block_rates, err=[stay_reward_err, stay_noreward_err], ax=ax)
    # ax.set_ylabel('p(Stay)')
    # ax.set_xlabel('Block Reward Probability (Left/Right %)')
    # ax.set_ylim(-0.05, 1.05)
    # ax.legend(loc='lower left')
    
    # # second row is empirical transition matrices
    # for i, br in enumerate(block_rates):
    #     ax = fig.add_subplot(gs[1, i])
    #     p_mat = trans_mats[br]['k']/trans_mats[br]['n']
    #     plot_utils.plot_value_matrix(p_mat, ax=ax, xticklabels=['high', 'low'], yticklabels=['high', 'low'], cbar=False, cmap='vlag')
    #     ax.set_ylabel('Choice Reward Rate')
    #     ax.set_xlabel('Next Choice Reward Rate')
    #     ax.xaxis.set_label_position('top')
    #     ax.set_title('Block Rate {}%'.format(br))
        
        
    # # Simple summary of choice behavior
    # fig = plt.figure(layout='constrained', figsize=(6, 3))
    # fig.suptitle('Choice Probabilities ({})'.format(subj_id))
    # gs = GridSpec(1, 2, figure=fig, width_ratios=[3,4])

    # data = choose_high_trial_probs['block_prob']
    # x_labels = data.index.to_list()
    # x_vals = np.array([int(x.split('/')[0]) for x in x_labels])
    
    # ax = fig.add_subplot(gs[0, 0])
    # ax.errorbar(x_vals, data['rate'], yerr=bah.convert_rate_err_to_mat(data), fmt='o', capsize=4)
    # plot_utils.plot_dashlines(x_vals/100, dir='h', ax=ax)
    # ax.set_ylabel('p(Choose High)')
    # ax.set_xlabel('Block Reward Probability (High/Low %)')
    # ax.set_xticks(x_vals, x_labels)
    # ax.set_xlim(50, 100)
    # ax.set_ylim(0.5, 1)
    # ax.set_title('Choose High Side')

    
    # stay_reward_vals = [comp_p(n_stay_reward_block[br]) for br in block_rates]
    # stay_reward_err = np.asarray([comp_err(n_stay_reward_block[br]) for br in block_rates]).T
    # stay_noreward_vals = [comp_p(n_stay_noreward_block[br]) for br in block_rates]
    # stay_noreward_err = np.asarray([comp_err(n_stay_noreward_block[br]) for br in block_rates]).T

    # ax = fig.add_subplot(gs[0, 1])
    # plot_utils.plot_stacked_bar([stay_reward_vals, stay_noreward_vals], value_labels=['Rewarded', 'Unrewarded'],
    #                             x_labels=block_rates, err=[stay_reward_err, stay_noreward_err], ax=ax)
    # ax.set_ylabel('p(Stay)')
    # ax.set_xlabel('Block Reward Probability (High/Low %)')
    # ax.set_ylim(-0.05, 1.05)
    # ax.legend(loc='lower right')
    # ax.set_title('Choose Previous Side')

    
            
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
        
        cpoke_in_bins = pd.cut(ind_sess['cpoke_in_time'], bins)[1:] # ignore first trial time since no trial before

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