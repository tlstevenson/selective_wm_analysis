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

sess_ids = db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 3)

# optionally limit sessions based on subject ids
subj_ids = [179,188,207,182]
sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

#sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=3)
#sess_ids = bah.limit_sess_ids(sess_ids, 12, last_idx=-1)

# get trial information
loc_db = db.LocalDB_BasicRLTasks('temporalChoice')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids)) #, reload=True
# make slow delay a string for better plot formatting
all_sess['slow_delay'] = all_sess['slow_delay'].apply(lambda x: '{:.0f}'.format(x))
all_sess['cpoke_out_time'] = all_sess['cpoke_out_time'].apply(lambda x: x if utils.is_scalar(x) else np.nan)
all_sess['cpoke_out_to_resp_time'] = all_sess['response_time'] - all_sess['cpoke_out_time']

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

    choose_fast_side_probs = bah.get_rate_dict(subj_sess_resp, 'chose_fast_port', [['block_rates', 'slow_delay'], ['block_rewards', 'slow_delay'], ['block_rates', 'fast_port'], ['block_rewards', 'fast_port']])
    choose_left_probs = bah.get_rate_dict(subj_sess_resp, 'chose_left', [['side_rewards', 'slow_delay']])

    # plot by reward rates
    fig = plt.figure(layout='constrained', figsize=(8, 7))
    fig.suptitle('Choice Probabilities (Rat {})'.format(subj_id))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1,1])

    # plot choose fast for reward rate and fast port side in heatmap
    ax = fig.add_subplot(gs[0,0])
    bah.plot_rate_heatmap(choose_fast_side_probs, 'fast_port', 'Fast Port', 'block_rates', 'Block Reward Rates (fast/slow, μL/s)', ax=ax)
    ax.set_title('Choose Fast Side')

    # line plot
    ax = fig.add_subplot(gs[0,1])
    data = choose_fast_side_probs['block_rates x slow_delay']
    #x_vals = [float(x) for x in choose_fast_side_probs['slow_delay'].index.to_list()]
    rates = choose_fast_side_probs['block_rates']['block_rates']

    for rate in rates:
        rate_data = data[data['block_rates'] == rate]
        rate_x_vals = [float(x) for x in rate_data['slow_delay']]
        ax.errorbar(rate_x_vals, rate_data['rate'], yerr=bah.convert_rate_err_to_mat(rate_data), fmt='o-', capsize=4, label=rate)

    ax.set_ylabel('p(Choose Fast)')
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylim(0, 1)
    ax.set_title('Choose Fast Side')
    ax.legend(title='Reward Rates (Fast/Slow, μL/s)', ncols=2)

    ax = fig.add_subplot(gs[1,:])
    bah.plot_rate_heatmap(choose_left_probs, 'side_rewards', 'Side Reward Volumes (left/right, μL)', 'slow_delay', 'Slow Reward Delay', ax=ax)
    ax.set_title('Choose Left Side')


    # plot same thing by reward volumes
    fig = plt.figure(layout='constrained', figsize=(8, 7))
    fig.suptitle('Choice Probabilities (Rat {})'.format(subj_id))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1,1])

    # plot choose fast for reward rate and fast port side in heatmap
    ax = fig.add_subplot(gs[0,0])
    bah.plot_rate_heatmap(choose_fast_side_probs, 'fast_port', 'Fast Port', 'block_rewards', 'Block Reward Volumes (fast/slow, μL)', ax=ax)
    ax.set_title('Choose Fast Side')

    # line plot
    ax = fig.add_subplot(gs[0,1])
    data = choose_fast_side_probs['block_rewards x slow_delay']
    #x_vals = [float(x) for x in choose_fast_side_probs['slow_delay'].index.to_list()]
    rates = choose_fast_side_probs['block_rewards']['block_rewards']

    for rate in rates:
        rate_data = data[data['block_rewards'] == rate]
        rate_x_vals = [float(x) for x in rate_data['slow_delay']]
        ax.errorbar(rate_x_vals, rate_data['rate'], yerr=bah.convert_rate_err_to_mat(rate_data), fmt='o-', capsize=4, label=rate)

    ax.set_ylabel('p(Choose Fast)')
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylim(0, 1)
    ax.set_title('Choose Fast Side')
    ax.legend(title='Reward Volumes (Fast/Slow, μL)', ncols=2)

    ax = fig.add_subplot(gs[1,:])
    bah.plot_rate_heatmap(choose_left_probs, 'side_rewards', 'Side Reward Volumes (left/right, μL)', 'slow_delay', 'Slow Reward Delay', ax=ax)
    ax.set_title('Choose Left Side')


    # SWITCHING PROBABILITIES
    # get switching rates by previous fast/slow choice in each block rate and slow choice delay
    block_rates = np.sort(subj_sess_resp['block_rates'].unique())
    slow_delays = np.sort(subj_sess_resp['slow_delay'].unique())
    trial_groups = ['fast', 'slow', 'all']

    n_switches = {br: {d: {t: {'k': 0, 'n': 0} for t in trial_groups} for d in slow_delays} for br in block_rates}

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess_resp[subj_sess_resp['sessid'] == sess_id].reset_index(drop=True)

        if len(ind_sess) == 0:
            continue

        choices = ind_sess['choice'].to_numpy()
        switches = choices[:-1] != choices[1:]
        block_rate = ind_sess['block_rates'].to_numpy()[:-1]
        slow_delay = ind_sess['slow_delay'].to_numpy()[:-1]
        fast_choice = ind_sess['chose_fast_port'].to_numpy()[:-1]

        # ignore any switches between block switches
        block_trans_sel = ind_sess['block_trial'].diff()[1:] < 0
        switches = switches[~block_trans_sel]
        block_rate = block_rate[~block_trans_sel]
        slow_delay = slow_delay[~block_trans_sel]
        fast_choice = fast_choice[~block_trans_sel]

        for br in block_rates:
            rate_sel = block_rate == br
            for d in slow_delays:
                delay_sel = (slow_delay == d) & rate_sel
                for group in trial_groups:
                    match group:
                        case 'fast':
                            sel = delay_sel & fast_choice
                        case 'slow':
                            sel = delay_sel & ~fast_choice
                        case 'all':
                            sel = delay_sel

                    n_switches[br][d][group]['k'] += sum(sel & switches)
                    n_switches[br][d][group]['n'] += sum(sel)


    # plot results
    # define reusable helper methods
    def comp_p(n_dict): return n_dict['k']/n_dict['n']
    def comp_err(n_dict): return abs(utils.binom_cis(n_dict['k'], n_dict['n']) - comp_p(n_dict))

    x_vals = [float(x) for x in slow_delays]

    fig, axs = plt.subplots(3, 1, layout='constrained', figsize=(5, 8))
    fig.suptitle('Switching Probabilities (Rat {})'.format(subj_id))

    trial_group_titles = {'fast': 'Fast', 'slow': 'Slow', 'all': 'Either'}
    for i, group in enumerate(trial_groups):

        ax = axs[i]

        for br in block_rates:
            y_vals = []
            y_err = []
            for d in slow_delays:
                n_dict = n_switches[br][d][group]
                y_vals.append(comp_p(n_dict))
                y_err.append(comp_err(n_dict))

            ax.errorbar(x_vals, y_vals, yerr=np.array(y_err).T, fmt='o-', capsize=4, label=br)

        ax.set_ylabel('p(Switch)')
        ax.set_xlabel('Slow Reward Delay (s)')
        ax.set_ylim(0, 1)
        ax.set_title('Switch Rates After Choosing {} Side'.format(trial_group_titles[group]))
        ax.legend(title='Reward Rates (Fast/Slow, μL/s)', ncols=2)


# %% Response Times

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_no_instruct = subj_sess[~subj_sess['instruct_trial']]
    # subj_sess_resp = subj_sess_no_instruct[subj_sess_no_instruct['choice'] != 'none']

    # organize time to reward and time to engage center port by prior reward, block probabilities, and whether switched response
    timing_data = pd.DataFrame(columns=['RT', 'cpoke_out_to_resp_time', 'cpoke_in_latency', 'next_cpoke_in_latency',
                                        'choice', 'port_speed_choice', 'choice_rate', 'choice_delay', 'reward',
                                        'block_rates', 'slow_delay', 'all'])

    for sess_id in sess_ids[subj_id]:
        ind_sess = subj_sess_no_instruct[subj_sess_no_instruct['sessid'] == sess_id].reset_index(drop=True)

        if len(ind_sess) == 0:
            continue

        data = ind_sess[['RT', 'cpoke_out_to_resp_time', 'cpoke_in_latency', 'next_cpoke_in_latency', 'choice',
                         'port_speed_choice', 'choice_rate', 'choice_delay', 'reward', 'block_rates', 'slow_delay']].iloc[1:] # ignore first trial because no trial before
        # add response type labels
        choices = ind_sess['choice'].to_numpy()
        stays = choices[:-1] == choices[1:]

        # ignore trials after a block switch ans trials where current response or previous response was none
        # ignore trials where the current response or the past response was none
        trial_sel = (choices[:-1] != 'none') & (choices[1:] != 'none') & ~(ind_sess['block_trial'].diff()[1:] < 0)
        stays = stays[trial_sel]
        data = data[trial_sel].reset_index(drop=True)

        data['response_type'] = pd.Series(stays).apply(lambda x: 'stay' if x else 'switch')
        data['all_rates'] = 'all'
        data['all_delays'] = 'all'

        timing_data = pd.concat([timing_data, data], ignore_index=True)

    # collapse trial types into one column and add 'all' to block rates and slow delays columns
    timing_data = pd.lreshape(timing_data, {'type': ['choice', 'response_type', 'port_speed_choice']})
    timing_data = pd.lreshape(timing_data, {'rates': ['block_rates', 'all_rates']})
    timing_data = pd.lreshape(timing_data, {'delays': ['slow_delay', 'all_delays']})

    # plot for RTs
    fig = plt.figure(layout='constrained', figsize=(8, 10))
    fig.suptitle('Response Latencies (Cue to Response) - Rat {}'.format(subj_id))
    gs = GridSpec(4, 1, figure=fig)

    ax = fig.add_subplot(gs[0])
    sb.boxplot(data=timing_data, x='rates', y='RT', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Block Rates by Choice Type')
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[1])
    sb.boxplot(data=timing_data, x='delays', y='RT', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Slow Choice Delays by Choice Type')
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[2])
    sb.boxplot(data=timing_data, x='rates', y='RT', hue='choice_delay', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Block Rates by Choice Delay')
    ax.legend(title='Choice Delay', loc='upper right', ncol=2)

    ax = fig.add_subplot(gs[3])
    sb.boxplot(data=timing_data, x='delays', y='RT', hue='reward', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Slow Choice Delays by Reward Volume')
    ax.legend(title='Reward Volume', loc='upper right', ncol=2)

    # plot for poke out to response times
    fig = plt.figure(layout='constrained', figsize=(8, 10))
    fig.suptitle('Response Latencies (Poke Out to Response) - Rat {}'.format(subj_id))
    gs = GridSpec(4, 1, figure=fig)

    ax = fig.add_subplot(gs[0])
    sb.boxplot(data=timing_data, x='rates', y='cpoke_out_to_resp_time', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Block Rates by Choice Type')
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[1])
    sb.boxplot(data=timing_data, x='delays', y='cpoke_out_to_resp_time', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Slow Choice Delays by Choice Type')
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[2])
    sb.boxplot(data=timing_data, x='rates', y='cpoke_out_to_resp_time', hue='choice_delay', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Block Rates by Choice Delay')
    ax.legend(title='Choice Delay', loc='upper right', ncol=2)

    ax = fig.add_subplot(gs[3])
    sb.boxplot(data=timing_data, x='delays', y='cpoke_out_to_resp_time', hue='reward', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Response Latency (s)')
    ax.set_title('Slow Choice Delays by Reward Volume')
    ax.legend(title='Reward Volume', loc='upper right', ncol=2)

    # plot for next cpoke in latencies
    y_max = 30
    cpoke_in_data = timing_data[timing_data['next_cpoke_in_latency'] > 0.001]

    fig = plt.figure(layout='constrained', figsize=(8, 10))
    fig.suptitle('Center Poke In Latencies (Port On to Poke In, Post Response) - Rat {}'.format(subj_id))
    gs = GridSpec(4, 1, figure=fig)

    ax = fig.add_subplot(gs[0])
    sb.boxplot(data=cpoke_in_data, x='rates', y='next_cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Block Rates by Choice Type')
    ax.set_ylim(0, y_max)
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[1])
    sb.boxplot(data=cpoke_in_data, x='delays', y='next_cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Slow Choice Delays by Choice Type')
    ax.set_ylim(0, y_max)
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[2])
    sb.boxplot(data=cpoke_in_data, x='rates', y='next_cpoke_in_latency', hue='choice_delay', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Block Rates by Choice Delay')
    ax.set_ylim(0, y_max)
    ax.legend(title='Choice Delay', loc='upper right', ncol=2)

    ax = fig.add_subplot(gs[3])
    sb.boxplot(data=cpoke_in_data, x='delays', y='next_cpoke_in_latency', hue='reward', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Slow Choice Delays by Reward Volume')
    ax.set_ylim(0, y_max)
    ax.legend(title='Reward Volume', loc='upper right', ncol=2)


    # plot for current trial cpoke in latencies
    cpoke_in_data = timing_data[timing_data['cpoke_in_latency'] > 0.001]

    fig = plt.figure(layout='constrained', figsize=(8, 10))
    fig.suptitle('Center Poke In Latencies (Port On to Poke In, Pre Response) - Rat {}'.format(subj_id))
    gs = GridSpec(4, 1, figure=fig)

    ax = fig.add_subplot(gs[0])
    sb.boxplot(data=cpoke_in_data, x='rates', y='cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Block Rates by Choice Type')
    ax.set_ylim(0, y_max)
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[1])
    sb.boxplot(data=cpoke_in_data, x='delays', y='cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Slow Choice Delays by Choice Type')
    ax.set_ylim(0, y_max)
    ax.legend(ncol=3, title='Choice Type', loc='upper right')

    ax = fig.add_subplot(gs[2])
    sb.boxplot(data=cpoke_in_data, x='rates', y='cpoke_in_latency', hue='choice_delay', ax=ax) # gap=0.1,
    ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Block Rates by Choice Delay')
    ax.set_ylim(0, y_max)
    ax.legend(title='Choice Delay', loc='upper right', ncol=2)

    ax = fig.add_subplot(gs[3])
    sb.boxplot(data=cpoke_in_data, x='delays', y='cpoke_in_latency', hue='reward', ax=ax) # gap=0.1,
    ax.set_xlabel('Slow Reward Delay (s)')
    ax.set_ylabel('Cpoke In Latency (s)')
    ax.set_title('Slow Choice Delays by Reward Volume')
    ax.set_ylim(0, y_max)
    ax.legend(title='Reward Volume', loc='upper right', ncol=2)
