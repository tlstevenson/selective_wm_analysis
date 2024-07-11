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
# stage = 4
# if active_subjects_only:
#     subject_info = subject_info[subject_info['stage'] == stage]
# else:
#     subject_info = subject_info[subject_info['stage'] >= stage]

# subj_ids = subject_info['subjid']

subj_ids = [179]
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=4)

# get trial information
loc_db = db.LocalDB_BasicRLTasks('foraging')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids)) #, reload=True

# %% PATCH COUNTS

# aggregate count tables into dictionary
count_columns = ['reward_port', 'initial_reward', 'depletion_rate', 'reward_depletion_rate', 'patch_switch_delay', 'reward_depletion_rate_switch_delay']
column_titles = ['Reward Port', 'Initial Reward', 'Depletion Rate', 'Initial Reward & Deplation Rate', 'Patch Switch Delay', 'Initial Reward, Deplation Rate, & Patch Switch Delay']

# only want unique patches
patch_info = all_sess[['sessid', 'subjid', 'block_num', *count_columns]].drop_duplicates()

count_dict = bah.get_count_dict(patch_info, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(patch_info, 'subjid', count_columns, normalize=True)

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

grouping_columns = ['reward_port', 'initial_reward', 'depletion_rate', 'patch_switch_delay']

for subj_id in subj_ids:
    subj_sess = all_sess[all_sess['subjid'] == subj_id]
    subj_sess_resp = subj_sess[subj_sess['choice'] != 'none']
    center_choice_sel = subj_sess['chose_center']

    # harvests per patch
    harvest_count_data = subj_sess[center_choice_sel][['patch_harvest_count', *grouping_columns]]
    # drop zero counts
    harvest_count_data = harvest_count_data[harvest_count_data['patch_harvest_count'] > 0]

    # Last reward before a switch
    prev_rew_data = subj_sess[center_choice_sel][['prev_reward', *grouping_columns]]
    # drop zero counts
    prev_rew_data = prev_rew_data[prev_rew_data['prev_reward'] > 0]

    # switch_delays = np.unique(harvest_count_data['patch_switch_delay'])
    # switch_delays = np.insert(switch_delays.astype(object), 0, 'All')
    switch_delays = ['All']

    for data, col, title, y_label in zip([harvest_count_data, prev_rew_data],
                                         ['patch_harvest_count', 'prev_reward'],
                                         ['Harvests Per Patch', 'Reward Before Switch'],
                                         ['# Harvests', 'Last Reward (μL)']):

        for delay in switch_delays:
            if delay != 'All':
                delay_data = data[data['patch_switch_delay'] == delay]
            else:
                delay_data = data

            delay_avg_info = bah.get_avg_value_dict(delay_data, col,
                                                    [*grouping_columns,
                                                    ['initial_reward', 'depletion_rate'],
                                                    ['initial_reward', 'reward_port'],
                                                    ['depletion_rate', 'reward_port'],
                                                    ['initial_reward', 'depletion_rate', 'reward_port']])

            def filter_df(df, col, val):
                return df[df[col] == val]


            fig = plt.figure(layout='constrained', figsize=(10, 7))
            fig.suptitle('{}, Switch Delay: {} (Rat {})'.format(title, delay, subj_id))
            gs = GridSpec(2, 6, figure=fig)

            init_rewards = delay_avg_info['initial_reward']['initial_reward']
            decay_rates = delay_avg_info['depletion_rate']['depletion_rate']
            sides = ['left', 'right']

            # plot by reward amount and side
            ax = fig.add_subplot(gs[0,0:3])
            ax.set_title('By Initial Reward')

            plot_values = [delay_avg_info['initial_reward']['avg'], *[filter_df(delay_avg_info['initial_reward x reward_port'], 'reward_port', s)['avg'] for s in sides]]
            plot_errors = [delay_avg_info['initial_reward']['se'], *[filter_df(delay_avg_info['initial_reward x reward_port'], 'reward_port', s)['se'] for s in sides]]
            plot_utils.plot_grouped_error_bar(plot_values, plot_errors, value_labels=['All', 'Left', 'Right'], x_labels=init_rewards, ax=ax)
            ax.set_ylabel(y_label)
            ax.set_xlabel('Initial Reward (μL)')
            ax.legend(title='Sides')

            # plot by decay rate and side
            ax = fig.add_subplot(gs[0,3:6], sharey=ax)
            ax.set_title('By Decay Rate')

            plot_values = [delay_avg_info['depletion_rate']['avg'], *[filter_df(delay_avg_info['depletion_rate x reward_port'], 'reward_port', s)['avg'] for s in sides]]
            plot_errors = [delay_avg_info['depletion_rate']['se'], *[filter_df(delay_avg_info['depletion_rate x reward_port'], 'reward_port', s)['se'] for s in sides]]
            plot_utils.plot_grouped_error_bar(plot_values, plot_errors, value_labels=['All', 'Left', 'Right'], x_labels=decay_rates, ax=ax)

            ax.set_xlabel('Depletion Rate')
            ax.legend(title='Sides')

            # plot by decay rate and side
            # all sides
            ax = fig.add_subplot(gs[1,0:2])
            ax.set_title('By Initial Reward & Decay Rate, All')

            plot_values = [filter_df(delay_avg_info['initial_reward x depletion_rate'], 'depletion_rate', r)['avg'] for r in decay_rates]
            plot_errors = [filter_df(delay_avg_info['initial_reward x depletion_rate'], 'depletion_rate', r)['se'] for r in decay_rates]
            plot_utils.plot_grouped_error_bar(plot_values, plot_errors, value_labels=decay_rates, x_labels=init_rewards, ax=ax)

            ax.set_ylabel(y_label)
            ax.set_xlabel('Initial Reward (μL)')
            ax.legend(title='Depletion Rate')

            # left side
            ax = fig.add_subplot(gs[1,2:4], sharey=ax)
            ax.set_title('By Initial Reward & Decay Rate, Left')

            plot_values = [filter_df(filter_df(delay_avg_info['initial_reward x depletion_rate x reward_port'], 'reward_port', 'left'), 'depletion_rate', r)['avg'] for r in decay_rates]
            plot_errors = [filter_df(filter_df(delay_avg_info['initial_reward x depletion_rate x reward_port'], 'reward_port', 'left'), 'depletion_rate', r)['se'] for r in decay_rates]
            plot_utils.plot_grouped_error_bar(plot_values, plot_errors, value_labels=decay_rates, x_labels=init_rewards, ax=ax)

            ax.set_xlabel('Initial Reward (μL)')
            ax.legend(title='Depletion Rate')

            # right side
            ax = fig.add_subplot(gs[1,4:6], sharey=ax)
            ax.set_title('By Initial Reward & Decay Rate, Right')

            plot_values = [filter_df(filter_df(delay_avg_info['initial_reward x depletion_rate x reward_port'], 'reward_port', 'right'), 'depletion_rate', r)['avg'] for r in decay_rates]
            plot_errors = [filter_df(filter_df(delay_avg_info['initial_reward x depletion_rate x reward_port'], 'reward_port', 'right'), 'depletion_rate', r)['se'] for r in decay_rates]
            plot_utils.plot_grouped_error_bar(plot_values, plot_errors, value_labels=decay_rates, x_labels=init_rewards, ax=ax)

            ax.set_xlabel('Initial Reward (μL)')
            ax.legend(title='Depletion Rate')

# %% Probability of switching patches by response

# TODO


# %% Response Times

# TODO
# for subj_id in subj_ids:
#     subj_sess = all_sess[all_sess['subjid'] == subj_id]
#     subj_sess_no_instruct = subj_sess[~subj_sess['instruct_trial']]
#     # subj_sess_resp = subj_sess_no_instruct[subj_sess_no_instruct['choice'] != 'none']

#     # organize time to reward and time to engage center port by prior reward, block probabilities, and whether switched response
#     timing_data = pd.DataFrame(columns=['RT', 'cpoke_out_to_resp_time', 'cpoke_in_latency', 'next_cpoke_in_latency',
#                                         'choice', 'port_speed_choice', 'choice_rate', 'choice_delay', 'reward',
#                                         'block_rates', 'slow_delay', 'all'])

#     for sess_id in sess_ids[subj_id]:
#         ind_sess = subj_sess_no_instruct[subj_sess_no_instruct['sessid'] == sess_id].reset_index(drop=True)

#         if len(ind_sess) == 0:
#             continue

#         data = ind_sess[['RT', 'cpoke_out_to_resp_time', 'cpoke_in_latency', 'next_cpoke_in_latency', 'choice',
#                          'port_speed_choice', 'choice_rate', 'choice_delay', 'reward', 'block_rates', 'slow_delay']].iloc[1:] # ignore first trial because no trial before
#         # add response type labels
#         choices = ind_sess['choice'].to_numpy()
#         stays = choices[:-1] == choices[1:]

#         # ignore trials after a block switch ans trials where current response or previous response was none
#         # ignore trials where the current response or the past response was none
#         trial_sel = (choices[:-1] != 'none') & (choices[1:] != 'none') & ~(ind_sess['block_trial'].diff()[1:] < 0)
#         stays = stays[trial_sel]
#         data = data[trial_sel].reset_index(drop=True)

#         data['response_type'] = pd.Series(stays).apply(lambda x: 'stay' if x else 'switch')
#         data['all_rates'] = 'all'
#         data['all_delays'] = 'all'

#         timing_data = pd.concat([timing_data, data], ignore_index=True)

#     # collapse trial types into one column and add 'all' to block rates and slow delays columns
#     timing_data = pd.lreshape(timing_data, {'type': ['choice', 'response_type', 'port_speed_choice']})
#     timing_data = pd.lreshape(timing_data, {'rates': ['block_rates', 'all_rates']})
#     timing_data = pd.lreshape(timing_data, {'delays': ['slow_delay', 'all_delays']})

#     # plot for RTs
#     fig = plt.figure(layout='constrained', figsize=(8, 10))
#     fig.suptitle('Response Latencies (Cue to Response) - Rat {}'.format(subj_id))
#     gs = GridSpec(4, 1, figure=fig)

#     ax = fig.add_subplot(gs[0])
#     sb.boxplot(data=timing_data, x='rates', y='RT', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Block Rates by Choice Type')
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[1])
#     sb.boxplot(data=timing_data, x='delays', y='RT', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Slow Choice Delays by Choice Type')
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[2])
#     sb.boxplot(data=timing_data, x='rates', y='RT', hue='choice_delay', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Block Rates by Choice Delay')
#     ax.legend(title='Choice Delay', loc='upper right', ncol=2)

#     ax = fig.add_subplot(gs[3])
#     sb.boxplot(data=timing_data, x='delays', y='RT', hue='reward', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Slow Choice Delays by Reward Volume')
#     ax.legend(title='Reward Volume', loc='upper right', ncol=2)

#     # plot for poke out to response times
#     fig = plt.figure(layout='constrained', figsize=(8, 10))
#     fig.suptitle('Response Latencies (Poke Out to Response) - Rat {}'.format(subj_id))
#     gs = GridSpec(4, 1, figure=fig)

#     ax = fig.add_subplot(gs[0])
#     sb.boxplot(data=timing_data, x='rates', y='cpoke_out_to_resp_time', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Block Rates by Choice Type')
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[1])
#     sb.boxplot(data=timing_data, x='delays', y='cpoke_out_to_resp_time', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Slow Choice Delays by Choice Type')
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[2])
#     sb.boxplot(data=timing_data, x='rates', y='cpoke_out_to_resp_time', hue='choice_delay', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Block Rates by Choice Delay')
#     ax.legend(title='Choice Delay', loc='upper right', ncol=2)

#     ax = fig.add_subplot(gs[3])
#     sb.boxplot(data=timing_data, x='delays', y='cpoke_out_to_resp_time', hue='reward', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Response Latency (s)')
#     ax.set_title('Slow Choice Delays by Reward Volume')
#     ax.legend(title='Reward Volume', loc='upper right', ncol=2)

#     # plot for next cpoke in latencies
#     y_max = 30
#     cpoke_in_data = timing_data[timing_data['next_cpoke_in_latency'] > 0.001]

#     fig = plt.figure(layout='constrained', figsize=(8, 10))
#     fig.suptitle('Center Poke In Latencies (Port On to Poke In, Post Response) - Rat {}'.format(subj_id))
#     gs = GridSpec(4, 1, figure=fig)

#     ax = fig.add_subplot(gs[0])
#     sb.boxplot(data=cpoke_in_data, x='rates', y='next_cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Block Rates by Choice Type')
#     ax.set_ylim(0, y_max)
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[1])
#     sb.boxplot(data=cpoke_in_data, x='delays', y='next_cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Slow Choice Delays by Choice Type')
#     ax.set_ylim(0, y_max)
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[2])
#     sb.boxplot(data=cpoke_in_data, x='rates', y='next_cpoke_in_latency', hue='choice_delay', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Block Rates by Choice Delay')
#     ax.set_ylim(0, y_max)
#     ax.legend(title='Choice Delay', loc='upper right', ncol=2)

#     ax = fig.add_subplot(gs[3])
#     sb.boxplot(data=cpoke_in_data, x='delays', y='next_cpoke_in_latency', hue='reward', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Slow Choice Delays by Reward Volume')
#     ax.set_ylim(0, y_max)
#     ax.legend(title='Reward Volume', loc='upper right', ncol=2)


#     # plot for current trial cpoke in latencies
#     cpoke_in_data = timing_data[timing_data['cpoke_in_latency'] > 0.001]

#     fig = plt.figure(layout='constrained', figsize=(8, 10))
#     fig.suptitle('Center Poke In Latencies (Port On to Poke In, Pre Response) - Rat {}'.format(subj_id))
#     gs = GridSpec(4, 1, figure=fig)

#     ax = fig.add_subplot(gs[0])
#     sb.boxplot(data=cpoke_in_data, x='rates', y='cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Block Rates by Choice Type')
#     ax.set_ylim(0, y_max)
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[1])
#     sb.boxplot(data=cpoke_in_data, x='delays', y='cpoke_in_latency', hue='type', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Slow Choice Delays by Choice Type')
#     ax.set_ylim(0, y_max)
#     ax.legend(ncol=3, title='Choice Type', loc='upper right')

#     ax = fig.add_subplot(gs[2])
#     sb.boxplot(data=cpoke_in_data, x='rates', y='cpoke_in_latency', hue='choice_delay', ax=ax) # gap=0.1,
#     ax.set_xlabel('Block Reward Rates (μL/s, Fast/Slow)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Block Rates by Choice Delay')
#     ax.set_ylim(0, y_max)
#     ax.legend(title='Choice Delay', loc='upper right', ncol=2)

#     ax = fig.add_subplot(gs[3])
#     sb.boxplot(data=cpoke_in_data, x='delays', y='cpoke_in_latency', hue='reward', ax=ax) # gap=0.1,
#     ax.set_xlabel('Slow Reward Delay (s)')
#     ax.set_ylabel('Cpoke In Latency (s)')
#     ax.set_title('Slow Choice Delays by Reward Volume')
#     ax.set_ylim(0, y_max)
#     ax.legend(title='Reward Volume', loc='upper right', ncol=2)
