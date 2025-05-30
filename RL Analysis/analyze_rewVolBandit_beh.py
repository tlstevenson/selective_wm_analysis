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
import bandit_beh_helpers as bbh
import vol_bandit_beh_helpers as vbbh
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
task_name = 'rewVolBandit'
subject_info = db_access.get_active_subj_stage(protocol='ClassicRLTasks', stage_name=task_name)
subj_ids = subject_info['subjid']

#sess_ids = db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 2)

# optionally limit sessions based on subject ids
#subj_ids = [401, 218, 217, 216] #[404, 275, 217] # #[179, 188, 191, 207, 182]
# sess_ids = {k: v for k, v in sess_ids.items() if k in subj_ids}

n_back = 6

#sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2, subj_ids=subj_ids)
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_name=task_name)
#sess_ids = {subj: sess[2:] for subj, sess in sess_ids.items()}
sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# get trial information
reload = False
loc_db = db.LocalDB_BasicRLTasks(task_name)
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

# # make sure RT column is filled in (it wasn't initially)
# all_sess['RT'] = all_sess['response_time'] - all_sess['response_cue_time']
# all_sess['cpoke_out_latency'] = all_sess['cpoke_out_time'] - all_sess['response_cue_time']

# # update reward rates

# bah.calc_trial_hist(all_sess, 5)
# bbh.make_trial_hist_labels(all_sess, 3)
# bbh.make_rew_hist_labels(all_sess, 3)

all_sess_resp = all_sess[all_sess['choice'] != 'none']

# %%

ind_subj = False
meta_subj = True

# %% TRIAL COUNTS

# aggregate count tables into dictionary
count_columns = ['high_side', 'epoch_label']
column_titles = ['High Side', 'Epoch Type (Stochasticity/Volatility)', ]
count_dict = bah.get_count_dict(all_sess_resp, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(all_sess_resp, 'subjid', count_columns, normalize=True)

# plot bar charts of trial distribution

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
for i, (col_name, title) in enumerate(zip(count_columns, column_titles)):
    bah.plot_counts(count_dict[col_name], axs[i], title, '# Trials', 'h')

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
for i, (col_name, title) in enumerate(zip(count_columns, column_titles)):
    bah.plot_counts(count_dict_pct[col_name], axs[i], title, '% Trials', 'v')

# %% Count Block Lengths

vbbh.count_block_lengths(all_sess_resp,  ind_subj=ind_subj, meta_subj=meta_subj)

# %% Analyze response metrics

plot_simple_summary = False

vbbh.analyze_choice_behavior(all_sess, plot_simple_summary=plot_simple_summary, meta_subj=meta_subj, ind_subj=ind_subj)
vbbh.analyze_trial_choice_behavior(all_sess, plot_simple_summary=plot_simple_summary, meta_subj=meta_subj, ind_subj=ind_subj)

# %% Logistic regression of choice by past choices and trial outcomes
separate_block_rates = True

n_back = 5

# whether to model the interaction as win-stay/lose switch or not
include_winstay = False
# whether to include reward as a predictor on its own
include_reward = False
# whether to have separate interaction terms for rewarded trials and unrewarded trials
separate_unreward = False
# whether to include all possible interaction combinations
include_full_interaction = False

plot_cis = True

bbh.logit_regress_side_choice(all_sess, n_back=n_back, separate_block_rates=separate_block_rates, include_winstay=include_winstay, include_reward=include_reward, 
                              separate_unreward=separate_unreward, include_full_interaction=include_full_interaction, plot_cis=plot_cis, ind_subj=ind_subj, meta_subj=meta_subj)
        
# %% Logistic regression of stay/switch choice by past choices and trial outcomes

separate_block_rates = True

n_back = 5

# whether to fit stays or switches
fit_switches = False
# whether to have reward predictors be 1/0 or +1/-1
include_unreward = False
# whether to include choice (same/diff of prior choice) as a predictor on its own
include_choice = True
# whether to model choice as 1/0 or +1/-1 for same/diff
include_diff_choice = True
# whether to include an interaction term of choice x reward
include_interaction = True
# whether to include all possible interaction combinations. Supersedes above options
include_full_interaction = True

plot_cis = True

bbh.logit_regress_stay_choice(all_sess, n_back=n_back, separate_block_rates=separate_block_rates, fit_switches=fit_switches, include_unreward=include_unreward, 
                              include_choice=include_choice, include_diff_choice=include_diff_choice, include_interaction=include_interaction, include_full_interaction=include_full_interaction, 
                              plot_cis=plot_cis, ind_subj=ind_subj, meta_subj=meta_subj)


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

# time_col = 'cpoke_in_time'
# plot_suffix = ' - Cpoke In Time'

time_col = 'cpoke_out_time'
plot_suffix = ' - Cpoke Out Time'

min_time = np.floor(all_sess[time_col].min())

# create uneven bins tiling the cpoke in times so bins have roughly the same amount of data points
bin_edges = min_time + np.concatenate((np.arange(0, 10, 2), [10,15,20,30,60,np.inf]))

bbh.analyze_choice_time_effects(all_sess, time_col, bin_edges, ind_subj=ind_subj, meta_subj=meta_subj, plot_suffix=plot_suffix)
