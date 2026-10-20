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
import fp_analysis_helpers as fpah
import bandit_beh_helpers as bbh
import numpy as np
import plot_style
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sb
import copy
import sklearn.linear_model as lm
from sklearn.metrics import r2_score
import statsmodels.api as sm
import re

# %% LOAD DATA

protocol = 'BanditTasks' #'ClassicRLTasks'
stage_name = 'twoArmBandit'
n_back = 20
fp_sess_only = False
active_subjects_only = True
reload = False
date_start = '2026-09-06'
date_end = '2026-09-23'
exclude_subjs = []

if active_subjects_only:
    subj_stage_info = db_access.get_active_subj_stage(protocol=protocol, stage_name=stage_name)
else:
    subj_stage_info = db_access.get_protocol_subject_info(protocol=protocol, stage_name=stage_name)

subj_ids = subj_stage_info['subjid']
# subj_ids = [198, 199, 274, 400, 402, 237, 238, 424, 483, 234, 235, 419, 421, 422]

subj_ids = [s for s in subj_ids if not s in exclude_subjs]

subj_info = db_access.get_subject_info(subj_ids)
# rename genotypes
name_mapping = {'FMR1-M2 Hemi': 'FMR1 Hemi', 'FMR1-M2 WT': 'FMR1 WT'}
subj_info['genotype'] = subj_info['genotype'].apply(lambda x: name_mapping.get(x, x))
genotype_subj_ids = subj_info.groupby('genotype')['subjid'].agg(list).to_dict()
genotypes = list(genotype_subj_ids.keys())

subj_labels = {}
for genotype, subjs in genotype_subj_ids.items():
    for subj in subjs:
        if genotype is None:
            subj_labels[subj] = str(subj)
        else:
            subj_labels[subj] = '{} ({})'.format(subj, genotype)

# get session ids
if fp_sess_only:
    sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids, protocol=protocol, stage_name=stage_name)
    sess_ids = {subjid: [sessid for sessid in sess_ids[subjid] if not sessid in fpah.__sess_ignore] for subjid in sess_ids.keys()}
else:
    sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_name=stage_name, protocol=protocol, date_start=date_start, date_end=date_end)
    sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# get trial information
loc_db = db.LocalDB_BasicRLTasks(stage_name)
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

# make sure RT column is filled in (it wasn't initially)
all_sess['RT'] = all_sess['response_time'] - all_sess['response_cue_time']
all_sess['cpoke_out_latency'] = all_sess['cpoke_out_time'] - all_sess['response_cue_time']

# make trial history labels
rew_rate_n_back = 3

bah.calc_trial_hist(all_sess, rew_rate_n_back)
bbh.make_trial_hist_labels(all_sess, rew_rate_n_back)
bbh.make_rew_hist_labels(all_sess, rew_rate_n_back)
all_sess['n_rew_hist'] = all_sess['rew_hist'].apply(lambda x: np.nansum(x))

all_sess_resp = all_sess[all_sess['choice'] != 'none']

block_rate_sess_ids = all_sess.groupby('block_prob')['sessid'].agg('unique').to_dict()
block_rate_sess_ids['All'] = all_sess['sessid'].unique()
block_rates = list(block_rate_sess_ids.keys())

bah.get_bandit_trial_outcome(all_sess)

# %% Calculate licking durations

all_trial_start_ts = bah.get_beh_trial_start_ts(all_sess)

bah.calc_lick_duration(all_sess, all_trial_start_ts)

# %% Analyze licking durations

plot_ind_subjects = False

plot_cols = [('lick_dur', 'Lick Duration', 'Lick Duration (s)')]
             # ('next_action_cpoke_in_latency', 'Next Action Cpoke Latency', 'Cpoke Latency (s)'),
             # ('next_cpoke_in_latency', 'Next Cport On Cpoke Latency', 'Cpoke Latency (s)'),
             # ('cpoke_out_to_response', 'Cpoke Out to Response', 'Response Latency (s)'),
             # ('RT', 'Cue to Response', 'Response Latency (s)')]

rew_hist_vals = np.arange(rew_rate_n_back+1)
hist_col = 'n_rew_hist'

for (plot_col, col_title, col_ylabel) in plot_cols:

    # all subjects by choice
    plot_outcomes = np.unique(all_sess['outcome_type'])

    fig, axs = plt.subplots(len(plot_outcomes), len(block_rates), layout='constrained',
                            figsize=(len(block_rates)*3,len(plot_outcomes)*4), sharey='row')
    fig.suptitle('Bandit: {} vs Reward History, All Subjects'.format(col_title))

    for j, br in enumerate(block_rates):
        br_sess_data = all_sess[all_sess['sessid'].isin(block_rate_sess_ids[br])]

        all_summary = (br_sess_data
                   .dropna(subset=[hist_col, plot_col])
                   .groupby([hist_col, 'outcome_type'])[plot_col]
                   .agg(['mean', 'sem'])
                   .reset_index())

        stay_trials = br_sess_data[br_sess_data['stay']]
        stay_summary = (stay_trials
                   .dropna(subset=[hist_col, plot_col])
                   .groupby([hist_col, 'outcome_type'])[plot_col]
                   .agg(['mean', 'sem'])
                   .reset_index())

        switch_trials = br_sess_data[br_sess_data['switch']]
        switch_summary = (switch_trials
                   .dropna(subset=[hist_col, plot_col])
                   .groupby([hist_col, 'outcome_type'])[plot_col]
                   .agg(['mean', 'sem'])
                   .reset_index())

        for i, outcome in enumerate(plot_outcomes):
            ax = axs[i,j]
            all_outcome_summary = all_summary[all_summary['outcome_type'] == outcome]
            plot_utils.plot_shaded_error(all_outcome_summary[hist_col], all_outcome_summary['mean'], y_err=all_outcome_summary['sem'], ax=ax, label='All Choices')
            stay_outcome_summary = stay_summary[stay_summary['outcome_type'] == outcome]
            plot_utils.plot_shaded_error(stay_outcome_summary[hist_col], stay_outcome_summary['mean'], y_err=stay_outcome_summary['sem'], ax=ax, label='Stays')
            switch_outcome_summary = switch_summary[switch_summary['outcome_type'] == outcome]
            plot_utils.plot_shaded_error(switch_outcome_summary[hist_col], switch_outcome_summary['mean'], y_err=switch_outcome_summary['sem'], ax=ax, label='Switches')

            ax.set_xlabel('# Rewards in Last {} Trials'.format(rew_rate_n_back))
            ax.set_ylabel(col_ylabel)
            ax.set_title('{}, {} Block'.format(outcome.title(), br))
            ax.legend()
            ax.set_xticks(rew_hist_vals)

    plt.show()

    # all subjects by genotype
    if len(genotypes) > 1:

        fig, axs = plt.subplots(len(plot_outcomes), len(block_rates), layout='constrained',
                                figsize=(len(block_rates)*3,len(plot_outcomes)*4), sharey='row')
        fig.suptitle('Bandit: {} vs Reward History, All Subjects'.format(col_title))

        for j, br in enumerate(block_rates):
            br_sess_data = all_sess[all_sess['sessid'].isin(block_rate_sess_ids[br])]

            genotype_data = {}

            for group, group_subjs in genotype_subj_ids.items():
                group_data = br_sess_data[br_sess_data['subjid'].isin(group_subjs)]

                genotype_data[group] = (group_data
                                    .dropna(subset=[hist_col, plot_col])
                                    .groupby([hist_col, 'outcome_type'])[plot_col]
                                    .agg(['mean', 'sem'])
                                    .reset_index())

            for i, outcome in enumerate(plot_outcomes):
                ax = axs[i,j]

                for group in genotypes:
                    group_outcome_summary = genotype_data[group][genotype_data[group]['outcome_type'] == outcome]
                    plot_utils.plot_shaded_error(group_outcome_summary[hist_col], group_outcome_summary['mean'],
                                                 y_err=group_outcome_summary['sem'], ax=ax, label=group)

                ax.set_xlabel('# Rewards in Last {} Trials'.format(rew_rate_n_back))
                ax.set_ylabel(col_ylabel)
                ax.tick_params(labelbottom=True, labelleft=True)
                ax.set_title('{}, {} Block'.format(outcome.title(), br))
                ax.legend()
                ax.set_xticks(rew_hist_vals)

        plt.show()

    # individual subjects
    if plot_ind_subjects:
        for outcome in plot_outcomes:
            fig, axs = plt.subplots(len(subj_ids), len(block_rates), layout='constrained', figsize=(len(block_rates)*4,len(subj_ids)*4))
            fig.suptitle('Bandit: {} vs Reward History, Individual Subjects, {} Trials'.format(col_title, outcome.title()))

            for j, br in enumerate(block_rates):
                br_sess_data = all_sess[all_sess['sessid'].isin(block_rate_sess_ids[br])]

                all_summary = (br_sess_data
                           .dropna(subset=[hist_col, plot_col])
                           .groupby(['subjid', hist_col, 'outcome_type'])[plot_col]
                           .agg(['mean', 'sem'])
                           .reset_index())

                stay_trials = br_sess_data[br_sess_data['stay']]
                stay_summary = (stay_trials
                           .dropna(subset=[hist_col, plot_col])
                           .groupby(['subjid', hist_col, 'outcome_type'])[plot_col]
                           .agg(['mean', 'sem'])
                           .reset_index())

                switch_trials = br_sess_data[br_sess_data['switch']]
                switch_summary = (switch_trials
                           .dropna(subset=[hist_col, plot_col])
                           .groupby(['subjid', hist_col, 'outcome_type'])[plot_col]
                           .agg(['mean', 'sem'])
                           .reset_index())

                for i, subj in enumerate(subj_ids):
                    ax = axs[i,j]
                    all_outcome_summary = all_summary[(all_summary['outcome_type'] == outcome) & (all_summary['subjid'] == subj)]
                    plot_utils.plot_shaded_error(all_outcome_summary[hist_col], all_outcome_summary['mean'], y_err=all_outcome_summary['sem'], ax=ax, label='All Choices')
                    stay_outcome_summary = stay_summary[(stay_summary['outcome_type'] == outcome) & (stay_summary['subjid'] == subj)]
                    plot_utils.plot_shaded_error(stay_outcome_summary[hist_col], stay_outcome_summary['mean'], y_err=stay_outcome_summary['sem'], ax=ax, label='Stays')
                    switch_outcome_summary = switch_summary[(switch_summary['outcome_type'] == outcome) & (switch_summary['subjid'] == subj)]
                    plot_utils.plot_shaded_error(switch_outcome_summary[hist_col], switch_outcome_summary['mean'], y_err=switch_outcome_summary['sem'], ax=ax, label='Switches')

                    ax.set_xlabel('# Rewards in Last {} Trials'.format(rew_rate_n_back))
                    ax.set_ylabel(col_ylabel)
                    ax.set_title('{}, {} Block'.format(subj_labels[subj], br))
                    ax.legend()
                    ax.set_xticks(rew_hist_vals)

        plt.show()

# %%

ind_subj = False
meta_subj = True

# %% TRIAL COUNTS

# aggregate count tables into dictionary
count_columns = ['side_prob', 'block_prob', 'high_side']
column_titles = ['Side Probability (L/R)', 'Block Probabilities', 'High Side']
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

# %% Plot trial history counts

bbh.analyze_trial_hist_counts(all_sess_resp, 3)

# %% Count Block Lengths

bbh.count_block_lengths(all_sess_resp,  ind_subj=ind_subj, meta_subj=meta_subj, subj_groups=genotype_subj_ids)

# %% Analyze response metrics

plot_simple_summary = True
plot_avg_block = False

figs = bbh.analyze_choice_behavior(all_sess, n_back_hist=3, plot_simple_summary=plot_simple_summary, meta_subj=meta_subj, ind_subj=ind_subj, subj_groups=genotype_subj_ids)
figs = bbh.analyze_trial_choice_behavior(all_sess, plot_simple_summary=plot_simple_summary, meta_subj=meta_subj, ind_subj=ind_subj, plot_avg_block=plot_avg_block, subj_groups=genotype_subj_ids)

# fpah.save_fig(figs['all'][1], r'C:\Users\tanne\OneDrive\Desktop\choose_high', format='pdf')
# fpah.save_fig(figs['all'][1], r'C:\Users\tanne\OneDrive\Desktop\choose_high', format='svg')

# %% Logistic regression of choice by past choices and trial outcomes
separate_block_rates = True

n_back = 5

include_unreward = False
separate_winstay_loseswitch = True
include_full_interaction = False

plot_cis = True

bah.calc_trial_hist(all_sess, n_back)

bbh.logit_regress_side_choice(all_sess, n_back=n_back, separate_block_rates=separate_block_rates, include_unreward=include_unreward,
                              separate_winstay_loseswitch=separate_winstay_loseswitch, include_full_interaction=include_full_interaction,
                              plot_cis=plot_cis, ind_subj=ind_subj, meta_subj=meta_subj, subj_groups=genotype_subj_ids)

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

time_col = 'response_cue_time'
plot_suffix = ' - Response Cue Time'

min_time = max(np.floor(all_sess[time_col].min()), 3)

# create uneven bins tiling the cpoke in times so bins have roughly the same amount of data points
bin_edges = min_time + np.concatenate((np.arange(0, 10, 2), [10,15,20,30,60,np.inf]))

bbh.analyze_choice_time_effects(all_sess, time_col, bin_edges, ind_subj=ind_subj, meta_subj=meta_subj, plot_suffix=plot_suffix)
