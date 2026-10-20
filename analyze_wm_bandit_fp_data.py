# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 22:33:24 2023

@author: tanne
"""

# %% imports

import init
import pandas as pd
import pyutils.utils as utils
from sys_neuro_tools import plot_utils, fp_utils
import plot_style
from hankslab_db import db_access
from hankslab_db import tonecatdelayresp_db as wm_db, basicRLtasks_db as bandit_db
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sb
import copy
import os.path as path
import pickle
import time
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.optimize import curve_fit
from collections import Counter
from sklearn.decomposition import FastICA, PCA
from sklearn.cluster import KMeans
import statsmodels.api as sm
import statsmodels.stats.multitest as smm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator, NullFormatter
import warnings
import sklearn
from collections import Counter

Align = fpah.Alignment

# %% Load behavior data

reload = False
for_debug = False

# used for saving plots
wm_beh_name = 'Single Tone WM'
wm_sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=7)

bandit_beh_name = 'Two-armed Bandit'
bandit_sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2)

tasks = ['wm', 'bandit']
beh_names = {'wm': 'Single Tone WM', 'bandit': 'Probabilistic Bandit'}

# optionally limit sessions based on subject ids
subj_ids = [198, 199, 274, 400, 402, 237, 238, 424, 483, 234, 235, 419, 421, 422] #
wm_sess_ids = {k: [s for s in v if not s in fpah.__sess_ignore] for k, v in wm_sess_ids.items() if k in subj_ids}
bandit_sess_ids = {k: [s for s in v if not s in fpah.__sess_ignore] for k, v in bandit_sess_ids.items() if k in subj_ids}

subj_ids = np.unique(list(wm_sess_ids.keys())+list(bandit_sess_ids.keys()))

wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

wm_sess_data = wm_loc_db.get_behavior_data(utils.flatten(wm_sess_ids), reload=reload)
bandit_sess_data = bandit_loc_db.get_behavior_data(utils.flatten(bandit_sess_ids), reload=reload)

bandit_block_rate_sess_ids = bandit_sess_data.groupby('block_prob')['sessid'].agg('unique').to_dict()
bandit_block_rate_sess_ids['All'] = bandit_sess_data['sessid'].unique()
bandit_block_rates = list(bandit_block_rate_sess_ids.keys())

implant_info = db_access.get_fp_implant_info(subj_ids)

# determine timing information between trials
if not for_debug:
    all_trial_start_ts = db_access.get_fp_trial_start_ts(np.concatenate((utils.flatten(wm_sess_ids), utils.flatten(bandit_sess_ids))))

    bah.calc_trial_creation_latency(wm_sess_data, all_trial_start_ts)
    bah.calc_trial_creation_latency(bandit_sess_data, all_trial_start_ts)

    # define the next trial latency threshold past which to not infer anything, in the absence of pose tracking info
    latency_thresh = 0.3

    bah.fill_missing_cpoke_times(wm_sess_data, all_trial_start_ts, latency_thresh=latency_thresh)
    bah.fill_missing_cpoke_times(bandit_sess_data, all_trial_start_ts, latency_thresh=latency_thresh)

    bah.get_bail_response_info(wm_sess_data, all_trial_start_ts, latency_thresh=latency_thresh)

    # recalculate stays and switches based on bail responses
    bah.calc_wm_stay_switch(wm_sess_data)

    # calculate lick duration
    bah.calc_lick_duration(wm_sess_data, all_trial_start_ts)
    bah.calc_lick_duration(bandit_sess_data, all_trial_start_ts)

    # calculate action cpoke in latency
    bah.calc_action_cpoke_latency(wm_sess_data, all_trial_start_ts)
    bah.calc_action_cpoke_latency(bandit_sess_data, all_trial_start_ts)

    # calculate licking duration bins by outcome
    n_bins = 4
    rew_sel = wm_sess_data['rewarded'] == True

    quantiles = np.nanquantile(wm_sess_data.loc[rew_sel,'lick_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
    dur_bins = np.unique(np.r_[np.floor(np.min(wm_sess_data.loc[rew_sel,'lick_dur'])), quantiles, np.ceil(np.max(wm_sess_data.loc[rew_sel,'lick_dur']))])
    dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

    wm_sess_data['rew_lick_dur_bin'] = pd.cut(wm_sess_data['lick_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

    quantiles = np.nanquantile(wm_sess_data.loc[~rew_sel,'lick_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
    dur_bins = np.unique(np.r_[np.floor(np.min(wm_sess_data.loc[~rew_sel,'lick_dur'])), quantiles, np.ceil(np.max(wm_sess_data.loc[~rew_sel,'lick_dur']))])
    dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

    wm_sess_data['unrew_lick_dur_bin'] = pd.cut(wm_sess_data['lick_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

    rew_sel = bandit_sess_data['rewarded'] == True

    quantiles = np.nanquantile(bandit_sess_data.loc[rew_sel,'lick_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
    dur_bins = np.unique(np.r_[np.floor(np.min(bandit_sess_data.loc[rew_sel,'lick_dur'])), quantiles, np.ceil(np.max(bandit_sess_data.loc[rew_sel,'lick_dur']))])
    dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

    bandit_sess_data['rew_lick_dur_bin'] = pd.cut(bandit_sess_data['lick_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

    quantiles = np.nanquantile(bandit_sess_data.loc[~rew_sel,'lick_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
    dur_bins = np.unique(np.r_[np.floor(np.min(bandit_sess_data.loc[~rew_sel,'lick_dur'])), quantiles, np.ceil(np.max(bandit_sess_data.loc[~rew_sel,'lick_dur']))])
    dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

    bandit_sess_data['unrew_lick_dur_bin'] = pd.cut(bandit_sess_data['lick_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

wm_sess_data['cpoke_out_latency'] = wm_sess_data['cpoke_out_time'] - wm_sess_data['response_cue_time']
bandit_sess_data['cpoke_out_latency'] = bandit_sess_data['cpoke_out_time'] - bandit_sess_data['response_cue_time']

wm_sess_data['cpoke_out_to_response'] = wm_sess_data['response_time'] - wm_sess_data['cpoke_out_time']
bandit_sess_data['cpoke_out_to_response'] = bandit_sess_data['response_time'] - bandit_sess_data['cpoke_out_time']

wm_sess_data['cpoke_dur'] = wm_sess_data['cpoke_out_time'] - wm_sess_data['cpoke_in_time']
bandit_sess_data['cpoke_dur'] = bandit_sess_data['cpoke_out_time'] - bandit_sess_data['cpoke_in_time']

# calculate center poke duration bins

# programmed trial duration
# bin_size = 1
# dur_bin_max = np.ceil(np.max(wm_sess_data['stim_dur'])/bin_size)
# dur_bin_min = np.floor(np.min(wm_sess_data['stim_dur'])/bin_size)
# dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
# dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

n_bins = 4
quantiles = np.nanquantile(wm_sess_data['stim_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
dur_bins = np.unique(np.r_[np.floor(np.min(wm_sess_data['stim_dur'])), quantiles, np.ceil(np.max(wm_sess_data['stim_dur']))])
dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

wm_sess_data['trial_dur_bin'] = pd.cut(wm_sess_data['stim_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

# actual center poke duration
quantiles = np.nanquantile(wm_sess_data['cpoke_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
dur_bins = np.unique(np.r_[np.floor(np.min(wm_sess_data['cpoke_dur'])), quantiles, np.ceil(np.max(wm_sess_data['cpoke_dur']))])
dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

wm_sess_data['cpoke_dur_bin'] = pd.cut(wm_sess_data['cpoke_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

# calculate WM response delay bins
wm_sess_data['resp_delay'] = wm_sess_data['stim_dur'] - wm_sess_data['rel_tone_end_times']

quantiles = np.quantile(wm_sess_data['resp_delay'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
delay_bins = np.unique(np.r_[np.floor(np.min(wm_sess_data['resp_delay'])), quantiles, np.ceil(np.max(wm_sess_data['resp_delay']))])
delay_bin_labels = ['{:.1f}-{:.1f}s'.format(delay_bins[i], delay_bins[i+1]) for i in range(len(delay_bins)-1)]

wm_sess_data['resp_delay_bin'] = pd.cut(wm_sess_data['resp_delay'], bins=delay_bins, labels=delay_bin_labels, include_lowest=True)

# programmed trial duration
# bin_size = 0.5
# dur_bin_max = np.ceil(np.max(bandit_sess_data['trial_length'])/bin_size)
# dur_bin_min = np.floor(np.min(bandit_sess_data['trial_length'])/bin_size)
# dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
# dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

quantiles = np.nanquantile(bandit_sess_data['trial_length'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
dur_bins = np.unique(np.r_[np.floor(np.min(bandit_sess_data['trial_length'])), quantiles, np.ceil(np.max(bandit_sess_data['trial_length']))])
dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

bandit_sess_data['trial_dur_bin'] = pd.cut(bandit_sess_data['trial_length'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

# actual center poke duration
n_bins = 4
quantiles = np.nanquantile(bandit_sess_data['cpoke_dur'], np.linspace(1/n_bins, 1, n_bins-1, endpoint=False))
dur_bins = np.unique(np.r_[np.floor(np.min(bandit_sess_data['cpoke_dur'])), quantiles, np.ceil(np.max(bandit_sess_data['cpoke_dur']))])
dur_bin_labels = ['{:.1f}-{:.1f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

bandit_sess_data['cpoke_dur_bin'] = pd.cut(bandit_sess_data['cpoke_dur'], bins=dur_bins, labels=dur_bin_labels, include_lowest=True)

# calculate reward history
rew_rate_n_back = 3

bah.calc_trial_hist(wm_sess_data, n_back=rew_rate_n_back, exclude_bails=True)
wm_sess_data['n_rew_hist'] = wm_sess_data['rew_hist'].apply(lambda x: np.nansum(x))
bah.calc_trial_hist(wm_sess_data, n_back=rew_rate_n_back, col_suffix='bail')
wm_sess_data['n_rew_hist_bail'] = wm_sess_data['rew_hist_bail'].apply(lambda x: np.nansum(x))

bah.calc_trial_hist(bandit_sess_data, n_back=rew_rate_n_back)
bandit_sess_data['n_rew_hist'] = bandit_sess_data['rew_hist'].apply(lambda x: np.nansum(x))

## TODO: Simplify using pd.cut as above
# get bins output by pandas for indexing
# make sure 0 is included in the first bin, intervals are one-sided
rew_hist_bin_edges = np.arange(-0.5, rew_rate_n_back+1.5, 1)
rew_hist_bins = pd.IntervalIndex.from_breaks(rew_hist_bin_edges)
rew_hist_bin_strs = {b:'{}'.format(i) for i,b in enumerate(rew_hist_bins)}

bah.get_wm_trial_outcome(wm_sess_data)
bah.get_bandit_trial_outcome(bandit_sess_data)


# %% Declare common functions and settings

all_regions = ['PL', 'NAc', 'DMS', 'DLS', 'TS']
region_colors = {'PL': '#BB6ED8', 'NAc': '#47BBBF', 'DMS': '#53C43B', 'DLS': '#E87A2C', 'TS': '#E6B900'}

plot_beh_names = {'wm': 'WM Task', 'bandit': 'Bandit Task', 'all': 'All Tasks'}

source_types = ['raw', 'filt']
source_type_labels = {'raw': 'True Signals', 'filt': 'Deconvolved Signals'}

tilt_t = False
traditional_fit = True
baseline_iso_fit = False
baseline_band_iso_fit = True
band_iso_fit = False

def sort_subj_regions(subj_regions):
    return fpah.sort_subj_regions(subj_regions, all_regions)

def get_subj_region_mapping(subj_regions):
    return {r: [k for k in all_regions if k in r][0] for r in subj_regions}

def get_gen_reg_subj_regions(subj_regions):
    return {r: [k for k in subj_regions if r in k] for r in all_regions}

def get_subj_regions(subj_id):
    subj_regions = list(implant_info[subj_id].keys())

    if subj_id in fpah.__region_ignore:
        subj_regions = [r for r in subj_regions if r not in fpah.__region_ignore[subj_id]]

    return sort_subj_regions(subj_regions)

def get_region_pairs(subj_regions):
    return [(region_i, region_j) for i, region_i in enumerate(subj_regions) for region_j in subj_regions[i+1:]]

# %% Session and trial counts

wm_sess_trial_counts_subj = wm_sess_data.groupby('subjid')['sessid'].agg(['nunique', 'count'])
wm_sess_trial_counts_total = wm_sess_trial_counts_subj.agg(['sum', 'mean', 'std'])

bandit_trial_counts_subj = bandit_sess_data.groupby(['subjid', 'block_prob'])['sessid'].agg(['nunique', 'count'])
bandit_trial_counts_total = bandit_trial_counts_subj.groupby('block_prob').agg(['sum', 'mean', 'std'])

# %% Look at trial creation latencies

bin_step = 0.1

plt.figure()
plt.hist(wm_sess_data['next_trial_start_latency'], bins=np.arange(max(wm_sess_data['next_trial_start_latency'])+bin_step, step=bin_step))
plt.title('WM Trial Creation Latency')

plt.figure()
plt.hist(bandit_sess_data['next_trial_start_latency'], bins=np.arange(max(bandit_sess_data['next_trial_start_latency'])+bin_step, step=bin_step))
plt.title('Bandit Trial Creation Latency')

# Note:
# The WM protocol had a bug when adding the bail tone where the stimulus would keep playing,
# so the next trial latencies were long after many bails because the next trail was waiting for the stimulus to end

# wm_bails = wm_sess_data[~np.isnan(wm_sess_data['bail_tone_time'])]
# plt.figure()
# plt.hist(wm_bails['next_trial_start_latency'], bins=np.arange(max(wm_bails['next_trial_start_latency'])+bin_step, step=bin_step))
# plt.title('WM Trial Creation Latency')

# wm_non_bails = wm_sess_data[np.isnan(wm_sess_data['bail_tone_time'])]
# plt.figure()
# plt.hist(wm_non_bails['next_trial_start_latency'], bins=np.arange(max(wm_non_bails['next_trial_start_latency'])+bin_step, step=bin_step))
# plt.title('WM Trial Creation Latency')

# %% Plot various timing psychometrics by choice/reward history

plot_ind_subjects = False

# split wm bails into different classes by the side of their choice
wm_sess_data['outcome_type_copy'] = wm_sess_data['outcome_type']
bail_sel = wm_sess_data['outcome_type'] == 'bail'
correct_choice_sel = wm_sess_data['choice'] == wm_sess_data['correct_port']
incorrect_choice_sel = (wm_sess_data['choice'] != wm_sess_data['correct_port']) & (wm_sess_data['choice'] != 'none')

wm_sess_data.loc[bail_sel & correct_choice_sel, 'outcome_type_copy'] = 'correct bail'
wm_sess_data.loc[bail_sel & incorrect_choice_sel, 'outcome_type_copy'] = 'incorrect bail'

# declare columns to plot vs reward history and their associated titles/labels
plot_cols = [('lick_dur', 'Lick Duration', 'Lick Duration (s)'),
             ('next_action_cpoke_in_latency', 'Next Action Cpoke Latency', 'Cpoke Latency (s)'),
             ('next_cpoke_in_latency', 'Next Cport On Cpoke Latency', 'Cpoke Latency (s)'),
             ('cpoke_out_to_response', 'Cpoke Out to Response', 'Response Latency (s)'),
             ('RT', 'Cue to Response', 'Response Latency (s)')]

wm_outcome_order = ['hit', 'miss', 'bail', 'correct bail', 'incorrect bail']
rew_hist_vals = np.arange(rew_rate_n_back+1)

for (plot_col, col_title, col_ylabel) in plot_cols:

    # WM, all subjects
    hist_cols = [('n_rew_hist', 'w/o Bails'),
                 ('n_rew_hist_bail', 'w/ Bails')]

    for hist_col, label in hist_cols:

        all_summary = (wm_sess_data
                   .dropna(subset=[hist_col, plot_col])
                   .groupby([hist_col, 'outcome_type_copy'])[plot_col]
                   .agg(['mean', 'sem'])
                   .reset_index())

        stay_trials = wm_sess_data[wm_sess_data['stay']]
        stay_summary = (stay_trials
                   .dropna(subset=[hist_col, plot_col])
                   .groupby([hist_col, 'outcome_type_copy'])[plot_col]
                   .agg(['mean', 'sem'])
                   .reset_index())

        switch_trials = wm_sess_data[wm_sess_data['switch']]
        switch_summary = (switch_trials
                   .dropna(subset=[hist_col, plot_col])
                   .groupby([hist_col, 'outcome_type_copy'])[plot_col]
                   .agg(['mean', 'sem'])
                   .reset_index())

        plot_outcomes = np.unique(all_summary['outcome_type_copy'])
        plot_outcomes = [o for o in wm_outcome_order if o in plot_outcomes]

        fig, axs = plt.subplots(1, len(plot_outcomes), layout='constrained', figsize=(len(plot_outcomes)*3,4))
        fig.suptitle('Working Memory: {} vs Reward History {}, All Subjects'.format(col_title, label))

        for i, outcome in enumerate(plot_outcomes):
            ax = axs[i]
            all_outcome_summary = all_summary[all_summary['outcome_type_copy'] == outcome]
            plot_utils.plot_shaded_error(all_outcome_summary[hist_col], all_outcome_summary['mean'], y_err=all_outcome_summary['sem'], ax=ax, label='All Choices')
            stay_outcome_summary = stay_summary[stay_summary['outcome_type_copy'] == outcome]
            plot_utils.plot_shaded_error(stay_outcome_summary[hist_col], stay_outcome_summary['mean'], y_err=stay_outcome_summary['sem'], ax=ax, label='Stays')
            switch_outcome_summary = switch_summary[switch_summary['outcome_type_copy'] == outcome]
            plot_utils.plot_shaded_error(switch_outcome_summary[hist_col], switch_outcome_summary['mean'], y_err=switch_outcome_summary['sem'], ax=ax, label='Switches')

            ax.set_xlabel('# Rewards in Last {} Trials'.format(rew_rate_n_back))
            ax.set_ylabel(col_ylabel)
            ax.set_title(outcome.title())
            ax.legend()
            ax.set_xticks(rew_hist_vals)

        plt.show()

    # WM, individual subjects
    if plot_ind_subjects:
        for hist_col, label in hist_cols:

            all_summary = (wm_sess_data
                       .dropna(subset=[hist_col, plot_col])
                       .groupby(['subjid', hist_col, 'outcome_type_copy'])[plot_col]
                       .agg(['mean', 'sem'])
                       .reset_index())

            stay_trials = wm_sess_data[wm_sess_data['stay']]
            stay_summary = (stay_trials
                       .dropna(subset=[hist_col, plot_col])
                       .groupby(['subjid', hist_col, 'outcome_type_copy'])[plot_col]
                       .agg(['mean', 'sem'])
                       .reset_index())

            switch_trials = wm_sess_data[wm_sess_data['switch']]
            switch_summary = (switch_trials
                       .dropna(subset=[hist_col, plot_col])
                       .groupby(['subjid', hist_col, 'outcome_type_copy'])[plot_col]
                       .agg(['mean', 'sem'])
                       .reset_index())

            plot_outcomes = np.unique(all_summary['outcome_type_copy'])
            plot_outcomes = [o for o in wm_outcome_order if o in plot_outcomes]

            fig, axs = plt.subplots(len(subj_ids), len(plot_outcomes), layout='constrained', figsize=(len(plot_outcomes)*3,len(subj_ids)*3))
            fig.suptitle('Working Memory: {} vs Reward History {}, Individual Subjects'.format(col_title, label))

            for i, subj in enumerate(subj_ids):
                for j, outcome in enumerate(plot_outcomes):
                    ax = axs[i,j]
                    all_outcome_summary = all_summary[(all_summary['outcome_type_copy'] == outcome) & (all_summary['subjid'] == subj)]
                    plot_utils.plot_shaded_error(all_outcome_summary[hist_col], all_outcome_summary['mean'], y_err=all_outcome_summary['sem'], ax=ax, label='All Choices')
                    stay_outcome_summary = stay_summary[(stay_summary['outcome_type_copy'] == outcome) & (stay_summary['subjid'] == subj)]
                    plot_utils.plot_shaded_error(stay_outcome_summary[hist_col], stay_outcome_summary['mean'], y_err=stay_outcome_summary['sem'], ax=ax, label='Stays')
                    switch_outcome_summary = switch_summary[(switch_summary['outcome_type_copy'] == outcome) & (switch_summary['subjid'] == subj)]
                    plot_utils.plot_shaded_error(switch_outcome_summary[hist_col], switch_outcome_summary['mean'], y_err=switch_outcome_summary['sem'], ax=ax, label='Switches')

                    ax.set_xlabel('# Rewards in Last {} Trials'.format(rew_rate_n_back))
                    ax.set_ylabel(col_ylabel)
                    ax.set_title('{}, {}'.format(subj, outcome.title()))
                    ax.legend()
                    ax.set_xticks(rew_hist_vals)

            plt.show()


    # bandit, all subjects
    hist_col = 'n_rew_hist'
    plot_outcomes = np.unique(bandit_sess_data['outcome_type'])

    fig, axs = plt.subplots(len(plot_outcomes), len(bandit_block_rates), layout='constrained',
                            figsize=(len(bandit_block_rates)*3,len(plot_outcomes)*4), sharey='row')
    fig.suptitle('Bandit: {} vs Reward History, All Subjects'.format(col_title))

    for j, br in enumerate(bandit_block_rates):
        br_sess_data = bandit_sess_data[bandit_sess_data['sessid'].isin(bandit_block_rate_sess_ids[br])]

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

    # bandit, individual subjects
    if plot_ind_subjects:
        for outcome in plot_outcomes:
            fig, axs = plt.subplots(len(subj_ids), len(bandit_block_rates), layout='constrained', figsize=(len(bandit_block_rates)*3,len(subj_ids)*3))
            fig.suptitle('Bandit: {} vs Reward History, Individual Subjects, {} Trials'.format(col_title, outcome.title()))

            for j, br in enumerate(bandit_block_rates):
                br_sess_data = bandit_sess_data[bandit_sess_data['sessid'].isin(bandit_block_rate_sess_ids[br])]

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
                    ax.set_title('{}, {} Block'.format(subj, br))
                    ax.legend()
                    ax.set_xticks(rew_hist_vals)

        plt.show()

# %% Preprocess photometry data and save preprocessing plots

reload = False
reprocess = False
save_processed_data = True
save_process_plots = True
show_process_plots = False

# Build signal matrices aligned to alignment points
for task in tasks:
    if task == 'wm':
        sess_ids = wm_sess_ids
        loc_db = wm_loc_db
    else:
        sess_ids = bandit_sess_ids
        loc_db = bandit_loc_db

    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue

        for sess_id in sess_ids[subj_id]:
            if sess_id in fpah.__sess_ignore:
                continue

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t, reload=reload, reprocess=reprocess)

            if save_process_plots or show_process_plots:
                fp_data = fp_data[subj_id][sess_id]

                fig = fpah.view_processed_signals(fp_data['processed_signals'], fp_data['time'], plot_traditional=traditional_fit,
                                                  plot_baseline=baseline_iso_fit, plot_fband=band_iso_fit, plot_baseline_fband=baseline_band_iso_fit,
                                                  title='Full Signals - Subject {}, Session {}'.format(subj_id, sess_id),
                                                  region_order=all_regions)
                if save_process_plots:
                    fpah.save_fig(fig, fpah.get_figure_save_path(beh_names[task], subj_id, 'sess_{}'.format(sess_id)))

                if show_process_plots:
                    plt.show()
                else:
                    plt.close(fig)

# %% Stack event-aligned photometry data

recalculate = False
recalculate_norm_tasks = [] # 'wm', 'bandit'

dec = 1

# ignored_signals = {'PL': [],
#                    'DMS': [],
#                    'DLS': [],
#                    'TS': []}

# ignored_subjects = []

reprocess_sess_ids = []
reprocess_subj_ids = []

signal_types = ['z_dff_iso_baseline_fband'] # 'dff_iso_baseline',

bandit_alignments = [Align.cport_on, Align.cpoke_in, Align.cue, Align.cpoke_out, Align.resp, Align.reward, Align.cue_reward, Align.cport_on_cpoke_in]
wm_alignments = bandit_alignments.copy()
wm_alignments.append(Align.tone)

default_xlims = {'PL': [-3,5], 'NAc': [-2,4], 'DMS': [-1,2], 'DLS': [-1,2], 'TS': [-1,2]}
xlims = {align: default_xlims.copy() for align in wm_alignments}
xlims[Align.reward]['PL'] = [-3,15]

# minimum interval between task events
min_dt = 0.1
# time before and after events
pre_cue_dt = 0.5
pre_cport_on_dt = 0.5
post_reward_dt = 1.5
post_cpoke_in_dt = 0.5
max_cport_on_cpoke_in = 10

filename = 'wm_bandit_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        aligned_signals = saved_data['aligned_signals']
        aligned_time = saved_data['aligned_time']
        aligned_metadata = saved_data['metadata']

        recalculate_regions = aligned_metadata['xlims'] != xlims

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    aligned_signals = {task: {subjid: {} for subjid in subj_ids}
                       for task in tasks}
    aligned_time = {task: {'events': {}, 't': {}} for task in tasks}

# get dt
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']*dec

# Build signal matrices aligned to alignment points
for task in tasks:
    if task == 'wm':
        sess_ids = wm_sess_ids
        loc_db = wm_loc_db
        sess_data = wm_sess_data
        alignments = wm_alignments
    else:
        sess_ids = bandit_sess_ids
        loc_db = bandit_loc_db
        sess_data = bandit_sess_data
        alignments = bandit_alignments

    # median time intervals across all subjects to determine size of time invariant signals
    cue_cpoke_out_dt = sess_data['cpoke_out_time'] - sess_data['response_cue_time']
    cpoke_out_resp_dt = sess_data['response_time'] - sess_data['cpoke_out_time']
    resp_reward_dt = sess_data['reward_time'] - sess_data['response_time']

    cport_on_cpoke_dt = sess_data['cpoke_in_time'] - sess_data['cport_on_time']

    med_poke_out_dt = np.median(cue_cpoke_out_dt[cue_cpoke_out_dt > 0])
    med_resp_dt = np.median(cpoke_out_resp_dt[cpoke_out_resp_dt > 0])
    med_rew_dt = np.median(resp_reward_dt[resp_reward_dt > 0])
    med_cpoke_dt = np.median(cport_on_cpoke_dt[(cport_on_cpoke_dt > 0) & (cport_on_cpoke_dt < max_cport_on_cpoke_in)])

    min_norm_bins = np.ceil(min_dt/dt)
    n_pre_cue_bins = max(np.ceil(pre_cue_dt/dt), min_norm_bins)
    n_cue_poke_out_bins = max(np.ceil(med_poke_out_dt/dt), min_norm_bins)
    n_poke_out_resp_bins = max(np.ceil(med_resp_dt/dt), min_norm_bins)
    n_resp_rew_bins = max(np.ceil(med_rew_dt/dt), min_norm_bins)
    n_post_rew_bins = max(np.ceil(post_reward_dt/dt), min_norm_bins)

    n_pre_cport_on_bins = max(np.ceil(pre_cport_on_dt/dt), min_norm_bins)
    n_cport_poke_in_bins = max(np.ceil(med_cpoke_dt/dt), min_norm_bins)
    n_post_poke_in_bins = max(np.ceil(post_cpoke_in_dt/dt), min_norm_bins)

    # persist median bin numbers for plotting
    aligned_time[task]['events'][Align.cport_on_cpoke_in] = {'cport_on': n_pre_cport_on_bins,
                                                             'cpoke_in': n_pre_cport_on_bins+n_cport_poke_in_bins}

    aligned_time[task]['events'][Align.cue_reward] = {'cue': n_pre_cue_bins,
                                                      'poke_out': n_pre_cue_bins+n_cue_poke_out_bins,
                                                      'response': n_pre_cue_bins+n_cue_poke_out_bins+n_poke_out_resp_bins,
                                                      'reward': n_pre_cue_bins+n_cue_poke_out_bins+n_poke_out_resp_bins+n_resp_rew_bins}


    aligned_time[task]['t'] = {align: {} for align in alignments}
    for align in alignments:
        for region in default_xlims.keys():
            if align == Align.cport_on_cpoke_in:
                aligned_time[task]['t'][align][region] = np.arange(0, n_pre_cport_on_bins+n_cport_poke_in_bins+n_post_poke_in_bins, 1)
            elif align == Align.cue_reward:
                aligned_time[task]['t'][align][region] = np.arange(0, n_pre_cue_bins+n_cue_poke_out_bins+
                                                                      n_poke_out_resp_bins+n_resp_rew_bins+n_post_rew_bins, 1)
            else:
                aligned_time[task]['t'][align][region] = np.arange(xlims[align][region][0], xlims[align][region][1]+dt, dt)


    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue

        if not subj_id in aligned_signals[task]:
            aligned_signals[task][subj_id] = {}

        for sess_id in sess_ids[subj_id]:
            if sess_id in fpah.__sess_ignore:
                continue

            if (sess_id in aligned_signals[task][subj_id] and not sess_id in reprocess_sess_ids and
                not task in recalculate_norm_tasks and not subj_id in reprocess_subj_ids):

                continue
            else:
                aligned_signals[task][subj_id][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]

            start = time.perf_counter()

            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_start_ts = fp_data['trial_start_ts'][:-1]
            cport_on_ts = trial_start_ts + trial_data['cport_on_time']
            cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
            cue_ts = trial_start_ts + trial_data['response_cue_time']
            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            resp_ts = trial_start_ts + trial_data['response_time']
            reward_ts = trial_start_ts + trial_data['reward_time']

            if task == 'wm':
                tone_start_ts = trial_start_ts + trial_data['abs_tone_start_times']

            for signal_type in signal_types:

                if not signal_type in aligned_signals[task][subj_id][sess_id]:
                    aligned_signals[task][subj_id][sess_id][signal_type] = {}

                for align in alignments:

                    norm_align = align in [Align.cport_on_cpoke_in, Align.cue_reward]

                    if align in aligned_signals[task][subj_id][sess_id][signal_type] and not (task in recalculate_norm_tasks and norm_align):
                        continue
                    else:
                        aligned_signals[task][subj_id][sess_id][signal_type][align] = {}

                    for region in fp_data['processed_signals'].keys():

                        if region in aligned_signals[task][subj_id][sess_id][signal_type][align]:
                            continue

                        signal, ts = fp_utils.decimate(fp_data['processed_signals'][region][signal_type], fp_data['time'], dec)

                        if norm_align:
                            match align:
                                case Align.cport_on_cpoke_in:

                                    poke_in_sel = ((cpoke_in_ts - cport_on_ts) < max_cport_on_cpoke_in).to_numpy()
                                    pre_cport_on = fp_utils.build_time_norm_signal_matrix(signal, ts, cport_on_ts-pre_cport_on_dt, cport_on_ts, n_pre_cport_on_bins, align_sel=poke_in_sel)
                                    cport_on_cpoke_in = fp_utils.build_time_norm_signal_matrix(signal, ts, cport_on_ts, cpoke_in_ts, n_cport_poke_in_bins, align_sel=poke_in_sel)
                                    post_cpoke_in = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_in_ts, cpoke_in_ts+post_cpoke_in_dt, n_post_poke_in_bins, align_sel=poke_in_sel)

                                    mat = np.hstack([pre_cport_on, cport_on_cpoke_in, post_cpoke_in])

                                case Align.cue_reward:
                                    pre_cue = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts-pre_cue_dt, cue_ts, n_pre_cue_bins)
                                    cue_poke_out = fp_utils.build_time_norm_signal_matrix(signal, ts, cue_ts, cpoke_out_ts, n_cue_poke_out_bins)
                                    poke_out_resp = fp_utils.build_time_norm_signal_matrix(signal, ts, cpoke_out_ts, resp_ts, n_poke_out_resp_bins)
                                    resp_rew = fp_utils.build_time_norm_signal_matrix(signal, ts, resp_ts, reward_ts, n_resp_rew_bins)
                                    post_rew = fp_utils.build_time_norm_signal_matrix(signal, ts, reward_ts, reward_ts+post_reward_dt, n_post_rew_bins)

                                    mat = np.hstack([pre_cue, cue_poke_out, poke_out_resp, resp_rew, post_rew])

                            aligned_signals[task][subj_id][sess_id][signal_type][align][region] = mat

                        else:
                            match align:
                                case Align.cport_on:
                                    align_ts = cport_on_ts
                                    mask_lims = None

                                case Align.cpoke_in:
                                    align_ts = cpoke_in_ts
                                    mask_lims = None

                                case Align.tone:
                                    align_ts = tone_start_ts
                                    mask_lims = None

                                case Align.cue:
                                    align_ts = cue_ts
                                    mask_lims = None
                                    # # mask out reward
                                    # reward_ts_mask = reward_ts.to_numpy(copy=True)
                                    # reward_ts_mask[~trial_data['rewarded']] = np.nan
                                    # if np.isnan(reward_ts_mask[-1]):
                                    #     reward_ts_mask[-1] = np.inf
                                    # reward_ts_mask = pd.Series(reward_ts_mask).bfill().to_numpy()

                                    # mask_lims = np.hstack((np.full_like(align_ts, 0)[:, None], reward_ts_mask[:, None]))

                                case Align.cpoke_out:
                                    align_ts = cpoke_out_ts
                                    mask_lims = None

                                case Align.resp:
                                    align_ts = resp_ts
                                    mask_lims = None

                                case Align.reward:
                                    align_ts = reward_ts
                                    mask_lims = None
                                    # # mask out next reward
                                    # next_reward_ts = reward_ts[1:].to_numpy(copy=True)
                                    # next_reward_ts[~trial_data['rewarded'][1:]] = np.nan
                                    # next_reward_ts = pd.Series(np.append(next_reward_ts, np.inf))
                                    # next_reward_ts = next_reward_ts.bfill().to_numpy()

                                    # mask_lims = np.hstack((np.zeros_like(align_ts)[:, None], next_reward_ts[:, None]))

                            reg_key = [k for k in xlims[align].keys() if k in region][0]
                            lims = xlims[align][reg_key]

                            mat, _ = fp_utils.build_signal_matrix(signal, ts, align_ts, -lims[0], lims[1], mask_lims=mask_lims)
                            aligned_signals[task][subj_id][sess_id][signal_type][align][region] = mat

            print('Stacked FP data for subject {} session {} in {:.1f} s'.format(subj_id, sess_id, time.perf_counter()-start))

            with open(save_path, 'wb') as f:
                pickle.dump({'aligned_signals': aligned_signals,
                             'aligned_time': aligned_time,
                             'metadata': {'signal_types': signal_types,
                                          'xlims': xlims}}, f)


# %% Construct selection vectors for various trial groupings

align_trial_selections = {task: {subjid: {} for subjid in subj_ids} for task in tasks}

rel_sides = ['contra', 'ipsi']
abs_sides = ['left', 'right']

for task in tasks:
    if task == 'wm':
        sess_ids = wm_sess_ids
        sess_data = wm_sess_data
        alignments = wm_alignments.copy()

        resp_delays = sess_data['resp_delay_bin'].cat.categories
        tone_types = np.unique(sess_data['relevant_tone_info'])

        # get tone side mapping
        tone_ports = sess_data[['subjid', 'correct_port', 'relevant_tone_info']].drop_duplicates()
        tone_ports = {i: tone_ports[tone_ports['subjid'] == i].set_index('relevant_tone_info')['correct_port'].to_dict() for i in sess_ids.keys()}

    else:
        sess_ids = bandit_sess_ids
        sess_data = bandit_sess_data

        resp_delays = sess_data['trial_dur_bin'].cat.categories
        alignments = bandit_alignments.copy()

    #alignments.append(Align.early_cpoke_in)

    if len(sess_data) == 0:
        continue

    cpoke_durs = sess_data['trial_dur_bin'].cat.categories
    rew_lick_durs = sess_data['rew_lick_dur_bin'].cat.categories
    unrew_lick_durs = sess_data['unrew_lick_dur_bin'].cat.categories

    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue

        subj_regions = get_subj_regions(subj_id)

        for sess_id in sess_ids[subj_id]:
            if not sess_id in align_trial_selections[task][subj_id]:
                align_trial_selections[task][subj_id][sess_id] = {}

            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_started = trial_data['trial_started']

            # create various trial selection criteria
            if task == 'wm':
                bail = (trial_data['bail'] == True).to_numpy()
                response = ~np.isnan(trial_data['response_time']).to_numpy() & ~bail

                tones = trial_data['relevant_tone_info'].to_numpy()
                prev_tones = trial_data['prev_tone_info'].apply(lambda x: x[0] if utils.is_list(x) else x).to_numpy()
                next_tones = trial_data['next_tone_info'].apply(lambda x: x[0] if utils.is_list(x) else x).to_numpy()

                delays = trial_data['resp_delay_bin'].to_numpy()

                prev_trial_same = tones == prev_tones
                prev_trial_diff = (tones != prev_tones) & (prev_tones != None)

                next_trial_same = tones == next_tones
                next_trial_diff = (tones != next_tones) & (next_tones != None)

                tone_heard_sel = ~np.isnan(trial_data['abs_tone_start_times']).to_numpy()
                prev_tone_heard_sel = np.insert(tone_heard_sel[:-1], 0, False)

                rew_hist_bail = pd.cut(trial_data['n_rew_hist_bail'], rew_hist_bins)

            if task == 'bandit':
                response = ~np.isnan(trial_data['response_time']).to_numpy()
                bail = np.full_like(response, False)
                delays = trial_data['trial_dur_bin'].to_numpy()

            reward = (trial_data['rewarded'] == True).to_numpy()
            unreward = (trial_data['rewarded'] == False).to_numpy() & response

            prev_resp = np.insert(response[:-1], 0, False)

            prev_reward = np.insert(reward[:-1], 0, False)
            prev_unreward = np.insert(unreward[:-1], 0, False)

            next_resp = np.append(response[1:], False)

            rew_hist = pd.cut(trial_data['n_rew_hist'], rew_hist_bins).to_numpy()

            choices = trial_data['choice']
            prev_choices = trial_data['prev_choice']

            switch = trial_data['switch'].to_numpy() & prev_resp & response
            stay = trial_data['stay'].to_numpy() & prev_resp & response
            next_switch = trial_data['next_switch'].to_numpy() & response & next_resp
            next_stay = trial_data['next_stay'].to_numpy() & response & next_resp

            prev_bail = np.insert(bail[:-1], 0, False)
            next_bail = np.append(bail[1:], False)

            sess_rew_lick_durs = trial_data['rew_lick_dur_bin'].to_numpy()
            sess_unrew_lick_durs = trial_data['unrew_lick_dur_bin'].to_numpy()

            # ignore cport on trials where they were poked before cport turned on
            cport_on_sel = (trial_data['cpoke_in_latency'] > 0.1).to_numpy()
            early_cpoke_in_sel = (trial_data['cpoke_in_latency'] < 0).to_numpy()
            norm_cpoke_in_sel = (trial_data['cpoke_in_latency'] > 0).to_numpy()
            engage_cpoke_in_sel = (trial_data['cpoke_in_latency'] < max_cport_on_cpoke_in).to_numpy()
            early_cpoke_out_sel = (trial_data['cpoke_out_latency'] < 0).to_numpy()
            norm_cpoke_out_sel = (trial_data['cpoke_out_latency'] > 0).to_numpy()

            for region in subj_regions:
                if not region in align_trial_selections[task][subj_id][sess_id]:
                    align_trial_selections[task][subj_id][sess_id][region] = {}

                region_side = implant_info[subj_id][region]['side']
                rel_choice_side = choices.apply(lambda x: fpah.get_implant_rel_side(x, region_side)).to_numpy().copy()
                rel_prev_choice_side = prev_choices.apply(lambda x: fpah.get_implant_rel_side(x, region_side)).to_numpy().copy()

                # get all trial selection vectors for different trial groupings
                align_trial_selections[task][subj_id][sess_id][region]['all'] = np.full_like(response, True)
                align_trial_selections[task][subj_id][sess_id][region]['trial_started'] = trial_started

                align_trial_selections[task][subj_id][sess_id][region]['prev_resp'] = prev_resp
                align_trial_selections[task][subj_id][sess_id][region]['prev_bail'] = prev_bail

                align_trial_selections[task][subj_id][sess_id][region]['bail'] = bail
                align_trial_selections[task][subj_id][sess_id][region]['response'] = response

                align_trial_selections[task][subj_id][sess_id][region]['early_cpoke_in'] = early_cpoke_in_sel
                align_trial_selections[task][subj_id][sess_id][region]['norm_cpoke_in'] = ~early_cpoke_in_sel

                align_trial_selections[task][subj_id][sess_id][region]['stay'] = stay
                align_trial_selections[task][subj_id][sess_id][region]['switch'] = switch
                align_trial_selections[task][subj_id][sess_id][region]['next_stay'] = next_stay
                align_trial_selections[task][subj_id][sess_id][region]['next_switch'] = next_switch

                align_trial_selections[task][subj_id][sess_id][region]['prev_reward'] = prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['prev_unreward'] = prev_unreward
                # need to have different name for reward so it doesn't get overwritten by Align.reward
                align_trial_selections[task][subj_id][sess_id][region]['rewarded'] = reward
                align_trial_selections[task][subj_id][sess_id][region]['unrewarded'] = unreward

                align_trial_selections[task][subj_id][sess_id][region]['reward_prev_reward'] = reward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['reward_prev_unreward'] = reward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['unreward_prev_reward'] = unreward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['unreward_prev_unreward'] = unreward & prev_unreward

                align_trial_selections[task][subj_id][sess_id][region]['stay_prev_reward'] = stay & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_prev_reward'] = switch & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_prev_unreward'] = stay & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_prev_unreward'] = switch & prev_unreward

                align_trial_selections[task][subj_id][sess_id][region]['stay_reward'] = stay & reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_reward'] = switch & reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_unreward'] = stay & unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_unreward'] = switch & unreward

                align_trial_selections[task][subj_id][sess_id][region]['stay_reward_prev_reward'] = stay & reward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_reward_prev_reward'] = switch & reward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_unreward_prev_reward'] = stay & unreward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['switch_unreward_prev_reward'] = switch & unreward & prev_reward
                align_trial_selections[task][subj_id][sess_id][region]['stay_reward_prev_unreward'] = stay & reward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_reward_prev_unreward'] = switch & reward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['stay_unreward_prev_unreward'] = stay & unreward & prev_unreward
                align_trial_selections[task][subj_id][sess_id][region]['switch_unreward_prev_unreward'] = switch & unreward & prev_unreward

                align_trial_selections[task][subj_id][sess_id][region]['reward_next_stay'] = next_stay & reward
                align_trial_selections[task][subj_id][sess_id][region]['reward_next_switch'] = next_switch & reward
                align_trial_selections[task][subj_id][sess_id][region]['unreward_next_stay'] = next_stay & unreward
                align_trial_selections[task][subj_id][sess_id][region]['unreward_next_switch'] = next_switch & unreward

                for rel_side in rel_sides:
                    side_sel = rel_choice_side == rel_side
                    prev_side_sel = rel_prev_choice_side == rel_side

                    align_trial_selections[task][subj_id][sess_id][region][rel_side] = side_sel
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side] = prev_side_sel & prev_resp

                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_prev_reward'] = prev_side_sel & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_prev_unreward'] = prev_side_sel & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_prev_bail'] = prev_side_sel & prev_bail

                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_bail'] = prev_side_sel & bail
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_response'] = prev_side_sel & response

                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_stay'] = prev_side_sel & stay
                    align_trial_selections[task][subj_id][sess_id][region]['prev_'+rel_side+'_switch'] = prev_side_sel & switch

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_prev_reward'] = side_sel & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_prev_unreward'] = side_sel & prev_unreward

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward'] = side_sel & reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward'] = side_sel & unreward

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay'] = side_sel & stay
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch'] = side_sel & switch
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_next_stay'] = side_sel & next_stay
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_next_switch'] = side_sel & next_switch

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_prev_reward'] = side_sel & stay & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_prev_reward'] = side_sel & switch & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_prev_unreward'] = side_sel & stay & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_prev_unreward'] = side_sel & switch & prev_unreward

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_prev_reward'] = side_sel & reward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_prev_unreward'] = side_sel & reward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_prev_reward'] = side_sel & unreward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_prev_unreward'] = side_sel & unreward & prev_unreward

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_reward'] = side_sel & stay & reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_reward'] = side_sel & switch & reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_unreward'] = side_sel & stay & unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_unreward'] = side_sel & switch & unreward

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_reward_prev_reward'] = side_sel & stay & reward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_reward_prev_reward'] = side_sel & switch & reward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_unreward_prev_reward'] = side_sel & stay & unreward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_unreward_prev_reward'] = side_sel & switch & unreward & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_reward_prev_unreward'] = side_sel & stay & reward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_reward_prev_unreward'] = side_sel & switch & reward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_stay_unreward_prev_unreward'] = side_sel & stay & unreward & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_switch_unreward_prev_unreward'] = side_sel & switch & unreward & prev_unreward

                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_next_stay'] = side_sel & next_stay & reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_reward_next_switch'] = side_sel & next_switch & reward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_next_stay'] = side_sel & next_stay & unreward
                    align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unreward_next_switch'] = side_sel & next_switch & unreward

                for rew_bin in rew_hist_bins:
                    rew_sel = rew_hist == rew_bin
                    bin_str = rew_hist_bin_strs[rew_bin]

                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str] = rew_sel
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_resp'] = rew_sel & response
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_prev_resp'] = rew_sel & prev_resp
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_prev_bail'] = rew_sel & prev_bail
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_reward'] = rew_sel & reward
                    align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_unreward'] = rew_sel & unreward

                    for rel_side in rel_sides:
                        side_sel = rel_choice_side == rel_side
                        prev_side_sel = rel_prev_choice_side == rel_side

                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_'+rel_side] = rew_sel & side_sel
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_prev_'+rel_side] = rew_sel & prev_side_sel
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_'+rel_side+'_reward'] = rew_sel & side_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_'+bin_str+'_'+rel_side+'_unreward'] = rew_sel & side_sel & unreward

                for resp_delay in resp_delays:
                    delay_sel = delays == resp_delay
                    align_trial_selections[task][subj_id][sess_id][region]['delay_'+resp_delay] = delay_sel
                    align_trial_selections[task][subj_id][sess_id][region]['delay_'+resp_delay+'_reward'] = delay_sel & reward
                    align_trial_selections[task][subj_id][sess_id][region]['delay_'+resp_delay+'_unreward'] = delay_sel & unreward

                    for rel_side in rel_sides:
                        side_sel = rel_choice_side == rel_side
                        align_trial_selections[task][subj_id][sess_id][region][rel_side+'_delay_'+resp_delay] = side_sel & delay_sel
                        align_trial_selections[task][subj_id][sess_id][region][rel_side+'_delay_'+resp_delay+'_reward'] = side_sel & delay_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region][rel_side+'_delay_'+resp_delay+'_unreward'] = side_sel & delay_sel & unreward

                for lick_dur in rew_lick_durs:
                    dur_sel = (sess_rew_lick_durs == lick_dur) & reward
                    align_trial_selections[task][subj_id][sess_id][region]['rew_lick_dur_'+lick_dur] = dur_sel

                    for rel_side in rel_sides:
                        side_sel = rel_choice_side == rel_side
                        align_trial_selections[task][subj_id][sess_id][region][rel_side+'_rew_lick_dur_'+lick_dur] = side_sel & dur_sel

                for lick_dur in unrew_lick_durs:
                    dur_sel = (sess_unrew_lick_durs == lick_dur) & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['unrew_lick_dur_'+lick_dur] = dur_sel

                    for rel_side in rel_sides:
                        side_sel = rel_choice_side == rel_side
                        align_trial_selections[task][subj_id][sess_id][region][rel_side+'_unrew_lick_dur_'+lick_dur] = side_sel & dur_sel

                if task == 'wm':
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_resp'] = prev_trial_same & response & prev_resp
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_resp'] = prev_trial_diff & response & prev_resp
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_resp_prev_reward'] = prev_trial_same & response & prev_resp & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_resp_prev_unreward'] = prev_trial_same & response & prev_resp & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_resp_prev_reward'] = prev_trial_diff & response & prev_resp & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_resp_prev_unreward'] = prev_trial_diff & response & prev_resp & prev_unreward

                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_prev_reward'] = prev_trial_same & stay & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_prev_reward'] = prev_trial_same & switch & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_prev_unreward'] = prev_trial_same & stay & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_prev_unreward'] = prev_trial_same & switch & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_prev_reward'] = prev_trial_diff & stay & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_prev_reward'] = prev_trial_diff & switch & prev_reward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_prev_unreward'] = prev_trial_diff & stay & prev_unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_prev_unreward'] = prev_trial_diff & switch & prev_unreward

                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_reward'] = prev_trial_same & stay & reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_reward'] = prev_trial_same & switch & reward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_stay_unreward'] = prev_trial_same & stay & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['same_tone_switch_unreward'] = prev_trial_same & switch & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_reward'] = prev_trial_diff & stay & reward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_reward'] = prev_trial_diff & switch & reward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_stay_unreward'] = prev_trial_diff & stay & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['diff_tone_switch_unreward'] = prev_trial_diff & switch & unreward

                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_stay_reward'] = next_trial_same & next_stay & reward
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_switch_reward'] = next_trial_same & next_switch & reward
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_stay_unreward'] = next_trial_same & next_stay & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['next_same_tone_switch_unreward'] = next_trial_same & next_switch & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_stay_reward'] = next_trial_diff & next_stay & reward
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_switch_reward'] = next_trial_diff & next_switch & reward
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_stay_unreward'] = next_trial_diff & next_stay & unreward
                    align_trial_selections[task][subj_id][sess_id][region]['next_diff_tone_switch_unreward'] = next_trial_diff & next_switch & unreward

                    for rew_bin in rew_hist_bins:
                        rew_sel = rew_hist_bail == rew_bin
                        bin_str = rew_hist_bin_strs[rew_bin]

                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str] = rew_sel
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_resp'] = rew_sel & response
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_prev_resp'] = rew_sel & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_prev_bail'] = rew_sel & prev_bail
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_reward'] = rew_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_unreward'] = rew_sel & unreward

                        for rel_side in rel_sides:
                            side_sel = rel_choice_side == rel_side
                            prev_side_sel = rel_prev_choice_side == rel_side

                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_'+rel_side] = rew_sel & side_sel
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_prev_'+rel_side] = rew_sel & prev_side_sel
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_'+rel_side+'_reward'] = rew_sel & side_sel & reward
                            align_trial_selections[task][subj_id][sess_id][region]['rew_hist_bail_'+bin_str+'_'+rel_side+'_unreward'] = rew_sel & side_sel & unreward

                    for tone in tone_types:
                        tone_abs_side = tone_ports[subj_id][tone]
                        tone_rel_side = fpah.get_implant_rel_side(tone_ports[subj_id][tone], region_side)
                        tone_sel = tones == tone

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_response'] = tone_sel & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_response'] = tone_sel & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_response'] = tone_sel & response

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_bail'] = tone_sel & bail
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_bail'] = tone_sel & bail
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_bail'] = tone_sel & bail

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_reward'] = tone_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_reward'] = tone_sel & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward'] = tone_sel & reward

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_unreward'] = tone_sel & unreward & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_abs_side+'_tone_unreward'] = tone_sel & unreward & response
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward'] = tone_sel & unreward & response

                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_resp'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_bail'] = tone_sel & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_reward'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_prev_unreward'] = tone_sel & prev_unreward

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_resp'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_bail'] = tone_sel & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_reward'] = tone_sel & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_prev_unreward'] = tone_sel & prev_unreward

                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward_prev_reward'] = tone_sel & prev_reward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_reward_prev_unreward'] = tone_sel & prev_unreward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward_prev_reward'] = tone_sel & prev_reward & unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_tone_unreward_prev_unreward'] = tone_sel & prev_unreward & unreward

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_reward_prev_reward'] = tone_sel & prev_reward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_reward_prev_unreward'] = tone_sel & prev_unreward & reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_unreward_prev_reward'] = tone_sel & prev_reward & unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_tone_unreward_prev_unreward'] = tone_sel & prev_unreward & unreward

                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp'] = tone_sel & prev_trial_same & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp'] = tone_sel & prev_trial_diff & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp_prev_reward'] = tone_sel & prev_trial_same & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_same_tone_resp_prev_unreward'] = tone_sel & prev_trial_same & response & prev_resp & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp_prev_reward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone_rel_side+'_diff_tone_resp_prev_unreward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_unreward

                        align_trial_selections[task][subj_id][sess_id][region][tone+'_same_tone_resp'] = tone_sel & prev_trial_same & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_diff_tone_resp'] = tone_sel & prev_trial_diff & response & prev_resp
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_same_tone_resp_prev_reward'] = tone_sel & prev_trial_same & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_same_tone_resp_prev_unreward'] = tone_sel & prev_trial_same & response & prev_resp & prev_unreward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_diff_tone_resp_prev_reward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_reward
                        align_trial_selections[task][subj_id][sess_id][region][tone+'_diff_tone_resp_prev_unreward'] = tone_sel & prev_trial_diff & response & prev_resp & prev_unreward


                # get alignment specific selection vectors
                for align in alignments:
                    match align:
                        case Align.cport_on:
                            align_trial_selections[task][subj_id][sess_id][region][align] = cport_on_sel & trial_started

                        # case Align.early_cpoke_in:
                        #     align_trial_selections[task][subj_id][sess_id][region][align] = early_cpoke_in_sel

                        case Align.cpoke_in:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_in_sel | early_cpoke_in_sel

                        case Align.tone:
                            align_trial_selections[task][subj_id][sess_id][region][align] = tone_heard_sel

                        case Align.cue | Align.cpoke_out:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_out_sel

                        case Align.resp | Align.reward:
                            align_trial_selections[task][subj_id][sess_id][region][align] = response

                        case Align.cue_reward:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_out_sel & response

                        case Align.cport_on_cpoke_in:
                            align_trial_selections[task][subj_id][sess_id][region][align] = norm_cpoke_in_sel & engage_cpoke_in_sel


# %% Set up average plot options

# modify these options to change what will be used in the average signal plots
plot_signals = ['z_dff_iso_baseline_fband'] # 'z_dff_iso',
plot_tasks = ['wm', 'bandit'] #
plot_ind_subj = False
plot_meta_subj = True
plot_per_subject = True # whether to have a plot per subject (True) or a plot per region (False)

use_se = True
ph = 3.5
pw = 5

gen_xlims = {'DMS': [-1,1.5], 'DLS': [-1,1.5], 'TS': [-1,1.5], 'NAc': [-1.5,2], 'PL': [-3,3]}
rew_xlims = {'DMS': [-1,2], 'DLS': [-1,2], 'TS': [-1,2], 'NAc': [-2,4], 'PL': [-3,10]}

all_xlims = {Align.cport_on: gen_xlims, Align.early_cpoke_in: gen_xlims, Align.cpoke_in: gen_xlims, Align.tone: gen_xlims,
            Align.cue: gen_xlims, Align.cpoke_out: gen_xlims, Align.resp: gen_xlims, Align.reward: rew_xlims,
            Align.cue_reward: None, Align.cport_on_cpoke_in: None}

tone_dashlines = [0.4]
norm_keys = {Align.cport_on_cpoke_in: ['cport_on', 'cpoke_in'], Align.cue_reward: ['cue', 'poke_out', 'response', 'reward']}
norm_labels = {Align.cport_on_cpoke_in: ['Cport On', 'Poke In'], Align.cue_reward: ['Cue', 'Out', 'Resp', 'Rew']}
norm_dashlines = {task: {align: [aligned_time[task]['events'][align][a] for a in norm_keys[align]] for align in [Align.cport_on_cpoke_in, Align.cue_reward]} for task in plot_tasks}

save_plots = True
show_plots = True
keep_figs_open = True

# declare method to stack data matrices for the given task, alignment, and selection grouping
def stack_mats(task, align, trial_groups, signal_types=plot_signals, sub_sess_ids=None, stack_meta_subj=plot_meta_subj):

    if not utils.is_list(signal_types):
        signal_types = [signal_types]

    stacked_mat_dict = {sig: {subj: {} for subj in subj_ids} for sig in signal_types}
    weights_mat_dict = {sig: {subj: {} for subj in subj_ids} for sig in signal_types}

    for signal_type in signal_types:
        for subj_id in subj_ids:
            subj_regions = get_subj_regions(subj_id)
            sess_ids = list(align_trial_selections[task][subj_id].keys())

            if not sub_sess_ids is None:
                sess_ids = [s for s in sess_ids if s in sub_sess_ids]

            for region in subj_regions:
                stacked_mat_dict[signal_type][subj_id][region] = {}
                weights_mat_dict[signal_type][subj_id][region] = {}

                for group in trial_groups:

                    stacked_mats = []

                    for sess_id in sess_ids:

                        if align == Align.early_cpoke_in:
                            signal_align = Align.cpoke_in
                        else:
                            signal_align = align

                        # some sessions have different regions
                        if not region in aligned_signals[task][subj_id][sess_id][signal_type][signal_align]:
                            continue

                        # get trial selections
                        trial_sel = align_trial_selections[task][subj_id][sess_id][region][align]
                        if '&' in group:
                            group_list = [g.strip() for g in group.split('&')]
                            for g in group_list:
                                trial_sel = trial_sel & align_trial_selections[task][subj_id][sess_id][region][g]
                        else:
                            trial_sel = trial_sel & align_trial_selections[task][subj_id][sess_id][region][group]

                        # get aligned data matrix
                        aligned_mat = aligned_signals[task][subj_id][sess_id][signal_type][signal_align][region]

                        stacked_mats.append(aligned_mat[trial_sel, :])

                    if len(stacked_mats) > 0:
                        stacked_mat_dict[signal_type][subj_id][region][group] = np.vstack(stacked_mats)
                    else:
                        stacked_mat_dict[signal_type][subj_id][region][group] = np.full_like(aligned_time[task]['t'][align][region][None,:], np.nan)

                    weights_mat_dict[signal_type][subj_id][region][group] = np.ones_like(stacked_mat_dict[signal_type][subj_id][region][group])

        if stack_meta_subj:
            # build region mapping since some animals have bilateral implants in the same region
            reg_mapping = {s: get_subj_region_mapping(implant_info[s].keys()) for s in subj_ids}

            stacked_mat_dict[signal_type]['all'] = {r: {g: np.full_like(aligned_time[task]['t'][align][r][None,:], np.nan) for g in trial_groups} for r in all_regions}
            weights_mat_dict[signal_type]['all'] = {r: {g: np.full_like(aligned_time[task]['t'][align][r][None,:], np.nan) for g in trial_groups} for r in all_regions}

            for s in subj_ids:
                subj_regions = get_subj_regions(s)

                for r in subj_regions:
                    n_dups = sum([1 for r2 in reg_mapping[s].values() if r2 == reg_mapping[s][r]])

                    for group in trial_groups:
                        subj_group_mat = stacked_mat_dict[signal_type][s][r][group]
                        n_trials = np.sum(~np.all(np.isnan(subj_group_mat), axis=1))

                        if n_trials == 0:
                            continue

                        stacked_mat_dict[signal_type]['all'][reg_mapping[s][r]][group] = np.vstack([stacked_mat_dict[signal_type]['all'][reg_mapping[s][r]][group],
                                                                                                    subj_group_mat])

                        weights_mat_dict[signal_type]['all'][reg_mapping[s][r]][group] = np.vstack([weights_mat_dict[signal_type]['all'][reg_mapping[s][r]][group],
                                                                                                    np.full_like(subj_group_mat, 1/(n_dups*n_trials))])

    return stacked_mat_dict, weights_mat_dict

# declare method to save plots
def save_plot(fig, task, subjects, plot_name):
    if save_plots and not plot_name is None:
        fpah.save_fig(fig, fpah.get_figure_save_path(beh_names[task], subjects, plot_name))

    if not show_plots:
        plt.close(fig)

# declare method to plot avg signals for the given plot_groups for one or more tasks, signal types, and alignments
def plot_avg_signals(plot_groups, group_labels, plot_group_titles, gen_title, aligns, subjects=subj_ids, sub_sess_ids=None, tasks=plot_tasks, signal_types=plot_signals,
                     xlims_dict=None, dashlines=None, legend_params=None, group_colors=None, gen_plot_name=None, include_norm_reward=True):

    if not utils.is_list(aligns):
        aligns = [aligns]

    if not utils.is_list(tasks):
        tasks = [tasks]

    if not utils.is_list(signal_types):
        signal_types = [signal_types]

    if xlims_dict is None:
        xlims_dict = all_xlims

    plot_subjects = []

    if plot_ind_subj:
        plot_subjects.extend(subjects)

    if plot_meta_subj:
        plot_subjects.append('all')

    all_groups = np.unique(utils.flatten(plot_groups))

    for task in tasks:
        for align in aligns:

            if task != 'wm' and align == Align.tone:
                continue

            align_title = fpah.get_align_title(align)
            x_label = fpah.get_align_xlabel(align)

            align_xlims = xlims_dict[align]

            if align == Align.early_cpoke_in:
                t = aligned_time[task]['t'][Align.cpoke_in]
            else:
                t = aligned_time[task]['t'][align]

            stacked_signals, weights = stack_mats(task, align, all_groups, signal_types=signal_types, sub_sess_ids=sub_sess_ids)

            x_ticks = None
            plot_x0 = True
            dashlines = None

            if align == Align.tone:
                dashlines = tone_dashlines
            elif align in [Align.cport_on_cpoke_in, Align.cue_reward]:
                plot_x0 = False
                dashlines = norm_dashlines[task][align]
                x_ticks = {'ticks': dashlines, 'labels': norm_labels[align]}

                if align == Align.cue_reward and not include_norm_reward:
                    align_xlims = {r: [0, dashlines[-1]] for r in default_xlims.keys()}

            for signal_type in signal_types:
                _, y_label = fpah.get_signal_type_labels(signal_type)

                if plot_per_subject:
                    for subj in plot_subjects:

                        plot_stacked_signals = stacked_signals[signal_type][subj]
                        plot_weights = weights[signal_type][subj]

                        # preserves ordering of regions
                        plot_regions = sort_subj_regions(plot_stacked_signals.keys())

                        if subj == 'all':
                            plot_region_labels = plot_regions
                        else:
                            plot_region_labels = ['{} ({})'.format(r, implant_info[subj][r]['side']) for r in plot_regions]

                        # handle bilateral regions
                        reg_mapping = get_subj_region_mapping(plot_regions)
                        plot_t = {r: t[reg_mapping[r]] for r in plot_regions}
                        plot_xlims = None if align_xlims is None else {r: align_xlims[reg_mapping[r]] for r in plot_regions}

                        title = '{} Aligned to {} - Subj {}, {}'.format(gen_title, align_title, subj, beh_names[task])

                        fig, plotted = fpah.plot_avg_signals(plot_stacked_signals, plot_regions, plot_region_labels, plot_groups, plot_group_titles, group_labels,
                                                             plot_t, title, x_label, y_label, parents_on_rows=True, xlims=plot_xlims, dashlines=dashlines,
                                                             x_ticks=x_ticks, plot_x0=plot_x0, legend_params=legend_params, use_se=use_se,
                                                             group_colors=group_colors, ph=ph, pw=pw, weights_dict=plot_weights)

                        if plotted and not gen_plot_name is None:
                            save_plot(fig, task, subj, '{}_{}_{}'.format(align, gen_plot_name, signal_type))
                else:
                    for region in all_regions:

                        plot_stacked_signals = {}
                        plot_weights = {}
                        reg_subjects = []
                        plot_subject_labels = []

                        for subj in plot_subjects:
                            # handle subjects with bilateral regions
                            for subj_reg in stacked_signals[signal_type][subj].keys():
                                if region in subj_reg:
                                    if subj == 'all':
                                        subj_label = subj
                                    else:
                                        subj_label = '{} ({})'.format(subj, implant_info[subj][subj_reg]['side'])

                                    plot_stacked_signals[subj_label] = stacked_signals[signal_type][subj][subj_reg]
                                    plot_weights[subj_label] = weights[signal_type][subj][subj_reg]
                                    reg_subjects.append(subj_label)
                                    plot_subject_labels.append(subj_label)

                        plot_xlims = None if align_xlims is None else align_xlims[region]

                        title = '{} Aligned to {} - Region {}, {}'.format(gen_title, align_title, region, beh_names[task])

                        fig, plotted = fpah.plot_avg_signals(plot_stacked_signals, reg_subjects, plot_subject_labels, plot_groups, plot_group_titles, group_labels,
                                                             t[region], title, x_label, y_label, parents_on_rows=False, xlims=plot_xlims, dashlines=dashlines,
                                                             x_ticks=x_ticks, plot_x0=plot_x0, legend_params=legend_params, use_se=use_se,
                                                             group_colors=group_colors, ph=ph, pw=pw, weights_dict=plot_weights)


                        if plotted and not gen_plot_name is None:
                            save_plot(fig, task, region, '{}_{}_{}'.format(align, gen_plot_name, signal_type))

                if not plotted:
                    plt.close(fig)
                else:
                    plt.show()
                    if not keep_figs_open:
                        plt.close(fig)


def plot_avg_signals_multi_align(plot_groups, group_labels, plot_group_titles, gen_title, plot_aligns, subjects=subj_ids, sub_sess_ids=None, tasks=plot_tasks, signal_types=plot_signals,
                                 xlims_dict=None, dashlines=None, legend_params=None, group_colors=None, gen_plot_name=None, include_norm_reward=True):

    if not utils.is_list(plot_aligns):
        plot_aligns = [plot_aligns]

    unique_aligns = np.unique(plot_aligns)

    if not utils.is_list(tasks):
        tasks = [tasks]

    if not utils.is_list(signal_types):
        signal_types = [signal_types]

    if xlims_dict is None:
        xlims_dict = all_xlims.copy()

    plot_subjects = []
    if plot_ind_subj:
        plot_subjects.extend(subjects)

    if plot_meta_subj:
        plot_subjects.append('all')

    all_groups = np.unique(utils.flatten(plot_groups))

    for task in tasks:

        # first build all plotting informaton dictionaries by align
        t = {}
        dash_dict = {}
        ticks_dict = {}
        plot_x0 = {}
        x_labels = {}
        stacked_mats_all = {}
        weights_all = {}

        for align in unique_aligns:

            if task != 'wm' and align == Align.tone:
                continue

            x_labels[align] = fpah.get_align_xlabel(align)

            if align == Align.early_cpoke_in:
                t_align = aligned_time[task]['t'][Align.cpoke_in]
            else:
                t_align = aligned_time[task]['t'][align]

            align_stacked_mats, align_weights = stack_mats(task, align, all_groups, signal_types=signal_types, sub_sess_ids=sub_sess_ids)

            t[align] = t_align
            stacked_mats_all[align] = align_stacked_mats
            weights_all[align] = align_weights

            dashes = None
            ticks = None
            px0 = True

            if align == Align.tone:
                dashes = tone_dashlines

            elif align in [Align.cport_on_cpoke_in, Align.cue_reward]:
                px0 = False
                dashes = norm_dashlines[task][align]
                ticks = {'ticks': dashes, 'labels': norm_labels[align]}

                if align == Align.cue_reward and not include_norm_reward:
                    xlims_dict[align] = {r: [0, dashes[-1]] for r in default_xlims.keys()}

            dash_dict[align] = dashes
            ticks_dict[align] = ticks
            plot_x0[align] = px0

        # perform plotting
        for signal_type in signal_types:

            _, y_label = fpah.get_signal_type_labels(signal_type)

            if plot_per_subject:
                for subj in plot_subjects:
                    plot_stacked_signals = {}
                    plot_weights = {}
                    plot_t = {}
                    plot_xlims = {}

                    if subj == 'all':
                        plot_regions = sort_subj_regions(all_regions)
                        plot_region_labels = plot_regions
                    else:
                        plot_regions = get_subj_regions(subj)
                        plot_region_labels = ['{} ({})'.format(r, implant_info[subj][r]['side']) for r in plot_regions]

                    # handle bilateral regions
                    reg_mapping = get_subj_region_mapping(plot_regions)

                    for align in unique_aligns:

                        plot_stacked_signals[align] = stacked_mats_all[align][signal_type][subj]
                        plot_weights[align] = weights_all[align][signal_type][subj]

                        plot_t[align] = {r: t[align][reg_mapping[r]] for r in plot_regions}
                        plot_xlims[align] = None if xlims_dict[align] is None else {r: xlims_dict[align][reg_mapping[r]] for r in plot_regions}

                    title = '{} - Subj {}, {}'.format(gen_title, subj, beh_names[task])

                    fig, plotted = fpah.plot_avg_signals_multi_align(plot_stacked_signals, plot_aligns, plot_regions, plot_region_labels, plot_groups,
                                                                     plot_group_titles, group_labels, plot_t, title, x_labels, y_label, parents_on_rows=True,
                                                                     weights_dict=plot_weights, xlims=plot_xlims, dashlines=dash_dict, x_ticks=ticks_dict,
                                                                     plot_x0=plot_x0, legend_params=legend_params, use_se=use_se, group_colors=group_colors, ph=ph, pw=pw)

                    if plotted and not gen_plot_name is None:
                        save_plot(fig, task, subj, '{}_{}_{}'.format('_'.join(unique_aligns), gen_plot_name, signal_type))

            else:
                for region in all_regions:
                    plot_stacked_signals = {a: {} for a in unique_aligns}
                    plot_weights = {a: {} for a in unique_aligns}
                    plot_t = {}
                    plot_xlims = {}
                    reg_subjects = []
                    plot_subject_labels = []

                    for align in unique_aligns:
                        plot_t[align] = t[align][region]
                        plot_xlims[align] = None if xlims_dict[align] is None else xlims_dict[align][region]

                        for subj in plot_subjects:
                            for subj_reg in stacked_mats_all[align][signal_type][subj].keys():
                                if region in subj_reg:
                                    if subj == 'all':
                                        subj_label = subj
                                    else:
                                        subj_label = '{} ({})'.format(subj, implant_info[subj][subj_reg]['side'])

                                    plot_stacked_signals[align][subj_label] = stacked_mats_all[align][signal_type][subj][subj_reg]
                                    plot_weights[align][subj_label] = weights_all[align][signal_type][subj][subj_reg]
                                    reg_subjects.append(subj_label)
                                    plot_subject_labels.append(subj_label)

                    title = '{} - Region {}, {}'.format(gen_title, region, beh_names[task])

                    fig, plotted = fpah.plot_avg_signals_multi_align(plot_stacked_signals, plot_aligns, reg_subjects, plot_subject_labels, plot_groups,
                                                                     plot_group_titles, group_labels, plot_t, title, x_labels, y_label, parents_on_rows=True,
                                                                     weights_dict=plot_weights, xlims=plot_xlims, dashlines=dash_dict, x_ticks=ticks_dict,
                                                                     plot_x0=plot_x0, legend_params=legend_params, use_se=use_se, group_colors=group_colors, ph=ph, pw=pw)

                    if plotted and not gen_plot_name is None:
                        save_plot(fig, task, region, '{}_{}_{}'.format('_'.join(unique_aligns), gen_plot_name, signal_type))

            if not plotted:
                plt.close(fig)
            else:
                plt.show()
                if not keep_figs_open:
                    plt.close(fig)

# %% Cue and Outcome in same plot

plot_aligns = [Align.cue_reward, Align.reward]
plot_groups = [['all'], ['rewarded', 'unrewarded']]
group_labels = {'all': '_', 'rewarded': 'Reward', 'unrewarded': 'Unreward'}
plot_titles = ['Cue to Response (All Choices)', 'Outcome']
gen_title = 'Cue and Outcome'
gen_plot_name = 'basic_cue_outcome'

group_colors = [np.array(plt.cm.Greens(0.9))[None,:], np.vstack([plt.cm.Blues(0.9), plt.cm.Reds(0.9)])]
plot_lims = {#Align.cue_reward: {r: [0, norm_dashlines[task][Align.cue_reward][-1]] for r in default_xlims.keys()},
             Align.reward: {'DMS': [0,1.5], 'DLS': [0,1.5], 'TS': [0,1.5], 'NAc': [0,3], 'PL': [0,10]}}

plot_avg_signals_multi_align(plot_groups, group_labels, plot_titles, gen_title, plot_aligns, gen_plot_name=gen_plot_name,
                             group_colors=group_colors, include_norm_reward=False,
                             xlims_dict=plot_lims)


# %% Choice and side

plot_groups = [['contra', 'ipsi'], ['stay', 'switch'], ['contra_stay', 'contra_switch', 'ipsi_stay', 'ipsi_switch']]
group_labels = {'stay': 'Stay', 'switch': 'Switch',
                'ipsi': 'Ipsi', 'contra': 'Contra',
                'contra_stay': 'Contra Stay', 'contra_switch': 'Contra Switch',
                'ipsi_stay': 'Ipsi Stay', 'ipsi_switch': 'Ipsi Switch'}

plot_titles = ['Choice Side', 'Stay/Switch', 'Stay/Switch & Side']
gen_title = 'Choice Side & Stay/Switch Groupings'
gen_plot_name = 'stay_switch_side'

aligns = [Align.reward] # Align.cport_on_cpoke_in, Align.cue_reward,  Align.cue, Align.cpoke_out, Align.resp,

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, include_norm_reward=False)

# %% Stay/Switch and Outcome

plot_groups = [['stay_reward', 'switch_reward'], ['stay_unreward', 'switch_unreward']]
group_labels = {'stay_reward': 'Stay', 'switch_reward': 'Switch',
                'stay_unreward': 'Stay', 'switch_unreward': 'Switch'}

plot_titles = ['Rewarded', 'Unrewarded']
gen_title = 'Stay/Switch by Outcome'
gen_plot_name = 'stay_switch_outcome'

aligns = [Align.reward] # Align.cport_on_cpoke_in, Align.cue_reward,  Align.cue, Align.cpoke_out, Align.resp,

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, include_norm_reward=False)


# %% choice side and outcome
plot_groups = [['contra', 'ipsi'],
               ['rewarded', 'unrewarded'],
               ['contra_reward', 'contra_unreward', 'ipsi_reward', 'ipsi_unreward']]
group_labels = {'contra': 'Contra', 'ipsi': 'Ipsi',
                'rewarded': 'Rewarded', 'unrewarded': 'Unrewarded',
                'contra_reward': 'Contra Rewarded', 'ipsi_reward': 'Ipsi Rewarded',
                'contra_unreward': 'Contra Unrewarded', 'ipsi_unreward': 'Ipsi Unrewarded'}

plot_titles = ['Choice Side', 'Outcome', 'Choice Side/Outcome']
gen_title = 'Choice Side by Outcome'
gen_plot_name = 'side_outcome'

aligns = [Align.reward, Align.cue_reward] # Align.cue, Align.cpoke_out, Align.resp,

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)


plot_groups = [['stay_reward', 'switch_reward', 'stay_unreward', 'switch_unreward'],
               ['contra_stay_reward', 'contra_switch_reward', 'contra_stay_unreward', 'contra_switch_unreward'],
               ['ipsi_stay_reward', 'ipsi_switch_reward', 'ipsi_stay_unreward', 'ipsi_switch_unreward']]
group_labels = {'stay_reward': 'Rewarded Stay', 'switch_reward': 'Rewarded Switch',
                'stay_unreward': 'Unrewarded Stay', 'switch_unreward': 'Unrewarded Switch',
                'contra_stay_reward': 'Rewarded Stay', 'contra_switch_reward': 'Rewarded Switch',
                'contra_stay_unreward': 'Unrewarded Stay', 'contra_switch_unreward': 'Unrewarded Switch',
                'ipsi_stay_reward': 'Rewarded Stay', 'ipsi_switch_reward': 'Rewarded Switch',
                'ipsi_stay_unreward': 'Unrewarded Stay', 'ipsi_switch_unreward': 'Unrewarded Switch'}

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Stay/Switch Choice by Outcome'
gen_plot_name = 'stay_switch_side_outcome'

aligns = [Align.reward, Align.cue_reward] # Align.cue, Align.cpoke_out, Align.resp,

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)


# # %% prior & future outcome
# plot_groups = [['prev_reward', 'prev_unreward', 'prev_bail'], ['reward', 'unreward', 'bail']]
# group_labels = {'prev_reward': 'Prev Reward', 'prev_unreward': 'Prev Unreward', 'prev_bail': 'Prev Bail',
#                 'reward': 'Rewarded', 'unreward': 'Unrewarded', 'bail': 'Bail'}

# plot_titles = ['Prior Trial Outcome', 'Current Trial Outcome']
# gen_title = 'Previous and Current Outcome'
# gen_plot_name = 'prev_future_outcome'

# aligns = [Align.cport_on, Align.early_cpoke_in, Align.cpoke_in, Align.cue, Align.cpoke_out]

# plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)

# %% prior choice side/outcome

plot_groups = [['prev_reward', 'prev_unreward'], ['prev_contra', 'prev_ipsi'], ['prev_contra_prev_reward', 'prev_contra_prev_unreward', 'prev_ipsi_prev_reward', 'prev_ipsi_prev_unreward']]
group_labels = {'prev_contra': 'Prev Contra', 'prev_ipsi': 'Prev Ipsi',
                'prev_reward': 'Prev Reward', 'prev_unreward': 'Prev Unreward',
                'prev_contra_prev_reward': 'Prev Contra Reward', 'prev_contra_prev_unreward': 'Prev Contra Unreward',
                'prev_ipsi_prev_reward': 'Prev Ipsi Reward', 'prev_ipsi_prev_unreward': 'Prev Ipsi Unreward'}

plot_titles = ['Prior Outcome', 'Prior Choice Side', 'Prior Choice Side & Outcome']
gen_title = 'Previous Outcome and Choice Side'
gen_plot_name = 'prev_side_prev_outcome'

aligns = [Align.cport_on_cpoke_in]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)


# %% prior reward/choice/side

plot_groups = [['stay_prev_reward', 'switch_prev_reward', 'stay_prev_unreward', 'switch_prev_unreward'],
               ['contra_prev_reward', 'ipsi_prev_reward', 'contra_prev_unreward', 'ipsi_prev_unreward'],
               ['contra_stay_prev_reward', 'contra_switch_prev_reward', 'ipsi_stay_prev_reward', 'ipsi_switch_prev_reward'],
               ['contra_stay_prev_unreward', 'contra_switch_prev_unreward', 'ipsi_stay_prev_unreward', 'ipsi_switch_prev_unreward']]

group_labels = {'stay_prev_reward': 'Stay | Reward', 'switch_prev_reward': 'Switch | Reward',
                'stay_prev_unreward': 'Stay | Unreward', 'switch_prev_unreward': 'Switch | Unreward',
                'contra_prev_reward': 'Contra | Reward', 'ipsi_prev_reward': 'Ipsi | Reward',
                'contra_prev_unreward': 'Contra | Unreward', 'ipsi_prev_unreward': 'Ipsi | Unreward',
                'contra_stay_prev_reward': 'Contra Stay', 'contra_switch_prev_reward': 'Contra Switch',
                'ipsi_stay_prev_reward': 'Ipsi Stay', 'ipsi_switch_prev_reward': 'Ipsi Switch',
                'contra_stay_prev_unreward': 'Contra Stay', 'contra_switch_prev_unreward': 'Contra Switch',
                'ipsi_stay_prev_unreward': 'Ipsi Stay', 'ipsi_switch_prev_unreward': 'Ipsi Switch'}

plot_titles = ['Stay/Switch by Prior Outcome', 'Choice Side by Prior Outcome', 'Prior Reward', 'Prior Unreward']
gen_title = 'Previous Outcome by Stay/Switch and Choice Side'
gen_plot_name = 'prev_outcome_stay_switch_side'

aligns = [Align.cport_on_cpoke_in, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, include_norm_reward=False)

# %% Outcome, future stay/switch

plot_groups = [['reward_next_stay', 'reward_next_switch', 'unreward_next_stay', 'unreward_next_switch'],
               ['contra_reward_next_stay', 'contra_reward_next_switch', 'contra_unreward_next_stay', 'contra_unreward_next_switch'],
               ['ipsi_reward_next_stay', 'ipsi_reward_next_switch', 'ipsi_unreward_next_stay', 'ipsi_unreward_next_switch']]

group_labels = {'reward_next_stay': 'Reward Next Stay', 'reward_next_switch': 'Reward Next Switch',
                'unreward_next_stay': 'Unreward Next Stay', 'unreward_next_switch': 'Unreward Next Switch',
                'contra_reward_next_stay': 'Reward Next Stay', 'contra_reward_next_switch': 'Reward Next Switch',
                'contra_unreward_next_stay': 'Unreward Next Stay', 'contra_unreward_next_switch': 'Unreward Next Switch',
                'ipsi_reward_next_stay': 'Reward Next Stay', 'ipsi_reward_next_switch': 'Reward Next Switch',
                'ipsi_unreward_next_stay': 'Unreward Next Stay', 'ipsi_unreward_next_switch': 'Unreward Next Switch'}

plot_titles = ['Current Outcome and Future Choice', 'Contra Choice Outcome and Future Choice', 'Ipsi Choice Outcome and Future Choice']
gen_title = 'Current Outcome & Choice by Next Choice'
gen_plot_name = 'outcome_side_next_stay_switch'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name)

# %% Reward History

task = 'bandit'
sep_block_rates = True

if sep_block_rates:
    plot_sess_ids = bandit_block_rate_sess_ids
else:
    plot_sess_ids = {'all': None}

add_filt = ''#'&norm_cpoke_in'

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

gen_group_labels = {'rew_hist_{}_resp'+add_filt: '{}', 'rew_hist_{}_prev_resp'+add_filt: '{}', 'rew_hist_{}_reward'+add_filt: '{} Rew', 'rew_hist_{}_unreward'+add_filt: '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_{}_{}'+add_filt: '{}', 'rew_hist_{}_prev_{}'+add_filt: '{}', 'rew_hist_{}_{}_reward'+add_filt: '{} Rew', 'rew_hist_{}_{}_unreward'+add_filt: '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins
                for s in rel_sides})

for block_name, plot_sub_sess in plot_sess_ids.items():

    # Reward History by previous choice side
    gen_plot_group = ['rew_hist_{}_prev_resp'+add_filt, 'rew_hist_{}_prev_contra'+add_filt, 'rew_hist_{}_prev_ipsi'+add_filt]
    plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

    plot_titles = ['All Prev Choices', 'Previous Contra Choices', 'Previous Ipsi Choices']
    gen_title = 'Reward History by Previous Choice Side. Reward Rate: {}'.format(block_name)
    gen_plot_name = 'rew_hist_prev_side_{}'.format(block_name.replace('/', '_'))

    aligns = [Align.cport_on_cpoke_in]

    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                     group_colors=rew_hist_all_colors, tasks=task, sub_sess_ids=plot_sub_sess)

    # Reward History by current choice side
    gen_plot_group = ['rew_hist_{}_resp'+add_filt, 'rew_hist_{}_contra'+add_filt, 'rew_hist_{}_ipsi'+add_filt]
    plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

    plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
    gen_title = 'Reward History by Choice Side. Reward Rate: {}'.format(block_name)
    gen_plot_name = 'rew_hist_side_{}'.format(block_name.replace('/', '_'))

    aligns = [Align.cue_reward]

    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                     group_colors=rew_hist_all_colors, include_norm_reward=False, tasks=task, sub_sess_ids=plot_sub_sess)

    # Reward History by current choice side & outcome

    gen_plot_group = ['rew_hist_{}_reward'+add_filt, 'rew_hist_{}_unreward'+add_filt]
    plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins]]

    gen_plot_group = ['rew_hist_{}_{}_reward'+add_filt, 'rew_hist_{}_{}_unreward'+add_filt]
    plot_groups.extend([[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides])

    plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
    gen_title = 'Reward History By Choice Side & Outcome. Reward Rate: {}'.format(block_name)
    gen_plot_name = 'rew_hist_side_outcome_{}'.format(block_name.replace('/', '_'))

    aligns = [Align.reward, Align.cue_reward]

    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, sub_sess_ids=plot_sub_sess,
                     group_colors=np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), legend_params={'ncol': 2}, tasks=task)

# %% WM Reward history w/ bails

add_filt = ''#'&norm_cpoke_in'
task = 'wm'
rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

gen_group_labels = {'rew_hist_bail_{}_resp'+add_filt: '{}', 'rew_hist_bail_{}_prev_resp'+add_filt: '{}',
                    'rew_hist_bail_{}_reward'+add_filt: '{} Rew', 'rew_hist_bail_{}_unreward'+add_filt: '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_bail_{}_{}'+add_filt: '{}', 'rew_hist_bail_{}_prev_{}'+add_filt: '{}',
                    'rew_hist_bail_{}_{}_reward'+add_filt: '{} Rew', 'rew_hist_bail_{}_{}_unreward'+add_filt: '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins
                for s in rel_sides})

# Reward History by Choice Side

# previous choice side
gen_plot_group = ['rew_hist_bail_{}_prev_resp'+add_filt, 'rew_hist_bail_{}_prev_contra'+add_filt, 'rew_hist_bail_{}_prev_ipsi'+add_filt]
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Prev Choices', 'Previous Contra Choices', 'Previous Ipsi Choices']
gen_title = 'Reward History with Bails by Previous Choice Side'
gen_plot_name = 'rew_hist_bail_prev_side'

aligns = [Align.cport_on_cpoke_in, Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                 group_colors=rew_hist_all_colors, tasks=task)

# current choice side
gen_plot_group = ['rew_hist_bail_{}_resp'+add_filt, 'rew_hist_bail_{}_contra'+add_filt, 'rew_hist_bail_{}_ipsi'+add_filt]
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for rew_bin in rew_hist_bins] for g in gen_plot_group]

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History with Bails by Choice Side'
gen_plot_name = 'rew_hist_bail_side'

aligns = [Align.cue_reward, Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                 group_colors=rew_hist_all_colors, tasks=task, include_norm_reward=False)

# choice side & outcome

gen_plot_group = ['rew_hist_bail_{}_reward'+add_filt, 'rew_hist_bail_{}_unreward'+add_filt]
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins]]

gen_plot_group = ['rew_hist_bail_{}_{}_reward'+add_filt, 'rew_hist_bail_{}_{}_unreward'+add_filt]
plot_groups.extend([[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides])

plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History with Bails By Choice Side & Outcome'
gen_plot_name = 'rew_hist_bail_side_outcome'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                 group_colors=np.vstack((rew_hist_rew_colors, rew_hist_unrew_colors)), tasks=task, legend_params={'ncol': 2})

# %% Rew Hist norm cue and reward in same plot

show_legend = False

rew_hist_all_colors = plt.cm.Greens(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_unrew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_hist_bins)))
rew_hist_rew_colors = plt.cm.Blues(np.linspace(0.4,1,len(rew_hist_bins)))

add_filt = ''#'&norm_cpoke_in'
task = 'bandit'
sep_block_rates = False

if sep_block_rates:
    plot_sess_ids = bandit_block_rate_sess_ids
else:
    plot_sess_ids = {'all': None}

plot_lims = {Align.cue_reward: {r: [0, norm_dashlines[task][Align.cue_reward][-1]] for r in default_xlims.keys()},
             Align.reward: {'DMS': [0,1.5], 'DLS': [0,1.5], 'TS': [0,1.5], 'NAc': [0,3], 'PL': [0,10]}}
if show_legend:
    legend_params = {Align.cue_reward: None, Align.reward: {'ncol': 2}}
else:
    legend_params = False

gen_group_labels = {'rew_hist_{}_resp'+add_filt: '{}', 'rew_hist_{}_reward'+add_filt: '{} Rew', 'rew_hist_{}_unreward'+add_filt: '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_{}_{}'+add_filt: '{}', 'rew_hist_{}_{}_reward'+add_filt: '{} Rew', 'rew_hist_{}_{}_unreward'+add_filt: '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins
                for s in rel_sides})

for block_name, plot_sub_sess in plot_sess_ids.items():

    # all choice sides
    plot_aligns = [Align.cue_reward, Align.reward]
    gen_plot_group = ['rew_hist_{}_resp'+add_filt]
    plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins]]

    gen_plot_group = ['rew_hist_{}_reward'+add_filt, 'rew_hist_{}_unreward'+add_filt]
    plot_groups.append([g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins])

    plot_titles = ['Cue to Response (All Choices)', 'Reward by Outcome']
    gen_title = 'Reward History by Outcome. Reward Rate: {}'.format(block_name)
    gen_plot_name = 'rew_hist_outcome_{}'.format(block_name.replace('/', '_'))
    group_colors = [rew_hist_all_colors, np.vstack([rew_hist_rew_colors, rew_hist_unrew_colors])]


    plot_avg_signals_multi_align(plot_groups, group_labels, plot_titles, gen_title, plot_aligns, gen_plot_name=gen_plot_name,
                                 group_colors=group_colors, tasks=task, sub_sess_ids=plot_sub_sess, include_norm_reward=False,
                                 xlims_dict=plot_lims, legend_params=legend_params)

    # separate choice sides
    plot_aligns = [Align.cue_reward, Align.cue_reward, Align.reward, Align.reward]
    gen_plot_group = ['rew_hist_{}_{}'+add_filt]
    plot_groups = [[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides]

    gen_plot_group = ['rew_hist_{}_{}_reward'+add_filt, 'rew_hist_{}_{}_unreward'+add_filt]
    plot_groups.extend([[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides])

    plot_titles = ['Contra Choices', 'Ipsi Choices', 'Contra Choices', 'Ipsi Choices']
    gen_title = 'Reward History by Choice Side and Outcome. Reward Rate: {}'.format(block_name)
    gen_plot_name = 'rew_hist_side_outcome_{}'.format(block_name.replace('/', '_'))
    group_colors = [rew_hist_all_colors, rew_hist_all_colors, np.vstack([rew_hist_rew_colors, rew_hist_unrew_colors]), np.vstack([rew_hist_rew_colors, rew_hist_unrew_colors])]


    plot_avg_signals_multi_align(plot_groups, group_labels, plot_titles, gen_title, plot_aligns, gen_plot_name=gen_plot_name,
                                 group_colors=group_colors, tasks=task, sub_sess_ids=plot_sub_sess, include_norm_reward=False,
                                 xlims_dict=plot_lims, legend_params=legend_params)

task = 'wm'

gen_group_labels = {'rew_hist_bail_{}_resp'+add_filt: '{}', 'rew_hist_bail_{}_reward'+add_filt: '{} Rew', 'rew_hist_bail_{}_unreward'+add_filt: '{} Unrew'}

group_labels = {k.format(rew_hist_bin_strs[rew_bin]): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins}

gen_group_labels = {'rew_hist_bail_{}_{}'+add_filt: '{}', 'rew_hist_bail_{}_{}_reward'+add_filt: '{} Rew', 'rew_hist_bail_{}_{}_unreward'+add_filt: '{} Unrew'}

group_labels.update({k.format(rew_hist_bin_strs[rew_bin], s): v.format(rew_hist_bin_strs[rew_bin])
                for k,v in gen_group_labels.items()
                for rew_bin in rew_hist_bins
                for s in rel_sides})


# all choice sides
plot_aligns = [Align.cue_reward, Align.reward]
gen_plot_group = ['rew_hist_bail_{}_resp'+add_filt]
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins]]

gen_plot_group = ['rew_hist_bail_{}_reward'+add_filt, 'rew_hist_bail_{}_unreward'+add_filt]
plot_groups.append([g.format(rew_hist_bin_strs[rew_bin]) for g in gen_plot_group for rew_bin in rew_hist_bins])

plot_titles = ['Cue to Response (All Choices)', 'Reward by Outcome']
gen_title = 'Reward History by Outcome'
gen_plot_name = 'rew_hist_outcome'
group_colors = [rew_hist_all_colors, np.vstack([rew_hist_rew_colors, rew_hist_unrew_colors])]


plot_avg_signals_multi_align(plot_groups, group_labels, plot_titles, gen_title, plot_aligns, gen_plot_name=gen_plot_name,
                             group_colors=group_colors, tasks=task, include_norm_reward=False, xlims_dict=plot_lims, legend_params=legend_params)

# separate choice sides
plot_aligns = [Align.cue_reward, Align.cue_reward, Align.reward, Align.reward]
gen_plot_group = ['rew_hist_bail_{}_{}'+add_filt]
plot_groups = [[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides]

gen_plot_group = ['rew_hist_bail_{}_{}_reward'+add_filt, 'rew_hist_bail_{}_{}_unreward'+add_filt]
plot_groups.extend([[g.format(rew_hist_bin_strs[rew_bin], s) for g in gen_plot_group for rew_bin in rew_hist_bins] for s in rel_sides])

plot_titles = ['Contra Choices', 'Ipsi Choices', 'Contra Choices', 'Ipsi Choices']
gen_title = 'Reward History by Choice Side and Outcome'
gen_plot_name = 'rew_hist_side_outcome'
group_colors = [rew_hist_all_colors, rew_hist_all_colors, np.vstack([rew_hist_rew_colors, rew_hist_unrew_colors]), np.vstack([rew_hist_rew_colors, rew_hist_unrew_colors])]


plot_avg_signals_multi_align(plot_groups, group_labels, plot_titles, gen_title, plot_aligns, gen_plot_name=gen_plot_name,
                             group_colors=group_colors, tasks=task, include_norm_reward=False, xlims_dict=plot_lims, legend_params=legend_params)


# %% Response Delay

for task in tasks:
    if task == 'wm':
        sess_data = wm_sess_data
        resp_delays = sess_data['resp_delay_bin'].cat.categories
    else:
        sess_data = bandit_sess_data
        resp_delays = sess_data['trial_dur_bin'].cat.categories

    delay_colors = plt.cm.Oranges(np.linspace(0.4,1,len(resp_delays)))
    delay_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(resp_delays)))
    delay_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(resp_delays)))

    # choice sides

    gen_group_labels = {'delay_{}': '{}', 'contra_delay_{}': '{}', 'ipsi_delay_{}': '{}',
                        'delay_{}_reward': '{} Rew', 'delay_{}_unreward': '{} Unrew'}

    group_labels = {k.format(delay): v.format(delay)
                    for k,v in gen_group_labels.items()
                    for delay in resp_delays}

    gen_group_labels = {'{}_delay_{}_reward': '{} Rew', '{}_delay_{}_unreward': '{} Unrew'}

    group_labels.update({k.format(s, delay): v.format(delay)
                    for k,v in gen_group_labels.items()
                    for delay in resp_delays
                    for s in rel_sides})

    gen_plot_group = ['delay_{}', 'contra_delay_{}', 'ipsi_delay_{}']
    plot_groups = [[g.format(delay) for delay in resp_delays] for g in gen_plot_group]

    plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
    gen_title = 'Response Delays by Choice Side'
    gen_plot_name = 'resp_delay_side'

    aligns = [Align.cue_reward, Align.cue]

    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                     group_colors=delay_colors, tasks=task, include_norm_reward=False)


    # outcomes and choices sides

    gen_plot_group = ['delay_{}_reward', 'delay_{}_unreward']
    plot_groups = [[g.format(delay) for g in gen_plot_group for delay in resp_delays]]

    gen_plot_group = ['{}_delay_{}_reward', '{}_delay_{}_unreward']
    plot_groups.extend([[g.format(s, delay) for g in gen_plot_group for delay in resp_delays] for s in rel_sides])

    plot_titles = ['All Choices', 'Contra Choices', 'Ipsi Choices']
    gen_title = 'Response Delays By Choice Side & Outcome'
    gen_plot_name = 'resp_delay_side_outcome'

    aligns = [Align.cue_reward, Align.cue]

    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                     group_colors=np.vstack((delay_rew_colors, delay_unrew_colors)), tasks=task, legend_params={'ncol': 2})

    # plot again without the reward
    gen_plot_name = 'resp_delay_side_outcome_no_rew'
    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                     group_colors=np.vstack((delay_rew_colors, delay_unrew_colors)), tasks=task, include_norm_reward=False, legend_params={'ncol': 2})


# %% Lick Duration

for task in tasks:
    if task == 'wm':
        sess_data = wm_sess_data
    else:
        sess_data = bandit_sess_data

    rew_lick_durs = sess_data['rew_lick_dur_bin'].cat.categories
    unrew_lick_durs = sess_data['unrew_lick_dur_bin'].cat.categories

    lick_colors = plt.cm.Oranges(np.linspace(0.4,1,len(rew_lick_durs)))
    lick_rew_colors = plt.cm.Reds(np.linspace(0.4,1,len(rew_lick_durs)))
    lick_unrew_colors = plt.cm.Blues(np.linspace(0.4,1,len(unrew_lick_durs)))

    group_labels = {'rew_lick_dur_{}'.format(d): d for d in rew_lick_durs}
    group_labels.update({'unrew_lick_dur_{}'.format(d): d for d in unrew_lick_durs})

    plot_groups = [['rew_lick_dur_{}'.format(d) for d in rew_lick_durs],
                   ['unrew_lick_dur_{}'.format(d) for d in unrew_lick_durs]]

    plot_titles = ['Rewarded', 'Unrewarded']
    gen_title = 'Licking Durations By Outcome'
    gen_plot_name = 'lick_dur'

    aligns = [Align.cue_reward, Align.reward]

    plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name,
                     group_colors=lick_colors, tasks=task)


# %% WM Tones

task = 'wm'

group_labels = {'response': 'Resp', 'bail': 'Bail', 'rewarded': 'Rew', 'unrewarded': 'Unrew'}

plot_groups = [['response', 'bail'], ['rewarded', 'unrewarded']]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Response and Outcome for All Tones'
gen_plot_name = 'all_tones_resp_outcome'

aligns = [Align.tone]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task)


gen_group_labels = {'{}_tone_response': '{} Resp', '{}_tone_bail': '{} Bail', '{}_tone_reward': '{} Rew', '{}_tone_unreward': '{} Unrew'}

gen_plot_groups = [['{}_tone_response', '{}_tone_bail'], ['{}_tone_reward', '{}_tone_unreward']]

# tone types
group_labels = {k.format(tone): v.format(tone.capitalize())
                for k,v in gen_group_labels.items()
                for tone in tone_types}

plot_groups = [[g.format(tone) for tone in tone_types for g in gs] for gs in gen_plot_groups]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Tone Types by Response and Outcome'
gen_plot_name = 'tone_type_resp_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# tone correct relative side
group_labels = {k.format(tone): v.format(tone.capitalize())
                for k,v in gen_group_labels.items()
                for tone in rel_sides}

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Tone Correct Relative Sides by Response and Outcome'
gen_plot_name = 'tone_rel_side_resp_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# tone correct absolute side
group_labels = {k.format(tone): v.format(tone.capitalize())
                for k,v in gen_group_labels.items()
                for tone in abs_sides}

plot_groups = [[g.format(tone) for tone in abs_sides for g in gs] for gs in gen_plot_groups]

plot_titles = ['Response Type', 'Outcome']
gen_title = 'Tone Correct Absolute Sides by Response and Outcome'
gen_plot_name = 'tone_abs_side_resp_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# %% Prev/Next stimulus by prev/next choice and current outcome

task = 'wm'

# previous and current outcome
group_labels = {'prev_reward': 'Prev Rew', 'prev_unreward': 'Prev Unrew', 'prev_bail': 'Prev Bail',
                'reward_prev_reward': 'Rew Prev Rew', 'reward_prev_unreward': 'Rew Prev Unrew',
                'unreward_prev_reward': 'Unrew Prev Rew', 'unreward_prev_unreward': 'Unrew Prev Unrew'}

plot_groups = [['prev_reward', 'prev_unreward', 'prev_bail'],
               ['reward_prev_reward', 'unreward_prev_reward', 'reward_prev_unreward', 'unreward_prev_unreward']]

plot_titles = ['Prior Outcome', 'Prior & Current Outcome']
gen_title = 'Previous & Future Outcome'
gen_plot_name = 'prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# previous and current outcome by tone type and side
gen_group_labels = {'{}_tone_prev_reward': '{} Prev Rew', '{}_tone_prev_unreward': '{} Prev Unrew', '{}_tone_prev_bail': '{} Prev Bail',
                    '{}_tone_reward': '{} Rew', '{}_tone_unreward': '{} Unrew', '{}_tone_bail': '{} Bail',
                    '{}_tone_reward_prev_reward': '{} Rew | Rew', '{}_tone_reward_prev_unreward': '{} Rew | Unrew',
                    '{}_tone_unreward_prev_reward': '{} Unrew | Rew', '{}_tone_unreward_prev_unreward': '{} Unrew | Unrew'}

group_labels = {k.format(tone): v.format(tone.capitalize())
                for k,v in gen_group_labels.items()
                for tone in rel_sides}

gen_plot_groups = [['{}_tone_prev_reward', '{}_tone_prev_unreward', '{}_tone_prev_bail'], ['{}_tone_reward', '{}_tone_unreward', '{}_tone_bail']]

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_tone_reward_prev_reward', '{}_tone_reward_prev_unreward', '{}_tone_unreward_prev_reward', '{}_tone_unreward_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in rel_sides])

plot_titles = ['Prior Outcome', 'Future Outcome', 'Contra Tones by Prev & Future Outcome', 'Ipsi Tones by Prev & Future Outcome']
gen_title = 'Tones by Relative Side, Previous & Future Outcomes'
gen_plot_name = 'tone_side_prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


group_labels = {k.format(tone): v.format(tone.capitalize())
                for k,v in gen_group_labels.items()
                for tone in tone_types}

gen_plot_groups = [['{}_tone_prev_reward', '{}_tone_prev_unreward', '{}_tone_prev_bail'], ['{}_tone_reward', '{}_tone_unreward', '{}_tone_bail']]

plot_groups = [[g.format(tone) for tone in tone_types for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_tone_reward_prev_reward', '{}_tone_reward_prev_unreward', '{}_tone_unreward_prev_reward', '{}_tone_unreward_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in tone_types])

plot_titles = ['Prior Outcome', 'Future Outcome', 'Contra Tones by Prev & Future Outcome', 'Ipsi Tones by Prev & Future Outcome']
gen_title = 'Tones by Relative Side, Previous & Future Outcomes'
gen_plot_name = 'tone_type_prev_future_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# consecutive stimuli by prior outcome, all side choices
group_labels = {'same_tone_resp': 'Same', 'diff_tone_resp': 'Diff',
                'prev_reward': 'Prev Rew', 'prev_unreward': 'Prev Unrew',
                'same_tone_resp_prev_reward': 'Same Prev Rew', 'same_tone_resp_prev_unreward': 'Same Prev Unrew',
                'diff_tone_resp_prev_reward': 'Diff Prev Rew', 'diff_tone_resp_prev_unreward': 'Diff Prev Unrew'}

plot_groups = [['same_tone_resp', 'diff_tone_resp'],
               ['prev_reward', 'prev_unreward'],
               ['same_tone_resp_prev_reward', 'same_tone_resp_prev_unreward', 'diff_tone_resp_prev_reward', 'diff_tone_resp_prev_unreward']]

plot_titles = ['Prior & Current Tone', 'Prior Outcome', 'Prior & Current Tone by Prev Outcome']
gen_title = 'Consecutive Trial Tones by Previous Outcome'
gen_plot_name = 'consec_tones_prev_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)


# consecutive stimuli by prior outcome & tone side
gen_group_labels = {'{}_same_tone_resp': '{} Same', '{}_diff_tone_resp': '{} Diff',
                    '{}_tone_prev_reward': '{} Prev Rew', '{}_tone_prev_unreward': '{} Prev Unrew',
                    '{}_same_tone_resp_prev_reward': '{} Same Prev Rew', '{}_same_tone_resp_prev_unreward': '{} Same Prev Unrew',
                    '{}_diff_tone_resp_prev_reward': '{} Diff Prev Rew', '{}_diff_tone_resp_prev_unreward': '{} Diff Prev Unrew'}

group_labels = {k.format(tone): v.format(tone.capitalize())
                for k,v in gen_group_labels.items()
                for tone in rel_sides}

gen_plot_groups = [['{}_same_tone_resp', '{}_diff_tone_resp'], ['{}_tone_prev_reward', '{}_tone_prev_unreward']]

plot_groups = [[g.format(tone) for tone in rel_sides for g in gs] for gs in gen_plot_groups]

gen_plot_groups = ['{}_same_tone_resp_prev_reward', '{}_same_tone_resp_prev_unreward', '{}_diff_tone_resp_prev_reward', '{}_diff_tone_resp_prev_unreward']

plot_groups.extend([[g.format(tone) for g in gen_plot_groups] for tone in rel_sides])

plot_titles = ['Prior & Current Tone', 'Prior Outcome', 'Prior & Current Contra Tone by Prev Outcome', 'Prior & Current Ipsi Tone by Prev Outcome']
gen_title = 'Consecutive Trial Tones by Relative Side and Previous Outcome'
gen_plot_name = 'consec_tones_side_prev_outcome'

aligns = [Align.tone, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task, include_norm_reward=False)

# next trial choice by consecutive stimuli and outcome
group_labels = {'next_same_tone_stay_reward': 'Next Same Stay', 'next_diff_tone_stay_reward': 'Next Diff Stay',
                'next_same_tone_switch_reward': 'Next Same Switch', 'next_diff_tone_switch_reward': 'Next Diff Switch',
                'next_same_tone_stay_unreward': 'Next Same Stay', 'next_diff_tone_stay_unreward': 'Next Diff Stay',
                'next_same_tone_switch_unreward': 'Next Same Switch', 'next_diff_tone_switch_unreward': 'Next Diff Switch'}

plot_groups = [['next_same_tone_stay_reward', 'next_diff_tone_stay_reward', 'next_same_tone_switch_reward', 'next_diff_tone_switch_reward'],
               ['next_same_tone_stay_unreward', 'next_diff_tone_stay_unreward', 'next_same_tone_switch_unreward', 'next_diff_tone_switch_unreward']]

plot_titles = ['Rewarded', 'Unrewarded']
gen_title = 'Future Tone & Choice by Current Outcome'
gen_plot_name = 'consec_tones_prev_future_outcome'

aligns = [Align.reward, Align.cue_reward]

plot_avg_signals(plot_groups, group_labels, plot_titles, gen_title, aligns, gen_plot_name=gen_plot_name, tasks=task)


# %% Prepare for event aligned regressions

fit_signals = ['z_dff_iso_baseline_fband']
regress_n_back = 3

# build predictor tables for subjects and regions to be used across regressions
all_regress_predictors = {task: {subjid: {} for subjid in subj_ids} for task in tasks}

for task in tasks:
    if task == 'wm':
        sess_ids = wm_sess_ids
        sess_data = wm_sess_data
    else:
        sess_ids = bandit_sess_ids
        sess_data = bandit_sess_data

    for subj_id in subj_ids:
        if not subj_id in sess_ids:
            continue

        subj_regions = get_subj_regions(subj_id)

        subj_data = sess_data[sess_data['subjid'] == subj_id].copy()
        bah.calc_trial_hist(subj_data, n_back=regress_n_back)

        # compute min/max lick durs for across subject scaling

        rewarded_trials = subj_data[subj_data['rewarded'] == True]
        unrewarded_trials = subj_data[(subj_data['rewarded'] == False) & ~np.isnan(subj_data['response_time'])]

        # Get the 95% quantile min/max to reduce outlier effects
        q = 0.025
        rew_lick_dur_min = rewarded_trials['lick_dur'].quantile(q/2)
        rew_lick_dur_max = rewarded_trials['lick_dur'].quantile(1-q/2)
        unrew_lick_dur_min = unrewarded_trials['lick_dur'].quantile(q/2)
        unrew_lick_dur_max = unrewarded_trials['lick_dur'].quantile(1-q/2)

        # compute min/max cue to response times (RT)
        rt_min = subj_data['RT'].quantile(q/2)
        rt_max = subj_data['RT'].quantile(1-q/2)

        # compute min/max response delays for WM task
        if task == 'wm':
            resp_delay_min = subj_data['resp_delay'].min()
            resp_delay_max = subj_data['resp_delay'].max()
        else:
            resp_delay_min = subj_data['trial_length'].min()
            resp_delay_max = subj_data['trial_length'].max()

        for sess_id in sess_ids[subj_id]:
            all_regress_predictors[task][subj_id][sess_id] = {}

            trial_data = subj_data[subj_data['sessid'] == sess_id]

            trial_started = trial_data['trial_started']

            # create various trial selection criteria
            if task == 'wm':
                bails = (trial_data['bail'] == True).to_numpy().astype(int)
                buffer = np.full(regress_n_back, np.nan)
                buff_bails = np.concatenate((buffer, bails))
                bail_hist = [buffer]

                for i in range(len(bails)-1):
                    bail_hist.append(np.flip(buff_bails[i+1:i+regress_n_back+1]))

                bail_hist = np.stack(bail_hist)

                tones = trial_data['relevant_tone_info'].to_numpy()
                tone_high = (tones == 'high').astype(int)
                tone_low = (tones == 'low').astype(int)

                prev_tones = trial_data['prev_tone_info'].apply(lambda x: x[0] if utils.is_list(x) else x).to_numpy()

                prev_tone_same = (tones == prev_tones).astype(int)
                prev_tone_diff = ((tones != prev_tones) & (prev_tones != None)).astype(int)

                delay_norm = (trial_data['resp_delay'].to_numpy() - resp_delay_min)/(resp_delay_max - resp_delay_min)

                responded = ((trial_data['bail'] == False) & trial_started).to_numpy()
            else:
                delay_norm = (trial_data['trial_length'].to_numpy() - resp_delay_min)/(resp_delay_max - resp_delay_min)
                responded = ~np.isnan(trial_data['response_time']).to_numpy()

            reward_sel = (trial_data['rewarded'] == True).to_numpy()
            unreward_sel = (trial_data['rewarded'] == False).to_numpy() & responded
            rewarded = reward_sel.astype(int)
            unrewarded = unreward_sel.astype(int)

            choices = trial_data['choice'].to_numpy()
            left_choices = (choices == 'left').astype(int)
            right_choices = (choices == 'right').astype(int)

            rew_hist = np.stack(trial_data['rew_hist'].to_numpy())
            choice_hist = np.stack(trial_data['choice_hist'].to_numpy())

            prev_resp = np.insert(responded[:-1], 0, False)
            next_resp = np.append(responded[1:], False)

            switch = (trial_data['switch'].to_numpy() & prev_resp & responded).astype(int)
            stay = (trial_data['stay'].to_numpy() & prev_resp & responded).astype(int)
            next_switch = (trial_data['next_switch'].to_numpy() & responded & next_resp).astype(int)
            next_stay = (trial_data['next_stay'].to_numpy() & responded & next_resp).astype(int)

            lick_dur_norm = np.full(len(trial_data), np.nan)
            lick_dur_norm[reward_sel] = ((trial_data.loc[reward_sel,'lick_dur'] - rew_lick_dur_min)/(rew_lick_dur_max - rew_lick_dur_min)).clip(0,1).to_numpy()
            lick_dur_norm[unreward_sel] = ((trial_data.loc[unreward_sel,'lick_dur'] - unrew_lick_dur_min)/(unrew_lick_dur_max - unrew_lick_dur_min)).clip(0,1).to_numpy()

            rt_norm = (trial_data['RT'].to_numpy() - rt_min)/(rt_max - rt_min)

            for region in subj_regions:

                region_side = implant_info[subj_id][region]['side']

                if region_side == 'left':
                    ipsi_choices = left_choices
                    contra_choices = right_choices
                    ipsi_hist = np.where(pd.isna(choice_hist), np.nan, (choice_hist == 'left').astype(float))
                    contra_hist = np.where(pd.isna(choice_hist), np.nan, (choice_hist == 'right').astype(float))
                else:
                    ipsi_choices = right_choices
                    contra_choices = left_choices
                    ipsi_hist = np.where(pd.isna(choice_hist), np.nan, (choice_hist == 'right').astype(float))
                    contra_hist = np.where(pd.isna(choice_hist), np.nan, (choice_hist == 'left').astype(float))

                # add all predictors and interactions to a dataframe
                preds = {'Intercept': np.ones_like(rewarded), 'Reward': rewarded, 'Unreward': unrewarded,
                         'Ipsi Choice': ipsi_choices, 'Contra Choice': contra_choices,
                         'Ipsi Reward': rewarded*ipsi_choices, 'Contra Reward': rewarded*contra_choices,
                         'Ipsi Unreward': unrewarded*ipsi_choices, 'Contra Unreward': unrewarded*contra_choices,
                         'Stay': stay, 'Switch': switch, 'Next Stay': next_stay, 'Next Switch': next_switch,
                         'Lick Dur': lick_dur_norm, 'RT': rt_norm, 'Resp Delay': delay_norm}

                preds.update({'Reward ({})'.format(-i-1): rew_hist[:,i] for i in range(regress_n_back)})
                preds.update({'Unreward ({})'.format(-i-1): 1-rew_hist[:,i] for i in range(regress_n_back)})
                preds.update({'Ipsi ({})'.format(-i-1): ipsi_hist[:,i] for i in range(regress_n_back)})
                preds.update({'Contra ({})'.format(-i-1): contra_hist[:,i] for i in range(regress_n_back)})
                preds.update({'Ipsi Reward ({})'.format(-i-1): ipsi_hist[:,i]*rew_hist[:,i] for i in range(regress_n_back)})
                preds.update({'Ipsi Unreward ({})'.format(-i-1): ipsi_hist[:,i]*(1-rew_hist[:,i]) for i in range(regress_n_back)})
                preds.update({'Contra Reward ({})'.format(-i-1): contra_hist[:,i]*rew_hist[:,i] for i in range(regress_n_back)})
                preds.update({'Contra Unreward ({})'.format(-i-1): contra_hist[:,i]*(1-rew_hist[:,i]) for i in range(regress_n_back)})

                if task == 'wm':
                    preds.update({'Bail': bails, 'High Tone': tone_high, 'Low Tone': tone_low,
                                  'Prev Same Tone': prev_tone_same, 'Prev Diff Tone': prev_tone_diff})
                    preds.update({'Bail ({})'.format(-i-1): bail_hist[:,i] for i in range(regress_n_back)})

                all_regress_predictors[task][subj_id][sess_id][region] = pd.DataFrame(preds)

# make history preditor labels for ease of use
rew_hist_labels = ['Reward ({})'.format(-i-1)for i in range(regress_n_back)]
unrew_hist_labels = ['Unreward ({})'.format(-i-1) for i in range(regress_n_back)]
ipsi_hist_labels = ['Ipsi ({})'.format(-i-1) for i in range(regress_n_back)]
contra_hist_labels = ['Contra ({})'.format(-i-1) for i in range(regress_n_back)]
ipsi_rew_hist_labels = ['Ipsi Reward ({})'.format(-i-1) for i in range(regress_n_back)]
ipsi_unrew_hist_labels = ['Ipsi Unreward ({})'.format(-i-1) for i in range(regress_n_back)]
contra_rew_hist_labels = ['Contra Reward ({})'.format(-i-1) for i in range(regress_n_back)]
contra_rew_hist_labels = ['Contra Unreward ({})'.format(-i-1) for i in range(regress_n_back)]
bail_hist_labels = ['Bail ({})'.format(-i-1) for i in range(regress_n_back)]

def select_regress_data(subj_id, region, task, align, predictors, fit_signal, trial_sel_key='trial_started',
                        sub_sess_ids=None, norm_signals=True, norm_trial_sel_key=None, norm_align=None, norm_t_range=[-0.1,0]):

    stacked_predictors = []
    stacked_signals = []

    if norm_trial_sel_key is None:
        norm_trial_sel_key = trial_sel_key

    if norm_align is None:
        norm_align = align

    if subj_id != 'all':
        subj_regions = get_subj_regions(subj_id)
        reg_mapping = get_subj_region_mapping(subj_regions)

        t = aligned_time[task]['t'][norm_align][reg_mapping[region]]
        norm_t_sel = (t >= norm_t_range[0]) & (t <= norm_t_range[1])

        sess_ids = list(all_regress_predictors[task][subj_id].keys())

        if not sub_sess_ids is None:
            sess_ids = [s for s in sess_ids if s in sub_sess_ids]

        for sess_id in sess_ids:

            # some sessions have different regions
            if not region in aligned_signals[task][subj_id][sess_id][fit_signal][align]:
                continue

            # get trial selections
            trial_sel = (align_trial_selections[task][subj_id][sess_id][region][align] &
                         align_trial_selections[task][subj_id][sess_id][region][trial_sel_key])

            # get aligned data matrix
            aligned_mat = aligned_signals[task][subj_id][sess_id][fit_signal][align][region].copy()
            if norm_signals:
                norm_trial_sel = align_trial_selections[task][subj_id][sess_id][region][norm_trial_sel_key]
                norm_aligned_mat = aligned_signals[task][subj_id][sess_id][fit_signal][norm_align][region]
                aligned_mat -= np.nanmean(norm_aligned_mat[np.ix_(norm_trial_sel, norm_t_sel)])

            stacked_signals.append(aligned_mat[trial_sel, :])

            # get predictors
            preds = {}
            for pred in predictors:
                if '&' in pred:
                    pred_list = [p.strip() for p in pred.split('&')]
                    preds[pred] = all_regress_predictors[task][subj_id][sess_id][region][pred_list].prod(axis=1)
                else:
                    preds[pred] = all_regress_predictors[task][subj_id][sess_id][region][pred].copy()

            preds = pd.DataFrame(preds)
            preds['subjid'] = subj_id
            stacked_predictors.append(preds[trial_sel])

    else:
        for subj in subj_ids:
            sess_ids = list(all_regress_predictors[task][subj].keys())
            subj_regions = get_subj_regions(subj)
            reg_mapping = get_gen_reg_subj_regions(subj_regions)

            t = aligned_time[task]['t'][norm_align][region]
            norm_t_sel = (t >= norm_t_range[0]) & (t <= norm_t_range[1])

            if not sub_sess_ids is None:
                sess_ids = [s for s in sess_ids if s in sub_sess_ids]

            for sess_id in sess_ids:
                # handle bilateral regions
                for r in reg_mapping[region]:

                    # some sessions have different regions
                    if not r in aligned_signals[task][subj][sess_id][fit_signal][align]:
                        continue

                    # get trial selections
                    trial_sel = (align_trial_selections[task][subj][sess_id][r][align] &
                                 align_trial_selections[task][subj][sess_id][r][trial_sel_key])

                    # get aligned data matrix
                    aligned_mat = aligned_signals[task][subj][sess_id][fit_signal][align][r].copy()
                    if norm_signals:
                        norm_trial_sel = align_trial_selections[task][subj][sess_id][r][norm_trial_sel_key]
                        norm_aligned_mat = aligned_signals[task][subj][sess_id][fit_signal][norm_align][r].copy()
                        # with warnings.catch_warnings():
                        #     warnings.simplefilter("error", RuntimeWarning)
                        #     try:
                        aligned_mat -= np.nanmean(norm_aligned_mat[np.ix_(norm_trial_sel, norm_t_sel)])
                            # except:
                            #     e=1

                    stacked_signals.append(aligned_mat[trial_sel, :])

                    # get predictors
                    preds = {}
                    for pred in predictors:
                        if '&' in pred:
                            pred_list = [p.strip() for p in pred.split('&')]
                            preds[pred] = all_regress_predictors[task][subj][sess_id][r][pred_list].prod(axis=1, skipna=False)
                        else:
                            preds[pred] = all_regress_predictors[task][subj][sess_id][r][pred].copy()

                    preds = pd.DataFrame(preds)
                    preds['subjid'] = subj
                    stacked_predictors.append(preds[trial_sel])

    stacked_signals = np.vstack(stacked_signals)
    stacked_predictors = pd.concat(stacked_predictors)

    return stacked_signals, stacked_predictors


def run_regress_over_time(tasks, aligns, predictors, trial_sel_key='trial_started', sub_sess_ids=None, fit_ind_subj=True,
                         add_intercept=False, norm_signals=True, norm_trial_sel_key=None, norm_align=None, norm_t_range=[-0.1,0]):

    if not utils.is_list(aligns):
        aligns = [aligns]

    if not utils.is_list(tasks):
        tasks = [tasks]

    fit_subj = ['all']
    if fit_ind_subj:
        fit_subj.extend(subj_ids)

    fit_results = {s: {t: {a: {d: {} for d in fit_signals} for a in aligns} for t in tasks} for s in fit_subj}

    for subj_id in fit_subj:
        if subj_id == 'all':
            subj_regions = all_regions
        else:
            subj_regions = get_subj_regions(subj_id)

        for task in tasks:
            for align in aligns:

                if task != 'wm' and align == Align.tone:
                    continue

                for fit_signal in fit_signals:
                    print('Regressing subject {} in {} task at {}...'.format(subj_id, task, align))
                    for region in subj_regions:
                        start = time.perf_counter()
                        signals, preds = select_regress_data(subj_id, region, task, align, predictors, fit_signal,
                                                             trial_sel_key=trial_sel_key, sub_sess_ids=sub_sess_ids,
                                                             norm_signals=norm_signals, norm_trial_sel_key=norm_trial_sel_key,
                                                             norm_align=norm_align, norm_t_range=norm_t_range)

                        regress_results = fp_utils.regress_over_time(signals, preds, group_col='subjid', add_intercept=add_intercept)

                        fit_results[subj_id][task][align][fit_signal][region] = regress_results

                        print('   Completed region {} in {:.1f}s'.format(region, time.perf_counter()-start))

    return fit_results

def plot_regress_over_time(fit_results, predictor_groups, group_titles, gen_title, predictor_labels=None,
                           plot_error='ci', plot_sig=True, sig_lvl=0.05, sig_y_dist=0.03, sig_correction='bonferroni', # 'bonferroni' 'holm'
                           xlims_dict=None, group_colors=None, gen_plot_name=None, include_norm_reward=True,
                           dec=1, x_inc_dict=None, y_inc_dict=None, ph=3.5, pw=5, legend_params=None, plot_ind_subj=True):

    if len(predictor_groups) == 0:
        return

    # make predictor groups be nested 3 levels
    while utils.get_list_depth(predictor_groups) < 3:
        predictor_groups = [predictor_groups]

    while utils.get_list_depth(group_titles) < 2:
        group_titles = [group_titles]

    if predictor_labels is None:
        predictor_labels = {}

    if xlims_dict is None:
        xlims_dict = all_xlims

    if plot_ind_subj:
        plot_subj = list(fit_results.keys())
    else:
        plot_subj = ['all']

    for subj_id in plot_subj:
        for task in fit_results[subj_id].keys():
            for align in fit_results[subj_id][task].keys():

                if task != 'wm' and align == Align.tone:
                    continue

                align_title = fpah.get_align_title(align)
                x_label = fpah.get_align_xlabel(align)

                align_xlims = xlims_dict.get(align, all_xlims[align])

                t = aligned_time[task]['t'][align]

                x_ticks = None
                plot_x0 = True
                dashlines = None

                if align == Align.tone:
                    dashlines = tone_dashlines
                elif align in [Align.cport_on_cpoke_in, Align.cue_reward]:
                    dashlines = norm_dashlines[task][align]
                    x_ticks = {'ticks': dashlines, 'labels': norm_labels[align]}
                    plot_x0 = False

                    if align == Align.cue_reward and not include_norm_reward:
                        align_xlims = {r: [0, dashlines[-1]] for r in default_xlims.keys()}

                for signal_type in fit_signals:
                    _, y_label = fpah.get_signal_type_labels(signal_type)

                    plot_results = fit_results[subj_id][task][align][signal_type]

                    plot_regions = sort_subj_regions(plot_results.keys())

                    reg_mapping = get_subj_region_mapping(plot_regions)
                    plot_t = {r: t[reg_mapping[r]] for r in plot_regions}
                    plot_xlims = None if align_xlims is None else {r: align_xlims[reg_mapping[r]] for r in plot_regions}

                    for fig_pred_groups, fig_group_titles in zip(predictor_groups, group_titles):

                        n_rows = len(plot_regions)
                        n_cols = len(fig_pred_groups)

                        fig, axs = plt.subplots(n_rows, n_cols, figsize=(pw*n_cols, ph*n_rows), layout='constrained', sharey='row')

                        axs = axs.reshape(n_rows, n_cols)

                        title = '{} Aligned to {} - Subj {}, {}'.format(gen_title, align_title, subj_id, beh_names[task])
                        fig.suptitle(title)

                        for i, r in enumerate(plot_regions):
                            region_results = plot_results[r]

                            if plot_xlims is None:
                                t_sel = np.full(plot_t[r].shape, True)
                            else:
                                t_sel = (plot_t[r] >= plot_xlims[r][0]) & (plot_t[r] <= plot_xlims[r][1])

                            if plot_sig:
                                p_vals_df = pd.DataFrame(region_results['p_vals'])
                                sig = p_vals_df.apply(lambda x: smm.multipletests(x, alpha=sig_lvl, method=sig_correction)[0], axis=1, result_type='broadcast').astype(bool)

                            y_max = -np.inf
                            y_min = np.inf
                            all_line_colors = []
                            for j, (preds, group_title) in enumerate(zip(fig_pred_groups, fig_group_titles)):
                                ax = axs[i,j]

                                if subj_id == 'all':
                                    region_label = r
                                else:
                                    region_label = '{} ({})'.format(r, implant_info[subj_id][r]['side'])

                                ax.set_title('{}, {}'.format(group_title, region_label))

                                group_line_colors = []
                                for k, pred in enumerate(preds):
                                    vals = region_results['params'][pred]
                                    if plot_error == 'ci':
                                        error = np.abs(np.vstack((region_results['ci_lower'][pred], region_results['ci_upper'][pred])) - vals[None,:])
                                        error = error[:,t_sel][:,::dec]
                                    else:
                                        error = region_results['se'][pred]
                                        error = error[t_sel][::dec]

                                    if group_colors is None:
                                        color = None
                                    elif type(group_colors) is dict:
                                        color = group_colors.get(pred, None)
                                    else:
                                        color = group_colors[k]

                                    pred_label = predictor_labels.get(pred, pred)

                                    line, _ = plot_utils.plot_psth(plot_t[r][t_sel][::dec], vals[t_sel][::dec], error, ax=ax, label=pred_label, plot_x0=False, color=color)

                                    group_line_colors.append(line.get_color())

                                all_line_colors.append(group_line_colors)

                                if not dashlines is None:
                                    plot_utils.plot_dashlines(dashlines, ax=ax)

                                if plot_x0:
                                    plot_utils.plot_dashlines(0, dir='v', ax=ax)
                                plot_utils.plot_dashlines(0, dir='h', ax=ax)

                                if j == 0:
                                    ax.set_ylabel(y_label)
                                else:
                                    ax.yaxis.set_tick_params(which='both', labelleft=True)
                                ax.set_xlabel(x_label)

                                if not x_ticks is None:
                                    ax.set_xticks(x_ticks['ticks'])
                                    ax.set_xticklabels(x_ticks['labels'])
                                elif not x_inc_dict is None:
                                    ax.xaxis.set_major_locator(MultipleLocator(x_inc_dict[reg_mapping[r]]))

                                if not y_inc_dict is None:
                                    ax.yaxis.set_major_locator(MultipleLocator(y_inc_dict[reg_mapping[r]]))

                                if not legend_params is None:
                                    if utils.is_list(legend_params) and not legend_params[j] is None:
                                        ax.legend(**legend_params[j])
                                    else:
                                        ax.legend(**legend_params)
                                else:
                                    ax.legend(loc='best')

                                ax_y_min, ax_y_max = ax.get_ylim()
                                if ax_y_min < y_min:
                                    y_min = ax_y_min
                                if ax_y_max > y_max:
                                    y_max = ax_y_max

                            # plot significance from 0
                            if plot_sig:
                                y_offset = (y_max-y_min)*sig_y_dist

                                for j, (preds, group_line_colors) in enumerate(zip(fig_pred_groups, all_line_colors)):
                                    ax = axs[i,j]

                                    for k, (pred, color) in enumerate(zip(preds, group_line_colors)):
                                        sig_t = plot_t[r][sig[pred] & t_sel]
                                        ax.scatter(sig_t, np.full_like(sig_t, y_max+k*y_offset), color=color, marker='.', s=10)

                        if not gen_plot_name is None:
                            save_plot(fig, task, subj_id, 'regress_{}_{}_{}'.format(align, gen_plot_name, signal_type))

                        plt.show()


# %% Event aligned regressions - Reward History Only

fit_ind_subj = False
plot_ind_subj = False
recalculate = False
plot_sig_lvl = 0.05
plot_error = 'se' # 'ci', 'se'
filename = 'regress_rew_hist'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        fit_results = saved_data['fit_results']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    fit_results = {}

rew_hist_all_colors = plt.cm.Greens(np.linspace(1,0.4,regress_n_back+1))
rew_hist_unrew_colors = plt.cm.Reds(np.linspace(1,0.4,regress_n_back+1))
rew_hist_rew_colors = plt.cm.Blues(np.linspace(1,0.4,regress_n_back+1))

plot_lims = {Align.reward: {'DMS': [-0.2,1.5], 'DLS': [-0.2,1.5], 'TS': [-0.2,1.5], 'NAc': [-0.4,3], 'PL': [-1.6,12]}}


# Reward History Only

# Bandit Task
regress_tasks = ['bandit']
key = 'bandit_cue_rew_hist_only'

pred_labels = {'Intercept': 'Avg'}

plot_groups = [['Intercept', *rew_hist_labels]]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

group_colors = {p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[0])}

plot_titles = ['All Choices']
gen_title = 'Cue Reward History'
gen_plot_name = 'rew_hist'

aligns = [Align.cue_reward]

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

key = 'bandit_rew_rew_hist_only'

rew_rew_hist = ['Reward & '+l for l in rew_hist_labels]
unrew_rew_hist = ['Unreward & '+l for l in rew_hist_labels]

pred_labels = {p: l for p,l in zip(rew_rew_hist, rew_hist_labels)}
pred_labels.update({p: l for p,l in zip(unrew_rew_hist, rew_hist_labels)})

plot_groups = [['Reward', *rew_rew_hist], ['Unreward', *unrew_rew_hist]]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

group_colors = {p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1])})

plot_titles = ['Rewarded', 'Unrewarded']
gen_title = 'Trial Outcome & History'
gen_plot_name = 'rew_hist'

aligns = [Align.reward]

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# WM
regress_tasks = ['wm']
key = 'wm_cue_rew_hist_only'

pred_labels = {'Intercept': 'Avg'}

plot_groups = [['Intercept', *rew_hist_labels], ['Intercept', *bail_hist_labels]]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_bail_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

group_colors = {p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[1])})

plot_titles = ['Reward History', 'Bail History']
gen_title = 'Cue Reward History'
gen_plot_name = 'rew_hist'

aligns = [Align.cue_reward]

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

key = 'wm_rew_rew_hist_only'

rew_rew_hist = ['Reward & '+l for l in rew_hist_labels]
rew_bail_hist = ['Reward & '+l for l in bail_hist_labels]
unrew_rew_hist = ['Unreward & '+l for l in rew_hist_labels]
unrew_bail_hist = ['Unreward & '+l for l in bail_hist_labels]

pred_labels = {p: l for p,l in zip(rew_rew_hist, rew_hist_labels)}
pred_labels.update({p: l for p,l in zip(rew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(unrew_bail_hist, bail_hist_labels)})

plot_groups = [['Reward', *rew_rew_hist], ['Reward', *rew_bail_hist], ['Unreward', *unrew_rew_hist], ['Unreward', *unrew_bail_hist]]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_bail_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

group_colors = {p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[1])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[2])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[3])})

plot_titles = ['Rewarded, Reward History', 'Rewarded, Bail History', 'Unrewarded, Reward History', 'Unrewarded, Bail History']
gen_title = 'Trial Outcome & History'
gen_plot_name = 'rew_hist'

aligns = [Align.reward]

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# %% Event aligned regressions - Reward History & all other relevant factors

fit_ind_subj = False
plot_ind_subj = False
recalculate = False
plot_sig_lvl = 0.05
plot_error = 'se' # 'ci', 'se'

filename = 'regress_rew_hist_full'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        fit_results = saved_data['fit_results']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    fit_results = {}

rew_hist_all_colors = plt.cm.Greens(np.linspace(1,0.4,regress_n_back+1))
rew_hist_unrew_colors = plt.cm.Reds(np.linspace(1,0.4,regress_n_back+1))
rew_hist_rew_colors = plt.cm.Blues(np.linspace(1,0.4,regress_n_back+1))

plot_lims = {Align.reward: {'DMS': [-0.2,1.5], 'DLS': [-0.2,1.5], 'TS': [-0.2,1.5], 'NAc': [-0.4,3], 'PL': [-1.6,12]},
             Align.cue: {'DMS': [-0.2,1], 'DLS': [-0.2,1], 'TS': [-0.2,1], 'NAc': [-0.2,1], 'PL': [-0.2,1]},
             Align.cpoke_out: {'DMS': [-0.2,1], 'DLS': [-0.2,1], 'TS': [-0.2,1], 'NAc': [-0.2,1], 'PL': [-0.2,1]}}

## Bandit Task
regress_tasks = ['bandit']

# cue
aligns = [Align.cue_reward, Align.cue, Align.cpoke_out]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

choice_mets = ['Stay', 'Resp Delay'] #'RT', 'Lick Dur'

key = 'bandit_cue_rew_hist_full'

pred_labels = {'Intercept': 'Avg'}

plot_groups = [['Intercept', *rew_hist_labels], ['Intercept', 'Contra Choice', *choice_mets]]

group_colors = {p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[0])}

plot_titles = ['Reward History', 'Choice Metrics']
gen_title = 'Cue Reward History & Choice Metrics'
gen_plot_name = 'rew_hist_full'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# replot with limited regressors
plot_groups = [['Intercept', *rew_hist_labels], ['Contra Choice', 'Stay']]#, 'Resp Delay']]
plot_titles = ['Reward History', 'Choice Metrics']
gen_title = 'Cue Reward History & Choice Metrics'
gen_plot_name = 'rew_hist_full_lim'

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# separate sides
key = 'bandit_cue_rew_hist_full_sep_sides'

contra_rew_hist = ['Contra Choice & '+l for l in rew_hist_labels]
ipsi_rew_hist = ['Ipsi Choice & '+l for l in rew_hist_labels]
contra_choice_mets = ['Contra Choice & '+l for l in choice_mets]
ipsi_choice_mets = ['Ipsi Choice & '+l for l in choice_mets]

pred_labels = {p: l for p,l in zip(contra_rew_hist, rew_hist_labels)}
pred_labels.update({p: l for p,l in zip(ipsi_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_choice_mets, choice_mets)})
pred_labels.update({p: l for p,l in zip(ipsi_choice_mets, choice_mets)})

plot_groups = [['Contra Choice', *contra_rew_hist], ['Ipsi Choice', *ipsi_rew_hist], ['Contra Choice', *contra_choice_mets], ['Ipsi Choice', *ipsi_choice_mets]]

group_colors = {p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[1])})

plot_titles = ['Reward History, Contra Choices', 'Reward History, Ipsi Choices', 'Choice Metrics, Contra Choices', 'Choice Metrics, Ipsi Choices']
gen_title = 'Cue Reward History & Choice Metrics by Choice Side'
gen_plot_name = 'rew_hist_full_sep_sides'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# reward

aligns = [Align.reward]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

choice_mets = ['Stay', 'Next Stay', 'Resp Delay', 'Lick Dur'] #'RT',

key = 'bandit_rew_rew_hist_full'

sub_choice_mets = ['Contra Choice', *choice_mets]

rew_rew_hist = ['Reward & '+l for l in rew_hist_labels]
unrew_rew_hist = ['Unreward & '+l for l in rew_hist_labels]
rew_choice_mets = ['Reward & '+l for l in sub_choice_mets]
unrew_choice_mets = ['Unreward & '+l for l in sub_choice_mets]

pred_labels = {p: l for p,l in zip(rew_rew_hist, rew_hist_labels)}
pred_labels.update({p: l for p,l in zip(unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(rew_choice_mets, sub_choice_mets)})
pred_labels.update({p: l for p,l in zip(unrew_choice_mets, sub_choice_mets)})

plot_groups = [['Reward', *rew_rew_hist], ['Reward', *rew_choice_mets], ['Unreward', *unrew_rew_hist], ['Unreward', *unrew_choice_mets]]

group_colors = {p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[2])})

plot_titles = ['Reward History, Rewarded Trials', 'Choice Metrics, Rewarded Trials',
               'Reward History, Unrewarded Trials', 'Choice Metrics, Unrewarded Trials']
gen_title = 'Reward History & Choice Metrics by Trial Outcome'
gen_plot_name = 'rew_hist_full'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# replot with limited regressors
lim_choice_mets = ['Stay', 'Lick Dur'] #'Next Stay', 'Contra Choice',
lim_rew_choice_mets = ['Reward & '+l for l in lim_choice_mets]
lim_unrew_choice_mets = ['Unreward & '+l for l in lim_choice_mets]
plot_groups = [['Reward', *rew_rew_hist], [*lim_rew_choice_mets], ['Unreward', *unrew_rew_hist], [*lim_unrew_choice_mets]]

plot_titles = ['Reward History, Rewarded Trials', 'Choice Metrics, Rewarded Trials',
               'Reward History, Unrewarded Trials', 'Choice Metrics, Unrewarded Trials']
gen_title = 'Reward History & Choice Metrics by Trial Outcome'
gen_plot_name = 'rew_hist_full_lim'

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels, plot_sig=False,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# separate sides
key = 'bandit_rew_rew_hist_full_sep_sides'

contra_rew_rew_hist = ['Contra Choice & Reward & '+l for l in rew_hist_labels]
contra_unrew_rew_hist = ['Contra Choice & Unreward & '+l for l in rew_hist_labels]
contra_rew_choice_mets = ['Contra Choice & Reward & '+l for l in choice_mets]
contra_unrew_choice_mets = ['Contra Choice & Unreward & '+l for l in choice_mets]

ipsi_rew_rew_hist = ['Ipsi Choice & Reward & '+l for l in rew_hist_labels]
ipsi_unrew_rew_hist = ['Ipsi Choice & Unreward & '+l for l in rew_hist_labels]
ipsi_rew_choice_mets = ['Ipsi Choice & Reward & '+l for l in choice_mets]
ipsi_unrew_choice_mets = ['Ipsi Choice & Unreward & '+l for l in choice_mets]

pred_labels = {'Contra Choice & Reward': 'Rewarded Contra Choice', 'Contra Choice & Unreward': 'Unrewarded Contra Choice',
               'Ipsi Choice & Reward': 'Rewarded Ipsi Choice', 'Ipsi Choice & Unreward': 'Unrewarded Ipsi Choice',}
pred_labels.update({p: l for p,l in zip(contra_rew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_rew_choice_mets, choice_mets)})
pred_labels.update({p: l for p,l in zip(contra_unrew_choice_mets, choice_mets)})

pred_labels.update({p: l for p,l in zip(ipsi_rew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_rew_choice_mets, choice_mets)})
pred_labels.update({p: l for p,l in zip(ipsi_unrew_choice_mets, choice_mets)})

plot_groups = [[['Contra Choice & Reward', *contra_rew_rew_hist], ['Ipsi Choice & Reward', *ipsi_rew_rew_hist],
                ['Contra Choice & Reward', *contra_rew_choice_mets], ['Ipsi Choice & Reward', *ipsi_rew_choice_mets]],
               [['Contra Choice & Unreward', *contra_unrew_rew_hist], ['Ipsi Choice & Unreward', *ipsi_unrew_rew_hist],
                ['Contra Choice & Unreward', *contra_unrew_choice_mets], ['Ipsi Choice & Unreward', *ipsi_unrew_choice_mets]]]

group_colors = {p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0][0])}
group_colors.update({p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0][1])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1][0])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1][1])})

plot_titles = [['Reward History, Rewarded Contra Choices', 'Reward History, Rewarded Ipsi Choices',
                'Choice Metrics, Rewarded Contra Choices', 'Choice Metrics, Rewarded Ipsi Choices'],
               ['Reward History, Unrewarded Contra Choices', 'Reward History, Unrewarded Ipsi Choices',
                'Choice Metrics, Unrewarded Contra Choices', 'Choice Metrics, Unrewarded Ipsi Choices']]
gen_title = 'Reward History & Choice Metrics by Choice Side and Trial Outcome'
gen_plot_name = 'rew_hist_full_sep_sides'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

## WM Task

regress_tasks = ['wm']

# cue
aligns = [Align.cue_reward, Align.cue, Align.cpoke_out]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_bail_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

choice_mets = ['Stay', 'Resp Delay'] #'RT', 'Lick Dur'

key = 'wm_cue_rew_hist_full'

pred_labels = {'Intercept': 'Avg'}

plot_groups = [['Intercept', *rew_hist_labels], ['Intercept', *bail_hist_labels], ['Intercept', 'Contra Choice', *choice_mets]]

group_colors = {p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[1])})

plot_titles = ['Reward History', 'Bail History', 'Choice Metrics']
gen_title = 'Cue Reward History & Choice Metrics'
gen_plot_name = 'rew_hist_full'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# # replot with limited regressors
# plot_groups = [['Intercept', *rew_hist_labels], ['Intercept', *bail_hist_labels], ['Intercept', 'Contra Choice', 'Stay', 'Resp Delay',]]

# plot_titles = ['Reward History', 'Bail History', 'Choice Metrics']
# gen_title = 'Cue Reward History & Choice Metrics'
# gen_plot_name = 'rew_hist_full_lim'

# plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
#                        sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
#                        group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# separate sides
key = 'wm_cue_rew_hist_full_sep_sides'

contra_rew_hist = ['Contra Choice & '+l for l in rew_hist_labels]
ipsi_rew_hist = ['Ipsi Choice & '+l for l in rew_hist_labels]
contra_bail_hist = ['Contra Choice & '+l for l in bail_hist_labels]
ipsi_bail_hist = ['Ipsi Choice & '+l for l in bail_hist_labels]
contra_choice_mets = ['Contra Choice & '+l for l in choice_mets]
ipsi_choice_mets = ['Ipsi Choice & '+l for l in choice_mets]

pred_labels = {p: l for p,l in zip(contra_rew_hist, rew_hist_labels)}
pred_labels.update({p: l for p,l in zip(ipsi_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_choice_mets, choice_mets)})
pred_labels.update({p: l for p,l in zip(ipsi_choice_mets, choice_mets)})

plot_groups = [['Contra Choice', *contra_rew_hist], ['Ipsi Choice', *ipsi_rew_hist],
               ['Contra Choice', *contra_bail_hist], ['Ipsi Choice', *ipsi_bail_hist],
               ['Contra Choice', *contra_choice_mets], ['Ipsi Choice', *ipsi_choice_mets]]

group_colors = {p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[1])})
group_colors.update({p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[2])})
group_colors.update({p: rew_hist_all_colors[i] for i,p in enumerate(plot_groups[3])})

plot_titles = ['Reward History, Contra Choices', 'Reward History, Ipsi Choices',
               'Bail History, Contra Choices', 'Bail History, Ipsi Choices',
               'Choice Metrics, Contra Choices', 'Choice Metrics, Ipsi Choices']
gen_title = 'Cue Reward History & Choice Metrics by Choice Side'
gen_plot_name = 'rew_hist_full_sep_sides'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# reward

aligns = [Align.reward]
trial_sel_key = 'response'
norm_trial_sel_key = 'rew_hist_bail_'+rew_hist_bin_strs[rew_hist_bins[0]]+'_resp'
norm_align = Align.cue

choice_mets = ['Stay', 'Next Stay', 'Resp Delay', 'Lick Dur'] #'RT',

key = 'wm_rew_rew_hist_full'

sub_choice_mets = ['Contra Choice', *choice_mets]

rew_rew_hist = ['Reward & '+l for l in rew_hist_labels]
unrew_rew_hist = ['Unreward & '+l for l in rew_hist_labels]
rew_bail_hist = ['Reward & '+l for l in bail_hist_labels]
unrew_bail_hist = ['Unreward & '+l for l in bail_hist_labels]
rew_choice_mets = ['Reward & '+l for l in sub_choice_mets]
unrew_choice_mets = ['Unreward & '+l for l in sub_choice_mets]

pred_labels = {p: l for p,l in zip(rew_rew_hist, rew_hist_labels)}
pred_labels.update({p: l for p,l in zip(unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(rew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(unrew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(rew_choice_mets, sub_choice_mets)})
pred_labels.update({p: l for p,l in zip(unrew_choice_mets, sub_choice_mets)})

plot_groups = [['Reward', *rew_rew_hist], ['Reward', *rew_bail_hist], ['Reward', *rew_choice_mets],
               ['Unreward', *unrew_rew_hist], ['Unreward', *unrew_bail_hist], ['Unreward', *unrew_choice_mets]]

group_colors = {p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0])}
group_colors.update({p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[1])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[3])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[4])})

plot_titles = ['Reward History, Rewarded Trials', 'Bail History, Rewarded Trials', 'Choice Metrics, Rewarded Trials',
               'Reward History, Unrewarded Trials', 'Bail History, Unrewarded Trials', 'Choice Metrics, Unrewarded Trials']
gen_title = 'Reward History & Choice Metrics by Trial Outcome'
gen_plot_name = 'rew_hist_full'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# replot with limited regressors
lim_choice_mets = ['Contra Choice', 'Stay', 'Next Stay', 'Lick Dur']
lim_rew_choice_mets = ['Reward & '+l for l in lim_choice_mets]
lim_unrew_choice_mets = ['Unreward & '+l for l in lim_choice_mets]
plot_groups = [['Reward', *rew_rew_hist], ['Reward', *rew_bail_hist], ['Reward', *lim_rew_choice_mets],
               ['Unreward', *unrew_rew_hist], ['Unreward', *unrew_bail_hist], ['Unreward', *lim_unrew_choice_mets]]

plot_titles = ['Reward History, Rewarded Trials', 'Bail History, Rewarded Trials', 'Choice Metrics, Rewarded Trials',
               'Reward History, Unrewarded Trials', 'Bail History, Unrewarded Trials', 'Choice Metrics, Unrewarded Trials']
gen_title = 'Reward History & Choice Metrics by Trial Outcome'
gen_plot_name = 'rew_hist_full_lim'

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims, include_norm_reward=False,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)

# separate sides
key = 'wm_rew_rew_hist_full_sep_sides'

contra_rew_rew_hist = ['Contra Choice & Reward & '+l for l in rew_hist_labels]
contra_unrew_rew_hist = ['Contra Choice & Unreward & '+l for l in rew_hist_labels]
contra_rew_bail_hist = ['Contra Choice & Reward & '+l for l in bail_hist_labels]
contra_unrew_bail_hist = ['Contra Choice & Unreward & '+l for l in bail_hist_labels]
contra_rew_choice_mets = ['Contra Choice & Reward & '+l for l in choice_mets]
contra_unrew_choice_mets = ['Contra Choice & Unreward & '+l for l in choice_mets]

ipsi_rew_rew_hist = ['Ipsi Choice & Reward & '+l for l in rew_hist_labels]
ipsi_unrew_rew_hist = ['Ipsi Choice & Unreward & '+l for l in rew_hist_labels]
ipsi_rew_bail_hist = ['Ipsi Choice & Reward & '+l for l in bail_hist_labels]
ipsi_unrew_bail_hist = ['Ipsi Choice & Unreward & '+l for l in bail_hist_labels]
ipsi_rew_choice_mets = ['Ipsi Choice & Reward & '+l for l in choice_mets]
ipsi_unrew_choice_mets = ['Ipsi Choice & Unreward & '+l for l in choice_mets]

pred_labels = {'Contra Choice & Reward': 'Rewarded Contra Choice', 'Contra Choice & Unreward': 'Unrewarded Contra Choice',
               'Ipsi Choice & Reward': 'Rewarded Ipsi Choice', 'Ipsi Choice & Unreward': 'Unrewarded Ipsi Choice',}
pred_labels.update({p: l for p,l in zip(contra_rew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_rew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_unrew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(contra_rew_choice_mets, choice_mets)})
pred_labels.update({p: l for p,l in zip(contra_unrew_choice_mets, choice_mets)})

pred_labels.update({p: l for p,l in zip(ipsi_rew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_unrew_rew_hist, rew_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_rew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_unrew_bail_hist, bail_hist_labels)})
pred_labels.update({p: l for p,l in zip(ipsi_rew_choice_mets, choice_mets)})
pred_labels.update({p: l for p,l in zip(ipsi_unrew_choice_mets, choice_mets)})

plot_groups = [[['Contra Choice & Reward', *contra_rew_rew_hist], ['Ipsi Choice & Reward', *ipsi_rew_rew_hist],
                ['Contra Choice & Reward', *contra_rew_bail_hist], ['Ipsi Choice & Reward', *ipsi_rew_bail_hist],
                ['Contra Choice & Reward', *contra_rew_choice_mets], ['Ipsi Choice & Reward', *ipsi_rew_choice_mets]],
               [['Contra Choice & Unreward', *contra_unrew_rew_hist], ['Ipsi Choice & Unreward', *ipsi_unrew_rew_hist],
                ['Contra Choice & Unreward', *contra_unrew_bail_hist], ['Ipsi Choice & Unreward', *ipsi_unrew_bail_hist],
                ['Contra Choice & Unreward', *contra_unrew_choice_mets], ['Ipsi Choice & Unreward', *ipsi_unrew_choice_mets]]]

group_colors = {p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0][0])}
group_colors.update({p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0][1])})
group_colors.update({p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0][2])})
group_colors.update({p: rew_hist_rew_colors[i] for i,p in enumerate(plot_groups[0][3])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1][0])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1][1])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1][2])})
group_colors.update({p: rew_hist_unrew_colors[i] for i,p in enumerate(plot_groups[1][3])})

plot_titles = [['Reward History, Rewarded Contra Choices', 'Reward History, Rewarded Ipsi Choices',
                'Bail History, Rewarded Contra Choices', 'Bail History, Rewarded Ipsi Choices',
                'Choice Metrics, Rewarded Contra Choices', 'Choice Metrics, Rewarded Ipsi Choices'],
               ['Reward History, Unrewarded Contra Choices', 'Reward History, Unrewarded Ipsi Choices',
                'Bail History, Unrewarded Contra Choices', 'Bail History, Unrewarded Ipsi Choices',
                'Choice Metrics, Unrewarded Contra Choices', 'Choice Metrics, Unrewarded Ipsi Choices']]
gen_title = 'Reward History & Choice Metrics by Choice Side and Trial Outcome'
gen_plot_name = 'rew_hist_full_sep_sides'

if not key in fit_results:
    fit_results[key] = run_regress_over_time(regress_tasks, aligns, np.unique(utils.flatten(plot_groups)), trial_sel_key=trial_sel_key,
                                             norm_trial_sel_key=norm_trial_sel_key, norm_align=norm_align, fit_ind_subj=fit_ind_subj)

with open(save_path, 'wb') as f:
    pickle.dump({'fit_results': fit_results}, f)

plot_regress_over_time(fit_results[key], plot_groups, plot_titles, gen_title, predictor_labels=pred_labels,
                       sig_lvl=plot_sig_lvl, plot_error=plot_error, xlims_dict=plot_lims,
                       group_colors=group_colors, gen_plot_name=gen_plot_name, plot_ind_subj=plot_ind_subj)



# %% Perform kernel regression

recalculate = False

signal_type = 'z_dff_iso_baseline_fband'

dec = 2

n_perms = 0
crossval = False
compute_param_stats = False

event_kernel_windows = {
    'Cport On': {'PL': [-0.1,1], 'NAc': [-0.1,1], 'DMS': [-0.1,1], 'DLS': [-0.1,1], 'TS': [-0.1,1]},
    'Cpoke In': {'PL': [-1,1], 'NAc': [-1,1], 'DMS': [-1,1], 'DLS': [-1,1], 'TS': [-1,1]},
    'Tone Start': {'PL': [-0.1,4], 'NAc': [-0.1,1], 'DMS': [-0.1,1], 'DLS': [-0.1,1], 'TS': [-0.1,1]},
    'Resp Cue': {'PL': [-0.1,2], 'NAc': [-0.1,2], 'DMS': [-0.1,1], 'DLS': [-0.1,1], 'TS': [-0.1,1]},
    'Cpoke Out': {'PL': [-1,1], 'NAc': [-1,1], 'DMS': [-1,1], 'DLS': [-1,1], 'TS': [-1,1]},
    'Resp Poke': {'PL': [-1,1], 'NAc': [-1,1], 'DMS': [-1,1], 'DLS': [-1,1], 'TS': [-1,1]},
    'Reward': {'PL': [-0.1,18], 'NAc': [-0.1,6], 'DMS': [-0.1,1], 'DLS': [-0.1,1], 'TS': [-0.1,1]},
    'Unreward': {'PL': [-0.1,18], 'NAc': [-0.1,6], 'DMS': [-0.1,1], 'DLS': [-0.1,1], 'TS': [-0.1,1]},
    'Bail': {'PL': [-0.1,18], 'NAc': [-0.1,6], 'DMS': [-0.1,1], 'DLS': [-0.1,1], 'TS': [-0.1,1]},
    'Lick End': {'PL': [-1,1], 'NAc': [-1,1], 'DMS': [-1,1], 'DLS': [-1,1], 'TS': [-1,1]},
    }

region_kernel_windows = {r: {e: event_kernel_windows[e][r] for e in event_kernel_windows.keys()} for r in all_regions}

filename = 'wm_bandit_kernel_regression'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        all_results = saved_data['all_results']
        metadata = saved_data['metadata']

        recalculate = metadata['event_kernel_windows'] != event_kernel_windows

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    all_results = {s: {t: {} for t in tasks} for s in subj_ids}

for subj_id in subj_ids:

    if not subj_id in all_results:
        all_results[subj_id] = {task: {} for task in tasks}

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    n_regions = len(subj_regions)

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids or len(all_results[subj_id][task].keys()) == n_regions:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        # aggregate signals by region and event timestamps
        signal_dict = {r: {} for r in subj_regions}
        ts_dict = {}
        event_ts_dict = {}

        for sess_id in sess_ids:

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            full_t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']
            _, dec_t = fp_utils.decimate(full_t, full_t, dec, dt=dt)
            dec_dt = dt*dec

            ts_dict[sess_id] = dec_t

            for r in subj_regions:
                if r in fp_data['processed_signals']:
                    signal = fp_data['processed_signals'][r][signal_type]
                    dec_signal, _ = fp_utils.decimate(signal, full_t, dec, dt=dt)
                    signal_dict[r][sess_id] = dec_signal

            trial_start_ts = fp_data['trial_start_ts'][:-1]

            trial_data = sess_data[sess_data['sessid'] == sess_id]

            cport_on_ts = trial_start_ts + trial_data['cport_on_time']
            cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
            cue_ts = trial_start_ts + trial_data['response_cue_time']
            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            response_ts = trial_start_ts + trial_data['response_time']
            outcome_ts = trial_start_ts + trial_data['reward_time']
            reward_ts = outcome_ts[trial_data['reward'] > 0]
            unreward_ts = outcome_ts[trial_data['reward'] == 0]
            lick_end_ts = trial_start_ts + trial_data['lick_stop_time']

            cport_on_pre_poke = cport_on_ts[cport_on_ts < cpoke_in_ts]
            cport_on_post_poke = cport_on_ts[cport_on_ts > cpoke_in_ts]
            cpoke_in_pre_cport_on = cpoke_in_ts[cport_on_ts > cpoke_in_ts]
            cpoke_in_post_cport_on = cpoke_in_ts[cport_on_ts < cpoke_in_ts]
            cue_pre_cpoke_out = cue_ts[cpoke_out_ts > cue_ts]
            cue_post_cpoke_out = cue_ts[cpoke_out_ts < cue_ts]

            # sess_events = {'Cport On - pre cpoke': cport_on_pre_poke, 'Cport On - post cpoke': cport_on_post_poke,
            #                'Cpoke In - pre port on': cpoke_in_pre_cport_on, 'Cpoke In - post port on': cpoke_in_post_cport_on,
            #                'Resp Cue - pre cpoke out': cue_pre_cpoke_out, 'Resp Cue - post cpoke out': cue_post_cpoke_out,
            #                'Cpoke Out': cpoke_out_ts, 'Resp Poke': response_ts, 'Reward': reward_ts, 'Unreward': unreward_ts,
            #                'Lick End': lick_end_ts}
            sess_events = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts,
                           'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Resp Poke': response_ts,
                           'Reward': reward_ts, 'Unreward': unreward_ts,
                           'Lick End': lick_end_ts}
            # sess_events = {'Resp Cue': cue_ts, 'Reward': reward_ts, 'Unreward': unreward_ts}

            if task == 'wm':
                tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
                cpoke_out_bail = cpoke_out_ts[trial_data['bail'] == True]
                bail_tone_ts = trial_start_ts + trial_data['bail_tone_time']
                bail_time = bail_tone_ts[trial_data['bail'] == True]
                bail_time[np.isnan(bail_time)] = cpoke_out_bail[np.isnan(bail_time)]
                sess_events.update({'Bail': bail_time, 'Tone Start': tone_ts})

            event_ts_dict[sess_id] = sess_events

        for r in subj_regions:
            start = time.perf_counter()
            all_results[subj_id][task][r] = fp_utils.kernel_regression(signal_dict[r], ts_dict, event_ts_dict,
                                                                       region_kernel_windows[reg_mapping[r]], dec_dt,
                                                                       n_perms=n_perms, crossval=crossval, compute_param_stats=compute_param_stats)
            print('Kernel regression for subject {}, task {}, region {} completed in {:.2f} s'.format(subj_id, task, r, time.perf_counter()-start))

        with open(save_path, 'wb') as f:
            pickle.dump({'all_results': all_results,
                         'metadata': {'event_kernel_windows': event_kernel_windows,
                                      'dec': dec, 'signal_type': signal_type}},
                        f)


# %% Plot Kernel Regression Results
ph = 3
pw = 3.5

# plot event kernels, one per region across all subjects, with each task on the same axis
# events = ['Cport On - pre cpoke', 'Cport On - post cpoke', 'Cpoke In - pre port on', 'Cpoke In - post port on',
#           'Tone Start', 'Resp Cue - pre cpoke out', 'Resp Cue - post cpoke out', 'Cpoke Out', 'Bail',
#           'Resp Poke', 'Reward', 'Unreward', 'Lick End']
events = ['Cport On', 'Cpoke In', 'Tone Start', #,
          'Resp Cue', 'Cpoke Out', #'Bail',
          'Reward', 'Unreward'] # 'Resp Poke', , 'Lick End'

n_events = len(events)

plot_sig = False
sig_alpha = 0.01

plot_ind_subjects = True

if plot_ind_subjects:
    for region in all_regions:

        reg_subjects = []
        reg_subj_regions = []
        reg_subj_labels = []
        for subj in subj_ids:
            subj_regions = get_subj_regions(subj)
            subj_regions = [r for r in subj_regions if region in r]
            for r in subj_regions:
                reg_subjects.append(subj)
                reg_subj_regions.append(r)
                reg_subj_labels.append('{} ({})'.format(subj, implant_info[subj][r]['side']))

        n_subj = len(reg_subjects)
        fig, axs = plt.subplots(n_subj, n_events, figsize=(n_events*pw, n_subj*ph), layout='constrained', sharey='row')
        axs = axs.reshape((n_subj, n_events))
        fig.suptitle('Kernel Regression Coefficients - {}'.format(region))

        for i, (subj_id, subj_reg, subj_label) in enumerate(zip(reg_subjects, reg_subj_regions, reg_subj_labels)):
            for j, event in enumerate(events):
                ax = axs[i,j]
                ax.set_title('{} - Subj {}'.format(event, subj_label))
                plot_utils.plot_x0line(ax=ax)
                plot_utils.plot_dashlines(0, dir='h', ax=ax)

                for k, task in enumerate(tasks):
                    subj_results = all_results[subj_id][task][subj_reg]

                    if event in subj_results['kernel_stats']:
                        kernel_stats = subj_results['kernel_stats'][event]
                        kernel_t = kernel_stats['t']
                        full_betas = kernel_stats['full_betas']
                        beta_se = kernel_stats['se']
                        beta_pvals = kernel_stats['p_vals']
                        sig_ps = beta_pvals < sig_alpha/subj_results['model_stats']['n_preds']
                        # cv_betas = np.vstack([b[None,:] for b in kernel_stats['cv_betas']])
                        # cv_mean = np.mean(cv_betas, axis=0)
                        # cv_se =  utils.stderr(cv_betas, axis=0)

                        # perm_betas = np.vstack([b[None,:] for b in kernel_stats['perm_betas']])
                        # beta_sig = np.sum(np.abs(full_betas)[None,:] < perm_betas, axis=0)
                        # sig_thresh = sig_alpha*perm_betas.shape[0]

                        # beta_sig = beta_sig < sig_thresh

                        plot_utils.plot_shaded_error(kernel_t, full_betas, y_err=beta_se, ax=ax, label=task, color='C{}'.format(k))
                        if plot_sig:
                            ax.fill_between(kernel_t, 0, 1, where=sig_ps, color='C{}'.format(k), alpha=0.3, transform=ax.get_xaxis_transform())

                ax.legend()
                handles, labels = ax.get_legend_handles_labels()
                ax.legend().remove()

        fig.legend(handles, labels, loc='outside right center')

# plot subject averages
fig, axs = plt.subplots(len(all_regions), n_events, figsize=(n_events*pw, len(all_regions)*ph), layout='constrained', sharey='row')

axs = np.atleast_2d(axs)

fig.suptitle('Kernel Regression Subject Averages by Region')

for i, region in enumerate(all_regions):
    for j, event in enumerate(events):

        ax = axs[i, j]

        plot_utils.plot_x0line(ax=ax)
        plot_utils.plot_dashlines(0, dir='h', ax=ax)

        for k, task in enumerate(tasks):

            subj_betas = []
            kernel_t = None

            for subj in subj_ids:

                subj_regions = get_subj_regions(subj)
                subj_regions = [r for r in subj_regions if region in r]

                # average across all matching regions within subject
                subj_region_betas = []

                for subj_reg in subj_regions:

                    subj_results = all_results[subj][task][subj_reg]

                    if event in subj_results['kernel_stats']:

                        kernel_stats = subj_results['kernel_stats'][event]

                        kernel_t = kernel_stats['t']
                        subj_region_betas.append(kernel_stats['full_betas'])

                # average bilateral regions within subject first
                if len(subj_region_betas) > 0:

                    subj_region_betas = np.vstack(subj_region_betas)
                    subj_avg_beta = np.mean(subj_region_betas, axis=0)

                    subj_betas.append(subj_avg_beta)

            if len(subj_betas) > 0:

                subj_betas = np.vstack(subj_betas)

                mean_beta = np.mean(subj_betas, axis=0)
                sem_beta = utils.stderr(subj_betas, axis=0)

                plot_utils.plot_shaded_error(kernel_t, mean_beta, y_err=sem_beta, ax=ax, label=task, color='C{}'.format(k))

        if i == 0:
            ax.set_title(event)
        if j == 0:
            ax.set_ylabel('{}\nz-dF/F'.format(region))
        if i == len(all_regions)-1:
            ax.set_xlabel('Time (s)')

        ax.legend()
        handles, labels = ax.get_legend_handles_labels()
        ax.legend().remove()

fig.legend(handles, labels, loc='outside right center')


# %% plot fraction of event power/variance by subject and over all subjects

plot_ind_subj = False

# first build dataframe for ease of plotting
frac_var_df = []
for s in subj_ids:

    subj_regions = get_subj_regions(s)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for t in tasks:
        for r in subj_regions:
            subj_results = all_results[s][t][r]
            s_side = '{} ({})'.format(s, implant_info[s][r]['side'])
            frac_var_df.append({'Subject': s, 'Subject (side)': s_side, 'Task': t, 'Subject Region': r, 'Region': reg_mapping[r], 'Event': 'All',
                                'Var Ratio, Log2(r)': np.log2(subj_results['model_stats']['var_ratio']),
                                'Mean Shift, d': subj_results['model_stats']['mean_shift'],
                                'Combined, Log2(r+d^2)': np.log2(subj_results['model_stats']['second_moment'])})

            for e in events:
                if e in subj_results['event_stats']:
                    frac_var_df.append({'Subject': s, 'Subject (side)': s_side, 'Task': t, 'Subject Region': r, 'Region': reg_mapping[r], 'Event': e,
                                        'Var Ratio, Log2(r)': np.log2(subj_results['event_stats'][e]['var_ratio']),
                                        'Mean Shift, d': subj_results['event_stats'][e]['mean_shift'],
                                        'Combined, Log2(r+d^2)': np.log2(subj_results['event_stats'][e]['second_moment'])})

all_mets = ['Var Ratio, Log2(r)', 'Mean Shift, d', 'Combined, Log2(r+d^2)']

frac_var_df = pd.DataFrame(frac_var_df)
all_frac_var = frac_var_df[frac_var_df['Event'] == 'All']

# plot whole model by subject/region
if plot_ind_subj:
    melt_all_frac_var = pd.melt(all_frac_var,
                                id_vars=['Subject (side)', 'Region', 'Task'],
                                value_vars=all_mets,
                                var_name='Metric', value_name='value')
    g = sb.catplot(melt_all_frac_var, x='Subject (side)', y='value', hue='Task',
                   col='Region', col_order=all_regions, row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar=None)
    fig = g.figure
    fig.suptitle('Model Event Signals Compared to Non-event Signals by Region')
    fig.subplots_adjust(top=0.9)

fig, axs = plt.subplots(len(all_mets),2, layout='constrained', figsize=(6,3*len(all_mets)), sharey='row')
fig.suptitle('Model Event Signals Compared to Non-event Signals')
for i, met in enumerate(all_mets):
    ax = axs[i,0]
    ax.set_title('{}, Separate Tasks'.format(met))
    sb.barplot(all_frac_var, x='Region', y=met, hue='Task', errorbar='se', ax=ax, order=all_regions)
    ax.legend(loc='upper right')
    ax = axs[i,1]
    ax.set_title('{}, All Tasks'.format(met))
    sb.barplot(all_frac_var, x='Region', y=met, errorbar='se', ax=ax, order=all_regions)

# plot variance fraction across events by subject/region
event_frac_var = frac_var_df[frac_var_df['Event'] != 'All']
# if plot_ind_subj:
#     for r in all_regions:
#         g = sb.catplot(event_frac_var[event_frac_var['Region'] == r], y='Event', x=met_title, hue='Task', col='Subject (side)',
#                        kind='bar', sharex=True, errorbar=None, col_wrap=4)
#         g.set_xlabels(met_title)
#         fig = g.figure
#         fig.suptitle('Event {} for {} by Subject - {}'.format(met_title, r, title_desc))
#         fig.subplots_adjust(top=0.9)

melt_event_frac_var = pd.melt(event_frac_var,
                            id_vars=['Subject (side)', 'Region', 'Task', 'Event'],
                            value_vars=all_mets,
                            var_name='Metric', value_name='value')

# separate tasks
g = sb.catplot(melt_event_frac_var, y='Event', x='value', hue='Task', col='Region', col_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar='se')

fig = g.figure
fig.suptitle('Individual Event Signals Compared to Non-event Signals, Separate Tasks')
fig.subplots_adjust(top=0.9)

g = sb.catplot(melt_event_frac_var, y='Event', x='value', hue='Region', col='Task', hue_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex='row', errorbar='se', palette=region_colors)
fig = g.figure
fig.suptitle('Individual Event Signals Compared to Non-event Signals, Separate Tasks')
fig.subplots_adjust(top=0.9)

# all tasks
g = sb.catplot(melt_event_frac_var, y='Event', x='value', col='Region', col_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar='se')
fig = g.figure
fig.suptitle('Individual Event Signals Compared to Non-event Signals, All Tasks')
fig.subplots_adjust(top=0.9)

g = sb.catplot(melt_event_frac_var, y='Event', x='value', hue='Region', hue_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar='se', palette=region_colors)
fig = g.figure
fig.suptitle('Individual Event Signals Compared to Non-event Signals, All Tasks')
fig.subplots_adjust(top=0.9)


# %% plot overall event significance by subject and over all subjects

plot_ind_subj = False
plot_metric = 'marginal_event' # 'marginal_event', 'event', 'masked', 'full'

match plot_metric:
    case 'event':
        all_key = 'R2_masked'
        e_key = 'R2_event_full'
        title_desc = 'During Individual Event Kernels'
        value_label = '% Explained Variance (R2)'
    case 'marginal_event':
        all_key = 'R2_masked'
        e_key = 'delta_R2_event'
        title_desc = 'During Individual Event Kernels'
        value_label = '% Marginal Explained Variance (ΔR2)'
    case 'masked':
        all_key = 'R2_masked'
        e_key = 'delta_R2_masked'
        title_desc = 'During All Event Kernels'
        value_label = '% Marginal Explained Variance (ΔR2)'
    case 'full':
        all_key = 'R2_full'
        e_key = 'delta_R2_full'
        title_desc = 'During Whole Recording'
        value_label = '% Marginal Explained Variance (ΔR2)'

# first build dataframe for ease of plotting
delta_r2_df = []
for s in subj_ids:

    subj_regions = get_subj_regions(s)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for t in tasks:
        for r in subj_regions:
            subj_results = all_results[s][t][r]
            s_side = '{} ({})'.format(s, implant_info[s][r]['side'])
            delta_r2_df.append({'Subject': s, 'Subject (side)': s_side, 'Task': t, 'Subject Region': r, 'Region': reg_mapping[r], 'Event': 'All',
                                '% R2': subj_results['model_stats'][all_key]*100})
            for e in events:
                if e in subj_results['event_stats']:
                    delta_r2_df.append({'Subject': s, 'Subject (side)': s_side, 'Task': t, 'Subject Region': r, 'Region': reg_mapping[r], 'Event': e,
                                        '% ΔR2': subj_results['event_stats'][e][e_key]*100})

delta_r2_df = pd.DataFrame(delta_r2_df)
avg_df = delta_r2_df.groupby(['Region', 'Task', 'Event']).agg(avg=('% ΔR2', 'mean'), se=('% ΔR2', lambda x: utils.stderr(x))).reset_index()

# plot whole model R2 by subject/region
all_r2 = delta_r2_df[delta_r2_df['Event'] == 'All']

if plot_ind_subj:
    g = sb.catplot(all_r2, x='Subject Region', y='% R2', hue='Task', col='Subject',
                   kind='bar', sharex=False, errorbar=None, col_wrap=5)
    g.set_ylabels('% Explained Variance (R2)')
    fig = g.figure
    fig.suptitle('Total Model % Explained Variance (R2) by Subject - {}'.format(title_desc))
    fig.subplots_adjust(top=0.9)

    g = sb.catplot(all_r2, x='Subject (side)', y='% R2', hue='Task', col='Region',
                   kind='bar', sharex=False, errorbar=None, col_wrap=5)
    g.set_ylabels('% Explained Variance (R2)')
    fig = g.figure
    fig.suptitle('Total Model % Explained Variance (R2) by Subject - {}'.format(title_desc))
    fig.subplots_adjust(top=0.9)

fig, axs = plt.subplots(1,2, layout='constrained', figsize=(6,3), sharey=True)
fig.suptitle('Avg Model % Explained Variance (R2)')
ax = axs[0]
ax.set_title('Separate Tasks')
sb.barplot(all_r2, x='Region', y='% R2', hue='Task', errorbar='se', ax=ax, order=all_regions)
ax.legend(loc='upper right')
ax = axs[1]
ax.set_title('All Tasks')
sb.barplot(all_r2, x='Region', y='% R2', errorbar='se', ax=ax, order=all_regions)

# plot delta R2 across events by subject/region
event_r2 = delta_r2_df[delta_r2_df['Event'] != 'All']
if plot_ind_subj:
    for r in all_regions:
        g = sb.catplot(event_r2[event_r2['Region'] == r], y='Event', x='% ΔR2', hue='Task', col='Subject (side)',
                       kind='bar', sharex=True, errorbar=None, col_wrap=4)
        g.set_xlabels(value_label)
        fig = g.figure
        fig.suptitle('Event {} for {} by Subject - {}'.format(value_label, r, title_desc))
        fig.subplots_adjust(top=0.9)

    for s in subj_ids:
        g = sb.catplot(event_r2[event_r2['Subject'] == s], y='Event', x='% ΔR2', hue='Task', col='Subject Region',
                       kind='bar', sharex=True, errorbar=None)
        g.set_xlabels(value_label)
        fig = g.figure
        fig.suptitle('Event {} for Subject {} - {}'.format(value_label, s, title_desc))
        fig.subplots_adjust(top=0.9)

# plot separate tasks
g = sb.catplot(event_r2, y='Event', x='% ΔR2', hue='Task', col='Region', col_order=all_regions,
               kind='bar', sharex=False, errorbar='se')
g.set_xlabels(value_label)
fig = g.figure
fig.suptitle('Event {}, Separate Tasks - {}'.format(value_label, title_desc))
fig.subplots_adjust(top=0.9)

g = sb.catplot(event_r2, y='Event', x='% ΔR2', hue='Region', col='Task', hue_order=all_regions,
               kind='bar', sharex=True, errorbar='se', palette=region_colors)
g.set_xlabels(value_label)
fig = g.figure
fig.suptitle('Event {}, Separate Tasks - {}'.format(value_label, title_desc))
fig.subplots_adjust(top=0.9)

# plot combined tasks
g = sb.catplot(event_r2, y='Event', x='% ΔR2', col='Region', col_order=all_regions,
               kind='bar', sharex=False, errorbar='se')
g.set_xlabels(value_label)
fig = g.figure
fig.suptitle('Event {}, All Tasks - {}'.format(value_label, title_desc))
fig.subplots_adjust(top=0.9)

g = sb.catplot(event_r2, y='Event', x='% ΔR2', hue='Region', hue_order=all_regions,
               kind='bar', sharex=True, errorbar='se', palette=region_colors)
g.set_xlabels(value_label)
fig = g.figure
fig.suptitle('Event {}, All Tasks - {}'.format(value_label, title_desc))
fig.subplots_adjust(top=0.9)


# %% Calculate task epoch labels

view_states = False
recalculate = False

def label_interval(start, end, label, sess_epoch_t_labels):
    if np.isnan(start) or np.isnan(end) or end <= start:
        return

    t_sel = (t >= start) & (t < end)
    sess_epoch_t_labels[t_sel] = label

# time cutoff between center port on or last lick, whichever is later, and next poke in to determine disengaged states
engage_iti_thresh = 15
cpoke_thresh = 0.5
sess_end_max = 60 # maximum time to label the session end
epochs = ['Engaged ITI', 'Disengaged ITI', 'Cpoke to Response', 'Reward Response to Lick End',
          'Unreward Response to Lick End', 'Session End', 'All']

filename = 'wm_bandit_task_epoch_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        epoch_t_labels = saved_data['epoch_t_labels']
        engage_metadata = saved_data['metadata']

        recalculate = engage_metadata['engage_iti_thresh'] != engage_iti_thresh

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    epoch_t_labels = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:

    if not subj_id in epoch_t_labels:
        epoch_t_labels[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in epoch_t_labels[subj_id][task]:
                continue

            fp_data = loc_db.get_sess_fp_data(sess_id)['fp_data'][subj_id][sess_id]
            t = fp_data['time']

            # find engaged and disengaged states
            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_start_ts = fp_data['trial_start_ts']

            cport_on_ts = trial_start_ts[:-1] + trial_data['cport_on_time'].to_numpy()
            cpoke_in_ts = trial_start_ts[:-1] + trial_data['cpoke_in_time'].to_numpy()
            cpoke_out_ts = trial_start_ts[:-1] + trial_data['cpoke_out_time'].to_numpy()
            response_ts = trial_start_ts[:-1] + trial_data['response_time'].to_numpy()
            outcome_ts = trial_start_ts[:-1] + trial_data['reward_time'].to_numpy()
            lick_end_ts = trial_start_ts[:-1] + trial_data['lick_stop_time'].to_numpy()
            if task == 'wm':
                bail_tone_ts = trial_start_ts[:-1] + trial_data['bail_tone_time'].to_numpy()

            rewarded = trial_data['rewarded'].to_numpy()

            sess_epoch_t_labels = np.full(t.shape, 'Disengaged ITI', dtype=object)

            for i in range(len(trial_data)):

                # determine ITI start from previous trial
                if i == 0:
                    iti_start = t[0]
                elif not np.isnan(lick_end_ts[i-1]):
                    iti_start = lick_end_ts[i-1]
                elif not np.isnan(outcome_ts[i-1]):
                    iti_start = outcome_ts[i-1]
                elif not np.isnan(response_ts[i-1]):
                    iti_start = response_ts[i-1]
                elif task == 'wm' and not np.isnan(bail_tone_ts[i-1]):
                    iti_start = bail_tone_ts[i-1]
                elif not np.isnan(cpoke_out_ts[i-1]):
                    iti_start = cpoke_out_ts[i-1]
                else:
                    iti_start = trial_start_ts[i]

                # some trials the animal poked in while the next trial was being created
                # without having the pose tracking, just start the cpoke in from the start of the trial
                if np.isnan(cpoke_in_ts[i]) and not np.isnan(cpoke_out_ts[i]):
                    cpoke_in = trial_start_ts[i]
                else:
                    cpoke_in = cpoke_in_ts[i]

                if (cpoke_in - iti_start) < engage_iti_thresh:
                    label_interval(iti_start, cpoke_in, 'Engaged ITI', sess_epoch_t_labels)

                # determine response time
                if not np.isnan(response_ts[i]):
                    resp_time = response_ts[i]
                else:
                    resp_time = cpoke_out_ts[i]

                if (resp_time - cpoke_in) > cpoke_thresh:
                    label_interval(cpoke_in, resp_time, 'Cpoke to Response', sess_epoch_t_labels)

                if rewarded[i]:
                    label_interval(resp_time, lick_end_ts[i], 'Reward Response to Lick End', sess_epoch_t_labels)
                else:
                    label_interval(resp_time, lick_end_ts[i], 'Unreward Response to Lick End', sess_epoch_t_labels)

            # label session end
            sess_end_start = max(t[-1]-sess_end_max, trial_start_ts[-1])
            label_interval(sess_end_start, t[-1], 'Session End', sess_epoch_t_labels)

            epoch_t_labels[subj_id][task][sess_id] = sess_epoch_t_labels

            if view_states:

                lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts, 'Lick End': lick_end_ts}

                fig, ax = plt.subplots(1, 1, sharex=True, layout='constrained', figsize=(12,4))

                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

                for j, e in enumerate(epochs):
                    if e == 'All':
                        continue
                    ax.fill_between(t, 0, 1, where=(sess_epoch_t_labels == e), label=e,
                                    color='C{}'.format(j), alpha=0.4, transform=ax.get_xaxis_transform())

                for j, (name, lines) in enumerate(lines_dict.items()):
                    ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j), linestyles='dashed',
                              transform=ax.get_xaxis_transform())

                ax.set_xlabel('Time (s)')
                ax.legend()
                plt.show()

with open(save_path, 'wb') as f:
    pickle.dump({'epoch_t_labels': epoch_t_labels,
                 'metadata': {'engage_iti_thresh': engage_iti_thresh}},
                f)


# define method to decimate epoch labels
def decimate_epoch_labels(labels, dec):
    if dec == 1:
        return labels

    pad_len = (-len(labels)) % dec

    if pad_len > 0:
        reshape_labels = np.concatenate([labels, np.full(pad_len, labels[-1])])
    else:
        reshape_labels = labels

    reshape_labels = np.reshape(reshape_labels, (-1, dec))

    dec_labels = np.array([Counter(row).most_common(1)[0][0] for row in reshape_labels])

    return dec_labels

# %% Calculate variance ratios of signals in each epoch

recalculate = False

signal_type = 'z_dff_iso_baseline_fband'

filename = 'wm_bandit_epoch_var_ratios'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        epoch_var_ratios = saved_data['epoch_var_ratios']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    epoch_var_ratios = {s: {t: {} for t in tasks} for s in subj_ids}

for subj_id in subj_ids:

    if not subj_id in epoch_var_ratios:
        epoch_var_ratios[subj_id] = {task: {} for task in tasks}

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    n_regions = len(subj_regions)

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids or len(epoch_var_ratios[subj_id][task].keys()) == n_regions:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        # aggregate signals by region and epoch labels
        stacked_signals = {r: [] for r in subj_regions}
        stacked_epoch_labels = []

        for sess_id in sess_ids:

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]

            stacked_epoch_labels.append(epoch_t_labels[subj_id][task][sess_id])

            for r in subj_regions:
                if r in fp_data['processed_signals']:
                    stacked_signals[r].append(fp_data['processed_signals'][r][signal_type])
                else:
                    stacked_signals[r].append(np.full(epoch_t_labels[subj_id][task][sess_id].shape, np.nan))

        stacked_epoch_labels = np.concatenate(stacked_epoch_labels)

        for r in subj_regions:
            stacked_signals[r] = np.concatenate(stacked_signals[r])
            epoch_var_ratios[subj_id][task][r] = {}

            # first calculate variance for each epoch in order to calculate pooled, within-epoch variance
            epoch_vars = {}

            for e in epochs:
                if e == 'All':
                    continue

                t_sel = stacked_epoch_labels == e
                epoch_signal = stacked_signals[r][t_sel]
                valid = ~np.isnan(epoch_signal)
                valid_signal = epoch_signal[valid]
                e_n = sum(valid)
                e_mean = np.mean(valid_signal)
                e_var = np.sum((valid_signal - e_mean)**2)/e_n
                epoch_vars[e] = {'mean': e_mean, 'var': e_var, 'n': e_n}

            all_valid = ~np.isnan(stacked_signals[r])
            all_mean = np.mean(stacked_signals[r][all_valid])
            all_n = sum(all_valid)
            pooled_var = np.sum([epoch_vars[e]['var']*epoch_vars[e]['n'] for e in epochs if not e == 'All'])/all_n
            pooled_std = np.sqrt(pooled_var)

            for e in epochs:
                if e == 'All':
                    continue

                var_ratio = epoch_vars[e]['var']/pooled_var
                var_shift = (np.sqrt(epoch_vars[e]['var']) - pooled_std)/pooled_std
                mean_shift = (epoch_vars[e]['mean'] - all_mean)/pooled_std

                epoch_var_ratios[subj_id][task][r][e] = {
                    'var': epoch_vars[e]['var'],
                    'mean': epoch_vars[e]['mean'],
                    'var_ratio': var_ratio,
                    'var_shift': var_shift,
                    'mean_shift': mean_shift,
                    'second_moment': var_ratio + mean_shift**2,
                    'n': epoch_vars[e]['n'],
                    'pooled_var': pooled_var,
                    'all_mean': all_mean
                }

        with open(save_path, 'wb') as f:
            pickle.dump({'epoch_var_ratios': epoch_var_ratios}, f)

# %% Plot epoch variance ratios TODO: plot var shift and scale shift in a scatter plot

plot_ind_subj = False

# first build dataframe for ease of plotting
frac_var_df = []
for s in subj_ids:

    subj_regions = get_subj_regions(s)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for t in tasks:
        for r in subj_regions:
            subj_results = epoch_var_ratios[s][t][r]
            s_side = '{} ({})'.format(s, implant_info[s][r]['side'])

            for e in epochs:
                if e == 'All':
                    continue

                frac_var_df.append({'Subject': s, 'Subject (side)': s_side, 'Task': t, 'Subject Region': r, 'Region': reg_mapping[r], 'Epoch': e,
                                    'Var Ratio': subj_results[e]['var_ratio'],
                                    'Second Moment': subj_results[e]['second_moment'],
                                    'Var Ratio, Log2(r)': np.log2(subj_results[e]['var_ratio']),
                                    'Mean Shift, d': subj_results[e]['mean_shift'],
                                    'Combined, Log2(r+d^2)': np.log2(subj_results[e]['second_moment'])})

all_mets = ['Var Ratio, Log2(r)', 'Mean Shift, d', 'Combined, Log2(r+d^2)']

frac_var_df = pd.DataFrame(frac_var_df)
melt_frac_var = pd.melt(frac_var_df,
                        id_vars=['Subject (side)', 'Region', 'Task', 'Epoch'],
                        value_vars=all_mets,
                        var_name='Metric', value_name='value')

# plot variance fraction across epochs by subject/region

if plot_ind_subj:
    for r in all_regions:
        g = sb.catplot(melt_frac_var[melt_frac_var['Region'] == r], y='Epoch', x='value', hue='Task',
                       col='Subject (side)', row='Metric', row_order=all_mets, kind='bar', sharex=True, errorbar=None)
        fig = g.figure
        fig.suptitle('Epoch Signal Differences for {} by Subject'.format(r))
        fig.subplots_adjust(top=0.9)


# separate tasks
g = sb.catplot(melt_frac_var, y='Epoch', x='value', hue='Task', col='Region', col_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar='se')

fig = g.figure
fig.suptitle('Epoch Signal Differences, Separate Tasks')
fig.subplots_adjust(top=0.9)

g = sb.catplot(melt_frac_var, y='Epoch', x='value', hue='Region', col='Task', hue_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex='row', errorbar='se', palette=region_colors)
fig = g.figure
fig.suptitle('Epoch Signal Differences, Separate Tasks')
fig.subplots_adjust(top=0.9)
fig.set_layout_engine("constrained")

# all tasks
g = sb.catplot(melt_frac_var, y='Epoch', x='value', col='Region', col_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar='se')
fig = g.figure
fig.suptitle('Epoch Signal Differences, All Tasks')
fig.subplots_adjust(top=0.9)

g = sb.catplot(melt_frac_var, y='Epoch', x='value', hue='Region', hue_order=all_regions,
               row='Metric', row_order=all_mets, kind='bar', sharex=False, errorbar='se', palette=region_colors)
fig = g.figure
fig.suptitle('Epoch Signal Differences, All Tasks')
fig.subplots_adjust(top=0.9)


# %% Calculate power spectra

f_min = 0.005

recalculate = False
reprocess_sess_ids = []

signal_types = ['z_dff_iso_baseline_fband'] # 'z_dff_iso_baseline'

filename = 'wm_bandit_spectral_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        ps_data = saved_data['ps_data']
        freqs = saved_data['freqs']

        if saved_data['f_min'] != f_min:
            recalculate = True

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    ps_data = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)

    if not subj_id in ps_data:
        ps_data[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in ps_data[subj_id][task] and not sess_id in reprocess_sess_ids and all([t in list(ps_data[subj_id][task][sess_id].keys()) for t in signal_types]):
                continue

            if sess_id not in ps_data[subj_id][task]:
                ps_data[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']

            for signal_type in signal_types:
                if signal_type in ps_data[subj_id][task][sess_id]:
                    continue
                else:
                    ps_data[subj_id][task][sess_id][signal_type] = {r: {e: [] for e in epochs} for r in subj_regions}

                for region in subj_regions:
                    if not region in fp_data['processed_signals']:
                        continue

                    signal = fp_data['processed_signals'][region][signal_type]

                    for e in epochs:
                        if e == 'All':
                            t_sel = np.full_like(t, True, dtype=bool)
                        else:
                            t_sel = epoch_t_labels[subj_id][task][sess_id] == e

                        try:
                            freqs, ps = fp_utils.calc_power_spectra(signal, dt, f_min=f_min, t_sel=t_sel)
                        except ValueError as err:
                            print('Could not calculate power spectrum for subject {}, session {}, task {}, region {}, signal {}, epoch {}: {}'.format(
                                subj_id, sess_id, task, region, signal_type, e, err))
                            continue

                        ps_data[subj_id][task][sess_id][signal_type][region][e] = ps

                with open(save_path, 'wb') as f:
                    pickle.dump({'ps_data': ps_data,
                                 'freqs': freqs,
                                 'f_min': f_min},
                                f)

# %% calculate frequency cutoff by percentage of total power per subject and region

freq_cutoffs = [0.95, 0.99, 0.999]
freq_range = [0, 10]
analyze_tasks = tasks + ['all']
freq_cutoff_df = []

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for signal_type in signal_types:
            for region in subj_regions:
                for task in analyze_tasks:
                    if task == 'all':
                        reg_ps = np.vstack([np.stack([ps_data[subj_id][t][s][signal_type][region] for s in ps_data[subj_id][t].keys()
                                                      if len(ps_data[subj_id][t][s][signal_type][region])> 0], axis=0) for t in tasks if len(ps_data[subj_id][t]) > 0])
                    else:
                        reg_ps = np.stack([ps_data[subj_id][task][s][signal_type][region] for s in ps_data[subj_id][task].keys()
                                           if len(ps_data[subj_id][task][s][signal_type][region]) > 0], axis=0)

                    freq_sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
                    tot_power = trapezoid(reg_ps[:, freq_sel], freqs[freq_sel], axis=1)

                    cum_power = cumulative_trapezoid(reg_ps[:, freq_sel], freqs[freq_sel], initial=0)

                    cum_power_prop = cum_power / tot_power[:,None]

                    results = {'subj_id': subj_id, 'task': task, 'region': region, 'general_region': reg_mapping[region]}
                    for cutoff in freq_cutoffs:
                        max_f = 0
                        for i in range(cum_power_prop.shape[0]):
                            cutoff_f = np.interp(cutoff, cum_power_prop[i,:], freqs[freq_sel])
                            if cutoff_f > max_f:
                                max_f = cutoff_f
                        results[cutoff] = max_f

                    freq_cutoff_df.append(results)

freq_cutoff_df = pd.DataFrame(freq_cutoff_df)

# first average over subject regions
avg_region_cutoff = freq_cutoff_df.groupby(['subj_id', 'general_region', 'task'])[freq_cutoffs].agg('mean').reset_index()
avg_region_cutoff = avg_region_cutoff.groupby(['general_region', 'task'])[freq_cutoffs].agg('mean').reset_index()

# %% Plot spectra

save_plots = False
show_plots = True

plot_ind_subj = False
plot_raw_spectra = False
plot_integrated_spectra = True
plot_band_prop = True

plot_sep_tasks = True
plot_sep_epochs = True
plot_all_epoch = False

freq_range = [0, 10]
prop_f_edges = np.concat([np.array([freq_range[0]]), 10**np.linspace(np.log10(0.1), np.log10(freq_range[1]), 20)])
prop_bands = [[v1, v2] for v1, v2 in zip(prop_f_edges[:-1], prop_f_edges[1:])]

# prop_bands = [[0,0.02], [0.02,0.1], [0.1,0.3], [0.3,1], [1,1.8], [1.8,4], [4,10]]
# prop_bands = [[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,1], [1,2], [2,6], [6,10]]

band_array = np.array(prop_bands)
band_centers = band_array[:,0] + np.diff(band_array, axis=1)[:,0]/2

plot_tasks = tasks.copy() if plot_sep_tasks else []
plot_tasks.append('all')

plot_epochs = epochs.copy() if plot_sep_epochs else ['All']
if not plot_all_epoch:
    plot_epochs.remove('All')

plot_subj = []
if plot_ind_subj:
    plot_subj.extend(subj_ids.copy())
plot_subj.append('all')

plot_signals = ['z_dff_iso_baseline_fband']

def get_task_spectra(stacked_spectra, signal_type, subj, task, epoch, region):
    if task == 'all':
        all_task_spectra = [np.stack(stacked_spectra[signal_type][subj][t][epoch][region], axis=0) for t in tasks
                            if len(stacked_spectra[signal_type][subj][t][epoch][region]) > 0]
        if len(all_task_spectra) == 0:
            return None
        return np.vstack(all_task_spectra)
    else:
        if len(stacked_spectra[signal_type][subj][task][epoch][region]) == 0:
            return None
        return np.stack(stacked_spectra[signal_type][subj][task][epoch][region], axis=0)

def prep_spectra_for_plot(stacked_ps, integrated=False, relative=False):
    if integrated:
        band_ps = []
        for band in prop_bands:
            freq_sel = (freqs >= band[0]) & (freqs <= band[1])
            band_ps.append(utils.nan_trapezoid(stacked_ps[:, freq_sel], freqs[freq_sel], axis=1))
        band_ps = np.stack(band_ps, axis=1)
        if relative:
            ps_tot = np.nansum(band_ps, axis=1, keepdims=True)
            band_ps = band_ps / ps_tot * 100
        return band_ps
    return stacked_ps

def plot_stacked_spectra(ax, stacked_ps, label, color=None, integrated=False, logy=True, relative=False):
    plot_ps = prep_spectra_for_plot(stacked_ps, integrated=integrated, relative=relative)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if integrated:
            plot_utils.plot_shaded_error(band_centers, np.nanmean(plot_ps, axis=0),
                                         y_err=utils.stderr(plot_ps, axis=0), ax=ax, label=label, color=color)
            ax.set_xlabel('Frequency Band Center (Hz)')
        else:
            freq_sel = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
            plot_utils.plot_shaded_error(freqs[freq_sel], np.nanmean(plot_ps, axis=0)[freq_sel],
                                         y_err=utils.stderr(plot_ps, axis=0)[freq_sel], ax=ax, label=label, color=color)
            ax.set_xlabel('Frequency (Hz)')

        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

def save_spectra_plot(fig, subj, plot_name):
    if save_plots and not plot_name is None:
        fpah.save_fig(fig, fpah.get_figure_save_path('Power Spectra', subj, plot_name))

    if not show_plots:
        plt.close(fig)

# stack spectra by epoch and average bilateral regions within session for the meta-subject
stacked_spectra = {sig: {s: {t: {e: {r: [] for r in get_subj_regions(s)} for e in epochs} for t in tasks} for s in subj_ids} for sig in plot_signals}
for signal_type in plot_signals:
    stacked_spectra[signal_type]['all'] = {t: {e: {r: [] for r in all_regions} for e in epochs} for t in tasks}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    subj_unique_regions = np.unique(list(reg_mapping.values()))

    for task in tasks:
        if not subj_id in ps_data or not task in ps_data[subj_id]:
            continue

        for sess_id in ps_data[subj_id][task].keys():
            for signal_type in plot_signals:
                if not signal_type in ps_data[subj_id][task][sess_id]:
                    continue

                for epoch in epochs:

                    all_region_spectra = {r: [] for r in subj_unique_regions}

                    for region in subj_regions:
                        if (not region in ps_data[subj_id][task][sess_id][signal_type]
                                or not epoch in ps_data[subj_id][task][sess_id][signal_type][region]):
                            continue

                        sess_reg_ps = ps_data[subj_id][task][sess_id][signal_type][region][epoch]
                        if len(sess_reg_ps) == 0:
                            continue

                        stacked_spectra[signal_type][subj_id][task][epoch][region].append(sess_reg_ps)
                        all_region_spectra[reg_mapping[region]].append(sess_reg_ps)

                    for region in subj_unique_regions:
                        if len(all_region_spectra[region]) > 0:
                            avg_ps = np.nanmean(np.stack(all_region_spectra[region], axis=0), axis=0)
                            stacked_spectra[signal_type]['all'][task][epoch][region].append(avg_ps)

spectra_plot_types = []
if plot_raw_spectra:
    spectra_plot_types.append(('raw', False, False, True, 'Power Spectral Density (V^2/Hz)', 'Power Spectra'))
if plot_integrated_spectra:
    spectra_plot_types.append(('integrated', True, False, True, 'Integrated Power (V^2)', 'Integrated Band Power'))
if plot_band_prop:
    spectra_plot_types.append(('band proportion', True, True, True, 'Relative Power (%)', 'Power Proportion'))

for signal_type in plot_signals:
    signal_label = fpah.get_signal_type_labels(signal_type)[0]

    for plot_type, integrated, relative, logy, y_label, plot_title in spectra_plot_types:

        # plot each epoch separately
        for subj in plot_subj:
            if subj == 'all':
                subj_regions = all_regions
            else:
                subj_regions = get_subj_regions(subj)

            fig, axs = plt.subplots(len(plot_tasks), len(plot_epochs), figsize=(4*len(plot_epochs), 4*len(plot_tasks)),
                                    layout='constrained', sharey=True)
            fig.suptitle('Regional {} for Subject {} by Epoch\n{}'.format(plot_title, subj, signal_label))

            axs = np.reshape(axs, (len(plot_tasks), len(plot_epochs)))
            handles = []
            labels = []

            for i, epoch in enumerate(plot_epochs):
                for j, task in enumerate(plot_tasks):
                    ax = axs[j,i]
                    ax.set_title('{}, {}'.format(plot_beh_names[task], epoch))

                    for k, region in enumerate(subj_regions):
                        stacked_ps = get_task_spectra(stacked_spectra, signal_type, subj, task, epoch, region)
                        if stacked_ps is None:
                            continue

                        if subj == 'all':
                            color = region_colors[region]
                        else:
                            color = 'C{}'.format(k)

                        plot_stacked_spectra(ax, stacked_ps, region, color=color, integrated=integrated, logy=logy, relative=relative)

                    plot_handles, plot_labels = ax.get_legend_handles_labels()
                    if len(plot_handles) > 0:
                        handles = plot_handles
                        labels = plot_labels
                    if i == 0:
                        ax.set_ylabel(y_label)

            if len(handles) > 0:
                fig.legend(handles, labels, loc='outside right center')

            save_spectra_plot(fig, subj, '{} by epoch {}'.format(plot_type, signal_type))

        # plot each region separately
        for subj in plot_subj:
            if subj == 'all':
                subj_regions = all_regions
            else:
                subj_regions = get_subj_regions(subj)

            fig, axs = plt.subplots(len(plot_tasks), len(subj_regions), figsize=(4*len(subj_regions), 4*len(plot_tasks)),
                                    layout='constrained', sharey=True)
            fig.suptitle('Regional {} for Subject {} by Region\n{}'.format(plot_title, subj, signal_label))

            axs = np.reshape(axs, (len(plot_tasks), len(subj_regions)))
            handles = []
            labels = []

            for i, region in enumerate(subj_regions):
                for j, task in enumerate(plot_tasks):
                    ax = axs[j,i]
                    ax.set_title('{}, {}'.format(region, plot_beh_names[task]))

                    for k, epoch in enumerate(plot_epochs):
                        stacked_ps = get_task_spectra(stacked_spectra, signal_type, subj, task, epoch, region)
                        if stacked_ps is None:
                            continue

                        plot_stacked_spectra(ax, stacked_ps, epoch, color='C{}'.format(k), integrated=integrated, logy=logy, relative=relative)

                    plot_handles, plot_labels = ax.get_legend_handles_labels()
                    if len(plot_handles) > 0:
                        handles = plot_handles
                        labels = plot_labels
                    if i == 0:
                        ax.set_ylabel(y_label)

            if len(handles) > 0:
                fig.legend(handles, labels, loc='outside right center')

            save_spectra_plot(fig, subj, '{} by region {}'.format(plot_type, signal_type))


# %% Calculate Coherence and Phase Angle between signals

recalculate = False
reprocess_sess_ids = []

signal_types = ['z_dff_iso_baseline_fband']
f_min = 0.005

filename = 'wm_bandit_coherence_phase_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        coherence_phase_data = saved_data['coherence_phase_data']
        freqs = saved_data['freqs']
        recalculate = saved_data['f_min'] != f_min

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    coherence_phase_data = {subjid: {task: {} for task in tasks} for subjid in subj_ids}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    region_pairs = get_region_pairs(subj_regions)

    if not subj_id in coherence_phase_data:
        coherence_phase_data[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db

        if not subj_id in sess_ids:
            continue

        if not task in coherence_phase_data[subj_id]:
            coherence_phase_data[subj_id][task] = {}

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if (sess_id in coherence_phase_data[subj_id][task] and not sess_id in reprocess_sess_ids
                and all(signal_type in coherence_phase_data[subj_id][task][sess_id]
                        and all(e in coherence_phase_data[subj_id][task][sess_id][signal_type] for e in epochs)
                        for signal_type in signal_types)):
                continue

            if not sess_id in coherence_phase_data[subj_id][task] or sess_id in reprocess_sess_ids:
                coherence_phase_data[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']

            print('Calculating coherence and phase for subject {} session {}:'.format(subj_id, sess_id))

            for signal_type in signal_types:
                start = time.perf_counter()

                if not signal_type in coherence_phase_data[subj_id][task][sess_id]:
                    coherence_phase_data[subj_id][task][sess_id][signal_type] = {}

                sess_signals = {}
                for region in subj_regions:
                    if (not region in fp_data['processed_signals']
                        or not signal_type in fp_data['processed_signals'][region]):
                        continue

                    signal = fp_data['processed_signals'][region][signal_type]
                    if np.any(np.isfinite(signal)):
                        sess_signals[region] = signal

                for epoch in epochs:
                    if epoch in coherence_phase_data[subj_id][task][sess_id][signal_type]:
                        continue

                    coherence_phase_data[subj_id][task][sess_id][signal_type][epoch] = {}

                    if epoch == 'All':
                        t_sel = np.full_like(t, True, dtype=bool)
                    else:
                        t_sel = epoch_t_labels[subj_id][task][sess_id] == epoch

                    for region_i, region_j in region_pairs:
                        if not region_i in sess_signals or not region_j in sess_signals:
                            continue

                        try:
                            freqs, coherence, phase = fp_utils.calc_coherence_phase(sess_signals[region_i], sess_signals[region_j],
                                                                                       dt, f_min=f_min, t_sel=t_sel)
                        except ValueError as err:
                            print('Could not calculate coherence and phase for subject {}, session {}, task {}, pair {} / {}, signal {}, epoch {}: {}'.format(
                                subj_id, sess_id, task, region_i, region_j, signal_type, epoch, err))
                            continue

                        # Phase is region_j minus region_i in radians; reversing the pair negates phase.
                        coherence_phase_data[subj_id][task][sess_id][signal_type][epoch][(region_i, region_j)] = {'coherence': coherence, 'phase': phase}

                print('  For {} in {:.1f} s'.format(signal_type, time.perf_counter()-start))

            with open(save_path, 'wb') as f:
                pickle.dump({'coherence_phase_data': coherence_phase_data,
                             'freqs': freqs,
                             'f_min': f_min},
                            f)


# %% Plot coherence and phase angle

save_plots = False
show_plots = True

plot_ind_subj = False
plot_comb_subj = True
plot_sep_tasks = False
plot_sep_epochs = True
plot_all_epoch = False
plot_by_epoch = True
plot_by_region = True
plot_by_pair = True
plot_bilateral = True
plot_band_avg = True

freq_range = [0, 10]
f_band_edges = np.concat([np.array([freq_range[0]]), 10**np.linspace(np.log10(0.1), np.log10(freq_range[1]), 25)])
f_bands = [[v1, v2] for v1, v2 in zip(f_band_edges[:-1], f_band_edges[1:])]

band_array = np.array(f_bands)
band_centers = band_array[:,0] + np.diff(band_array, axis=1)[:,0]/2

min_phase_coherence = 0 # increase to exclude weakly coherent frequencies from phase averages
ax_size = 3

plot_tasks = tasks.copy() if plot_sep_tasks else []
plot_tasks.append('all')

plot_epochs = epochs.copy() if plot_sep_epochs else ['All']
if plot_sep_epochs and not plot_all_epoch and 'All' in plot_epochs:
    plot_epochs.remove('All')

plot_subj = []
if plot_ind_subj:
    plot_subj.extend(subj_ids.copy())
if plot_comb_subj:
    plot_subj.append('all')

plot_signals = ['z_dff_iso_baseline_fband']

def average_frequency_band(values, band_freqs, circular=False):
    values = np.where(np.isfinite(values), values, np.nan)
    # valid_width = utils.nan_trapezoid(np.where(np.isfinite(values), 1., np.nan), band_freqs)
    # if not np.isfinite(valid_width) or valid_width <= 0:
    #     return np.nan

    # if circular:
    #     real = utils.nan_trapezoid(np.cos(values), band_freqs) / valid_width
    #     imag = utils.nan_trapezoid(np.sin(values), band_freqs) / valid_width
    #     return np.angle(real + 1j*imag) if np.hypot(real, imag) > 1e-12 else np.nan
    # else:
    #     return utils.nan_trapezoid(values, band_freqs) / valid_width

    if not np.any(np.isfinite(values)):
        return np.nan

    if circular:
        real = np.nanmean(np.cos(values))
        imag = np.nanmean(np.sin(values))
        return np.angle(real + 1j*imag) if np.hypot(real, imag) > 1e-12 else np.nan
    else:
        return np.nanmean(values)

def average_coherence_phase_bands(pair_data, unsigned_phase=False):
    coherence = pair_data['coherence']
    phase = np.where(np.isfinite(coherence) & (coherence > min_phase_coherence), pair_data['phase'], np.nan)
    if unsigned_phase:
        phase = np.abs(np.angle(np.exp(1j*phase)))

    band_coherence = []
    band_phase = []
    for band in f_bands:
        freq_sel = (freqs >= band[0]) & (freqs <= band[1])
        band_freqs = freqs[freq_sel]
        band_coherence.append(average_frequency_band(coherence[freq_sel], band_freqs))
        band_phase.append(average_frequency_band(phase[freq_sel], band_freqs, circular=not unsigned_phase))

    return {'coherence': np.array(band_coherence), 'phase': np.array(band_phase)}

def average_coherence_phase(pair_data, unsigned_phase=False):
    coherence = np.stack([d['coherence'] for d in pair_data], axis=0)
    phase = np.stack([d['phase'] for d in pair_data], axis=0)
    phase = np.where(np.isfinite(coherence) & (coherence > min_phase_coherence), phase, np.nan)
    if unsigned_phase:
        phase = np.abs(np.angle(np.exp(1j*phase)))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        coherence_avg = np.nanmean(coherence, axis=0)
        if unsigned_phase:
            phase_avg = np.nanmean(phase, axis=0)
        else:
            phase_vector = np.nanmean(np.exp(1j*phase), axis=0)
            phase_avg = np.where(np.abs(phase_vector) > 1e-12, np.angle(phase_vector), np.nan)

    return {'coherence': coherence_avg, 'phase': phase_avg}

def get_coherence_phase_summary(signal_type, subj, task, epoch, region_i, region_j, bilateral=False):
    summary_key = (signal_type, subj, task, epoch, region_i, region_j, bilateral)
    if summary_key in coherence_phase_summaries:
        return coherence_phase_summaries[summary_key]

    reg_mapping = get_subj_region_mapping([region_i, region_j])
    unsigned_phase = bilateral or reg_mapping[region_i] == reg_mapping[region_j]
    pair_data = []
    plot_pair = (region_i, region_j)
    reverse_pair = (region_j, region_i)
    pair_tasks = tasks if task == 'all' else [task]
    pair_subjects = subj_ids if subj == 'all' else [subj]
    stacked_data = stacked_region_coherence_phase if subj == 'all' or bilateral else stacked_coherence_phase

    for subj_id in pair_subjects:
        subj_pair_data = []
        for pair_task in pair_tasks:
            epoch_data = stacked_data[signal_type][subj_id][pair_task][epoch]
            if plot_pair in epoch_data:
                subj_pair_data.extend(epoch_data[plot_pair])
            elif reverse_pair in epoch_data:
                subj_pair_data.extend([{'coherence': d['coherence'], 'phase': -d['phase']} for d in epoch_data[reverse_pair]])

        if len(subj_pair_data) > 0:
            if subj == 'all':
                pair_data.append(average_coherence_phase(subj_pair_data, unsigned_phase=unsigned_phase))
            else:
                pair_data.extend(subj_pair_data)

    if len(pair_data) == 0:
        coherence_phase_summaries[summary_key] = None
        return None

    avg = average_coherence_phase(pair_data, unsigned_phase=unsigned_phase)
    avg['unsigned_phase'] = unsigned_phase
    coherence = np.stack([d['coherence'] for d in pair_data], axis=0)
    phase = np.stack([d['phase'] for d in pair_data], axis=0)
    phase = np.where(np.isfinite(coherence) & (coherence > min_phase_coherence), phase, np.nan)
    if unsigned_phase:
        phase_residuals = np.abs(np.angle(np.exp(1j*phase))) - avg['phase']
    else:
        # Angular residuals give errors around the circular mean without a jump at +/- pi.
        phase_residuals = np.angle(np.exp(1j*(phase - avg['phase'])))
    err = {'coherence': utils.stderr(coherence, axis=0), 'phase': utils.stderr(phase_residuals, axis=0)}
    coherence_phase_summaries[summary_key] = (avg, err)
    return avg, err

def unwrap_valid_phase(phase):
    unwrapped = np.full_like(phase, np.nan, dtype=float)
    valid_idxs = np.flatnonzero(np.isfinite(phase))
    for idxs in np.split(valid_idxs, np.flatnonzero(np.diff(valid_idxs) > 1)+1):
        if len(idxs) > 0:
            unwrapped[idxs] = np.unwrap(phase[idxs])
    return unwrapped

def plot_coherence_phase(ax, summary, metric, label, color):
    ax.set_xscale('log')
    ax.set_xlabel('Frequency Band Center (Hz)' if plot_band_avg else 'Frequency (Hz)')
    plot_utils.plot_dashlines(0, dir='h', ax=ax)

    # if metric == 'coherence':
    #     ax.set_ylim(-0.05, 1)

    if summary is None:
        return

    avg, err = summary
    plot_freqs = band_centers if plot_band_avg else freqs
    freq_sel = (plot_freqs > 0) & (plot_freqs >= freq_range[0]) & (plot_freqs <= freq_range[1])
    plot_avg = avg[metric][freq_sel]
    if metric == 'phase' and not avg['unsigned_phase']:
        plot_avg = unwrap_valid_phase(plot_avg)

    if np.any(np.isfinite(plot_avg)):
        line, _ = plot_utils.plot_shaded_error(plot_freqs[freq_sel], plot_avg, y_err=err[metric][freq_sel], ax=ax, label=label, color=color)
        # if plot_band_avg:
        #     line.set_marker('o')

def finish_coherence_phase_plot(fig, subj, plot_name, legend_title):
    handles = {}
    for ax in fig.axes:
        plot_handles, plot_labels = ax.get_legend_handles_labels()
        handles.update(zip(plot_labels, plot_handles))
    if len(handles) > 0:
        fig.legend(list(handles.values()), list(handles.keys()), loc='outside right center', title=legend_title)

    if save_plots:
        plot_name = '{} {}'.format('band averaged' if plot_band_avg else 'raw', plot_name)
        fpah.save_fig(fig, fpah.get_figure_save_path('Coherence and Phase', subj, plot_name), bbox_inches='tight')
    if not show_plots:
        plt.close(fig)

# Keep anatomical pairs for individual subjects and average bilateral pairs within each session for population plots.
stacked_coherence_phase = {sig: {s: {t: {e: {} for e in epochs} for t in tasks} for s in subj_ids} for sig in plot_signals}
stacked_region_coherence_phase = {sig: {s: {t: {e: {} for e in epochs} for t in tasks} for s in subj_ids} for sig in plot_signals}

for subj_id in subj_ids:
    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for task in tasks:
        for sess_data in coherence_phase_data.get(subj_id, {}).get(task, {}).values():
            for signal_type in plot_signals:
                for epoch in epochs:
                    region_pair_data = {}
                    for pair, pair_data in sess_data.get(signal_type, {}).get(epoch, {}).items():
                        region_i, region_j = pair
                        if not region_i in reg_mapping or not region_j in reg_mapping:
                            continue

                        general_i, general_j = reg_mapping[region_i], reg_mapping[region_j]
                        if plot_band_avg:
                            pair_data = average_coherence_phase_bands(pair_data, unsigned_phase=general_i == general_j)
                        stacked_coherence_phase[signal_type][subj_id][task][epoch].setdefault(pair, []).append(pair_data)
                        if general_i == general_j:
                            side_i = implant_info[subj_id][region_i]['side']
                            side_j = implant_info[subj_id][region_j]['side']
                            if {side_i, side_j} != {'left', 'right'}:
                                continue
                            # Fold each bilateral phase before averaging so opposite orderings reinforce.
                            general_pair = (general_i, general_i)
                            general_data = {'coherence': pair_data['coherence'], 'phase': np.abs(np.angle(np.exp(1j*pair_data['phase'])))}
                        elif all_regions.index(general_i) < all_regions.index(general_j):
                            general_pair = (general_i, general_j)
                            general_data = pair_data
                        else:
                            general_pair = (general_j, general_i)
                            general_data = {'coherence': pair_data['coherence'], 'phase': -pair_data['phase']}
                        region_pair_data.setdefault(general_pair, []).append(general_data)

                    for pair, pair_data in region_pair_data.items():
                        stacked_region_coherence_phase[signal_type][subj_id][task][epoch].setdefault(pair, []).append(
                            average_coherence_phase(pair_data, unsigned_phase=pair[0] == pair[1]))

coherence_phase_summaries = {}

for signal_type in plot_signals:
    signal_title = fpah.get_signal_type_labels(signal_type)[0]

    for subj in plot_subj:
        subj_regions = all_regions if subj == 'all' else get_subj_regions(subj)
        n_regions = len(subj_regions)
        if n_regions == 0 or len(plot_epochs) == 0:
            continue

        colors = {r: region_colors[r] if subj == 'all' else 'C{}'.format(i) for i,r in enumerate(subj_regions)}
        bilat_subjects = subj_ids if subj == 'all' else [subj]
        plot_bilat_regions = [r for r in all_regions
                              if any((r,r) in stacked_region_coherence_phase[signal_type][s][t][e]
                                     for s in bilat_subjects for t in tasks for e in plot_epochs)]

        for metric, metric_title, y_label in [('coherence', 'Coherence', 'Magnitude-squared Coherence'),
                                             ('phase', 'Unwrapped Phase', 'Phase Difference (rad)')]:
            if plot_band_avg:
                metric_title = 'Band-averaged {}'.format(metric_title)
            metric_signal_title = '{}\nPhase = compared - primary'.format(signal_title) if metric == 'phase' else signal_title
            if plot_by_epoch and n_regions > 1:
                # Primary regions in rows, epochs in columns, compared regions as lines.
                for task in plot_tasks:
                    fig, axs = plt.subplots(n_regions, len(plot_epochs), squeeze=False, sharex=True, sharey=True,
                                            figsize=(ax_size*len(plot_epochs)+2, ax_size*n_regions), layout='constrained')
                    fig.suptitle('Regional {} for Subject {} in {} by Epoch\n{}'.format(metric_title, subj, plot_beh_names[task], metric_signal_title))

                    for i, region_i in enumerate(subj_regions):
                        for j, epoch in enumerate(plot_epochs):
                            ax = axs[i,j]
                            ax.set_title('{}, {}'.format(region_i, epoch))
                            for region_j in subj_regions:
                                if region_i == region_j:
                                    continue
                                summary = get_coherence_phase_summary(signal_type, subj, task, epoch, region_i, region_j)
                                plot_coherence_phase(ax, summary, metric, region_j, colors[region_j])
                            if j == 0:
                                ax.set_ylabel(y_label)

                    finish_coherence_phase_plot(fig, subj, '{} by epoch {} {}'.format(metric, task, signal_type), 'Compared Region')

            if plot_by_region and n_regions > 1:
                # Tasks in rows, primary regions in columns, compared regions as lines.
                for epoch in plot_epochs:
                    fig, axs = plt.subplots(len(plot_tasks), n_regions, squeeze=False, sharex=True, sharey=True,
                                            figsize=(ax_size*n_regions+2, ax_size*len(plot_tasks)), layout='constrained')
                    fig.suptitle('Regional {} for Subject {} during {} by Region\n{}'.format(metric_title, subj, epoch, metric_signal_title))

                    for i, task in enumerate(plot_tasks):
                        for j, region_i in enumerate(subj_regions):
                            ax = axs[i,j]
                            ax.set_title('{}, {}'.format(region_i, plot_beh_names[task]))
                            for region_j in subj_regions:
                                if region_i == region_j:
                                    continue
                                summary = get_coherence_phase_summary(signal_type, subj, task, epoch, region_i, region_j)
                                plot_coherence_phase(ax, summary, metric, region_j, colors[region_j])
                            if j == 0:
                                ax.set_ylabel(y_label)

                    finish_coherence_phase_plot(fig, subj, '{} by region {} {}'.format(metric, epoch, signal_type), 'Compared Region')

            if plot_by_pair and n_regions > 1:
                for task in (plot_tasks if plot_sep_epochs else ['all']):
                    if metric == 'phase':
                        n_rows, n_cols = n_regions, n_regions
                        plot_pairs = [(region_i, region_j) for region_i in subj_regions for region_j in subj_regions]
                    else:
                        plot_pairs = get_region_pairs(subj_regions)
                        n_cols = min(n_regions, len(plot_pairs))
                        n_rows = int(np.ceil(len(plot_pairs)/n_cols))

                    fig, axs = plt.subplots(n_rows, n_cols, squeeze=False, sharex=True, sharey=True,
                                            figsize=(ax_size*n_cols+2, ax_size*n_rows), layout='constrained')
                    fig.suptitle('Regional {} for Subject {} in {} by Pair\n{}'.format(metric_title, subj, plot_beh_names[task], metric_signal_title))

                    for i, ax in enumerate(axs.flat):
                        if i >= len(plot_pairs):
                            ax.set_axis_off()
                            continue
                        region_i, region_j = plot_pairs[i]
                        if region_i == region_j:
                            ax.set_axis_off()
                            continue

                        ax.set_title('{} vs {}'.format(region_i, region_j))
                        ax.set_ylabel(y_label)
                        ax.tick_params(axis='both', which='both', labelbottom=True, labelleft=True)
                        line_groups = plot_epochs if plot_sep_epochs else plot_tasks
                        for j, group in enumerate(line_groups):
                            pair_task = task if plot_sep_epochs else group
                            pair_epoch = group if plot_sep_epochs else 'All'
                            label = group if plot_sep_epochs else plot_beh_names[group]
                            summary = get_coherence_phase_summary(signal_type, subj, pair_task, pair_epoch, region_i, region_j)
                            plot_coherence_phase(ax, summary, metric, label, 'C{}'.format(j))

                    finish_coherence_phase_plot(fig, subj, '{} by pair {} {}'.format(metric, task, signal_type),
                                                'Epoch' if plot_sep_epochs else 'Task')

            if plot_bilateral and len(plot_bilat_regions) > 0:
                bilat_metric_title = 'Unsigned Phase Separation' if metric == 'phase' else metric_title
                if plot_band_avg and metric == 'phase':
                    bilat_metric_title = 'Band-averaged {}'.format(bilat_metric_title)
                bilat_y_label = 'Phase Separation (rad)' if metric == 'phase' else y_label
                if plot_sep_epochs:
                    # Tasks in rows, bilateral regions in columns, epochs as lines.
                    fig, axs = plt.subplots(len(plot_tasks), len(plot_bilat_regions), squeeze=False, sharex=True, sharey=True,
                                            figsize=(ax_size*len(plot_bilat_regions)+2, ax_size*len(plot_tasks)), layout='constrained')
                    fig.suptitle('Bilateral {} for Subject {} by Region\n{}'.format(bilat_metric_title, subj, signal_title))

                    for i, task in enumerate(plot_tasks):
                        for j, region in enumerate(plot_bilat_regions):
                            ax = axs[i,j]
                            ax.set_title('{}, {}'.format(region, plot_beh_names[task]))
                            for k, epoch in enumerate(plot_epochs):
                                summary = get_coherence_phase_summary(signal_type, subj, task, epoch, region, region, bilateral=True)
                                plot_coherence_phase(ax, summary, metric, epoch, 'C{}'.format(k))
                            if j == 0:
                                ax.set_ylabel(bilat_y_label)
                            if metric == 'phase':
                                ax.set_ylim(0, np.pi)

                    finish_coherence_phase_plot(fig, subj, 'bilateral {} by region {}'.format(metric, signal_type), 'Epoch')

                    # Tasks in rows, epochs in columns, bilateral regions as lines.
                    fig, axs = plt.subplots(len(plot_tasks), len(plot_epochs), squeeze=False, sharex=True, sharey=True,
                                            figsize=(ax_size*len(plot_epochs)+2, ax_size*len(plot_tasks)), layout='constrained')
                    fig.suptitle('Bilateral {} for Subject {} by Epoch\n{}'.format(bilat_metric_title, subj, signal_title))

                    for i, task in enumerate(plot_tasks):
                        for j, epoch in enumerate(plot_epochs):
                            ax = axs[i,j]
                            ax.set_title('{}, {}'.format(plot_beh_names[task], epoch))
                            for region in plot_bilat_regions:
                                summary = get_coherence_phase_summary(signal_type, subj, task, epoch, region, region, bilateral=True)
                                plot_coherence_phase(ax, summary, metric, region, region_colors[region])
                            if j == 0:
                                ax.set_ylabel(bilat_y_label)
                            if metric == 'phase':
                                ax.set_ylim(0, np.pi)

                    finish_coherence_phase_plot(fig, subj, 'bilateral {} by epoch {}'.format(metric, signal_type), 'Region')

                else:
                    fig, axs = plt.subplots(1, len(plot_tasks), squeeze=False, sharex=True, sharey=True,
                                            figsize=(ax_size*len(plot_tasks)+2, ax_size), layout='constrained')
                    fig.suptitle('Bilateral {} for Subject {}\n{}'.format(bilat_metric_title, subj, signal_title))

                    for i, task in enumerate(plot_tasks):
                        ax = axs[0,i]
                        ax.set_title(plot_beh_names[task])
                        for region in plot_bilat_regions:
                            summary = get_coherence_phase_summary(signal_type, subj, task, 'All', region, region, bilateral=True)
                            plot_coherence_phase(ax, summary, metric, region, region_colors[region])
                        if i == 0:
                            ax.set_ylabel(bilat_y_label)
                        if metric == 'phase':
                            ax.set_ylim(0, np.pi)

                    finish_coherence_phase_plot(fig, subj, 'bilateral {} {}'.format(metric, signal_type), 'Region')

if show_plots:
    plt.show()



# %% Estimate auto-correlation timescales by subject

# Will use the dominant decay timescales calculated with an AR(2) process. For real roots, this will be the largest timescale,
# for complex roots, this will be the magnitude of the root.

plot_fits = False

region_lags = {'PL': 12, 'NAc': 6, 'DMS': 2, 'DLS': 2, 'TS': 2}
region_ar_lags = {'PL': 4, 'NAc': 0.5, 'DMS': 0.2, 'DLS': 0.2, 'TS': 0.2}

recalculate = False
recalculate_ar_params = False
reprocess_sess_ids = []
reprocess_reg = []

signal_type = 'dff_iso_baseline_fband'

filename = 'wm_bandit_autocorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        auto_corrs = saved_data['auto_corrs']
        auto_corr_taus = saved_data['auto_corr_taus']
        auto_corr_lags = saved_data['auto_corr_lags']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    auto_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    auto_corr_taus = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    auto_corr_lags = {}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    subj_reprocess_reg = [s for s in subj_regions if reg_mapping[s] in reprocess_reg]

    if not subj_id in auto_corrs:
        auto_corrs[subj_id] = {task: {} for task in tasks}
        auto_corr_taus[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if (sess_id in auto_corrs[subj_id][task] and not sess_id in reprocess_sess_ids
                and len(subj_reprocess_reg) == 0 and not recalculate_ar_params):
                continue

            if sess_id not in auto_corrs[subj_id][task]:
                auto_corrs[subj_id][task][sess_id] = {}
                auto_corr_taus[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']

            print('Calculating auto-correlation timescales for subject {} session {}'.format(subj_id, sess_id))

            if plot_fits:
                fig, axs = plt.subplots(len(epochs), len(subj_regions), figsize=(5*len(subj_regions),4*len(epochs)))
                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

            for i, region in enumerate(subj_regions):
                if not region in fp_data['processed_signals'] or (region in auto_corrs[subj_id][task][sess_id]
                                                                  and not region in subj_reprocess_reg
                                                                  and not recalculate_ar_params):
                    continue
                signal = fp_data['processed_signals'][region][signal_type]
                sn = fp_utils.get_ar_noise(signal)

                if region not in auto_corrs[subj_id][task][sess_id]:
                    auto_corrs[subj_id][task][sess_id][region] = {}
                    auto_corr_taus[subj_id][task][sess_id][region] = {}

                if all(np.isnan(signal)):
                    continue

                for j, e in enumerate(epochs):

                    if e == 'All':
                        t_sel = np.full_like(t, True, dtype=bool)
                    else:
                        t_sel = epoch_t_labels[subj_id][task][sess_id] == e

                    if e in auto_corrs[subj_id][task][sess_id][region]:
                        auto_corr = auto_corrs[subj_id][task][sess_id][region][e]
                        corr_lags = auto_corr_lags[region]
                    else:
                        auto_corr, corr_lags = fp_utils.correlate(signal, signal, dt, neg_lag=0, pos_lag=region_lags[reg_mapping[region]], t_sel=t_sel)

                    if all(np.isnan(auto_corr)):
                        continue

                    # calculate auto-regressive decay timescale
                    ar_lags = int(region_ar_lags[reg_mapping[region]] / dt)
                    g_ar1 = fp_utils.get_acf_ar_params(auto_corr, corr_lags, 1, ar_lags, sn=sn)
                    ar1_mode = fp_utils.get_ar_modes(g_ar1, dt)[0]
                    g_ar2 = fp_utils.get_acf_ar_params(auto_corr, corr_lags, 2, ar_lags, sn=sn)
                    ar2_modes = fp_utils.get_ar_modes(g_ar2, dt)
                    max_ar2_mode = {'tau': 0}
                    for m in ar2_modes:
                        if m['tau'] > max_ar2_mode['tau']:
                            max_ar2_mode = m

                    auto_corrs[subj_id][task][sess_id][region][e] = auto_corr
                    auto_corr_taus[subj_id][task][sess_id][region][e] = {'tau': max_ar2_mode['tau'],
                                                                         'freq': max_ar2_mode['freq'],
                                                                         'tau_ar1': ar1_mode['tau']}
                    auto_corr_lags[region] = corr_lags

                    if plot_fits:
                        ax = axs[j,i]
                        ax.set_title('{}, {}'.format(region, e))
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        ax.plot(corr_lags, auto_corr, label='Autocorr')

                        ar2_autocorr = fp_utils.simulate_ar_acf(g_ar2, len(corr_lags)-1)
                        ar2to1_autocorr = fp_utils.simulate_ar_acf([max_ar2_mode['mag']], len(corr_lags)-1)

                        ax.plot(corr_lags, ar2_autocorr, label='Full AR(2)')
                        ax.plot(corr_lags, ar2to1_autocorr, label='AR(2) Tau')

                        ax.legend()

            if plot_fits:
                plt.show()

            with open(save_path, 'wb') as f:
                pickle.dump({'auto_corrs': auto_corrs,
                             'auto_corr_taus': auto_corr_taus,
                             'auto_corr_lags': auto_corr_lags},
                            f)

# Construct pandas dataframe of the taus and find the mean and median values
rows = []

for subj_id, subj_dict in auto_corr_taus.items():
    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    for task, task_dict in subj_dict.items():
        for sess_id, sess_dict in task_dict.items():
            trial_data = loc_db.get_behavior_data(sess_id)
            for region, region_dict in sess_dict.items():
                for epoch, params in region_dict.items():
                    rows.append({
                        'subj_id': subj_id,
                        'task': task,
                        'sess_id': sess_id,
                        'region': region,
                        'general_region': reg_mapping[region],
                        'epoch': epoch,
                        'tau': params['tau'],
                        'freq': params['freq'],
                        'tau_ar1': params['tau_ar1'],
                        'n_trials': len(trial_data),
                        'n_rewards': np.sum(trial_data['rewarded'])
                    })

auto_corrs_df = pd.DataFrame(rows)
auto_corrs_df['log_tau'] = auto_corrs_df['tau'].apply(lambda t: np.log(t))
auto_corrs_df['log_tau_ar1'] = auto_corrs_df['tau_ar1'].apply(lambda t: np.log(t))

# average across duplicate regions per session
norm_auto_corrs = auto_corrs_df.groupby(['subj_id', 'sess_id', 'general_region', 'epoch'])[['log_tau', 'freq', 'log_tau_ar1']].agg('mean').reset_index()
norm_auto_corrs['tau'] = norm_auto_corrs['log_tau'].apply(lambda t: np.exp(t))
norm_auto_corrs['tau_ar1'] = norm_auto_corrs['log_tau_ar1'].apply(lambda t: np.exp(t))

norm_auto_corrs_task = auto_corrs_df.groupby(['subj_id', 'sess_id', 'general_region', 'epoch', 'task'])[['log_tau', 'freq']].agg('mean').reset_index()
norm_auto_corrs_task['tau'] = norm_auto_corrs_task['log_tau'].apply(lambda t: np.exp(t))

avg_params_task = auto_corrs_df.groupby(['subj_id', 'region', 'task', 'epoch'])[['log_tau', 'freq']].agg(['median', 'mean']).reset_index()
avg_params_task = avg_params_task.rename(columns={'log_tau': 'tau'}, level=0)
avg_params_task.loc[:,('tau','median')] = avg_params_task.loc[:,('tau','median')].apply(lambda t: np.exp(t))
avg_params_task.loc[:,('tau','mean')] = avg_params_task.loc[:,('tau','mean')].apply(lambda t: np.exp(t))

avg_params_all = auto_corrs_df.groupby(['subj_id', 'region', 'epoch'])[['log_tau', 'freq']].agg(['median', 'mean']).reset_index()
avg_params_all = avg_params_all.rename(columns={'log_tau': 'tau'}, level=0)
avg_params_all.loc[:,('tau','median')] = avg_params_all.loc[:,('tau','median')].apply(lambda t: np.exp(t))
avg_params_all.loc[:,('tau','mean')] = avg_params_all.loc[:,('tau','mean')].apply(lambda t: np.exp(t))

# %% Plot autocorrelation taus

melt_autocorr_df = auto_corrs_df.melt(['subj_id', 'region', 'general_region', 'task', 'epoch'], ['tau', 'freq', 'tau_ar1'], 'param')

all_epoch_autocorr = melt_autocorr_df[melt_autocorr_df['epoch'] == 'All']

# scatter plot of all subjects by general region colored by task
p = sb.catplot(all_epoch_autocorr, x='subj_id', y='value', hue='task', col='general_region',
               col_order=all_regions, row='param', dodge=True, sharex=False, sharey='row', height=4, aspect=1)
axs = p.axes
for i in range(axs.shape[1]):
    axs[0,i].set_yscale('log')
    axs[0,i].set_title('{}, Decay Tau'.format(all_regions[i]))
    axs[1,i].set_title('{}, Decay Freq'.format(all_regions[i]))
axs[0,0].set_ylabel('Decay Tau (s)')
axs[1,0].set_ylabel('Decay Freq (Hz)')

# boxplot of average over all subjects
fig, axs = plt.subplots(len(epochs), 2, layout='constrained', figsize=(8,4*len(epochs)))
for i, e in enumerate(epochs):
    epoch_autocorrs = norm_auto_corrs[norm_auto_corrs['epoch'] == e]
    sb.boxplot(epoch_autocorrs, x='general_region', y='tau', order=all_regions, ax=axs[i,0],
               hue='general_region', palette=region_colors, whis=(2.5, 97.5), log_scale=(False, True), showfliers=False)
    sb.boxplot(epoch_autocorrs, x='general_region', y='freq', order=all_regions, ax=axs[i,1],
               hue='general_region', palette=region_colors, whis=(2.5, 97.5), showfliers=False)

    axs[i,0].set_title('Taus, {}'.format(e))
    axs[i,0].set_ylabel('Tau (s)')
    axs[i,0].set_xlabel('Region')
    axs[i,1].set_title('Freqs, {}'.format(e))
    axs[i,1].set_ylabel('Freq (Hz)')
    axs[i,1].set_xlabel('Region')

fig.suptitle('Dominant AR(2) Modes, All Subjects, By Epoch')

fig, ax = plt.subplots(1,1, layout='constrained')
all_epoch_autocorr = norm_auto_corrs[norm_auto_corrs['epoch'] == 'All']
p = sb.boxplot(all_epoch_autocorr, x='general_region', y='tau_ar1', order=all_regions, ax=ax,
           hue='general_region', palette=region_colors, log_scale=(False, True), whis=(2.5, 97.5), showfliers=False)
ax = p.axes
ax.set_ylabel('Decay Tau (s)')
ax.set_xlabel('Region')

# boxplot of average over all subjects
fig, axs = plt.subplots(len(all_regions), 2, layout='constrained', figsize=(10,4*len(epochs)))
for i, r in enumerate(all_regions):
    region_autocorrs = norm_auto_corrs_task[norm_auto_corrs_task['general_region'] == r]
    sb.boxplot(region_autocorrs, x='epoch', y='tau', hue='task', order=epochs, ax=axs[i,0], whis=(2.5, 97.5), log_scale=(False, True))
    sb.boxplot(region_autocorrs, x='epoch', y='freq', hue='task', order=epochs, ax=axs[i,1], whis=(2.5, 97.5))

    axs[i,0].set_title('Taus, {}'.format(r))
    axs[i,0].set_ylabel('Tau (s)')
    axs[i,0].set_xlabel('Region')
    axs[i,0].set_xticklabels(axs[i,0].get_xticklabels(), rotation=45, ha='right')
    axs[i,1].set_title('Freqs, {}'.format(r))
    axs[i,1].set_ylabel('Freq (Hz)')
    axs[i,1].set_xlabel('Region')
    axs[i,1].set_xticklabels(axs[i,0].get_xticklabels(), rotation=45, ha='right')

    axs[i,1].legend()
    handles, labels = axs[i,1].get_legend_handles_labels()
    axs[i,1].legend().remove()

fig.legend(handles, labels, loc='outside right center')

fig.suptitle('Dominant AR(2) Modes, All Subjects, By Region')

# %% stack auto-correlations by epoch
stacked_autocorrs = {s: {t: {e: {r: [] for r in get_subj_regions(s)} for e in epochs} for t in tasks} for s in subj_ids}
stacked_autocorrs['all'] = {t: {e: {r: [] for r in all_regions} for e in epochs} for t in tasks}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    subj_unique_regions = np.unique(list(reg_mapping.values()))

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:
            for e in epochs:

                all_region_autocorrs = {r: [] for r in subj_unique_regions}

                for region in subj_regions:
                    if not region in auto_corrs[subj_id][task][sess_id] or not e in auto_corrs[subj_id][task][sess_id][region]:
                        continue

                    sess_reg_ac = auto_corrs[subj_id][task][sess_id][region][e]

                    stacked_autocorrs[subj_id][task][e][region].append(sess_reg_ac)
                    all_region_autocorrs[reg_mapping[region]].append(sess_reg_ac)

                for region in subj_unique_regions:
                    if len(all_region_autocorrs[region]) > 0:
                        avg_autocorr = np.nanmean(np.stack(all_region_autocorrs[region], axis=0), axis=0)
                        stacked_autocorrs['all'][task][e][region].append(avg_autocorr)

# %% plot autocorrelations
plot_ind_subj = False

plot_lim = 2
reg_plot_lim = {'PL': 6, 'NAc': 3, 'DMS': 1, 'DLS': 1, 'TS': 1}
plot_sep_tasks = True
plot_tasks = tasks.copy() if plot_sep_tasks else []
plot_tasks.append('all')

plot_sep_epochs = True
plot_all_epoch = False
plot_epochs = epochs.copy() if plot_sep_epochs else ['All']
if not plot_all_epoch:
    plot_epochs.remove('All')

plot_subj = []
if plot_ind_subj:
    plot_subj.extend(subj_ids.copy())
plot_subj.append('all')

# plot each epoch separately
for subj in plot_subj:
    if subj == 'all':
        subj_regions = all_regions
    else:
        subj_regions = get_subj_regions(subj)
    reg_mapping = get_subj_region_mapping(subj_regions)

    fig, axs = plt.subplots(len(plot_tasks), len(plot_epochs), figsize=(4*len(plot_epochs), 4*len(plot_tasks)),
                            layout='constrained', sharey=True)
    fig.suptitle('Regional Autocorrelations for Subject {} by Epoch'.format(subj))

    axs = np.reshape(axs, (len(plot_tasks), len(plot_epochs)))

    for i, epoch in enumerate(plot_epochs):
        for j, task in enumerate(plot_tasks):
            ax = axs[j,i]
            ax.set_title('{}, {}'.format(plot_beh_names[task], epoch))
            plot_utils.plot_dashlines(0, dir='h', ax=ax)

            for k, region in enumerate(subj_regions):
                region_lags = auto_corr_lags[reg_mapping[region]]
                t_sel = region_lags < plot_lim

                if task == 'all':
                    stacked_acs = np.vstack([np.stack(stacked_autocorrs[subj][t][epoch][region], axis=0) for t in tasks])
                else:
                    stacked_acs = np.stack(stacked_autocorrs[subj][task][epoch][region], axis=0)

                if subj == 'all':
                    color = region_colors[region]
                else:
                    color = 'C{}'.format(k)

                plot_utils.plot_psth(region_lags[t_sel], np.nanmean(stacked_acs, axis=0)[t_sel],
                                     utils.stderr(stacked_acs, axis=0)[t_sel], ax, plot_x0=False, label=region, color=color)

            ax.legend()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()
            ax.set_xlabel('Time lag (s)')
            if i == 0:
                ax.set_ylabel('Pearson r')

    fig.legend(handles, labels, loc='outside right center')

# plot each region separately
for subj in plot_subj:
    if subj == 'all':
        subj_regions = all_regions
    else:
        subj_regions = get_subj_regions(subj)
    reg_mapping = get_subj_region_mapping(subj_regions)

    fig, axs = plt.subplots(len(plot_tasks), len(subj_regions), figsize=(4*len(subj_regions), 4*len(plot_tasks)),
                            layout='constrained', sharey=True)
    fig.suptitle('Regional Autocorrelations for Subject {} by Region'.format(subj))

    axs = np.reshape(axs, (len(plot_tasks), len(subj_regions)))

    for i, region in enumerate(subj_regions):
        region_lags = auto_corr_lags[reg_mapping[region]]
        t_sel = region_lags < reg_plot_lim[reg_mapping[region]]

        for j, task in enumerate(plot_tasks):
            ax = axs[j,i]
            ax.set_title('{}, {}'.format(region, plot_beh_names[task]))
            plot_utils.plot_dashlines(0, dir='h', ax=ax)

            for k, epoch in enumerate(plot_epochs):

                if task == 'all':
                    stacked_acs = np.vstack([np.stack(stacked_autocorrs[subj][t][epoch][region], axis=0) for t in tasks])
                else:
                    stacked_acs = np.stack(stacked_autocorrs[subj][task][epoch][region], axis=0)

                plot_utils.plot_psth(region_lags[t_sel], np.nanmean(stacked_acs, axis=0)[t_sel],
                                     utils.stderr(stacked_acs, axis=0)[t_sel], ax, plot_x0=False, label=epoch)

            ax.legend()
            handles, labels = ax.get_legend_handles_labels()
            ax.legend().remove()
            ax.set_xlabel('Time lag (s)')
            if i == 0:
                ax.set_ylabel('Pearson r')

    fig.legend(handles, labels, loc='outside right center')

# %% Compute cross-correlations

recalculate = False
reprocess_sess_ids = []

signal_types = ['dff_iso_baseline_fband'] # 'z_dff_iso_baseline'
region_lpf = {'PL': 4, 'NAc': 6, 'DMS': 8, 'DLS': 8, 'TS': 8}
max_lags = {'raw': 4, 'filt': 1}
dec = 1

filename = 'wm_bandit_xcorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        x_corrs = saved_data['x_corrs']
        x_corr_metadata = saved_data['metadata']
        x_corr_regions = x_corr_metadata['regions']
        corr_lags = x_corr_metadata['corr_lags']

        recalculate = x_corr_metadata['max_lags'] != max_lags or x_corr_metadata['dec'] != dec

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    x_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    x_corr_regions = {}

# get dt information to calculate the number of timesteps in the xcorr results
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']
dec_dt = dt*dec

corr_lags = {}
for s in source_types:
    lag = utils.convert_to_multiple(max_lags[s], dec_dt)
    corr_lags[s] = np.arange(-int(np.round(lag/dec_dt)), int(np.round(lag/dec_dt))+1)*dec_dt

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    x_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)

    if not subj_id in x_corrs:
        x_corrs[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in x_corrs[subj_id][task] and all(s in x_corrs[subj_id][task][sess_id].keys() for s in signal_types):
                continue

            if not sess_id in x_corrs[subj_id][task]:
                x_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']

            print('Calculating cross-correlation for subject {} session {}:'.format(subj_id, sess_id))

            for signal_type in signal_types:
                if not signal_type in x_corrs[subj_id][task][sess_id]:
                    x_corrs[subj_id][task][sess_id][signal_type] = {}

                for source in source_types:
                    start = time.perf_counter()

                    if not source in x_corrs[subj_id][task][sess_id][signal_type]:
                        x_corrs[subj_id][task][sess_id][signal_type][source] = {}

                    sess_signals = {}

                    for region in fp_data['processed_signals']:
                        raw_signal = fp_data['processed_signals'][region][signal_type]
                        if all(np.isnan(raw_signal)):
                            continue

                        if source == 'filt':
                            # perform deconvolution
                            filt_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
                            dec_filt_signal, dec_t = fp_utils.decimate(filt_signal, t, dec, dt=dt)

                            deconv_tau = avg_params_all.loc[(avg_params_all['subj_id'] == subj_id) & (avg_params_all['region'] == region) & (avg_params_all['epoch'] == 'All'),
                                                            ('tau','mean')]
                            deconv_params = fp_utils.get_ar_params_from_modes({'tau': deconv_tau, 'freq': 0}, dec_dt)

                            sess_signals[region] = fp_utils.extract_ar_source(dec_filt_signal, deconv_params)

                        else:
                            dec_signal, dec_t = fp_utils.decimate(raw_signal, t, dec, dt=dt)
                            sess_signals[region] = dec_signal

                    for epoch in epochs:
                        if epoch in x_corrs[subj_id][task][sess_id][signal_type][source] and not sess_id in reprocess_sess_ids:
                            continue
                        else:
                            x_corrs[subj_id][task][sess_id][signal_type][source][epoch] = np.full((n_regions, n_regions, len(corr_lags[source])), np.nan, dtype=float)

                        if epoch == 'All':
                            t_sel = np.full_like(dec_t, True, dtype=bool)
                        else:
                            dec_labels = decimate_epoch_labels(epoch_t_labels[subj_id][task][sess_id], dec)
                            t_sel = dec_labels == epoch

                        for reg_i, region_i in enumerate(subj_regions):
                            if not region_i in sess_signals or all(np.isnan(sess_signals[region_i])):
                                continue

                            signal_i = sess_signals[region_i]

                            for reg_j in np.arange(reg_i, n_regions):
                                region_j = subj_regions[reg_j]
                                if not region_j in sess_signals or all(np.isnan(sess_signals[region_j])):
                                    continue

                                signal_j = sess_signals[region_j]

                                xcorr, _ = fp_utils.correlate(signal_i, signal_j, dec_dt, neg_lag=max_lags[source], pos_lag=max_lags[source], t_sel=t_sel)

                                x_corrs[subj_id][task][sess_id][signal_type][source][epoch][reg_i, reg_j, :] = xcorr
                                if reg_i != reg_j:
                                    x_corrs[subj_id][task][sess_id][signal_type][source][epoch][reg_j, reg_i, :] = np.flip(xcorr)

                    print('  For {} signals in {:.1f} s'.format(source, time.perf_counter()-start))

            with open(save_path, 'wb') as f:
                pickle.dump({'x_corrs': x_corrs,
                             'metadata': {'regions': x_corr_regions,
                                          'max_lags': max_lags,
                                          'corr_lags': corr_lags,
                                          'dec': dec}},
                            f)


# %% Plot cross-correlations

plot_lags = {'raw': 1, 'filt': 1}
t_sels = {s: np.abs(corr_lags[s]) <= plot_lags[s] for s in source_types}

plot_sep_tasks = False
plot_sep_epochs = True
plot_all_epoch = False
plot_ind_subj = False
plot_comb_subj = True
plot_filt = False
plot_diag = False

ax_size = 3

plot_tasks = []
if plot_sep_tasks:
    plot_tasks.extend(tasks.copy())

plot_tasks.append('all')

plot_epochs = epochs.copy() if plot_sep_epochs else ['All']
if not plot_all_epoch:
    plot_epochs.remove('All')

plot_source_types = source_types if plot_filt else ['raw']

if plot_comb_subj:
    comb_xcorr = {t: {s: {st: {e: [] for e in epochs} for st in plot_source_types} for s in signal_types} for t in tasks}
    comb_xcorr_weights = {t: {s: {st: {e: [] for e in epochs} for st in plot_source_types} for s in signal_types} for t in tasks}

for subj_id in subj_ids:

    subj_regions = x_corr_regions[subj_id]
    sorted_regions = sort_subj_regions(subj_regions)
    sorted_region_idxs = [subj_regions.index(r) for r in sorted_regions]

    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]

    # stack across subjects
    if plot_comb_subj:
        for task in tasks:

            if len(x_corrs[subj_id][task]) == 0:
                continue

            for signal_type in signal_types:

                # stack by session across regions
                if plot_comb_subj and task != 'all':

                    for s in x_corrs[subj_id][task].keys():
                        for source in plot_source_types:
                            for e in epochs:
                                sess_xcorr = x_corrs[subj_id][task][s][signal_type][source][e]

                                # get mapping of subject regions to all regions order
                                all_region_idx_mapping = fpah.get_region_idx_mapping(subj_regions, all_regions)
                                dup_counts = fpah.count_corr_pairs(subj_regions, all_regions)

                                if all([v == 1 for v in dup_counts.values()]):
                                    # if no duplicates, can simply use indexing
                                    sess_comb_xcorr = np.full((len(all_regions), len(all_regions), len(corr_lags[source])), np.nan)
                                    sess_comb_xcorr[np.ix_(all_region_idx_mapping, all_region_idx_mapping, np.arange(len(corr_lags[source])))] = sess_xcorr

                                    comb_xcorr[task][signal_type][source][e].append(sess_comb_xcorr)

                                    weights = np.ones_like(sess_comb_xcorr)
                                    weights[np.isnan(sess_comb_xcorr)] = np.nan
                                    comb_xcorr_weights[task][signal_type][source][e].append(weights)
                                else:
                                    # add each duplicate pair separately
                                    for subj_i, map_i in enumerate(all_region_idx_mapping):
                                        for subj_j, map_j in enumerate(all_region_idx_mapping):
                                            # only do each cross-regional comparison once
                                            if subj_j < subj_i:
                                                continue
                                            # don't add bilateral cross-correlations of the same region as auto-correlations
                                            if map_i == map_j and subj_i != subj_j:
                                                continue

                                            sess_comb_xcorr = np.full((len(all_regions), len(all_regions), len(corr_lags[source])), np.nan)
                                            sess_comb_xcorr[map_i,map_j,:] = sess_xcorr[subj_i,subj_j,:]
                                            if map_i != map_j:
                                                sess_comb_xcorr[map_j,map_i,:] = sess_xcorr[subj_j,subj_i,:]

                                            comb_xcorr[task][signal_type][source][e].append(sess_comb_xcorr)

                                            dup_key = tuple(sorted([map_i,map_j]))
                                            weights = np.full_like(sess_comb_xcorr, 1/dup_counts[dup_key])
                                            weights[np.isnan(sess_comb_xcorr)] = np.nan
                                            comb_xcorr_weights[task][signal_type][source][e].append(weights)

    # plot individual subject averages
    if plot_ind_subj:
        for source in plot_source_types:
            t_sel = t_sels[source]
            plot_corr_lags = corr_lags[source]

            if plot_sep_epochs:
                # if plotting separate epochs, plot them on the same axis per task
                for task in plot_tasks:
                    for signal_type in signal_types:

                        signal_title, _ = fpah.get_signal_type_labels(signal_type)

                        fig, axs = plt.subplots(n_regions, n_regions, sharex=True, sharey=True, figsize=(ax_size*n_regions+2, ax_size*n_regions), layout='constrained')
                        fig.suptitle('{} Regional Cross-correlation for Subject {} in {}. (- first leading, + first lagging)\n{}'.format(source_type_labels[source], subj_id, plot_beh_names[task], signal_title))

                        for reg_i in range(n_regions):
                            sorted_i = sorted_region_idxs[reg_i]
                            for reg_j in range(n_regions):
                                sorted_j = sorted_region_idxs[reg_j]

                                ax = axs[reg_i, reg_j]

                                if not plot_diag and reg_j == reg_i:
                                    ax.set_axis_off()
                                    continue

                                plot_utils.plot_dashlines(0, dir='h', ax=ax)
                                plot_utils.plot_dashlines(0, dir='v', ax=ax)

                                for e in plot_epochs:
                                    if task == 'all':
                                        sess_corrs = np.hstack([np.stack([x_corrs[subj_id][t][s][signal_type][source][e][sorted_i, sorted_j, :] for s in x_corrs[subj_id][t].keys()], axis=1) for t in tasks if len(x_corrs[subj_id][t]) > 0])
                                    else:
                                        sess_corrs = np.stack([x_corrs[subj_id][task][s][signal_type][source][e][sorted_i, sorted_j, :] for s in x_corrs[subj_id][task].keys()], axis=1)

                                    plot_utils.plot_psth(plot_corr_lags[t_sel], np.nanmean(sess_corrs, axis=1)[t_sel], utils.stderr(sess_corrs, axis=1)[t_sel], ax, plot_x0=False, label=e)

                                ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'],
                                                                         sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
                                ax.legend()
                                handles, labels = ax.get_legend_handles_labels()
                                ax.legend().remove()

                                if reg_i == n_regions-1:
                                    ax.set_xlabel('Time lag (s)')
                                if reg_j == 0:
                                    ax.set_ylabel('Pearson r')

                        fig.legend(handles, labels, loc='outside right center', title='Epoch')

            else:
                # if not plotting separate epochs, plot tasks on same axes
                for signal_type in signal_types:

                    signal_title, _ = fpah.get_signal_type_labels(signal_type)

                    fig, axs = plt.subplots(n_regions, n_regions, sharex=True, sharey=True, figsize=(ax_size*n_regions+2, ax_size*n_regions), layout='constrained')
                    fig.suptitle('{} Regional Cross-correlation for Subject {}. (- first leading, + first lagging)\n{}'.format(source_type_labels[source], subj_id, signal_title))

                    for reg_i in range(n_regions):
                        sorted_i = sorted_region_idxs[reg_i]
                        for reg_j in range(n_regions):
                            sorted_j = sorted_region_idxs[reg_j]

                            ax = axs[reg_i, reg_j]

                            if not plot_diag and reg_j == reg_i:
                                ax.set_axis_off()
                                continue

                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            plot_utils.plot_dashlines(0, dir='v', ax=ax)

                            for task in plot_tasks:
                                if task == 'all':
                                    sess_corrs = np.hstack([np.stack([x_corrs[subj_id][t][s][signal_type][source]['All'][sorted_i, sorted_j, :] for s in x_corrs[subj_id][t].keys()], axis=1) for t in tasks if len(x_corrs[subj_id][t]) > 0])
                                else:
                                    sess_corrs = np.stack([x_corrs[subj_id][task][s][signal_type][source]['All'][sorted_i, sorted_j, :] for s in x_corrs[subj_id][task].keys()], axis=1)

                                plot_utils.plot_psth(plot_corr_lags[t_sel], np.nanmean(sess_corrs, axis=1)[t_sel], utils.stderr(sess_corrs, axis=1)[t_sel], ax, plot_x0=False, label=plot_beh_names[task])

                            ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'],
                                                                     sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
                            ax.legend()
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend().remove()

                            if reg_i == n_regions-1:
                                ax.set_xlabel('Time lag (s)')
                            if reg_j == 0:
                                ax.set_ylabel('Pearson r')

                    fig.legend(handles, labels, loc='outside right center', title='Task')


# plot combined subject averages
if plot_comb_subj:
    # stack all matrices together
    comb_xcorr = {t: {s: {st: {e: np.stack(comb_xcorr[t][s][st][e], axis=-1) for e in epochs} for st in plot_source_types} for s in signal_types} for t in tasks}
    comb_xcorr_weights = {t: {s: {st: {e: np.stack(comb_xcorr_weights[t][s][st][e], axis=-1) for e in epochs} for st in plot_source_types} for s in signal_types} for t in tasks}

    for source in plot_source_types:
        t_sel = t_sels[source]
        plot_corr_lags = corr_lags[source]

        for signal_type in signal_types:
            signal_title, _ = fpah.get_signal_type_labels(signal_type)

            if plot_sep_epochs:
                for task in plot_tasks:

                    fig, axs = plt.subplots(len(all_regions), len(all_regions), sharey=True,
                                            figsize=(ax_size*len(all_regions)+2, ax_size*len(all_regions)), layout='constrained')
                    fig.suptitle('{} Regional Cross-correlation for All Subjects in {}. (- first leading, + first lagging)\n{}'.format(source_type_labels[source], plot_beh_names[task], signal_title))

                    for reg_i in range(len(all_regions)):
                        for reg_j in range(len(all_regions)):

                            ax = axs[reg_i, reg_j]

                            if not plot_diag and reg_j == reg_i:
                                ax.set_axis_off()
                                continue

                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            plot_utils.plot_dashlines(0, dir='v', ax=ax)

                            for e in plot_epochs:
                                if task == 'all':
                                    comb_corrs = np.hstack([comb_xcorr[t][signal_type][source][e][reg_i, reg_j, :, :] for t in tasks])
                                    comb_weights = np.hstack([comb_xcorr_weights[t][signal_type][source][e][reg_i, reg_j, :, :] for t in tasks])
                                else:
                                    comb_corrs = comb_xcorr[task][signal_type][source][e][reg_i, reg_j, :, :]
                                    comb_weights = comb_xcorr_weights[task][signal_type][source][e][reg_i, reg_j, :, :]

                                plot_utils.plot_psth(plot_corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel],
                                                     utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=False, label=e)

                            ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
                            ax.legend()
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend().remove()

                            ax.set_xlabel('Time lag (s)')
                            if reg_j == 0:
                                ax.set_ylabel('Pearson r')

                    fig.legend(handles, labels, loc='outside right center', title='Epoch')

            else:

                fig, axs = plt.subplots(len(all_regions), len(all_regions), sharey=True,
                                        figsize=(ax_size*len(all_regions)+2, ax_size*len(all_regions)), layout='constrained')
                fig.suptitle('{} Regional Cross-correlation for All Subjects. (- first leading, + first lagging)\n{}'.format(source_type_labels[source], signal_title))

                for reg_i in range(len(all_regions)):
                    for reg_j in range(len(all_regions)):

                        ax = axs[reg_i, reg_j]

                        if not plot_diag and reg_j == reg_i:
                            ax.set_axis_off()
                            continue

                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=ax)

                        for task in plot_tasks:
                            if task == 'all':
                                comb_corrs = np.hstack([comb_xcorr[t][signal_type][source]['All'][reg_i, reg_j, :, :] for t in tasks])
                                comb_weights = np.hstack([comb_xcorr_weights[t][signal_type][source]['All'][reg_i, reg_j, :, :] for t in tasks])
                            else:
                                comb_corrs = comb_xcorr[task][signal_type][source]['All'][reg_i, reg_j, :, :]
                                comb_weights = comb_xcorr_weights[task][signal_type][source]['All'][reg_i, reg_j, :, :]

                            plot_utils.plot_psth(plot_corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel],
                                                 utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=False, label=plot_beh_names[task])

                        ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
                        ax.legend()
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend().remove()

                        ax.set_xlabel('Time lag (s)')
                        if reg_j == 0:
                            ax.set_ylabel('Pearson r')

                fig.legend(handles, labels, loc='outside right center', title='Task')

            # plot cross-correlations between each region and all others on the same axis
            if plot_sep_epochs:

                for task in plot_tasks:

                    fig, axs = plt.subplots(len(all_regions), len(plot_epochs), sharex=False, sharey=True,
                                            figsize=(ax_size*len(plot_epochs)+2, ax_size*len(all_regions)), layout='constrained')
                    fig.suptitle('{} Cross-correlations by Epoch for All Subject in {}. (- leading, + lagging)\n{}'.format(source_type_labels[source], plot_beh_names[task], signal_title))

                    for epoch_i, epoch in enumerate(plot_epochs):
                        for reg_i in range(len(all_regions)):
                            ax = axs[reg_i, epoch_i]
                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            plot_utils.plot_dashlines(0, dir='v', ax=ax)
                            ax.set_title('{}, {}'.format(all_regions[reg_i], epoch))
                            ax.set_xlabel('Time lag (s)')

                            for reg_j in range(len(all_regions)):

                                if reg_i == reg_j:
                                    continue

                                if task == 'all':
                                    comb_corrs = np.hstack([comb_xcorr[t][signal_type][source][epoch][reg_i, reg_j, :, :] for t in tasks])
                                    comb_weights = np.hstack([comb_xcorr_weights[t][signal_type][source][epoch][reg_i, reg_j, :, :] for t in tasks])
                                else:
                                    comb_corrs = comb_xcorr[task][signal_type][source][epoch][reg_i, reg_j, :, :]
                                    comb_weights = comb_xcorr_weights[task][signal_type][source][epoch][reg_i, reg_j, :, :]

                                plot_utils.plot_psth(plot_corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel],
                                                     utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=False,
                                                     color=region_colors[all_regions[reg_j]])

                                if epoch_i == 0:
                                    ax.set_ylabel('Pearson r')

                    handles = []
                    labels = []
                    for reg in all_regions:
                        handles.append(mpatches.Patch(color=region_colors[reg]))
                        labels.append(reg)

                    fig.legend(handles, labels, loc='outside right center')

            else:

                fig, axs = plt.subplots(len(plot_tasks), len(all_regions), sharex=False, sharey=True,
                                        figsize=(ax_size*len(all_regions)+2, ax_size*len(plot_tasks)), layout='constrained')
                axs = axs.reshape(len(plot_tasks), len(all_regions))

                fig.suptitle('{} Cross-correlations for All Subjects. (- leading, + lagging)\n{}'.format(source_type_labels[source], signal_title))

                for task_i, task in enumerate(plot_tasks):

                    for reg_i in range(len(all_regions)):
                        ax = axs[task_i, reg_i]
                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=ax)
                        ax.set_title('{}, {}'.format(all_regions[reg_i], plot_beh_names[task]))
                        ax.set_xlabel('Time lag (s)')

                        for reg_j in range(len(all_regions)):

                            if reg_i == reg_j:
                                continue

                            if task == 'all':
                                comb_corrs = np.hstack([comb_xcorr[t][signal_type][source]['All'][reg_i, reg_j, :, :] for t in tasks])
                                comb_weights = np.hstack([comb_xcorr_weights[t][signal_type][source]['All'][reg_i, reg_j, :, :] for t in tasks])
                            else:
                                comb_corrs = comb_xcorr[task][signal_type][source]['All'][reg_i, reg_j, :, :]
                                comb_weights = comb_xcorr_weights[task][signal_type][source]['All'][reg_i, reg_j, :, :]

                            plot_utils.plot_psth(plot_corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel],
                                                 utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=False,
                                                 color=region_colors[all_regions[reg_j]])

                            if reg_i == 0:
                                ax.set_ylabel('Pearson r')

                handles = []
                labels = []
                for reg in all_regions:
                    handles.append(mpatches.Patch(color=region_colors[reg]))
                    labels.append(reg)

                fig.legend(handles, labels, loc='outside right center')


            # plot bilateral correlations
            bilat_xcorr = {t: {e: {r: [] for r in all_regions} for e in epochs} for t in tasks}
            bilat_corr_diff = {t: {e: {r: [] for r in all_regions} for e in epochs} for t in tasks}

            for subj_id in subj_ids:
                subj_regions = x_corr_regions[subj_id]
                reg_mapping = get_subj_region_mapping(subj_regions)
                subj_bilat_regions = [r for r in all_regions if sum([reg_mapping[sr] == r for sr in subj_regions]) > 1]

                for task in tasks:
                    if len(x_corrs[subj_id][task]) == 0:
                        continue

                    for e in epochs:
                        for region in subj_bilat_regions:
                            reg_idxs = [i for i,r in enumerate(subj_regions) if reg_mapping[r] == region]
                            subj_xcorrs = []
                            subj_corr_diffs = []

                            for reg_i in range(len(reg_idxs)-1):
                                for reg_j in np.arange(reg_i+1, len(reg_idxs)):
                                    idx_i = reg_idxs[reg_i]
                                    idx_j = reg_idxs[reg_j]

                                    for s in x_corrs[subj_id][task].keys():
                                        sess_xcorr = np.nanmean(np.stack([x_corrs[subj_id][task][s][signal_type][source][e][idx_i, idx_j, :],
                                                                          x_corrs[subj_id][task][s][signal_type][source][e][idx_j, idx_i, :]], axis=1), axis=1)
                                        sess_autocorr = np.nanmean(np.stack([x_corrs[subj_id][task][s][signal_type][source][e][idx_i, idx_i, :],
                                                                             x_corrs[subj_id][task][s][signal_type][source][e][idx_j, idx_j, :]], axis=1), axis=1)
                                        sess_corr_diff = sess_autocorr - sess_xcorr

                                        subj_xcorrs.append(sess_xcorr)
                                        subj_corr_diffs.append(sess_corr_diff)

                            if len(subj_xcorrs) > 0:
                                bilat_xcorr[task][e][region].append(np.nanmean(np.stack(subj_xcorrs, axis=1), axis=1))
                                bilat_corr_diff[task][e][region].append(np.nanmean(np.stack(subj_corr_diffs, axis=1), axis=1))

            plot_bilat_regions = [r for r in all_regions if any([len(bilat_xcorr[t][e][r]) > 0 for t in tasks for e in epochs])]

            if len(plot_bilat_regions) > 0:
                if plot_sep_epochs:

                    for i in range(2):

                        if i == 0:
                            plot_data = bilat_xcorr
                            data_title = 'Cross-correlations'
                            data_ylabel = 'Pearson r'
                        else:
                            plot_data = bilat_corr_diff
                            data_title = 'Correlation Diffs'
                            data_ylabel = 'Pearson r Diff (auto - cross)'

                        # group by region

                        fig, axs = plt.subplots(len(plot_tasks), len(plot_bilat_regions), sharex=True, sharey=True,
                                                figsize=(ax_size*len(plot_bilat_regions)+2, ax_size*len(plot_tasks)), layout='constrained')
                        axs = np.reshape(axs, (len(plot_tasks), len(plot_bilat_regions)))
                        fig.suptitle('{} Bilateral {} by Epoch for All Subjects. (- leading, + lagging)\n{}'.format(source_type_labels[source], data_title, signal_title))

                        for task_i, task in enumerate(plot_tasks):
                            for reg_i, region in enumerate(plot_bilat_regions):
                                ax = axs[task_i, reg_i]
                                plot_utils.plot_dashlines(0, dir='h', ax=ax)
                                plot_utils.plot_dashlines(0, dir='v', ax=ax)
                                ax.set_title('{}, {}'.format(region, plot_beh_names[task]))
                                ax.set_xlabel('Time lag (s)')

                                for epoch in plot_epochs:
                                    if task == 'all':
                                        bilat_corrs = [corr for t in tasks for corr in plot_data[t][epoch][region]]
                                    else:
                                        bilat_corrs = plot_data[task][epoch][region]

                                    if len(bilat_corrs) == 0:
                                        continue

                                    bilat_corrs = np.stack(bilat_corrs, axis=1)
                                    plot_utils.plot_psth(plot_corr_lags[t_sel], np.nanmean(bilat_corrs, axis=1)[t_sel],
                                                         utils.stderr(bilat_corrs, axis=1)[t_sel], ax, plot_x0=False,
                                                         label=epoch)

                                if reg_i == 0:
                                    ax.set_ylabel(data_ylabel)

                                ax.legend()
                                handles, labels = ax.get_legend_handles_labels()
                                ax.legend().remove()

                        fig.legend(handles, labels, loc='outside right center')

                        # group by epoch

                        fig, axs = plt.subplots(len(plot_tasks), len(plot_epochs), sharex=True, sharey=True,
                                                figsize=(ax_size*len(plot_epochs)+2, ax_size*len(plot_tasks)), layout='constrained')
                        axs = np.reshape(axs, (len(plot_tasks), len(plot_epochs)))
                        fig.suptitle('{} Bilateral {} by Epoch for All Subjects. (- leading, + lagging)\n{}'.format(source_type_labels[source], data_title, signal_title))

                        for task_i, task in enumerate(plot_tasks):
                            for epoch_i, epoch in enumerate(plot_epochs):
                                ax = axs[task_i, epoch_i]
                                plot_utils.plot_dashlines(0, dir='h', ax=ax)
                                plot_utils.plot_dashlines(0, dir='v', ax=ax)
                                ax.set_title('{}, {}'.format(plot_beh_names[task], epoch))
                                ax.set_xlabel('Time lag (s)')

                                for region in plot_bilat_regions:
                                    if task == 'all':
                                        bilat_corrs = [corr for t in tasks for corr in plot_data[t][epoch][region]]
                                    else:
                                        bilat_corrs = plot_data[task][epoch][region]

                                    if len(bilat_corrs) == 0:
                                        continue

                                    bilat_corrs = np.stack(bilat_corrs, axis=1)
                                    plot_utils.plot_psth(plot_corr_lags[t_sel], np.nanmean(bilat_corrs, axis=1)[t_sel],
                                                         utils.stderr(bilat_corrs, axis=1)[t_sel], ax, plot_x0=False,
                                                         label=region, color=region_colors[region])

                                if reg_i == 0:
                                    ax.set_ylabel(data_ylabel)

                                ax.legend()
                                handles, labels = ax.get_legend_handles_labels()
                                ax.legend().remove()

                        fig.legend(handles, labels, loc='outside right center')

                else:

                    fig, axs = plt.subplots(2, len(plot_tasks), sharex=True, sharey=True,
                                            figsize=(ax_size*len(plot_tasks)+2, 2*ax_size), layout='constrained')
                    axs = np.reshape(axs, (2, len(plot_tasks)))
                    fig.suptitle('{} Bilateral Cross-correlations for All Subjects. (- leading, + lagging)\n{}'.format(source_type_labels[source], signal_title))

                    for task_i, task in enumerate(plot_tasks):
                        xcorr_ax = axs[0,task_i]
                        diff_ax = axs[1,task_i]

                        plot_utils.plot_dashlines(0, dir='h', ax=xcorr_ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=xcorr_ax)

                        plot_utils.plot_dashlines(0, dir='h', ax=diff_ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=diff_ax)

                        xcorr_ax.set_title('Cross-correlations, {}'.format(plot_beh_names[task]))
                        xcorr_ax.set_xlabel('Time lag (s)')

                        diff_ax.set_title('Correlation Diffs, {}'.format(plot_beh_names[task]))
                        diff_ax.set_xlabel('Time lag (s)')

                        if task_i == 0:
                            xcorr_ax.set_ylabel('Pearson r')
                            diff_ax.set_ylabel('Pearson r Diff (auto - cross)')

                        for region in plot_bilat_regions:

                            if task == 'all':
                                bilat_xcorrs = [corr for t in tasks for corr in bilat_xcorr[t]['All'][region]]
                                bilat_corr_diffs = [corr for t in tasks for corr in bilat_corr_diff[t]['All'][region]]
                            else:
                                bilat_xcorrs = bilat_xcorr[task]['All'][region]
                                bilat_corr_diffs = bilat_corr_diff[task]['All'][region]

                            if len(bilat_xcorrs) == 0:
                                continue

                            bilat_xcorrs = np.stack(bilat_xcorrs, axis=1)
                            bilat_corr_diffs = np.stack(bilat_corr_diffs, axis=1)

                            plot_utils.plot_psth(plot_corr_lags[t_sel], np.nanmean(bilat_xcorrs, axis=1)[t_sel],
                                                 utils.stderr(bilat_corrs, axis=1)[t_sel], xcorr_ax, plot_x0=False,
                                                 color=region_colors[region])

                            plot_utils.plot_psth(plot_corr_lags[t_sel], np.nanmean(bilat_corr_diffs, axis=1)[t_sel],
                                                 utils.stderr(bilat_corrs, axis=1)[t_sel], diff_ax, plot_x0=False,
                                                 color=region_colors[region])

                    handles = []
                    labels = []
                    for reg in plot_bilat_regions:
                        handles.append(mpatches.Patch(color=region_colors[reg]))
                        labels.append(reg)

                    fig.legend(handles, labels, loc='outside right center')

# %% Perform delay-embedded regression

dec = 2
delay_window = 2

# %% Plot delay-embedded regression results



# %% Compute ICA Mixing Matrices

recalculate = False
reprocess_sess_ids = []
reprocess_subj_ids = []
max_attempts = 10

signal_type = 'dff_iso_baseline_fband'

region_lpf = {'PL': 4, 'NAc': 6, 'DMS': 8, 'DLS': 8, 'TS': 8}
dec = 1

filename = 'wm_bandit_ica_mix_mats'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        ica_results = saved_data['ica_results']
        metadata = saved_data['metadata']
        recalculate = metadata['signal_type'] != signal_type
        ignored_sess = saved_data['ignored_sess']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    ica_results = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    ignored_sess = set()

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    n_regions = len(subj_regions)

    if not subj_id in ica_results:
        ica_results[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if (sess_id in ica_results[subj_id][task] or sess_id in ignored_sess) and (not sess_id in reprocess_sess_ids and not subj_id in reprocess_subj_ids):
                continue

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            dt = fp_data['dec_info']['decimated_dt']
            trial_start_ts = fp_data['trial_start_ts']

            if not np.all([r in fp_data['processed_signals'] for r in subj_regions]):
                ignored_sess.add(sess_id)
                continue

            mix_mats = {s: {e: [] for e in epochs} for s in source_types}
            demix_mats = {s: {e: [] for e in epochs} for s in source_types}
            ev_details = {s: {e: [] for e in epochs} for s in source_types}

            for source in source_types:
                signal_mat = []
                for r in subj_regions:
                    signal = fp_data['processed_signals'][r][signal_type]

                    if source == 'filt':
                        signal = fp_utils.filter_signal(signal, region_lpf[reg_mapping[r]], 1/dt)
                        signal, _ = fp_utils.decimate(signal, t, dec, dt=dt)

                        deconv_tau = avg_params_all.loc[(avg_params_all['subj_id'] == subj_id) & (avg_params_all['region'] == r) & (avg_params_all['epoch'] == 'All'), ('tau','mean')]
                        deconv_params = fp_utils.get_ar_params_from_modes({'tau': deconv_tau, 'freq': 0}, dt*dec)

                        signal = fp_utils.extract_ar_source(signal, deconv_params)
                    else:
                        signal, _ = fp_utils.decimate(signal, t, dec, dt=dt)

                    signal_mat.append(signal.T)

                signal_mat = np.stack(signal_mat, axis=1)

                # remove nans
                nans = np.any(np.isnan(signal_mat), axis=1)
                if np.all(nans):
                    ignored_sess.add(sess_id)
                    continue

                signal_mat = signal_mat[~nans,:]

                for epoch in epochs:
                    if epoch == 'All':
                        t_sel = np.full_like(signal, True, dtype=bool)
                    else:
                        dec_labels = decimate_epoch_labels(epoch_t_labels[subj_id][task][sess_id], dec)
                        t_sel = dec_labels == epoch

                    t_sel = t_sel[~nans]
                    epoch_mat = signal_mat[t_sel,:]
                    epoch_mat = utils.z_score(epoch_mat, axis=0)

                    if epoch_mat.shape[0] < 2:
                        continue

                    # do ICA
                    converged = False
                    attempt = 0
                    while not converged and attempt < max_attempts:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter('always', category=sklearn.exceptions.ConvergenceWarning)

                            ica = FastICA(max_iter=1000)
                            epoch_ica = ica.fit_transform(epoch_mat)

                        if any(issubclass(w.category, sklearn.exceptions.ConvergenceWarning) for w in caught_warnings):
                            attempt += 1
                            converged = False
                            print('  ICA for session {} did not converge. Trying again...'.format(sess_id))
                        else:
                            converged = True

                    # if attempt >= max_attempts:
                    #     print('Exceeded max attempts for session {}. Skipping...'.format(sess_id))
                    #     continue

                    mix_mat = ica.mixing_
                    demix_mat = ica.components_

                    # rectify columns of mixing matrix so that the largest values are all positive
                    mix_mat, epoch_ica, rect_mult = fp_utils.rectify_ica_output(mix_mat, epoch_ica)
                    demix_mat = demix_mat*rect_mult.T

                    epoch_ev_details = fp_utils.ic_explained_variance(mix_mat, epoch_ica, epoch_mat)

                    mix_mats[source][epoch] = mix_mat
                    demix_mats[source][epoch] = demix_mat
                    ev_details[source][epoch] = epoch_ev_details

            ica_results[subj_id][task][sess_id] = {'mix_mat': mix_mats, 'demix_mat': demix_mats, 'ev_details': ev_details}

            with open(save_path, 'wb') as f:
                pickle.dump({'ica_results': ica_results,
                             'ignored_sess': ignored_sess,
                             'metadata': {'signal_type': signal_type,
                                          'region_lpf': region_lpf,
                                          'dec': dec}},
                            f)

# %% Plot ICA results

plot_raw_loadings = False
plot_deconv = False
plot_ind_subj = True
use_marginal_ev = True

if plot_deconv:
    plot_source_types = source_types.copy()
else:
    plot_source_types = ['raw']

ev_key = 'ev_component_region_marg' if use_marginal_ev else 'ev_component_region'

# cluster ICA mixing and demixing matrices
plot_tasks = tasks.copy()
plot_tasks.append('all')

all_mix_mats = {s: {t: {s: {e: {'norm': [], 'raw': []} for e in epochs} for s in plot_source_types} for t in plot_tasks} for s in subj_ids}
all_expl_vars = {s: {t: {s: {e: [] for e in epochs} for s in plot_source_types} for t in plot_tasks} for s in subj_ids}
all_dom_regs = {s: {t: {s: {e: [] for e in epochs} for s in plot_source_types} for t in plot_tasks} for s in subj_ids}
#all_demix_mats = {s: {t: {s: {e: {'norm': [], 'raw': []} for e in epochs} for s in plot_source_types} for t in plot_tasks} for s in subj_ids}

avg_mix_mats = {t: {s: {e: {r: [] for r in all_regions} for e in epochs} for s in plot_source_types} for t in plot_tasks}
avg_expl_vars = {t: {s: {e: {r: [] for r in all_regions} for e in epochs} for s in plot_source_types} for t in plot_tasks}
region_to_idx = {r:i for i,r in enumerate(all_regions)}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    subj_region_idxs = {r: i for i,r in enumerate(subj_regions)}
    subj_region_mapping_idxs = {reg_mapping[r]: [] for r in subj_regions}

    for i, r in enumerate(subj_regions):
        subj_region_mapping_idxs[reg_mapping[r]].append(i)

    subj_reg_mix_mats = {t: {s: {e: {r: [] for r in subj_region_mapping_idxs} for e in epochs} for s in plot_source_types} for t in plot_tasks}
    subj_reg_expl_vars = {t: {s: {e: {r: [] for r in subj_region_mapping_idxs} for e in epochs} for s in plot_source_types} for t in plot_tasks}

    n_regions = len(subj_regions)

    reg_colors = {r: 'C{}'.format(i) for i, r in enumerate(subj_regions)}

    for s in plot_source_types:
        task_order = {e: [] for e in epochs}

        for i, task in enumerate(tasks):
            for e in epochs:

                mix_vectors = []
                expl_vars = []
                dom_regs = []

                for sess_id in ica_results[subj_id][task].keys():
                    sess_mix_mat = ica_results[subj_id][task][sess_id]['mix_mat'][s][e]

                    if sess_id in ignored_sess or len(sess_mix_mat) == 0:
                        continue

                    sess_mix_mat = sess_mix_mat.T
                    sess_ev = ica_results[subj_id][task][sess_id]['ev_details'][s][e][ev_key]

                    mix_vectors.append(sess_mix_mat)
                    expl_vars.append(sess_ev)

                    # calculate the subject average grouped by primary mixing vector loading region
                    # first average loadings and explained variances over duplicate regions for each IC
                    session_reg_mix = {r: [] for r in subj_region_mapping_idxs}
                    session_reg_ev = {r: [] for r in subj_region_mapping_idxs}

                    for ic in range(sess_mix_mat.shape[0]):

                        # save dominant region for clustering
                        dom_regs.append(subj_regions[np.nanargmax(sess_ev[ic,:])])

                        canon_mix = np.full(len(all_regions), np.nan)
                        canon_ev = np.full(len(all_regions), np.nan)

                        for canon_reg, idxs in subj_region_mapping_idxs.items():

                            canon_idx = region_to_idx[canon_reg]

                            # Collapse duplicate regions
                            canon_mix[canon_idx] = np.mean(sess_mix_mat[ic, idxs])
                            canon_ev[canon_idx] = np.mean(sess_ev[ic, idxs])

                        # Normalize loading vector
                        valid = ~np.isnan(canon_mix)
                        canon_mix[valid] /= np.linalg.norm(canon_mix[valid])

                        # Assign IC to dominant canonical region
                        dom_idx = np.nanargmax(canon_ev)
                        dom_reg = all_regions[dom_idx]

                        session_reg_mix[dom_reg].append(canon_mix)
                        session_reg_ev[dom_reg].append(canon_ev)

                    # Then average over duplicate region ICs for this session
                    for canon_reg in subj_region_mapping_idxs:

                        if len(session_reg_mix[canon_reg]) == 0:
                            continue

                        mixes = np.vstack(session_reg_mix[canon_reg])
                        evs = np.vstack(session_reg_ev[canon_reg])

                        dom_idx = region_to_idx[canon_reg]

                        weights = evs[:, dom_idx]
                        weights = weights / weights.sum()

                        avg_mix = np.sum(mixes * weights[:, None], axis=0)
                        avg_mix /= np.linalg.norm(avg_mix[~np.isnan(avg_mix)])

                        total_ev = np.sum(evs, axis=0)

                        subj_reg_mix_mats[task][s][e][canon_reg].append(avg_mix)
                        subj_reg_mix_mats['all'][s][e][canon_reg].append(avg_mix)

                        subj_reg_expl_vars[task][s][e][canon_reg].append(total_ev)
                        subj_reg_expl_vars['all'][s][e][canon_reg].append(total_ev)

                # average subject loadings and EVs over all sessions
                for canon_reg in subj_region_mapping_idxs:
                    avg_mix_mat = np.nanmean(np.vstack(subj_reg_mix_mats[task][s][e][canon_reg]), axis=0)
                    avg_ev = np.nanmean(np.vstack(subj_reg_expl_vars[task][s][e][canon_reg]), axis=0)

                    avg_mix_mats[task][s][e][canon_reg].append(avg_mix_mat)
                    avg_expl_vars[task][s][e][canon_reg].append(avg_ev)
                    avg_mix_mats['all'][s][e][canon_reg].append(avg_mix_mat)
                    avg_expl_vars['all'][s][e][canon_reg].append(avg_ev)

                mix_vectors = np.vstack(mix_vectors)
                mix_vectors_norm = mix_vectors / np.linalg.norm(mix_vectors, axis=1, keepdims=True)

                expl_vars = np.vstack(expl_vars)
                dom_regs = np.vstack(dom_regs)

                all_mix_mats[subj_id][task][s][e]['raw'] = mix_vectors
                all_mix_mats[subj_id][task][s][e]['norm'] = mix_vectors_norm
                all_mix_mats[subj_id]['all'][s][e]['raw'].append(mix_vectors)
                all_mix_mats[subj_id]['all'][s][e]['norm'].append(mix_vectors_norm)

                all_expl_vars[subj_id][task][s][e] = expl_vars
                all_expl_vars[subj_id]['all'][s][e].append(expl_vars)

                all_dom_regs[subj_id][task][s][e] = dom_regs
                all_dom_regs[subj_id]['all'][s][e].append(dom_regs)

                # demix_vectors = np.vstack([ica_results[subj_id][task][sess]['demix_mat'][s][e] for sess in ica_results[subj_id][task].keys() if len(ica_results[subj_id][task][sess]['demix_mat'][s][e]) > 0])
                # demix_vectors_norm = demix_vectors / np.linalg.norm(demix_vectors, axis=1, keepdims=True)

                # all_demix_mats[subj_id][task][s][e]['raw'] = demix_vectors
                # all_demix_mats[subj_id][task][s][e]['norm'] = demix_vectors_norm
                # all_demix_mats[subj_id]['all'][s][e]['raw'].append(demix_vectors)
                # all_demix_mats[subj_id]['all'][s][e]['norm'].append(demix_vectors_norm)

                task_order[e].extend(mix_vectors.shape[0]*[i])

    if plot_ind_subj:
        task_order = {e: np.array(task_order[e]) for e in epochs}
        key = 'raw' if plot_raw_loadings else 'norm'

        for s in plot_source_types:
            n_rows = len(epochs)
            n_cols = 3
            fig = plt.figure(figsize=(5*n_cols,4*n_rows), layout='constrained')
            fig.suptitle('Subject {}, {}, ICA Results by Epoch'.format(subj_id, source_type_labels[s]))

            for e_i, e in enumerate(epochs):
                all_mix_mats[subj_id]['all'][s][e][key] = np.vstack(all_mix_mats[subj_id]['all'][s][e][key])
                all_expl_vars[subj_id]['all'][s][e] = np.vstack(all_expl_vars[subj_id]['all'][s][e])
                all_dom_regs[subj_id]['all'][s][e] = np.vstack(all_dom_regs[subj_id]['all'][s][e]).flatten()
                # all_demix_mats[subj_id]['all'][s][e][key] = np.vstack(all_demix_mats[subj_id]['all'][s][e][key])

                # cluster
                # mix_wm_kmeans = KMeans(n_clusters=n_regions).fit(all_mix_mats[subj_id]['wm'][s][e][key])
                # mix_wm_centers = mix_wm_kmeans.cluster_centers_

                # mix_bandit_kmeans = KMeans(n_clusters=n_regions).fit(all_mix_mats[subj_id]['bandit'][s][e][key])
                # mix_bandit_centers = mix_bandit_kmeans.cluster_centers_

                # mix_all_kmeans = KMeans(n_clusters=n_regions).fit(all_mix_mats[subj_id]['all'][s][e][key])
                # mix_all_centers = mix_all_kmeans.cluster_centers_

                # demix_wm_kmeans = KMeans(n_clusters=n_regions).fit(all_demix_mats[subj_id]['wm'][s][e][key])
                # demix_wm_centers = demix_wm_kmeans.cluster_centers_

                # demix_bandit_kmeans = KMeans(n_clusters=n_regions).fit(all_demix_mats[subj_id]['bandit'][s][e][key])
                # demix_bandit_centers = demix_bandit_kmeans.cluster_centers_

                # demix_all_kmeans = KMeans(n_clusters=n_regions).fit(all_demix_mats[subj_id]['all'][s][e][key])
                # demix_all_centers = demix_all_kmeans.cluster_centers_

                # # Compute 3D projections
                # pca = PCA(n_components=3)
                # pca.fit(all_mix_mats[subj_id]['all'][s][e][key])

                # proj_mix_wm = pca.transform(all_mix_mats[subj_id]['wm'][s][e][key])
                # proj_mix_bandit = pca.transform(all_mix_mats[subj_id]['bandit'][s][e][key])
                # proj_mix_wm_centers = pca.transform(mix_wm_centers)
                # proj_mix_bandit_centers = pca.transform(mix_bandit_centers)

                # pca.fit(all_demix_mats[subj_id]['all'][s][e][key])

                # proj_demix_wm = pca.transform(all_demix_mats[subj_id]['wm'][s][e][key])
                # proj_demix_bandit = pca.transform(all_demix_mats[subj_id]['bandit'][s][e][key])
                # proj_demix_wm_centers = pca.transform(demix_wm_centers)
                # proj_demix_bandit_centers = pca.transform(demix_bandit_centers)

                # compute cosine similarity matrices
                mix_cos_sim = all_mix_mats[subj_id]['all'][s][e][key] @ all_mix_mats[subj_id]['all'][s][e][key].T
                # demix_cos_sim = all_demix_mats[subj_id]['all'][s][e][key] @ all_demix_mats[subj_id]['all'][s][e][key].T

                # # sort kmeans labels by dominant region
                # clust_dom_reg = {}
                # clust_reg_idx = {}
                # clust_idx_reg = {}
                # for i in range(n_regions):
                #     label_sel = mix_all_kmeans.labels_ == i
                #     clust_avg_expl_var = np.mean(all_expl_vars[subj_id]['all'][s][e][label_sel, :], axis=0)
                #     dom_idx = np.nanargmax(clust_avg_expl_var)
                #     dom_reg = subj_regions[dom_idx]
                #     clust_dom_reg[dom_reg] = i
                #     clust_reg_idx[i] = dom_idx
                #     clust_idx_reg[i] = dom_reg

                idxs = np.arange(all_mix_mats[subj_id]['all'][s][e][key].shape[0])
                mix_order = sorted(idxs, key=lambda i: (subj_region_idxs[all_dom_regs[subj_id]['all'][s][e][i]], task_order[e][i]))
                #demix_order = sorted(idxs, key=lambda i: (demix_all_kmeans.labels_[i], task_order[i]))

                mix_cos_sim = mix_cos_sim[np.ix_(mix_order, mix_order)]
                # demix_cos_sim = demix_cos_sim[np.ix_(mix_order, mix_order)]

                # plot

                # # first col are 3D projections
                # ax = fig.add_subplot(n_rows, n_cols, 1, projection='3d')
                # ax.set_title('Mixing Vectors - 3D Projection')
                # ax.scatter(proj_mix_wm[:,0], proj_mix_wm[:,1], proj_mix_wm[:,2], label='WM', alpha=0.5, color='C0')
                # ax.scatter(proj_mix_bandit[:,0], proj_mix_bandit[:,1], proj_mix_bandit[:,2], label='Bandit', alpha=0.5, color='C1')
                # ax.scatter(proj_mix_wm_centers[:,0], proj_mix_wm_centers[:,1], proj_mix_wm_centers[:,2], label='WM Centers', color='C0', marker='x', s=50)
                # ax.scatter(proj_mix_bandit_centers[:,0], proj_mix_bandit_centers[:,1], proj_mix_bandit_centers[:,2], label='Bandit Centers', color='C1', marker='x', s=50)
                # ax.legend()

                # ax = fig.add_subplot(n_rows, n_cols, n_cols+1, projection='3d')
                # ax.set_title('Demixing Vectors - 3D Projection')
                # ax.scatter(proj_demix_wm[:,0], proj_demix_wm[:,1], proj_demix_wm[:,2], label='WM', alpha=0.5, color='C0')
                # ax.scatter(proj_demix_bandit[:,0], proj_demix_bandit[:,1], proj_demix_bandit[:,2], label='Bandit', alpha=0.5, color='C1')
                # ax.scatter(proj_demix_wm_centers[:,0], proj_demix_wm_centers[:,1], proj_demix_wm_centers[:,2], label='WM Centers', color='C0', marker='x', s=50)
                # ax.scatter(proj_demix_bandit_centers[:,0], proj_demix_bandit_centers[:,1], proj_demix_bandit_centers[:,2], label='Bandit Centers', color='C1', marker='x', s=50)
                # ax.legend()

                # correlation matrix
                ax = fig.add_subplot(n_rows, n_cols, e_i*n_cols+1)
                ax.set_title('Mixing Cosine Similarity, {}'.format(e))
                im = ax.imshow(mix_cos_sim, vmin=0, vmax=1, cmap='plasma')
                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label('Similarity')

                # add boundary lines and labels
                bounds = [0]
                clusters_sorted = [all_dom_regs[subj_id]['all'][s][e][i] for i in mix_order]
                tasks_sorted = [task_order[e][i] for i in mix_order]

                for i in range(1, len(clusters_sorted)):
                    if clusters_sorted[i-1] != clusters_sorted[i] or tasks_sorted[i-1] != tasks_sorted[i]:
                        ax.axhline(i-0.5, color='white', lw=1)
                        ax.axvline(i-0.5, color='white', lw=1)
                        bounds.append(i)

                bounds.append(i+1)
                bounds = np.array(bounds)
                label_centers = np.diff(bounds)/2 + bounds[0:-1]
                labels = ['{} - {}'.format(r, t) for r in subj_regions for t in tasks]

                ax.set_xticks(label_centers, labels, rotation=45, ha='right', rotation_mode='anchor')
                ax.set_yticks(label_centers, labels)

                # ax = fig.add_subplot(n_rows, n_cols, n_cols+2)
                # ax.set_title('Demixing Vectors - Cosine Similarity')
                # im = ax.imshow(demix_cos_sim, vmin=0, vmax=1, cmap='plasma')
                # cbar = fig.colorbar(im, ax=ax)
                # cbar.set_label('Similarity')

                # # add boundary lines and labels
                # # clusters_sorted = [demix_all_kmeans.labels_[i] for i in mix_order]
                # # tasks_sorted = [task_order[i] for i in mix_order]

                # for i in range(1, len(clusters_sorted)):
                #     if clusters_sorted[i-1] != clusters_sorted[i] or tasks_sorted[i-1] != tasks_sorted[i]:
                #         ax.axhline(i-0.5, color='white', lw=1)
                #         ax.axvline(i-0.5, color='white', lw=1)

                # ax.set_xticks(label_centers, labels, rotation=45)
                # ax.set_yticks(label_centers, labels)
                # ax.set_xlabel('Cluster - Task')
                # ax.set_ylabel('Cluster - Task')

                # mixing vectors per cluster
                ax = fig.add_subplot(n_rows, n_cols, e_i*n_cols+2)
                ax.set_title('Mixing Vectors, {}'.format(e))
                x = np.arange(n_regions)
                ax.axhline(0, color='black', linewidth=0.5)
                for i in x:
                    ax.axvline(i, color='black', linewidth=0.5)
                # plot all sessions
                for i in idxs:
                    linestyle = '-' if task_order[e][i] == 0 else '--'
                    ax.plot(x, all_mix_mats[subj_id]['all'][s][e][key][i,:], color=reg_colors[all_dom_regs[subj_id]['all'][s][e][i]],
                            linestyle=linestyle, alpha=0.4, linewidth=1)

                # plot averages
                wm_sel = task_order[e] == 0
                bandit_sel = task_order[e] == 1
                for r in subj_regions:
                    label_sel = all_dom_regs[subj_id]['all'][s][e] == r

                    wm_mean = np.mean(all_mix_mats[subj_id]['all'][s][e][key][label_sel & wm_sel, :], axis=0)
                    bandit_mean = np.mean(all_mix_mats[subj_id]['all'][s][e][key][label_sel & bandit_sel, :], axis=0)

                    ax.plot(x, wm_mean, color=reg_colors[r], linestyle='-', alpha=1, linewidth=2, label=r)
                    ax.plot(x, bandit_mean, color=reg_colors[r], linestyle='--', alpha=1, linewidth=2)

                ax.plot(x, x*np.nan, linestyle='-', color='black', label='WM')
                ax.plot(x, x*np.nan, linestyle='--', color='black', label='Bandit')

                ax.set_xticks(x, subj_regions)
                ax.set_ylabel('IC Loading')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                # ax = fig.add_subplot(n_rows, n_cols, n_cols+3)
                # ax.set_title('Demixing Vectors - Region Loadings')
                # ax.axhline(0, color='black', linewidth=0.5)
                # for i in x:
                #     ax.axvline(i, color='black', linewidth=0.5)
                # # plot all sessions
                # for i in idxs:
                #     linestyle = '-' if task_order[i] == 0 else '--'
                #     ax.plot(x, all_demix_mats[subj_id]['all'][s][e][key][i,:], color='C{}'.format(mix_all_kmeans.labels_[i]),
                #             linestyle=linestyle, alpha=0.4, linewidth=1)

                # # plot averages
                # for i in range(n_regions):
                #     label_sel = mix_all_kmeans.labels_ == i

                #     wm_mean = np.mean(all_demix_mats[subj_id]['all'][s][e][key][label_sel & wm_sel, :], axis=0)
                #     bandit_mean = np.mean(all_demix_mats[subj_id]['all'][s][e][key][label_sel & bandit_sel, :], axis=0)

                #     ax.plot(x, wm_mean, color='C{}'.format(i), linestyle='-', alpha=1, linewidth=2, label='Cluster {}'.format(i))
                #     ax.plot(x, bandit_mean, color='C{}'.format(i), linestyle='--', alpha=1, linewidth=2)

                # ax.plot(x, x*np.nan, linestyle='-', color='black', label='WM')
                # ax.plot(x, x*np.nan, linestyle='--', color='black', label='Bandit')

                # ax.set_xticks(x, subj_regions)
                # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

                # explained variances per cluster
                ax = fig.add_subplot(n_rows, n_cols, e_i*n_cols+3)
                ax.set_title('Explained Variance, {}'.format(e))
                x = np.arange(n_regions)
                ax.axhline(0, color='black', linewidth=0.5)
                for i in x:
                    ax.axvline(i, color='black', linewidth=0.5)
                # plot all sessions
                for i in idxs:
                    linestyle = '-' if task_order[e][i] == 0 else '--'
                    ax.plot(x, all_expl_vars[subj_id]['all'][s][e][i,:], color=reg_colors[all_dom_regs[subj_id]['all'][s][e][i]],
                            linestyle=linestyle, alpha=0.4, linewidth=1)

                # plot averages
                wm_sel = task_order[e] == 0
                bandit_sel = task_order[e] == 1
                for r in subj_regions:
                    label_sel = all_dom_regs[subj_id]['all'][s][e] == r

                    wm_mean = np.mean(all_expl_vars[subj_id]['all'][s][e][label_sel & wm_sel, :], axis=0)
                    bandit_mean = np.mean(all_expl_vars[subj_id]['all'][s][e][label_sel & bandit_sel, :], axis=0)

                    ax.plot(x, wm_mean, color=reg_colors[r], linestyle='-', alpha=1, linewidth=2, label=r)
                    ax.plot(x, bandit_mean, color=reg_colors[r], linestyle='--', alpha=1, linewidth=2)

                ax.plot(x, x*np.nan, linestyle='-', color='black', label='WM')
                ax.plot(x, x*np.nan, linestyle='--', color='black', label='Bandit')

                ax.set_xticks(x, subj_regions)
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            plt.show()

# %% plot subject averages

plot_type = 'bar' # 'bar'
plot_sep_epochs = True
plot_all_epoch = False

plot_epochs = epochs.copy() if plot_sep_epochs else ['All']
if not plot_all_epoch:
    plot_epochs.remove('All')

# plot loadings by epoch per task & source
for s in plot_source_types:
    for metric in ['Mixing Vectors', 'Explained Variance']:

        if metric == 'Mixing Vectors':
            met_dict = avg_mix_mats
            y_label = 'IC Loading'
        else:
            met_dict = avg_expl_vars
            y_label = 'Explained Variance (R2)'

        # plot by dominant region on each row
        fig, axs = plt.subplots(len(all_regions), len(plot_tasks), figsize=(4*len(plot_tasks),3*len(all_regions)), layout='constrained', sharey=True)
        fig.suptitle('All Subjects {} IC {} by Epoch'.format(source_type_labels[s], metric))

        for i, dom_reg in enumerate(all_regions):
            for j, task in enumerate(plot_tasks):
                ax = axs[i,j]
                ax.set_title('{}, {}'.format(dom_reg, plot_beh_names[task]))
                # ax.grid(axis='y')

                if plot_type == 'bar':
                    epoch_avg = []
                    epoch_se = []
                    plot_colors = []

                    for k, e in enumerate(plot_epochs):
                        mat = np.vstack(met_dict[task][s][e][dom_reg])
                        avg = np.nanmean(mat, axis=0)
                        se = utils.stderr(mat, axis=0)

                        # remove dominant region
                        reg_idx = region_to_idx[dom_reg]
                        avg[reg_idx] = np.nan
                        se[reg_idx] = np.nan

                        epoch_avg.append(avg)
                        epoch_se.append(se)
                        plot_colors.append('C{}'.format(k))

                    plot_utils.plot_stacked_bar(epoch_avg, value_labels=plot_epochs, x_labels=all_regions, ax=ax, err=epoch_se, colors=plot_colors, draw_legend=False)

                else:
                    x = np.arange(len(all_regions))

                    ax.axhline(0, color='black', linewidth=0.5)
                    for k in x:
                        ax.axvline(k, color='black', linewidth=0.5)

                    for k, e in enumerate(plot_epochs):

                        plot_mat = np.vstack(met_dict[task][s][e][dom_reg])
                        epoch_color = 'C{}'.format(k)

                        # plot individual subject averages
                        for k in range(plot_mat.shape[0]):
                            ax.plot(x, plot_mat[k,:], alpha=0.4, linewidth=1, color=epoch_color)

                        # plot total average
                        ax.plot(x, np.nanmean(plot_mat, axis=0), alpha=1, linewidth=2, color=epoch_color, label=e)
                        ax.set_xticks(x, all_regions)

                if j == 0:
                    ax.set_ylabel(y_label)
                if j == len(plot_tasks)-1:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')

                ax.set_xlabel('Loading Region')

        # plot by epoch on each row
        fig, axs = plt.subplots(len(plot_epochs), len(plot_tasks), figsize=(4*len(plot_tasks),3*len(plot_epochs)), layout='constrained', sharey=True)
        fig.suptitle('All Subjects {} IC {} by Dominant Region'.format(source_type_labels[s], metric))

        axs = axs.reshape(len(plot_epochs), len(plot_tasks))

        for i, e in enumerate(plot_epochs):
            for j, task in enumerate(plot_tasks):
                ax = axs[i,j]
                ax.set_title('{}, {}'.format(e, plot_beh_names[task]))
                # ax.grid(axis='y')


                if plot_type == 'bar':
                    reg_avg = []
                    reg_se = []
                    plot_colors = []

                    for dom_reg in all_regions:
                        mat = np.vstack(met_dict[task][s][e][dom_reg])
                        avg = np.nanmean(mat, axis=0)
                        se = utils.stderr(mat, axis=0)

                        # remove dominant region
                        reg_idx = region_to_idx[dom_reg]
                        avg[reg_idx] = np.nan
                        se[reg_idx] = np.nan

                        reg_avg.append(avg)
                        reg_se.append(se)
                        plot_colors.append(region_colors[dom_reg])

                    plot_utils.plot_stacked_bar(reg_avg, value_labels=all_regions, x_labels=all_regions, ax=ax, err=reg_se, colors=plot_colors, draw_legend=False)

                else:
                    x = np.arange(len(all_regions))

                    ax.axhline(0, color='black', linewidth=0.5)
                    for k in x:
                        ax.axvline(k, color='black', linewidth=0.5)

                    for dom_reg in all_regions:

                        plot_mat = np.vstack(met_dict[task][s][e][dom_reg])

                        # plot individual subject averages
                        for k in range(plot_mat.shape[0]):
                            ax.plot(x, plot_mat[k,:], alpha=0.4, linewidth=1, color=region_colors[dom_reg])

                        # plot total average
                        ax.plot(x, np.nanmean(plot_mat, axis=0), alpha=1, linewidth=2, color=region_colors[dom_reg], label=dom_reg)
                        ax.set_xticks(x, all_regions)

                if j == 0:
                    ax.set_ylabel(y_label)
                if j == len(plot_tasks)-1:
                    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Dominant Region')

                ax.set_xlabel('Loading Region')


# %% Perform convolutional delay-embedded ICA

import conv_dictionary_learning as cdl
import gc

def plot_kernels(kernels, subj_regions, dt, dec, title, evs=None):
    n_kerns = len(kernels)

    # plot component loadings
    n_cols = len(subj_regions)
    n_rows = int(np.ceil(n_kerns/n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=[4*n_cols,4*n_rows], sharey=True)
    fig.suptitle(title)

    axs = np.reshape(axs, (n_rows, n_cols))

    for i in range(n_kerns):
        row = int(np.floor(i/n_cols))
        col = int(i % n_cols)

        # plot component
        ax = axs[row,col]

        kern = kernels[i]
        kern_t = np.arange(kern.shape[1])*dt*dec

        for k, r in enumerate(subj_regions):
            ax.plot(kern_t, kern[k,:], label=r, alpha=0.8)

        if evs is None:
            ax.set_title('IC {}'.format(i))
        else:
            ax.set_title('IC {}, EV: {:.3f}'.format(i, evs[i]))

        ax.set_ylabel('Loading')
        ax.set_xlabel('Time (s)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend()
        factor_handles, factor_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

    fig.legend(factor_handles, factor_labels, loc='outside right center')
    plt.show()

reprocess_ica_results = False
recalculate = False
reprocess_sess_ids = []
reprocess_subj_ids = []
max_attempts = 1

ica_epochs = ['All'] #epochs
signal_type = 'z_dff_iso_baseline_fband'

t_dec = 5
n_ic_dec = 1.5

delay_window = 1.5
sim_t_window = 2000
sim_thresh = 0.7

ev_thresh = 0.02

filename = 'wm_bandit_conv_ica'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        conv_ica_results = saved_data['conv_ica_results']
        metadata = saved_data['metadata']
        recalculate = metadata['t_dec'] != t_dec or metadata['delay_window'] != delay_window
        ignored_sess = saved_data['ignored_sess']

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    conv_ica_results = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    ignored_sess = set()

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    n_regions = len(subj_regions)

    if not subj_id in conv_ica_results:
        conv_ica_results[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if (sess_id in ignored_sess
                or ((sess_id in conv_ica_results[subj_id][task] and all(e in conv_ica_results[subj_id][task][sess_id] for e in ica_epochs))
                    and (not sess_id in reprocess_sess_ids and not subj_id in reprocess_subj_ids)
                    and not reprocess_ica_results)):

                continue

            if not sess_id in conv_ica_results[subj_id][task]:
                conv_ica_results[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            dt = fp_data['dec_info']['decimated_dt']

            n_delay = int(delay_window/(dt*t_dec))

            full_t = fp_data['time']
            _, dec_t = fp_utils.decimate(full_t, full_t, t_dec, dt=dt)

            if not np.all([r in fp_data['processed_signals'] for r in subj_regions]):
                ignored_sess.add(sess_id)
                continue

            # build signal matrices
            delay_signal_mat = np.empty([len(dec_t)-n_delay, n_delay*n_regions])
            full_signal_mat = np.empty([len(dec_t), n_regions])

            for i, region in enumerate(subj_regions):
                signal = fp_data['processed_signals'][region][signal_type]
                dec_signal, _ = fp_utils.decimate(signal, full_t, t_dec, dt=dt)
                full_signal_mat[:,i] = dec_signal
                for j in range(n_delay):
                    delay_signal_mat[:,i+j*n_regions] = dec_signal[j:-n_delay+j]

            # remove nans from delay mat
            nans = np.any(np.isnan(delay_signal_mat), axis=1)
            if np.all(nans):
                ignored_sess.add(sess_id)
                continue

            delay_signal_mat = delay_signal_mat[~nans,:]

            dec_labels = decimate_epoch_labels(epoch_t_labels[subj_id][task][sess_id], t_dec)
            dec_labels_start = dec_labels[:-n_delay]
            dec_labels_end = dec_labels[n_delay:]

            ica_results = {'delay_ica': {'mix_mat': [], 'demix_mat': [], 'ev_details': [],
                                         'clust_labels': [], 'mix_kernels': [], 'demix_kernels': []},
                           'conv_ica': {'mix_kernels': [], 'demix_kernels': [],
                                        'ev_details': [], 'center_idx': []}}
            for epoch in ica_epochs:

                if epoch in conv_ica_results[subj_id][task][sess_id] and not reprocess_ica_results:
                    continue

                if not epoch in conv_ica_results[subj_id][task][sess_id]:
                    conv_ica_results[subj_id][task][sess_id][epoch] = copy.deepcopy(ica_results)

                if epoch == 'All':
                    t_sel_delay = np.full(len(dec_labels_start), True, dtype=bool)
                    t_sel_full = np.full(len(dec_labels), True, dtype=bool)
                else:
                    t_sel_delay = (dec_labels_start == epoch) & (dec_labels_end == epoch)
                    t_sel_full = dec_labels == epoch

                t_sel_delay = t_sel_delay[~nans]
                delay_epoch_mat = delay_signal_mat[t_sel_delay,:]
                delay_epoch_mat = utils.z_score(delay_epoch_mat, axis=0)

                # fill in nans outside of epoch so those timepoints are ignored when refining kernels
                signal_epoch_mat = full_signal_mat.copy()
                signal_epoch_mat[~t_sel_full,:] = np.nan
                signal_epoch_mat = utils.z_score(signal_epoch_mat, axis=0)

                # do delay-embedded ICA
                start = time.perf_counter()
                if len(conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['mix_mat']) == 0:
                    print('Performing delay-embedded ICA for session {}, epoch {}'.format(sess_id, epoch))
                    converged = False
                    attempt = 0
                    while not converged and attempt < max_attempts:
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter('always', category=sklearn.exceptions.ConvergenceWarning)

                            ica = FastICA(max_iter=1000, whiten_solver='eigh', n_components=int(n_delay*n_regions/n_ic_dec), tol=1e-3)
                            epoch_sources = ica.fit_transform(delay_epoch_mat)

                        if any(issubclass(w.category, sklearn.exceptions.ConvergenceWarning) for w in caught_warnings):
                            attempt += 1
                            converged = False
                            print('  ICA for session {} did not converge. Trying again...'.format(sess_id))
                        else:
                            converged = True

                    print('  Completed ICA in {:.1f} s in {} iterations'.format(time.perf_counter()-start, ica.n_iter_))

                    mix_mat = ica.mixing_
                    demix_mat = ica.components_

                    mix_mat, epoch_sources, rect_mult = fp_utils.rectify_ica_output(mix_mat, epoch_sources)
                    demix_mat = demix_mat*rect_mult.T

                    raw_ev_details = fp_utils.ic_explained_variance(mix_mat, epoch_sources, delay_epoch_mat)
                else:
                    mix_mat = conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['mix_mat']
                    demix_mat = conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['demix_mat']
                    raw_ev_details = conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['ev_details']
                    epoch_sources = delay_epoch_mat @ demix_mat.T

                ev_components = raw_ev_details['ev_component_marg']

                # find unique ICs that explain 99% variance in the all columns
                sorted_ic_idxs = np.argsort(ev_components)[::-1]
                sorted_evs = ev_components[sorted_ic_idxs]
                cum_ev = np.cumsum(ev_components)
                thresh_idx = np.argmin((cum_ev < 0.991) & (sorted_evs >= 0.0005))
                if thresh_idx == 0:
                    thresh_idx = len(sorted_ic_idxs)
                primary_ic_idxs = sorted_ic_idxs[:thresh_idx]
                primary_ic_evs = ev_components[primary_ic_idxs]

                # aggregate and average ic kernels
                mix_kernels = []
                demix_kernels = []
                for i in primary_ic_idxs:
                    mix_kernels.append(mix_mat[:,i].reshape(n_delay, n_regions).T)
                    demix_kernels.append(demix_mat[i,:].reshape(n_delay, n_regions).T)

                # title = 'Subject {}, Session {} Mixing Kernels'.format(subj_id, sess_id)
                # plot_kernels(mix_kernels, subj_regions, dt, t_dec, title, evs=primary_ic_evs)
                # title = 'Subject {}, Session {} Demixing Kernels'.format(subj_id, sess_id)
                # plot_kernels(demix_kernels, subj_regions, dt, dec, title)

                # cluster kernels by their loadings
                source_mid = epoch_sources.shape[0] // 2
                source_list = [epoch_sources[source_mid-sim_t_window:source_mid+sim_t_window,i][None,:] for i in primary_ic_idxs]

                similarity = fp_utils.calc_similarity(source_list, search_window=n_delay, shift_penalty=0, use_deriv=False, use_centroid=False)

                del source_list, epoch_sources

                ic_labels = fp_utils.cluster_similarity(similarity, threshold=sim_thresh)

                mix_templates, fit_details = fp_utils.create_average_templates(mix_kernels, ic_labels, primary_ic_evs, search_window=n_delay,
                                                                               use_deriv=True, use_centroid=True)

                template_ic_evs = []
                demix_templates = []
                for i in np.unique(ic_labels):
                    label_sel = ic_labels == i
                    label_ic_idxs = primary_ic_idxs[label_sel]
                    template_ic_evs.append(np.sum(ev_components[label_ic_idxs]))
                    clust_demix = [demix_kernels[i] for i, sel in enumerate(label_sel) if sel]
                    clust_shifts = fit_details['shifts'][label_sel]
                    clust_scales = fit_details['scales'][label_sel]
                    demix_templates.append(fp_utils.build_template(clust_demix, clust_shifts, 1/clust_scales))

                template_ic_evs = np.array(template_ic_evs)

                # remove templates with explained variance below a threshold to make kernel selection a little faster
                sorted_ic_idxs = np.flatnonzero(template_ic_evs >= ev_thresh)
                sorted_ic_idxs = sorted_ic_idxs[np.argsort(template_ic_evs[sorted_ic_idxs])[::-1]]

                mix_templates = [mix_templates[i] for i in sorted_ic_idxs]
                demix_templates = [demix_templates[i] for i in sorted_ic_idxs]
                template_ic_evs = template_ic_evs[sorted_ic_idxs]

                # title = 'Subject {}, Session {}, Demixing Kernels'.format(subj_id, sess_id)
                # plot_kernels(demix_templates, subj_regions, dt, dec, title, powers=template_ic_evs)
                title = 'Subject {}, Session {}, Mixing Kernels'.format(subj_id, sess_id)
                plot_kernels(mix_templates, subj_regions, dt, t_dec, title, evs=template_ic_evs)

                # stack mix and demix kernels
                stacked_mix_templates, stack_details = fp_utils.stack_centered_kernels(mix_templates, trunc_thresh=0.002,
                                                                                       debug_plot=False, extrapolate_edges=False)
                stacked_demix_templates, _ = fp_utils.stack_centered_kernels(demix_templates, shifts=stack_details['shifts'],
                                                                             trunc_min=stack_details['trunc_min'], trunc_max=stack_details['trunc_max'])

                # find kernels that explain unique variance
                keep, results, details = cdl.find_unique_kernels(signal_epoch_mat.T, stacked_mix_templates, zero_idx=stack_details['center_idx'],
                                                                 ev_thresh=ev_thresh, order=np.arange(len(mix_templates)), verbose=False,
                                                                 atol=1e-4, btol=1e-4)

                # refine the kernels using gradient descent
                model = cdl.ConvAutoencoder(W_mix_init=results['W_mix_orig'], W_demix_init=results['W_demix_est'], zero_idx=stack_details['center_idx'],
                                            n_iter=20000, early_stopping_tol=1e-6, verbose=True, lr=1e-3,
                                            lambda_w=0.01, lambda_w_alpha=0, lambda_smooth=0.1, lambda_ica=0, lambda_recon_sim=1)

                # title = 'Subject {}, Session {}, Kernel Selection Results'.format(subj_id, sess_id)
                # model.plot_templates(title=title)
                fit_results_demix = model.fit(signal_epoch_mat.T, fix_w_mix=True, fix_w_alpha=True)
                fit_results_mix = model.fit(signal_epoch_mat.T, fix_w_demix=True, fix_w_alpha=True)
                processed_ev_details = model.calc_ev(signal_epoch_mat.T)
                title = 'Subject {}, Session {}, Epoch: {}, EV: {:.3f}'.format(subj_id, sess_id, epoch, processed_ev_details['ev_total'])
                model.plot_templates(title=title)
                plt.show()

                conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['mix_mat'] = mix_mat
                conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['demix_mat'] = demix_mat
                conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['ev_details'] = raw_ev_details
                conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['clust_labels'] = ic_labels
                conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['mix_kernels'] = mix_templates
                conv_ica_results[subj_id][task][sess_id][epoch]['delay_ica']['demix_kernels'] = demix_templates

                conv_ica_results[subj_id][task][sess_id][epoch]['conv_ica']['mix_kernels'] = model.get_W_mix()
                conv_ica_results[subj_id][task][sess_id][epoch]['conv_ica']['demix_kernels'] = model.get_W_demix()
                conv_ica_results[subj_id][task][sess_id][epoch]['conv_ica']['ev_details'] = processed_ev_details
                conv_ica_results[subj_id][task][sess_id][epoch]['conv_ica']['center_idx'] = stack_details['center_idx']

                print('Completed Convolutional ICA for session {}, epoch {} in {:.1f} s. EV: {:.3f}'.format(sess_id, epoch, time.perf_counter()-start, processed_ev_details['ev_total']))

                with open(save_path, 'wb') as f:
                    pickle.dump({'conv_ica_results': conv_ica_results,
                                 'ignored_sess': ignored_sess,
                                 'metadata': {'signal_type': signal_type,
                                              't_dec': t_dec,
                                              'delay_window': delay_window}},
                                f)

                del delay_epoch_mat, signal_epoch_mat
                gc.collect()


# %% Investigate ICA

signal_type = 'dff_iso_baseline_fband'

region_lpf = {'PL': 4, 'NAc': 6, 'DMS': 8, 'DLS': 8, 'TS': 8}
dec = 2

component_dec = 0
sess_ids = {199: [116828, 117565], 237: [124638, 124973], 274: [119776, 119993], 400: [119713, 119974],
            402: [119481, 119742], 424: [124644, 124772], 483: [124924, 124957]}
#sess_ids = { 400: [119785, 120001], 402: [119839, 120009], 483: [124746, 125164], 424: [124674]} #
loc_db = bandit_loc_db

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    n_regions = len(subj_regions)
    n_components = n_regions-component_dec

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        t = fp_data['time']
        dt = fp_data['dec_info']['decimated_dt']
        trial_start_ts = fp_data['trial_start_ts']

        trial_data = loc_db.get_behavior_data(sess_id)

        trial_start_ts = trial_start_ts[:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        for f in source_types:
            signal_mat = []
            for r in subj_regions:
                if f == 'raw':
                    signal = fp_data['processed_signals'][r][signal_type]
                    signal, dec_t = fp_utils.decimate(signal, t, dec, dt=dt)
                else:
                    signal = fp_utils.filter_signal(fp_data['processed_signals'][r][signal_type], region_lpf[reg_mapping[r]], 1/dt)
                    signal, dec_t = fp_utils.decimate(signal, t, dec, dt=dt)

                    deconv_tau = avg_params_all.loc[(avg_params_all['subj_id'] == subj_id) & (avg_params_all['region'] == r), ('tau','mean')]
                    deconv_params = fp_utils.get_ar_params_from_modes({'tau': deconv_tau, 'freq': 0}, dt*dec)

                    signal = fp_utils.extract_ar_source(signal, deconv_params)

                signal_mat.append(signal.T)

            signal_mat = np.stack(signal_mat, axis=1)

            # remove nans
            nans = np.any(np.isnan(signal_mat), axis=1)
            if np.all(nans):
                ignored_sess.append(sess_id)
                continue

            signal_mat = signal_mat[~nans,:]

            signal_mat = utils.z_score(signal_mat, axis=0)

            # do ICA
            ica = FastICA(max_iter=1000)
            signal_ica = ica.fit_transform(signal_mat)

            mix_mat = ica.mixing_
            demix_mat = ica.components_

            # rearrange columns of mixing matrix so that the diagonal has all the largest values
            mix_mat, signal_ica, rect_mult = fp_utils.rectify_ica_output(mix_mat, signal_ica)
            demix_mat = demix_mat*rect_mult.T

            power_total, power_ic_total, power_ic_channel = fp_utils.ic_explained_power(mix_mat, signal_ica, signal_mat)
            recon_signals = fp_utils.reconstruct_ica(mix_mat, signal_ica)

            title = 'Subject {}, Session {}, {}'.format(subj_id, sess_id, f)
            fig, axs = plt.subplots(n_components+1, 2, layout='constrained', figsize=[24,4*(n_components+1)], width_ratios=[1,10], sharex='col')
            fig.suptitle('{}, Reconstruction Power {:.3f}'.format(title, power_total))

            plot_time = dec_t[~nans]

            # plot full signal with behavioral events
            ax = axs[0,1]
            for j, region in enumerate(subj_regions):
                ax.plot(plot_time, signal_mat[:,j], label=region, alpha=0.5)

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                          transform=ax.get_xaxis_transform())

            ax.set_title('Region Signals')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('dF/F')
            ax.legend()
            sig_handles, sig_labels = ax.get_legend_handles_labels()
            ax.legend().remove()

            for j in range(n_components):

                # plot component
                ax = axs[j+1,0]
                ax.bar(subj_regions, mix_mat[:,j])

                ax.set_title('IC {} Region Loadings'.format(j))
                ax.set_xlabel('Regions')
                ax.set_ylabel('Loading')
                ax.xaxis.set_tick_params(which='both', labelbottom=True)
                ax.legend()
                factor_handles, factor_labels = ax.get_legend_handles_labels()
                ax.legend().remove()

                # plot factor reconstruction with behavioral events
                ax = axs[j+1,1]

                # plot reconstruction
                for k, region in enumerate(subj_regions):
                    ax.plot(plot_time, recon_signals[:,k,j], label=region, alpha=0.5)

                for k, (name, lines) in enumerate(lines_dict.items()):
                    ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
                              transform=ax.get_xaxis_transform())

                ax.set_title('IC Factor {} Reconstruction. Power: {:.3f}'.format(j, power_ic_total[j]))
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Signal Reconstruction')
                ax.xaxis.set_tick_params(which='both', labelbottom=True)

            ax = axs[0,0]
            ax.set_axis_off()
            ax.legend(factor_handles, factor_labels)
            fig.legend(sig_handles, sig_labels, loc='outside right upper')
            plt.show()


        # fig, axs = plt.subplots(len(subj_regions), 1, sharex=True, figsize=(20, 4*len(subj_regions)))
        # fig.suptitle('Subject {}, Session {}, Region Signals Processing'.format(subj_id, sess_id))
        # for i,r in enumerate(subj_regions):
        #     signal = fp_data['processed_signals'][r][signal_type]
        #     signal, dec_t = fp_utils.decimate(signal, t, dec, dt=dt)

        #     tau = avg_taus.loc[(subj_id, r)]['mean']
        #     #lpf = freq_cutoff_df[(freq_cutoff_df['subj_id'] == subj_id) & (freq_cutoff_df['region'] == r) & (freq_cutoff_df['task'] == 'all')][filter_fcol].iloc[0]
        #     lpf = region_lpf[reg_mapping[r]]
        #     filtered_signal = fp_utils.filter_signal(signal, lpf, 1/dt)
        #     deconvolved_signal = signal[1:] - np.exp(-(dt*dec)/tau)*signal[:-1]
        #     filt_deconvolved_signal = filtered_signal[1:] - np.exp(-(dt*dec)/tau)*filtered_signal[:-1]

        #     ax = axs[i]
        #     ax.set_title('{}, tau: {}, LPF: {}'.format(r, tau, lpf))
        #     ax.plot(dec_t, signal, label='original', alpha=0.5)
        #     ax.plot(dec_t, filtered_signal, label='filtered', alpha=0.5)
        #     ax.plot(dec_t[1:], utils.z_score(deconvolved_signal), label='deconvolved', alpha=0.5)
        #     ax.plot(dec_t[1:], utils.z_score(filt_deconvolved_signal), label='filtered deconvolved', alpha=0.5)
        #     ax.legend()

        #     # if filter_signal:
        #     #     signal = fp_utils.filter_signal(signal, filter_mult/tau, 1/dt)

        #     # if deconvolve:
        #     #     signal[1:] = signal[1:] - np.exp(-(dt*dec)/tau)*signal[:-1]
        #     #     signal = signal[1:]

        #     # # z-score again
        #     # signal = utils.z_score(signal)
        #     # signal_mat.append(signal)

        # # signal_mat = np.stack(signal_mat, axis=1)
        # # if deconvolve:
        # #     t = t[1:]

        # # # build signal matrix
        # # if use_fbands:

        # #     fband_signals = {}
        # #     fband_mix_mats = {}
        # #     fband_tot_powers = {}
        # #     fband_ic_powers = {}
        # #     fband_ic_channel_powers = {}
        # #     fband_ic_recon = {}

        # #     # limit filtered ICA signals to avoid edge effects from filtering
        # #     t_sel = np.full_like(t, False, dtype=bool)
        # #     t_sel[t < trial_start_ts[0]] = True
        # #     t_sel[t > trial_start_ts[-1]] = True

        # #     signal_mat[t_sel,:] = np.nan
        # #     nans = np.any(np.isnan(signal_mat), axis=1)
        # #     signal_mat = signal_mat[~nans,:]

        # #     for fband, label in zip(freq_bands, band_labels):
        # #         fband_signal_mat = np.concatenate([fp_utils.decompose_signal_fbands(
        # #             fp_data['processed_signals'][r][signal_type], [fband], 1/dt).T for r in subj_regions], axis=1)

        # #         # remove nans
        # #         fband_signal_mat = fband_signal_mat[~nans,:]

        # #         fband_signal_mat = utils.z_score(fband_signal_mat, axis=0)

        # #         # do ICA
        # #         ica = FastICA(n_components=n_components)
        # #         signal_ica = ica.fit_transform(fband_signal_mat)
        # #         ica_mix_mat = ica.mixing_
        # #         ica_demix_mat = ica.components_

        # #         # rearrange columns of mixing matrix so that the diagonal has all the largest values
        # #         ica_mix_mat, signal_ica, _ = fp_utils.rectify_ica_output(ica_mix_mat, signal_ica)

        # #         power_total, power_ic_total, power_ic_channel = fp_utils.ic_explained_power(ica_mix_mat, signal_ica, fband_signal_mat)

        # #         # sort by power per region
        # #         order = []
        # #         all_cols = np.arange(n_components)
        # #         for i in range(n_components):
        # #             max_idx = np.argmax(power_ic_channel[i,all_cols])
        # #             max_idx = all_cols[max_idx]
        # #             all_cols = all_cols[all_cols != max_idx]
        # #             order.append(max_idx)

        # #         power_ic_total = power_ic_total[order]
        # #         power_ic_channel = power_ic_channel[:, order]
        # #         ica_mix_mat = ica_mix_mat[:, order]
        # #         signal_ica = signal_ica[:, order]

        # #         # calculate reconstruction of each channel from each IC
        # #         ic_reconstruction = fp_utils.reconstruct_ica(ica_mix_mat, signal_ica)

        # #         fband_signals[label] = fband_signal_mat
        # #         fband_mix_mats[label] = ica_mix_mat
        # #         fband_tot_powers[label] = power_total
        # #         fband_ic_powers[label] = power_ic_total
        # #         fband_ic_channel_powers[label] = power_ic_channel
        # #         fband_ic_recon[label] = ic_reconstruction
        # # else:

        # #     # remove nans
        # #     nans = np.any(np.isnan(signal_mat), axis=1)
        # #     signal_mat = signal_mat[~nans,:]

        # #     # do ICA
        # #     ica = FastICA(n_components=n_components)
        # #     signal_ica = ica.fit_transform(signal_mat)
        # #     ica_mix_mat = ica.mixing_
        # #     ica_demix_mat = ica.components_

        # #     # rectify the columns of the mixing matrix so that the largest values are all positive
        # #     ica_mix_mat, signal_ica, _ = fp_utils.rectify_ica_output(ica_mix_mat, signal_ica)

        # #     power_total, power_ic_total, power_ic_channel = fp_utils.ic_explained_power(ica_mix_mat, signal_ica, signal_mat)

        # #     # sort by power per region
        # #     order = []
        # #     all_cols = np.arange(n_components)
        # #     for i in range(n_components):
        # #         max_idx = np.argmax(power_ic_channel[i,all_cols])
        # #         max_idx = all_cols[max_idx]
        # #         all_cols = all_cols[all_cols != max_idx]
        # #         order.append(max_idx)

        # #     power_ic_total = power_ic_total[order]
        # #     power_ic_channel = power_ic_channel[:, order]
        # #     ica_mix_mat = ica_mix_mat[:, order]
        # #     signal_ica = signal_ica[:, order]

        # #     # calculate reconstruction of each channel from each IC
        # #     ic_reconstruction = fp_utils.reconstruct_ica(ica_mix_mat, signal_ica)

        # # trial_data = loc_db.get_behavior_data(sess_id)

        # # plot_time = t[~nans]
        # # trial_start_ts = trial_start_ts[:-1]
        # # cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        # # cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        # # #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        # # cue_ts = trial_start_ts + trial_data['response_cue_time']
        # # cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        # # response_ts = trial_start_ts + trial_data['response_time']
        # # outcome_ts = trial_start_ts + trial_data['reward_time']
        # # reward_ts = outcome_ts[trial_data['reward'] > 0]
        # # unreward_ts = outcome_ts[trial_data['reward'] == 0]

        # # lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
        # #               'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
        # #               'Reward': reward_ts, 'Unreward': unreward_ts}

        # #

        # # if use_fbands:

        # #     # build full signal reconstructions and total power of full signal
        # #     full_signal_recon = np.stack([fband_ic_recon[l] for l in band_labels], axis=-1)
        # #     full_signal_recon_ic = np.sum(full_signal_recon, axis=-1)
        # #     full_signal_recon = np.sum(full_signal_recon_ic, axis=-1)

        # #     # calculate total power and power per ic x region
        # #     region_power = np.nansum(signal_mat**2, axis=0)
        # #     tot_power = np.sum(region_power)

        # #     power_ic_region = np.zeros((n_regions, n_components))

        # #     power_total = 1 - np.sum((signal_mat - full_signal_recon)**2)/tot_power

        # #     for i in range(n_components):
        # #         recon_ic = full_signal_recon_ic[:,:,i]
        # #         power_ic_region[:, i] =  1 - np.sum((signal_mat - recon_ic)**2, axis=0) / region_power

        # #     for fband, label in zip(freq_bands, band_labels):
        # #         fig, axs = plt.subplots(n_components+1, 2, layout='constrained', figsize=[24,4*(n_components+1)], width_ratios=[1,10], sharex='col')
        # #         fig.suptitle('{}, Band: {}, Reconstruction Power {:.3f}'.format(title, label, fband_tot_powers[label]))

        # #         # plot full signal with behavioral events
        # #         ax = axs[0,1]
        # #         for j, region in enumerate(subj_regions):
        # #             ax.plot(plot_time, fband_signals[label][:,j], label=region, alpha=0.5)

        # #         for j, (name, lines) in enumerate(lines_dict.items()):
        # #             ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
        # #                       transform=ax.get_xaxis_transform())

        # #         ax.set_title('Region Signals')
        # #         ax.set_xlabel('Time (s)')
        # #         ax.set_ylabel('dF/F')
        # #         ax.legend()
        # #         sig_handles, sig_labels = ax.get_legend_handles_labels()
        # #         ax.legend().remove()

        # #         for j in range(n_components):

        # #             # plot component
        # #             ax = axs[j+1,0]
        # #             ax.bar(subj_regions, fband_mix_mats[label][:,j])

        # #             ax.set_title('IC {} Region Loadings'.format(j))
        # #             ax.set_xlabel('Regions')
        # #             ax.set_ylabel('Loading')
        # #             ax.xaxis.set_tick_params(which='both', labelbottom=True)
        # #             ax.legend()
        # #             factor_handles, factor_labels = ax.get_legend_handles_labels()
        # #             ax.legend().remove()

        # #             # plot factor reconstruction with behavioral events
        # #             ax = axs[j+1,1]

        # #             # plot reconstruction
        # #             for k, region in enumerate(subj_regions):
        # #                 ax.plot(plot_time, fband_ic_recon[label][:,k,j], label=region, alpha=0.5)

        # #             for k, (name, lines) in enumerate(lines_dict.items()):
        # #                 ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
        # #                           transform=ax.get_xaxis_transform())

        # #             ax.set_title('IC Factor {} Reconstruction. Power: {:.3f}'.format(j, fband_ic_powers[label][j]))
        # #             ax.set_xlabel('Time (s)')
        # #             ax.set_ylabel('Signal Reconstruction')
        # #             ax.xaxis.set_tick_params(which='both', labelbottom=True)

        # #         ax = axs[0,0]
        # #         ax.set_axis_off()
        # #         ax.legend(factor_handles, factor_labels)
        # #         fig.legend(sig_handles, sig_labels, loc='outside right upper')
        # #         plt.show()
        # # else:

        # #

# %% Investigate Delay-embedded ICA

# epoch check:
# 237: 124737 cpoke to response, response to lick end, 124771 resp lick end, 125118 Cpoke to Response

import conv_dictionary_learning as cdl

def plot_kernels(kernels, subj_regions, n_regions, dt, dec, title, powers=None):
    n_kerns = len(kernels)

    # plot component loadings
    n_cols = len(subj_regions)
    n_rows = int(np.ceil(n_kerns/n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, layout='constrained', figsize=[4*n_cols,4*n_rows], sharey=True)
    fig.suptitle(title)

    axs = np.reshape(axs, (n_rows, n_cols))

    for i in range(n_kerns):
        row = int(np.floor(i/n_cols))
        col = int(i % n_cols)

        # plot component
        ax = axs[row,col]

        kern = kernels[i]
        kern_t = np.arange(kern.shape[1])*dt*dec

        for j in range(n_regions):
            r = subj_regions[j]
            ax.plot(kern_t, kern[j,:], label=r, alpha=0.8)

        if powers is None:
            ax.set_title('IC {}'.format(i))
        else:
            ax.set_title('IC {}, EV: {:.3f}'.format(i, powers[i]))

        ax.set_ylabel('Loading')
        ax.set_xlabel('Time (s)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend()
        factor_handles, factor_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

    fig.legend(factor_handles, factor_labels, loc='outside right center')
    plt.show()

signal_type = 'dff_iso_baseline_fband'

dec = 5
ic_dec = 2

window = 1.5
t_window = 2000

sess_ids = {199: [116828, 116914, 117565, 117317], 237: [124638, 124737, 124973, 125155], 238: [124607, 124755, 124965, 125146], #
            274: [119187, 119418, 119776, 119993], 400: [119247, 119479, 119713, 119974], 402: [119255, 119481, 119742, 120009],
            424: [124644, 124772, 124946, 125129], 483: [124622, 124735, 124924, 124957]}
sess_ids = {238: [124607]} # 400: [119974]
loc_db = bandit_loc_db

filename = 'conv_ica_mix_mats_new' #'conv_ica_mix_mats_exp'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

recalculate = False

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        all_ic_details = saved_data['all_ic_details']
        metadata = saved_data['metadata']
        recalculate = metadata['dec'] != dec or metadata['window'] != window

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    all_ic_details = {}

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)

    n_regions = len(subj_regions)
    # n_regions = 1

    subj_templates = []
    subj_template_powers = []

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        dt = fp_data['dec_info']['decimated_dt']

        n_tot = int(window/(dt*dec))

        # find middle 5 indices
        mid_idx = int(n_tot/2)
        mid_idxs = mid_idx + np.arange(-2,3)

        full_t = fp_data['time']
        _, dec_t = fp_utils.decimate(full_t, full_t, dec, dt=dt)
        plot_t = dec_t[:-n_tot]

        if sess_id in wm_sess_ids[subj_id]:
            trial_data = wm_sess_data[wm_sess_data['sessid'] == sess_id]
        else:
            trial_data = bandit_sess_data[bandit_sess_data['sessid'] == sess_id]

        # build signal matrix
        delay_signal_mat = np.empty([len(plot_t), n_tot*n_regions])
        full_signal_mat = np.empty([len(dec_t), n_regions])

        dec_signals = {}
        for i in range(n_regions):
            region = subj_regions[i]
            signal = fp_data['processed_signals'][region][signal_type]
            dec_signal, _ = fp_utils.decimate(signal, full_t, dec, dt=dt)
            dec_signals[region] = dec_signal
            full_signal_mat[:,i] = dec_signal
            for j in range(n_tot):
                delay_signal_mat[:,i+j*n_regions] = dec_signal[j:-n_tot+j]

        # remove nans from delay mat
        nans = np.any(np.isnan(delay_signal_mat), axis=1)
        if np.all(nans):
            continue

        delay_signal_mat = delay_signal_mat[~nans,:]
        delay_signal_mat = utils.z_score(delay_signal_mat, axis=0)

        full_signal_mat = utils.z_score(full_signal_mat, axis=0)

        if not sess_id in all_ic_details:

            # do ICA
            ica = FastICA(max_iter=1000, whiten_solver='eigh', n_components=int(n_tot*n_regions/ic_dec), tol=1e-3) #fun='exp', tol=1e-3,
            signal_ica = ica.fit_transform(delay_signal_mat)

            mix_mat = ica.mixing_
            demix_mat = ica.components_

            mix_mat, signal_ica, rect_mult = fp_utils.rectify_ica_output(mix_mat, signal_ica)
            demix_mat = demix_mat*rect_mult.T

            ev_details = fp_utils.ic_explained_variance(mix_mat, signal_ica, delay_signal_mat)

            all_ic_details[sess_id] = {'mix_mat': mix_mat, 'demix_mat': demix_mat, 'ev_details': ev_details}

        else:
            mix_mat = all_ic_details[sess_id]['mix_mat']
            demix_mat = all_ic_details[sess_id]['demix_mat']

        ev_component = all_ic_details[sess_id]['ev_details']['ev_component_marg']

        sources = delay_signal_mat @ demix_mat.T

        # recon_signals = fp_utils.reconstruct_ica(mix_mat, signal_ica)

        # find unique ICs that explain 99% variance in the all columns
        sorted_ic_idxs = np.argsort(ev_component)[::-1]
        sorted_evs = ev_component[sorted_ic_idxs]
        cum_ev = np.cumsum(sorted_evs)
        thresh_idx = np.argmin((cum_ev < 0.991) & (sorted_evs >= 0.0005))
        if thresh_idx == 0:
            thresh_idx = len(sorted_ic_idxs)
        primary_ic_idxs = sorted_ic_idxs[:thresh_idx]
        primary_ic_evs = ev_component[primary_ic_idxs]

        # ev_thresh = 0.005
        # primary_ic_idxs = set()
        # for i, region in enumerate(subj_regions):
        #     reg_ic_powers = np.mean(power_ic_channel[mid_idxs*n_regions+i,:], axis=0)
        #     sorted_ic_idxs = np.argsort(reg_ic_powers)[::-1]
        #     sorted_reg_powers = reg_ic_powers[sorted_ic_idxs]
        #     cum_power = np.cumsum(sorted_reg_powers)
        #     thresh_idx = np.argmin((cum_power < 0.95) & (sorted_reg_powers > ev_thresh))
        #     primary_ic_idxs.update(sorted_ic_idxs[:thresh_idx])

        # primary_ic_idxs = np.array(list(primary_ic_idxs))

        # aggregate and average ic kernels
        ic_kernels = []
        demix_kernels = []
        for i in primary_ic_idxs:
            ic_kernels.append(mix_mat[:,i].reshape(n_tot, n_regions).T)
            demix_kernels.append(demix_mat[i,:].reshape(n_tot, n_regions).T)

        title = 'Subject {}, Session {} Mixing Kernels'.format(subj_id, sess_id)
        plot_kernels(ic_kernels, subj_regions, n_regions, dt, dec, title, powers=primary_ic_evs)
        # title = 'Subject {}, Session {} Demixing Kernels'.format(subj_id, sess_id)
        # plot_kernels(demix_kernels, subj_regions, dt, dec, title)

        source_mid = sources.shape[0] // 2
        source_list = [sources[source_mid-t_window:source_mid+t_window,i][None,:] for i in primary_ic_idxs]
        # source_list = [sources[:,i][None,:] for i in primary_ic_idxs]

        similarity = fp_utils.calc_similarity(source_list, search_window=n_tot, shift_penalty=0, use_deriv=False, use_centroid=False)

        ic_labels = fp_utils.cluster_similarity(similarity, threshold=0.7)

        templates, fit_details = fp_utils.create_average_templates(ic_kernels, ic_labels, primary_ic_evs, search_window=mid_idx,
                                                                   use_deriv=True, use_centroid=True)

        template_evs = []
        demix_templates = []
        for i in np.unique(ic_labels):
            label_sel = ic_labels == i
            label_ic_idxs = primary_ic_idxs[label_sel]
            # template_ic_powers.append(np.sum(np.mean(power_ic_channel[mid_idx*n_regions:(mid_idx+1)*n_regions, label_ic_idxs], axis=0)))
            template_evs.append(np.sum(ev_component[label_ic_idxs]))
            clust_demix = [demix_kernels[i] for i, sel in enumerate(label_sel) if sel]
            clust_shifts = fit_details['shifts'][label_sel]
            clust_scales = fit_details['scales'][label_sel]
            demix_templates.append(fp_utils.build_template(clust_demix, clust_shifts, 1/clust_scales))

        template_evs = np.array(template_evs)

        # remove templates with explained variance below a threshold
        sorted_ic_idxs = np.flatnonzero(template_evs >= 0.02)
        sorted_ic_idxs = sorted_ic_idxs[np.argsort(template_evs[sorted_ic_idxs])[::-1]]

        templates = [templates[i] for i in sorted_ic_idxs]
        demix_templates = [demix_templates[i] for i in sorted_ic_idxs]
        template_evs = template_evs[sorted_ic_idxs]

        tot_power = np.sum(template_evs)

        # title = 'Subject {}, Session {}, Demixing Kernels, Reconstruction Power {:.3f}'.format(subj_id, sess_id, tot_power)
        # plot_kernels(demix_templates, subj_regions, dt, dec, title, powers=template_evs)
        title = 'Subject {}, Session {}, Mixing Kernels, Reconstruction Power {:.3f}'.format(subj_id, sess_id, tot_power)
        plot_kernels(templates, subj_regions, n_regions, dt, dec, title, powers=template_evs)

        stacked_templates, stack_details = fp_utils.stack_centered_kernels(templates, trunc_thresh=0.002, debug_plot=False, extrapolate_edges=False)
        stacked_demix_templates, _ = fp_utils.stack_centered_kernels(demix_templates, shifts=stack_details['shifts'], trunc_min=stack_details['trunc_min'], trunc_max=stack_details['trunc_max'])

        plot_model = cdl.ConvAutoencoder(W_mix_init=stacked_templates, W_demix_init=stacked_demix_templates)
        # plot_model.rescale_w()
        plot_model.plot_templates(dt=dt*dec)

        # t_mid = len(dec_t) // 2
        # signals_tensor = cdl.to_tensor(full_signal_mat[t_mid-t_window:t_mid+t_window,:].T)
        # signals_tensor = cdl.to_tensor(full_signal_mat[t_mid-5000:t_mid+5000,:].T)
        keep, results, details = cdl.find_unique_kernels(full_signal_mat.T, stacked_templates, zero_idx=stack_details['center_idx'],
                                                         ev_thresh=0.02, order=np.arange(len(templates)), verbose=False,
                                                         atol=1e-4, btol=1e-4)

        model = cdl.ConvAutoencoder(W_mix_init=results['W_mix_orig'], W_demix_init=results['W_demix_est'], zero_idx=stack_details['center_idx'],
                                    n_iter=20000, early_stopping_tol=1e-6, verbose=True, lr=1e-3,
                                    lambda_w=0.01, lambda_w_alpha=0, lambda_smooth=0.1, lambda_ica=0, lambda_recon_sim=1)
        title = 'Subject {}, Session {}, Kernel Selection Results'.format(subj_id, sess_id)
        model.plot_templates(title=title, dt=dt*dec)
        fit_results_demix = model.fit(full_signal_mat.T, fix_w_mix=True, fix_w_alpha=True)
        fit_results_mix = model.fit(full_signal_mat.T, fix_w_demix=True, fix_w_alpha=True)
        ev_details = model.calc_ev(full_signal_mat.T)
        title = 'Subject {}, Session {}, Reconstruction EV {:.3f}'.format(subj_id, sess_id, ev_details['ev_total'])
        model.plot_templates(title=title, dt=dt*dec)

        all_ic_details[sess_id]['best_fit_ev_details'] = ev_details

        plt.show()

        # model.plot_reconstruction(full_signal_mat.T)
        # model.plot_impulse_response()

        # model = cdl.ConvAutoencoder(W_mix_init=results['W_mix_est'], W_demix_init=results['W_demix_est'])
        # model.rescale_w()
        # # title = 'Subject {}, Session {}, Mixing Kernels, Reconstruction Power {:.3f}'.format(subj_id, sess_id, results['ev_est'].mean())
        # model.plot_templates()

        # h_corr = cdl.measure_h_corr(results['H_lsq'])
        # h_corr_demix = cdl.measure_h_corr(results['H_est'])

        # model_fit, results = cdl.solve_conv_autoencoder(full_signal_mat.T, stacked_templates, zero_idx=stack_details['center_idx'], ev_thresh=0.005,
        #                                            order=np.arange(len(templates)), max_iter=50, error_tol=1e-4, eval_iter=3,
        #                                            ridge_lam=0.1, update_alpha=0.5, atol=1e-3, btol=1e-3, verbose=False)

        # plot_model = cdl.ConvAutoencoder(W_mix_init=model_fit['W_mix'], W_demix_init=model_fit['W_demix'])
        # plot_model.plot_templates()

        # h_corr = cdl.calc_h_corr(model_fit['H'])

        # model_solve = cdl.ConvAutoencoder(W_mix_init=stacked_templates[results['keep']], W_demix_init=model_fit['W_demix'],
        #                             verbose=True, lambda_w=1e-3, lambda_smooth=0.01, lambda_ica=0, lambda_recon_sim=0.1, n_iter=20000, lr=1e-3)
        # fit_results_solve = model_solve.fit(full_signal_mat.T, fix_w_mix=True)
        # model_solve.plot_templates()
        # ev_details_solve = model_solve.calc_ev(full_signal_mat.T)

        # model_ica = cdl.ConvAutoencoder(W_mix_init=stacked_templates[results['keep']], W_demix_init=stacked_demix_templates[results['keep']],
        #                             verbose=True, lambda_w=1e-3, lambda_smooth=0.01, lambda_ica=0, lambda_recon_sim=0.1, n_iter=20000, lr=1e-3)
        # fit_results_ica = model_ica.fit(full_signal_mat.T, fix_w_mix=True)
        # model_ica.plot_templates()
        # ev_details_ica = model_ica.calc_ev(full_signal_mat.T)

        # model = cdl.ConvAutoencoder(W_mix_init=stacked_templates, W_demix_init=stacked_demix_templates, zero_idx=stack_details['center_idx'], early_stopping_tol=1e-5,
        #                             verbose=True, lambda_w=0.01, lambda_w_alpha=0.01, lambda_smooth=0.01, lambda_ica=0, lambda_recon_sim=0.1, lr=1e-3)

        # keep, results, details = model.find_unique_kernels(full_signal_mat.T, order=np.arange(len(templates)), recalc_W_demix=True)

        # plot_model = cdl.ConvAutoencoder(W_mix_init=results['W_mix'].clone(), W_demix_init=results['W_demix'].clone(), W_alpha_init=results['W_alpha'].clone(),
        #                                  verbose=True, lambda_w=0.01, lambda_w_alpha=0.01, lambda_smooth=0.001, lambda_ica=0, lambda_recon_sim=0.1, lr=1e-3)
        # plot_model.plot_templates()
        # fit_results = plot_model.fit(full_signal_mat.T, fix_w_mix=False)
        # plot_model.plot_templates()
        # ev_details = plot_model.calc_ev(full_signal_mat.T)

        # fit_results = model.fit(full_signal_mat.T, fix_w_mix=True)
        # model.plot_templates()
        # ev_details_pre = model.calc_ev(full_signal_mat.T)
        # fit_results = model.fit(full_signal_mat.T, fix_w_mix=False)
        # model.plot_templates()
        # ev_details_post = model.calc_ev(full_signal_mat.T)

        # sources, x_recon = model.extract_and_reconstruct(full_signal_mat.T)
        # loss, cosine = model.reconstruction_similarity_loss(x_recon)


        # # templates = []
        # # demix_templates = []
        # # for i in np.unique(ic_labels):
        # #     label_sel = ic_labels == i
        # #     clust_mix = [ic_kernels[i] for i, sel in enumerate(label_sel) if sel]
        # #     clust_demix = [demix_kernels[i] for i, sel in enumerate(label_sel) if sel]
        # #     clust_shifts = fit_details['shifts'][label_sel]
        # #     clust_scales = fit_details['scales'][label_sel]
        # #     clust_shifts *= -1
        # #     clust_shifts -= min(clust_shifts)
        # #     templates.append(fp_utils.build_template(clust_mix, clust_shifts, 1/clust_scales))
        # #     demix_templates.append(fp_utils.build_template(clust_demix, clust_shifts, clust_scales))

        # # cluster mixing kernels
        # # templates, ic_labels, fit_details = fp_utils.cluster_kernels(ic_kernels, similarity_threshold=0.1, search_window=n_tot//4, overlap_prop_thresh=0.5,
        # #                                                         linkage_method='average', shift_penalty=0.5)

        # # average demixing kernels and sources
        # demix_templates = []
        # for i in np.unique(ic_labels):
        #     label_sel = ic_labels == i
        #     clust_demix = [demix_kernels[i] for i, sel in enumerate(label_sel) if sel]
        #     clust_shifts = fit_details['shifts'][label_sel]
        #     clust_scales = fit_details['scales'][label_sel]
        #     demix_templates.append(fp_utils.build_template(clust_demix, clust_shifts, 1/clust_scales))

        # # use mixing kernel indices to compute the average

        # # now sort the templates by their total explained variance in the middle signal columns
        # template_ic_powers = []
        # for i in np.unique(ic_labels):
        #     label_sel = ic_labels == i
        #     label_ic_idxs = primary_ic_idxs[label_sel]
        #     # template_ic_powers.append(np.sum(np.mean(power_ic_channel[mid_idx*n_regions:(mid_idx+1)*n_regions, label_ic_idxs], axis=0)))
        #     template_ic_powers.append(np.sum(power_ic_total[label_ic_idxs]))

        # template_ic_powers = np.array(template_ic_powers)

        # # remove templates under a threshold
        # template_sel = template_ic_powers >= 0.01
        # template_ic_powers = template_ic_powers[template_sel]
        # templates = [templates[i] for i, sel in enumerate(template_sel) if sel]
        # demix_templates = [demix_templates[i] for i, sel in enumerate(template_sel) if sel]

        # sorted_ic_idxs = np.argsort(template_ic_powers)[::-1]

        # templates = [templates[i] for i in sorted_ic_idxs]
        # demix_templates = [demix_templates[i] for i in sorted_ic_idxs]
        # template_ic_powers = template_ic_powers[sorted_ic_idxs]

        # tot_power = np.sum(template_ic_powers)

        # # title = 'Subject {}, Session {}, Demixing Kernels, Reconstruction Power {:.3f}'.format(subj_id, sess_id, tot_power)
        # # plot_kernels(demix_templates, subj_regions, dt, dec, title, powers=template_ic_powers)
        # title = 'Subject {}, Session {}, Mixing Kernels, Reconstruction Power {:.3f}'.format(subj_id, sess_id, tot_power)
        # plot_kernels(templates, subj_regions, dt, dec, title, powers=template_ic_powers)

        # subj_templates.extend(templates)
        # subj_template_powers.extend(template_ic_powers)

        # # clust_templates, clust_labels, clust_fit_details = fp_utils.cluster_matched_filter_similarity(full_signal_mat.T, templates)

        # # refine templates through convolutional dictionary learning
        # stacked_templates, stack_details = fp_utils.stack_centered_kernels(templates, trunc_thresh=0.002)
        # # stacked_demix_templates, _ = fp_utils.stack_centered_kernels(demix_templates, **stack_details)

        # model = cdl.ConvDictionaryLearning(stacked_templates.shape[0], stacked_templates.shape[2],
        #                                    W_init=stacked_templates[:,:,:], verbose=True,
        #                                    lambda_sparse=1, lambda_smooth=0,
        #                                    lambda_template=0, lambda_ortho=1,
        #                                    lambda_w_l1=0, lambda_x_ortho=0,
        #                                    lambda_ica=0,
        #                                    n_iter = 10000, early_stopping_tol=1e-6, h_init_method='random')
        # model.rescale_w()
        # model.plot_templates()
        # fit_results = model.fit(full_signal_mat.T, fix_W=True)

        # model = cdl.ConvAutoencoder(W_mix_init=stacked_templates, W_demix_init=stacked_demix_templates, #[:-1,:,:]
        #                             verbose=True, lambda_w=1e-3, lambda_ica=0, lambda_recon_sim=0.1, n_iter=20000, lr=1e-3)
        # # scales = model.rescale_w_to_data(full_signal_mat.T)
        # scales = model.rescale_w()
        # model.plot_templates()

        # fit_results = model.fit(full_signal_mat.T, fix_w_mix=True)
        # fit_results = model.fit(full_signal_mat.T, fix_w_mix=False)
        # model.plot_templates()
        # model.plot_reconstruction(full_signal_mat.T)
        # model.plot_loadings(full_signal_mat.T)

        # model.plot_impulse_response()
        # sources, x_recon = model.extract_and_reconstruct(full_signal_mat.T)
        # loss, cosine = model.reconstruction_similarity_loss(x_recon)


        with open(save_path, 'wb') as f:
            pickle.dump({'all_ic_details': all_ic_details,
                         'metadata': {'dec': dec,
                                      'window': window}},
                        f)


    # subj_template_powers = np.array(subj_template_powers)

    # # cluster all kernels across sessions
    # comb_templates, comb_labels, _ = fp_utils.cluster_kernels(subj_templates, similarity_threshold=0.1, min_overlap=10, overlap_prop_thresh=0.5,
    #                                                        linkage_method='average', shift_penalty=0.5)

    # # # get rid of clusters with only one label
    # # ref_comb_templates = []
    # # for i, cluster in enumerate(np.unique(comb_labels)):
    # #     if sum(comb_labels == cluster) > 1:
    # #         ref_comb_templates.append(comb_templates[i])

    # ref_comb_templates = comb_templates

    # # sort combined templates by average explained power
    # comb_template_powers = []
    # for i in np.unique(comb_labels):
    #     label_sel = comb_labels == i
    #     comb_template_powers.append(np.mean(subj_template_powers[label_sel]))

    # comb_template_powers = np.array(comb_template_powers)

    # tot_comb_power = np.sum(comb_template_powers)

    # sorted_template_idxs = np.argsort(comb_template_powers)[::-1]

    # ref_comb_templates = [ref_comb_templates[i] for i in sorted_template_idxs]
    # comb_template_powers = comb_template_powers[sorted_template_idxs]

    # tot_power = np.sum(comb_template_powers)

    # title = 'Subject {} Combined Templates, Avg Reconstruction Power {:.3f}'.format(subj_id, tot_comb_power)
    # plot_kernels(ref_comb_templates, subj_regions, dt, dec, title, powers=comb_template_powers)

        # trial_start_ts = fp_data['trial_start_ts'][:-1]
        # cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        # cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        # #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        # cue_ts = trial_start_ts + trial_data['response_cue_time']
        # cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        # response_ts = trial_start_ts + trial_data['response_time']
        # outcome_ts = trial_start_ts + trial_data['reward_time']
        # reward_ts = outcome_ts[trial_data['reward'] > 0]
        # unreward_ts = outcome_ts[trial_data['reward'] == 0]
        # lick_end_ts = trial_start_ts + trial_data['lick_stop_time']

        # lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
        #               'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
        #               'Reward': reward_ts, 'Unreward': unreward_ts, 'Lick Stop': lick_end_ts}

        # fig, axs = plt.subplots(n_components+1, 2, layout='constrained', figsize=[24,4*(n_components+1)], width_ratios=[1,10], sharex='col')
        # fig.suptitle('Subject {}, Session {}, Reconstruction Power {:.3f}'.format(subj_id, sess_id, tot_power))

        # plot_time = plot_t[~nans]

        # # plot full signal with behavioral events
        # ax = axs[0,1]
        # for i, region in enumerate(subj_regions):
        #     ax.plot(plot_time, signal_mat[:,i], alpha=0.5)

        # for i, (name, lines) in enumerate(lines_dict.items()):
        #     ax.vlines(lines, 0, 1, label=name, color='C{}'.format(i+n_regions), linestyles='dashed',
        #               transform=ax.get_xaxis_transform())

        # ax.set_title('Region Signals')
        # ax.set_xlabel('Time (s)')
        # ax.set_ylabel('dF/F')
        # ax.legend()
        # sig_handles, sig_labels = ax.get_legend_handles_labels()
        # ax.legend().remove()

        # for i in range(n_components):

        #     # plot component
        #     ax = axs[i+1,0]

        #     comp_mix_mat = mix_mat[:,primary_ic_idxs[i]].reshape(n_tot, n_regions)

        #     for k, r in enumerate(subj_regions):
        #         ax.plot(comp_t, comp_mix_mat[:,k], label=r, alpha=0.5)

        #     ax.set_title('IC {} Region Loadings'.format(i))
        #     ax.set_ylabel('Loading')
        #     ax.set_xlabel('Time (s)')
        #     ax.xaxis.set_tick_params(which='both', labelbottom=True)
        #     ax.legend()
        #     factor_handles, factor_labels = ax.get_legend_handles_labels()
        #     ax.legend().remove()

        #     # plot IC value with behavioral events
        #     ax = axs[i+1,1]

        #     ax.plot(plot_time, signal_ica[:,i])

        #     for k, (name, lines) in enumerate(lines_dict.items()):
        #         ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
        #                   transform=ax.get_xaxis_transform())

        #     ax.set_title('IC Factor {}. Power: {:.3f}'.format(i, primary_ic_powers[i]))
        #     ax.set_xlabel('Time (s)')
        #     ax.set_ylabel('Signal IC')
        #     ax.xaxis.set_tick_params(which='both', labelbottom=True)

        # ax = axs[0,0]
        # ax.set_axis_off()
        # ax.legend(factor_handles, factor_labels)
        # fig.legend(sig_handles, sig_labels, loc='outside right upper')
        # plt.show()


# %% Investigate frequency band ICA

signal_type = 'z_dff_iso_baseline_fband'

use_fbands = True
freq_bands = [[0,0.03], [0.03,0.3], [0.3,2], [2,10]]
band_labels = ['{}-{}'.format(b[0], b[1]) for b in freq_bands]

component_dec = 1

sess_ids = {274: [119776, 119993], 400: [119785, 120001], 402: [119839, 120009]} #
loc_db = bandit_loc_db

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)

    n_regions = len(subj_regions)
    n_components = n_regions-component_dec

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        t = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts']

        signal_mat = np.stack([fp_data['processed_signals'][r][signal_type] for r in subj_regions], axis=1)

        # build signal matrix
        if use_fbands:

            fband_signals = {}
            fband_mix_mats = {}
            fband_tot_powers = {}
            fband_ic_powers = {}
            fband_ic_channel_powers = {}
            fband_ic_recon = {}

            # limit filtered ICA signals to avoid edge effects from filtering
            t_sel = np.full_like(t, False, dtype=bool)
            t_sel[t < trial_start_ts[0]] = True
            t_sel[t > trial_start_ts[-1]] = True

            signal_mat[t_sel,:] = np.nan
            nans = np.any(np.isnan(signal_mat), axis=1)
            signal_mat = signal_mat[~nans,:]

            for fband, label in zip(freq_bands, band_labels):
                fband_signal_mat = np.concatenate([fp_utils.decompose_signal_fbands(
                    fp_data['processed_signals'][r][signal_type], [fband], 1/dt).T for r in subj_regions], axis=1)

                # remove nans
                fband_signal_mat = fband_signal_mat[~nans,:]

                fband_signal_mat = utils.z_score(fband_signal_mat, axis=0)

                # do ICA
                ica = FastICA(n_components=n_components)
                signal_ica = ica.fit_transform(fband_signal_mat)
                ica_mix_mat = ica.mixing_
                ica_demix_mat = ica.components_

                # rearrange columns of mixing matrix so that the diagonal has all the largest values
                ica_mix_mat, signal_ica, _ = fp_utils.rectify_ica_output(ica_mix_mat, signal_ica)

                power_total, power_ic_total, power_ic_channel = fp_utils.ic_explained_power(ica_mix_mat, signal_ica, fband_signal_mat)

                # sort by power per region
                order = []
                all_cols = np.arange(n_components)
                for i in range(n_components):
                    max_idx = np.argmax(power_ic_channel[i,all_cols])
                    max_idx = all_cols[max_idx]
                    all_cols = all_cols[all_cols != max_idx]
                    order.append(max_idx)

                power_ic_total = power_ic_total[order]
                power_ic_channel = power_ic_channel[:, order]
                ica_mix_mat = ica_mix_mat[:, order]
                signal_ica = signal_ica[:, order]

                # calculate reconstruction of each channel from each IC
                ic_reconstruction = fp_utils.reconstruct_ica(ica_mix_mat, signal_ica)

                fband_signals[label] = fband_signal_mat
                fband_mix_mats[label] = ica_mix_mat
                fband_tot_powers[label] = power_total
                fband_ic_powers[label] = power_ic_total
                fband_ic_channel_powers[label] = power_ic_channel
                fband_ic_recon[label] = ic_reconstruction
        else:

            # remove nans
            nans = np.any(np.isnan(signal_mat), axis=1)
            signal_mat = signal_mat[~nans,:]

            # do ICA
            ica = FastICA(n_components=n_components)
            signal_ica = ica.fit_transform(signal_mat)
            ica_mix_mat = ica.mixing_
            ica_demix_mat = ica.components_

            # rectify the columns of the mixing matrix so that the largest values are all positive
            ica_mix_mat, signal_ica, _ = fp_utils.rectify_ica_output(ica_mix_mat, signal_ica)

            power_total, power_ic_total, power_ic_channel = fp_utils.ic_explained_power(ica_mix_mat, signal_ica, signal_mat)

            # sort by power per region
            order = []
            all_cols = np.arange(n_components)
            for i in range(n_components):
                max_idx = np.argmax(power_ic_channel[i,all_cols])
                max_idx = all_cols[max_idx]
                all_cols = all_cols[all_cols != max_idx]
                order.append(max_idx)

            power_ic_total = power_ic_total[order]
            power_ic_channel = power_ic_channel[:, order]
            ica_mix_mat = ica_mix_mat[:, order]
            signal_ica = signal_ica[:, order]

            # calculate reconstruction of each channel from each IC
            ic_reconstruction = fp_utils.reconstruct_ica(ica_mix_mat, signal_ica)

        trial_data = loc_db.get_behavior_data(sess_id)

        plot_time = t[~nans]
        trial_start_ts = trial_start_ts[:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        title = 'Subject {}, Session {}'.format(subj_id, sess_id)

        if use_fbands:

            # build full signal reconstructions and total power of full signal
            full_signal_recon = np.stack([fband_ic_recon[l] for l in band_labels], axis=-1)
            full_signal_recon_ic = np.sum(full_signal_recon, axis=-1)
            full_signal_recon = np.sum(full_signal_recon_ic, axis=-1)

            # calculate total power and power per ic x region
            region_power = np.nansum(signal_mat**2, axis=0)
            tot_power = np.sum(region_power)

            power_ic_region = np.zeros((n_regions, n_components))

            power_total = 1 - np.sum((signal_mat - full_signal_recon)**2)/tot_power

            for i in range(n_components):
                recon_ic = full_signal_recon_ic[:,:,i]
                power_ic_region[:, i] =  1 - np.sum((signal_mat - recon_ic)**2, axis=0) / region_power

            for fband, label in zip(freq_bands, band_labels):
                fig, axs = plt.subplots(n_components+1, 2, layout='constrained', figsize=[20,6*(n_components+1)], width_ratios=[1,8], sharex='col')
                fig.suptitle('Subject {}, Session {}, Band: {}, Reconstruction Power {:.3f}'.format(subj_id, sess_id, label, fband_tot_powers[label]))

                # plot full signal with behavioral events
                ax = axs[0,1]
                for j, region in enumerate(subj_regions):
                    ax.plot(plot_time, fband_signals[label][:,j], label=region, alpha=0.5)

                for j, (name, lines) in enumerate(lines_dict.items()):
                    ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                              transform=ax.get_xaxis_transform())

                ax.set_title('Region Signals')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('dF/F')
                ax.legend()
                sig_handles, sig_labels = ax.get_legend_handles_labels()
                ax.legend().remove()

                for j in range(n_components):

                    # plot component
                    ax = axs[j+1,0]
                    ax.bar(subj_regions, fband_mix_mats[label][:,j])

                    ax.set_title('IC {} Region Loadings'.format(j))
                    ax.set_xlabel('Regions')
                    ax.set_ylabel('Loading')
                    ax.xaxis.set_tick_params(which='both', labelbottom=True)
                    ax.legend()
                    factor_handles, factor_labels = ax.get_legend_handles_labels()
                    ax.legend().remove()

                    # plot factor reconstruction with behavioral events
                    ax = axs[j+1,1]

                    # plot reconstruction
                    for k, region in enumerate(subj_regions):
                        ax.plot(plot_time, fband_ic_recon[label][:,k,j], label=region, alpha=0.5)

                    for k, (name, lines) in enumerate(lines_dict.items()):
                        ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
                                  transform=ax.get_xaxis_transform())

                    ax.set_title('IC Factor {} Reconstruction. Power: {:.3f}'.format(j, fband_ic_powers[label][j]))
                    ax.set_xlabel('Time (s)')
                    ax.set_ylabel('Signal Reconstruction')
                    ax.xaxis.set_tick_params(which='both', labelbottom=True)

                ax = axs[0,0]
                ax.set_axis_off()
                ax.legend(factor_handles, factor_labels)
                fig.legend(sig_handles, sig_labels, loc='outside right upper')
                plt.show()
        else:

            fig, axs = plt.subplots(n_components+1, 2, layout='constrained', figsize=[20,6*(n_components+1)], width_ratios=[1,8], sharex='col')
            fig.suptitle('Subject {}, Session {}, Reconstruction Power {:.3f}'.format(subj_id, sess_id, power_total))

            # plot full signal with behavioral events
            ax = axs[0,1]
            for j, region in enumerate(subj_regions):
                ax.plot(plot_time, signal_mat[:,j], label=region, alpha=0.5)

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                          transform=ax.get_xaxis_transform())

            ax.set_title('Region Signals')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('dF/F')
            ax.legend()
            sig_handles, sig_labels = ax.get_legend_handles_labels()
            ax.legend().remove()

            for j in range(n_components):

                # plot component
                ax = axs[j+1,0]
                ax.bar(subj_regions, ica_mix_mat[:,j])

                ax.set_title('IC {} Region Loadings'.format(j))
                ax.set_xlabel('Regions')
                ax.set_ylabel('Loading')
                ax.xaxis.set_tick_params(which='both', labelbottom=True)
                ax.legend()
                factor_handles, factor_labels = ax.get_legend_handles_labels()
                ax.legend().remove()

                # plot factor reconstruction with behavioral events
                ax = axs[j+1,1]

                # plot reconstruction
                for k, region in enumerate(subj_regions):
                    ax.plot(plot_time, ic_reconstruction[:,k,j], label=region, alpha=0.5)

                for k, (name, lines) in enumerate(lines_dict.items()):
                    ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
                              transform=ax.get_xaxis_transform())

                ax.set_title('IC Factor {} Reconstruction. Power: {:.3f}'.format(j, power_ic_total[j]))
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Signal Reconstruction')
                ax.xaxis.set_tick_params(which='both', labelbottom=True)

            ax = axs[0,0]
            ax.set_axis_off()
            ax.legend(factor_handles, factor_labels)
            fig.legend(sig_handles, sig_labels, loc='outside right upper')
            plt.show()


# %% Investigate NMF

# Produces inconsistent solutions from run to run

from sklearn.decomposition import NMF

sess_ids = {274: [119776, 119993], 400: [119785, 120001], 402: [119839, 120009]} #
loc_db = bandit_loc_db

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)

    n_regions = len(subj_regions)

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        t = fp_data['time']

        # build signal matrix
        signal_mat = np.stack([fp_data['processed_signals'][r][signal_type] for r in subj_regions], axis=1)

        # remove nans
        nans = np.any(np.isnan(signal_mat), axis=1)
        signal_mat = signal_mat[~nans,:]

        # make each signal positive and have equal power
        offset = np.nanmin(signal_mat, axis=0, keepdims=True) - 0.01
        signal_mat -= offset

        # normalize the power across channels
        power = np.sqrt(np.mean(signal_mat**2, axis=0, keepdims=True))
        signal_mat /= power

        # do NMF
        model = NMF(n_components=n_regions, init='random')
        signal_factors = model.fit_transform(signal_mat) # (T, n_components)
        mix_mat = model.components_ # (n_components, n_channels)

        power_total, power_ic_total, power_ic_channel = fp_utils.ic_explained_power(mix_mat.T, signal_factors, signal_mat)

        # sort by total power
        order = np.argsort(power_ic_total)[::-1]

        power_ic_total = power_ic_total[order]
        power_ic_channel = power_ic_channel[:, order]
        mix_mat = mix_mat[order, :]
        signal_factors = signal_factors[:, order]

        # calculate reconstruction of each channel from each factorization
        nmf_reconstruction = fp_utils.reconstruct_ica(mix_mat.T, signal_factors)

        trial_data = loc_db.get_behavior_data(sess_id)
        plot_time = t[~nans]

        ts = fp_data['time']
        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        title = 'Subject {}, Session {}'.format(subj_id, sess_id)

        fig, axs = plt.subplots(n_regions+1, 2, layout='constrained', figsize=[20,6*(n_regions+1)], width_ratios=[1,8], sharex='col')
        fig.suptitle('Subject {}, Session {}, Reconstruction Power {:.3f}'.format(subj_id, sess_id, power_total))

        # plot full signal with behavioral events
        ax = axs[0,1]
        for j, region in enumerate(subj_regions):
            ax.plot(plot_time, signal_mat[:,j], label=region, alpha=0.5)

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                      transform=ax.get_xaxis_transform())

        ax.set_title('Region Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.legend()
        sig_handles, sig_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        for j in range(n_regions):

            # plot component
            ax = axs[j+1,0]
            ax.bar(subj_regions, mix_mat[j,:])

            ax.set_title('NMF {} Region Loadings'.format(j))
            ax.set_xlabel('Regions')
            ax.set_ylabel('Loading')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()
            factor_handles, factor_labels = ax.get_legend_handles_labels()
            ax.legend().remove()

            # plot factor reconstruction with behavioral events
            ax = axs[j+1,1]

            # plot reconstruction
            for k, region in enumerate(subj_regions):
                ax.plot(plot_time, nmf_reconstruction[:,k,j], label=region, alpha=0.5)

            for k, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
                          transform=ax.get_xaxis_transform())

            ax.set_title('NMF Factor {} Reconstruction. Power: {:.3f}'.format(j, power_ic_total[j]))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal Reconstruction')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)

        ax = axs[0,0]
        ax.set_axis_off()
        ax.legend(factor_handles, factor_labels)
        fig.legend(sig_handles, sig_labels, loc='outside right upper')
        plt.show()

# %% Time delay embedded PCA

# perform PCA on signals stacked across time delays
# for example, if I have a 1xN signal from a region, I can build a TxN signal by sequentially stacking
# the signal delayed by 1 to T timepoints. The resulting loading vectors for each PCA axis will end up
# giving periodic components that are similar to a dominant frequency/wavelet

from sklearn.decomposition import PCA

t_pre = 4
t_post = 4

signal_type = 'z_dff_iso_baseline_fband'

ignore_regions = []

sess_ids = {400: [119974]} #{274: [119776, 119993], 400: [119785, 120001], 402: [119839, 120009]}
loc_db = bandit_loc_db

n_pcs = 6
dec = 4

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)

    subj_regions = [r for r in subj_regions if r not in ignore_regions]

    n_regions = len(subj_regions)

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        dt = fp_data['dec_info']['decimated_dt']

        n_pre = int(t_pre/(dt*dec))
        n_post = int(t_post/(dt*dec))
        n_tot = n_pre + n_post

        comp_t = np.arange(-n_pre, n_post)*dt*dec

        trial_data = loc_db.get_behavior_data(sess_id)

        full_t = fp_data['time']
        _, dec_t = fp_utils.decimate(full_t, full_t, dec, dt=dt)
        plot_t = dec_t[n_pre:-n_post]

        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        # build signal matrix
        signal_mat = np.empty([len(plot_t), n_tot*n_regions])

        dec_signals = {}
        for i, region in enumerate(subj_regions):
            signal = fp_data['processed_signals'][region][signal_type]
            dec_signal, _ = fp_utils.decimate(signal, full_t, dec, dt=dt)
            dec_signals[region] = dec_signal
            for j in range(n_tot):
                signal_mat[:,i*n_tot+j] = dec_signal[j:-n_tot+j]

        # remove nans
        nans = np.any(np.isnan(signal_mat), axis=1)
        signal_mat = signal_mat[~nans,:]

        # do PCA
        pca = PCA()
        signal_pca = pca.fit_transform(signal_mat)
        explained_var = pca.explained_variance_ratio_
        components = pca.components_

        # plot results
        fig, axs = plt.subplots(n_pcs+1, 2, layout='constrained', figsize=[20,6*(n_pcs+1)], width_ratios=[1,5], sharex='col')
        fig.suptitle('Subject {}, Session {}, {}'.format(subj_id, sess_id, region))

        # plot full signal with behavioral events
        ax = axs[0,1]
        for region in subj_regions:
            ax.plot(dec_t, dec_signals[region], label=region, alpha=0.4)

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                      transform=ax.get_xaxis_transform())

        ax.set_title('Region Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.legend()

        for i in range(n_pcs):

            # plot component
            ax = axs[i+1,0]
            for j, region in enumerate(subj_regions):
                ax.plot(comp_t, components[i,j*n_tot:(j+1)*n_tot], label=region)

            ax.set_title('PC {}'.format(i))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()

            # plot component loading with behavioral events
            ax = axs[i+1,1]
            ax.plot(plot_t[~nans], signal_pca[:,i], label='_')

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+1), linestyles='dashed',
                          transform=ax.get_xaxis_transform())

            ax.set_title('PC {} Loading. Explained Var: {:.2f}'.format(i, explained_var[i]))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PC Loading')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()

        plt.show()

# %% Perform Dynamic Mode Decomposition

from pydmd import BOPDMD, DMD, HankelDMD
from pydmd.preprocessing import hankel_preprocessing
from pydmd.plotter import plot_summary

t_pre = 1
t_post = 1

signal_type = 'z_dff_iso_baseline_fband'

sess_ids = {274: [119776, 119993], 400: [119785, 120001], 402: [119839, 120009]}
loc_db = bandit_loc_db

svd_rank = 0
n_plot = 4
dec = 4

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)

    n_regions = len(subj_regions)

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        dt = fp_data['dec_info']['decimated_dt']

        n_pre = int(utils.convert_to_multiple(t_pre, dt)/(dt*dec))
        n_post = int(utils.convert_to_multiple(t_post, dt)/(dt*dec))+1
        n_tot = n_pre + n_post

        comp_t = np.arange(-n_pre, n_post)*dt*dec

        trial_data = loc_db.get_behavior_data(sess_id)

        full_t = fp_data['time']
        _, dec_t = fp_utils.decimate(full_t, full_t, dec, dt=dt)
        plot_t = dec_t[n_pre:-n_post]

        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        # build signal matrix
        signal_mat = np.empty([n_tot*n_regions, len(plot_t)])

        dec_signals = {}
        for i, region in enumerate(subj_regions):
            signal = fp_data['processed_signals'][region][signal_type]
            dec_signal, _ = fp_utils.decimate(signal, full_t, dec, dt=dt)
            dec_signals[region] = dec_signal
            for j in range(n_tot):
                signal_mat[i*n_tot+j,:] = dec_signal[j:-n_tot+j]

        # remove nans
        nans = np.any(np.isnan(signal_mat), axis=0)
        signal_mat = signal_mat[:,~nans]

        # do DMD
        #dmd = BOPDMD(svd_rank=svd_rank)
        #dmd.fit(signal_mat, plot_t[~nans])
        dmd = DMD(svd_rank=svd_rank, sorted_eigs='abs')
        dmd.fit(signal_mat)
        eigs = dmd.eigs
        modes = dmd.modes
        dynamics = dmd.dynamics
        amps = dmd.amplitudes

        plot_summary(dmd, max_sval_plot=190)

        # plot results
        fig, axs = plt.subplots(n_plot+1, 2, layout='constrained', figsize=[20,6*n_plot], width_ratios=[1,7], sharex='col')
        fig.suptitle('Subject {}, Session {}, {}'.format(subj_id, sess_id, region))

        # plot full signal with behavioral events
        ax = axs[0,1]
        for region in subj_regions:
            ax.plot(dec_t, dec_signals[region], label=region, alpha=0.4)

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                      transform=ax.get_xaxis_transform())

        ax.set_title('Region Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.legend()

        for i in range(n_plot):

            # plot component
            ax = axs[i+1,0]
            for j, region in enumerate(subj_regions):
                ax.plot(comp_t, modes[j*n_tot:(j+1)*n_tot,i].real, label=region)

            ax.set_title('DMD Mode {}'.format(i))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()

            # plot component loading with behavioral events
            ax = axs[i+1,1]
            ax.plot(plot_t[~nans], dynamics[i].real, label='_')

            for j, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+1), linestyles='dashed',
                          transform=ax.get_xaxis_transform())

            ax.set_title('DMD Mode {} Dynamics. Amplitude: {:.2f}'.format(i, amps[i]))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('PC Loading')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()

        plt.show()

# %% Investigate caiman deconvolution methods

# # this seems to work well for dorsal striatum, but not so well in PL or NAc,
# # these two regions seem to have multiple variable timescales

# import caiman.source_extraction.cnmf.deconvolution as caim_deconv

# sess_ids = {400: [119785, 120001], 402: [119839, 120009]} # 274: [119776, 119993],
# loc_db = bandit_loc_db
# region_lpf = {'PL': 4, 'NAc': 6, 'DMS': 8, 'DLS': 8, 'TS': 8}
# region_lags = {'PL': 6, 'NAc': 2, 'DMS': 1, 'DLS': 1, 'TS': 1}
# region_ar_lags = {'PL': 4, 'NAc': 0.5, 'DMS': 0.2, 'DLS': 0.2, 'TS': 0.2}
# dec = 2
# signal_type = 'dff_iso_baseline_fband'

# for subj_id in sess_ids.keys():

#     subj_regions = get_subj_regions(subj_id)

#     reg_mapping = get_subj_region_mapping(subj_regions)

#     for sess_id in sess_ids[subj_id]:

#         fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
#                                        baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
#         fp_data = fp_data[subj_id][sess_id]
#         dt = fp_data['dec_info']['decimated_dt']

#         t = fp_data['time']

#         fig, axs = plt.subplots(len(subj_regions), 1, figsize=(20,5*len(subj_regions)), layout='constrained')
#         fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

#         for i, region in enumerate(subj_regions):
#             signal = fp_data['processed_signals'][region][signal_type]

#             if all(np.isnan(signal)):
#                 continue

#             dec_signal, dec_t = fp_utils.decimate(signal, t, dec, dt=dt)

#             auto_corr, corr_lags = fp_utils.correlate(dec_signal, dec_signal, dt*dec, neg_lag=0, pos_lag=region_lags[reg_mapping[region]])

#             # calculate auto-regressive decay timescale
#             ar_lags = int(region_ar_lags[reg_mapping[region]] / (dec*dt))
#             g_ar2 = fp_utils.get_signal_ar_params(dec_signal, 2, ar_lags)

#             ar2_modes = fp_utils.get_ar_modes(g_ar2, dt*dec)
#             max_ar2_mode = {'tau': 0}
#             for m in ar2_modes:
#                 if m['tau'] > max_ar2_mode['tau']:
#                     max_ar2_mode = m

#             g_ar1 = fp_utils.get_ar_params_from_modes({'tau': max_ar2_mode['tau'], 'freq': 0}, dt*dec)

#             lpf_signal = fp_utils.filter_signal(signal, region_lpf[reg_mapping[region]], 1/dt)
#             dec_lpf_signal, dec_t = fp_utils.decimate(lpf_signal, t, dec, dt=dt)

#             # AR(1) deconvolution
#             ar1_deconv = fp_utils.extract_ar_source(dec_lpf_signal, g_ar1)

#             dec_signal_nans, nan_idxs = fp_utils.fill_signal_nans(dec_lpf_signal)
#             foopsi_deconv, bl, cl, g, sn, foopsi_spikes, lam = caim_deconv.constrained_foopsi(dec_signal_nans, p=2, bas_nonneg=False, optimize_g=False)

#             _, sn_oasis = caim_deconv.estimate_parameters(dec_signal_nans, p=2)

#             oasis_deconv, oasis_spikes, bl_oasis, g_oasis, lam_oasis = caim_deconv.constrained_oasisAR2(dec_signal_nans.astype(np.float32), g_ar2, sn_oasis, optimize_b=True, b_nonneg=False, optimize_g=False,
#                                                                                       decimate=1, shift=ar_lags, window=None, tol=1e-9, max_iter=10, penalty=0.1, s_min=None, )

#             ax = axs[i]
#             ax.set_title(region)
#             ax.plot(dec_t, utils.z_score(dec_signal), alpha=0.6, label='raw')
#             ax.plot(dec_t, utils.z_score(ar1_deconv), alpha=0.6, label='simple')
#             ax.plot(dec_t, utils.z_score(foopsi_deconv), alpha=0.6, label='oasis default')
#             ax.plot(dec_t, utils.z_score(oasis_deconv), alpha=0.6, label='oasis shift 100')
#             ax.legend()


# %% Perform SeqNMF

import sys
sys.path.append(r"C:\Users\Tanner\repos\python\seqnmf\seqnmf")
from seqnmf.seqnmf import helpers
from seqnmf.seqnmf.seqnmf import seqnmf, seqnmf_old

pl_lpf = 1

deconvolve = True

signal_type = 'z_dff_iso_baseline_fband'
ignore_regions = [] #['PL'] #

sess_ids = {400: [119974]} #{274: [119776, 119993], 400: [119785, 120001], 402: [119839, 120009]} #
loc_db = bandit_loc_db

n_factors = 4
factor_len = 1 # in seconds
# 1e-5 too little, not sparse enough, 1e-3 too large, too sparse
penalty = 6e-5
h_ortho_penalty = 1e-4
max_iter = 5000
tol = 1e-5
dec = 4
n_plot = 4

results = {}

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)

    subj_regions = [r for r in subj_regions if r not in ignore_regions]

    n_regions = len(subj_regions)

    sess_data_mats = []
    event_times = []
    dec_ts = []

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        dt = fp_data['dec_info']['decimated_dt']

        trial_data = loc_db.get_behavior_data(sess_id)

        full_t = fp_data['time']
        _, dec_t = fp_utils.decimate(full_t, full_t, dec, dt=dt)

        trial_start_ts = fp_data['trial_start_ts'][:-1]
        cport_on_ts = trial_start_ts + trial_data['cport_on_time']
        cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
        #tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
        cue_ts = trial_start_ts + trial_data['response_cue_time']
        cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
        response_ts = trial_start_ts + trial_data['response_time']
        outcome_ts = trial_start_ts + trial_data['reward_time']
        reward_ts = outcome_ts[trial_data['reward'] > 0]
        unreward_ts = outcome_ts[trial_data['reward'] == 0]

        lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                      'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                      'Reward': reward_ts, 'Unreward': unreward_ts}

        event_times.append(lines_dict)
        dec_ts.append(dec_t)

        # build signal matrix
        signal_mat = np.empty([n_regions, len(dec_t)])

        signal_mods = {}
        for i, region in enumerate(subj_regions):
            signal = fp_data['processed_signals'][region][signal_type]

            if region == 'PL':
                signal = fp_utils.filter_signal(signal, pl_lpf, 1/dt, filter_type='lowpass')

            dec_signal, _ = fp_utils.decimate(signal, full_t, dec, dt=dt)
            # if deconvolve:
            #     tau = avg_taus.loc[(subj_id, region)]['median']
            #     dec_signal[1:] = dec_signal[1:] - np.exp(-(dt*dec)/tau)*dec_signal[:-1]

            # make each signal positive and have equal power
            offset = np.nanmin(dec_signal) - 0.01
            signal_mat[i,:] = dec_signal - offset

        # set nans to zero so they do not affect the pattern extraction but their timecourse stays the same
        nans = np.any(np.isnan(signal_mat), axis=0)
        signal_mat[:,nans] = 0

        # normalize the power across channels
        power = np.sqrt(np.mean(signal_mat**2, axis=1, keepdims=True))
        signal_mat = signal_mat / power

        sess_data_mats.append(signal_mat)

    # perform seqNMF across sessions
    max_sess_len = np.max([m.shape[1] for m in sess_data_mats])

    sess_data_mat = np.zeros([len(sess_ids[subj_id]), n_regions, max_sess_len], dtype=np.float32)

    for i, mat in enumerate(sess_data_mats):
        sess_data_mat[i,:,:mat.shape[1]] = mat

    # do seqNMF

    factor_t = np.arange(0, factor_len, step=dt*dec)

    W, H, power, loadings, costs = seqnmf(sess_data_mat, K=n_factors, L=len(factor_t), Lambda=penalty,
                                          max_iter=max_iter, tol=tol, lambda_OrthH=h_ortho_penalty)

    results[subj_id] = {'W': W, 'H': H, 'power': power, 'loadings': loadings, 'costs': costs}

    factor_recons = helpers.reconstruct_factors(W, H)
    total_recons = np.sum(factor_recons, axis=-1)

    # plot results

    for i, sess_id in enumerate(sess_ids[subj_id]):

        signal_mat = sess_data_mats[i]
        lines_dict = event_times[i]
        dec_t = dec_ts[i]
        fit_resid = sess_data_mat[i,...] - total_recons[i,...]

        fig, axs = plt.subplots(n_plot+2, 2, layout='constrained', figsize=[20,6*n_plot], width_ratios=[1,6], sharex='col')
        fig.suptitle('Subject {}, Session {}, Total Power Explained: {:.3f}'.format(subj_id, sess_id, power[i]))

        # plot full signal with behavioral events
        ax = axs[0,1]
        for j, region in enumerate(subj_regions):
            ax.plot(dec_t, signal_mat[j,:], label=region, alpha=0.5)

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                      transform=ax.get_xaxis_transform())

        ax.set_title('Region Signals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.legend()
        sig_handles, sig_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        # plot residuals
        ax = axs[1,1]
        for j, region in enumerate(subj_regions):
            ax.plot(dec_t, fit_resid[j,:len(dec_t)], label=region, alpha=0.5)

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j+n_regions), linestyles='dashed',
                      transform=ax.get_xaxis_transform())

        ax.set_title('Fit Residuals')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('dF/F')
        ax.legend()
        sig_handles, sig_labels = ax.get_legend_handles_labels()
        ax.legend().remove()

        for j in range(n_plot):

            # plot component
            ax = axs[j+2,0]
            for k, region in enumerate(subj_regions):
                ax.plot(factor_t, W[k,j,:], label=region, alpha=0.5)

            ax.set_title('seqNMF Factor {}'.format(j))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Signal')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)
            ax.legend()
            factor_handles, factor_labels = ax.get_legend_handles_labels()
            ax.legend().remove()

            # plot factor reconstruction with behavioral events
            ax = axs[j+2,1]

            # plot reconstruction
            for k, region in enumerate(subj_regions):
                ax.plot(dec_t, factor_recons[i,k,:len(dec_t),j], label=region, alpha=0.5)

            for k, (name, lines) in enumerate(lines_dict.items()):
                ax.vlines(lines, 0, 1, label=name, color='C{}'.format(k+n_regions), linestyles='dashed',
                          transform=ax.get_xaxis_transform())

            ax.set_title('seqNMF Factor {} Loading and Reconstruction. Power: {:.3f}'.format(j, loadings[i,j]))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Loading')
            ax.xaxis.set_tick_params(which='both', labelbottom=True)

        ax = axs[0,0]
        ax.set_axis_off()
        ax.legend(factor_handles, factor_labels)
        fig.legend(sig_handles, sig_labels, loc='outside right upper')
        plt.show()

# %% Investigate source extraction performance with and without filtering

sess_ids = {199: [116828, 117565], 237: [124638, 124973], 400: [119713, 119974],
            402: [119481, 119742], 424: [124644, 124772], 483: [124924, 124957]} #
loc_db = bandit_loc_db

region_lpf = {'PL': 4, 'NAc': 6, 'DMS': 8, 'DLS': 8, 'TS': 8}
dec = 2

for subj_id in sess_ids.keys():

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)

    for sess_id in sess_ids[subj_id]:

        fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                       baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
        fp_data = fp_data[subj_id][sess_id]
        t = fp_data['time']
        dt = fp_data['dec_info']['decimated_dt']

        n_rows = len(subj_regions)
        fig, axs = plt.subplots(n_rows, 1, figsize=(20,4*n_rows), layout='constrained', sharex=True)
        fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

        for i, region in enumerate(subj_regions):
            if not region in fp_data['processed_signals']:
                continue
            raw_signal = fp_data['processed_signals'][region][signal_type]

            if all(np.isnan(raw_signal)):
                continue

            lpf_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
            dec_raw_signal, dec_t = fp_utils.decimate(raw_signal, t, dec, dt=dt)
            dec_lpf_signal, dec_t = fp_utils.decimate(lpf_signal, t, dec, dt=dt)

            deconv_tau = avg_params_all.loc[(avg_params_all['subj_id'] == subj_id) & (avg_params_all['region'] == region), ('tau','mean')]
            deconv_params = fp_utils.get_ar_params_from_modes({'tau': deconv_tau, 'freq': 0}, dt*dec)

            raw_source = fp_utils.extract_ar_source(dec_raw_signal, deconv_params)
            lpf_source = fp_utils.extract_ar_source(dec_lpf_signal, deconv_params)

            ax = axs[i]
            ax.set_title(region)
            ax.plot(dec_t, utils.z_score(dec_raw_signal), label='Raw Signal', alpha=0.5)
            ax.plot(dec_t, utils.z_score(dec_lpf_signal), label='LPF Signal', alpha=0.5)
            ax.plot(dec_t, utils.z_score(raw_source), label='Raw Source', alpha=0.5)
            ax.plot(dec_t, utils.z_score(lpf_source), label='LPF Source', alpha=0.5)

            ax.legend()

        plt.show()


# %% Compute source extraction cross-correlations with and without filtering

recalculate = False
reprocess_sess_ids = []

signal_types = ['dff_iso_baseline_fband'] # 'z_dff_iso_baseline'
filt_types = ['filtered', 'raw']

max_lag = 2
region_lpf = {'PL': 4, 'NAc': 6, 'DMS': 8, 'DLS': 8, 'TS': 8}
dec = 2

# get dt information to calculate the number of timesteps in the xcorr results
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']
n_lags = int(np.round(utils.convert_to_multiple(max_lag, dt*dec)/(dt*dec)))*2 + 1

tasks = ['wm', 'bandit']

filename = 'wm_bandit_source_xcorr_data2'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        x_corrs = saved_data['x_corrs']
        x_corr_metadata = saved_data['metadata']
        x_corr_regions = x_corr_metadata['regions']
        corr_lags = x_corr_metadata['corr_lags']

        recalculate = (x_corr_metadata['max_lag'] != max_lag) or (x_corr_metadata['region_lpf'] != region_lpf)

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    x_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    x_corr_regions = {}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)
    reg_mapping = get_subj_region_mapping(subj_regions)
    x_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)

    if not subj_id in x_corrs:
        x_corrs[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in x_corrs[subj_id][task] and not sess_id in reprocess_sess_ids and list(x_corrs[subj_id][task][sess_id].keys()) == signal_types:
                continue

            if sess_id not in x_corrs[subj_id][task]:
                x_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']

            print('Calculating cross-correlation for subject {} session {}'.format(subj_id, sess_id))

            for signal_type in signal_types:
                if signal_type in x_corrs[subj_id][task][sess_id]:
                    continue
                else:
                    x_corrs[subj_id][task][sess_id][signal_type] = {f: np.full((n_regions, n_regions, n_lags), np.nan, dtype=float) for f in filt_types}

                # calculate signal sources
                sess_resids = {f: {} for f in filt_types}

                for region in subj_regions:
                    if not region in fp_data['processed_signals']:
                        continue
                    raw_signal = fp_data['processed_signals'][region][signal_type]

                    if all(np.isnan(raw_signal)):
                        continue

                    filt_signal = fp_utils.filter_signal(raw_signal, region_lpf[reg_mapping[region]], 1/dt)
                    dec_raw_signal, _ = fp_utils.decimate(raw_signal, t, dec, dt=dt)
                    dec_filt_signal, _ = fp_utils.decimate(filt_signal, t, dec, dt=dt)

                    deconv_tau = avg_params_all.loc[(avg_params_all['subj_id'] == subj_id) & (avg_params_all['region'] == region), ('tau','mean')]
                    deconv_params = fp_utils.get_ar_params_from_modes({'tau': deconv_tau, 'freq': 0}, dt*dec)

                    sess_resids['raw'][region] = fp_utils.extract_ar_source(dec_raw_signal, deconv_params)
                    sess_resids['filtered'][region] = fp_utils.extract_ar_source(dec_filt_signal, deconv_params)

                for f in filt_types:
                    for reg_i in range(n_regions):
                        if not subj_regions[reg_i] in sess_resids[f]:
                            continue

                        signal_i = sess_resids[f][subj_regions[reg_i]]

                        for reg_j in np.arange(reg_i, n_regions):
                            if not subj_regions[reg_j] in sess_resids[f]:
                                continue

                            signal_j = sess_resids[f][subj_regions[reg_j]]

                            start = time.perf_counter()

                            xcorr, corr_lags = fp_utils.correlate(signal_i, signal_j, dt*dec, neg_lag=max_lag, pos_lag=max_lag)

                            x_corrs[subj_id][task][sess_id][signal_type][f][reg_i, reg_j, :] = xcorr
                            if reg_i != reg_j:
                                x_corrs[subj_id][task][sess_id][signal_type][f][reg_j, reg_i, :] = np.flip(xcorr)


                with open(save_path, 'wb') as f:
                    pickle.dump({'x_corrs': x_corrs,
                                 'metadata': {'regions': x_corr_regions,
                                              'max_lag': max_lag,
                                              'corr_lags': corr_lags,
                                              'region_lpf': region_lpf}},
                                f)

# %% Plot source extraction cross-correlations

plot_lag = 1
t_sel = np.abs(corr_lags) <= plot_lag

plot_sep_tasks = False
plot_filt_types = ['filtered', 'raw']
plot_ind_subj = True
plot_comb_subj = True
ax_size = 4

plot_beh_names = {'wm': 'WM Task', 'bandit': 'Bandit Task', 'all': 'All Tasks'}

if plot_comb_subj:
    comb_xcorr = {t: {s: {f: [] for f in plot_filt_types} for s in signal_types} for t in tasks}
    comb_xcorr_weights = {t: {s: {f: [] for f in plot_filt_types} for s in signal_types} for t in tasks}

for subj_id in subj_ids:

    subj_regions = x_corr_regions[subj_id]
    sorted_regions = sort_subj_regions(subj_regions)
    sorted_region_idxs = [subj_regions.index(r) for r in sorted_regions]

    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]

    # stack across subjects
    if plot_comb_subj:
        for task in tasks:

            if len(x_corrs[subj_id][task]) == 0:
                continue

            for signal_type in signal_types:

                # stack by session across regions
                if plot_comb_subj and task != 'all':

                    for s in x_corrs[subj_id][task].keys():
                        for f in plot_filt_types:
                            sess_xcorr = x_corrs[subj_id][task][s][signal_type][f]

                            # get mapping of subject regions to all regions order
                            all_region_idx_mapping = fpah.get_region_idx_mapping(subj_regions, all_regions)
                            dup_counts = fpah.count_corr_pairs(subj_regions, all_regions)

                            if all([v == 1 for v in dup_counts.values()]):
                                # if no duplicates, can simply use indexing
                                sess_comb_xcorr = np.full((len(all_regions), len(all_regions), len(corr_lags)), np.nan)
                                sess_comb_xcorr[np.ix_(all_region_idx_mapping, all_region_idx_mapping, np.arange(len(corr_lags)))] = sess_xcorr

                                comb_xcorr[task][signal_type][f].append(sess_comb_xcorr)

                                weights = np.ones_like(sess_comb_xcorr)
                                weights[np.isnan(sess_comb_xcorr)] = np.nan
                                comb_xcorr_weights[task][signal_type][f].append(weights)
                            else:
                                # add each duplicate pair separately
                                for subj_i, map_i in enumerate(all_region_idx_mapping):
                                    for subj_j, map_j in enumerate(all_region_idx_mapping):
                                        # only do each cross-regional comparison once
                                        if subj_j < subj_i:
                                            continue
                                        # don't add bilateral cross-correlations of the same region as auto-correlations
                                        if map_i == map_j and subj_i != subj_j:
                                            continue

                                        sess_comb_xcorr = np.full((len(all_regions), len(all_regions), len(corr_lags)), np.nan)
                                        sess_comb_xcorr[map_i,map_j,:] = sess_xcorr[subj_i,subj_j,:]
                                        if map_i != map_j:
                                            sess_comb_xcorr[map_j,map_i,:] = sess_xcorr[subj_j,subj_i,:]

                                        comb_xcorr[task][signal_type][f].append(sess_comb_xcorr)

                                        dup_key = tuple(sorted([map_i,map_j]))
                                        weights = np.full_like(sess_comb_xcorr, 1/dup_counts[dup_key])
                                        weights[np.isnan(sess_comb_xcorr)] = np.nan
                                        comb_xcorr_weights[task][signal_type][f].append(weights)

    # plot individual subject averages
    if plot_ind_subj:
        subj_region_colors = {r: 'C{}'.format(i) for i, r in enumerate(subj_regions)}
        if plot_sep_tasks:
            for signal_type in signal_types:

                signal_title, _ = fpah.get_signal_type_labels(signal_type)

                fig, axs = plt.subplots(n_regions, n_regions, sharex=True, sharey=True, figsize=(ax_size*n_regions, ax_size*n_regions), layout='constrained')
                fig.suptitle('Regional Cross-correlations for Subject {}. (- first leading, + first lagging)\n{}'.format(subj_id, signal_title))

                for reg_i in range(n_regions):
                    sorted_i = sorted_region_idxs[reg_i]
                    for reg_j in range(n_regions):
                        sorted_j = sorted_region_idxs[reg_j]
                        ax = axs[reg_i, reg_j]

                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=ax)

                        for task in tasks:
                            for f in plot_filt_types:

                                sess_corrs = np.stack([x_corrs[subj_id][task][s][signal_type][f][sorted_i, sorted_j, :] for s in x_corrs[subj_id][task].keys()], axis=1)

                                plot_utils.plot_psth(corr_lags[t_sel], np.nanmean(sess_corrs, axis=1)[t_sel], utils.stderr(sess_corrs, axis=1)[t_sel], ax, plot_x0=False,
                                                     label='{}, {}'.format(plot_beh_names[task], f), color=subj_region_colors[sorted_regions[reg_j]])

                            ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'],
                                                                     sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))
                            ax.legend()

                            if reg_i == n_regions-1:
                                ax.set_xlabel('Time lag (s)')
                            if reg_j == 0:
                                ax.set_ylabel('Pearson r')
        else:
            for signal_type in signal_types:

                signal_title, _ = fpah.get_signal_type_labels(signal_type)

                fig, axs = plt.subplots(len(plot_filt_types), n_regions, sharex=True, sharey=True, figsize=(ax_size*n_regions, ax_size*len(plot_filt_types)), layout='constrained')
                fig.suptitle('Regional Cross-correlations for Subject {}. (- first leading, + first lagging)\n{}'.format(subj_id, signal_title))

                for reg_i in range(n_regions):
                    sorted_i = sorted_region_idxs[reg_i]
                    for reg_j in range(n_regions):
                        sorted_j = sorted_region_idxs[reg_j]

                        if reg_i == reg_j:
                            continue

                        for f_i, f in enumerate(plot_filt_types):
                            ax = axs[f_i, reg_i]

                            plot_utils.plot_dashlines(0, dir='h', ax=ax)
                            plot_utils.plot_dashlines(0, dir='v', ax=ax)

                            sess_corrs = np.hstack([np.stack([x_corrs[subj_id][t][s][signal_type][f][sorted_i, sorted_j, :] for s in x_corrs[subj_id][t].keys()], axis=1) for t in tasks if len(x_corrs[subj_id][t]) > 0])

                            plot_utils.plot_psth(corr_lags[t_sel], np.nanmean(sess_corrs, axis=1)[t_sel], utils.stderr(sess_corrs, axis=1)[t_sel], ax, plot_x0=False,
                                                 label='{} ({})'.format(sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']), color=subj_region_colors[sorted_regions[reg_j]])

                            ax.set_title('{} ({}), {}'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'], f))
                            ax.legend()

                            if f_i == len(plot_filt_types)-1:
                                ax.set_xlabel('Time lag (s)')
                            if reg_i == 0:
                                ax.set_ylabel('Pearson r')


# plot combined subject averages
if plot_comb_subj:
    # stack all matrices together
    comb_xcorr = {t: {s: {f: np.stack(comb_xcorr[t][s][f], axis=-1) for f in plot_filt_types} for s in signal_types} for t in tasks}
    comb_xcorr_weights = {t: {s: {f: np.stack(comb_xcorr_weights[t][s][f], axis=-1) for f in plot_filt_types} for s in signal_types} for t in tasks}

    if plot_sep_tasks:
        for signal_type in signal_types:

            signal_title, _ = fpah.get_signal_type_labels(signal_type)

            fig, axs = plt.subplots(len(all_regions), len(all_regions), sharex=True, sharey=True,
                                    figsize=(ax_size*len(all_regions), ax_size*len(all_regions)), layout='constrained')
            fig.suptitle('Regional Cross-correlation for All Subjects. (- first leading, + first lagging)\n{}'.format(signal_title))

            for reg_i in range(len(all_regions)):
                for reg_j in range(len(all_regions)):

                    ax = axs[reg_i, reg_j]

                    plot_utils.plot_dashlines(0, dir='h', ax=ax)
                    plot_utils.plot_dashlines(0, dir='v', ax=ax)

                    for task in tasks:
                        for f in plot_filt_types:
                            comb_corrs = comb_xcorr[task][signal_type][f][reg_i, reg_j, :, :]
                            comb_weights = comb_xcorr_weights[task][signal_type][f][reg_i, reg_j, :, :]

                            plot_utils.plot_psth(corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel],
                                                 utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=False,
                                                 label='{}, {}'.format(plot_beh_names[task], f), color=region_colors[all_regions[reg_j]])

                    ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))
                    ax.legend()

                    if reg_i == len(all_regions)-1:
                        ax.set_xlabel('Time lag (s)')
                    if reg_j == 0:
                        ax.set_ylabel('Pearson r')
    else:
        for signal_type in signal_types:

            signal_title, _ = fpah.get_signal_type_labels(signal_type)

            fig, axs = plt.subplots(len(plot_filt_types), len(all_regions), sharex=True, sharey=True,
                                    figsize=(ax_size*len(all_regions), ax_size*len(plot_filt_types)), layout='constrained')
            fig.suptitle('Regional Cross-correlation for All Subjects. (- first leading, + first lagging)\n{}'.format(signal_title))

            for reg_i in range(len(all_regions)):
                for reg_j in range(len(all_regions)):

                    if reg_i == reg_j:
                        continue

                    for f_i, f in enumerate(plot_filt_types):
                        ax = axs[f_i, reg_i]

                        plot_utils.plot_dashlines(0, dir='h', ax=ax)
                        plot_utils.plot_dashlines(0, dir='v', ax=ax)

                        comb_corrs = np.hstack([comb_xcorr[t][signal_type][f][reg_i, reg_j, :, :] for t in tasks])
                        comb_weights = np.hstack([comb_xcorr_weights[t][signal_type][f][reg_i, reg_j, :, :] for t in tasks])

                        plot_utils.plot_psth(corr_lags[t_sel], utils.weighted_mean(comb_corrs, comb_weights, axis=1)[t_sel],
                                             utils.weighted_se(comb_corrs, comb_weights, axis=1)[t_sel], ax, plot_x0=False,
                                             label=all_regions[reg_j], color=region_colors[all_regions[reg_j]])

                        ax.set_title('{}, {}'.format(all_regions[reg_i], f))
                        ax.legend()
                    # handles, labels = ax.get_legend_handles_labels()
                    # ax.legend().remove()

                    if f_i == len(plot_filt_types)-1:
                        ax.set_xlabel('Time lag (s)')
                    if reg_i == 0:
                        ax.set_ylabel('Pearson r')

            # fig.legend(handles, labels, loc='outside right center')



# %% compute correlations by frequency band

freq_vals = [0] + list(np.logspace(np.log10(0.01), np.log10(10), 20))
freq_vals = np.round(freq_vals, 3)
freq_bands = list(zip(freq_vals[:-1], freq_vals[1:]))
# freq_bands = [[0,0.3], [0.3,10]]
#freq_bands = [[0,0.03], [0.03,0.3], [0.3,2], [2,10]]
#freq_bands = [[0,0.02], [0.02,0.1], [0.1,0.3], [0.3,1], [1,1.8], [1.8,4], [4,10]]
#freq_bands = [[0,0.06], [0.06,0.2], [0.2,0.6], [0.6,1], [1,2], [2,6], [6,10]]

recalculate = False
reprocess_sess_ids = []

signal_types = ['z_dff_iso_baseline_fband'] # 'z_dff_iso_baseline'

# get dt information
tmp_subj = list(wm_sess_ids.keys())[0]
tmp_sess = wm_sess_ids[tmp_subj][0]
dt = wm_loc_db.get_sess_fp_data([tmp_sess])['fp_data'][tmp_subj][tmp_sess]['dec_info']['decimated_dt']

tasks = ['wm', 'bandit']

filename = 'wm_bandit_fband_xcorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        fband_corrs = saved_data['fband_corrs']
        fband_corr_metadata = saved_data['metadata']
        fband_corr_regions = fband_corr_metadata['regions']

        recalculate = fband_corr_metadata['freq_bands'] != freq_bands

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    fband_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    fband_corr_regions = {}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)

    fband_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)

    if not subj_id in fband_corrs:
        fband_corrs[subj_id] = {task: {} for task in tasks}

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in fband_corrs[subj_id][task] and not sess_id in reprocess_sess_ids and list(fband_corrs[subj_id][task][sess_id].keys()) == signal_types:
                continue

            if sess_id not in fband_corrs[subj_id][task]:
                fband_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']
            trial_start_ts = fp_data['trial_start_ts']

            # remove start and end of signals to avoid edge effects of filtering
            outer_sel = np.full_like(t, False, dtype=bool)
            outer_sel[t < trial_start_ts[0]] = True
            outer_sel[t > trial_start_ts[-1]] = True

            sess_engaged_t_sel = engaged_t_sel[subj_id][task][sess_id]

            print('Calculating frequency band cross-correlation for subject {} session {}:'.format(subj_id, sess_id))

            for signal_type in signal_types:
                if signal_type in fband_corrs[subj_id][task][sess_id]:
                    continue
                else:
                    fband_corrs[subj_id][task][sess_id][signal_type] = {e: np.full((n_regions, n_regions, len(freq_bands)+1), np.nan, dtype=float) for e in engage_states}

                for reg_i in range(n_regions):
                    if not subj_regions[reg_i] in fp_data['processed_signals']:
                        continue
                    signal_i = fp_data['processed_signals'][subj_regions[reg_i]][signal_type]
                    signal_i_fbands = fp_utils.decompose_signal_fbands(signal_i, freq_bands, 1/dt)

                    signal_i[outer_sel] = np.nan
                    signal_i_fbands[:, outer_sel] = np.nan

                    for reg_j in np.arange(reg_i+1, n_regions):
                        if not subj_regions[reg_j] in fp_data['processed_signals']:
                            continue
                        signal_j = fp_data['processed_signals'][subj_regions[reg_j]][signal_type]
                        signal_j_fbands = fp_utils.decompose_signal_fbands(signal_j, freq_bands, 1/dt)

                        signal_j[outer_sel] = np.nan
                        signal_j_fbands[:, outer_sel] = np.nan

                        start = time.perf_counter()

                        for e in engage_states:

                            if e == 'engaged':
                                t_sel = sess_engaged_t_sel
                            elif e == 'disengaged':
                                t_sel = ~sess_engaged_t_sel
                            else:
                                t_sel = None

                            tot_xcorr, _ = fp_utils.correlate(signal_i, signal_j, dt, neg_lag=0, pos_lag=0, t_sel=t_sel, z_score_t_sel=True)

                            fband_corrs[subj_id][task][sess_id][signal_type][e][reg_i, reg_j, -1] = tot_xcorr[0]
                            fband_corrs[subj_id][task][sess_id][signal_type][e][reg_j, reg_i, -1] = tot_xcorr[0]

                            for k in range(len(freq_bands)):
                                band_xcorr, _ = fp_utils.correlate(signal_i_fbands[k,:], signal_j_fbands[k,:], dt, neg_lag=0, pos_lag=0, t_sel=t_sel, z_score_t_sel=True)

                                fband_corrs[subj_id][task][sess_id][signal_type][e][reg_i, reg_j, k] = band_xcorr[0]
                                fband_corrs[subj_id][task][sess_id][signal_type][e][reg_j, reg_i, k] = band_xcorr[0]

                        print('  Between {} and {} in {:.1f} s'.format(
                            subj_regions[reg_i], subj_regions[reg_j], time.perf_counter()-start))

                with open(save_path, 'wb') as f:
                    pickle.dump({'fband_corrs': fband_corrs,
                                 'metadata': {'regions': fband_corr_regions,
                                              'freq_bands': freq_bands}},
                                f)

# %% Plot frequency band correlations

plot_sep_tasks = True
plot_sep_states = False
plot_ind_subj = False
plot_comb_subj = True
ax_height = 3
ax_width = 5
x_label_rot = 90

plot_tasks = []
if plot_sep_tasks:
    plot_tasks.extend(tasks.copy())

plot_tasks.append('all')

if plot_sep_states:
    plot_states = engage_states.copy()
else:
    plot_states = ['all']

plot_beh_names = {'wm': 'WM', 'bandit': 'Bandit', 'all': 'All'}
engage_labels = {'engaged': 'engaged', 'disengaged': 'disengaged', 'all': 'all'}
band_labels = ['{}-{}'.format(b[0], b[1]) for b in freq_bands]

task_names = [plot_beh_names[t] for t in plot_tasks]
engage_names = [engage_labels[e] for e in plot_states]

if plot_comb_subj:
    comb_fband_corr = {t: {s: {e: [] for e in plot_states} for s in signal_types} for t in tasks}
    comb_fband_weights = {t: {s: {e: [] for e in plot_states} for s in signal_types} for t in tasks}

for subj_id in subj_ids:
    subj_regions = fband_corr_regions[subj_id]
    sorted_regions = sort_subj_regions(subj_regions)
    sorted_region_idxs = [subj_regions.index(r) for r in sorted_regions]

    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]

    # stack across subjects
    if plot_comb_subj:
        for task in tasks:

            if len(fband_corrs[subj_id][task]) == 0:
                continue

            for signal_type in signal_types:

                # stack by session across regions
                for s in fband_corrs[subj_id][task].keys():
                    for e in plot_states:

                        sess_fband_corr = fband_corrs[subj_id][task][s][signal_type][e]

                        # get mapping of subject regions to all regions order
                        all_region_idx_mapping = fpah.get_region_idx_mapping(subj_regions, all_regions)
                        dup_counts = fpah.count_corr_pairs(subj_regions, all_regions)

                        if all([v == 1 for v in dup_counts.values()]):
                            # if no duplicates, can simply use indexing
                            sess_comb_fband_corr = np.full((len(all_regions), len(all_regions), len(freq_bands)+1), np.nan, dtype=float)
                            sess_comb_fband_corr[np.ix_(all_region_idx_mapping, all_region_idx_mapping, np.arange(len(freq_bands)+1))] = sess_fband_corr

                            comb_fband_corr[task][signal_type][e].append(sess_comb_fband_corr)

                            weights = np.ones_like(sess_comb_fband_corr)
                            weights[np.isnan(sess_comb_fband_corr)] = np.nan
                            comb_fband_weights[task][signal_type][e].append(weights)
                        else:
                            # add each duplicate pair separately
                            for subj_i, map_i in enumerate(all_region_idx_mapping):
                                for subj_j, map_j in enumerate(all_region_idx_mapping):
                                    # only do each cross-regional comparison once and don't do autocorrelations
                                    if subj_j <= subj_i:
                                        continue
                                    # don't add bilateral cross-correlations of the same region as auto-correlations
                                    if map_i == map_j and subj_i != subj_j:
                                        continue

                                    sess_comb_fband_corr = np.full((len(all_regions), len(all_regions), len(freq_bands)+1), np.nan, dtype=float)
                                    sess_comb_fband_corr[map_i,map_j,:] = sess_fband_corr[subj_i,subj_j,:]
                                    if map_i != map_j:
                                        sess_comb_fband_corr[map_j,map_i,:] = sess_fband_corr[subj_j,subj_i,:]

                                    comb_fband_corr[task][signal_type][e].append(sess_comb_fband_corr)

                                    dup_key = tuple(sorted([map_i,map_j]))
                                    weights = np.full_like(sess_comb_fband_corr, 1/dup_counts[dup_key])
                                    weights[np.isnan(sess_comb_fband_corr)] = np.nan
                                    comb_fband_weights[task][signal_type][e].append(weights)

    if plot_ind_subj:
        if plot_sep_states:
            for task in plot_tasks:
                for signal_type in signal_types:

                    signal_title, _ = fpah.get_signal_type_labels(signal_type)

                    fig, axs = plt.subplots(n_regions-1, n_regions-1, figsize=(ax_width*(n_regions-1)+1, ax_height*(n_regions-1)), layout='constrained')
                    fig.suptitle('Regional Frequency Band Correlations for Subject {} in {}.\n{}'.format(subj_id, plot_beh_names[task], signal_title))

                    if task == 'all':
                        stacked_subj_corrs = {e: np.concatenate([np.stack([fband_corrs[subj_id][t][s][signal_type][e] for s in fband_corrs[subj_id][t].keys()], axis=-1) for t in tasks], axis=3) for e in plot_states}
                    else:
                        stacked_subj_corrs = {e: np.stack([fband_corrs[subj_id][task][s][signal_type][e] for s in fband_corrs[subj_id][task].keys()], axis=-1) for e in plot_states}

                    subj_corr_avg = {e: np.nanmean(stacked_subj_corrs[e], axis=3) for e in plot_states}
                    subj_corr_err = {e: utils.stderr(stacked_subj_corrs[e], axis=3) for e in plot_states}

                    colors = ['C{}'.format(i) for i in range(len(plot_states))]

                    for reg_i in range(n_regions-1):
                        sorted_i = sorted_region_idxs[reg_i]
                        for reg_j in np.arange(1,n_regions):
                            sorted_j = sorted_region_idxs[reg_j]

                            ax = axs[reg_i, reg_j-1]

                            if reg_j <= reg_i:
                                ax.axis('off')
                                continue

                            fband_vals = [subj_corr_avg[e][sorted_i, sorted_j, :-1] for e in plot_states]
                            fband_err = [subj_corr_err[e][sorted_i, sorted_j, :-1] for e in plot_states]

                            tot_corr_vals = [subj_corr_avg[e][sorted_i, sorted_j, -1] for e in plot_states]
                            tot_corr_err = [subj_corr_err[e][sorted_i, sorted_j, -1] for e in plot_states]

                            plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=engage_names, x_labels=band_labels,
                                                        orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)

                            for i, e in enumerate(plot_states):
                                plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax,
                                                             color=colors[i], label='{} Avg'.format(engage_labels[e]), linestyle='--')

                            ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'],
                                                                     sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))

                            ax.set_xlabel('Frequency Band')
                            ax.set_ylabel('Pearson r')

                            ax.legend()
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend().remove()

                    fig.legend(handles, labels, loc='outside right center')
        else:
            for signal_type in signal_types:

                signal_title, _ = fpah.get_signal_type_labels(signal_type)

                fig, axs = plt.subplots(n_regions-1, n_regions-1, figsize=(ax_width*(n_regions-1)+1, ax_height*(n_regions-1)), layout='constrained')
                fig.suptitle('Regional Frequency Band Correlations for Subject {}.\n{}'.format(subj_id, signal_title))

                stacked_subj_corrs = {t: np.stack([fband_corrs[subj_id][t][s][signal_type]['all'] for s in fband_corrs[subj_id][t].keys()], axis=-1) for t in tasks}
                stacked_subj_corrs['all'] = np.concatenate([stacked_subj_corrs[t] for t in tasks], axis=3)

                subj_corr_avg = {t: np.nanmean(stacked_subj_corrs[t], axis=3) for t in plot_tasks}
                subj_corr_err = {t: utils.stderr(stacked_subj_corrs[t], axis=3) for t in plot_tasks}

                colors = ['C{}'.format(i) for i in range(len(plot_tasks))]

                for reg_i in range(n_regions-1):
                    sorted_i = sorted_region_idxs[reg_i]
                    for reg_j in np.arange(1,n_regions):
                        sorted_j = sorted_region_idxs[reg_j]

                        ax = axs[reg_i, reg_j-1]

                        if reg_j <= reg_i:
                            ax.axis('off')
                            continue

                        fband_vals = [subj_corr_avg[t][sorted_i, sorted_j, :-1] for t in plot_tasks]
                        fband_err = [subj_corr_err[t][sorted_i, sorted_j, :-1] for t in plot_tasks]

                        tot_corr_vals = [subj_corr_avg[t][sorted_i, sorted_j, -1] for t in plot_tasks]
                        tot_corr_err = [subj_corr_err[t][sorted_i, sorted_j, -1] for t in plot_tasks]

                        plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=task_names, x_labels=band_labels,
                                                    orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)

                        for i, t in enumerate(plot_tasks):
                            plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax,
                                                         color=colors[i], label='{} Avg'.format(plot_beh_names[t]), linestyle='--')

                        ax.set_title('{} ({}) vs {} ({})'.format(sorted_regions[reg_i], implant_side_info[sorted_regions[reg_i]]['side'],
                                                                 sorted_regions[reg_j], implant_side_info[sorted_regions[reg_j]]['side']))

                        ax.set_xlabel('Frequency Band')
                        ax.set_ylabel('Pearson r')

                        ax.legend()
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend().remove()

                fig.legend(handles, labels, loc='outside right center')


# plot combined subject averages
if plot_comb_subj:
    # stack all matrices together
    comb_fband_corr = {t: {s: {e: np.stack(comb_fband_corr[t][s][e], axis=-1) for e in plot_states} for s in signal_types} for t in tasks}
    comb_fband_weights = {t: {s: {e: np.stack(comb_fband_weights[t][s][e], axis=-1) for e in plot_states} for s in signal_types} for t in tasks}

    if plot_sep_states:
        for task in plot_tasks:
            for signal_type in signal_types:
                signal_title, _ = fpah.get_signal_type_labels(signal_type)

                fig, axs = plt.subplots(len(all_regions)-1, len(all_regions)-1,
                                        figsize=(ax_width*(len(all_regions)-1)+1, ax_height*(len(all_regions)-1)), layout='constrained')
                fig.suptitle('Regional Frequency Band Correlations for All Subjects in {}.\n{}'.format(plot_beh_names[task], signal_title))

                if task == 'all':
                    stacked_subj_corrs = {e: np.concatenate([comb_fband_corr[t][signal_type][e] for t in tasks], axis=3) for e in plot_states}
                    stacked_subj_weights = {e: np.concatenate([comb_fband_weights[t][signal_type][e] for t in tasks], axis=3) for e in plot_states}
                else:
                    stacked_subj_corrs = {e: comb_fband_corr[task][signal_type][e] for e in plot_states}
                    stacked_subj_weights = {e: comb_fband_weights[task][signal_type][e] for e in plot_states}

                subj_corr_avg = {e: utils.weighted_mean(stacked_subj_corrs[e], stacked_subj_weights[e], axis=3) for e in plot_states}
                subj_corr_err = {e: utils.weighted_se(stacked_subj_corrs[e], stacked_subj_weights[e], axis=3) for e in plot_states}

                colors = ['C{}'.format(i) for i in range(len(plot_states))]

                for reg_i in range(len(all_regions)-1):
                    for reg_j in np.arange(1, len(all_regions)):

                        ax = axs[reg_i, reg_j-1]

                        if reg_j <= reg_i:
                            ax.axis('off')
                            continue

                        fband_vals = [subj_corr_avg[e][reg_i, reg_j, :-1] for e in plot_states]
                        fband_err = [subj_corr_err[e][reg_i, reg_j, :-1] for e in plot_states]

                        tot_corr_vals = [subj_corr_avg[e][reg_i, reg_j, -1] for e in plot_states]
                        tot_corr_err = [subj_corr_err[e][reg_i, reg_j, -1] for e in plot_states]

                        plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=engage_names, x_labels=band_labels,
                                                    orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)

                        for i, e in enumerate(plot_states):
                            plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax,
                                                         color=colors[i], label='{} Avg'.format(engage_labels[e]), linestyle='--')

                        ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))

                        ax.set_xlabel('Frequency Band')
                        ax.set_ylabel('Pearson r')

                        ax.legend()
                        handles, labels = ax.get_legend_handles_labels()
                        ax.legend().remove()

                fig.legend(handles, labels, loc='outside right center')

    else:
        for signal_type in signal_types:
            signal_title, _ = fpah.get_signal_type_labels(signal_type)

            fig, axs = plt.subplots(len(all_regions)-1, len(all_regions)-1,
                                    figsize=(ax_width*(len(all_regions)-1)+1, ax_height*(len(all_regions)-1)), layout='constrained')
            fig.suptitle('Regional Frequency Band Correlations for All Subjects.\n{}'.format(signal_title))

            stacked_subj_corrs = {t: comb_fband_corr[t][signal_type]['all'] for t in tasks}
            stacked_subj_corrs['all'] = np.concatenate([stacked_subj_corrs[t] for t in tasks], axis=3)

            stacked_subj_weights = {t: comb_fband_weights[t][signal_type]['all'] for t in tasks}
            stacked_subj_weights['all'] = np.concatenate([stacked_subj_weights[t] for t in tasks], axis=3)

            subj_corr_avg = {t: utils.weighted_mean(stacked_subj_corrs[t], stacked_subj_weights[t], axis=3) for t in plot_tasks}
            subj_corr_err = {t: utils.weighted_se(stacked_subj_corrs[t], stacked_subj_weights[t], axis=3) for t in plot_tasks}

            colors = ['C{}'.format(i) for i in range(len(plot_tasks))]

            for reg_i in range(len(all_regions)-1):
                for reg_j in np.arange(1, len(all_regions)):

                    ax = axs[reg_i, reg_j-1]

                    if reg_j <= reg_i:
                        ax.axis('off')
                        continue

                    fband_vals = [subj_corr_avg[t][reg_i, reg_j, :-1] for t in plot_tasks]
                    fband_err = [subj_corr_err[t][reg_i, reg_j, :-1] for t in plot_tasks]

                    tot_corr_vals = [subj_corr_avg[t][reg_i, reg_j, -1] for t in plot_tasks]
                    tot_corr_err = [subj_corr_err[t][reg_i, reg_j, -1] for t in plot_tasks]

                    plot_utils.plot_stacked_bar(fband_vals, err=fband_err, value_labels=task_names, x_labels=band_labels,
                                                orientation='h', ax=ax, x_label_rot=x_label_rot, colors=colors)

                    for i, t in enumerate(plot_tasks):
                        plot_utils.plot_shaded_error(np.arange(len(band_labels)), tot_corr_vals[i], y_err=tot_corr_err[i], ax=ax,
                                                     color=colors[i], label='{} Avg'.format(plot_beh_names[t]), linestyle='--')

                    ax.set_title('{} vs {}'.format(all_regions[reg_i], all_regions[reg_j]))

                    ax.set_xlabel('Frequency Band')
                    ax.set_ylabel('Pearson r')

                    ax.legend()
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()

            fig.legend(handles, labels, loc='outside right center')


    # plot frequency band correlations between regions on the same axes
    for signal_type in signal_types:

        signal_title, _ = fpah.get_signal_type_labels(signal_type)

        stacked_subj_corrs = np.concatenate([comb_fband_corr[t][signal_type]['all'] for t in tasks], axis=3)
        stacked_subj_weights = np.concatenate([comb_fband_weights[t][signal_type]['all'] for t in tasks], axis=3)

        subj_corr_avg = utils.weighted_mean(stacked_subj_corrs, stacked_subj_weights, axis=3)
        subj_corr_err = utils.weighted_se(stacked_subj_corrs, stacked_subj_weights, axis=3)

        colors = ['C{}'.format(i) for i in range(len(plot_tasks))]

        fig, axs = plt.subplots(1, len(all_regions), sharex=True, sharey=True,
                                figsize=(ax_width*len(all_regions), ax_height+1), layout='constrained')
        fig.suptitle('Regional Frequency Band Correlations for All Subjects.\n{}'.format(signal_title))

        x_labels = band_labels + ['all']

        for reg_i in range(len(all_regions)):
            ax = axs[reg_i]
            ax.set_title(all_regions[reg_i])
            fband_vals = []
            fband_err = []
            region_labels = []
            plot_colors = []

            for reg_j in range(len(all_regions)):

                if reg_j == reg_i:
                    continue

                fband_vals.append(subj_corr_avg[reg_i, reg_j, :])
                fband_err.append(subj_corr_err[reg_i, reg_j, :])
                region_labels.append(all_regions[reg_j])
                plot_colors.append(region_colors[all_regions[reg_j]])

            plot_utils.plot_stacked_bar(fband_vals, err=fband_err, x_labels=x_labels, #value_labels=region_labels,
                                        orientation='h', ax=ax, x_label_rot=45, colors=plot_colors)

            ax.grid(axis='y')
            ax.set_xlabel('Frequency Band')
            if reg_i == 0:
                ax.set_ylabel('Pearson r')

        handles = []
        labels = []
        for reg in all_regions:
            handles.append(mpatches.Patch(color=region_colors[reg]))
            labels.append(reg)

        fig.legend(handles, labels, loc='outside right center')


# %% compute correlations over time
t_width = 0.5
recalculate = False

filename = 'wm_bandit_tcorr_data'
save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path) and not recalculate:
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        t_corrs = saved_data['t_corrs']
        t_corr_metadata = saved_data['metadata']
        t_corr_regions = t_corr_metadata['regions']

        recalculate = t_corr_metadata['t_width'] != t_width

elif not path.exists(save_path):
    recalculate = True

if recalculate:
    t_corrs = {subjid: {task: {} for task in tasks} for subjid in subj_ids}
    t_corr_regions = {subjid: [] for subjid in subj_ids}

for subj_id in subj_ids:

    subj_regions = get_subj_regions(subj_id)

    if not subj_id in t_corrs:
        t_corrs[subj_id] = {task: {} for task in tasks}

    t_corr_regions[subj_id] = subj_regions
    n_regions = len(subj_regions)

    for task in tasks:
        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            if sess_id in t_corrs[subj_id][task] and not sess_id in reprocess_sess_ids:
                continue
            else:
                t_corrs[subj_id][task][sess_id] = {}

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']

            t_corrs[subj_id][task][sess_id]['t'] = t

            print('Calculating correlation over time for subject {} session {}'.format(subj_id, sess_id))

            for reg_i in range(n_regions):
                signal_i = fp_data['processed_signals'][subj_regions[reg_i]][signal_type]

                for reg_j in np.arange(reg_i+1, n_regions):
                    signal_j = fp_data['processed_signals'][subj_regions[reg_j]][signal_type]

                    start = time.perf_counter()

                    t_corr = fp_utils.correlate_over_time(signal_i, signal_j, dt, t_width=t_width)

                    t_corrs[subj_id][task][sess_id]['{} & {}'.format(subj_regions[reg_i], subj_regions[reg_j])] = t_corr

            with open(save_path, 'wb') as f:
                pickle.dump({'t_corrs': t_corrs,
                             'metadata': {'regions': t_corr_regions,
                                         't_width': t_width}},
                            f)

# %% Investigate engaged vs disengaged

signal_types = ['z_dff_iso_baseline_fband'] #, 'z_dff_iso_fband']
plot_tasks = ['wm']
dec=2
baseline_iso_fit = True
tilt_t = True

for subj_id in subj_ids:

    subj_regions = x_corr_regions[subj_id]
    n_regions = len(subj_regions)

    for task in plot_tasks:
        if task == 'wm':
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        sess_ids = list(x_corrs[subj_id][task].keys())

        for sess_id in sess_ids:

            fp_data, _ = fpah.load_fp_data(loc_db, {subj_id: [sess_id]}, traditional_fit=traditional_fit, baseline_iso_fit=baseline_iso_fit,
                                           baseline_band_iso_fit=baseline_band_iso_fit, band_iso_fit=band_iso_fit, tilt_t=tilt_t)
            fp_data = fp_data[subj_id][sess_id]
            t = fp_data['time']

            sess_engaged_t_sel = engaged_t_sel[subj_id][task][sess_id]

            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_start_ts = fp_data['trial_start_ts'][:-1]

            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']

            response = ~np.isnan(trial_data['response_time']).to_numpy()
            reward = (trial_data['rewarded'] == True).to_numpy()
            unreward = (trial_data['rewarded'] == False).to_numpy() & response

            lines_dict = {'Rewarded': cpoke_out_ts[reward], 'Unrewarded': cpoke_out_ts[unreward]}

            if task == 'wm':
                bail = (trial_data['bail'] == True).to_numpy()
                lines_dict['Bail'] = cpoke_out_ts[bail]

            for signal_type in signal_types:
                fig, axs = plt.subplots(len(subj_regions), 1, sharex=True, layout='constrained', figsize=(12,3*len(subj_regions)))

                fig.suptitle('Subject {}, Session {}'.format(subj_id, sess_id))

                for i, region in enumerate(subj_regions):
                    ax = axs[i]

                    if not region in fp_data['processed_signals']:
                        continue
                    signal = fp_data['processed_signals'][region][signal_type]

                    ax.plot(t[::dec], signal[::dec])
                    ax.fill_between(t, 0, 1, where=sess_engaged_t_sel,
                                    color='grey', alpha=0.4, transform=ax.get_xaxis_transform())

                    for j, (name, lines) in enumerate(lines_dict.items()):
                        ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j), linestyles='dashed',
                                  transform=ax.get_xaxis_transform())

                    ax.set_title(region)
                    ax.legend()
                    handles, labels = ax.get_legend_handles_labels()
                    ax.legend().remove()

                fig.legend(handles, labels, loc='outside right upper')

            plt.show()


# %% Plot helper

def plot_signal_details(signal_mat, t, title, col_names, lines_dict={}, dec=10):

    t = t[::dec].copy()

    fig, axs = plt.subplots(len(col_names), 1, layout='constrained', figsize=[18,6*len(col_names)], sharex=True)
    if len(col_names) == 1:
        axs = [axs]

    fig.suptitle(title)

    for i, col_name in enumerate(col_names):

        ax = axs[i]
        ax.plot(t, signal_mat[::dec,i], label='_')

        for j, (name, lines) in enumerate(lines_dict.items()):
            ax.vlines(lines, 0, 1, label=name, color='C{}'.format(j), linestyles='dashed',
                      transform=ax.get_xaxis_transform())

        ax.set_title(col_name)
        ax.set_xlabel('Time (s)')
        ax.xaxis.set_tick_params(which='both', labelbottom=True)
        ax.legend()

# %% Plot correlations over time with behavioral events

plot_tasks = tasks.copy()

beh_names = {'wm': 'WM Task', 'bandit': 'Bandit Task'}

for subj_id in subj_ids:

    subj_regions = t_corr_regions[subj_id]
    n_regions = len(subj_regions)
    implant_side_info = implant_info[subj_id]

    for task in tasks:

        if task == 'wm':
            sess_ids = wm_sess_ids
            loc_db = wm_loc_db
            sess_data = wm_sess_data
        else:
            sess_ids = bandit_sess_ids
            loc_db = bandit_loc_db
            sess_data = bandit_sess_data

        if not subj_id in sess_ids:
            continue

        sess_ids = [s for s in sess_ids[subj_id] if s not in fpah.__sess_ignore]

        for sess_id in sess_ids:

            trial_data = sess_data[sess_data['sessid'] == sess_id]

            trial_start_ts = loc_db.get_sess_fp_data([sess_id])['fp_data'][subj_id][sess_id]['trial_start_ts'][:-1]

            cport_on_ts = trial_start_ts + trial_data['cport_on_time']
            cpoke_in_ts = trial_start_ts + trial_data['cpoke_in_time']
            cue_ts = trial_start_ts + trial_data['response_cue_time']
            cpoke_out_ts = trial_start_ts + trial_data['cpoke_out_time']
            response_ts = trial_start_ts + trial_data['response_time']
            outcome_ts = trial_start_ts + trial_data['reward_time']
            reward_ts = outcome_ts[trial_data['reward'] > 0]
            unreward_ts = outcome_ts[trial_data['reward'] == 0]

            lines_dict = {'Cport On': cport_on_ts, 'Cpoke In': cpoke_in_ts, #'Tone': tone_ts,
                          'Resp Cue': cue_ts, 'Cpoke Out': cpoke_out_ts, 'Response': response_ts,
                          'Reward': reward_ts, 'Unreward': unreward_ts}

            if task == 'wm':
                tone_ts = trial_start_ts + trial_data['abs_tone_start_times']
                lines_dict.update({'Tone Start': tone_ts})

            corr_keys = list(t_corrs[subj_id][task][sess_id].keys())
            corr_keys = [k for k in corr_keys if k != 't']

            t = t_corrs[subj_id][task][sess_id]['t']
            corr_mat = np.stack([t_corrs[subj_id][task][sess_id][k] for k in corr_keys], axis=1)

            plot_signal_details(corr_mat, t, 'Regional correlations over time for Subject {} in {}'.format(subj_id, beh_names[task]),
                                corr_keys, lines_dict=lines_dict)



