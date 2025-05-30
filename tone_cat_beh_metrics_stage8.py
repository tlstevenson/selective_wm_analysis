# -*- coding: utf-8 -*-
"""
Script to investigate performance on the tone categorization task stage 8 - multi tone

@author: tanner stevenson
"""

import init

import pyutils.utils as utils
from sys_neuro_tools import plot_utils
import hankslab_db.tonecatdelayresp_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import re
import seaborn as sb
import copy

plot_timing = False
exclude_instruct_from_hitrate = True
# plot_summary = True

# %% LOAD DATA

stage = 8
n_back = 15
active_subjects_only = True
reload = False

subject_info = db_access.get_active_subj_stage('ToneCatDelayResp')
if active_subjects_only:
    subject_info = subject_info[subject_info['stage'] == stage]
else:
    subject_info = subject_info[subject_info['stage'] >= stage]

subj_ids = subject_info['subjid']
#subj_ids = subj_ids[subj_ids != 187]
#subj_ids = [193,192]

# get session ids
sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_num=stage, protocol='ToneCatDelayResp')
sess_ids = bah.limit_sess_ids(sess_ids, n_back)
# sess_ids = {180: [80585, 80542, 80497, 80452, 80407, 81250, 81205, 81160, 81115, 81070, 81025, 80980, 80933, 80876]}
#sess_ids = {179: [92562, 92600, 93412]} #[92562, 92600, 92646, 92692, 93412]

# get trial information
loc_db = db.LocalDB_ToneCatDelayResp()
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)
# remove trials where the stimulus didn't start
all_sess = all_sess[all_sess['trial_started']]

# %% Add formatted information columns to ease analysis and plotting

# group stimulus durations into bins
bin_size = 1
dur_bin_max = np.ceil(np.max(all_sess['stim_dur'])/bin_size)
dur_bin_min = np.floor(np.min(all_sess['stim_dur'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

all_sess['stim_dur_bin'] = all_sess['stim_dur'].apply(lambda x: dur_bin_labels[np.where(x >= dur_bins)[0][-1]])

# calculate the delay from the last relevant tone and group delays into bins
def calc_rel_tone_delay(row):
    rel_tone_pos = np.array(row['tone_info']) == row['relevant_tone_info']
    if not any(rel_tone_pos):
        rel_tone_time = row['rel_tone_end_times'][-1]
    else:
        rel_tone_time = row['rel_tone_end_times'][rel_tone_pos][-1]
        
    return row['stim_dur'] - rel_tone_time

all_sess['rel_tone_delay'] = all_sess.apply(lambda x: calc_rel_tone_delay(x), axis=1)

bin_size = 1
delay_bin_max = np.ceil(np.max(all_sess['rel_tone_delay'])/bin_size)
delay_bin_min = np.floor(np.min(all_sess['rel_tone_delay'])/bin_size)
delay_bins = np.arange(delay_bin_min, delay_bin_max+1)*bin_size
delay_bin_labels = ['{:.0f}-{:.0f}s'.format(delay_bins[i], delay_bins[i+1]) for i in range(len(delay_bins)-1)]

all_sess['delay_bin'] = all_sess['rel_tone_delay'].apply(lambda x: delay_bin_labels[np.where(x >= delay_bins)[0][-1]])

# Make trial type category labels based on tone types
def simplify_tone_info(x):
    return x.replace('low', 'L').replace('high', 'H').replace('left', 'L').replace('right', 'R')

all_sess['tone_info_str'] = all_sess['tone_info'].apply(
    lambda x: x if not type(x) is list else ', '.join(x)).apply(simplify_tone_info)
all_sess['variant_tone_info_str'] = all_sess['relevant_tone_info'].apply(simplify_tone_info) + ' - (' + all_sess['tone_info_str'] + ')'
all_sess['trial_type'] = all_sess['tone_info'].apply(
    lambda x: '1 tone' if not type(x) is list else
    '{0} tones - same'.format(len(x)) if x[0] == x[-1] else '{0} tones - diff'.format(len(x)))
all_sess['first_tone_info'] = all_sess['tone_info'].apply(lambda x: x if not type(x) is list else x[0]).apply(simplify_tone_info)
all_sess['second_tone_info'] = all_sess['tone_info'].apply(lambda x: '' if not type(x) is list else x[1]).apply(simplify_tone_info)

# get custom sorted values
tone_pitch_labels = np.array(sorted(all_sess['tone_info_str'].unique().tolist(), key=lambda x: str(len(x)) + x))
variant_tone_pitch_labels = np.array(sorted(all_sess['variant_tone_info_str'].unique().tolist(), key=lambda x: str(len(x)) + x))

# make sure these values are always sorted appropriately using categories
all_sess['tone_info_str'] = pd.Categorical(all_sess['tone_info_str'], categories=tone_pitch_labels)
all_sess['variant_tone_info_str'] = pd.Categorical(all_sess['variant_tone_info_str'], categories=variant_tone_pitch_labels)

# %% INVESTIGATE TRIAL TYPE COUNTS

# ignore bails because they are repeated
all_sess_no_bails = all_sess[all_sess['bail'] == False]

# aggregate count tables into dictionary
count_columns = ['correct_port', 'relevant_tone_info', 'stim_dur_bin', 'delay_bin', 'tone_info_str', 'variant_tone_info_str']
count_dict = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=True)

# plot bar charts and tables of trial distribution

def plotCounts(count_dict, stack, subj_ids, title, y_label):

    fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                            figsize=(3+0.5*len(subj_ids), 3*len(count_dict.keys())))
    fig.suptitle(title)

    bah.plot_counts(count_dict['correct_port'], axs[0], 'Correct Port', y_label, stack)
    bah.plot_counts(count_dict['relevant_tone_info'], axs[1], 'Relevant Tone', y_label, stack)
    bah.plot_counts(count_dict['stim_dur_bin'], axs[2], 'Trial Duration', y_label, stack)
    bah.plot_counts(count_dict['delay_bin'], axs[3], 'Response Delay', y_label, stack)
    bah.plot_counts(count_dict['tone_info_str'], axs[4], 'Tone Positions', y_label, stack)
    bah.plot_counts(count_dict['variant_tone_info_str'], axs[5], 'Relevant Tone x Tone Positions', y_label, stack, legend_cols=2)

plotCounts(count_dict_pct, 'v', subj_ids, 'Trial Distribution', '% Trials')
plotCounts(count_dict, 'h', subj_ids, 'Trial Counts', '# Trials')

# %% LOOK AT HIT & BAIL RATES

subj_ids = [190, 198, 199, 400] 

plot_bail = False
ind_subj = False
meta_subj = True

# calculate per-subject metrics
# hit rates over time
n_back = 10
hit_rate_signal = []
irr_tone_volume = []

plot_subjs = []
if ind_subj:
    plot_subjs.extend(subj_ids)
    
if meta_subj:
    plot_subjs.append('all')

for subj_id in plot_subjs:
    if subj_id == 'all':
        subj_sess = all_sess[all_sess['subjid'].isin(subj_ids)]
    else:
        subj_sess = all_sess[all_sess['subjid'] == subj_id]
        
    subj_sess_ids = np.unique(subj_sess['sessid'])

    rel_tones = np.unique(subj_sess['relevant_tone_info'])

    # CALCULATE HIT/BAIL METRICS
    # ignore bails and no responses
    rate_columns = [['first_tone_info', 'second_tone_info'],
                    ['stim_dur_bin', 'tone_info_str'],
                    ['trial_type', 'delay_bin']]
    
    hit_metrics_dict = {}
    bail_metrics_dict = {}

    for rel_tone in rel_tones:
        rel_subj_sess = subj_sess[subj_sess['relevant_tone_info'] == rel_tone]
        rel_subj_sess_no_bails = rel_subj_sess[(rel_subj_sess['bail'] == False) & (rel_subj_sess['choice'] != 'none')]
        if exclude_instruct_from_hitrate:
            rel_subj_sess_no_bails = rel_subj_sess_no_bails[rel_subj_sess_no_bails['tone_db_offsets'].apply(lambda x: all(x == 0))]
        
        rel_subj_sess_tone_heard = rel_subj_sess[rel_subj_sess['abs_tone_start_times'].str[0] < rel_subj_sess['cpoke_out_time']]

        hit_metrics_dict[rel_tone] = bah.get_rate_dict(rel_subj_sess_no_bails, 'hit', rate_columns)
        bail_metrics_dict[rel_tone] = bah.get_rate_dict(rel_subj_sess_tone_heard, 'bail', rate_columns)

    # COMPUTE METRICS SESSION BY SESSION

    # PROBABILITY OF OUTCOME BASED ON PREVIOUS OUTCOME:
    # p(incorrectly choose high|any high)
    # p(incorrectly choose low|any low)
    # p(choose right|previously chose right)
    # p(choose right|previously chose left)
    # p(stay with previous choice)
    # p(win-stay)
    # p(lose-switch)
    # p(bail|previous bail)
    # p(bail|previously incorrect)
    # p(bail|previously correct)

    n_high_any_high = {t: {'num': 0, 'denom': 0} for t in rel_tones}
    n_low_any_low = copy.deepcopy(n_high_any_high)
    n_right_prev_right = copy.deepcopy(n_high_any_high)
    n_right_prev_left = copy.deepcopy(n_high_any_high)
    n_repeat_choice = copy.deepcopy(n_high_any_high)
    n_win_stay = copy.deepcopy(n_high_any_high)
    n_lose_switch = copy.deepcopy(n_high_any_high)
    n_bail_prev_bail = copy.deepcopy(n_high_any_high)
    n_bail_prev_correct = copy.deepcopy(n_high_any_high)
    n_bail_prev_incorrect = copy.deepcopy(n_high_any_high)
    n_correct_prev_bail = copy.deepcopy(n_high_any_high)
    n_correct = copy.deepcopy(n_high_any_high)

    # ROLLING PERFORMANCE OVER TIME
    subj_hit_rate_signal = []
    subj_irr_tone_vol = []

    for sess_id in subj_sess_ids:
        ind_sess = subj_sess[subj_sess['sessid'] == sess_id]
        ind_sess_no_bails = ind_sess[(ind_sess['bail'] == False) & (ind_sess['choice'] != 'none')]

        if len(ind_sess) == 0:
            continue

        # p(incorrect side|tone presence)
        # find which side is the high tone
        if len(ind_sess[ind_sess['relevant_tone_info'] == 'high']) > 0:
            high_tone_side = ind_sess[ind_sess['relevant_tone_info'] == 'high'].iloc[0]['relevant_tone_port']
        else:
            low_tone_side = ind_sess[ind_sess['relevant_tone_info'] == 'low'].iloc[0]['relevant_tone_port']
            sides = np.array(['left', 'right'])
            high_tone_side = sides[sides != low_tone_side][0]
            
        high_tone_choice = ind_sess_no_bails['choice'].to_numpy() == high_tone_side
        incorrect_no_bails = ind_sess_no_bails['hit'].to_numpy() == False
        any_tone_high = ind_sess_no_bails['tone_info'].apply(
            lambda x: x == 'high' if not type(x) is list else any([val == 'high' for val in x])).to_numpy()
        any_tone_low = ind_sess_no_bails['tone_info'].apply(
            lambda x: x == 'low' if not type(x) is list else any([val == 'low' for val in x])).to_numpy()
        
        choices = ind_sess_no_bails['choice'].to_numpy()
        prev_choice_right = choices[:-1] == 'right'
        cur_choice_right = choices[1:] == 'right'
        stays = choices[:-1] == choices[1:]
        hits_no_bails = ind_sess_no_bails['hit'].astype(bool).to_numpy()
        bails = ind_sess['bail'].to_numpy()
        hits_all = ind_sess['hit'].to_numpy()
        prev_bail = bails[:-1] == True
        prev_correct = hits_all[:-1] == True
        prev_incorrect = hits_all[:-1] == False
        cur_correct = hits_all[1:] == True
        cur_bail = bails[1:] == True
            
        # break out metric by relevant tone block
        for rel_tone in rel_tones:
            rel_tone_sel = ind_sess['relevant_tone_info'] == rel_tone
            rel_tone_sel_no_bails = ind_sess_no_bails['relevant_tone_info'] == rel_tone
            
            if sum(rel_tone_sel) == 0:
                continue
            
            n_high_any_high[rel_tone]['num'] += sum(high_tone_choice & incorrect_no_bails & any_tone_high & rel_tone_sel_no_bails)
            n_high_any_high[rel_tone]['denom'] += sum(any_tone_high & rel_tone_sel_no_bails)
            n_low_any_low[rel_tone]['num'] += sum(~high_tone_choice & incorrect_no_bails & any_tone_low & rel_tone_sel_no_bails)
            n_low_any_low[rel_tone]['denom'] += sum(any_tone_low & rel_tone_sel_no_bails)
    
            # p(choice|previous choice)
            n_right_prev_right[rel_tone]['num'] += sum(cur_choice_right & prev_choice_right & rel_tone_sel_no_bails[1:])
            n_right_prev_right[rel_tone]['denom'] += sum(prev_choice_right & rel_tone_sel_no_bails[1:])
            n_right_prev_left[rel_tone]['num'] += sum(cur_choice_right & ~prev_choice_right & rel_tone_sel_no_bails[1:])
            n_right_prev_left[rel_tone]['denom'] += sum(~prev_choice_right & rel_tone_sel_no_bails[1:])
            n_repeat_choice[rel_tone]['num'] += sum(stays & rel_tone_sel_no_bails[1:])
            n_repeat_choice[rel_tone]['denom'] += sum(rel_tone_sel_no_bails[1:])
    
            # p(win-stay/lose-switch)
            n_win_stay[rel_tone]['num'] += sum(stays & hits_no_bails[1:] & rel_tone_sel_no_bails[1:])
            n_win_stay[rel_tone]['denom'] += sum(hits_no_bails[1:] & rel_tone_sel_no_bails[1:])
            n_lose_switch[rel_tone]['num'] += sum(~stays & ~hits_no_bails[1:] & rel_tone_sel_no_bails[1:])
            n_lose_switch[rel_tone]['denom'] += sum(~hits_no_bails[1:] & rel_tone_sel_no_bails[1:])
    
            # p(bail|previous result)
            n_bail_prev_bail[rel_tone]['num'] += sum(cur_bail & prev_bail & rel_tone_sel[1:])
            n_bail_prev_bail[rel_tone]['denom'] += sum(prev_bail & rel_tone_sel[1:])
            n_bail_prev_correct[rel_tone]['num'] += sum(cur_bail & prev_correct & rel_tone_sel[1:])
            n_bail_prev_correct[rel_tone]['denom'] += sum(prev_correct & rel_tone_sel[1:])
            n_bail_prev_incorrect[rel_tone]['num'] += sum(cur_bail & prev_incorrect & rel_tone_sel[1:])
            n_bail_prev_incorrect[rel_tone]['denom'] += sum(prev_incorrect & rel_tone_sel[1:])
            
            # p(correct|previous bail & current response)
            n_correct_prev_bail[rel_tone]['num'] += sum(cur_correct & prev_bail & rel_tone_sel[1:])
            n_correct_prev_bail[rel_tone]['denom'] += sum(~cur_bail & prev_bail & rel_tone_sel[1:])
            n_correct[rel_tone]['num'] += sum(cur_correct & rel_tone_sel[1:])
            n_correct[rel_tone]['denom'] += sum(~cur_bail & rel_tone_sel[1:])

        # performance over time
        subj_hit_rate_signal.extend(np.convolve(hits_no_bails, np.ones(n_back)/n_back, 'valid'))
        subj_irr_tone_vol.extend(ind_sess[ind_sess['bail'] == False]['tone_db_offsets'].iloc[n_back:].apply(lambda x: np.min(x)))

    hit_rate_signal.append(subj_hit_rate_signal)
    irr_tone_volume.append(subj_irr_tone_vol)

    # PLOT HIT/BAIL RATES AND RESPONSE PROBABILITIES

    # plot rate metrics

    if plot_bail:
        fig = plt.figure(constrained_layout=True, figsize=(9*len(rel_tones), 13))
        gs = GridSpec(4, 2*len(rel_tones), figure=fig)
    else:
        fig = plt.figure(constrained_layout=True, figsize=(8*len(rel_tones), 10))
        gs = GridSpec(3, 2*len(rel_tones), figure=fig, width_ratios=np.tile([2, 1], len(rel_tones)))

    fig.suptitle('Psychometrics (subj {0})'.format(str(subj_id)))

    for i, rel_tone in enumerate(rel_tones):
        ax = fig.add_subplot(gs[0, 2*i])

        ax.set_title('Hit Rates - {0} Tone Relevant'.format(rel_tone.capitalize()))

        # plot hit rates for first tone by second tone
        bah.plot_rate_heatmap(hit_metrics_dict[rel_tone], 'second_tone_info', 'Second Tone', 'first_tone_info', 'First Tone', ax)

        # plot hit rates for tones by stimulus duration
        ax = fig.add_subplot(gs[1, 2*i])

        bah.plot_rate_heatmap(hit_metrics_dict[rel_tone], 'stim_dur_bin', 'Stimulus Duration',
                              'tone_info_str', 'Unique Stimuli', ax, x_rot=30)
        
        # plot hit rates for tones by stimulus duration
        ax = fig.add_subplot(gs[2, 2*i])

        bah.plot_rate_heatmap(hit_metrics_dict[rel_tone], 'delay_bin', 'Delay from Last Relevant Tone',
                              'trial_type', 'Trial Type', ax, x_rot=30)

        # repeat above plots for bails
        if plot_bail:
            # plot bail rates
            ax = fig.add_subplot(gs[0, 2*i+1])

            ax.set_title('Bail Rates - {0} Tone Relevant'.format(rel_tone.capitalize()))

            bah.plot_rate_heatmap(bail_metrics_dict[rel_tone], 'second_tone_info', 'Second Tone', 'first_tone_info', 'First Tone', ax)

            ax = fig.add_subplot(gs[1, 2*i+1])

            bah.plot_rate_heatmap(bail_metrics_dict[rel_tone], 'stim_dur_bin', 'Stimulus Duration',
                                  'tone_info_str', 'Unique Stimuli', ax, x_rot=30)
            
            # plot hit rates for tones by stimulus duration
            ax = fig.add_subplot(gs[2, 2*i+1])
 
            bah.plot_rate_heatmap(bail_metrics_dict[rel_tone], 'delay_bin', 'Delay from Last Relevant Tone',
                                  'trial_type', 'Trial Type', ax, x_rot=30)

        # plot probabilities
        def comp_p(n_dict): return n_dict['num']/n_dict['denom']
        prob_labels = ['p(wrong high|any high)', 'p(wrong low|any low)', 'p(right|prev right)', 'p(right|prev left)', 'p(repeat choice)',
                       'p(stay|correct)', 'p(switch|incorrect)', 'p(bail|bail)', 'p(bail|incorrect)', 'p(bail|correct)', 'p(correct|bail,resp)', 'p(correct)']
        prob_values = [comp_p(n_high_any_high[rel_tone]), comp_p(n_low_any_low[rel_tone]), comp_p(n_right_prev_right[rel_tone]), comp_p(n_right_prev_left[rel_tone]),
                       comp_p(n_repeat_choice[rel_tone]), comp_p(n_win_stay[rel_tone]), comp_p(n_lose_switch[rel_tone]), comp_p(n_bail_prev_bail[rel_tone]),
                       comp_p(n_bail_prev_incorrect[rel_tone]), comp_p(n_bail_prev_correct[rel_tone]), comp_p(n_correct_prev_bail[rel_tone]), comp_p(n_correct[rel_tone])]

        if plot_bail:
            ax = fig.add_subplot(gs[3, 2*i:2*i+2])
            ax.plot(np.arange(len(prob_labels)), prob_values, 'o')
            ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
            ax.set_ylabel('Probability')
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.25, 0.25))
            ax.set_xticks(np.arange(len(prob_labels)), prob_labels, rotation=-45)
            ax.yaxis.grid(True)
        else:
            ax = fig.add_subplot(gs[:, 2*i+1])
            ax.plot(np.flip(prob_values), np.arange(len(prob_labels)), 'o')
            ax.axvline(0.5, dashes=[4, 4], c='k', lw=1)
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            ax.set_xticks(np.arange(0, 1.25, 0.25))
            ax.set_yticks(np.arange(len(prob_labels)), np.flip(prob_labels), rotation=30)
            ax.xaxis.grid(True)
        ax.set_title('Response Probabilities')


# plot performance over time
n_cols = 1 if len(subj_ids) == 1 else 2
n_rows = int(np.ceil(len(subj_ids)/n_cols))*2
fig, axs = plt.subplots(n_rows, n_cols,
                        figsize=(8*n_cols, n_rows*2), constrained_layout=True)
axs = axs.reshape((-1, n_cols))
fig.suptitle('Rolling task performance (last {0} trials) and instructor cue'.format(n_back))
for i, subj_id in enumerate(subj_ids):
    ax = axs[int(np.floor(i/2)*2), i % 2]
    ax.plot(hit_rate_signal[i])
    ax.set_title('Subject {0}'.format(subj_id))
    ax.set_xticks([])
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.grid(True)
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
    if i % 2 == 0:
        ax.set_ylabel('Hit Rate')

    ax = axs[int(np.floor(i/2)*2+1), i % 2]
    ax.plot(irr_tone_volume[i])
    ax.set_xticks([])
    ax.set_ylim(-52, 2)
    ax.yaxis.grid(True)
    if i % 2 == 0:
        ax.set_ylabel('Irrelevant Tone Loudness (dB)')

plt.show(block=False)

# %% LOOK AT HIT REACTION TIMES/BAIL TIMES BY STIMULUS TYPE

# if plot_timing:
#     # format the data appropriately
#     rt_data = all_sess.rename(
#         columns={'subjid': 'Subject', 'RT': 'Reaction Time'})
#     rt_data['Subject'] = rt_data['Subject'].astype(str)
#     rt_data['Outcome'] = rt_data['hit'].apply(lambda x: 'Correct' if x else 'Incorrect')

#     # add bail time
#     rt_data['Bail Time'] = rt_data['cpoke_out_time'] - rt_data['stim_start_time']
#     rt_data.loc[rt_data['bail'] == False, 'Bail Time'] = np.nan

#     # first look at reaction times grouped by tones and stimulus layout accross subjects

#     plot = (ggplot(rt_data[rt_data['bail'] == False], aes(x='Subject', y='Reaction Time', fill='Outcome'))
#             + geom_violin(alpha=0.7)
#             + facet_grid('tone_info_str~tone_pos_dur')
#             + stat_summary(fun_data='mean_se', position=position_dodge(0.9))
#             + labs(title='Reaction Time Distributions'))

#     plot.draw(show=True)

#     # Then look at bail times, filtering out bails that happen before any tone is played
#     # construct dataframe with horizontal lines for each tone
#     tone_times = rt_data.groupby(['tone_info_str', 'tone_pos_dur'])['rel_tone_start_times_str'].agg(np.unique).apply(
#         lambda x: float(x) if not type(x) is np.ndarray else np.array([float(i) for i in x[0].split(',')])).reset_index()
#     # need to separate out the lines
#     unique_tone_times = [i for i in tone_times['rel_tone_start_times_str'].to_list() if type(i) is np.ndarray]
#     unique_tone_times = np.unique([i for l in unique_tone_times for i in l])
#     tone_times_dict = {}
#     for i, t in enumerate(unique_tone_times):
#         tone_times_dict[i] = tone_times.copy()
#         tone_times_dict[i]['rel_tone_start_times_str'] = tone_times_dict[i]['rel_tone_start_times_str'].apply(
#             lambda x: np.nan if np.isnan(x).any() else x[x == t][0] if any(x == t) else np.nan)

#     plot = (ggplot(rt_data[(rt_data['bail'] == True)], aes(x='Subject', y='Bail Time', fill='Subject'))
#             + geom_violin(alpha=0.7)
#             + geom_hline(data=tone_times_dict[0], mapping=aes(yintercept='rel_tone_start_times_str'), linetype='dashed')
#             + geom_hline(data=tone_times_dict[1], mapping=aes(yintercept='rel_tone_start_times_str'), linetype='dashed')
#             + geom_hline(data=tone_times_dict[2], mapping=aes(yintercept='rel_tone_start_times_str'), linetype='dashed')
#             + facet_grid('tone_info_str~tone_pos_dur')
#             + stat_summary(fun_data='mean_se', position=position_dodge(0.9))
#             + labs(title='Bail Time Distributions'))

#     plot.draw(show=True)

# %% Plot combined performance for multiple subjects

# This is updated code based on the above, just didn't want to rewrite all the prior stuff to make this

# subj_ids = [179, 180, 182, 186, 188, 202]
# sess_ids = db_access.get_subj_sess_ids(subj_ids, stage=8, date_end='2023-09-17')

# # get trial information
# loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True
# all_sess = loc_db.get_behavior_data(utils.flatten_dict_array(sess_ids))
# # remove trials where the stimulus didn't start
# all_sess = all_sess[all_sess['trial_started']]

# variants = all_sess['task_variant'].unique()

# # format columns for ease of aggregating and display
# # round stimulus duration
# all_sess['stim_dur'] = all_sess['stim_dur'].round(2).apply(lambda x: str(x) + 's')
# # reformat tone start times and high tones arrays into strings to be hashable for value counting and for display
# all_sess['trial_type'] = all_sess['tone_info'].apply(
#     lambda x: '1 tone' if not type(x) is list or len(x) == 1 else
#         '{0} tones - same'.format(len(x)) if x[0] == x[-1] else '{0} tones - diff'.format(len(x)))

# rate_columns = ['stim_dur', 'trial_type', ['trial_type', 'stim_dur']]
# hit_metrics_dict = {}
# data_metrics = {}

# for variant in variants:
#     var_sess = all_sess[all_sess['task_variant'] == variant]
#     var_sess_no_bails = var_sess[(var_sess['bail'] == False) & (var_sess['choice'] != 'none')]

#     data_metrics[variant] = {}
#     data_metrics[variant]['n_subjects'] = var_sess['subjid'].nunique()
#     data_metrics[variant]['n_sessions'] = var_sess['sessid'].nunique()
#     data_metrics[variant]['n_trials'] = len(var_sess_no_bails)

#     hit_metrics_dict[variant] = bah.get_rate_dict(var_sess_no_bails, 'hit', rate_columns)

# fig, axs = plt.subplots(1, 2, figsize=(8, 2.5), constrained_layout=True)

# for i, variant in enumerate(variants):
#     ax = axs[i]
#     ax.set_title('\'{0}\' Variant'.format(variant))

#     bah.plot_rate_heatmap(hit_metrics_dict[variant], 'stim_dur', 'Stimulus Duration', 'trial_type', 'Trial Type', ax, fmt='.2f')

# plt.rcParams["svg.fonttype"] = 'none'
# fig.savefig(path.join(utils.get_user_home(), 'downloads', 'beh_performance.svg'), format='svg', dpi=1200)

