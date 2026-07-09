# -*- coding: utf-8 -*-
"""
Script to investigate performance on the tone categorization task stage 7 - single tone

@author: tanner stevenson
"""

import init

import pyutils.utils as utils
import hankslab_db.tonecatdelayresp_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

import statsmodels.api as sm
import warnings

# %% LOAD DATA

stage = 7
stage_name = 'growDelay'
n_back = 6
fp_sess_only = True
active_subjects_only = False
reload = False

if active_subjects_only:
    subject_info = db_access.get_active_subj_stage(protocol='ToneCatDelayResp', stage_num=stage)
else:
    subject_info = db_access.get_protocol_subject_info(protocol='ToneCatDelayResp', stage_num=stage, stage_name=stage_name)

#subj_ids = subject_info['subjid']
#subj_ids = subj_ids[subj_ids != 187]
#subj_ids = [187,190,192,193,198,199,400,402]
#subj_ids = [198, 199, 237, 238, 274, 400, 402, 424, 483]  # updated subj_ids
subj_ids = [198, 199, 274, 400] # short list for testing code

# get session ids
if fp_sess_only:
    sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids, protocol='ToneCatDelayResp', stage_num=stage)
else:
    sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_num=stage, protocol='ToneCatDelayResp')
    # sess_ids = db_access.get_fp_data_sess_ids(protocol='ToneCatDelayResp', stage_num=stage)
    sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# get trial information
loc_db = db.LocalDB_ToneCatDelayResp()
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)
# remove trials where the stimulus didn't start
all_sess = all_sess[all_sess['trial_started']]

# %% Format Data

# calculate delay time
tone_dur = 0.4
all_sess['delay_time'] = all_sess['stim_dur'] - all_sess['rel_tone_start_times'] - tone_dur

# format columns for ease of aggregating and display

# reformat tone infos into a single string for hashability
all_sess['tone_info_str'] = all_sess['tone_info'].apply(
    lambda x: x if not type(x) is list else ', '.join(x))

tone_info_order = ['high', 'low', 'left', 'right']
all_sess['tone_info_str'] = pd.Categorical(all_sess['tone_info_str'], categories=tone_info_order)


bin_size = 1
delay_bin_max = np.ceil(np.max(all_sess['delay_time'])/bin_size)
delay_bin_min = np.floor(np.min(all_sess['delay_time'])/bin_size)
delay_bins = np.arange(delay_bin_min, delay_bin_max+1)*bin_size
delay_bin_labels = ['{:.0f}-{:.0f}s'.format(delay_bins[i], delay_bins[i+1]) for i in range(len(delay_bins)-1)]

all_sess['delay_bin'] = all_sess['delay_time'].apply(lambda x: delay_bin_labels[np.where(x >= delay_bins)[0][-1]])

bin_size = 1
dur_bin_max = np.ceil(np.max(all_sess['stim_dur'])/bin_size)
dur_bin_min = np.floor(np.min(all_sess['stim_dur'])/bin_size)
dur_bins = np.arange(dur_bin_min, dur_bin_max+1)*bin_size
dur_bin_labels = ['{:.0f}-{:.0f}s'.format(dur_bins[i], dur_bins[i+1]) for i in range(len(dur_bins)-1)]

all_sess['dur_bin'] = all_sess['stim_dur'].apply(lambda x: dur_bin_labels[np.where(x >= dur_bins)[0][-1]])


# %% INVESTIGATE TRIAL TYPE COUNTS

# ignore bails because they are repeated
all_sess_no_bails = all_sess[all_sess['bail'] == False]

# aggregate count tables into dictionary
count_columns = ['correct_port', 'dur_bin', 'delay_bin', 'tone_info_str']
count_dict = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=False)
count_dict_pct = bah.get_count_dict(all_sess_no_bails, 'subjid', count_columns, normalize=True)

# plot bar charts and tables of trial distribution

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
bah.plot_counts(count_dict['correct_port'], axs[0], 'Correct Port', '# Trials', 'h')
bah.plot_counts(count_dict['dur_bin'], axs[1], 'Stimulus Duration', '# Trials', 'h')
bah.plot_counts(count_dict['delay_bin'], axs[2], 'Response Delay', '# Trials', 'h')
bah.plot_counts(count_dict['tone_info_str'], axs[3], 'Stimulus Type', '# Trials', 'h')

fig, axs = plt.subplots(len(count_dict.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict.keys())))
bah.plot_counts(count_dict_pct['correct_port'], axs[0], 'Correct Port', '% Trials', 'v')
bah.plot_counts(count_dict_pct['dur_bin'], axs[1], 'Stimulus Duration', '% Trials', 'v')
bah.plot_counts(count_dict_pct['delay_bin'], axs[2], 'Response Delay', '% Trials', 'v')
bah.plot_counts(count_dict_pct['tone_info_str'], axs[3], 'Stimulus Type', '% Trials', 'v')

# %% LOOK AT HIT & BAIL RATES

plot_bail = True
ind_subj = True
meta_subj = True

# CALCULATE HIT/BAIL METRICS
# ignore bails and no responses
rate_columns = ['tone_info_str', 'delay_bin', ['tone_info_str', 'delay_bin']]

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

    subj_sess_no_bails = subj_sess[(subj_sess['bail'] == False) & (subj_sess['choice'] != 'none')]
    subj_sess_tone_heard = subj_sess[subj_sess['cpoke_out_time'] > subj_sess['abs_tone_start_times']]

    hit_metrics_dict = bah.get_rate_dict(subj_sess_no_bails, 'hit', rate_columns)
    bail_metrics_dict = bah.get_rate_dict(subj_sess_tone_heard, 'bail', rate_columns)

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
    
    # p(correct | previously same)
    # p(correct | previously diff)
    # p(correct | previously same & correct)
    # p(correct | previously same & incorrect)
    # p(correct | previously diff & correct)
    # p(correct | previously diff & incorrect)
    
    # p(correct | prev bail)
    # p(correct | prev bail & same tone)
    # p(correct | prev bail & diff tone)
    
    # p(correct | response)

    # Create dictionaries for each rate metric
    
    n_right_prev_right = {'num': 0, 'denom': 0}
    n_right_prev_left = {'num': 0, 'denom': 0}
    n_repeat_choice = {'num': 0, 'denom': 0}
    n_win_stay = {'num': 0, 'denom': 0}
    n_lose_switch = {'num': 0, 'denom': 0}
    n_bail_prev_bail = {'num': 0, 'denom': 0}
    n_bail_prev_correct = {'num': 0, 'denom': 0}
    n_bail_prev_incorrect = {'num': 0, 'denom': 0}
    n_bail_prev_correct_diff = {'num': 0, 'denom': 0}
    
    n_hit_prev_stim_same = {'num': 0, 'denom': 0}
    n_hit_prev_stim_diff = {'num': 0, 'denom': 0}
    n_hit_prev_stim_same_prev_correct = {'num': 0, 'denom': 0}
    n_hit_prev_stim_same_prev_incorrect = {'num': 0, 'denom': 0}
    n_hit_prev_stim_diff_prev_correct = {'num': 0, 'denom': 0}
    n_hit_prev_stim_diff_prev_incorrect = {'num': 0, 'denom': 0}
    
    n_hit_prev_bail = {'num': 0, 'denom': 0}
    n_hit_prev_bail_same_tone = {'num': 0, 'denom': 0}
    n_hit_prev_bail_diff_tone = {'num': 0, 'denom': 0}
    
    n_hit_response = {'num': 0, 'denom': 0}


    for sess_id in subj_sess_ids:
        ind_sess = subj_sess[subj_sess['sessid'] == sess_id]
        ind_sess_no_bails = ind_sess[(ind_sess['bail'] == False) & (ind_sess['choice'] != 'none')]

        if len(ind_sess) == 0:
            continue
        
        # Accumulate counts for each rate metric
        
        # p(choice|previous choice)
        choices = ind_sess_no_bails['choice'].to_numpy()
        prev_choice_right = choices[:-1] == 'right'
        cur_choice_right = choices[1:] == 'right'
        n_right_prev_right['num'] += sum(cur_choice_right & prev_choice_right)
        n_right_prev_right['denom'] += sum(prev_choice_right)
        n_right_prev_left['num'] += sum(cur_choice_right & ~prev_choice_right)
        n_right_prev_left['denom'] += sum(~prev_choice_right)
        n_repeat_choice['num'] += sum(choices[:-1] == choices[1:])
        n_repeat_choice['denom'] += len(choices)-1

        # p(win-stay/lose-switch)
        stays = choices[:-1] == choices[1:]
        hits_no_bails = ind_sess_no_bails['hit'].astype(bool).to_numpy()
        n_win_stay['num'] += sum(stays & hits_no_bails[:-1])
        n_win_stay['denom'] += sum(hits_no_bails[:-1])
        n_lose_switch['num'] += sum(~stays & ~hits_no_bails[:-1])
        n_lose_switch['denom'] += sum(~hits_no_bails[:-1])

        # p(bail|previous result)
        bails = ind_sess['bail'].to_numpy()
        hits = ind_sess['hit'].to_numpy()
        prev_bail = bails[:-1] == True
        prev_correct = hits[:-1] == True
        prev_incorrect = hits[:-1] == False
        cur_bail = bails[1:] == True
        n_bail_prev_bail['num'] += sum(cur_bail & prev_bail)
        n_bail_prev_bail['denom'] += sum(prev_bail)
        n_bail_prev_correct['num'] += sum(cur_bail & prev_correct)
        n_bail_prev_correct['denom'] += sum(prev_correct)
        n_bail_prev_incorrect['num'] += sum(cur_bail & prev_incorrect)
        n_bail_prev_incorrect['denom'] += sum(prev_incorrect)
        
        # p(correct | previously same)
        stims = ind_sess_no_bails['relevant_tone_info'].to_numpy()
        prev_stim_same = stims[:-1] == stims[1:]
        n_hit_prev_stim_same['num'] += sum(hits_no_bails[1:] & prev_stim_same)
        n_hit_prev_stim_same['denom'] += sum(prev_stim_same)
        
        # p(correct | previously diff)
        n_hit_prev_stim_diff['num'] += sum(hits_no_bails[1:] & ~prev_stim_same)
        n_hit_prev_stim_diff['denom'] += sum(~prev_stim_same)
        
        # p(correct | previously same & correct)
        n_hit_prev_stim_same_prev_correct['num'] += sum(hits_no_bails[1:] & prev_stim_same & hits_no_bails[:-1])
        n_hit_prev_stim_same_prev_correct['denom'] += sum(prev_stim_same & hits_no_bails[:-1])
        
        # p(correct | previously same & incorrect)
        n_hit_prev_stim_same_prev_incorrect['num'] += sum(hits_no_bails[1:] & prev_stim_same & ~hits_no_bails[:-1])
        n_hit_prev_stim_same_prev_incorrect['denom'] += sum(prev_stim_same & ~hits_no_bails[:-1])
        
        # p(correct | previously diff & correct)
        n_hit_prev_stim_diff_prev_correct['num'] += sum(hits_no_bails[1:] & ~prev_stim_same & hits_no_bails[:-1])
        n_hit_prev_stim_diff_prev_correct['denom'] += sum(~prev_stim_same & hits_no_bails[:-1])
        
        # p(correct | previously diff & incorrect)
        n_hit_prev_stim_diff_prev_incorrect['num'] += sum(hits_no_bails[1:] & ~prev_stim_same & ~hits_no_bails[:-1])
        n_hit_prev_stim_diff_prev_incorrect['denom'] += sum(~prev_stim_same & ~hits_no_bails[:-1])
        
        # p(correct | prev bail)
        stims = ind_sess['relevant_tone_info'].to_numpy()
        prev_stim_same = stims[:-1] == stims[1:]
        current_correct = hits[1:] == True
        response = bails[1:] == False
        n_hit_prev_bail['num'] += sum(current_correct & prev_bail & response)
        n_hit_prev_bail['denom'] += sum(prev_bail & response)
        
        # p(correct | prev bail & same tone)
        n_hit_prev_bail_same_tone['num'] += sum(current_correct & prev_bail & prev_stim_same & response)
        n_hit_prev_bail_same_tone['denom'] += sum(prev_bail & prev_stim_same & response)
        
        # p(correct | prev bail & diff tone)
        n_hit_prev_bail_diff_tone['num'] += sum(current_correct & prev_bail & ~prev_stim_same & response)
        n_hit_prev_bail_diff_tone['denom'] += sum(prev_bail & ~prev_stim_same & response)
        
        # Overall hit rate
        n_hit_response['num'] += sum(hits_no_bails)
        n_hit_response['denom'] += len(hits_no_bails)

    # PLOT HIT/BAIL RATES AND RESPONSE PROBABILITIES

    # plot hit metrics
    if plot_bail:
        fig = plt.figure(layout='constrained', figsize=(8, 5))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[2,1])
    else:
        fig = plt.figure(layout='constrained', figsize=(5, 5))
        gs = GridSpec(2, 1, figure=fig, height_ratios=[2,1])

    fig.suptitle('Psychometrics (subj {0})'.format(str(subj_id)))

    ax = fig.add_subplot(gs[0, 0])
    ax.set_title('Hit Rates')

    bah.plot_rate_heatmap(hit_metrics_dict, 'delay_bin', 'Response Delay', 'tone_info_str', 'Tone', ax)

    if plot_bail:
        ax = fig.add_subplot(gs[0, 1])
        ax.set_title('Bail Rates')

        bah.plot_rate_heatmap(bail_metrics_dict, 'delay_bin', 'Response Delay', 'tone_info_str', 'Tone', ax)

    # plot probabilities
    def comp_p(n_dict): return n_dict['num']/n_dict['denom']
    
    #Add label for each metric
    prob_labels = ['p(right|prev right)', 'p(right|prev left)', 'p(repeat choice)', 'p(stay|correct)', 'p(switch|incorrect)',
                   'p(hit|same tone)', 'p(hit|same tone & hit)', 'p(hit|same tone & miss)',
                   'p(hit|diff tone)', 'p(hit|diff tone & hit)', 'p(hit|diff tone & miss)',
                   'p(hit|bail)', 'p(hit|bail & same tone)', 'p(hit|bail & diff tone)',
                   'p(bail|bail)', 'p(bail|miss)', 'p(bail|hit)']
    # Add call to 'comp_p' for each metric's dictionary
    prob_values = [comp_p(n_right_prev_right), comp_p(n_right_prev_left), comp_p(n_repeat_choice), comp_p(n_win_stay), comp_p(n_lose_switch), 
                   comp_p(n_hit_prev_stim_same), comp_p(n_hit_prev_stim_same_prev_correct), comp_p(n_hit_prev_stim_same_prev_incorrect), 
                   comp_p(n_hit_prev_stim_diff), comp_p(n_hit_prev_stim_diff_prev_correct), comp_p(n_hit_prev_stim_diff_prev_incorrect), 
                   comp_p(n_hit_prev_bail), comp_p(n_hit_prev_bail_same_tone), comp_p(n_hit_prev_bail_diff_tone),
                   comp_p(n_bail_prev_bail), comp_p(n_bail_prev_incorrect), comp_p(n_bail_prev_correct)] 

    hit_rate = comp_p(n_hit_response)
    
    ax = fig.add_subplot(gs[1, :])
    ax.plot(np.arange(len(prob_labels)), prob_values, 'o')
    ax.axhline(0.5, dashes=[4, 4], c='k', lw=1)
    ax.axhline(hit_rate, dashes=[4, 4], c='r', lw=1)
    ax.set_ylabel('Probability')
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.set_xticks(np.arange(len(prob_labels)), prob_labels, rotation=-60)
    ax.yaxis.grid(True)
    ax.set_title('Response Probabilities')


#%% Multinomial Logistic Regression

# Suppress pandas fragmentation and future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None 

print("Setting up Regression Data...")

# Map labels to numbers for regression
# Left = -1, Right = 1, None/Bail = 0
val_map = {'left': -1, 'right': 1, 'none': 0}

all_sess['curr_target'] = all_sess['correct_port'].map(val_map)
all_sess['curr_choice_val'] = all_sess['choice'].map(val_map)

# Defining outcomes: 0 = Left choice, 1 = Right choice, 2 = Bail
def get_outcome(row):
    if row['bail']: return 2
    if row['choice'] == 'right': return 1
    return 0
all_sess['outcome'] = all_sess.apply(get_outcome, axis=1)

# History + Predictors (Grouped by subjid and sessid)
# Recalculated to ensure trial 1 of a new session doesn't look at the previous session
all_sess['prev_choice'] = all_sess.groupby(['subjid', 'sessid'])['curr_choice_val'].shift(1)
all_sess['prev_target'] = all_sess.groupby(['subjid', 'sessid'])['curr_target'].shift(1)
all_sess['prev_bail_val'] = all_sess.groupby(['subjid', 'sessid'])['bail'].shift(1).astype(float)
all_sess['prev_rew_bool'] = all_sess.groupby(['subjid', 'sessid'])['rewarded'].shift(1)

# Win-Stay Case
# If choice = right(1) and reward=True, interaction is 1.
# If choice = left(-1) and reward=True, interaction is -1.
all_sess['prev_choice_reward'] = all_sess['prev_choice'] * (all_sess['prev_rew_bool'] == True).astype(float)

# Lose-Switch Case:
all_sess['prev_choice_unreward'] = all_sess['prev_choice'] * ((all_sess['prev_rew_bool'] == False) & (all_sess['prev_choice'] != 0)).astype(float)

# Stimulus + Time Factors
d_max = all_sess['delay_time'].max()
all_sess['delay_norm'] = (all_sess['delay_time']) / (d_max)  # removed d_min subtraction on num and denom
all_sess['target_delay_interaction'] = all_sess['curr_target'] * all_sess['delay_norm']

# Filter for Regression Ready Data
# List of predictors:
# We exclude raw prev_reward because it's included in the interactions, to prevent degeneracy
predictors = [
    'curr_target', 
    'delay_norm', 
    'target_delay_interaction', 
    'prev_target',
    'prev_bail_val', 
    'prev_choice_reward', 
    'prev_choice_unreward'
]

## prev_choice was degenerating for some trials, causing very large numbers or NaNs

# Drop NaNs (including trial 1's which now have NaNs in history columns)
reg_ready_df = all_sess.dropna(subset=predictors + ['outcome']).copy()

# Fitting Loop (Aggregate and Individual)
fit_mode = 'both' # options: 'all', 'individual', 'both'
plot_results = True 

if fit_mode == 'all':
    subj_list = ['all']
elif fit_mode == 'individual':
    subj_list = reg_ready_df['subjid'].unique().tolist()
else:
    subj_list = ['all'] + reg_ready_df['subjid'].unique().tolist()

for subj in subj_list:
    if subj == 'all':
        print(f"\n{'-'*50}\nFitting Model: ALL SUBJECTS\n{'-'*50}")
        subset = reg_ready_df.copy()
    else:
        print(f"\n{'-'*50}\nFitting Model: SUBJECT {subj}\n{'-'*50}")
        subset = reg_ready_df[reg_ready_df['subjid'] == subj].copy()

    if len(subset) < 50:
        continue

    X = sm.add_constant(subset[predictors])
    y = subset['outcome']

    try:
        model = sm.MNLogit(y, X).fit(method='newton', maxiter=100, disp=False)
        
        # Metric: Avg P that model makes right choice (Geometric Mean)
        avg_p_choice = np.exp(model.llf / model.nobs)
        
        print(f"Log-Likelihood: {model.llf:.2f}")
        print(f"Avg P(Choice) per trial: {avg_p_choice:.3f}")
        print("\n--- Regression Summary ---")
        print(model.summary())
        
        # Display p-values for quick filtering
        p_vals = model.pvalues
        print("\nInsignificant predictors (P > 0.05):")
        print(p_vals[p_vals > 0.05].dropna(how='all'))
        
        # Plotting Trajectories for each subject
        if plot_results and subj != 'all':
            sample_sess = subset['sessid'].unique()[0]
            sess_subset = subset[subset['sessid'] == sample_sess]
            X_sess = sm.add_constant(sess_subset[predictors])
            y_probs = model.predict(X_sess) 

            fig, ax = plt.subplots(figsize=(10, 4))
            trials = np.arange(len(sess_subset))
            
            ax.plot(trials, y_probs[0], color='blue', label='P(Left)', alpha=0.8)
            ax.plot(trials, y_probs[1], color='green', label='P(Right)', alpha=0.8)
            ax.plot(trials, y_probs[2], color='red', label='P(Bail)', alpha=0.8)

            # Choice indicators at the top
            for i, (idx, row) in enumerate(sess_subset.iterrows()):
                c = 'blue' if row['outcome']==0 else ('green' if row['outcome']==1 else 'red')
                h = 1.15 if row['rewarded'] else 1.05 
                ax.vlines(i, 1.0, h, colors=c, linewidth=2)

            ax.set_ylim(0, 1.25)
            ax.set_title(f"Predictions vs Choice (Subj {subj}, Sess {sample_sess})")
            ax.set_ylabel("Probability")
            ax.legend(loc='lower left', ncol=3)
            plt.show()

    except Exception as e:
        print(f"Fitting failed for {subj}: {e}")