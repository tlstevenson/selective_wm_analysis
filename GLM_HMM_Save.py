# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:05:08 2026

HMM + per-state multinomial logistic regression for stage 7 tone categorization behavior.
Completely self-contained data loading, feature engineering, fitting, and cross-validation script.

Baseline Category:
- Category 2 (Bail) is the baseline reference category.
- Plotting and console output are presented as "vs. Bail" to preserve 
  the natural cognitive hierarchy (Engagement Level 1 and Spatial Choice Level 2).

Predictors:
- curr_target
- delay_norm 
- target_delay_interaction
- prev_target
- prev_bail_val
- prev_choice_reward
- prev_choice_unreward   

@author: alex truong
"""

import os
import pickle
import ssm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings

import init
import pyutils.utils as utils
import hankslab_db.tonecatdelayresp_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah

# Force a global random seed to maintain identical state mapping across runs
np.random.seed(42)

# Ensure interactive plots pop up
plt.ion()

#%% Save Path for Persisting Data
save_path = 'glm_hmm_results.pkl'

if os.path.exists(save_path):
    with open(save_path, 'rb') as f:
        cv_results = pickle.load(f)
    print("Loaded existing cross validation HMM results. Resuming...")
else:
    cv_results = {}

#%% Loading data
stage = 7
stage_name = 'growDelay'
n_back = 6
active_subjects_only = False
reload = False

if active_subjects_only:
    subject_info = db_access.get_active_subj_stage(protocol='ToneCatDelayResp', stage_num=stage)
else:
    subject_info = db_access.get_protocol_subject_info(protocol='ToneCatDelayResp', stage_num=stage, stage_name=stage_name)

subj_ids = [198, 199, 237, 238, 274, 400, 402, 424, 483]  # updated subj_ids

# Get session ids
sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_num=stage, protocol='ToneCatDelayResp')
sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# Get trial information
loc_db = db.LocalDB_ToneCatDelayResp()
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

# Remove trials where the stimulus didnt start
all_sess = all_sess[all_sess['trial_started']]

#%% Formatting data
# Calculate delay time
tone_dur = 0.4
all_sess['delay_time'] = all_sess['stim_dur'] - all_sess['rel_tone_start_times'] - tone_dur

# Reformat tone infos into a single string for hashability
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

#%% Preparing Predictors
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
# (Since 2 is the last index, SSM naturally uses Category 2 [Bail] as the baseline reference!)
def get_outcome(row):
    if row['bail']: return 2
    if row['choice'] == 'right': return 1
    return 0

all_sess['outcome'] = all_sess.apply(get_outcome, axis=1)

# History + Predictors (Grouped by subjid and sessid)
all_sess['prev_choice'] = all_sess.groupby(['subjid', 'sessid'])['curr_choice_val'].shift(1)
all_sess['prev_target'] = all_sess.groupby(['subjid', 'sessid'])['curr_target'].shift(1)
all_sess['prev_bail_val'] = all_sess.groupby(['subjid', 'sessid'])['bail'].shift(1).astype(float)
all_sess['prev_rew_bool'] = all_sess.groupby(['subjid', 'sessid'])['rewarded'].shift(1)

# Win-Stay Case
all_sess['prev_choice_reward'] = all_sess['prev_choice'] * (all_sess['prev_rew_bool'] == True).astype(float)

# Lose-Switch Case
all_sess['prev_choice_unreward'] = all_sess['prev_choice'] * ((all_sess['prev_rew_bool'] == False) & (all_sess['prev_choice'] != 0)).astype(float)

# Stimulus + Time Factors
d_max = all_sess['delay_time'].max()
all_sess['delay_norm'] = (all_sess['delay_time']) / (d_max) 
all_sess['target_delay_interaction'] = all_sess['curr_target'] * all_sess['delay_norm']

# List of predictors
predictors = [
    'curr_target', 
    'delay_norm', 
    'target_delay_interaction', 
    'prev_target',
    'prev_bail_val', 
    'prev_choice_reward', 
    'prev_choice_unreward'
]

# Drop NaNs to clean up the matrix edges
reg_ready_df = all_sess.dropna(subset=predictors + ['outcome']).copy()

# Add standard intercept mapping column to design inputs
reg_ready_df['const'] = 1.0
ssm_predictors = ['const'] + predictors

#%% Restructure into chronological session lists and dicts for ssm
print("Restructuring regression-ready data into arrays for HMM...")

inpts = []
true_choices = []
sess_data_dict = {} 

session_groups = reg_ready_df.groupby(['subjid', 'sessid'])

for (subj, sess), group in session_groups:
    # Ensure trials within the individual session are processed in strict time sequence
    sorted_group = group.sort_index()
    
    # Isolate predictors matrix (N_trials x 8) and outcomes vector (N_trials x 1)
    x_matrix = sorted_group[ssm_predictors].to_numpy(dtype=float)
    y_vector = sorted_group['outcome'].to_numpy(dtype=int).reshape(-1, 1)
    
    inpts.append(x_matrix)
    true_choices.append(y_vector)
    
    if subj not in sess_data_dict:
        sess_data_dict[subj] = {}
        
    sess_data_dict[subj][sess] = {
        'inputs': x_matrix,       # Shape: (N_trials, 8)
        'choices': y_vector,      # Shape: (N_trials, 1)
    }

print(f"Data conversion complete: packaged data for {len(sess_data_dict)} subjects.")

#%% Fit Model, Plot, and Print Weights per Subject
num_states = 2        # Set to the state count you want to visualize (e.g., 2, 3, or 4)
obs_dim = 1           
num_categories = 3    
input_dim = len(ssm_predictors) 

print(f"\nFitting {num_states}-State GLM-HMM and plotting for EACH subject...")

for subj in sess_data_dict.keys():
    print(f"\n" + "="*60)
    print(f"                SUBJECT {subj}")
    print("="*60)
    
    # 1. Extract this specific subject's data arrays
    subj_inpts = []
    subj_choices = []
    for sess in sess_data_dict[subj].keys():
        subj_inpts.append(sess_data_dict[subj][sess]['inputs'])
        subj_choices.append(sess_data_dict[subj][sess]['choices'])
        
    # 2. Initialize and Fit Model
    subj_glmhmm = ssm.HMM(num_states, 
                          obs_dim,          # D: observation dim
                          input_dim,        # M: input dim
                          observations="input_driven_obs", 
                          observation_kwargs=dict(C=num_categories), 
                          transitions="standard")
    
    print(f"Running Expectation-Maximization (EM) loop for Subject {subj}...")
    subj_glmhmm.fit(subj_choices, inputs=subj_inpts, method="em", num_iters=200, tolerance=10**-4)
    
    # 3. Extract Parameters
    weights = subj_glmhmm.observations.params
    recovered_trans_mat = np.exp(subj_glmhmm.transitions.params)[0]
    
    # 4. Plotting (Modified to represent raw parameters relative to BAIL baseline)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6), dpi=100)
    fig.suptitle(f"Subject {subj} - {num_states}-State GLM-HMM Profile", fontsize=16, fontweight='bold')
    
    # Visual markers
    line_styles = ['solid', 'dashed']
    markers = ['o', 'x']
    contrast_labels = ["Engage Left (vs Bail)", "Engage Right (vs Bail)"]
    
    # Distinct palettes: blues for State 1, reds for State 2
    colors_state_1 = ['#1f77b4', '#2980b9']  # Blue tones
    colors_state_2 = ['#d62728', '#c0392b']  # Red tones
    state_colors = [colors_state_1, colors_state_2]
    
    for k in range(num_states):
        # Category 2 (Bail) is baseline, so raw weights are:
        # weights[k][0] -> Category 0 (Left) vs. Category 2 (Bail)
        # weights[k][1] -> Category 1 (Right) vs. Category 2 (Bail)
        w_left_vs_bail = weights[k][0]   
        w_right_vs_bail = weights[k][1]  
        
        # Plot Left vs Bail (solid line)
        axs[0].plot(range(input_dim), w_left_vs_bail, marker=markers[0], linestyle=line_styles[0],
                    color=state_colors[k % len(state_colors)][0], lw=2.5, 
                    label=f"State {k+1}: {contrast_labels[0]}")
        
        # Plot Right vs Bail (dashed line)
        axs[0].plot(range(input_dim), w_right_vs_bail, marker=markers[1], linestyle=line_styles[1],
                    color=state_colors[k % len(state_colors)][1], lw=2.0, 
                    label=f"State {k+1}: {contrast_labels[1]}")

    axs[0].set_ylabel("GLM Weight Parameter (Log-Odds vs. Bail)")
    axs[0].set_xticks(range(input_dim))
    axs[0].set_xticklabels(ssm_predictors, rotation=45, ha='right')
    axs[0].axhline(y=0, color="k", alpha=0.3, ls="-")
    axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')
    axs[0].set_title(f"Subject {subj} - Recovered GLM State Weights (vs Bail)")
    axs[0].grid(True, axis='y', alpha=0.2)

    im = axs[1].imshow(recovered_trans_mat, vmin=0, vmax=1, cmap='bone')
    for i in range(num_states):
        for j in range(num_states):
            axs[1].text(j, i, f"{recovered_trans_mat[i, j]:.2f}", 
                        ha="center", va="center", color="red", weight='bold')

    axs[1].set_xticks(range(num_states))
    axs[1].set_xticklabels([f"State {i+1}" for i in range(num_states)])
    axs[1].set_yticks(range(num_states))
    axs[1].set_yticklabels([f"State {i+1}" for i in range(num_states)])
    axs[1].set_ylabel("State at t")
    axs[1].set_xlabel("State at t+1")
    axs[1].set_title(f"Subject {subj} - Transition Probabilities")

    plt.tight_layout()
    plt.show()

    # 5. Print Tables to Console (Modified to match vs. Bail Baseline)
    print(f"\n--- GLM-HMM COEFFICIENTS (BASELINE = BAIL) ---")
    for k in range(num_states):
        print(f"\n{'-'*15} STATE {k+1} {'-'*15}")
        w_left_vs_bail = weights[k][0]   
        w_right_vs_bail = weights[k][1]  
        
        df_weights = pd.DataFrame({
            "Left (vs Bail)": w_left_vs_bail,
            "Right (vs Bail)":  w_right_vs_bail
        }, index=ssm_predictors)
        
        print(df_weights.round(4))

#%% Cross-Validation Loop

# States to test (1 state = basic multinomial regression)
num_states_to_test = [1, 2, 3, 4]
subjects = list(sess_data_dict.keys())

# Cross-validation loop
for subj_id in subjects:
    if subj_id not in cv_results:
        cv_results[subj_id] = {}

    sessions = list(sess_data_dict[subj_id].keys())

    for num_states in num_states_to_test:
        state_key = f"{num_states}_states"
        if state_key not in cv_results[subj_id]:
            cv_results[subj_id][state_key] = {}

        # Leave-One-Out Cross-Validation (LOOCV)
        for test_sess in sessions:
            if test_sess in cv_results[subj_id][state_key]:
                continue
                
            print(f"Running Subj: {subj_id} | States: {num_states} | Held-out: {test_sess}")

            # Split data into Train and Test sets
            train_inputs, train_choices = [], []
            test_inputs, test_choices = [], []

            for sess_id in sessions:
                if sess_id == test_sess:
                    test_inputs.append(sess_data_dict[subj_id][sess_id]['inputs'])
                    test_choices.append(sess_data_dict[subj_id][sess_id]['choices'])
                else:
                    train_inputs.append(sess_data_dict[subj_id][sess_id]['inputs'])
                    train_choices.append(sess_data_dict[subj_id][sess_id]['choices'])

            input_dim = train_inputs[0].shape[1]

            # Initialize GLM-HMM
            model = ssm.HMM(num_states, 
                            1,                   # D: observation dimension        
                            input_dim,           # M: input dimensions
                            observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), 
                            transitions="standard")

            # Fit the model on the training data
            model.fit(train_choices, inputs=train_inputs, method="em", num_iters=200, tolerance=10**-4)

            # Evaluate Log-Likelihood on the test data
            test_ll = model.log_likelihood(test_choices, inputs=test_inputs)

            # Evaluate Log-Likelihood on the training data
            train_ll = model.log_likelihood(train_choices, inputs=train_inputs)
            n_train_trials = sum([len(c) for c in train_choices])
            n_trials = sum([len(c) for c in test_choices])

            # Calculate degrees of freedom (k)
            k_transitions = num_states * (num_states - 1)
            k_glm_weights = num_states * input_dim * (num_categories - 1)
            k = k_transitions + k_glm_weights
            
            # BIC Formula: k*ln(n_train) - 2*train_ll
            bic = k * np.log(n_train_trials) - 2 * train_ll

            # Store results in dictionary
            cv_results[subj_id][state_key][test_sess] = {
                'test_ll': float(test_ll),
                'bic': float(bic),
                'n_trials': int(n_trials)
            }

            # Dump to disk
            with open(save_path, 'wb') as f:
                pickle.dump(cv_results, f)
                
#%% Graphing Log-Likelihoods and BIC across num_states

# Load Data
with open(save_path, 'rb') as f:
    cv_results = pickle.load(f)

# Parse Nested Dictionary into a Flat DataFrame
cv_rows = []
for subj_id, states_dict in cv_results.items():
    for state_key, sessions_dict in states_dict.items():
        num_states = int(state_key.split('_')[0])
        for sess_id, metrics in sessions_dict.items():
            cv_rows.append({
                'subject': subj_id,
                'num_states': num_states,
                'session': sess_id,
                'test_ll': metrics['test_ll'],
                'bic': metrics['bic'],
                'n_trials': metrics['n_trials']
            })

df_cv = pd.DataFrame(cv_rows)

# Aggregate Metrics Across Held-Out Sessions
summary_df = df_cv.groupby(['subject', 'num_states']).agg({
    'test_ll': 'mean',
    'bic': 'mean'
}).reset_index()

# Plot Metrics per Subject
unique_subjects = summary_df['subject'].unique()
n_subjects = len(unique_subjects)

fig, axs = plt.subplots(n_subjects, 2, figsize=(12, 3.5 * n_subjects), squeeze=False)

for idx, subj in enumerate(unique_subjects):
    subj_data = summary_df[summary_df['subject'] == subj].sort_values('num_states')
    
    # Left Column: Test Log-Likelihood (Higher is better)
    axs[idx, 0].plot(subj_data['num_states'], subj_data['test_ll'], marker='o', linestyle='-', lw=2)
    axs[idx, 0].set_title(f"Subj {subj} - Cross-Validated Log-Likelihood")
    axs[idx, 0].set_ylabel("Mean Test LL")
    axs[idx, 0].set_xticks(subj_data['num_states'])
    axs[idx, 0].grid(True, alpha=0.3)
    
    # Right Column: BIC Score (Lower is better)
    axs[idx, 1].plot(subj_data['num_states'], subj_data['bic'], marker='s', linestyle='--', lw=2)
    axs[idx, 1].set_title(f"Subj {subj} - Cross-Validated BIC")
    axs[idx, 1].set_ylabel("Mean BIC")
    axs[idx, 1].set_xticks(subj_data['num_states'])
    axs[idx, 1].grid(True, alpha=0.3)
    
    # Highlight the minimum BIC point (optimal state count)
    min_bic_row = subj_data.loc[subj_data['bic'].idxmin()]
    axs[idx, 1].plot(min_bic_row['num_states'], min_bic_row['bic'], marker='X', markersize=12)

# Label formatting
for ax in axs[-1, :]:
    ax.set_xlabel("Number of Latent States")

plt.tight_layout()
plt.show()

#%% Notes
"""
Graph Interpretation + Logic
                  
1. 1D Multinomial Workaround for d=1:
   As a workaround to the requirement for d=1, we are forced to map our 2D behavioral 
   outcomes (Engagement [Bail vs. Respond] and Spatial Choice [Left vs. Right]) into 
   a single 1D categorical variable with C=3 distinct classes:
     - Category 0: Bail (Default / Aborted trial / 'none' choice)
     - Category 1: Respond Left
     - Category 2: Respond Right
   This satisfies the ssm library's strict univariate constraint while mathematically 
   preserving the full joint distribution and hierarchical decision structure of the task.

2. Level 1: Engagement (Bail vs. Response):
   When comparing bail vs response, look for the heights of the lines with respect to baseline (0.0). 
   If the animal has at least 1 line (either pref right or pref left) significantly higher than 
   baseline, it is likely that for the given predictor, a larger (more positive) predictor value 
   drives the rat to respond more than bail. 
   If both lines are significantly below baseline, the opposite is true (larger positive values 
   of the predictor drive the animal to bail).

3. Level 2: Choice Given a Response (Right vs. Left):
   When comparing left vs right given a response, check the vertical distance between the lines of 
   the same color (dashed Right vs. solid Left). Depending on which is higher, that side is preferred 
   for larger (more positive) values of that predictor.
     - If the Dashed line (Right) is higher: Positive predictor values drive Rightward choices.
     - If the Solid line (Left) is higher: Positive predictor values drive Leftward choices.

"""

#%% Final Model Fitting & Posterior Extraction (2 States Standard)

"""
State-Specific Psychometric and GLM-HMM Behavioral Analysis

- Categorizes each trial as belonging to State 1 or State 2.
- Runs psychometrics, rate heatmaps, and transition probabilities
  separately for trials belonging to each strategy.

"""

print("\n" + "="*60)
print("EXTRACTING POSTERIOR STATE PROBABILITIES PER SUBJECT (K=2)")
print("="*60)

# Dictionary to store the final dataframes with state probabilities
final_subject_dfs = {}
final_num_states = 2 
obs_dim = 1
num_categories = 3
input_dim = len(ssm_predictors)

for subj in sess_data_dict.keys():
    print(f"\nProcessing Final Posteriors for Subject {subj}...")
    print(f"  -> Forcing exactly {final_num_states} states for uniform cross-subject psychometrics.")
    
    # 1. Gather ALL sessions for this subject to train the final global model
    subj_inpts = []
    subj_choices = []
    sess_order = list(sess_data_dict[subj].keys()) # Keep track of session order
    
    for sess in sess_order:
        subj_inpts.append(sess_data_dict[subj][sess]['inputs'])
        subj_choices.append(sess_data_dict[subj][sess]['choices'])
        
    # 2. Initialize and Fit the FINAL model for this specific animal (Forced K=2)
    final_model = ssm.HMM(final_num_states, 
                          obs_dim, 
                          input_dim, 
                          observations="input_driven_obs", 
                          observation_kwargs=dict(C=num_categories), 
                          transitions="standard")
                          
    print(f"  -> Fitting final model on all sessions...")
    final_model.fit(subj_choices, inputs=subj_inpts, method="em", num_iters=200, tolerance=10**-4)
    
    # 3. Extract Posteriors (Smoothed Probabilities) and map to DataFrame
    # Filter the original regression dataframe for this subject
    subj_df = reg_ready_df[reg_ready_df['subjid'] == subj].copy()
    
    # Create columns for state probabilities (fill with NaNs initially)
    for k in range(final_num_states):
        subj_df[f'state_{k+1}_prob'] = np.nan
        
    # 4. Map posteriors back to the specific session trials
    for i, sess in enumerate(sess_order):
        # expected_states returns a tuple: (E[z_t], E[z_t, z_{t+1}], loglike)
        # E[z_t] is shape (n_trials, num_states), giving the probability of each state per trial
        # Singular input= ensures correct parsing by ssm
        posteriors = final_model.expected_states(subj_choices[i], input=subj_inpts[i])[0] 
        
        # Mask for this specific session
        sess_mask = (subj_df['sessid'] == sess)
        
        # Assign probabilities to columns
        for k in range(final_num_states):
            subj_df.loc[sess_mask, f'state_{k+1}_prob'] = posteriors[:, k]
            
    # 5. Assign the "Dominant State" (argmax) for easy binary grouping later
    prob_cols = [f'state_{k+1}_prob' for k in range(final_num_states)]
    subj_df['assigned_state'] = subj_df[prob_cols].idxmax(axis=1).apply(lambda x: int(x.split('_')[1]))
    
    final_subject_dfs[subj] = subj_df

# 6. Combine all subjects into one master dataframe and save
master_posterior_df = pd.concat(final_subject_dfs.values())
master_posterior_df.to_csv('glm_hmm_state_posteriors.csv', index=False)
print("\nSaved 'glm_hmm_state_posteriors.csv' successfully.")

#%% Post-Hoc Analysis: State-Specific Behavioral Rates & Psychometrics
print("\n" + "="*60)
print("RUNNING COMPARATIVE STATE-SPECIFIC PSYCHOMETRIC ANALYSIS")
print("="*60)

# Global configuration
plot_bail = True
rate_columns = ['tone_info_str', 'delay_bin', ['tone_info_str', 'delay_bin']]

# Map outcomes
master_posterior_df['is_bail'] = (master_posterior_df['outcome'] == 2).astype(int)
master_posterior_df['is_choice'] = (master_posterior_df['outcome'] != 2).astype(int)

# 1. Output general subject state table
state_summary = master_posterior_df.groupby(['subjid', 'assigned_state']).agg({
    'is_bail': 'mean',
    'is_choice': 'mean',
    'curr_choice_val': lambda x: x[x != 0].mean() 
})
state_summary.columns = ['Bail Rate', 'Engagement Rate', 'Rightward Bias (Choice Trials)']
print("\nSUBJECT-SPECIFIC STATE INTERPRETATION SUMMARY:")
print(state_summary.round(3))
          
# 2. Loop through subjects and accumulate state-specific metrics
for subj_id in master_posterior_df['subjid'].unique():
    subj_sess = master_posterior_df[master_posterior_df['subjid'] == subj_id]
    subj_sess_ids = np.unique(subj_sess['sessid'])
    
    states_data = {}
    
    for state_val in [1, 2]:
        state_sess = subj_sess[subj_sess['assigned_state'] == state_val]
        if len(state_sess) < 10: continue

        state_sess_no_bails = state_sess[(state_sess['bail'] == False) & (state_sess['choice'] != 'none')]
        state_sess_tone_heard = state_sess[state_sess['cpoke_out_time'] > state_sess['abs_tone_start_times']]

        # Initialize counters
        counters = {k: {'num': 0, 'denom': 0} for k in ['right_pr', 'right_pl', 'repeat', 'win_stay', 'lose_switch', 
                    'bail_pb', 'bail_pc', 'bail_pi', 'hit_ps_same', 'hit_ps_diff', 'hit_ps_same_c', 'hit_ps_same_i', 
                    'hit_ps_diff_c', 'hit_ps_diff_i', 'hit_pb', 'hit_pb_same', 'hit_pb_diff', 'hit_resp']}

        for sess_id in subj_sess_ids:
            ind_sess = state_sess[state_sess['sessid'] == sess_id]
            ind_sess_no_bails = ind_sess[(ind_sess['bail'] == False) & (ind_sess['choice'] != 'none')]
            if len(ind_sess) < 3: continue
            
            # Logic: Spatial choice transitions
            choices = ind_sess_no_bails['choice'].to_numpy()
            if len(choices) > 1:
                p_r = choices[:-1] == 'right'; c_r = choices[1:] == 'right'
                counters['right_pr']['num'] += sum(c_r & p_r); counters['right_pr']['denom'] += sum(p_r)
                counters['right_pl']['num'] += sum(c_r & ~p_r); counters['right_pl']['denom'] += sum(~p_r)
                counters['repeat']['num'] += sum(choices[:-1] == choices[1:]); counters['repeat']['denom'] += len(choices) - 1
                stays = choices[:-1] == choices[1:]; hits_nb = ind_sess_no_bails['hit'].astype(bool).to_numpy()
                counters['win_stay']['num'] += sum(stays & hits_nb[:-1]); counters['win_stay']['denom'] += sum(hits_nb[:-1])
                counters['lose_switch']['num'] += sum(~stays & ~hits_nb[:-1]); counters['lose_switch']['denom'] += sum(~hits_nb[:-1])

            # Logic: Engagement state transitions
            bails_arr = ind_sess['bail'].fillna(False).astype(bool).to_numpy()
            hits_arr = ind_sess['hit'].fillna(False).astype(bool).to_numpy()
            
            if len(bails_arr) > 1:
                counters['bail_pb']['num'] += sum(bails_arr[1:] & bails_arr[:-1])
                counters['bail_pb']['denom'] += sum(bails_arr[:-1])
                
                counters['bail_pc']['num'] += sum(bails_arr[1:] & hits_arr[:-1])
                counters['bail_pc']['denom'] += sum(hits_arr[:-1])
                
                counters['bail_pi']['num'] += sum(bails_arr[1:] & ~hits_arr[:-1])
                counters['bail_pi']['denom'] += sum(~hits_arr[:-1])

            # Logic: Stimulus history
            if len(ind_sess_no_bails) > 1:
                stims = ind_sess_no_bails['relevant_tone_info'].to_numpy(); p_s = stims[:-1] == stims[1:]
                counters['hit_ps_same']['num'] += sum(hits_nb[1:] & p_s); counters['hit_ps_same']['denom'] += sum(p_s)
                counters['hit_ps_diff']['num'] += sum(hits_nb[1:] & ~p_s); counters['hit_ps_diff']['denom'] += sum(~p_s)
                counters['hit_ps_same_c']['num'] += sum(hits_nb[1:] & p_s & hits_nb[:-1]); counters['hit_ps_same_c']['denom'] += sum(p_s & hits_nb[:-1])
                counters['hit_ps_same_i']['num'] += sum(hits_nb[1:] & p_s & ~hits_nb[:-1]); counters['hit_ps_same_i']['denom'] += sum(p_s & ~hits_nb[:-1])
                counters['hit_ps_diff_c']['num'] += sum(hits_nb[1:] & ~p_s & hits_nb[:-1]); counters['hit_ps_diff_c']['denom'] += sum(~p_s & hits_nb[:-1])
                counters['hit_ps_diff_i']['num'] += sum(hits_nb[1:] & ~p_s & ~hits_nb[:-1]); counters['hit_ps_diff_i']['denom'] += sum(~p_s & ~hits_nb[:-1])

            # Post-bail sensory response
            if len(ind_sess) > 1:
                stims_f = ind_sess['relevant_tone_info'].to_numpy()
                p_s_f = (stims_f[:-1] == stims_f[1:])
                res = (bails_arr[1:] == False)
                
                counters['hit_pb']['num'] += sum(hits_arr[1:] & bails_arr[:-1] & res)
                counters['hit_pb']['denom'] += sum(bails_arr[:-1] & res)
                
                counters['hit_pb_same']['num'] += sum(hits_arr[1:] & bails_arr[:-1] & p_s_f & res)
                counters['hit_pb_same']['denom'] += sum(bails_arr[:-1] & p_s_f & res)
                
                counters['hit_pb_diff']['num'] += sum(hits_arr[1:] & bails_arr[:-1] & ~p_s_f & res)
                counters['hit_pb_diff']['denom'] += sum(bails_arr[:-1] & ~p_s_f & res)
            
            counters['hit_resp']['num'] += sum(hits_arr); counters['hit_resp']['denom'] += len(hits_arr)

        # Compile state data
        def comp_p(n): return n['num']/n['denom'] if n['denom'] > 0 else 0.0
        states_data[state_val] = {
            'hit_map': bah.get_rate_dict(state_sess_no_bails, 'hit', rate_columns),
            'bail_map': bah.get_rate_dict(state_sess_tone_heard, 'bail', rate_columns),
            'probs': [comp_p(counters[k]) for k in ['right_pr', 'right_pl', 'repeat', 'win_stay', 'lose_switch', 'hit_ps_same', 'hit_ps_same_c', 'hit_ps_same_i', 'hit_ps_diff', 'hit_ps_diff_c', 'hit_ps_diff_i', 'hit_pb', 'hit_pb_same', 'hit_pb_diff', 'bail_pb', 'bail_pi', 'bail_pc']],
            'avg_hit': comp_p(counters['hit_resp'])
        }

    # 3. Comparative Plot
    if not states_data: continue
    fig = plt.figure(layout='constrained', figsize=(15, 8))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 3])
    
    # Updated labels matching your visual format
    labels = ['p(right|prev R)', 'p(right|prev L)', 'p(repeat)', 'p(stay|win)', 'p(switch|lose)', 
              'p(hit|same)', 'p(hit|same,win)', 'p(hit|same,lose)', 'p(hit|diff)', 'p(hit|diff,win)', 
              'p(hit|diff,lose)', 'p(hit|bail)', 'p(hit|bail,same)', 'p(hit|bail,diff)', 
              'p(bail|bail)', 'p(bail|miss)', 'p(bail|hit)']
    
    # Generate Heatmaps
    for i, s in enumerate([1, 2]):
        if s not in states_data: continue
        ax_h = fig.add_subplot(gs[0, i]); ax_b = fig.add_subplot(gs[1, i])
        bah.plot_rate_heatmap(states_data[s]['hit_map'], 'delay_bin', 'Delay', 'tone_info_str', 'Tone', ax_h)
        bah.plot_rate_heatmap(states_data[s]['bail_map'], 'delay_bin', 'Delay', 'tone_info_str', 'Tone', ax_b)
        ax_h.set_title(f"State {s} Hit Rate"); ax_b.set_title(f"State {s} Bail Rate")

    # Probability & History Dot Plot with Horizontal Reference Lines
    ax_p = fig.add_subplot(gs[2, :])
    
    # Add constant Chance line
    ax_p.axhline(0.5, color='black', linestyle='--', alpha=0.5, label="Chance")
    
    # Plot dots and state-specific Hit Rate lines
    colors = ['#1f77b4', '#d62728']
    for (s, data), color in zip(states_data.items(), colors):
        # Dots
        ax_p.plot(range(len(labels)), data['probs'], 'o', markersize=8, color=color, label=f"State {s} Prob")
        
        # Horizontal reference line for state hit rate
        avg_hit = data['avg_hit']
        ax_p.axhline(avg_hit, color=color, linestyle='--', alpha=0.6, label=f"State {s} Hit Rate ({avg_hit:.2f})")
    
    ax_p.set_xticks(range(len(labels)))
    ax_p.set_xticklabels(labels, rotation=-30, ha='left')
    ax_p.set_ylabel("Probability")
    ax_p.set_title(f"Response & History Probabilities - Subj {subj_id}")
    ax_p.set_ylim(-0.05, 1.05)
    ax_p.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.show()