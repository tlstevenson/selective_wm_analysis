# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 20:05:08 2026

HMM + per-state multinomial logistic regression for stage 7 tone categorization behavior.
Completely self-contained data loading, feature engineering, fitting, and cross-validation script.
Includes multi-initialization consistency checks, robust LOOCV across random restarts,
persisted results, state-specific psychometric analysis, Test BIC model selection, 
and transition matrix heatmaps.

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
import seaborn as sns
import warnings
from scipy.optimize import linear_sum_assignment
from scipy.stats import sem
from scipy.spatial.distance import cdist

import init
import pyutils.utils as utils
import hankslab_db.tonecatdelayresp_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah
import joblib
from itertools import combinations

# Ensure interactive plots pop up
plt.ion()

#%% Global Configuration & Save Paths
consistency_save_path = 'glm_hmm_consistency.pkl'
cv_save_path = 'glm_hmm_cv_multi_init.pkl'
full_metrics_save_path = 'glm_hmm_full_metrics.pkl'

n_inits_consistency = 20    # Number of fits to test model weight consistency
n_inits_cv = 20             # Number of random initializations per CV fold
n_inits_criteria = 20       # Number of random initializations for EM fitting
num_states_to_test = [1, 2, 3, 4]
target_num_states = 2       # Target K for state-specific diagnostics
tol = 1e-4                  # Convergence tolerance for EM fitting
regen_params = False        # Set to True to re-fit models; False to load cached pkl files

#%% Loading Data
stage = 7
stage_name = 'growDelay'
n_back = 6
active_subjects_only = False
reload = False

if active_subjects_only:
    subject_info = db_access.get_active_subj_stage(protocol='ToneCatDelayResp', stage_num=stage)
else:
    subject_info = db_access.get_protocol_subject_info(protocol='ToneCatDelayResp', stage_num=stage, stage_name=stage_name)

subj_ids = [198, 199, 234, 235, 237, 238, 274, 400, 402, 419, 421, 422, 424, 483]

# Get session ids
sess_ids = db_access.get_subj_sess_ids(subj_ids, stage_num=stage, protocol='ToneCatDelayResp')
sess_ids = bah.limit_sess_ids(sess_ids, n_back)

# Get trial information
loc_db = db.LocalDB_ToneCatDelayResp()
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

# Remove trials where the stimulus didnt start
all_sess = all_sess[all_sess['trial_started']].copy()

#%% Formatting Data
tone_dur = 0.4
bin_size = 1.0

all_sess['delay_time'] = all_sess['stim_dur'] - all_sess['rel_tone_start_times'] - tone_dur

# Reformat tone infos into a single categorical series
all_sess['tone_info_str'] = pd.Categorical(
    all_sess['tone_info'].apply(lambda x: x if not isinstance(x, list) else ', '.join(x)),
    categories=['high', 'low', 'left', 'right']
)

# Vectorized binning with pd.cut (include_lowest=True prevents boundary NaNs)
d_min, d_max = np.floor(all_sess['delay_time'].min()), np.ceil(all_sess['delay_time'].max())
d_bins = np.arange(d_min, d_max + bin_size, bin_size)
d_labels = [f'{b[0]:.1f}-{b[1]:.1f}s' for b in zip(d_bins[:-1], d_bins[1:])]
all_sess['delay_bin'] = pd.cut(all_sess['delay_time'], bins=d_bins, labels=d_labels, include_lowest=True)

s_min, s_max = np.floor(all_sess['stim_dur'].min()), np.ceil(all_sess['stim_dur'].max())
s_bins = np.arange(s_min, s_max + bin_size, bin_size)
s_labels = [f'{b[0]:.1f}-{b[1]:.1f}s' for b in zip(s_bins[:-1], s_bins[1:])]
all_sess['dur_bin'] = pd.cut(all_sess['stim_dur'], bins=s_bins, labels=s_labels, include_lowest=True)

#%% Preparing Predictors
warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.chained_assignment = None 

print("Setting up Regression Data...")

val_map = {'left': -1, 'right': 1, 'none': 0}
all_sess['curr_target'] = all_sess['correct_port'].map(val_map)
all_sess['curr_choice_val'] = all_sess['choice'].map(val_map)

# Fast vectorized outcome mapping: 0 = Left, 1 = Right, 2 = Bail
all_sess['outcome'] = np.select(
    [all_sess['bail'].astype(bool), (all_sess['choice'] == 'right').astype(bool)],
    [2, 1],
    default=0
)

# History + Predictors (Grouped once)
grouped = all_sess.groupby(['subjid', 'sessid'])
all_sess['prev_choice'] = grouped['curr_choice_val'].shift(1)
all_sess['prev_target'] = grouped['curr_target'].shift(1)
all_sess['prev_bail_val'] = grouped['bail'].shift(1).astype(float)
all_sess['prev_rew_bool'] = grouped['rewarded'].shift(1)

# Win-Stay & Lose-Switch Cases
all_sess['prev_choice_reward'] = all_sess['prev_choice'] * (all_sess['prev_rew_bool'] == True).astype(float)
all_sess['prev_choice_unreward'] = all_sess['prev_choice'] * ((all_sess['prev_rew_bool'] == False) & (all_sess['prev_choice'] != 0)).astype(float)

# Stimulus + Time Factors
d_max_val = all_sess['delay_time'].max()
all_sess['delay_norm'] = all_sess['delay_time'] / d_max_val
all_sess['target_delay_interaction'] = all_sess['curr_target'] * all_sess['delay_norm']

predictors = [
    'curr_target', 
    'delay_norm', 
    'target_delay_interaction', 
    'prev_target',
    'prev_bail_val', 
    'prev_choice_reward', 
    'prev_choice_unreward'
]

reg_ready_df = all_sess.dropna(subset=predictors + ['outcome']).copy()
reg_ready_df['const'] = 1.0
ssm_predictors = ['const'] + predictors

# =========================================================
# PREDICTOR DEFINITIONS & FORMULAS (GLM-HMM Design Matrix)
# =========================================================

# 1. const (Baseline Intercept)
# - Definition: Fixed constant at 1.0 for all trials (x_0 = 1.0)
# - Role: Captures intrinsic spatial choice bias toward Right vs. Left/Bail, independent of cues.

# 2. curr_target (Current Target Stimulus)
# - Definition: Discrete target direction on trial t: Right (+1), Left (-1), None (0)
# - Role: Quantifies baseline sensory sensitivity (cue-driven choice steering).

# 3. delay_norm (Normalized Cue Delay Duration)
# - Formula: delay_norm = delay_time / max(delay_time)  ∈ [0, 1]
# - Role: Tests if wait time alters decision thresholds or choice bias (impatience/state shifts).

# 4. target_delay_interaction (Target x Delay Interaction)
# - Formula: target_delay_interaction = curr_target * delay_norm  ∈ [-1, 1]
# - Role: Evaluates whether longer delays amplify or dampen sensory cue influence.

# 5. prev_target (Previous Target Stimulus)
# - Definition: Target direction on trial t-1: Right (+1), Left (-1), None (0)
# - Role: Measures stimulus perseveration (repeating prior target location regardless of outcome).

# 6. prev_bail_val (Previous Trial Bail Indicator)
# - Formula: prev_bail_val = 1.0 if bailed on t-1 else 0.0
# - Role: Tracks post-bail strategy resets or bias shifts following an aborted trial.

# 7. prev_choice_reward (Rewarded Choice History / Win-Stay)
# - Formula: prev_choice * (prev_rew_bool == True)  ∈ {-1, 0, +1}
#   (+1 = Rewarded Right choice, -1 = Rewarded Left choice, 0 = Unrewarded/Bailed/None)
# - Role: Quantifies Win-Stay reinforcement dynamics (probability of repeating a rewarded action).

# 8. prev_choice_unreward (Unrewarded Choice History / Lose-Switch)
# - Formula: prev_choice * ((prev_rew_bool == False) & (prev_choice != 0))  ∈ {-1, 0, +1}
#   (+1 = Error Right choice, -1 = Error Left choice, 0 = Rewarded/Bailed/None)
# - Role: Quantifies Lose-Shift dynamics (avoidance of an error-producing port).

#%% Restructure into chronological session arrays for SSM
print("Restructuring regression-ready data into arrays for HMM...")

sess_data_dict = {} 
for (subj, sess), group in reg_ready_df.groupby(['subjid', 'sessid']):
    sorted_group = group.sort_index()
    if subj not in sess_data_dict:
        sess_data_dict[subj] = {}
        
    sess_data_dict[subj][sess] = {
        'inputs': sorted_group[ssm_predictors].to_numpy(dtype=float),
        'choices': sorted_group['outcome'].to_numpy(dtype=int).reshape(-1, 1)
    }

print(f"Data conversion complete: packaged data for {len(sess_data_dict)} subjects.")

#%% Helper Function: State Alignment via Trial Posteriors
def align_states_by_posteriors(ref_posteriors, target_posteriors, target_weights, target_trans):
    """
    Aligns state indices using Hungarian Algorithm on vectorized L2 distance matrix.
    """
    # 3D broadcasting computes the full K x K Euclidean distance matrix in one step
    cost_matrix = np.linalg.norm(ref_posteriors[:, :, None] - target_posteriors[:, None, :], axis=0)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    # Use np.ix_ to cleanly align transition matrix rows and columns
    aligned_trans = target_trans[np.ix_(col_ind, col_ind)]
    
    return target_weights[col_ind], aligned_trans, target_posteriors[:, col_ind], col_ind

#%% SECTION 1: Multi-Fit Posterior & State Consistency Check
print("\n" + "="*60)
print("SECTION 1: MULTI-FIT CONSISTENCY CHECK ACROSS RANDOM INITIALIZATIONS")
print("="*60)

# Check cache while respecting the regen_params flag
if not regen_params and os.path.exists(consistency_save_path):
    with open(consistency_save_path, 'rb') as f:
        consistency_results = pickle.load(f)
    print("Loaded existing consistency results.")
else:
    consistency_results = {}
    if regen_params:
        print("regen_params is True. Forcing a clean re-fit of all consistency checks...")
    else:
        print("No cache found. Starting fresh...")

num_states_check = 2
obs_dim = 1
num_categories = 3
input_dim = len(ssm_predictors)

for subj in sess_data_dict.keys():
    # Respect regen_params when deciding whether to skip a subject
    if not regen_params and subj in consistency_results and len(consistency_results[subj]['fits']) >= n_inits_consistency:
        print(f"Subject {subj} consistency fits already completed. Skipping...")
        continue
        
    print(f"\nRunning {n_inits_consistency} fits for Subject {subj} to check parameter consistency...")
    
    subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in sess_data_dict[subj]]
    subj_choices = [sess_data_dict[subj][s]['choices'] for s in sess_data_dict[subj]]
    
    fits_data = []
    
    for init_seed in range(n_inits_consistency):
        np.random.seed(init_seed * 100 + 7)
        model = ssm.HMM(num_states_check, obs_dim, input_dim, 
                        observations="input_driven_obs",
                        observation_kwargs=dict(C=num_categories), 
                        transitions="standard")
        
        train_lls = model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
        
        # Safely extract expected states across session lists
        posterior_list = []
        for c, inp in zip(subj_choices, subj_inputs):
            res = model.expected_states(c, input=inp)
            post = res[0] if isinstance(res, tuple) else res
            posterior_list.append(post)
            
        # Vectorized posterior concatenation across session lists
        concat_posteriors = np.vstack(posterior_list)
        
        # Safely extract transition matrix probabilities
        trans_matrix = model.transitions.transition_matrix.copy()
        
        fits_data.append({
            'seed': init_seed,
            'model': model,
            'weights': model.observations.params.copy(),
            'trans_mat': trans_matrix,
            'posteriors': concat_posteriors,
            'final_ll': train_lls[-1]
        })
        
    # Align all initializations to the best fitting initialization (highest LL)
    best_fit = max(fits_data, key=lambda x: x['final_ll'])
    ref_posteriors = best_fit['posteriors']
    
    aligned_weights_list = []
    aligned_trans_list = []
    
    for fit in fits_data:
        w_aligned, t_aligned, _, _ = align_states_by_posteriors(
            ref_posteriors, fit['posteriors'], fit['weights'], fit['trans_mat']
        )
        aligned_weights_list.append(w_aligned)
        aligned_trans_list.append(t_aligned)
        
    consistency_results[subj] = {
        'fits': fits_data,
        'aligned_weights': np.array(aligned_weights_list),
        'aligned_trans': np.array(aligned_trans_list),
        'best_fit_seed': best_fit['seed']
    }
    
    with open(consistency_save_path, 'wb') as f:
        pickle.dump(consistency_results, f)

# Summary Plot for Weight Consistency with Proper SEM Error Bars
subjs_to_plot = [s for s in sess_data_dict.keys() if s in consistency_results]
fig, axs = plt.subplots(len(subjs_to_plot), 1, figsize=(10, 3.5 * len(subjs_to_plot)), sharex=True)
if len(subjs_to_plot) == 1: 
    axs = [axs]

for idx, subj in enumerate(subjs_to_plot):
    res = consistency_results[subj]
    weights_arr = res['aligned_weights'] # Shape: (n_inits, K, C-1, input_dim)
    mean_weights = np.mean(weights_arr, axis=0)
    
    # Calculate proper Standard Error of the Mean (SEM) across initializations (ddof=1)
    n_runs = weights_arr.shape[0]
    sem_weights = np.std(weights_arr, axis=0, ddof=1) / np.sqrt(n_runs)
    
    for k in range(num_states_check):
        # Category 0 (Left vs Bail)
        axs[idx].errorbar(np.arange(input_dim) - 0.1, mean_weights[k, 0, :], yerr=sem_weights[k, 0, :], 
                          fmt='o-', capsize=3, label=f"State {k+1} Left vs Bail")
        # Category 1 (Right vs Bail)
        axs[idx].errorbar(np.arange(input_dim) + 0.1, mean_weights[k, 1, :], yerr=sem_weights[k, 1, :], 
                          fmt='s--', capsize=3, label=f"State {k+1} Right vs Bail")
                            
    axs[idx].set_title(f"Subj {subj} - Weight Consistency Across {n_inits_consistency} Initializations (Mean ± SEM)")
    axs[idx].set_ylabel("Log-Odds vs. Bail")
    axs[idx].set_xticks(range(input_dim))
    axs[idx].set_xticklabels(ssm_predictors, rotation=45, ha='right')
    axs[idx].axhline(0, color='k', alpha=0.3)
    axs[idx].grid(True, alpha=0.3)
    axs[idx].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize='small')

plt.tight_layout()
plt.show()

#%% Automatically Populate best_models_dict for K = 1, 2, 3, 4 (With Caching)
best_models_save_path = 'glm_hmm_best_models.pkl'
k_states_to_fit = [1, 2, 3, 4]

# Ensure global variables are set if not already defined above:
n_inits_criteria = 5   # Example: Adjust to your preferred number of initializations
n_inits_cv = 3         # Example: Number of initializations for cross-validation
num_states_to_test = [1, 2, 3, 4]

# 1. Load from cache if regen_params is False and file exists
if not regen_params and os.path.exists(best_models_save_path):
    with open(best_models_save_path, 'rb') as f:
        best_models_dict = pickle.load(f)
    print(f"Loaded existing best_models_dict from {best_models_save_path}.")

else:
    print("Fitting/Extracting best models across K = 1, 2, 3, 4 for all subjects...")
    best_models_dict = {}

    for subj, res in consistency_results.items():
        best_models_dict[subj] = {}
        
        # Grab K=2 best fit from Section 1 consistency runs
        best_fit_k2 = max(res['fits'], key=lambda x: x['final_ll'])
        best_models_dict[subj][2] = best_fit_k2['model']
        
        # Prepare data arrays
        subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in sess_data_dict[subj]]
        subj_choices = [sess_data_dict[subj][s]['choices'] for s in sess_data_dict[subj]]
        
        # Fit models for missing K values
        for k in k_states_to_fit:
            if k == 2:
                continue  # Already saved from Section 1
                
            print(f"  -> Fitting full-data model for Subject {subj} | K={k}...")
            best_ll = -np.inf
            best_k_model = None
            
            for init_seed in range(n_inits_criteria):
                np.random.seed(init_seed * 100 + 7)
                model = ssm.HMM(
                    k, 1, input_dim,
                    observations="input_driven_obs",
                    observation_kwargs=dict(C=num_categories),
                    transitions="standard"
                )
                train_lls = model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
                
                if train_lls[-1] > best_ll:
                    best_ll = train_lls[-1]
                    best_k_model = model
                    
            best_models_dict[subj][k] = best_k_model

    # 2. Save fitted models to disk
    with open(best_models_save_path, 'wb') as f:
        pickle.dump(best_models_dict, f)
    print(f"Saved all K=1..4 best models to {best_models_save_path}.")

#%% SECTION 2: Multi-Run Cross-Validation Loop
print("\n" + "="*60)
print("SECTION 2: MULTI-RUN LEAVE-ONE-OUT CROSS-VALIDATION")
print("="*60)

# Check cache while respecting the regen_params flag
if not regen_params and os.path.exists(cv_save_path):
    with open(cv_save_path, 'rb') as f:
        cv_results = pickle.load(f)
    print("Loaded existing multi-run CV results. Resuming...")
else:
    cv_results = {}
    if regen_params:
        print("regen_params is True. Forcing a clean re-fit of all cross-validation models...")
    else:
        print("No cache found. Starting fresh...")

subjects = list(sess_data_dict.keys())

for subj_id in subjects:
    if subj_id not in cv_results:
        cv_results[subj_id] = {}

    sessions = list(sess_data_dict[subj_id].keys())
    
    # Pre-extract session data once per subject
    all_inputs = [sess_data_dict[subj_id][s]['inputs'] for s in sessions]
    all_choices = [sess_data_dict[subj_id][s]['choices'] for s in sessions]
    input_dim = all_inputs[0].shape[1]

    for num_states in num_states_to_test:
        state_key = f"{num_states}_states"
        if state_key not in cv_results[subj_id]:
            cv_results[subj_id][state_key] = {}

        # Correct degrees of freedom calculation
        k_initial = num_states - 1
        k_transitions = num_states * (num_states - 1)
        k_glm_weights = num_states * input_dim * (num_categories - 1)
        k_params = k_initial + k_transitions + k_glm_weights

        # Leave-One-Out Cross-Validation (LOOCV)
        for s_idx, test_sess in enumerate(sessions):
            if test_sess not in cv_results[subj_id][state_key]:
                cv_results[subj_id][state_key][test_sess] = []

            existing_inits = len(cv_results[subj_id][state_key][test_sess])
            if existing_inits >= n_inits_cv:
                continue

            print(f"Subj: {subj_id} | States: {num_states} | Fold (Held-out): {test_sess} | Running inits {existing_inits+1}..{n_inits_cv}")

            # Fast LOOCV Split using list slicing
            train_inputs = all_inputs[:s_idx] + all_inputs[s_idx+1:]
            train_choices = all_choices[:s_idx] + all_choices[s_idx+1:]
            test_inputs = [all_inputs[s_idx]]
            test_choices = [all_choices[s_idx]]

            n_train_trials = sum(len(c) for c in train_choices)
            n_test_trials = len(test_choices[0])

            for init_idx in range(existing_inits, n_inits_cv):
                np.random.seed(init_idx * 50 + 13)
                
                # Initialize GLM-HMM
                model = ssm.HMM(num_states, 1, input_dim, 
                                observations="input_driven_obs", 
                                observation_kwargs=dict(C=num_categories), 
                                transitions="standard")

                # Fit the model on training data using tolerance and increased max iterations
                train_lls = model.fit(train_choices, inputs=train_inputs, method="em", num_iters=1000, tolerance=tol)
                
                final_train_ll = train_lls[-1]
                test_ll = model.log_likelihood(test_choices, inputs=test_inputs)
                
                # Test BIC and Test AIC using corrected k_params
                test_bic = k_params * np.log(n_test_trials) - 2 * test_ll
                test_aic = 2 * k_params - 2 * test_ll

                run_data = {
                    'init_idx': init_idx,
                    'train_ll': float(final_train_ll),
                    'test_ll': float(test_ll),
                    'test_ll_per_trial': float(test_ll / n_test_trials),
                    'bic': float(test_bic),
                    'aic': float(test_aic),
                    'n_train_trials': int(n_train_trials),
                    'n_test_trials': int(n_test_trials)
                }

                cv_results[subj_id][state_key][test_sess].append(run_data)

            # Dump to disk incrementally
            with open(cv_save_path, 'wb') as f:
                pickle.dump(cv_results, f)

print("Cross-validation model fitting complete.")

#%% SECTION 3: Cross-Validation Aggregation & Visualization
print("\n" + "="*60)
print("SECTION 3: AGGREGATING CV RESULTS & PLOTTING")
print("="*60)

# Parse raw dictionary into flat DataFrame
cv_rows = [
    {
        'subject': subj_id,
        'num_states': int(state_key.split('_')[0]),
        'session': sess_id,
        'init_idx': run_info['init_idx'],
        'train_ll': run_info['train_ll'],
        'test_ll': run_info['test_ll'],
        'test_ll_per_trial': run_info['test_ll_per_trial'],
        'bic': run_info['bic'],
        'aic': run_info.get('aic', np.nan),
        'n_test_trials': run_info['n_test_trials']
    }
    for subj_id, states_dict in cv_results.items()
    for state_key, sessions_dict in states_dict.items()
    for sess_id, inits_list in sessions_dict.items()
    for run_info in inits_list
]

df_cv = pd.DataFrame(cv_rows)

if df_cv.empty:
    print("No cross-validation results found to aggregate.")
else:
    # Method 1: Best init per fold, then calculate Mean and SEM across folds (N = number of sessions)
    best_runs_per_fold = df_cv.loc[df_cv.groupby(['subject', 'num_states', 'session'])['test_ll'].idxmax()]

    summary_best_model = best_runs_per_fold.groupby(['subject', 'num_states']).agg(
        test_ll_mean=('test_ll_per_trial', 'mean'),
        test_ll_sem=('test_ll_per_trial', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        bic_mean=('bic', 'mean'),
        bic_sem=('bic', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        aic_mean=('aic', 'mean'),
        aic_sem=('aic', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)
    ).reset_index()

    # Method 2: Average across initializations *first* per fold, then calculate Mean and SEM across folds
    avg_per_fold = df_cv.groupby(['subject', 'num_states', 'session']).agg(
        test_ll_per_trial=('test_ll_per_trial', 'mean'),
        bic=('bic', 'mean'),
        aic=('aic', 'mean')
    ).reset_index()

    summary_avg_all = avg_per_fold.groupby(['subject', 'num_states']).agg(
        test_ll_mean=('test_ll_per_trial', 'mean'),
        test_ll_sem=('test_ll_per_trial', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        bic_mean=('bic', 'mean'),
        bic_sem=('bic', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0),
        aic_mean=('aic', 'mean'),
        aic_sem=('aic', lambda x: np.std(x, ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)
    ).reset_index()

    # Filter plotting strictly to subjects present in the aggregated results
    subjects_to_plot = [s for s in subjects if s in df_cv['subject'].unique()]

    fig, axs = plt.subplots(len(subjects_to_plot), 3, figsize=(18, 3.5 * len(subjects_to_plot)), sharex=True, squeeze=False)

    for idx, subj in enumerate(subjects_to_plot):
        subj_best = summary_best_model[summary_best_model['subject'] == subj].sort_values('num_states')
        subj_avg = summary_avg_all[summary_avg_all['subject'] == subj].sort_values('num_states')
        states = subj_best['num_states']
        
        # ----------------------------------------------------
        # Column 1: Test Log-Likelihood per Trial (Higher is better)
        # ----------------------------------------------------
        axs[idx, 0].errorbar(states, subj_best['test_ll_mean'], yerr=subj_best['test_ll_sem'], fmt='o-', lw=2, capsize=4, label="Best Init/Fold (Mean±SEM)")
        axs[idx, 0].errorbar(states, subj_avg['test_ll_mean'], yerr=subj_avg['test_ll_sem'], fmt='s--', lw=1.5, alpha=0.7, capsize=4, label="Avg across All Inits")
        axs[idx, 0].set_title(f"Subj {subj} - Test LL / Trial")
        axs[idx, 0].set_ylabel("Normalized Test LL")
        axs[idx, 0].set_xticks(num_states_to_test)
        axs[idx, 0].grid(True, alpha=0.3)
        axs[idx, 0].legend(fontsize='x-small', loc='lower left')
        
        # ----------------------------------------------------
        # Column 2: Test BIC Score (Lower is better)
        # ----------------------------------------------------
        axs[idx, 1].errorbar(states, subj_best['bic_mean'], yerr=subj_best['bic_sem'], fmt='o-', lw=2, color='red', capsize=4, label="Best Init/Fold (Mean±SEM)")
        axs[idx, 1].errorbar(states, subj_avg['bic_mean'], yerr=subj_avg['bic_sem'], fmt='s--', lw=1.5, alpha=0.7, color='mediumseagreen', capsize=4, label="Avg across All Inits")
        axs[idx, 1].set_title(f"Subj {subj} - Test BIC Score")
        axs[idx, 1].set_ylabel("Mean Test BIC")
        axs[idx, 1].set_xticks(num_states_to_test)
        axs[idx, 1].grid(True, alpha=0.3)
        axs[idx, 1].legend(fontsize='x-small', loc='upper left')

        # ----------------------------------------------------
        # Column 3: Test AIC Score (Lower is better)
        # ----------------------------------------------------
        axs[idx, 2].errorbar(states, subj_best['aic_mean'], yerr=subj_best['aic_sem'], fmt='s-', lw=2, color='purple', capsize=4, label="Best Init/Fold (Mean±SEM)")
        axs[idx, 2].errorbar(states, subj_avg['aic_mean'], yerr=subj_avg['aic_sem'], fmt='o--', lw=1.5, alpha=0.7, color='gold', capsize=4, label="Avg across All Inits")
        axs[idx, 2].set_title(f"Subj {subj} - Test AIC Score")
        axs[idx, 2].set_ylabel("Mean Test AIC")
        axs[idx, 2].set_xticks(num_states_to_test)
        axs[idx, 2].grid(True, alpha=0.3)
        axs[idx, 2].legend(fontsize='x-small', loc='upper left')

    # Label formatting for bottom row
    for col in range(3):
        axs[-1, col].set_xlabel("Number of Latent States")

    plt.tight_layout()
    plt.show()
    
#%% Full-Dataset Training LL (Normalized by Trial), BIC, and AIC Calculation & Visualization

full_metrics_save_path = 'glm_hmm_full_metrics.pkl'

# Pull state configuration and upstream variables safely from session state
num_states_to_test = globals().get('num_states_to_test', [1, 2, 3, 4])
n_inits_criteria = globals().get('n_inits_criteria', 20)

# Maintain persistent model dictionary across script blocks if already defined
if 'best_models_dict' not in globals():
    best_models_dict = {}

# Check cache while respecting the regen_params flag
if not regen_params and os.path.exists(full_metrics_save_path):
    with open(full_metrics_save_path, 'rb') as f:
        full_metrics_results = pickle.load(f)
    print("Loaded existing full-dataset metrics results from cache. Skipping re-fit...")
else:
    full_metrics_results = {}
    if regen_params:
        print("regen_params is True. Forcing a clean re-fit of full-dataset metrics...")
    else:
        print("No full-metrics cache found. Starting fresh...")

    for subj in sess_data_dict.keys():
        print(f"Calculating criteria and Training LL with multi-init for Subject {subj}...")
        
        subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in sess_data_dict[subj]]
        subj_choices = [sess_data_dict[subj][s]['choices'] for s in sess_data_dict[subj]]
        n_total_trials = sum(len(c) for c in subj_choices)
        
        subj_seed_offset = int(subj) if str(subj).isdigit() else abs(hash(str(subj))) % 1000
        
        full_metrics_results[subj] = {st: {'train_ll_per_trials': [], 'bics': [], 'aics': []} for st in num_states_to_test}
        best_models_dict[subj] = best_models_dict.get(subj, {})
        
        for num_states in num_states_to_test:
            best_ll_for_state = -np.inf
            best_model_for_state = None
            
            for init_idx in range(n_inits_criteria):
                if 'consistency_results' in globals() and subj in consistency_results:
                    base_seed = consistency_results[subj].get('best_fit_seed', 42) * 100
                else:
                    base_seed = 42 + subj_seed_offset * 10
                    
                np.random.seed(base_seed + init_idx)
                
                model = ssm.HMM(num_states, 1, len(ssm_predictors), 
                                observations="input_driven_obs", 
                                observation_kwargs=dict(C=3), 
                                transitions="standard")
                model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
                full_ll = model.log_likelihood(subj_choices, inputs=subj_inputs)
                full_ll_per_trial = full_ll / n_total_trials  # Normalized per trial
                
                if full_ll > best_ll_for_state:
                    best_ll_for_state = full_ll
                    best_model_for_state = model
                
                # Corrected degrees of freedom including initial state probabilities
                k_initial = num_states - 1
                k_trans = num_states * (num_states - 1)
                k_glm = num_states * len(ssm_predictors) * 2
                k_params = k_initial + k_trans + k_glm
                
                full_bic = k_params * np.log(n_total_trials) - 2 * full_ll
                full_aic = 2 * k_params - 2 * full_ll
                
                full_metrics_results[subj][num_states]['train_ll_per_trials'].append(full_ll_per_trial)
                full_metrics_results[subj][num_states]['bics'].append(full_bic)
                full_metrics_results[subj][num_states]['aics'].append(full_aic)
            
            best_models_dict[subj][num_states] = best_model_for_state

        # Dump incrementally to disk per subject
        with open(full_metrics_save_path, 'wb') as f:
            pickle.dump(full_metrics_results, f)

    print("Multi-init Training LL (per trial), BIC, AIC calculated and saved to cache!")

# Summarize metrics across initializations (using SEM instead of SD)
summary_rows = []
for s in full_metrics_results:
    for st in full_metrics_results[s]:
        data = full_metrics_results[s][st]
        n_inits = len(data['train_ll_per_trials'])
        
        summary_rows.append({
            'Subject': str(s), 'States': st, 
            'Best_Train_LL': np.max(data['train_ll_per_trials']), 
            'Mean_Train_LL': np.mean(data['train_ll_per_trials']), 
            'Sem_Train_LL': np.std(data['train_ll_per_trials'], ddof=1) / np.sqrt(n_inits) if n_inits > 1 else 0.0,
            'Best_BIC': np.min(data['bics']), 
            'Mean_BIC': np.mean(data['bics']), 
            'Sem_BIC': np.std(data['bics'], ddof=1) / np.sqrt(n_inits) if n_inits > 1 else 0.0,
            'Best_AIC': np.min(data['aics']), 
            'Mean_AIC': np.mean(data['aics']), 
            'Sem_AIC': np.std(data['aics'], ddof=1) / np.sqrt(n_inits) if n_inits > 1 else 0.0,
        })

df_full_metrics = pd.DataFrame(summary_rows)
subjects_list = df_full_metrics['Subject'].unique()

# Plot 3-column figure grid per subject
fig, axs = plt.subplots(len(subjects_list), 3, figsize=(18, 3.2 * len(subjects_list)), sharex=True, squeeze=False)

for idx, subj in enumerate(subjects_list):
    subj_df = df_full_metrics[df_full_metrics['Subject'] == subj].sort_values('States')
    states = subj_df['States']
    
    # 1. Training Log-Likelihood per Trial
    axs[idx, 0].plot(states, subj_df['Best_Train_LL'], 'o-', lw=2, label='Best Train LL / Trial')
    axs[idx, 0].errorbar(states, subj_df['Mean_Train_LL'], yerr=subj_df['Sem_Train_LL'], fmt='s--', lw=1.5, capsize=4, label='Mean ± SEM')
    axs[idx, 0].set_title(f"Subj {subj} - Train LL / Trial (Higher is Better)", fontweight='bold')
    axs[idx, 0].set_ylabel("Normalized Train LL")
    axs[idx, 0].set_xticks(num_states_to_test)
    axs[idx, 0].grid(True, alpha=0.3)
    axs[idx, 0].legend(loc='lower right', fontsize='x-small')
    
    # 2. Bayesian Information Criterion (BIC)
    axs[idx, 1].plot(states, subj_df['Best_BIC'], 'o-', lw=2, color='red', label='Best BIC')
    axs[idx, 1].errorbar(states, subj_df['Mean_BIC'], yerr=subj_df['Sem_BIC'], fmt='s--', lw=1.5, color='mediumseagreen', capsize=4, label='Mean ± SEM')
    axs[idx, 1].set_title(f"Subj {subj} - BIC (Lower is Better)", fontweight='bold')
    axs[idx, 1].set_ylabel("BIC Score")
    axs[idx, 1].set_xticks(num_states_to_test)
    axs[idx, 1].grid(True, alpha=0.3)
    axs[idx, 1].legend(loc='upper right', fontsize='x-small')
    
    # 3. Akaike Information Criterion (AIC)
    axs[idx, 2].plot(states, subj_df['Best_AIC'], 's-', lw=2, color='purple', label='Best AIC')
    axs[idx, 2].errorbar(states, subj_df['Mean_AIC'], yerr=subj_df['Sem_AIC'], fmt='o--', lw=1.5, color='gold', capsize=4, label='Mean ± SEM')
    axs[idx, 2].set_title(f"Subj {subj} - AIC (Lower is Better)", fontweight='bold')
    axs[idx, 2].set_ylabel("AIC Score")
    axs[idx, 2].set_xticks(num_states_to_test)
    axs[idx, 2].grid(True, alpha=0.3)
    axs[idx, 2].legend(loc='upper right', fontsize='x-small')

for col in range(3):
    axs[-1, col].set_xlabel("Number of Latent States")

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

#%% SECTION 4: Final Model Extraction & Posteriors (Fixed K for Initial Inspection)
print("\n" + "="*60)
print("SECTION 4: EXTRACTING POSTERIOR PROBABILITIES FOR FINAL ANALYSIS")
print("="*60)

final_num_states = 2 

final_subject_dfs = {}
obs_dim = 1
num_categories = 3
input_dim = len(ssm_predictors)

for subj in sess_data_dict.keys():
    print(f"\nProcessing Final Posteriors for Subject {subj}...")
    print(f"  -> Using fixed K = {final_num_states} for initial inspection.")
    
    sess_order = list(sess_data_dict[subj].keys())
    subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in sess_order]
    subj_choices = [sess_data_dict[subj][s]['choices'] for s in sess_order]
    
    # Check if we can reuse the best model already fitted in Section 3
    if 'best_models_dict' in globals() and subj in best_models_dict and final_num_states in best_models_dict[subj] and best_models_dict[subj][final_num_states] is not None:
        print(f"  -> Reusing pre-fitted best model from 'best_models_dict' for Subject {subj}.")
        final_model = best_models_dict[subj][final_num_states]
    else:
        print(f"  -> Fitting final model on all sessions from scratch...")
        # Pick seed from Section 1 if applicable
        if subj in consistency_results and final_num_states == num_states_check:
            best_seed = consistency_results[subj]['best_fit_seed']
            np.random.seed(best_seed * 100 + 7)
        else:
            np.random.seed(42)
            
        final_model = ssm.HMM(final_num_states, obs_dim, input_dim, 
                            observations="input_driven_obs", 
                            observation_kwargs=dict(C=num_categories), 
                            transitions="standard")
        final_model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
    
    # Extract transition matrix safely using the built-in property
    trans_mat = final_model.transitions.transition_matrix.copy()
    
    # Extract trial posteriors per session safely
    posterior_list = []
    for c, inp in zip(subj_choices, subj_inputs):
        res = final_model.expected_states(c, input=inp)
        post = res[0] if isinstance(res, tuple) else res
        posterior_list.append(post)
        
    concat_posteriors = np.vstack(posterior_list)
    weights = final_model.observations.params.copy()
    
    # Align states back to Section 1 reference fit to prevent state-label flipping
    if subj in consistency_results and final_num_states == num_states_check:
        ref_posteriors = consistency_results[subj]['fits'][consistency_results[subj]['best_fit_seed']]['posteriors']
        weights, trans_mat, concat_posteriors, _ = align_states_by_posteriors(
            ref_posteriors, concat_posteriors, weights, trans_mat
        )

    # Plot & inspect transition matrix with 1-indexed labels ('State 1', 'State 2')
    fig_tm, ax_tm = plt.subplots(figsize=(5, 4))
    sns.heatmap(trans_mat, annot=True, fmt=".3f", cmap="Blues", vmin=0, vmax=1, cbar=True,
                xticklabels=['State 1', 'State 2'], 
                yticklabels=['State 1', 'State 2'], 
                ax=ax_tm)
    ax_tm.set_title(f"Transition Matrix - Subj {subj} (K={final_num_states})")
    ax_tm.set_xlabel("To State")
    ax_tm.set_ylabel("From State")
    plt.tight_layout()
    plt.show()
    
    # Build dataframe with state probabilities
    subj_df = reg_ready_df[reg_ready_df['subjid'] == subj].copy()
    
    # Sanity check to ensure trial count matches expectations
    assert len(subj_df) == len(concat_posteriors), \
        f"Row mismatch for Subject {subj}: reg_ready_df has {len(subj_df)} trials, but model produced {len(concat_posteriors)} posteriors."
    
    for k in range(final_num_states):
        subj_df[f'state_{k+1}_prob'] = concat_posteriors[:, k]
        
    # Fast vectorized state assignment (1-indexed)
    subj_df['assigned_state'] = np.argmax(concat_posteriors, axis=1) + 1
    final_subject_dfs[subj] = subj_df

# Combine all subjects into one master dataframe and save
master_posterior_df = pd.concat(final_subject_dfs.values())
master_posterior_df.to_csv('glm_hmm_state_posteriors.csv', index=False)
print("\nSaved 'glm_hmm_state_posteriors.csv' successfully.")

#%% SECTION 5: Post-Hoc Analysis: State-Specific Behavioral Rates & Psychometrics
print("\n" + "="*60)
print("RUNNING COMPARATIVE STATE-SPECIFIC PSYCHOMETRIC ANALYSIS")
print("="*60)

# Global configuration
rate_columns = ['tone_info_str', 'delay_bin', ['tone_info_str', 'delay_bin']]

# Ensure final_num_states is defined (default to 2 if not set)
final_num_states = globals().get('final_num_states', 2)

# 1. Ensure outcome booleans & compute continuous session lags BEFORE state slicing
df = master_posterior_df.copy()
df['is_bail'] = df['bail'].astype(int) if 'bail' in df.columns else (df['outcome'] == 0).astype(int)
df['is_choice'] = 1 - df['is_bail']

# Compute contiguous trial-to-trial lag features per subject and session safely
sess_col = 'sessid' if 'sessid' in df.columns else 'session'
df_grouped = df.groupby(['subjid', sess_col])

df['prev_choice'] = df_grouped['choice'].shift(1)
df['prev_hit'] = df_grouped['hit'].shift(1)
df['prev_bail'] = df_grouped['bail'].shift(1)
df['prev_stim'] = df_grouped['relevant_tone_info'].shift(1)

# Subject State Summary Table (safely handles missing curr_choice_val columns)
agg_dict = {
    'is_bail': 'mean',
    'is_choice': 'mean'
}
if 'curr_choice_val' in df.columns:
    agg_dict['curr_choice_val'] = lambda x: x[x != 0].mean() if (x != 0).any() else np.nan

state_summary = df.groupby(['subjid', 'assigned_state']).agg(agg_dict)
if 'curr_choice_val' in agg_dict:
    state_summary.columns = ['Bail Rate', 'Engagement Rate', 'Rightward Bias (Choice Trials)']
else:
    state_summary.columns = ['Bail Rate', 'Engagement Rate']

print("\nSUBJECT-SPECIFIC STATE INTERPRETATION SUMMARY:")
print(state_summary.round(3))

# Vectorized rate helper: num_condition / denom_condition
def calc_rate(sub_df, num_mask, denom_mask):
    denom = denom_mask.sum()
    return (num_mask & denom_mask).sum() / denom if denom > 0 else 0.0

# 2. Comparative Post-Hoc State Loop
for subj_id in df['subjid'].unique():
    subj_df = df[df['subjid'] == subj_id].copy()
    total_subj_trials = len(subj_df)
    if total_subj_trials == 0:
        continue

    states_data = {}
    
    for s in range(1, final_num_states + 1):
        state_sess = subj_df[subj_df['assigned_state'] == s]
        n_trials = len(state_sess)
        if n_trials < 10:
            continue

        prop = (n_trials / total_subj_trials) * 100
        state_no_bails = state_sess[(state_sess['bail'] == False) & (state_sess['choice'] != 'none')]
        state_tone_heard = state_sess[state_sess['cpoke_out_time'] > state_sess['abs_tone_start_times']]

        # Define boolean masks for trial t evaluating lag t-1
        valid_choice = (state_sess['bail'] == False) & (state_sess['choice'] != 'none')
        c_r = (state_sess['choice'] == 'right') & valid_choice
        p_r = (state_sess['prev_choice'] == 'right')
        p_l = (state_sess['prev_choice'] == 'left')
        has_prev_choice = state_sess['prev_choice'].isin(['left', 'right']) & valid_choice
        
        p_hit = (state_sess['prev_hit'] == True)
        p_miss = (state_sess['prev_hit'] == False) & (state_sess['prev_bail'] == False)
        stay = (state_sess['choice'] == state_sess['prev_choice']) & has_prev_choice
        
        p_bail = (state_sess['prev_bail'] == True)
        is_b = (state_sess['bail'] == True)
        is_hit = (state_sess['hit'] == True) & valid_choice
        
        same_stim = (state_sess['relevant_tone_info'] == state_sess['prev_stim']) & has_prev_choice
        diff_stim = (state_sess['relevant_tone_info'] != state_sess['prev_stim']) & has_prev_choice

        # Calculate contiguous state history metrics
        probs = [
            calc_rate(state_sess, c_r, p_r & has_prev_choice),       # right_pr
            calc_rate(state_sess, c_r, p_l & has_prev_choice),       # right_pl
            calc_rate(state_sess, stay, has_prev_choice),            # repeat
            calc_rate(state_sess, stay, p_hit & has_prev_choice),    # win_stay
            calc_rate(state_sess, ~stay, p_miss & has_prev_choice),  # lose_switch
            calc_rate(state_sess, is_hit, same_stim),                # hit_ps_same
            calc_rate(state_sess, is_hit, same_stim & p_hit),        # hit_ps_same_c
            calc_rate(state_sess, is_hit, same_stim & p_miss),       # hit_ps_same_i
            calc_rate(state_sess, is_hit, diff_stim),                # hit_ps_diff
            calc_rate(state_sess, is_hit, diff_stim & p_hit),        # hit_ps_diff_c
            calc_rate(state_sess, is_hit, diff_stim & p_miss),       # hit_ps_diff_i
            calc_rate(state_sess, is_hit, p_bail),                   # hit_pb
            calc_rate(state_sess, is_hit, p_bail & same_stim),       # hit_pb_same
            calc_rate(state_sess, is_hit, p_bail & diff_stim),       # hit_pb_diff
            calc_rate(state_sess, is_b, p_bail),                     # bail_pb
            calc_rate(state_sess, is_b, p_miss),                     # bail_pi
            calc_rate(state_sess, is_b, p_hit)                       # bail_pc
        ]

        avg_hit = state_no_bails['hit'].mean() if len(state_no_bails) > 0 else 0.0

        states_data[s] = {
            'hit_map': bah.get_rate_dict(state_no_bails, 'hit', rate_columns),
            'bail_map': bah.get_rate_dict(state_tone_heard, 'bail', rate_columns),
            'probs': probs,
            'avg_hit': avg_hit,
            'proportion': prop,
            'n_trials': n_trials
        }

    if not states_data:
        continue

    # 3. Comparative Plot Setup
    fig = plt.figure(layout='constrained', figsize=(15, 3.5 * final_num_states + 3))
    gs = GridSpec(final_num_states + 1, final_num_states, figure=fig)
    
    labels = ['right_pr', 'right_pl', 'repeat', 'win_stay', 'lose_switch', 
              'hit_ps_same', 'hit_ps_same_c', 'hit_ps_same_i', 'hit_ps_diff', 
              'hit_ps_diff_c', 'hit_ps_diff_i', 'hit_pb', 'hit_pb_same', 
              'hit_pb_diff', 'bail_pb', 'bail_pi', 'bail_pc']
    colors = plt.cm.tab10(np.linspace(0, 1, final_num_states))

    valid_states = sorted([s for s in states_data.keys()])
    for idx, s in enumerate(valid_states):
        ax_h = fig.add_subplot(gs[0, idx])
        ax_b = fig.add_subplot(gs[1, idx])
        bah.plot_rate_heatmap(states_data[s]['hit_map'], 'delay_bin', 'Delay', 'tone_info_str', 'Tone', ax_h)
        bah.plot_rate_heatmap(states_data[s]['bail_map'], 'delay_bin', 'Delay', 'tone_info_str', 'Tone', ax_b)
        
        n = states_data[s]['n_trials']
        prop = states_data[s]['proportion']
        avg_hit = states_data[s]['avg_hit']
        
        ax_h.set_title(f"State {s} Hit Rate (Avg: {avg_hit:.2f})\n({prop:.1f}% of trials, N={n})")
        ax_b.set_title(f"State {s} Bail Rate")

    ax_p = fig.add_subplot(gs[2:, :])
    for idx, s in enumerate(valid_states):
        ax_p.plot(range(len(labels)), states_data[s]['probs'], 'o-', color=colors[idx], label=f"State {s}")
        ax_p.axhline(states_data[s]['avg_hit'], color=colors[idx], linestyle='--', alpha=0.6, 
                     label=f"State {s} Hit Rate ({states_data[s]['avg_hit']:.2f})")

    ax_p.set_xticks(range(len(labels)))
    ax_p.set_xticklabels(labels, rotation=45, ha='right')
    fig.suptitle(f"Comparative History Probabilities - Subj {subj_id}", fontsize=14, fontweight='bold')
    ax_p.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
#%% Label Meanings

'''
right_pr = P(right|prev right)
right_pl = P(right|prev left)
repeat = P(repeat choice)
win_stay = P(stay|prev hit)
lose_switch = P(switch|prev miss)
hit_ps_same = P(hit|prev stim same)
hit_ps_same_c = P(hit|prev stim same and prev hit)
hit_ps_same_i = P(hit|prev stim same and prev miss)
hit_ps_diff = P(hit|prev stim diff)
hit_ps_diff_c = P(hit|prev stim diff and prev hit)
hit_ps_diff_i = P(hit|prev stim diff and prev miss)
hit_pb = P(hit|prev bail)
hit_pb_same = P(hit|prev bail and same tone)
hit_pb_diff = P(hit|prev bail and diff tone)
bail_pb = P(bail|prev bail)
bail_pi = P(bail|prev miss)
bail_pc = P(bail|prev hit)
'''

#%% Measure label consistency across consistency initializations

'''
Currently, label consistency is performed this way:
  - 1st run is selected as fixed reference standard. Every other random initialization run is 
    aligned to this reference using the Hungarian algorithm (align_states_by_posteriors) 
    to ensure states match up semantically.
  - For every trial, look at the aligned posterior probabilities across all initialization runs, then
    extracts the discrete winning state (argmax) for each run. 
    This creates a matrix of shape (n_inits, n_trials).
  - For each individual trial, calculate the mode (the most frequent state label chosen across all initializations).
    Then, it counts what percentage of the runs agreed with that modal state.
  - Then, average those percentages across all trials to yield subject-level consistency score shown in the graph.
'''

consistency_trials = []
consistency_summary = {}

# Pull expected number of states safely from globals or default to 2
K_consistency = globals().get('num_states_check', 2)

for subj, res in consistency_results.items():
    n_inits = len(res['fits'])
    if n_inits < 2:
        print(f"Subj {subj}: Skipping (fewer than 2 initialization runs).")
        continue

    # Safety check: If 'posteriors' wasn't saved in the old pkl, recompute them on the fly
    if 'posteriors' not in res['fits'][0]:
        print(f"Recomputing missing posteriors for Subject {subj}...")
        subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in sess_data_dict[subj]]
        subj_choices = [sess_data_dict[subj][s]['choices'] for s in sess_data_dict[subj]]

        for fit in res['fits']:
            model = ssm.HMM(K_consistency, 1, len(ssm_predictors), 
                            observations="input_driven_obs", 
                            observation_kwargs=dict(C=3), 
                            transitions="standard")
            model.observations.params = fit['weights']
            
            # Clip transitions to prevent log(0) -> -inf errors during expected_states
            safe_trans = np.clip(fit['trans_mat'], 1e-12, 1.0)
            model.transitions.params = np.log(safe_trans)[None, ...]

            posteriors_list = []
            for c, inp in zip(subj_choices, subj_inputs):
                res_expected = model.expected_states(c, input=inp)
                post = res_expected[0] if isinstance(res_expected, tuple) else res_expected
                posteriors_list.append(post)
            fit['posteriors'] = np.vstack(posteriors_list)

    # Now safely run the alignment and consistency calculation
    ref_p = res['fits'][0]['posteriors']
    aligned_ps = []
    for fit in res['fits']:
        _, _, p_al, _ = align_states_by_posteriors(ref_p, fit['posteriors'], fit['weights'], fit['trans_mat'])
        aligned_ps.append(p_al)
    aligned_ps = np.array(aligned_ps)  # Shape: (n_inits, n_trials, K)

    # Get argmax state for each run
    argmax_states = np.argmax(aligned_ps, axis=2)  # Shape: (n_inits, n_trials)

    # Calculate percentage of runs that agree with the mode state label per trial (Generalized for any K)
    n_trials = argmax_states.shape[1]
    mode_counts = np.array([
        np.max(np.bincount(argmax_states[:, t], minlength=K_consistency)) 
        for t in range(n_trials)
    ])

    trial_consistency = (mode_counts / n_inits) * 100
    consistency_percentage = np.mean(trial_consistency)

    consistency_trials.append(consistency_percentage)
    consistency_summary[subj] = {
        'mean_consistency': consistency_percentage,
        'trial_consistency': trial_consistency
    }

    print(f"Subj {subj} - Mean Trial Label Consistency Across Inits: {consistency_percentage:.2f}%")
    
#%% Visualize Trial Label Consistency, State Posterior, & Choice Posterior Similarities

# Helper function to compute flattened cosine similarity
def calc_cos_sim(mat1, mat2):
    vec1 = mat1.flatten()
    vec2 = mat2.flatten()
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

print("\n" + "="*60)
print("PLOTTING CONSISTENCY & POSTERIOR SIMILARITIES ACROSS INITIALIZATIONS")
print("="*60)

consistency_plot_data = []
state_post_sim_data = []
choice_post_sim_data = []

# Pull expected number of states safely from globals or default to 2
K_consistency = globals().get('num_states_check', 2)

# Compute metrics dynamically for all subjects
for subj, res in consistency_results.items():
    n_inits = len(res['fits'])
    if n_inits < 2:
        continue

    # Safety check: If 'posteriors' wasn't saved in the old pkl, recompute them on the fly
    if 'posteriors' not in res['fits'][0]:
        subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in sess_data_dict[subj]]
        subj_choices = [sess_data_dict[subj][s]['choices'] for s in sess_data_dict[subj]]
        for fit in res['fits']:
            model = ssm.HMM(K_consistency, 1, len(ssm_predictors), 
                            observations="input_driven_obs", 
                            observation_kwargs=dict(C=3), 
                            transitions="standard")
            model.observations.params = fit['weights']
            
            # Clip transition matrix to avoid log(0) -> -inf
            safe_trans = np.clip(fit['trans_mat'], 1e-12, 1.0)
            model.transitions.params = np.log(safe_trans)[None, ...]
            
            posteriors_list = []
            for c, inp in zip(subj_choices, subj_inputs):
                res_expected = model.expected_states(c, input=inp)
                post = res_expected[0] if isinstance(res_expected, tuple) else res_expected
                posteriors_list.append(post)
            fit['posteriors'] = np.vstack(posteriors_list)

    ref_p = res['fits'][0]['posteriors']
    aligned_ps = []
    aligned_choice_posts = []
    
    subj_inputs_all = [sess_data_dict[subj][s]['inputs'] for s in sess_data_dict[subj]]
    cat_inputs = np.vstack(subj_inputs_all)

    for fit in res['fits']:
        w_al, t_al, p_al, _ = align_states_by_posteriors(ref_p, fit['posteriors'], fit['weights'], fit['trans_mat'])
        aligned_ps.append(p_al) # Shape: (n_trials, K)
        
        # Compute choice probabilities for this alignment/fit using einsum
        n_trials = cat_inputs.shape[0]
        K = w_al.shape[0]
        sub_logits = np.einsum('nd,kcd->nkc', cat_inputs, w_al) # (n_trials, K, C-1)
        full_logits = np.concatenate([sub_logits, np.zeros((n_trials, K, 1))], axis=2)
        exp_logits = np.exp(full_logits - np.max(full_logits, axis=2, keepdims=True))
        choice_probs_k = exp_logits / np.sum(exp_logits, axis=2, keepdims=True) # (n_trials, K, C)
        
        # Marginalize over states: sum(p(state) * p(choice | state))
        marginal_choice_prob = np.sum(p_al[:, :, None] * choice_probs_k, axis=1) # (n_trials, C)
        aligned_choice_posts.append(marginal_choice_prob)

    aligned_ps = np.array(aligned_ps) # Shape: (n_inits, n_trials, K)
    n_trials = aligned_ps.shape[1]
    
    # --- 1. Calculate Trial Label Consistency (Argmax Agreement) ---
    argmax_states = np.argmax(aligned_ps, axis=2) # Shape: (n_inits, n_trials)
    mode_counts = np.array([
        np.max(np.bincount(argmax_states[:, t], minlength=K_consistency)) 
        for t in range(n_trials)
    ])
    consistency_percentage = (np.mean(mode_counts / n_inits)) * 100
    consistency_plot_data.append({'Subject': str(subj), 'Consistency (%)': consistency_percentage})

    # --- 2. State Posterior Cosine Similarity ---
    state_sims = []
    for i, j in combinations(range(n_inits), 2):
        state_sims.append(calc_cos_sim(aligned_ps[i], aligned_ps[j]))
    mean_state_sim = np.mean(state_sims) if state_sims else 0.0
    state_post_sim_data.append({'Subject': str(subj), 'Cosine Similarity': mean_state_sim})

    # --- 3. Choice Posterior Cosine Similarity ---
    choice_sims = []
    for i, j in combinations(range(n_inits), 2):
        choice_sims.append(calc_cos_sim(aligned_choice_posts[i], aligned_choice_posts[j]))
    mean_choice_sim = np.mean(choice_sims) if choice_sims else 0.0
    choice_post_sim_data.append({'Subject': str(subj), 'Cosine Similarity': mean_choice_sim})

df_consistency = pd.DataFrame(consistency_plot_data)
df_state_sim = pd.DataFrame(state_post_sim_data)
df_choice_sim = pd.DataFrame(choice_post_sim_data)

# =====================================================================
# Graph 1: Trial Label Consistency
# =====================================================================
plt.figure(figsize=(10, 5))
ax1 = sns.barplot(data=df_consistency, x='Subject', y='Consistency (%)', hue='Subject', palette='crest', legend=False)

for p in ax1.patches:
    height = p.get_height()
    if height > 0:
        ax1.annotate(f'{height:.2f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')

plt.axhline(90.0, color='darkorange', linestyle='--', linewidth=1.5, label='90% Robustness Threshold')
plt.title("Trial Label Consistency Across Random Initializations (Hungarian Aligned)", fontsize=13, fontweight='bold')
plt.xlabel("Subject ID", fontsize=11)
plt.ylabel("Mean Trial State Agreement (%)", fontsize=11)
plt.ylim(0, 108)
plt.legend(loc='lower left', fontsize='small')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =====================================================================
# Graph 2: State Posteriors Cosine Similarity
# =====================================================================
plt.figure(figsize=(10, 5))
ax2 = sns.barplot(data=df_state_sim, x='Subject', y='Cosine Similarity', hue='Subject', palette='mako', legend=False)

for p in ax2.patches:
    height = p.get_height()
    if height > 0:
        ax2.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')

plt.title("State Posteriors Cosine Similarity Across Initializations", fontsize=13, fontweight='bold')
plt.xlabel("Subject ID", fontsize=11)
plt.ylabel("Mean Cosine Similarity", fontsize=11)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# =====================================================================
# Graph 3: Choice Posteriors Cosine Similarity
# =====================================================================
plt.figure(figsize=(10, 5))
ax3 = sns.barplot(data=df_choice_sim, x='Subject', y='Cosine Similarity', hue='Subject', palette='magma', legend=False)

for p in ax3.patches:
    height = p.get_height()
    if height > 0:
        ax3.annotate(f'{height:.3f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontsize=8.5, fontweight='bold',
                    xytext=(0, 3), textcoords='offset points')

plt.title("Choice Posteriors Cosine Similarity Across Initializations", fontsize=13, fontweight='bold')
plt.xlabel("Subject ID", fontsize=11)
plt.ylabel("Mean Cosine Similarity", fontsize=11)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

#%% Visualize 2-State HMM Model Choice Prediction Hit Rate & Outcome Breakdown

# Visualize Distribution of Trial-Level Agreements as Grouped Bars Per Subject

accuracy_records_2state = []
eval_num_states = 2

for subj, df in final_subject_dfs.items():
    subj_sessions = sess_data_dict[subj]
    subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in subj_sessions]
    subj_choices = [sess_data_dict[subj][s]['choices'] for s in subj_sessions]
    
    # Check if we can reuse the pre-fitted model from Section 3 or 4
    if 'best_models_dict' in globals() and subj in best_models_dict and eval_num_states in best_models_dict[subj] and best_models_dict[subj][eval_num_states] is not None:
        print(f"Reusing pre-fitted {eval_num_states}-state model for Subject {subj}...")
        model = best_models_dict[subj][eval_num_states]
    else:
        print(f"Fitting {eval_num_states}-state model for Subject {subj} from scratch...")
        if subj in consistency_results:
            best_seed = consistency_results[subj]['best_fit_seed']
            np.random.seed(best_seed * 100 + 7)
        else:
            np.random.seed(42)
            
        model = ssm.HMM(eval_num_states, 1, len(ssm_predictors), 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=3), 
                        transitions="standard")
        model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
    
    all_pred_choices = []
    all_true_choices = []
    
    for sess_id, sess_data in subj_sessions.items():
        inp = sess_data['inputs']            # Shape: (n_trials, input_dim)
        raw_choices = sess_data['choices']    # Shape: (n_trials,) or (n_trials, 1)
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choice = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices])
        else:
            true_choice = np.array(raw_choices).astype(int).reshape(-1, 1)

        res_expected = model.expected_states(true_choice, input=inp)
        posterior_states = res_expected[0] if isinstance(res_expected, tuple) else res_expected  # Shape: (n_trials, K)
        
        n_trials = len(true_choice)
        p_cat_given_state = np.zeros((n_trials, model.K, 3))
        
        # Vectorized probability calculation per state across all session trials
        for k in range(model.K):
            w_k = model.observations.params[k]    # Shape: (2, input_dim)
            logits_non_base = inp @ w_k.T        # Shape: (n_trials, 2)
            logits = np.hstack([logits_non_base, np.zeros((n_trials, 1))])  # Append baseline 0.0
            
            # Softmax across category axis
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            p_cat_given_state[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
        # Marginalize over state posteriors: P(choice) = sum_k P(state_k) * P(choice | state_k)
        trial_probs = np.sum(posterior_states[:, :, None] * p_cat_given_state, axis=1) # Shape: (n_trials, 3)
            
        pred_choice = np.argmax(trial_probs, axis=1)
        all_pred_choices.extend(pred_choice)
        all_true_choices.extend(true_choice.flatten())
        
    all_pred = np.array(all_pred_choices)
    all_true = np.array(all_true_choices)
    
    overall_hr = np.mean(all_pred == all_true) * 100
    accuracy_records_2state.append({'Subject': str(subj), 'Category': 'Overall', 'Hit Rate (%)': overall_hr})
    
    # Category mapping: 0=Left, 1=Right, 2=Bail
    outcome_names = {0: 'Left', 1: 'Right', 2: 'Bail'}
    for cat_val, cat_name in outcome_names.items():
        mask = (all_true == cat_val)
        cat_hr = np.mean(all_pred[mask] == all_true[mask]) * 100 if np.sum(mask) > 0 else 0.0
        accuracy_records_2state.append({'Subject': str(subj), 'Category': cat_name, 'Hit Rate (%)': cat_hr})

df_accuracy_2state = pd.DataFrame(accuracy_records_2state)

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_accuracy_2state, x='Subject', y='Hit Rate (%)', hue='Category', palette='Set2')
plt.axhline(33.33, color='red', linestyle='--', linewidth=1.5, label='Chance Level (33.3%)')

plt.title(f"{eval_num_states}-State HMM Model: Choice Prediction Hit Rate by Subject and Outcome Category", fontsize=14, fontweight='bold')
plt.xlabel("Subject ID", fontsize=12)
plt.ylabel("Hit Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Metric / Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='solid', alpha=0.5)

plt.tight_layout()
plt.show()
    
#%% Visualize 1-State Model Choice Prediction Hit Rate & Outcome Breakdown

accuracy_records_1state = []
eval_num_states_1s = 1

for subj, df in final_subject_dfs.items():
    subj_sessions = sess_data_dict[subj]
    subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in subj_sessions]
    subj_choices = [sess_data_dict[subj][s]['choices'] for s in subj_sessions]
    
    # Check if we can reuse the pre-fitted 1-state model from Section 3
    if 'best_models_dict' in globals() and subj in best_models_dict and eval_num_states_1s in best_models_dict[subj] and best_models_dict[subj][eval_num_states_1s] is not None:
        print(f"Reusing pre-fitted 1-state model for Subject {subj}...")
        model_1s = best_models_dict[subj][eval_num_states_1s]
    else:
        print(f"Fitting 1-state model for Subject {subj} from scratch...")
        np.random.seed(42)
        model_1s = ssm.HMM(eval_num_states_1s, 1, len(ssm_predictors), 
                           observations="input_driven_obs", 
                           observation_kwargs=dict(C=3), 
                           transitions="standard")
        model_1s.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
    
    all_pred_choices = []
    all_true_choices = []
    
    for sess_id, sess_data in subj_sessions.items():
        inp = sess_data['inputs']            # Shape: (n_trials, input_dim)
        raw_choices = sess_data['choices']    # Shape: (n_trials,) or (n_trials, 1)
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choice = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices])
        else:
            true_choice = np.array(raw_choices).astype(int).flatten()

        n_trials = len(true_choice)
        
        # Vectorized multinomial logit calculation for K=1 state
        w_0 = model_1s.observations.params[0]                    # Shape: (2, input_dim)
        logits_non_base = inp @ w_0.T                            # Shape: (n_trials, 2)
        logits = np.hstack([logits_non_base, np.zeros((n_trials, 1))]) # Append 0.0 for baseline choice category
        
        # Softmax across choice categories
        logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits_shifted)
        trial_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)  # Shape: (n_trials, 3)
        
        pred_choice = np.argmax(trial_probs, axis=1)
        all_pred_choices.extend(pred_choice)
        all_true_choices.extend(true_choice.flatten())
        
    all_pred = np.array(all_pred_choices)
    all_true = np.array(all_true_choices)
    
    overall_hr = np.mean(all_pred == all_true) * 100
    accuracy_records_1state.append({'Subject': str(subj), 'Category': 'Overall', 'Hit Rate (%)': overall_hr})
    
    outcome_names = {0: 'Left', 1: 'Right', 2: 'Bail'}
    for cat_val, cat_name in outcome_names.items():
        mask = (all_true == cat_val)
        cat_hr = np.mean(all_pred[mask] == all_true[mask]) * 100 if np.sum(mask) > 0 else 0.0
        accuracy_records_1state.append({'Subject': str(subj), 'Category': cat_name, 'Hit Rate (%)': cat_hr})

df_accuracy_1state = pd.DataFrame(accuracy_records_1state)

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_accuracy_1state, x='Subject', y='Hit Rate (%)', hue='Category', palette='Set2')
plt.axhline(33.33, color='red', linestyle='--', linewidth=1.5, label='Chance Level (33.3%)')

plt.title("1-State Model: Choice Prediction Hit Rate by Subject and Outcome Category", fontsize=14, fontweight='bold')
plt.xlabel("Subject ID", fontsize=12)
plt.ylabel("Hit Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Metric / Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='solid', alpha=0.5)

plt.tight_layout()
plt.show()

#%% Calculate Overall and Outcome-Specific Model Choice Accuracy

model_accuracy_rows = []
outcome_names = {0: 'Left', 1: 'Right', 2: 'Bail'}
eval_num_states = 2

for subj, df in final_subject_dfs.items():
    print(f"\nEvaluating choice prediction accuracy for Subject {subj}...")
    subj_sessions = sess_data_dict[subj]
    
    subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in subj_sessions]
    subj_choices = [sess_data_dict[subj][s]['choices'] for s in subj_sessions]
    
    # Check if we can reuse the pre-fitted 2-state model from Section 3
    if 'best_models_dict' in globals() and subj in best_models_dict and eval_num_states in best_models_dict[subj] and best_models_dict[subj][eval_num_states] is not None:
        print(f"  -> Reusing pre-fitted {eval_num_states}-state model for Subject {subj}...")
        model = best_models_dict[subj][eval_num_states]
    else:
        print(f"  -> Fitting {eval_num_states}-state model for Subject {subj} from scratch...")
        if subj in consistency_results:
            best_seed = consistency_results[subj]['best_fit_seed']
            np.random.seed(best_seed * 100 + 7)
        else:
            np.random.seed(42)
            
        model = ssm.HMM(eval_num_states, 1, len(ssm_predictors), 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=3), 
                        transitions="standard")
        model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
    
    all_pred_choices = []
    all_true_choices = []
    
    for sess_id, sess_data in subj_sessions.items():
        inp = sess_data['inputs']            # Shape: (n_trials, input_dim)
        raw_choices = sess_data['choices']    # Shape: (n_trials,) or (n_trials, 1)
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choice = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices])
        else:
            true_choice = np.array(raw_choices).astype(int).flatten()

        res_expected = model.expected_states(true_choice, input=inp)
        posterior_states = res_expected[0] if isinstance(res_expected, tuple) else res_expected  # Shape: (n_trials, K)
        
        n_trials = len(true_choice)
        num_categories = 3
        p_cat_given_state = np.zeros((n_trials, model.K, num_categories))
        
        # Vectorized probability calculation per state across all session trials
        for k in range(model.K):
            w_k = model.observations.params[k]  # Shape: (2, input_dim)
            logits_non_base = inp @ w_k.T        # Shape: (n_trials, 2)
            logits = np.hstack([logits_non_base, np.zeros((n_trials, 1))])  # Bail baseline reference
            
            # Softmax across choice categories
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            p_cat_given_state[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
        # Marginalize over state posteriors: P(choice) = sum_k P(state_k) * P(choice | state_k)
        trial_probs = np.sum(posterior_states[:, :, None] * p_cat_given_state, axis=1)  # Shape: (n_trials, 3)
            
        pred_choice = np.argmax(trial_probs, axis=1)
        all_pred_choices.extend(pred_choice)
        all_true_choices.extend(true_choice.flatten())
        
    all_pred_choices = np.array(all_pred_choices)
    all_true_choices = np.array(all_true_choices)
    
    overall_hit_rate = np.mean(all_pred_choices == all_true_choices) * 100
    print(f"    -> Overall Model Hit Rate: {overall_hit_rate:.2f}%")
    
    row_data = {
        'subject': subj,
        'overall_hit_rate': overall_hit_rate
    }
    
    for cat_val, cat_name in outcome_names.items():
        mask = (all_true_choices == cat_val)
        if np.sum(mask) > 0:
            cat_hit_rate = np.mean(all_pred_choices[mask] == all_true_choices[mask]) * 100
            print(f"      * Hit Rate for '{cat_name}' trials: {cat_hit_rate:.2f}% (n={np.sum(mask)})")
            row_data[f'hit_rate_{cat_name.lower()}'] = cat_hit_rate
        else:
            print(f"      * Hit Rate for '{cat_name}' trials: N/A (0 trials)")
            row_data[f'hit_rate_{cat_name.lower()}'] = np.nan
            
    model_accuracy_rows.append(row_data)

print("\nFinished calculating model choice prediction accuracies!")

#%% Visualize Model Choice Prediction Accuracy & Outcome Breakdown

# Visualize Distribution of Trial-Level Agreements as Grouped Bars Per Subject

accuracy_records = []
eval_num_states = 2

for subj, df in final_subject_dfs.items():
    subj_sessions = sess_data_dict[subj]
    subj_inputs = [sess_data_dict[subj][s]['inputs'] for s in subj_sessions]
    subj_choices = [sess_data_dict[subj][s]['choices'] for s in subj_sessions]
    
    # Check if we can reuse the pre-fitted model from previous sections
    if 'best_models_dict' in globals() and subj in best_models_dict and eval_num_states in best_models_dict[subj] and best_models_dict[subj][eval_num_states] is not None:
        print(f"Reusing pre-fitted {eval_num_states}-state model for Subject {subj}...")
        model = best_models_dict[subj][eval_num_states]
    else:
        print(f"Fitting {eval_num_states}-state model for Subject {subj} from scratch...")
        if subj in consistency_results:
            best_seed = consistency_results[subj]['best_fit_seed']
            np.random.seed(best_seed * 100 + 7)
        else:
            np.random.seed(42)
            
        model = ssm.HMM(eval_num_states, 1, len(ssm_predictors), 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=3), 
                        transitions="standard")
        model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
    
    all_pred_choices = []
    all_true_choices = []
    
    for sess_id, sess_data in subj_sessions.items():
        inp = sess_data['inputs']            # Shape: (n_trials, input_dim)
        raw_choices = sess_data['choices']    # Shape: (n_trials,) or (n_trials, 1)
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choice = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices])
        else:
            true_choice = np.array(raw_choices).astype(int).flatten()

        res_expected = model.expected_states(true_choice, input=inp)
        posterior_states = res_expected[0] if isinstance(res_expected, tuple) else res_expected  # Shape: (n_trials, K)
        
        n_trials = len(true_choice)
        p_cat_given_state = np.zeros((n_trials, model.K, 3))
        
        # Vectorized probability calculation per state across all session trials
        for k in range(model.K):
            w_k = model.observations.params[k]  # Shape: (2, input_dim)
            logits_non_base = inp @ w_k.T        # Shape: (n_trials, 2)
            logits = np.hstack([logits_non_base, np.zeros((n_trials, 1))])  # Bail baseline reference
            
            # Softmax across choice categories
            logits_shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(logits_shifted)
            p_cat_given_state[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
        # Marginalize over state posteriors: P(choice) = sum_k P(state_k) * P(choice | state_k)
        trial_probs = np.sum(posterior_states[:, :, None] * p_cat_given_state, axis=1)  # Shape: (n_trials, 3)
            
        pred_choice = np.argmax(trial_probs, axis=1)
        all_pred_choices.extend(pred_choice)
        all_true_choices.extend(true_choice.flatten())
        
    all_pred = np.array(all_pred_choices)
    all_true = np.array(all_true_choices)
    
    overall_hr = np.mean(all_pred == all_true) * 100
    accuracy_records.append({'Subject': str(subj), 'Category': 'Overall', 'Hit Rate (%)': overall_hr})
    
    outcome_names = {0: 'Left', 1: 'Right', 2: 'Bail'}
    for cat_val, cat_name in outcome_names.items():
        mask = (all_true == cat_val)
        cat_hr = np.mean(all_pred[mask] == all_true[mask]) * 100 if np.sum(mask) > 0 else 0.0
        accuracy_records.append({'Subject': str(subj), 'Category': cat_name, 'Hit Rate (%)': cat_hr})

df_accuracy = pd.DataFrame(accuracy_records)

plt.figure(figsize=(12, 6))
ax = sns.barplot(data=df_accuracy, x='Subject', y='Hit Rate (%)', hue='Category', palette='Set2')
plt.axhline(33.33, color='red', linestyle='--', linewidth=1.5, label='Chance Level (33.3%)')

plt.title("Model Choice Prediction Hit Rate by Subject and Outcome Category", fontsize=14, fontweight='bold')
plt.xlabel("Subject ID", fontsize=12)
plt.ylabel("Hit Rate (%)", fontsize=12)
plt.ylim(0, 100)
plt.legend(title='Metric / Outcome', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.grid(axis='y', linestyle='solid', alpha=0.5)

plt.tight_layout()
plt.show()

#%% Individual Subject Session Diagnostic Plots (2-Panel: Choice Posteriors & State Posteriors with Shading)

def diagnose_subject_sessions(target_subj, max_sessions=3):
    """
    Loops through multiple sessions for a given subject and generates 
    a streamlined 2-panel diagnostic plot:
    - Panel 1: Choice Posterior Probabilities + Floating Choice Ticks on top
    - Panel 2: State Posterior Probabilities + Viterbi State Shading in background
    
    Parameters:
    - target_subj: int or str, the subject ID (e.g., 274)
    - max_sessions: int or 'all', number of sessions to plot (default: 3)
    """
    # Normalize subject key lookup (handles int vs string key mismatches)
    if target_subj not in sess_data_dict:
        alt_key = str(target_subj) if not isinstance(target_subj, str) else int(target_subj)
        if alt_key in sess_data_dict:
            target_subj = alt_key
        else:
            print(f"Error: Subject {target_subj} not found in sess_data_dict.")
            print(f"Available subjects: {list(sess_data_dict.keys())}")
            return
    
    # 1. Get all available sessions for this subject
    available_sessions = list(sess_data_dict[target_subj].keys())
    print(f"==================================================")
    print(f"Subject {target_subj}: Found {len(available_sessions)} total sessions.")
    print(f"Available session IDs: {available_sessions}")
    print(f"==================================================")
    
    # 2. Determine which sessions to loop through
    if max_sessions == 'all':
        sessions_to_plot = available_sessions
    else:
        sessions_to_plot = available_sessions[:max_sessions]
        
    print(f"Generating diagnostic plots for {len(sessions_to_plot)} session(s)...\n")
    
    # 3. Check model fits existence (support both consistency_results and best_models_dict)
    best_fit = None
    if target_subj in consistency_results and 'fits' in consistency_results[target_subj] and len(consistency_results[target_subj]['fits']) > 0:
        best_fit = consistency_results[target_subj]['fits'][0]
    elif 'best_models_dict' in globals() and target_subj in best_models_dict and 2 in best_models_dict[target_subj]:
        # Extract weights and trans_mat from pre-fitted model object if dictionary fit isn't found
        m_obj = best_models_dict[target_subj][2]
        best_fit = {
            'weights': m_obj.observations.params,
            'trans_mat': np.exp(m_obj.transitions.params[0])
        }
        
    if best_fit is None:
        print(f"Error: No model fits found for subject {target_subj}.")
        return
        
    weights = best_fit['weights'] # Shape: (num_states, C-1, input_dim)
    num_states = weights.shape[0]
    trans_mat = best_fit['trans_mat']

    # 4. Loop through each chosen session
    for idx, sess_key in enumerate(sessions_to_plot, 1):
        print(f"--- [{idx}/{len(sessions_to_plot)}] Rat {target_subj} | Session {sess_key} ---")
        
        raw_choices = sess_data_dict[target_subj][sess_key]['choices']
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choices_1d = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices])
        else:
            true_choices_1d = np.array(raw_choices).astype(int).flatten()
            
        true_choices = true_choices_1d.reshape(-1, 1)
        sess_inpt = sess_data_dict[target_subj][sess_key]['inputs']
        input_dim = sess_inpt.shape[1]

        choice_labels = {0: 'Left', 1: 'Right', 2: 'Bail'}
        choice_colors = {0: 'cornflowerblue', 1: 'orange', 2: 'forestgreen'}

        # Rebuild model for this session's input/choices shape using dynamic num_states
        model = ssm.HMM(num_states, 1, input_dim, 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=3), 
                        transitions="standard")
        model.observations.params = weights
        model.transitions.params = np.log(trans_mat + 1e-12)[None, ...]

        # Compute posteriors and Viterbi states
        posterior_res = model.expected_states(true_choices, input=sess_inpt)
        posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res
        viterbi_states = model.most_likely_states(true_choices, input=sess_inpt)

        # Compute choice posterior probabilities across states
        n_trials = len(true_choices_1d)
        choice_probs_all_states = np.zeros((n_trials, num_states, 3))
        for k in range(num_states):
            sub_logits = sess_inpt @ weights[k].T # Shape: (n_trials, C-1)
            full_logits = np.hstack([sub_logits, np.zeros((n_trials, 1))]) # Add 0 logit for baseline Category 2 (Bail)
            exp_logits = np.exp(full_logits - np.max(full_logits, axis=1, keepdims=True))
            choice_probs_all_states[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        # Marginalize choice probabilities across state posteriors
        marginal_choice_probs = np.sum(posterior_probs[:, :, None] * choice_probs_all_states, axis=1)

        # Print Class Balance / Denominator Check
        unique_classes, counts = np.unique(true_choices_1d, return_counts=True)
        for cls, cnt in zip(unique_classes, counts):
            pct = (cnt / len(true_choices_1d)) * 100
            print(f"  - Category {cls} ({choice_labels.get(cls, 'Unknown')}): {cnt} trials ({pct:.1f}%)")
        print("-" * 50)

        # 5. Plotting 2-Panel Diagnostic Figure
        fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
        trials = np.arange(len(true_choices_1d))

        # --- PANEL 1: Choice Posterior Probabilities + Floating Choice Ticks ---
        ax1 = axes[0]
        ax1.plot(trials, marginal_choice_probs[:, 0], label='P(Left)', color='cornflowerblue', linewidth=1.5, zorder=2)
        ax1.plot(trials, marginal_choice_probs[:, 1], label='P(Right)', color='orange', linewidth=1.5, zorder=2)
        ax1.plot(trials, marginal_choice_probs[:, 2], label='P(Bail)', color='forestgreen', linewidth=1.5, zorder=2)
        
        # Helper to plot short floating choice ticks above y=1.0 on Panel 1
        for t in range(n_trials):
            c = int(true_choices_1d[t])
            color = choice_colors.get(c, 'gray')
            ymin = 1.03
            ymax = 1.07 if c == 2 else 1.11  # Left/Right span 0.08, Bail spans 0.04
            ax1.vlines(x=t, ymin=ymin, ymax=ymax, color=color, alpha=1.0, lw=1.2, zorder=5)

        ax1.set_ylabel("Choice Prob.", fontsize=10, fontweight='semibold')
        ax1.set_ylim(-0.05, 1.15)  # Extended headroom for choice ticks above 1.0
        ax1.set_title(f"Rat {target_subj} - Session {sess_key}: Choice Probabilities & Inferred States", fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize='x-small')
        ax1.grid(True, linestyle='--', alpha=0.4)

        # --- PANEL 2: State Posterior Probabilities + Viterbi Shading Background ---
        ax2 = axes[1]
        
        # Background shading for Viterbi states
        current_state = viterbi_states[0]
        start_idx = 0
        state_colors = ['#e0f2fe', '#fef3c7'] # Light blue and light amber

        for t in range(1, len(viterbi_states)):
            if viterbi_states[t] != current_state:
                ax2.axvspan(start_idx, t, color=state_colors[current_state % len(state_colors)], alpha=0.5, ec=None, zorder=1)
                current_state = viterbi_states[t]
                start_idx = t
        ax2.axvspan(start_idx, len(viterbi_states), color=state_colors[current_state % len(state_colors)], alpha=0.5, ec=None, zorder=1)

        # Plot state posterior probability curves on top of the shading
        for k in range(posterior_probs.shape[1]):
            ax2.plot(trials, posterior_probs[:, k], label=f'State {k+1} Posterior', linewidth=1.8, color=['#0284c7', '#d97706'][k % 2], zorder=3)

        ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.7, label='Chance / Ambiguity Line (0.5)', zorder=3)
        ax2.set_ylabel("State Prob.", fontsize=10, fontweight='semibold')
        ax2.set_xlabel("Trial Number", fontsize=10, fontweight='semibold')
        ax2.set_ylim(-0.05, 1.05)
        ax2.legend(loc='upper right', fontsize='x-small')
        ax2.grid(True, linestyle='--', alpha=0.4, zorder=2)

        plt.tight_layout()
        plt.show()

# ==========================================
# HOW TO USE IT:
# ==========================================

subj_ids = [198, 199, 234, 235, 237, 238, 274, 400, 402, 419, 421, 422, 424, 483]

for subj in subj_ids:
    diagnose_subject_sessions(subj, max_sessions=1)
    
    
# ==============================================================================
# HMM STATE VISUALIZATION MECHANICS: MARGINALS VS. VITERBI DECODING (PANEL 2)
# ==============================================================================
# 
# 1. THE LINES (Marginal Posterior Probabilities):
#    - Generated via model.expected_states() using the forward-backward algorithm.
#    - Represents the soft, context-aware posterior probability of being in each 
#      state at any given trial, incorporating information from neighboring trials 
#      (both past and future) via forward and backward smoothing passes. 
#    - The continuous blue and orange curves cross precisely where probability 
#      mass hits 0.5 (indicated by the horizontal dotted line).
#
# 2. THE BACKGROUND SHADING (Viterbi Decoding):
#    - Generated via model.most_likely_states() using the Viterbi algorithm.
#    - Finds the single globally optimal discrete state sequence across the 
#      entire session, incorporating transition matrix penalties.
#    - Why shading/transitions can occur slightly before/after lines cross 0.5:
#      The Viterbi path penalizes rapid trial-by-trial flickering, meaning it 
#      can officially commit to a state transition slightly earlier or later 
#      based on temporal continuity and global sequence likelihood, rather 
#      than reacting strictly to a local 50% line-crossing threshold.
# ==============================================================================

#%% Compare 1-State vs 2-State Choice Probabilities for Subjects with Trial Choices Overlay

def compare_1s_2s_choice_posteriors(target_subjects=[400, 402, 424], session_idx=0):
    """
    Compares trial-by-trial choice posterior/predictive probabilities between 
    1-state (GLM) and 2-state (GLM-HMM) models for specified subjects,
    displaying actual subject choices as short tick marks floating above the probability axes.
    """
    choice_labels = {0: 'Left', 1: 'Right', 2: 'Bail'}
    choice_colors = {0: 'cornflowerblue', 1: 'orange', 2: 'forestgreen'}

    for subj in target_subjects:
        # Handle string vs integer subject key mismatches safely
        subj_key = subj
        if subj_key not in sess_data_dict:
            alt_key = str(subj) if not isinstance(subj, str) else int(subj)
            if alt_key in sess_data_dict:
                subj_key = alt_key
            else:
                print(f"Subject {subj} not found in sess_data_dict. Skipping.")
                continue
                
        available_sessions = list(sess_data_dict[subj_key].keys())
        if not available_sessions:
            continue
        sess_key_to_plot = available_sessions[min(session_idx, len(available_sessions) - 1)]
        
        sess_data = sess_data_dict[subj_key][sess_key_to_plot]
        raw_choices = sess_data['choices']
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choices_array = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices])
        else:
            true_choices_array = np.array(raw_choices).astype(int).flatten()
            
        true_choices_2d = true_choices_array.reshape(-1, 1)
        
        sess_inpts = sess_data['inputs']
        input_dim = sess_inpts.shape[1]
        n_trials = len(true_choices_array)
        trials = np.arange(n_trials)
        
        # 1. Get or Fit 1-State Model
        if 'best_models_dict' in globals() and subj_key in best_models_dict and 1 in best_models_dict[subj_key] and best_models_dict[subj_key][1] is not None:
            model_1s = best_models_dict[subj_key][1]
        else:
            model_1s = ssm.HMM(1, 1, input_dim, 
                               observations="input_driven_obs", 
                               observation_kwargs=dict(C=3), 
                               transitions="standard")
            model_1s.fit([true_choices_2d], inputs=[sess_inpts], method="em", num_iters=1000, tolerance=tol)
            
        # Compute 1-State choice probabilities
        weights_1s = model_1s.observations.params # Shape: (1, C-1, input_dim)
        sub_logits_1s = sess_inpts @ weights_1s[0].T # Shape: (n_trials, C-1)
        full_logits_1s = np.hstack([sub_logits_1s, np.zeros((n_trials, 1))]) # Bail baseline
        exp_logits_1s = np.exp(full_logits_1s - np.max(full_logits_1s, axis=1, keepdims=True))
        choice_probs_1s = exp_logits_1s / np.sum(exp_logits_1s, axis=1, keepdims=True) # Shape: (n_trials, 3)

        # 2. Get or Fit 2-State Model
        if 'best_models_dict' in globals() and subj_key in best_models_dict and 2 in best_models_dict[subj_key] and best_models_dict[subj_key][2] is not None:
            model_2s = best_models_dict[subj_key][2]
        elif subj_key in consistency_results and 'fits' in consistency_results[subj_key] and len(consistency_results[subj_key]['fits']) > 0:
            best_fit = consistency_results[subj_key]['fits'][0]
            model_2s = ssm.HMM(2, 1, input_dim, 
                               observations="input_driven_obs", 
                               observation_kwargs=dict(C=3), 
                               transitions="standard")
            model_2s.observations.params = best_fit['weights']
            model_2s.transitions.params = np.log(best_fit['trans_mat'] + 1e-12)[None, ...]
        else:
            model_2s = ssm.HMM(2, 1, input_dim, 
                               observations="input_driven_obs", 
                               observation_kwargs=dict(C=3), 
                               transitions="standard")
            model_2s.fit([true_choices_2d], inputs=[sess_inpts], method="em", num_iters=1000, tolerance=tol)
            
        # Compute 2-State posteriors and marginal choice probabilities
        posterior_res = model_2s.expected_states(true_choices_2d, input=sess_inpts)
        post_2s = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res # Shape: (n_trials, 2)
        
        weights_2s = model_2s.observations.params # Shape: (2, C-1, input_dim)
        choice_probs_all_states = np.zeros((n_trials, 2, 3))
        for k in range(2):
            sub_logits_2s = sess_inpts @ weights_2s[k].T # Shape: (n_trials, C-1)
            full_logits_2s = np.hstack([sub_logits_2s, np.zeros((n_trials, 1))])
            exp_logits_2s = np.exp(full_logits_2s - np.max(full_logits_2s, axis=1, keepdims=True))
            choice_probs_all_states[:, k, :] = exp_logits_2s / np.sum(exp_logits_2s, axis=1, keepdims=True)
            
        # Marginalize choice probabilities across the 2-state posteriors
        marginal_choice_probs_2s = np.sum(post_2s[:, :, None] * choice_probs_all_states, axis=1) # Shape: (n_trials, 3)

        # 3. Plotting comparison figure with choices floating above the graph area
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        
        # Helper function: short tick marks above y=1.0 (Bails = half height of Left/Right)
        def plot_choice_vlines(ax):
            for t in range(n_trials):
                c = int(true_choices_array[t])
                color = choice_colors.get(c, 'gray')
                ymin = 1.03
                ymax = 1.07 if c == 2 else 1.11  # Left/Right span 0.08, Bail spans 0.04
                ax.vlines(x=t, ymin=ymin, ymax=ymax, color=color, alpha=1.0, lw=1.2, zorder=5)

        # Panel 1: 1-State Model Choice Probabilities
        axes[0].plot(trials, choice_probs_1s[:, 0], label='P(Left)', color='cornflowerblue', lw=1.5, zorder=2)
        axes[0].plot(trials, choice_probs_1s[:, 1], label='P(Right)', color='orange', lw=1.5, zorder=2)
        axes[0].plot(trials, choice_probs_1s[:, 2], label='P(Bail)', color='forestgreen', lw=1.5, zorder=2)
        plot_choice_vlines(axes[0])
        
        axes[0].set_ylim(-0.05, 1.15)  # Extended headroom for choice ticks above 1.0
        axes[0].set_ylabel("Choice Prob.")
        axes[0].set_title(f"Subject {subj} | Session {sess_key_to_plot}: 1-State Model Choice Probabilities (Static GLM)", fontsize=11, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.4)
        axes[0].legend(loc='upper right', fontsize='small')
        
        # Panel 2: 2-State Model Marginalized Choice Probabilities
        axes[1].plot(trials, marginal_choice_probs_2s[:, 0], label='P(Left)', color='cornflowerblue', lw=1.5, zorder=2)
        axes[1].plot(trials, marginal_choice_probs_2s[:, 1], label='P(Right)', color='orange', lw=1.5, zorder=2)
        axes[1].plot(trials, marginal_choice_probs_2s[:, 2], label='P(Bail)', color='forestgreen', lw=1.5, zorder=2)
        plot_choice_vlines(axes[1])
        
        axes[1].set_ylim(-0.05, 1.15)  # Extended headroom for choice ticks above 1.0
        axes[1].set_ylabel("Choice Prob.")
        axes[1].set_xlabel("Trial Number")
        axes[1].set_title(f"Subject {subj} | Session {sess_key_to_plot}: 2-State Model Marginalized Choice Probabilities (GLM-HMM)", fontsize=11, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.4)
        axes[1].legend(loc='upper right', fontsize='small')
        
        plt.tight_layout()
        plt.show()

# Execute choice posterior comparison for subjects
compare_1s_2s_choice_posteriors(target_subjects=[400, 402, 424], session_idx=0)

#%% Continuous Predictive Performance (Pseudo-Hit Rate & Cosine Sim Axis 1 vs Axis 0)
print("\n" + "="*60)
print("SECTION 4: PSEUDO-HIT RATES & COSINE SIMILARITY (AXIS 1 VS AXIS 0)")
print("="*60)

perf_rows = []
eval_num_states = target_num_states if 'target_num_states' in globals() else 2

for subj in sess_data_dict.keys():
    # Normalize key lookup if necessary
    subj_key = subj
    if subj_key not in best_models_dict:
        alt_key = str(subj) if not isinstance(subj, str) else int(subj)
        if alt_key in best_models_dict:
            subj_key = alt_key
        else:
            continue
            
    if eval_num_states not in best_models_dict[subj_key]:
        continue
        
    model = best_models_dict[subj_key][eval_num_states]
    weights = model.observations.params
    
    # Store aggregated probabilities across all sessions
    all_marginal_probs = []
    all_true_choices = []
    
    for sess_key in sess_data_dict[subj].keys():
        sess_inpt = sess_data_dict[subj][sess_key]['inputs']
        raw_choices = sess_data_dict[subj][sess_key]['choices']
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choices = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices]).reshape(-1, 1)
        else:
            true_choices = np.array(raw_choices).astype(int).reshape(-1, 1)
        
        # Get State Posteriors
        posterior_res = model.expected_states(true_choices, input=sess_inpt)
        posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res
        
        # Calculate Choice Probs per state, then marginalize
        n_trials = len(true_choices)
        choice_probs_all_states = np.zeros((n_trials, eval_num_states, 3))
        for k in range(eval_num_states):
            sub_logits = sess_inpt @ weights[k].T
            full_logits = np.hstack([sub_logits, np.zeros((n_trials, 1))])
            exp_logits = np.exp(full_logits - np.max(full_logits, axis=1, keepdims=True))
            choice_probs_all_states[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
        marginal_choice_probs = np.sum(posterior_probs[:, :, None] * choice_probs_all_states, axis=1)
        
        all_marginal_probs.append(marginal_choice_probs)
        all_true_choices.append(true_choices.flatten())
        
    # Concatenate all sessions for this subject
    P_choice = np.vstack(all_marginal_probs)
    T_choice = np.concatenate(all_true_choices)
    T_one_hot = np.eye(3)[T_choice]
    
    # 1. Pseudo-Hit Rates (Per Outcome & Pooled Mean via exp(mean LL))
    log_probs = np.log(P_choice + 1e-12)
    labels = {0: 'Left', 1: 'Right', 2: 'Bail'}
    pseudo_hits = {}
    
    for c in [0, 1, 2]:
        mask = (T_choice == c)
        if np.sum(mask) > 0:
            mean_lp = np.mean(log_probs[mask, c])
            pseudo_hits[labels[c]] = np.exp(mean_lp)
        else:
            pseudo_hits[labels[c]] = np.nan
            
    # Pooled Overall Pseudo-Hit Rate: exp(mean log-prob across ALL true choices)
    all_chosen_log_probs = log_probs[np.arange(len(T_choice)), T_choice]
    pooled_mean_pseudo_hit = np.exp(np.mean(all_chosen_log_probs))
            
    # 2. Cosine Similarity: AXIS = 1 (Trial-Wise Mean) with safe division norm
    dot_prods_row = np.sum(P_choice * T_one_hot, axis=1)
    norms_row = np.linalg.norm(P_choice, axis=1) + 1e-12
    cos_sims_row = dot_prods_row / norms_row
    mean_cos_sim_axis1 = np.mean(cos_sims_row)
    
    # 3. Cosine Similarity: AXIS = 0 (Column-Wise / Choice Timelines) with safe division norms
    dot_prods_col = np.sum(P_choice * T_one_hot, axis=0)
    norm_p_col = np.linalg.norm(P_choice, axis=0) + 1e-12
    norm_t_col = np.linalg.norm(T_one_hot, axis=0) + 1e-12
    cos_sim_cols_axis0 = dot_prods_col / (norm_p_col * norm_t_col)
    mean_cos_sim_axis0 = np.nanmean(cos_sim_cols_axis0)
    
    perf_rows.append({
        'Subject': str(subj),
        'Pseudo_Hit_Left': pseudo_hits['Left'],
        'Pseudo_Hit_Right': pseudo_hits['Right'],
        'Pseudo_Hit_Bail': pseudo_hits['Bail'],
        'Pseudo_Hit_Mean': pooled_mean_pseudo_hit,
        'Mean_Cosine_Axis1': mean_cos_sim_axis1,
        'Cosine_Axis0_Left': cos_sim_cols_axis0[0],
        'Cosine_Axis0_Right': cos_sim_cols_axis0[1],
        'Cosine_Axis0_Bail': cos_sim_cols_axis0[2],
        'Cosine_Axis0_Mean': mean_cos_sim_axis0
    })

df_perf = pd.DataFrame(perf_rows)
print(df_perf.to_string(index=False))

# --- Numerical Comparison Summary Between Axis 1 and Axis 0 ---
print("\n" + "-"*60)
print("NUMERICAL COMPARISON: AXIS 1 (Trial-Wise) vs AXIS 0 (Timeline Mean)")
print("-"*60)
axis1_vals = df_perf['Mean_Cosine_Axis1'].values
axis0_vals = df_perf['Cosine_Axis0_Mean'].values
diffs = axis1_vals - axis0_vals

comparison_summary = pd.DataFrame({
    'Subject': df_perf['Subject'],
    'Axis_1 (Trial-Wise)': axis1_vals,
    'Axis_0 (Timeline Mean)': axis0_vals,
    'Difference (1 - 0)': diffs
})
print(comparison_summary.to_string(index=False))
print("-" * 60)
print(f"Grand Mean across subjects -> Axis 1: {np.mean(axis1_vals):.4f} | Axis 0 (Mean): {np.mean(axis0_vals):.4f}")
print(f"Mean Absolute Difference: {np.mean(np.abs(diffs)):.4f}")
print("="*60)

# ==========================================
# 3-Panel Comparison Layout (Consistent X-Axis Alignment)
# ==========================================
fig, ax = plt.subplots(1, 3, figsize=(21, 5))
x = np.arange(len(df_perf['Subject']))
width_4 = 0.2  # 4 bars per cluster

# --- Panel 1: Pseudo-Hit Rates (Per Outcome + Pooled Mean Bar) ---
ax[0].bar(x - 1.5 * width_4, df_perf['Pseudo_Hit_Left'], width_4, label='Left', color='cornflowerblue')
ax[0].bar(x - 0.5 * width_4, df_perf['Pseudo_Hit_Right'], width_4, label='Right', color='orange')
ax[0].bar(x + 0.5 * width_4, df_perf['Pseudo_Hit_Bail'], width_4, label='Bail', color='forestgreen')
ax[0].bar(x + 1.5 * width_4, df_perf['Pseudo_Hit_Mean'], width_4, label='Mean', color='slateblue')
ax[0].set_xticks(x)
ax[0].set_xticklabels(df_perf['Subject'])
ax[0].set_title(f"Pseudo-Hit Rates: Outcomes & Pooled Mean (K={eval_num_states})")
ax[0].set_ylabel("exp(Mean Log-Prob)")
ax[0].set_ylim(0, 1.05)
ax[0].legend(loc='lower right')
ax[0].grid(axis='y', alpha=0.3)

# --- Panel 2: Cosine Similarity (Axis = 1: Trial-Wise Mean) ---
ax[1].bar(x, df_perf['Mean_Cosine_Axis1'], width=0.4, color='slateblue', alpha=0.7)
ax[1].set_xticks(x)
ax[1].set_xticklabels(df_perf['Subject'])
ax[1].set_title("Cosine Sim: Axis = 1 (Trial-Wise Mean)")
ax[1].set_ylabel("Mean Cosine Similarity")
ax[1].set_ylim(0, 1.05)
ax[1].grid(axis='y', alpha=0.3)

# --- Panel 3: Cosine Similarity (Axis = 0: Choice Timelines + Mean Bar) ---
ax[2].bar(x - 1.5 * width_4, df_perf['Cosine_Axis0_Left'], width_4, label='Left', color='cornflowerblue')
ax[2].bar(x - 0.5 * width_4, df_perf['Cosine_Axis0_Right'], width_4, label='Right', color='orange')
ax[2].bar(x + 0.5 * width_4, df_perf['Cosine_Axis0_Bail'], width_4, label='Bail', color='forestgreen')
ax[2].bar(x + 1.5 * width_4, df_perf['Cosine_Axis0_Mean'], width_4, label='Mean', color='slateblue')
ax[2].set_xticks(x)
ax[2].set_xticklabels(df_perf['Subject'])
ax[2].set_title("Cosine Sim: Axis = 0 (Timelines & Macro-Avg)")
ax[2].set_ylabel("Timeline Cosine Similarity")
ax[2].set_ylim(0, 1.05)
ax[2].legend(loc='lower right')
ax[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

#%% State Interpretation via State-Choice Cosine Similarity
print("\n" + "="*60)
print("SECTION 5: STATE-TO-CHOICE POSTERIOR COSINE SIMILARITY")
print("="*60)

# ==============================================================================
# STATE-TO-CHOICE POSTERIOR COSINE SIMILARITY
# ==============================================================================
# 
# 1. MATH & INTERPRETATION:
#    - Treats each session's trial sequence as a continuous temporal vector.
#    - Computes the cosine similarity (normalized dot product) between a hidden 
#      state's posterior probability vector (S_k) and a choice's marginal 
#      probability vector (C_c) across all trials.
#    - Quantifies behavioral "labels" by showing which choices are actively 
#      driven or synchronized with each inferred hidden state (scores near 1.0 
#      indicate strong temporal alignment).
#
# 2. HANDLING VECTOR ALIGNMENT ACROSS SESSIONS:
#    - Because different sessions have varying trial lengths, direct comparison 
#      is achieved by vertically stacking (np.vstack) trial matrices across all 
#      sessions. 
#    - This builds unified global vectors of length equal to total trials (Sum of T), 
#      preserving trial-by-trial synchronization for the dot product.
# ==============================================================================

eval_num_states = target_num_states if 'target_num_states' in globals() else 2

for subj in list(sess_data_dict.keys()):
    # Normalize subject key lookup for models dictionary
    subj_key = subj
    if subj_key not in best_models_dict:
        alt_key = str(subj) if not isinstance(subj, str) else int(subj)
        if alt_key in best_models_dict:
            subj_key = alt_key
        else:
            continue
            
    if eval_num_states not in best_models_dict[subj_key]:
        continue
        
    model = best_models_dict[subj_key][eval_num_states]
    weights = model.observations.params
    
    all_state_probs = []
    all_choice_probs = []
    
    # Normalize subject key lookup for session data dictionary
    data_subj_key = subj
    if data_subj_key not in sess_data_dict:
        alt_key = str(subj) if not isinstance(subj, str) else int(subj)
        if alt_key in sess_data_dict:
            data_subj_key = alt_key
        else:
            continue
    
    for sess_key in sess_data_dict[data_subj_key].keys():
        sess_inpt = sess_data_dict[data_subj_key][sess_key]['inputs']
        raw_choices = sess_data_dict[data_subj_key][sess_key]['choices']
        
        # Ensure true choices are formatted as integers matching observation categories (0, 1, 2)
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            true_choices = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices]).reshape(-1, 1)
        else:
            true_choices = np.array(raw_choices).astype(int).reshape(-1, 1)
        
        # State Posteriors
        posterior_res = model.expected_states(true_choices, input=sess_inpt)
        posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res
        
        # Marginal Choice Posteriors
        n_trials = len(true_choices)
        choice_probs_all_states = np.zeros((n_trials, eval_num_states, 3))
        for k in range(eval_num_states):
            sub_logits = sess_inpt @ weights[k].T
            full_logits = np.hstack([sub_logits, np.zeros((n_trials, 1))])
            exp_logits = np.exp(full_logits - np.max(full_logits, axis=1, keepdims=True))
            choice_probs_all_states[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
        marginal_choice_probs = np.sum(posterior_probs[:, :, None] * choice_probs_all_states, axis=1)
        
        all_state_probs.append(posterior_probs)
        all_choice_probs.append(marginal_choice_probs)
        
    if not all_state_probs:
        continue
        
    # Stack arrays across all sessions for a global similarity check
    S_probs = np.vstack(all_state_probs)  # Shape: (Total Trials, eval_num_states)
    C_probs = np.vstack(all_choice_probs) # Shape: (Total Trials, 3)
    
    # Calculate Cosine Similarity Matrix (K states x 3 choices) with safe division norms
    sim_matrix = np.zeros((eval_num_states, 3))
    for k in range(eval_num_states):
        for c in range(3):
            vec_S = S_probs[:, k]
            vec_C = C_probs[:, c]
            
            num = np.dot(vec_S, vec_C)
            den = (np.linalg.norm(vec_S) * np.linalg.norm(vec_C)) + 1e-12
            sim_matrix[k, c] = num / den
            
    # Plot Heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu", vmin=0, vmax=1, 
                xticklabels=['Left (0)', 'Right (1)', 'Bail (2)'],
                yticklabels=[f'State {k+1}' for k in range(eval_num_states)])
    plt.title(f"Rat {subj}: State vs Choice Posterior Similarity")
    plt.ylabel("Inferred Hidden States")
    plt.xlabel("Choice Categories")
    plt.tight_layout()
    plt.show()
    
#%% Initialization Stability (Cross-Iteration Similarity per Session)
print("\n" + "="*60)
print("SECTION 6: CROSS-ITERATION POSTERIOR STABILITY (PER SESSION)")
print("="*60)

n_stability_inits = 5  # Number of random initializations to compare
eval_num_states = target_num_states if 'target_num_states' in globals() else 2

for subj in list(sess_data_dict.keys()):
    # Normalize subject key lookup
    subj_key = subj
    if subj_key not in sess_data_dict:
        alt_key = str(subj) if not isinstance(subj, str) else int(subj)
        if alt_key in sess_data_dict:
            subj_key = alt_key
        else:
            continue
            
    print(f"\nEvaluating stability for Rat {subj_key} across {n_stability_inits} initializations...")
    
    session_keys = list(sess_data_dict[subj_key].keys())
    if not session_keys:
        continue
        
    subj_inputs = []
    subj_choices = []
    
    for s in session_keys:
        sess_inpt = sess_data_dict[subj_key][s]['inputs']
        raw_choices = sess_data_dict[subj_key][s]['choices']
        
        # Ensure choices are formatted as integers (0, 1, 2) for ssm fitting
        if isinstance(raw_choices[0], (str, np.str_)):
            choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
            int_choices = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices]).astype(int)
        else:
            int_choices = np.array(raw_choices).astype(int)
            
        subj_inputs.append(sess_inpt)
        subj_choices.append(int_choices.reshape(-1, 1))
        
    input_dim = subj_inputs[0].shape[1]
    
    iter_posteriors = {i: {} for i in range(n_stability_inits)}
    iter_full_states = [] 
    
    for i in range(n_stability_inits):
        np.random.seed(42 + i * 100)
        
        model = ssm.HMM(eval_num_states, 1, input_dim, 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=3), 
                        transitions="standard")
        model.fit(subj_choices, inputs=subj_inputs, method="em", num_iters=1000, tolerance=tol)
        weights = model.observations.params
        
        full_state_probs_list = []
        for sess_idx, sess_key in enumerate(session_keys):
            sess_inpt = subj_inputs[sess_idx]
            true_choices = subj_choices[sess_idx]
            n_trials = len(true_choices)
            
            posterior_res = model.expected_states(true_choices, input=sess_inpt)
            posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res
            full_state_probs_list.append(posterior_probs)
            
            choice_probs_all_states = np.zeros((n_trials, eval_num_states, 3))
            for k in range(eval_num_states):
                sub_logits = sess_inpt @ weights[k].T
                full_logits = np.hstack([sub_logits, np.zeros((n_trials, 1))])
                exp_logits = np.exp(full_logits - np.max(full_logits, axis=1, keepdims=True))
                choice_probs_all_states[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
                
            marginal_choice_probs = np.sum(posterior_probs[:, :, None] * choice_probs_all_states, axis=1)
            
            iter_posteriors[i][sess_key] = {
                'choice': marginal_choice_probs,
                'state': posterior_probs
            }
            
        iter_full_states.append(np.vstack(full_state_probs_list))

    # Align State Labels to Iteration 0 using Hungarian matching
    S0 = iter_full_states[0]
    for i in range(1, n_stability_inits):
        Si = iter_full_states[i]
        cost_matrix = -np.dot(S0.T, Si) 
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for sess_key in session_keys:
            iter_posteriors[i][sess_key]['state'] = iter_posteriors[i][sess_key]['state'][:, col_ind]

    # Calculate Average Pairwise Similarity per Session with safe division norms
    session_stability_data = []
    
    for sess_key in session_keys:
        choice_sims = []
        state_sims = []
        
        # Upper triangle pairwise combinations (Init 0 vs 1, 0 vs 2, etc.)
        for i in range(n_stability_inits):
            for j in range(i + 1, n_stability_inits):
                c_i = iter_posteriors[i][sess_key]['choice'].flatten()
                c_j = iter_posteriors[j][sess_key]['choice'].flatten()
                choice_sim = np.dot(c_i, c_j) / ((np.linalg.norm(c_i) * np.linalg.norm(c_j)) + 1e-12)
                choice_sims.append(choice_sim)
                
                s_i = iter_posteriors[i][sess_key]['state'].flatten()
                s_j = iter_posteriors[j][sess_key]['state'].flatten()
                state_sim = np.dot(s_i, s_j) / ((np.linalg.norm(s_i) * np.linalg.norm(s_j)) + 1e-12)
                state_sims.append(state_sim)
                
        session_stability_data.append({
            'Session': str(sess_key),
            'Choice_Stability': np.mean(choice_sims),
            'State_Stability': np.mean(state_sims)
        })
        
    df_stability = pd.DataFrame(session_stability_data)
    
    # Plotting Session-by-Session Stability
    fig, ax = plt.subplots(figsize=(10, 4))
    
    ax.plot(df_stability['Session'], df_stability['Choice_Stability'], marker='o', lw=2, 
            color='teal', label='Choice Posterior Stability')
    ax.plot(df_stability['Session'], df_stability['State_Stability'], marker='s', lw=2, linestyle='--', 
            color='darkred', label='State Posterior Stability')
    
    ax.set_title(f"Rat {subj_key}: Model Initialization Stability per Session (K={eval_num_states})")
    ax.set_ylabel("Mean Pairwise Cosine Sim")
    ax.set_xlabel("Session Number")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.7, label='0.9 Threshold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
#%% Selected Subjects Comparison (Subject on X-Axis, Category as Hue)

print("\n" + "="*60)
print("SUBJECT COMPARISON (SUBJECT-CENTRIC): 237, 402, 424")
print("="*60)

target_subjects = ['237', '402', '424']

df_ph_sub = df_ph[df_ph['Subject'].isin(target_subjects)].copy()
df_c1_sub = df_c1[df_c1['Subject'].isin(target_subjects)].copy()
df_c0_sub = df_c0[df_c0['Subject'].isin(target_subjects)].copy()

if df_ph_sub.empty:
    print("Warning: Target subjects not found. Make sure you ran the population extraction cell first.")
else:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('SELECTED SUBJECTS SUMMARY (SUBJECT-CENTRIC): PSEUDO-HIT RATE AND COSINE SIMILARITY', fontsize=13, fontweight='bold', y=0.98)

    palette = {'Left': '#5C82E6', 'Right': '#FFA726', 'Bail': '#2E7D32', 'Mean': '#7E57C2'}
    subject_order = ['237', '402', '424']

    # Panel 1: Pseudo-Hit Rates
    sns.barplot(data=df_ph_sub, x='Subject', y='Value', hue='Category', order=subject_order, hue_order=lrb_categories, palette=palette, ax=axes[0])
    axes[0].set_title('Pseudo-Hit Rates: Outcomes & Pooled Mean (K=2)', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_ylabel('exp(Mean Log-Prob)', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Subject ID', fontsize=10, fontweight='bold')
    axes[0].set_ylim([0.0, 1.05])
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[0].legend(title='Category', loc='upper right')

    # Panel 2: Cosine Sim Axis = 1
    sns.barplot(data=df_c1_sub, x='Subject', y='Value', hue='Category', order=subject_order, hue_order=lrb_categories, palette=palette, ax=axes[1])
    axes[1].set_title('Cosine Sim: Axis = 1 (Trial-Wise Mean)', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_ylabel('Mean Cosine Similarity', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Subject ID', fontsize=10, fontweight='bold')
    axes[1].set_ylim([0.0, 1.05])
    axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[1].legend(title='Category', loc='upper right')

    # Panel 3: Cosine Sim Axis = 0
    sns.barplot(data=df_c0_sub, x='Subject', y='Value', hue='Category', order=subject_order, hue_order=lrb_categories, palette=palette, ax=axes[2])
    axes[2].set_title('Cosine Sim: Axis = 0 (Timelines & Macro-Avg)', fontsize=11, fontweight='bold', pad=10)
    axes[2].set_ylabel('Timeline Cosine Similarity', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Subject ID', fontsize=10, fontweight='bold')
    axes[2].set_ylim([0.0, 1.05])
    axes[2].grid(True, axis='y', linestyle=':', alpha=0.6)
    axes[2].legend(title='Category', loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Subject-centric comparison plot generated successfully!")
    
#%% Population-Wide Summary (Pseudo-Hit Rates & Cosine Similarities)

print("\n" + "="*60)
print("LAB PRESENTATION: POPULATION-WIDE SUMMARY (SLIDE 3)")
print("="*60)

# 1. Load session data and best models explicitly
sess_data_path = 'sess_data_dict.pkl'
if os.path.exists(sess_data_path) and ('sess_data_dict' not in globals() or not sess_data_dict):
    with open(sess_data_path, 'rb') as f:
        sess_data_dict = pickle.load(f)
    print(f"Loaded sess_data_dict ({len(sess_data_dict)} subjects).")

models_path = 'glm_hmm_best_models.pkl'
if os.path.exists(models_path):
    with open(models_path, 'rb') as f:
        best_models_dict = pickle.load(f)
    print(f"Loaded best models from {models_path} ({len(best_models_dict)} subjects).")
else:
    raise FileNotFoundError(f"Could not find model file at {models_path}")

def extract_model_object(val):
    if hasattr(val, 'observations'):
        return val
    if isinstance(val, dict):
        for sub_k in ['model', 'fitted_model', 'best_model', 'hmms', 'hmm']:
            if sub_k in val:
                res = extract_model_object(val[sub_k])
                if res is not None:
                    return res
        for sub_v in val.values():
            res = extract_model_object(sub_v)
            if res is not None:
                return res
    return None

def get_matching_model(subj, models_dict):
    subj_entry = models_dict.get(subj)
    if subj_entry is None:
        for alt in [str(subj), int(subj) if isinstance(subj, str) or np.issubdtype(type(subj), np.integer) else str(subj)]:
            if alt in models_dict:
                subj_entry = models_dict[alt]
                break
    if subj_entry is None or not isinstance(subj_entry, dict):
        return None
    for k_key, val in subj_entry.items():
        try:
            if int(str(k_key).split('_')[0]) == 2:
                model_obj = extract_model_object(val)
                if model_obj is not None:
                    return model_obj
        except (ValueError, TypeError):
            continue
    return None

choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}
lrb_categories = ['Left', 'Right', 'Bail', 'Mean']
labels_map = {0: 'Left', 1: 'Right', 2: 'Bail'}

pop_pseudohit_rows = []
pop_cos1_rows = []
pop_cos0_rows = []

for subj in list(sess_data_dict.keys()):
    model = get_matching_model(subj, best_models_dict)
    if model is None:
        continue

    weights = model.observations.params
    sess_dict = sess_data_dict[subj]
    sessions_list = list(sess_dict.values()) if isinstance(sess_dict, dict) else list(sess_dict)
    if not sessions_list:
        continue

    # 1. Pseudo-Hit Rates
    all_marginal_probs = []
    all_true_choices = []
    for sess in sessions_list:
        sess_inpt = sess['inputs']
        raw_choices = sess['choices']
        if isinstance(raw_choices[0], (str, np.str_)):
            true_choices = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices]).reshape(-1, 1)
        else:
            true_choices = np.array(raw_choices).astype(int).reshape(-1, 1)
        
        n_t = sess_inpt.shape[0]
        posterior_res = model.expected_states(true_choices, input=sess_inpt)
        posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res

        choice_probs = np.zeros((n_t, 2, 3))
        for k in range(2):
            sub_logits = sess_inpt @ weights[k].T
            full_logits = np.hstack([sub_logits, np.zeros((n_t, 1))])
            exp_logits = np.exp(full_logits - np.max(full_logits, axis=1, keepdims=True))
            choice_probs[:, k, :] = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        marginal = np.sum(posterior_probs[:, :, None] * choice_probs, axis=1)
        all_marginal_probs.append(marginal)
        all_true_choices.append(true_choices.flatten())

    if not all_marginal_probs:
        continue

    P_choice = np.vstack(all_marginal_probs)
    T_choice = np.concatenate(all_true_choices)
    log_probs = np.log(P_choice + 1e-12)

    subj_ph = {}
    for c_val, c_name in labels_map.items():
        mask = (T_choice == c_val)
        subj_ph[c_name] = np.exp(np.mean(log_probs[mask, c_val])) if np.sum(mask) > 0 else np.nan
    subj_ph['Mean'] = np.exp(np.mean(log_probs[np.arange(len(T_choice)), T_choice]))

    for cat in lrb_categories:
        pop_pseudohit_rows.append({'Subject': str(subj), 'Category': cat, 'Value': subj_ph[cat]})

    # 2. Cosine Similarities
    choice_sims_a1 = {0: [], 1: [], 2: []}
    all_sims_a1 = []
    sess_choice_sims_a0 = {0: [], 1: [], 2: []}
    sess_mean_sims_a0 = []

    for sess in sessions_list:
        sess_inpt = sess['inputs']
        raw_choices = sess['choices']
        if isinstance(raw_choices[0], (str, np.str_)):
            raw_choices_arr = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices]).ravel()
        else:
            raw_choices_arr = np.array(raw_choices).astype(int).ravel()
        true_choices = raw_choices_arr.reshape(-1, 1)

        posterior_res = model.expected_states(true_choices, input=sess_inpt)
        posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res

        n_trials = len(raw_choices_arr)
        choice_sims_sess = {0: [], 1: [], 2: []}
        all_sims_sess = []

        for t in range(n_trials):
            x_t = sess_inpt[t]
            sub_logits = np.array([x_t @ weights[k].T for k in range(2)])
            full_logits = np.hstack([sub_logits, np.zeros((2, 1))])
            exp_l = np.exp(full_logits - np.max(full_logits, axis=1, keepdims=True))
            p_k = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            pred_vec = posterior_probs[t] @ p_k

            true_vec = np.zeros(3)
            c = int(raw_choices_arr[t])
            true_vec[c] = 1.0

            norm_p = np.linalg.norm(pred_vec)
            norm_t = np.linalg.norm(true_vec)
            sim = np.dot(pred_vec, true_vec) / (norm_p * norm_t) if (norm_p > 0 and norm_t > 0) else 0.0

            choice_sims_a1[c].append(sim)
            all_sims_a1.append(sim)
            choice_sims_sess[c].append(sim)
            all_sims_sess.append(sim)

        for c in [0, 1, 2]:
            if len(choice_sims_sess[c]) > 0:
                sess_choice_sims_a0[c].append(np.mean(choice_sims_sess[c]))
        if len(all_sims_sess) > 0:
            sess_mean_sims_a0.append(np.mean(all_sims_sess))

    for c_val, c_name in labels_map.items():
        val_a1 = np.mean(choice_sims_a1[c_val]) if len(choice_sims_a1[c_val]) > 0 else np.nan
        pop_cos1_rows.append({'Subject': str(subj), 'Category': c_name, 'Value': val_a1})
    pop_cos1_rows.append({'Subject': str(subj), 'Category': 'Mean', 'Value': np.mean(all_sims_a1) if len(all_sims_a1) > 0 else np.nan})

    for c_val, c_name in labels_map.items():
        val_a0 = np.mean(sess_choice_sims_a0[c_val]) if len(sess_choice_sims_a0[c_val]) > 0 else np.nan
        pop_cos0_rows.append({'Subject': str(subj), 'Category': c_name, 'Value': val_a0})
    pop_cos0_rows.append({'Subject': str(subj), 'Category': 'Mean', 'Value': np.mean(sess_mean_sims_a0) if len(sess_mean_sims_a0) > 0 else np.nan})

df_ph = pd.DataFrame(pop_pseudohit_rows, columns=['Subject', 'Category', 'Value'])
df_c1 = pd.DataFrame(pop_cos1_rows, columns=['Subject', 'Category', 'Value'])
df_c0 = pd.DataFrame(pop_cos0_rows, columns=['Subject', 'Category', 'Value'])

if not df_ph.empty:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    fig.suptitle('POPULATION-WIDE 2-STATE MODEL SUMMARY: PSEUDO-HIT RATE AND COSINE SIMILARITY', fontsize=13, fontweight='bold', y=0.98)

    palette = {'Left': '#5C82E6', 'Right': '#FFA726', 'Bail': '#2E7D32', 'Mean': '#7E57C2'}

    # Panel 1: Pseudo-Hit Rates
    sns.barplot(data=df_ph, x='Category', y='Value', order=lrb_categories, palette=palette, ax=axes[0], 
                errorbar='se', capsize=0.1, err_kws={'linewidth': 1.5, 'color': '#333333'})
    axes[0].set_title('Pseudo-Hit Rates: Outcomes & Pooled Mean (K=2)', fontsize=11, fontweight='bold', pad=10)
    axes[0].set_ylabel('exp(Mean Log-Prob) ± SEM', fontsize=10, fontweight='bold')
    axes[0].set_xlabel('Choice Category', fontsize=10, fontweight='bold')
    axes[0].set_ylim([0.0, 1.05])
    axes[0].grid(True, axis='y', linestyle=':', alpha=0.6)

    # Panel 2: Cosine Sim Axis = 1
    sns.barplot(data=df_c1, x='Category', y='Value', order=lrb_categories, palette=palette, ax=axes[1], 
                errorbar='se', capsize=0.1, err_kws={'linewidth': 1.5, 'color': '#333333'})
    axes[1].set_title('Cosine Sim: Axis = 1 (Trial-Wise Mean)', fontsize=11, fontweight='bold', pad=10)
    axes[1].set_ylabel('Mean Cosine Similarity ± SEM', fontsize=10, fontweight='bold')
    axes[1].set_xlabel('Choice Category', fontsize=10, fontweight='bold')
    axes[1].set_ylim([0.0, 1.05])  # Fixed from [-0.05, 1.05]
    axes[1].grid(True, axis='y', linestyle=':', alpha=0.6)

    # Panel 3: Cosine Sim Axis = 0
    sns.barplot(data=df_c0, x='Category', y='Value', order=lrb_categories, palette=palette, ax=axes[2], 
                errorbar='se', capsize=0.1, err_kws={'linewidth': 1.5, 'color': '#333333'})
    axes[2].set_title('Cosine Sim: Axis = 0 (Timelines & Macro-Avg)', fontsize=11, fontweight='bold', pad=10)
    axes[2].set_ylabel('Timeline Cosine Similarity ± SEM', fontsize=10, fontweight='bold')
    axes[2].set_xlabel('Choice Category', fontsize=10, fontweight='bold')
    axes[2].set_ylim([0.0, 1.05])  # Fixed from [-0.05, 1.05]
    axes[2].grid(True, axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    print("Population-wide summary plot generated successfully!")

#%% Heatmap of Cosine Similarity by State and Rat's Choice for Subjects [237, 402, 424]

# Define your target subjects
target_subjects = [237, 402, 424]
K_states = globals().get('num_states_check', 2)

print("\n" + "="*60)
print("COMPUTING STATE-CHOICE COSINE SIMILARITIES FOR TARGET SUBJECTS")
print("="*60)

for subj in target_subjects:
    # Handle string/int key lookups safely
    subj_key = subj if subj in consistency_results else str(subj)
    if subj_key not in consistency_results:
        print(f"Subject {subj} not found in consistency_results. Skipping.")
        continue

    res = consistency_results[subj_key]
    if len(res['fits']) == 0:
        continue

    # Use the first fit (or reference fit)
    fit = res['fits'][0]
    
    # Gather inputs and choices across sessions for this subject
    subj_sessions = sess_data_dict.get(subj, sess_data_dict.get(str(subj), {}))
    if not subj_sessions:
        continue
        
    subj_inputs = [subj_sessions[s]['inputs'] for s in subj_sessions]
    subj_choices = [subj_sessions[s]['choices'] for s in subj_sessions]
    
    cat_inputs = np.vstack(subj_inputs)
    cat_choices = np.array([c for sess in subj_choices for c in np.array(sess).flatten()]).astype(int)

    n_trials = cat_inputs.shape[0]
    C = 3  # Number of choice categories (0: Left, 1: Right, 2: Bail)

    # Get state posteriors
    if 'posteriors' in fit:
        p_states = fit['posteriors'] # Shape: (n_trials, K)
    else:
        # Fallback if posteriors aren't stored in fit
        model = ssm.HMM(K_states, 1, len(ssm_predictors), 
                        observations="input_driven_obs", 
                        observation_kwargs=dict(C=C), 
                        transitions="standard")
        model.observations.params = fit['weights']
        safe_trans = np.clip(fit['trans_mat'], 1e-12, 1.0)
        model.transitions.params = np.log(safe_trans)[None, ...]
        
        posteriors_list = []
        for c_sess, inp_sess in zip(subj_choices, subj_inputs):
            c_arr = np.array(c_sess).astype(int).reshape(-1, 1)
            res_expected = model.expected_states(c_arr, input=inp_sess)
            posteriors_list.append(res_expected[0] if isinstance(res_expected, tuple) else res_expected)
        p_states = np.vstack(posteriors_list)

    # Compute state-conditional choice probabilities using einsum
    weights = fit['weights'] # Shape: (K, C-1, D)
    sub_logits = np.einsum('nd,kcd->nkc', cat_inputs, weights) # (n_trials, K, C-1)
    full_logits = np.concatenate([sub_logits, np.zeros((n_trials, K, 1))], axis=2)
    exp_logits = np.exp(full_logits - np.max(full_logits, axis=2, keepdims=True))
    choice_probs_k = exp_logits / np.sum(exp_logits, axis=2, keepdims=True) # Shape: (n_trials, K, C)

    # Compute trial-by-trial cosine similarity between each state's prediction and actual choice
    y_onehot = np.zeros((n_trials, C))
    y_onehot[np.arange(n_trials), cat_choices] = 1.0

    state_norms = np.linalg.norm(choice_probs_k, axis=2) # Shape: (n_trials, K)
    prob_actual = np.array([[choice_probs_k[t, k, cat_choices[t]] for k in range(K_states)] for t in range(n_trials)]) # (n_trials, K)
    
    trial_cos_sims = prob_actual / (state_norms + 1e-12) # Shape: (n_trials, K)

    # Aggregate: Average cosine similarity broken down by State and Rat's Choice Category
    sim_matrix = np.zeros((K_states, C))
    for k in range(K_states):
        for c in range(C):
            mask = (cat_choices == c)
            if np.sum(mask) > 0:
                sim_matrix[k, c] = np.average(trial_cos_sims[mask, k], weights=p_states[mask, k] + 1e-5)

    # Plot Heatmap with 1-indexed state labels ('State 1', 'State 2')
    plt.figure(figsize=(7, 4.5))
    ax = sns.heatmap(sim_matrix, annot=True, fmt=".3f", cmap="mako", cbar=True,
                     xticklabels=['Left', 'Right', 'Bail'],
                     yticklabels=[f"State {k+1}" for k in range(K_states)])
    
    plt.title(f"Subject {subj}: Mean Cosine Similarity (State Prediction vs Actual Choice)", fontsize=11, fontweight='bold')
    plt.xlabel("Rat's Actual Choice Category", fontsize=10)
    plt.ylabel("Latent State", fontsize=10)
    plt.tight_layout()
    plt.show()
    
#%% MODEL SELECTION PLOTTING SCRIPT

print("\n" + "="*60)
print("LOADING CACHED METRICS & GENERATING 4 MODEL SELECTION GRAPHS (LL, BIC, AIC)")
print("="*60)

# Configuration settings
cv_save_path = globals().get('cv_save_path', 'glm_hmm_cv_results.pkl')
full_metrics_save_path = globals().get('full_metrics_save_path', 'glm_hmm_full_metrics.pkl')
num_states_to_test = globals().get('num_states_to_test', [1, 2, 3, 4])
C_cat = globals().get('num_categories', 3)

available_subjects = list(sess_data_dict.keys()) if 'sess_data_dict' in globals() else []
example_subj = 237
if example_subj not in available_subjects and str(example_subj) in available_subjects:
    example_subj = str(example_subj)
elif example_subj not in available_subjects and available_subjects:
    example_subj = available_subjects[0]

# =====================================================================
# 1. Load Cross-Validated (LOOCV) Test Metrics from Cache
# =====================================================================
if 'cv_results' not in globals() or not cv_results:
    if os.path.exists(cv_save_path):
        with open(cv_save_path, 'rb') as f:
            cv_results = pickle.load(f)
        print(f"Loaded CV results from {cv_save_path}.")
    else:
        raise FileNotFoundError(f"Could not find CV cache at {cv_save_path}.")

cv_summary_rows = []
for subj_id, states_dict in cv_results.items():
    subj_sessions = sess_data_dict.get(subj_id, sess_data_dict.get(str(subj_id), {}))
    if not subj_sessions:
        continue
    total_trials = sum(len(sess['choices']) for sess in subj_sessions.values())
    
    for state_key, sessions_dict in states_dict.items():
        num_states = int(state_key.split('_')[0])
        input_dim = list(subj_sessions.values())[0]['inputs'].shape[1]
        k_params = (num_states - 1) + (num_states * (num_states - 1)) + (num_states * input_dim * (C_cat - 1))
        
        fold_test_lls = []
        for sess_id, inits_list in sessions_dict.items():
            if len(inits_list) > 0:
                best_init = max(inits_list, key=lambda x: x['test_ll'])
                fold_test_lls.append(best_init['test_ll'])
                
        if len(fold_test_lls) == len(sessions_dict):
            total_test_ll = sum(fold_test_lls)
            test_ll_per_trial = total_test_ll / total_trials
            test_aic = 2 * k_params - 2 * total_test_ll
            test_bic = k_params * np.log(total_trials) - 2 * total_test_ll
            
            cv_summary_rows.append({
                'subject': str(subj_id),
                'num_states': num_states,
                'LL_per_trial': test_ll_per_trial,
                'AIC': test_aic,
                'BIC': test_bic
            })

df_cv_summary = pd.DataFrame(cv_summary_rows)

# =====================================================================
# 2. Load Full Dataset (In-Sample) Metrics from Cache
# =====================================================================
full_summary_rows = []
if os.path.exists(full_metrics_save_path):
    with open(full_metrics_save_path, 'rb') as f:
        full_metrics_results = pickle.load(f)
    print(f"Loaded full dataset metrics from {full_metrics_save_path}.")
    
    for s, states_dict in full_metrics_results.items():
        for st, data in states_dict.items():
            if isinstance(data, dict) and 'train_ll_per_trials' in data and len(data['train_ll_per_trials']) > 0:
                full_summary_rows.append({
                    'subject': str(s),
                    'num_states': int(st),
                    'LL_per_trial': np.mean(data['train_ll_per_trials']),
                    'AIC': np.mean(data['aics']),
                    'BIC': np.mean(data['bics'])
                })
else:
    raise FileNotFoundError(f"Could not find full metrics cache at {full_metrics_save_path}.")

df_full_summary = pd.DataFrame(full_summary_rows)

# =====================================================================
# 3. Dual-Axis Plotting Helper Function (LL on Left, BIC & AIC on Right)
# =====================================================================
def plot_dual_axis_with_aic(data_df, title_text, is_population=False):
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    if is_population:
        grouped = data_df.groupby('num_states').agg({
            'LL_per_trial': ['mean', 'sem'],
            'BIC': ['mean', 'sem'],
            'AIC': ['mean', 'sem']
        }).reset_index()
        states = grouped['num_states']
        ll_mean, ll_err = grouped[('LL_per_trial', 'mean')], grouped[('LL_per_trial', 'sem')]
        bic_mean, bic_err = grouped[('BIC', 'mean')], grouped[('BIC', 'sem')]
        aic_mean, aic_err = grouped[('AIC', 'mean')], grouped[('AIC', 'sem')]
    else:
        states = data_df['num_states']
        ll_mean, ll_err = data_df['LL_per_trial'], np.zeros_like(data_df['LL_per_trial'])
        bic_mean, bic_err = data_df['BIC'], np.zeros_like(data_df['BIC'])
        aic_mean, aic_err = data_df['AIC'], np.zeros_like(data_df['AIC'])
    
    # Left Axis: Log-Likelihood / Trial (Higher is Better)
    color_ll = 'tab:blue'
    ax1.set_xlabel('Number of Latent States (K)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Log-Likelihood / Trial (Higher is Better)', color=color_ll, fontsize=11, fontweight='bold')
    
    if is_population:
        p1 = ax1.errorbar(states, ll_mean, yerr=ll_err, fmt='-o', color=color_ll, linewidth=2, capsize=4, label='Log-Likelihood')
    else:
        p1 = ax1.plot(states, ll_mean, '-o', color=color_ll, linewidth=2, label='Log-Likelihood')
        
    ax1.tick_params(axis='y', labelcolor=color_ll)
    ax1.set_xticks(num_states_to_test)
    ax1.grid(True, linestyle='--', alpha=0.4)
    
    # Right Axis: BIC & AIC Scores (Lower is Better)
    ax2 = ax1.twinx()
    color_bic = 'tab:red'
    color_aic = 'purple'
    ax2.set_ylabel('Information Criteria (BIC / AIC) - Lower is Better', color='black', fontsize=10, fontweight='bold')
    
    if is_population:
        p2 = ax2.errorbar(states, bic_mean, yerr=bic_err, fmt='--s', color=color_bic, linewidth=2, capsize=4, label='BIC')
        p3 = ax2.errorbar(states, aic_mean, yerr=aic_err, fmt=':^', color=color_aic, linewidth=2, capsize=4, label='AIC')
    else:
        p2 = ax2.plot(states, bic_mean, '--s', color=color_bic, linewidth=2, label='BIC')
        p3 = ax2.plot(states, aic_mean, ':^', color=color_aic, linewidth=2, label='AIC')
        
    ax2.tick_params(axis='y')
    
    # Combine legends from both axes
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center left', fontsize='small')
    
    plt.title(title_text, fontsize=12, fontweight='bold')
    fig.tight_layout()
    plt.show()

# =====================================================================
# 4. Generate the 4 Required Graphs Sequentially
# =====================================================================

# Graph 1: Example Subject - Cross-Validated (Test) Metrics
df_cv_ex = df_cv_summary[df_cv_summary['subject'] == str(example_subj)].sort_values('num_states')
if not df_cv_ex.empty:
    plot_dual_axis_with_aic(df_cv_ex, f"Subject {example_subj}: Cross-Validated Test Metrics (LOOCV)")

# Graph 2: Example Subject - Full Dataset (In-Sample) Metrics
df_full_ex = df_full_summary[df_full_summary['subject'] == str(example_subj)].sort_values('num_states')
if not df_full_ex.empty:
    plot_dual_axis_with_aic(df_full_ex, f"Subject {example_subj}: Full Dataset Metrics (In-Sample)")

# Graph 3: Population-Wide - Cross-Validated (Test) Metrics
if not df_cv_summary.empty:
    plot_dual_axis_with_aic(df_cv_summary, "Population-Wide: Cross-Validated Test Metrics (LOOCV)", is_population=True)

# Graph 4: Population-Wide - Full Dataset (In-Sample) Metrics
if not df_full_summary.empty:
    plot_dual_axis_with_aic(df_full_summary, "Population-Wide: Full Dataset Metrics (In-Sample)", is_population=True)
    
#%% 1-STATE VS 2-STATE PSEUDOHIT COMPARISON SCRIPT

print("\n" + "="*60)
print("RUNNING BULLETPROOF PSEUDOHIT COMPARISON SCRIPT")
print("="*60)

if 'best_models_dict' not in globals():
    best_models_dict = {}

comparison_choice_rows = []
comparison_overall_rows = []

choice_labels_dict = {0: 'Left', 1: 'Right', 2: 'Bail'}
choice_palette = {'Left': 'tab:blue', 'Right': 'tab:orange', 'Bail': 'tab:green'}
C_cat = globals().get('num_categories', 3)
choice_map = {'left': 0, 'right': 1, 'bail': 2, 'none': 2}

def compute_choice_probs(model, sessions_list, C_cat, K):
    w = model.observations.params
    all_probs = []
    
    for sess in sessions_list:
        sess_inpt = sess['inputs']
        n_t = sess_inpt.shape[0]
        
        state_logits = []
        for k in range(K):
            wk = w[k]
            sub_l = sess_inpt @ wk.T
            full_l = np.hstack([sub_l, np.zeros((n_t, 1))])
            state_logits.append(full_l)
            
        state_logits = np.array(state_logits) # Shape: (K, n_t, C_cat)
        
        max_l = np.max(state_logits, axis=2, keepdims=True)
        exp_l = np.exp(state_logits - max_l)
        state_probs = exp_l / np.sum(exp_l, axis=2, keepdims=True) # Shape: (K, n_t, C_cat)
        
        if K == 1:
            all_probs.append(state_probs[0])
        else:
            sess_ch = sess['choices']
            if isinstance(sess_ch[0], (str, np.str_)):
                sess_t_ch = np.array([choice_map.get(str(c).lower(), 2) for c in sess_ch]).reshape(-1, 1)
            else:
                sess_t_ch = np.array(sess_ch).astype(int).reshape(-1, 1)
                
            posterior_res = model.expected_states(sess_t_ch, input=sess_inpt)
            posterior_probs = posterior_res[0] if isinstance(posterior_res, tuple) else posterior_res
            
            sp_t = np.transpose(state_probs, (1, 0, 2)) # Shape: (n_t, K, C_cat)
            marginal = np.sum(posterior_probs[:, :, None] * sp_t, axis=1) # Shape: (n_t, C_cat)
            all_probs.append(marginal)
            
    return np.vstack(all_probs)

for subj in sess_data_dict.keys():
    print(f"Processing Subject {subj}...")
    subj_data = sess_data_dict[subj]
    sessions_list = list(subj_data.values()) if isinstance(subj_data, dict) else list(subj_data)
    
    subj_inputs = np.vstack([sess['inputs'] for sess in sessions_list])
    
    raw_choices_all = []
    for sess in sessions_list:
        raw_choices_all.extend(sess['choices'])
        
    if isinstance(raw_choices_all[0], (str, np.str_)):
        true_choices = np.array([choice_map.get(str(c).lower(), 2) for c in raw_choices_all]).ravel().astype(int)
    else:
        true_choices = np.array(raw_choices_all).ravel().astype(int)
        
    n_trials = len(true_choices)
    input_dim = subj_inputs.shape[1]

    if subj not in best_models_dict:
        best_models_dict[subj] = {}

    models = {}
    for K in [1, 2]:
        if K in best_models_dict[subj] and best_models_dict[subj][K] is not None:
            models[K] = best_models_dict[subj][K]
        else:
            print(f"  -> Fitting K={K} model for Subject {subj}...")
            try:
                m = ssm.HMM(K, 1, input_dim, observations="input_driven_obs", 
                            observation_kwargs=dict(C=C_cat), transitions="standard")
                
                train_inputs_list = [sess['inputs'] for sess in sessions_list]
                train_choices_list = []
                for sess in sessions_list:
                    sess_ch = sess['choices']
                    if isinstance(sess_ch[0], (str, np.str_)):
                        ch_arr = np.array([choice_map.get(str(c).lower(), 2) for c in sess_ch])
                    else:
                        ch_arr = np.array(sess_ch).astype(int)
                    train_choices_list.append(ch_arr.reshape(-1, 1))
                
                m.fit(train_choices_list, inputs=train_inputs_list, method="em", num_iters=100, tolerance=1e-3)
                best_models_dict[subj][K] = m
                models[K] = m
            except Exception as e:
                print(f"  ❌ Error fitting K={K} for Subject {subj}: {e}")
                models[K] = None

    if models.get(1) is None or models.get(2) is None:
        print(f"  Skipping Subject {subj}: Could not secure K=1 and K=2 models.")
        continue

    # Compute choice probabilities
    probs_1 = compute_choice_probs(models[1], sessions_list, C_cat, K=1)
    probs_2 = compute_choice_probs(models[2], sessions_list, C_cat, K=2)

    log_probs_1 = np.log(probs_1 + 1e-12)
    log_probs_2 = np.log(probs_2 + 1e-12)
    
    # Choice-Specific Breakdown (using 1D boolean mask indexing)
    for c_val, c_name in choice_labels_dict.items():
        mask = (true_choices == c_val)
        if np.sum(mask) > 0:
            comparison_choice_rows.append({
                'Subject': str(subj),
                'Choice': c_name,
                'K1_pseudohit': np.exp(np.mean(log_probs_1[mask, c_val])),
                'K2_pseudohit': np.exp(np.mean(log_probs_2[mask, c_val]))
            })
            
    # Overall Average
    comparison_overall_rows.append({
        'Subject': str(subj),
        'K1_pseudohit': np.exp(np.mean(log_probs_1[np.arange(n_trials), true_choices])),
        'K2_pseudohit': np.exp(np.mean(log_probs_2[np.arange(n_trials), true_choices]))
    })

df_comp_choice = pd.DataFrame(comparison_choice_rows)
df_comp_overall = pd.DataFrame(comparison_overall_rows)

print(f"\nSuccessfully processed {len(df_comp_overall)} subjects for comparison plots.")

# =====================================================================
# PLOTTING
# =====================================================================
if not df_comp_choice.empty:
    plt.figure(figsize=(7, 6))
    seaborn_plt.scatterplot(
        data=df_comp_choice, x='K1_pseudohit', y='K2_pseudohit',
        hue='Choice', style='Subject', palette=choice_palette, s=120, alpha=0.9
    )
    all_c_vals = pd.concat([df_comp_choice['K1_pseudohit'], df_comp_choice['K2_pseudohit']])
    min_c, max_c = all_c_vals.min() * 0.95, all_c_vals.max() * 1.05
    plt.plot([min_c, max_c], [min_c, max_c], 'k--', alpha=0.6, label='y = x (Equal Performance)')
    plt.xlabel('1-State Model Pseudohit Rate', fontsize=11, fontweight='bold')
    plt.ylabel('2-State Model Pseudohit Rate', fontsize=11, fontweight='bold')
    plt.title('Choice-Specific Pseudohit Rates: 1-State vs. 2-State Model', fontsize=12, fontweight='bold')
    plt.xlim(min_c, max_c)
    plt.ylim(min_c, max_c)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()

if not df_comp_overall.empty:
    plt.figure(figsize=(6, 6))
    seaborn_plt.scatterplot(
        data=df_comp_overall, x='K1_pseudohit', y='K2_pseudohit',
        hue='Subject', s=160, palette='deep', alpha=0.9
    )
    for _, row in df_comp_overall.iterrows():
        plt.text(row['K1_pseudohit'] + 0.003, row['K2_pseudohit'], f"Subj {row['Subject']}", fontsize=10, fontweight='bold')
    all_o_vals = pd.concat([df_comp_overall['K1_pseudohit'], df_comp_overall['K2_pseudohit']])
    min_o, max_o = all_o_vals.min() * 0.95, all_o_vals.max() * 1.05
    plt.plot([min_o, max_o], [min_o, max_o], 'k--', alpha=0.6, label='y = x (Equal Performance)')
    plt.xlabel('1-State Model Overall Pseudohit Rate', fontsize=11, fontweight='bold')
    plt.ylabel('2-State Model Overall Pseudohit Rate', fontsize=11, fontweight='bold')
    plt.title('Overall Average Pseudohit Rates: 1-State vs. 2-State Model', fontsize=12, fontweight='bold')
    plt.xlim(min_o, max_o)
    plt.ylim(min_o, max_o)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()