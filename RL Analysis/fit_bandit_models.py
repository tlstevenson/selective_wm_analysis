# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:35:16 2024

@author: tanne
"""

# %% Imports

import init
import pandas as pd
from pyutils import utils
import hankslab_db.basicRLtasks_db as db
from hankslab_db import db_access
import beh_analysis_helpers as bah
import fp_analysis_helpers as fpah
from fp_analysis_helpers import Alignment as Align
from sys_neuro_tools import plot_utils, fp_utils
from modeling import agents
import modeling.training_helpers as th
import modeling.sim_helpers as sh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sb
from os import path
import pickle

# %% Load behavioral data

subj_ids = [179, 188, 191, 207] # 182

save_path = r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\fit_models_masked.json'
if path.exists(save_path):
    all_models = agents.load_model(save_path)
else:
    all_models = {}

# load data
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=2)

# start from the third session (so index=2)-->do not account for the first two sessions
sess_ids = {subj: sess[2:] for subj, sess in sess_ids.items()}

# get session data
reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

all_sess = th.define_choice_outcome(all_sess)

# %% Run Fitting

skip_existing_fits = True
refit_existing = False
print_train_params = False
limit_mask = True
n_limit_hist = 2

n_fits = 1
n_steps = 10000
end_tol = 5e-6

for subj in subj_ids: 
    print("\nSubj", subj)
        
    sess_data = all_sess[all_sess['subjid'] == subj]
    
    training_data = th.get_model_training_data(sess_data, limit_mask=limit_mask, n_limit_hist=n_limit_hist)

    basic_inputs = training_data['basic_inputs']
    two_side_inputs = training_data['two_side_inputs']
    left_choice_labels = training_data['left_choice_labels']
    choice_class_labels = training_data['choice_class_labels']
    trial_mask_train = training_data['trial_mask_train']
    trial_mask_eval = training_data['trial_mask_eval']
    
    ## BASIC MODEL FITS
    th.fit_basic_models(basic_inputs, left_choice_labels, trial_mask_train, trial_mask_eval, subj, save_path, n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, 
                        skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, print_train_params=print_train_params)
    
    ## Q-MODEL FITS

    # declare model fit settings
    settings = {# All Alphas Free, Different K Fixes
                # 'All Free': {},
                
                # 'All Alpha Free, S/R K Fixed': {'k_same_rew': {'fit': False}},
                
                # 'All Alpha Free, Same K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                # 'All Alpha Free, All K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'All Alpha Free, D/R K Free': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
            
                # 'All Alpha Free, All K Fixed, Diff K=0.5': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                             'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
                # # All Alphas shared, Different K Fixes
                'All Alpha Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                                                  'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                                                  'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'All Alpha Shared, D/R K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                   'alpha_diff_unrew': {'share': 'alpha_same_rew'},'k_same_rew': {'fit': False}, 
                #                                   'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'All Alpha Shared, Same K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                    'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                # 'All Alpha Shared, All K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_rew'}},
                
                # 'All Alpha Shared, All K Fixed, Diff K=0.5': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                               'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
                
                # # Models with limited different choice updating
                # 'Same Alpha Only Shared, K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                #                                'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Only Shared, Counter K': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                #                                'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Shared, K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 
                #                               'alpha_diff_unrew': {'fit': False, 'init': 0}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Only, K Fixed': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                #                              'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Only, K Free': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                #                             'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Only, Counter K': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                #                                'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 
                #                                'k_diff_unrew': {'fit': False}},
                
                # 'No Alpha D/U, All K Fixed': {'alpha_diff_unrew': {'fit': False, 'init': 0}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'No Alpha D/R, All K Fixed': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                
                # # Constrained Alpha Pairs
                # 'Alpha Same/Diff Shared, Same K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                # 'Alpha Same/Diff Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                #                                          'k_diff_unrew': {'fit': False}},
                
                # 'Alpha Same/Diff Shared, D/R K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                        'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Alpha Rew/Unrew Shared, Same K Fixed': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
                #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                # 'Alpha Rew/Unrew Shared, All K Fixed': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
                #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                #                                          'k_diff_unrew': {'fit': False}},
                
                # 'Alpha Rew/Unrew Shared, D/R K Free': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
                #                                        'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                
                # # Counterfactual models
                # 'All Alpha Shared, Counter D/U K=1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                       'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 
                #                                       'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False, 'init': 1}},
                
                # 'Alpha Same/Diff Shared, Counter D/U K=1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                #                                             'k_diff_unrew': {'fit': False, 'init': 1}},
                
                # 'All Alpha Shared, Counter D/R K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                       'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 
                #                                       'k_diff_rew': {'fit': False, 'init': -1}, 'k_diff_unrew': {'fit': False}},
                
                # 'Alpha Same/Diff Shared, Counter D/R K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False, 'init': -1}, 
                #                                             'k_diff_unrew': {'fit': False}},
                
                # 'All Alpha Shared, Counter S/U K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                       'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                #                                       'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Alpha Same/Diff Shared, Counter S/U K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 
                #                                             'k_diff_unrew': {'fit': False}},
                
                }

    add_agent_dict = {'': [], 
                      # 'Persev': [agents.PerseverativeAgent(n_vals=2)],
                      # 'Fall': [agents.FallacyAgent(n_vals=2)]
                      }
    agent_gen = lambda s: agents.QValueAgent(constraints=s)
    
    th.fit_two_side_model(agent_gen, 'Q', settings, two_side_inputs, choice_class_labels, trial_mask_train, trial_mask_eval, subj, save_path, 
                       n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, add_agent_dict=add_agent_dict, 
                       skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, print_train_params=print_train_params)
    
                    
    ## DYNAMIC Q-MODEL FITS

    # # declare model fit settings
    # settings = {
    #             'All Alpha Shared, All K Fixed, Global λ': {'global_lam': True, 'constraints': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
    #                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
    #                                               'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}},
                
    #             }

    # agent_gen = lambda s: agents.DynamicQAgent(**s)
    
    # th.fit_two_side_model(agent_gen, 'Dynamic Q', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
    #                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
    #                    print_train_params=print_train_params)
    
    ## UNCERTAINTY DYNAMIC Q-MODEL FITS

    # # declare model fit settings
    # settings = {
    #             'All Alpha Shared, All K Fixed, Global λ': {'global_lam': True, 'constraints': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
    #                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
    #                                               'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}},
                
    #             }

    # agent_gen = lambda s: agents.UncertaintyDynamicQAgent(**s)
    
    # th.fit_two_side_model(agent_gen, 'Uncertainty Dynamic Q', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
    #                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
    #                    print_train_params=print_train_params)
            
    ## Q-VALUE STATE INFERENCE MODEL FITS

    # settings = {'Alpha Free, K Free': {},
                
    #             # 'Alpha Free, K Free, Value Update First': {'update_order': 'value_first'},
                
    #             # 'Alpha Free, K Free, Belief Update First': {'update_order': 'belief_first'},
        
    #             # 'Alpha Free, K Fixed': {'constraints': {'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False},
    #             #                                         'k_low_rew': {'fit': False}, 'k_low_unrew': {'fit': False}}},
                
    #             # 'All Alpha Shared, K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 'alpha_low_rew': {'share': 'alpha_high_rew'}, 
    #             #                                              'alpha_low_unrew': {'share': 'alpha_high_rew'}}},
                
    #             # 'Shared High Alpha, Const Low K, High K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
    #             #                                                                 'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
    #             #                                                                 'k_low_unrew': {'share': 'k_low_rew', 'init': None}}},
                
    #             # 'Shared High Alpha, Fixed Low K, High K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
    #             #                                                                 'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
    #             #                                                                 'k_low_unrew': {'share': 'k_low_rew', 'fit': False, 'init': 0.1}}},
                
    #             # 'Separate High Alphas, Const Low K, High K Free': {'constraints': {'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
    #             #                                                                    'k_low_unrew': {'share': 'k_low_rew', 'init': None}}},
                
    #             # 'Separate Low Alphas, Const High K, Low K Free': {'constraints': {'alpha_high_rew': {'share': 'alpha_high_unrew', 'fit': False, 'init': 0}, 
    #             #                                                                    'k_high_unrew': {'share': 'k_high_rew', 'init': None}}},
                
    #             # 'Shared High Alpha, Const Low K, High K Fixed': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
    #             #                                                                  'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
    #             #                                                                  'k_low_unrew': {'share': 'k_low_rew', 'init': None}, 
    #             #                                                                  'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False}}},
                
    #             # 'Separate High Alphas, Const Low K, High K Fixed': {'constraints': {'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
    #             #                                                                     'k_low_unrew': {'share': 'k_low_rew', 'init': None}, 
    #             #                                                                     'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False}}}
    #             }

    # agent_gen = lambda s: agents.QValueStateInferenceAgent(**s)
    
    # th.fit_two_side_model(agent_gen, 'Q SI', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
    #                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
    #                    print_train_params=print_train_params)
    
    
    # ## RL STATE INFERENCE MODEL FITS

    # settings = {
    #             # 'Shared Evidence': {'complement_c_rew':True, 'complement_c_diff':True}, 
    #             # 'Separate Same/Diff Evidence': {'complement_c_rew':True, 'complement_c_diff':False},
    #             # 'Separate Rew/Unrew Evidence': {'complement_c_rew':False, 'complement_c_diff':True}, 
    #             # 'All Separate Evidence': {'complement_c_rew':False, 'complement_c_diff':False},
    #             'Free Same/Diff Rew Evidence': {'complement_c_rew':False, 'complement_c_diff':False,
    #                                             'constraints': {'c_same_unrew': {'fit': False, 'init': 0}, 
    #                                                             'c_diff_unrew': {'fit': False, 'init': 0}}}
    #             }
    
    # agent_gen = lambda s: agents.RLStateInferenceAgent(**s)
    
    # th.fit_two_side_model(agent_gen, 'RL SI', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
    #                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
    #                    print_train_params=print_train_params)
    
    
    ## STATE INFERENCE MODEL FITS

    settings = {
                # 'Shared Evidence': {'complement_c_rew':True, 'complement_c_diff':True}, 
                # 'Separate Same/Diff Evidence': {'complement_c_rew':True, 'complement_c_diff':False},
                'Separate Rew/Unrew Evidence': {'complement_c_rew':False, 'complement_c_diff':True}, 
                # 'All Separate Evidence': {'complement_c_rew':False, 'complement_c_diff':False},
                # 'Free Same/Diff Rew Evidence': {'complement_c_rew':False, 'complement_c_diff':False,
                #                                 'constraints': {'c_same_unrew': {'fit': False, 'init': 0}, 
                #                                                 'c_diff_unrew': {'fit': False, 'init': 0}}}
                }
    
    agent_gen = lambda s: agents.StateInferenceAgent(**s)
    
    th.fit_two_side_model(agent_gen, 'SI', settings, two_side_inputs, choice_class_labels, trial_mask_train, trial_mask_eval, subj, save_path, 
                       n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
                       print_train_params=print_train_params)
    
                
    # ## FULL BAYESIAN MODEL FITS

    # p_step = 0.01
    
    # settings = {
    #             # 'No Switch Scatter, Switch Update First': {'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False}}}, 
    #             'No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates': 
    #                 {'update_p_switch_first': False,
    #                  'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False},
    #                                  'imperfect_update_alpha': {'init': 0, 'fit': False},
    #                                  'stay_bias_lam': {'init': 0, 'fit': False}}}, 
    #             # 'Fixed 0.5 Prior Rew Mean, No Switch Scatter': {'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False}, 
    #             #                                                          'init_high_rew_mean': {'init': 0, 'fit': False}, 
    #             #                                                          'init_low_rew_mean': {'init': 1, 'fit': False}}}
    #             }
    
    # agent_gen = lambda s: agents.BayesianAgent(p_step=p_step, **s)
    
    # th.fit_two_side_model(agent_gen, 'Bayes', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
    #                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
    #                    print_train_params=print_train_params)
    
            