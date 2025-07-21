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
from models import agents
import models.training_helpers as th
import models.sim_helpers as sh
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

save_path = r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\fit_models_CELoss.json'
if path.exists(save_path):
    all_models = agents.load_model(save_path)
else:
    all_models = {}

# load data
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=2)

### EDIT: start from the third session (so index=2)-->do not account for the first two sessions
sess_ids = {subj: sess[2:] for subj, sess in sess_ids.items()}

# get session data
reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
all_sess = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)

# add needed columns
## add a column to represent choice (+1 left and -1 right) and outcome (+1 rewarded and -1 unrewarded)
choice_input_dict = {'left': 1, 'right': -1}
outcome_dict = {False: -1, True: 1}

# Add choice and outcome inputs as new columns to sess data
all_sess['choice_inputs'] = all_sess['choice'].apply(lambda choice: choice_input_dict[choice] if choice in choice_input_dict.keys() else np.nan).to_numpy()
all_sess['outcome_inputs'] = all_sess['rewarded'].apply(lambda reward: outcome_dict[reward] if reward in outcome_dict.keys() else np.nan).to_numpy()
all_sess['rewarded_int'] = all_sess['rewarded'].astype(int)
all_sess['chose_left'] = all_sess['chose_left'].astype(int)
all_sess['chose_right'] = all_sess['chose_right'].astype(int)


# %% Run Fitting

#torch.autograd.set_detect_anomaly(False)

plot_fits = False
skip_existing_fits = True
refit_existing = False

n_fits = 1
n_steps = 10000
end_tol = 5e-6

for subj in subj_ids: 
    print("\nSubj", subj)
    
    if not str(subj) in all_models:
        all_models[str(subj)] = {}
        
    sess_data = all_sess[all_sess['subjid'] == subj]
    # filter out no responses
    sess_data = sess_data[sess_data['hit']==True]
    
    ## Create 3-D inputs tensor and 3-D labels tensor
    n_sess = len(sess_ids[subj])

    max_trials = np.max(sess_data.groupby('sessid').size())

    ## FIT BASIC MODEL
    
    #create empty tensors with the max number of trials across all sessions
    inputs = torch.zeros(n_sess, max_trials-1, 2)
    ### EDIT: changed to left choice from right choice labels
    left_choice_labels = torch.zeros(n_sess, max_trials-1, 1)
    
    trial_mask = torch.zeros_like(left_choice_labels)
    
    # populate tensors from behavioral data
    for i, sess_id in enumerate(sess_ids[subj]):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data) - 1 # one less because we predict the next choice based on the prior choice
        
        inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['choice_inputs'][:-1], trial_data['outcome_inputs'][:-1]]).T).type(torch.float)
        ### EDIT:changed to left choice from right choice labels
        left_choice_labels[i, :n_trials, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)
        trial_mask[i, :n_trials, :] = 1
    
    basic_model_agents = [[agents.SingleValueAgent()], [agents.SingleValueAgent(), agents.PerseverativeAgent()], [agents.SingleValueAgent(), agents.FallacyAgent()],
                          [agents.SingleValueAgent(), agents.PerseverativeAgent(), agents.FallacyAgent()]]
    basic_model_names = ['Basic - Value', 'Basic - Value/Persev', 'Basic - Value/Fall', 'Basic - Value/Persev/Fall']
    
    for model_agents, model_name in zip(basic_model_agents, basic_model_names):
        if not model_name in all_models[str(subj)]:
            if refit_existing:
                continue
            else:
                all_models[str(subj)][model_name] = []
                n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            elif refit_existing:
                n_model_fits = len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits

        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            # fit the model
            if refit_existing:
                fit_model = all_models[str(subj)][model_name][i]['model']
            else:
                fit_model = agents.SummationModule(model_agents)
                fit_model.reset_params()
            
            loss = nn.BCEWithLogitsLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, left_choice_labels, n_steps, 
                                           trial_mask=trial_mask, loss_diff_exit_thresh=end_tol)
                
                #print("loss vals", loss_vals)
                
                # evaluate the fit
                fit_output, agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                                   output_transform=lambda x: 1/(1+np.exp(-x)))
                
                print('\nFit Model Params:')
                print(fit_model.print_params())
                
                print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
                print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
                print('Avg LL: {:.5f}'.format(fit_perf['ll_avg']))
                
                # plot fit results
                if plot_fits:
                    th.plot_single_val_fit_results(sess_data, fit_output, agent_states[:,:,0,:], 
                                        ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], betas=fit_model.beta.weight[0],
                                        title_prefix= 'Subj {}: Basic Model {} - '.format(subj, i))
                    plt.show(block=False)
                
                if refit_existing:
                    all_models[str(subj)][model_name][i] = {'model': fit_model, 'perf': fit_perf}
                else:
                    all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                
                agents.save_model(all_models, save_path)
                
                i += 1
                
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))
    
    ## Q-MODEL FITS

    ##### EDIT ####### Q-model    
    #extended inputs to have three dimensions for Q-learning model
    inputs = torch.zeros(n_sess, max_trials-1, 3)
    #created new choie class labels
    choice_class_labels = torch.zeros(n_sess, max_trials-1, 1).type(torch.long)
    
    #Iterate through every individual session
    # populate tensors from behavioral data
    for i, sess_id in enumerate(sess_ids[subj]):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data) - 1 # one less because we predict the next choice based on the prior choice
        
        inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['chose_left'][:-1], trial_data['chose_right'][:-1], trial_data['rewarded_int'][:-1]]).T).type(torch.float)
        #EDIT: created class labels based on "chose left" instead of choice right
        choice_class_labels[i, :n_trials, :] = torch.from_numpy(np.array(trial_data['chose_right'][1:])[:,None]).type(torch.long)


    # declare model fit settings
    settings = {# All Alphas Free, Different K Fixes
                'All Free': {},
                
                # 'All Alpha Free, S/R K Fixed': {'k_same_rew': {'fit': False}},
                
                # 'All Alpha Free, Same K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                # 'All Alpha Free, All K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'All Alpha Free, D/R K Free': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
            
                # 'All Alpha Free, All K Fixed, Diff K=0.5': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                             'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
                # # All Alphas shared, Different K Fixes
                # 'All Alpha Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                   'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                #                                   'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
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

    add_model_agents = [[]] #[[], [agents.PerseverativeAgent(n_vals=2)], [agents.FallacyAgent(n_vals=2)]]
    model_prefixes = ['Q'] #['Q', 'Q/Persev', 'Q/Fall']
    agent_names = [['Value']] # [['Value'], ['Value', 'Persev'], ['Value', 'Fallacy']]
    
    for add_agents, prefix, names in zip(add_model_agents, model_prefixes, agent_names):
        for label, s in settings.items():
            model_name = '{} - {}'.format(prefix, label)
            
            if not model_name in all_models[str(subj)]:
                if refit_existing:
                    continue
                else:
                    all_models[str(subj)][model_name] = []
                    n_model_fits = n_fits
            else:
                if skip_existing_fits:
                    n_model_fits = n_fits - len(all_models[str(subj)][model_name])
                elif refit_existing:
                    n_model_fits = len(all_models[str(subj)][model_name])
                else:
                    n_model_fits = n_fits
            
            i = 0
            while i < n_model_fits:
                print('\n{}, fit {}\n'.format(model_name, i))
    
                if refit_existing:
                    fit_model = all_models[str(subj)][model_name][i]['model']
                else:
                    #fit the model formulated in the same way as the single value model
                    q_agent = agents.QValueAgent(alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
                                                  k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, 
                                                  constraints=s)
                    
                    fit_model = agents.SummationModule([q_agent]+add_agents) #, agents.PerseverativeAgent(n_vals=2), agents.FallacyAgent(n_vals=2)
                    fit_model.reset_params()
                
                loss = nn.CrossEntropyLoss(reduction='none')
                
                optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
                
                try:
                    # need to reshape the output to work with the cross entropy loss function
                    loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                                               output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1), loss_diff_exit_thresh=end_tol)
                    
                    # evaluate the fit
                    fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                                           output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
                    
                    print('\n{} Params:'.format(model_name))
                    print(fit_model.print_params())
                    
                    print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
                    print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
                    print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
                
                    # plot fit results
                    if plot_fits:
                        th.plot_multi_val_fit_results(sess_data, fit_output,  fit_agent_states, 
                                                      names, ['Left', 'Right'], betas=fit_model.beta.weight[0], 
                                                      title_prefix='Subi {}: Fit Model {} - '.format(subj, '{} {}'.format(model_name, i)))
                        plt.show(block=False)
                    
                    if refit_existing:
                        all_models[str(subj)][model_name][i] = {'model': fit_model, 'perf': fit_perf}
                    else:
                        all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                    
                    agents.save_model(all_models, save_path)
                    i += 1
                    
                except RuntimeError as e:
                    print('Error: {}. \nTrying Again...'.format(e))
                    
                    
    ## DYNAMIC Q-MODEL FITS

    # # declare model fit settings
    # settings = {
    #             'All Alpha Shared, All K Fixed, Global λ': {'global_lam': True, 'constraints': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
    #                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
    #                                               'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}},
                
    #             }

    # add_model_agents = [[]]#, [agents.PerseverativeAgent(n_vals=2)], [agents.FallacyAgent(n_vals=2)]]
    # model_prefixes = ['Q']#, 'Q/Persev', 'Q/Fall']
    # agent_names = [['Value'], ['Value', 'Persev'], ['Value', 'Fallacy']]
    
    # for add_agents, prefix, names in zip(add_model_agents, model_prefixes, agent_names):
    #     for label, s in settings.items():
    #         model_name = '{} - {}'.format(prefix, label)
            
    #         if not model_name in all_models[str(subj)]:
    #             all_models[str(subj)][model_name] = []
    #             n_model_fits = n_fits
    #         else:
    #             if skip_existing_fits:
    #                 n_model_fits = n_fits - len(all_models[str(subj)][model_name])
    #             else:
    #                 n_model_fits = n_fits
            
    #         i = 0
    #         while i < n_model_fits:
    #             print('\n{}, fit {}\n'.format(model_name, i))
    
    #             #fit the model formulated in the same way as the single value model
    #             q_agent = agents.QValueAgent(alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
    #                                           k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, 
    #                                           constraints=s)
                
    #             fit_model = agents.SummationModule([q_agent]+add_agents) #, agents.PerseverativeAgent(n_vals=2), agents.FallacyAgent(n_vals=2)
    #             fit_model.reset_params()
                
    #             loss = nn.CrossEntropyLoss(reduction='none')
                
    #             optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
                
    #             try:
    #                 # need to reshape the output to work with the cross entropy loss function
    #                 loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
    #                                            output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1))
                    
    #                 # evaluate the fit
    #                 fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, trial_mask=trial_mask,
    #                                                                        output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
                    
    #                 print('\n{} Params:'.format(model_name))
    #                 print(fit_model.print_params())
                    
    #                 print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
    #                 print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
    #                 print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
                
    #                 # plot fit results
    #                 if plot_fits:
    #                     th.plot_multi_val_fit_results(sess_data, fit_output,  fit_agent_states, 
    #                                                   names, ['Left', 'Right'], betas=fit_model.beta.weight[0], 
    #                                                   title_prefix='Subi {}: Fit Model {} - '.format(subj, '{} {}'.format(model_name, i)))
    #                     plt.show(block=False)
                    
    #                 all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                    
    #                 agents.save_model(all_models, save_path)
    #                 i += 1
                    
    #             except RuntimeError as e:
    #                 print('Error: {}. \nTrying Again...'.format(e))
            
    ## Q-VALUE STATE INFERENCE MODEL FITS

    settings = {'Alpha Free, K Free': {},
                
                # 'Alpha Free, K Free, Value Update First': {'update_order': 'value_first'},
                
                # 'Alpha Free, K Free, Belief Update First': {'update_order': 'belief_first'},
        
                # 'Alpha Free, K Fixed': {'constraints': {'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False},
                #                                         'k_low_rew': {'fit': False}, 'k_low_unrew': {'fit': False}}},
                
                # 'All Alpha Shared, K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 'alpha_low_rew': {'share': 'alpha_high_rew'}, 
                #                                              'alpha_low_unrew': {'share': 'alpha_high_rew'}}},
                
                # 'Shared High Alpha, Const Low K, High K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
                #                                                                 'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
                #                                                                 'k_low_unrew': {'share': 'k_low_rew', 'init': None}}},
                
                # 'Shared High Alpha, Fixed Low K, High K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
                #                                                                 'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
                #                                                                 'k_low_unrew': {'share': 'k_low_rew', 'fit': False, 'init': 0.1}}},
                
                # 'Separate High Alphas, Const Low K, High K Free': {'constraints': {'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
                #                                                                    'k_low_unrew': {'share': 'k_low_rew', 'init': None}}},
                
                # 'Separate Low Alphas, Const High K, Low K Free': {'constraints': {'alpha_high_rew': {'share': 'alpha_high_unrew', 'fit': False, 'init': 0}, 
                #                                                                    'k_high_unrew': {'share': 'k_high_rew', 'init': None}}},
                
                # 'Shared High Alpha, Const Low K, High K Fixed': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
                #                                                                  'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
                #                                                                  'k_low_unrew': {'share': 'k_low_rew', 'init': None}, 
                #                                                                  'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False}}},
                
                # 'Separate High Alphas, Const Low K, High K Fixed': {'constraints': {'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
                #                                                                     'k_low_unrew': {'share': 'k_low_rew', 'init': None}, 
                #                                                                     'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False}}}
                }

    
    for label, s in settings.items():
        model_name = 'Q SI - ' + label

        if not model_name in all_models[str(subj)]:
            if refit_existing:
                continue
            else:
                all_models[str(subj)][model_name] = []
                n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            elif refit_existing:
                n_model_fits = len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits
        
        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            if refit_existing:
                fit_model = all_models[str(subj)][model_name][i]['model']
            else:
                #fit the model formulated in the same way as the single value model
                si_agent = agents.QValueStateInferenceAgent(**s)
                
                fit_model = agents.SummationModule([si_agent])
            
            loss = nn.CrossEntropyLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                # need to reshape the output to work with the cross entropy loss function
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                                           output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1), loss_diff_exit_thresh=end_tol)
            
                # evaluate the fit
                fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                                       output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
                
                print('\n{} Params:'.format(model_name))
                print(fit_model.print_params())
                
                print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
                print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
                print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
                
                if refit_existing:
                    all_models[str(subj)][model_name][i] = {'model': fit_model, 'perf': fit_perf}
                else:
                    all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                
                agents.save_model(all_models, save_path)
                
                i += 1
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))
    
    ## RL STATE INFERENCE MODEL FITS

    settings = {
                'Shared Evidence': {'complement_c_rew':True, 'complement_c_diff':True}, 
                # 'Separate Same/Diff Evidence': {'complement_c_rew':True, 'complement_c_diff':False},
                # 'Separate Rew/Unrew Evidence': {'complement_c_rew':False, 'complement_c_diff':True}, 
                # 'All Separate Evidence': {'complement_c_rew':False, 'complement_c_diff':False},
                # 'Free Same/Diff Rew Evidence': {'complement_c_rew':False, 'complement_c_diff':False,
                #                                 'constraints': {'c_same_unrew': {'fit': False, 'init': 0}, 
                #                                                 'c_diff_unrew': {'fit': False, 'init': 0}}}
                }
    
    for label, s in settings.items():
        model_name = 'RL SI - ' + label

        if not model_name in all_models[str(subj)]:
            if refit_existing:
                continue
            else:
                all_models[str(subj)][model_name] = []
                n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            elif refit_existing:
                n_model_fits = len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits
        
        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            if refit_existing:
                fit_model = all_models[str(subj)][model_name][i]['model']
            else:
                si_agent = agents.RLStateInferenceAgent(**s)
                fit_model = agents.SummationModule([si_agent])
            
            loss = nn.CrossEntropyLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                # need to reshape the output to work with the cross entropy loss function
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                                           output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1), loss_diff_exit_thresh=end_tol)
            
                # evaluate the fit
                fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                                       output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
                
                print('\n{} Params:'.format(model_name))
                print(fit_model.print_params())
                
                print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
                print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
                print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
                
                if refit_existing:
                    all_models[str(subj)][model_name][i] = {'model': fit_model, 'perf': fit_perf}
                else:
                    all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                
                agents.save_model(all_models, save_path)
                
                i += 1
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))
                
                
    ## FULL BAYESIAN MODEL FITS

    p_step = 0.01
    # p_step_init = 0.05
    # p_step_final = 0.01
    
    # end_tol_init = 1e-4
    # end_tol_final = end_tol
    
    settings = {
                # 'No Switch Scatter, Switch Update First': {'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False}}}, 
                'No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates': 
                    {'update_p_switch_first': False,
                     'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False},
                                     'imperfect_update_alpha': {'init': 0, 'fit': False},
                                     'stay_bias_lam': {'init': 0, 'fit': False}}}, 
                # 'Fixed 0.5 Prior Rew Mean, No Switch Scatter': {'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False}, 
                #                                                          'init_high_rew_mean': {'init': 0, 'fit': False}, 
                #                                                          'init_low_rew_mean': {'init': 1, 'fit': False}}}
                }
    
    for label, s in settings.items():
        model_name = 'Bayes - ' + label

        if not model_name in all_models[str(subj)]:
            if refit_existing:
                continue
            else:
                all_models[str(subj)][model_name] = []
                n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            elif refit_existing:
                n_model_fits = len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits
        
        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            if refit_existing:
                fit_model = all_models[str(subj)][model_name][i]['model']
            else:
                agent = agents.BayesianAgent(p_step=p_step, **s)
                fit_model = agents.SummationModule([agent])

            loss = nn.CrossEntropyLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                # fit with a larger p_step at first
                #fit_model.agents[0].p_step = p_step_init
                # need to reshape the output to work with the cross entropy loss function
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                                           output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1), 
                                           loss_diff_exit_thresh=end_tol, print_params=True)
                
                # # then fit with a more refined p_step to finalize the model
                # fit_model.agents[0].p_step = p_step_final
                # # need to reshape the output to work with the cross entropy loss function
                # loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                #                            output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1), 
                #                            loss_diff_exit_thresh=end_tol_final, print_params=True)
            
                # evaluate the fit
                fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                                       output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
                
                print('\n{} Params:'.format(model_name))
                print(fit_model.print_params())
                
                print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
                print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
                print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))

                if refit_existing:
                    all_models[str(subj)][model_name][i] = {'model': fit_model, 'perf': fit_perf}
                else:
                    all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})

                agents.save_model(all_models, save_path)

                i += 1
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))  
            

# %% Investigate Fit Results

ignore_subj = ['182']
# plot accuracy and total LL
subjids = list(all_models.keys())
subjids = [s for s in subjids if not s in ignore_subj]

ignore_models = ['Q/Persev/Fall', 'SI/Persev', 'Q SI', 'RL SI'] # ['basic - value only']

model_names = list(all_models[subjids[0]].keys())
model_names.sort(key=str.lower)

model_names = [n for n in model_names if not any(im in n for im in ignore_models)]

# build dataframe with accuracy and LL per fit
fit_mets = []
for subj in subjids:
    for model_name in model_names:
        for i in range(len(all_models[subj][model_name])):
            model = all_models[subj][model_name][i]['model']
            perf = all_models[subj][model_name][i]['perf']
            
            fit_mets.append({'subjid': subj, 'model': '{} ({})'.format(model_name, th.count_params(model)), 'n_params': th.count_params(model), **perf})
            
fit_mets = pd.DataFrame(fit_mets)
fit_mets['n_trials'] = (fit_mets['ll_total']/fit_mets['ll_avg']).astype(int)
fit_mets['norm_llh'] = np.exp(fit_mets['ll_avg'])
fit_mets['bic'] = th.calc_bic(fit_mets['ll_total'], fit_mets['n_params'], fit_mets['n_trials'])
fit_mets['ll_total'] = -fit_mets['ll_total']
fit_mets['ll_avg'] = -fit_mets['ll_avg']
fit_mets['acc'] = fit_mets['acc']*100

model_names = fit_mets['model'].unique().tolist()
model_names.sort(key=str.lower)

best_model_counts = {m: {'norm_llh': 0, 'acc': 0, 'bic': 0} for m in model_names}

# calculate percent difference from best fitting model per subject
fit_mets[['diff_ll_avg', 'diff_norm_llh', 'diff_acc', 'diff_bic']] = 0.0
for subj in subjids:
    subj_sel = fit_mets['subjid'] == subj
    subj_mets = fit_mets[subj_sel]
    best_ll_avg = subj_mets['ll_avg'].min()
    best_norm_llh = subj_mets['norm_llh'].max()
    best_acc = subj_mets['acc'].max()
    best_bic = subj_mets['bic'].min()

    fit_mets.loc[subj_sel, 'diff_ll_avg'] = (subj_mets['ll_avg'] - best_ll_avg)/best_ll_avg*100
    fit_mets.loc[subj_sel, 'diff_norm_llh'] = -(subj_mets['norm_llh'] - best_norm_llh)/best_norm_llh*100
    fit_mets.loc[subj_sel, 'diff_acc'] = -(subj_mets['acc'] - best_acc)/best_acc*100
    fit_mets.loc[subj_sel, 'diff_bic'] = (subj_mets['bic'] - best_bic)/best_bic*100
    
    # count best models
    best_ll_names = subj_mets[subj_mets['norm_llh'] == best_norm_llh]['model'].unique()
    best_acc_names = subj_mets[subj_mets['acc'] == best_acc]['model'].unique()
    best_bic_names = subj_mets[subj_mets['bic'] == best_bic]['model'].unique()
    for name in best_ll_names:
        best_model_counts[name]['norm_llh'] += 1
        
    for name in best_acc_names:
        best_model_counts[name]['acc'] += 1
        
    for name in best_bic_names:
        best_model_counts[name]['bic'] += 1
    
best_model_counts = pd.DataFrame(best_model_counts).transpose().reset_index().rename(columns={'index': 'model'})

perf_cols = ['diff_ll_avg', 'diff_norm_llh', 'diff_acc', 'diff_bic']

avg_diffs = fit_mets.groupby(['subjid', 'model'])[perf_cols].min().reset_index()
avg_diffs = avg_diffs.groupby('model')[perf_cols].mean().reset_index()

ax_height = len(model_names)/5
# Plot model fit performances
fig, axs = plt.subplots(2, 1, figsize=(10,ax_height*2), layout='constrained')

sb.stripplot(fit_mets, y='model', x='norm_llh', hue='subjid', ax=axs[0], palette='colorblind')
sb.stripplot(fit_mets, y='model', x='acc', hue='subjid', ax=axs[1], palette='colorblind')

fig.suptitle('Model Performance Comparison - Values')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[0].set_xlabel('Avg p(correct) per trial')
axs[1].set_xlabel('Accuracy (%)')
axs[0].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
axs[1].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))


# plot performance differences from best fit
fig, axs = plt.subplots(3, 1, figsize=(10,ax_height*3), layout='constrained')

for ax in axs:
    plot_utils.plot_x0line(ax=ax)

sb.stripplot(fit_mets, y='model', x='diff_norm_llh', hue='subjid', ax=axs[0], palette='colorblind')
sb.stripplot(fit_mets, y='model', x='diff_acc', hue='subjid', ax=axs[1], palette='colorblind')
sb.stripplot(fit_mets, y='model', x='diff_bic', hue='subjid', ax=axs[2], palette='colorblind')

fig.suptitle('Model Performance Comparison - % Worse from Best Model')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[2].set_title('Model BIC')
axs[0].set_xlabel('% Worse Avg p(correct) per trial')
axs[1].set_xlabel('% Worse Accuracy')
axs[2].set_xlabel('% Worse BIC')
axs[0].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
axs[1].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
axs[2].legend(loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))


# plot best model counts
fig, axs = plt.subplots(3, 1, figsize=(10,ax_height*3), layout='constrained')

sb.barplot(best_model_counts, y='model', x='norm_llh', ax=axs[0], errorbar=None)
sb.barplot(best_model_counts, y='model', x='acc', ax=axs[1], errorbar=None)
sb.barplot(best_model_counts, y='model', x='bic', ax=axs[2], errorbar=None)

fig.suptitle('Best Model Performance Counts')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[2].set_title('Model BIC')
axs[0].set_xlabel('# Best Models')
axs[1].set_xlabel('# Best Models')
axs[2].set_xlabel('# Best Models')

# plot average best model differences
fig, axs = plt.subplots(3, 1, figsize=(10,ax_height*3), layout='constrained')

sb.barplot(avg_diffs, y='model', x='diff_norm_llh', ax=axs[0], errorbar=None)
sb.barplot(avg_diffs, y='model', x='diff_acc', ax=axs[1], errorbar=None)
sb.barplot(avg_diffs, y='model', x='diff_bic', ax=axs[2], errorbar=None)

fig.suptitle('Average Model Difference from Best Model per Subject')
axs[0].set_title('Model Normalized Likelihood')
axs[1].set_title('Model Accuracy')
axs[2].set_title('Model BIC')
axs[0].set_xlabel('% Worse Avg p(correct) per trial')
axs[1].set_xlabel('% Worse Accuracy')
axs[2].set_xlabel('% Worse BIC')


# %% Get FP data and analyze peaks

fp_sess_ids = db_access.get_fp_data_sess_ids(protocol='ClassicRLTasks', stage_num=2, subj_ids=subj_ids)
implant_info = db_access.get_fp_implant_info(subj_ids)

filename = 'two_arm_bandit_data'

save_path = path.join(utils.get_user_home(), 'db_data', filename+'.pkl')

if path.exists(save_path):
    with open(save_path, 'rb') as f:
        saved_data = pickle.load(f)
        aligned_signals = saved_data['aligned_signals']
        aligned_metadata = saved_data['metadata']

alignments = [Align.cue, Align.reward] #  
signal_type = 'dff_iso' # , 'z_dff_iso'

filter_props = {Align.cue: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}},
                Align.reward: {'DMS': {'filter': True, 'use_filt_signal_props': False, 'cutoff_f': 8},
                            'PL': {'filter': True, 'use_filt_signal_props': True, 'cutoff_f': 1}}}

peak_find_props = {Align.cue: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                               'PL': {'min_dist': 0.2, 'peak_tmax': 1.5, 'peak_edge_buffer': 0.2, 'lim_peak_width_to_edges': True}},
                   Align.reward: {'DMS': {'min_dist': 0.05, 'peak_tmax': 0.45, 'peak_edge_buffer': 0.08, 'lim_peak_width_to_edges': True},
                                  'PL': {'min_dist': 0.5, 'peak_tmax': 3.5, 'peak_edge_buffer': 0.2, 'lim_peak_width_to_edges': False}}}

sides = ['contra', 'ipsi']
regions = ['DMS', 'PL']

ignored_signals = {'PL': [],
                   'DMS': []}

t = aligned_signals['t']

peak_metrics = []

for subj_id in subj_ids:
    print('Analyzing peaks for subj {}'.format(subj_id))
    for sess_id in fp_sess_ids[subj_id]:
        
        trial_data = all_sess[all_sess['sessid'] == sess_id]
        rewarded = trial_data['rewarded'].to_numpy()
        responded = ~np.isnan(trial_data['response_time']).to_numpy()
        choice = trial_data['choice']
        reward_time = trial_data['reward_time'].to_numpy()[:,None]
        stays = choice[:-1].to_numpy() == choice[1:].to_numpy()
        switches = np.insert(~stays, 0, False)
        stays = np.insert(stays, 0, False)
        prev_rewarded = np.insert(rewarded[:-1], 0, False)
        prev_unrewarded = np.insert(~rewarded[:-1], 0, False)
        
        resp_rewarded = rewarded[responded]
        
        for region in regions:
            if sess_id in ignored_signals[region]:
                continue

            region_side = implant_info[subj_id][region]['side']
            choice_side = choice.apply(lambda x: fpah.get_implant_side_type(x, region_side) if not x == 'none' else 'none').to_numpy()

            for align in alignments:
                if not align in aligned_signals[subj_id][sess_id][signal_type]:
                    continue

                t_r = t[align][region]
                mat = aligned_signals[subj_id][sess_id][signal_type][align][region]

                # calculate peak properties on a trial-by-trial basis
                contra_choices = choice_side == 'contra'
                contra_choices = contra_choices[responded]
                
                resp_trial = 0
                for i in range(mat.shape[0]):
                    if responded[i]:
                        metrics = fpah.calc_peak_properties(mat[i,:], t_r, 
                                                            filter_params=filter_props[align][region],
                                                            peak_find_params=peak_find_props[align][region],
                                                            fit_decay=False)

                        peak_metrics.append(dict([('subj_id', subj_id), ('sess_id', sess_id), ('signal_type', signal_type), 
                                                 ('align', align.name), ('region', region), ('trial', resp_trial),
                                                 ('rewarded', rewarded[i]), ('side', choice_side[i]),
                                                 ('reward_time', reward_time[i]), ('RT', trial_data['RT'].iloc[i]),
                                                 ('cpoke_out_latency', trial_data['cpoke_out_latency'].iloc[i]), *metrics.items()]))
                        
                        resp_trial += 1
                            

peak_metrics = pd.DataFrame(peak_metrics)
# drop unused columns
peak_metrics.drop(['decay_tau', 'decay_params', 'decay_form'], axis=1, inplace=True)

# filter peak metric outliers
# make subject ids categories

ignore_outliers = True
ignore_any_outliers = True
outlier_thresh = 10

t_min = 0.02
t_max = {a: {r: peak_find_props[a][r]['peak_tmax'] - t_min for r in regions} for a in alignments} 

parameters = ['peak_time', 'peak_height'] #, 'decay_tau'

filt_peak_metrics = peak_metrics.copy()

# remove outliers on a per-subject basis:
if ignore_outliers:
    
    # first get rid of peaks with times too close to the edges of the peak window (10ms from each edge)
    peak_sel = np.full(len(peak_metrics), False)
    for align in alignments:    
        for region in regions:
            align_region_sel = (peak_metrics['align'] == align) & (peak_metrics['region'] == region)
            sub_peak_metrics = peak_metrics[align_region_sel]
            peak_sel[align_region_sel] = ((sub_peak_metrics['peak_height'] > 0) & 
                                          (sub_peak_metrics['peak_time'] > t_min) &
                                          (sub_peak_metrics['peak_time'] < t_max[align][region]))
            
    # look at potentially problematic peaks
    # t = aligned_signals['t']
    # rem_peak_info = peak_metrics[~peak_sel]
    # rem_peak_info =  rem_peak_info[rem_peak_info['signal_type'] == 'dff_iso']
    # rem_subj_ids = np.unique(rem_peak_info['subj_id'])
    # for subj_id in rem_subj_ids:
    #     subj_peak_info = rem_peak_info[rem_peak_info['subj_id'] == subj_id]
    #     for _, row in subj_peak_info.iterrows():
    #         mat = aligned_signals[row['subj_id']][row['sess_id']]['dff_iso'][row['align']][row['region']]
    #         _, ax = plt.subplots(1,1)
    #         ax.set_title('{} - {}, {} {}-aligned, trial {}'.format(row['subj_id'], row['sess_id'], row['region'], row['align'], row['trial']))
    #         ax.plot(t[row['align']][row['region']], mat[row['trial'], :])
    #         plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
    #         peak_idx = np.argmin(np.abs(t[row['align']][row['region']] - row['peak_time']))
    #         ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
    #         ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')

    filt_peak_metrics = filt_peak_metrics[peak_sel]
    
    # first add iqr multiple columns
    for param in parameters:
        filt_peak_metrics['iqr_mult_'+param] = np.nan
    
    # calculate iqr multiple for potential outliers
    outlier_grouping = ['subj_id', 'sess_id']
    
    # compute IQR on different groups of trials based on the alignment and region
    for align in alignments:
        # separate peaks by outcome at time of reward
        if align == Align.reward:
            align_outlier_grouping = outlier_grouping+['rewarded']
        else:
            align_outlier_grouping = outlier_grouping
            
        for region in regions:
            # separate peaks by side for DMS since very sensitive to choice side
            if region == 'DMS':
                region_outlier_grouping = align_outlier_grouping+['side']
            else:
                region_outlier_grouping = align_outlier_grouping
                
            align_region_sel = (filt_peak_metrics['align'] == align) & (filt_peak_metrics['region'] == region)
            
            filt_peak_metrics.loc[align_region_sel, :] = fpah.calc_iqr_multiple(filt_peak_metrics[align_region_sel], region_outlier_grouping, parameters)
    
    # then remove outlier values
    if ignore_any_outliers:

        outlier_sel = np.full(len(filt_peak_metrics), False)
        for param in parameters:
            outlier_sel = outlier_sel | (np.abs(filt_peak_metrics['iqr_mult_'+param]) >= outlier_thresh)
            
        filt_peak_metrics.loc[outlier_sel, parameters] = np.nan
        
    else:
        for param in parameters:
            outlier_sel = np.abs(filt_peak_metrics['iqr_mult_'+param]) >= outlier_thresh
            
            if any(outlier_sel):
                # look at outlier peaks
                # t = aligned_signals['t']
                # rem_peak_info = filt_peak_metrics[outlier_sel]
                # rem_peak_info =  rem_peak_info[rem_peak_info['signal_type'] == 'dff_iso']
                # rem_subj_ids = np.unique(rem_peak_info['subj_id'])
                # for subj_id in rem_subj_ids:
                #     subj_peak_info = rem_peak_info[rem_peak_info['subj_id'] == subj_id]
                #     for _, row in subj_peak_info.iterrows():
                #         mat = aligned_signals[row['subj_id']][row['sess_id']]['dff_iso'][row['align']][row['region']]
                #         _, ax = plt.subplots(1,1)
                #         ax.set_title('{} - {}, {} {}-aligned, trial {}'.format(row['subj_id'], row['sess_id'], row['region'], row['align'], row['trial']))
                #         ax.plot(t[row['align']][row['region']], mat[row['trial'], :])
                #         plot_utils.plot_dashlines([t_min, t_max[row['align']][row['region']]], ax=ax)
                #         peak_idx = np.argmin(np.abs(t[row['align']][row['region']] - row['peak_time']))
                #         ax.plot(row['peak_time'], mat[row['trial'], peak_idx], marker=7, markersize=10, color='C1')
                #         ax.vlines(row['peak_time'], mat[row['trial'], peak_idx]-row['peak_height'], mat[row['trial'], peak_idx], color='C2', linestyles='dashed')
            
                filt_peak_metrics.loc[outlier_sel, param] = np.nan

# %% Compare model fit outputs

subjids = [179, 188, 191, 207] # [179, 188, 191, 207]

use_fp_sess_only = True
plot_ind_sess = True
plot_output = True
plot_output_diffs = False
plot_agent_states = True
plot_rpes = False
plot_fp_peak_amps = False

#['Q - Same Alpha Only, K Fixed', 'Q - All Alpha Free, All K Fixed']
#['Q - All Alpha Shared, All K Fixed', 'Q - All Alpha Shared, All K Free']
#['Q - All Alpha Shared, All K Fixed', 'Q - Same Alpha Only Shared, K Fixed']
#['Q - All Alpha Shared, All K Fixed', 'Q - All Alpha Shared, Counter D/R K=-1']
#['Q - All Alpha Shared, All K Free', 'SI - Free Same/Diff Rew Evidence']
#['Q - All Alpha Shared, All K Fixed', 'SI - Separate Rew/Unrew Evidence']
# compare_model_info = {'Q SI - Alpha Free, K Free': {'agent_names': ['State', 'Value', 'Belief']}, 
#                       'Q SI - Separate High Alphas, Const Low K, High K Free': {'agent_names': ['State', 'Value', 'Belief']}}
compare_model_info = {'Q SI - Alpha Free, K Free': {'agent_names': ['State', 'Value', 'Belief']}, 
                      'Q SI - Alpha Free, K Free, Belief Update First': {'agent_names': ['State', 'Value', 'Belief']}}
                      
compare_models = list(compare_model_info.keys())
model_outputs = {s: {m: {} for m in compare_models} for s in subj_ids}

plot_sess_ids = fp_sess_ids if use_fp_sess_only else sess_ids
    
for subj in subjids: 

    # Format model inputs to re-run fit models
    sess_data = all_sess[all_sess['sessid'].isin(plot_sess_ids[subj])]
    # filter out no responses
    sess_data = sess_data[sess_data['hit']==True]
    
    ## Create 3-D inputs tensor and 3-D labels tensor
    n_sess = len(plot_sess_ids[subj])
    max_trials = np.max(sess_data.groupby('sessid').size())

    inputs = torch.zeros(n_sess, max_trials-1, 3)
    left_choice_labels = torch.zeros(n_sess, max_trials-1, 1)
    trial_mask = torch.zeros(n_sess, max_trials-1, 1)

    # populate tensors from behavioral data
    for i, sess_id in enumerate(plot_sess_ids[subj]):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data) - 1 # one less because we predict the next choice based on the prior choice
        
        left_choice_labels[i, :n_trials, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)
        inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['chose_left'][:-1], trial_data['chose_right'][:-1], trial_data['rewarded_int'][:-1]]).T).type(torch.float)
        trial_mask[i, :n_trials, :] = 1
        
    for model_name in compare_models:
        
        # get best model
        best_model_idx = 0
        for i in range(len(all_models[str(subj)][model_name])):
            if all_models[str(subj)][model_name][i]['perf']['norm_llh'] > all_models[str(subj)][model_name][best_model_idx]['perf']['norm_llh']:
                best_model_idx = i    

        model = all_models[str(subj)][model_name][best_model_idx]['model'].clone()

        # run model
        output, agent_states, fit_perf = th.eval_model(model, inputs, left_choice_labels, trial_mask=trial_mask,
                                                       output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
        
        state_diff_hist = torch.stack([torch.stack(agent.state_diff_hist, dim=1) for agent in model.agents], dim=-1).numpy()
        #state_delta_hist = torch.stack([torch.stack(agent.state_delta_hist, dim=1) for agent in model.agents], dim=-1).numpy()
        
        if isinstance(model.agents[0], agents.QValueStateInferenceAgent):
            value_hist = torch.stack(model.agents[0].v_hist[1:], dim=1).numpy()
            belief_hist = torch.stack(model.agents[0].belief_hist[1:], dim=1).numpy()
            agent_states = np.insert(agent_states, 1, value_hist, axis=3)
            agent_states = np.insert(agent_states, 2, belief_hist, axis=3)
        
        model_outputs[subj][model_name] = {'model': model, 'output': output, 'agent_states': agent_states, 'perf': fit_perf,
                                           'agent_state_diff_hist': state_diff_hist} #, 'agent_state_delta_hist': state_diff_hist}


if plot_ind_sess:
    for subj in subjids: 
    
        # Format model inputs to re-run fit models
        sess_data = all_sess[all_sess['sessid'].isin(plot_sess_ids[subj])]
        # filter out no responses
        sess_data = sess_data[sess_data['hit']==True]
                
        # compare output similarities between the models
        for i, sess_id in enumerate(plot_sess_ids[subj]):
            trial_data = sess_data[sess_data['sessid'] == sess_id]
            
            # output_diffs = {}
            # for model_name in compare_models:
            #     output_diffs[model_name] = {}
            #     outputs = np.stack([mo['output'][i,:,0] for mo in model_outputs[subj][model_name]], axis=1)
            #     output_diffs[model_name]['outputs'] = outputs
            #     output_diffs[model_name]['diff'] = np.mean(np.abs(np.diff(outputs, axis=1)), axis=1)
            #     output_diffs[model_name]['avg'] = np.mean(outputs, axis=1)
            #     # assuming the value agent is the first agent
            #     agent_states = np.stack([mo['agent_states'][i,:,:,0] for mo in model_outputs[subj][model_name]], axis=1)
            #     output_diffs[model_name]['agent_states'] = agent_states
            #     output_diffs[model_name]['avg_agent_states'] = np.mean(agent_states, axis=1)
            #     output_diffs[model_name]['state_diffs'] = np.stack([mo['agent_state_diff_hist'][i,:,:,0] for mo in model_outputs[subj][model_name]], axis=1)
            #     output_diffs[model_name]['avg_state_diffs'] = np.mean(output_diffs[model_name]['state_diffs'], axis=1)
            #     output_diffs[model_name]['trans_state_diffs'] = utils.rescale(output_diffs[model_name]['state_diffs'], 0, 1, axis=1)
            #     output_diffs[model_name]['trans_avg_state_diffs'] = np.mean(output_diffs[model_name]['trans_state_diffs'], axis=1)
            
            output_diffs = np.abs(np.diff(np.stack([model_outputs[subj][m]['output'][i,:,0] for m in compare_models], axis=1), axis=1))
            n_agents = np.max([model_outputs[subj][m]['agent_states'].shape[3] for m in compare_models])
    
            n_rows = 0
            if plot_output:
                n_rows += 1
            if plot_output_diffs:
                n_rows += 1
            if plot_agent_states:
                n_rows += n_agents
            if plot_rpes:
                n_rows += 1
            if plot_fp_peak_amps:
                n_rows += len(regions)
                
            fig, axs = plt.subplots(n_rows, 1, figsize=(15,n_rows*4), layout='constrained')
                
            # get block transitions
            block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
            block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
            block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
                
            fig.suptitle('Subj {} Session {} Model Comparison'.format(subj, sess_id))
    
            # label trials from 1 to the last trial
            x = np.arange(len(trial_data)-1)+1
            
            ax_idx = 0
            
            if plot_output:
                ax = axs[ax_idx]
                # plot model outputs
                for j, model_name in enumerate(compare_models):
                    ax.plot(x, model_outputs[subj][model_name]['output'][i,:len(trial_data)-1,0], color='C{}'.format(j), alpha=0.6, label=model_name)
                    
                ax.set_ylabel('p(Choose Left)')
                ax.set_xlabel('Trial')
                ax.set_title('Model Outputs', fontsize=10)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                ax.axhline(y=0.5, color='black', linestyle='dashed')
                ax.margins(x=0.01)
                
                th._draw_choices(trial_data, ax)
                th._draw_blocks(block_switch_trials, block_rates, ax)
                
                ax_idx += 1
            
            # Plot output diffs between models
            if plot_output_diffs:
                ax = axs[ax_idx]
                
                ax.plot(x, output_diffs[:len(trial_data)-1])
        
                ax.set_ylabel('Output Diffs')
                ax.set_xlabel('Trial')
                ax.set_title('Avg Model Output Differences', fontsize=10)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.axhline(y=0, color='black', linestyle='dashed')
                ax.margins(x=0.01)
                
                th._draw_choices(trial_data, ax)
                th._draw_blocks(block_switch_trials, block_rates, ax)
                
                ax_idx += 1
                
            if plot_agent_states:
                for j in range(n_agents):
                    ax = axs[ax_idx]
                    
                    for k, model_name in enumerate(compare_models):
                        agent_names = compare_model_info[model_name]['agent_names']
                        if j < len(agent_names):
                            ax.plot(x-1, model_outputs[subj][model_name]['agent_states'][i,:len(trial_data)-1,0,j], color='C{}'.format(k), alpha=0.6, label='{}, {} left'.format(model_name, agent_names[j]))
                            ax.plot(x-1, model_outputs[subj][model_name]['agent_states'][i,:len(trial_data)-1,1,j], color='C{}'.format(k), alpha=0.6, linestyle='dotted', label='{}, {} right'.format(model_name, agent_names[j]))
                        
                    ax.set_ylabel('Agent State Values')
                    ax.set_xlabel('Trial')
                    ax.set_title('Agent State Values', fontsize=10)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                    ax.axhline(y=0, color='black', linestyle='dashed')
                    ax.margins(x=0.01)
                    
                    th._draw_choices(trial_data, ax)
                    th._draw_blocks(block_switch_trials, block_rates, ax)
                
                    ax_idx += 1
    
            if plot_rpes:
                ax = axs[ax_idx]
                # plot all model RPEs for each side
                for j, model_name in enumerate(compare_models):
                    ax.plot(x-1, model_outputs[subj][model_name]['agent_state_diff_hist'][i,:len(trial_data)-1,0,0], color='C{}'.format(j), alpha=0.6, label='{}, left'.format(model_name))
                    ax.plot(x-1, model_outputs[subj][model_name]['agent_state_diff_hist'][i,:len(trial_data)-1,1,0], color='C{}'.format(j), alpha=0.6, linestyle='dotted', label='{}, right'.format(model_name))
                    
                ax.set_ylabel('Model RPEs')
                ax.set_xlabel('Trial')
                ax.set_title('All Model RPEs', fontsize=10)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                ax.axhline(y=0, color='black', linestyle='dashed')
                ax.margins(x=0.01)
            
                th._draw_choices(trial_data, ax)
                th._draw_blocks(block_switch_trials, block_rates, ax)
                
                ax_idx += 1
    
            if plot_fp_peak_amps:
                for j, region in enumerate(regions):
                    ax = axs[ax_idx+j]
                        
                    sess_region_metrics = filt_peak_metrics[(filt_peak_metrics['sess_id'] == sess_id) & (filt_peak_metrics['region'] == region) & (filt_peak_metrics['align'] == Align.reward)].sort_values('trial')
                    peak_amps = sess_region_metrics['peak_height'].to_numpy()
                    peak_trials = sess_region_metrics['trial'].to_numpy()
                    
                    for k, side in enumerate(sides):
                        for rewarded in [True, False]:
                            side_outcome_sel = (sess_region_metrics['side'] == side) & (sess_region_metrics['rewarded'] == rewarded)
                            color = 'C{}'.format(k+3)
                            # change color lightness based on outcome
                            color = utils.change_color_lightness(color, -0.30) if rewarded else utils.change_color_lightness(color, 0.30)
                            rew_label = 'rew' if rewarded else 'unrew'
                            ax.vlines(x=peak_trials[side_outcome_sel], ymin=0, ymax=peak_amps[side_outcome_sel], color=color, label='{} choice, {}'.format(side, rew_label))
                        
                    _, y_label = fpah.get_signal_type_labels(signal_type)
                    ax.set_ylabel(y_label)
                    ax.set_xlabel('Trial')
                    ax.set_title('{} - Reward Peak Amplitude'.format(region), fontsize=10)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                    ax.axhline(y=0, color='black', linestyle='dashed')
                    ax.margins(x=0.01)
                    
                    th._draw_choices(trial_data, ax)
                    th._draw_blocks(block_switch_trials, block_rates, ax)
                
# plot comparison of fit performance for each model

# first build dataframe
model_fit_comparison = []
for subj in subjids:
    for model_name in compare_models:
        
        model_fit_comparison.append({'subj': subj, 'model': model_name, **model_outputs[subj][model_name]['perf']})
        
model_fit_comparison = pd.DataFrame(model_fit_comparison)

# then plot
metrics = ['norm_llh'] #, 'bic', 'acc'
metric_labels = ['Normalized Likelihood', 'BIC', 'Accuracy']
n_cols =  len(metrics)
fig, axs = plt.subplots(1, n_cols, figsize=(3*n_cols,3), layout='constrained')
axs = np.resize(np.array(axs), n_cols)
fig.suptitle('Fit Performance Comparison: {} vs {}'.format(compare_models[0], compare_models[1]))

for i, metric in enumerate(metrics):
    pivot_metrics = model_fit_comparison.pivot(index='subj', columns='model', values=metric).reset_index()
    ax = axs[i]
    sb.scatterplot(pivot_metrics, x=compare_models[0], y=compare_models[1], ax=ax)
    plot_utils.plot_unity_line(ax)
    ax.set_title(metric_labels[i])
    ax.set_xlabel(compare_models[0])
    ax.set_ylabel(compare_models[1])
    
# %% Plot model fits

use_simple_plot = False

model_name = 'Bayes - No Switch Scatter, Simul Updates' #'Q/Fall - All Alpha Shared, All K Fixed'
agent_names = ['State'] #['Value', 'Perseverative']

subjids = [179, 188, 191] # [179, 188, 191, 207]

use_fp_sess_only = True

plot_sess_ids = fp_sess_ids if use_fp_sess_only else sess_ids
    
for subj in subjids: 

    # Format model inputs to re-run fit models
    sess_data = all_sess[all_sess['sessid'].isin(plot_sess_ids[subj])]
    # filter out no responses
    sess_data = sess_data[sess_data['hit']==True]
    
    ## Create 3-D inputs tensor and 3-D labels tensor
    n_sess = len(plot_sess_ids[subj])
    max_trials = np.max(sess_data.groupby('sessid').size())

    inputs = torch.zeros(n_sess, max_trials-1, 3)
    left_choice_labels = torch.zeros(n_sess, max_trials-1, 1)
    trial_mask = torch.zeros(n_sess, max_trials-1, 1)

    # populate tensors from behavioral data
    for i, sess_id in enumerate(plot_sess_ids[subj]):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data) - 1 # one less because we predict the next choice based on the prior choice
        
        left_choice_labels[i, :n_trials, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)
        inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['chose_left'][:-1], trial_data['chose_right'][:-1], trial_data['rewarded_int'][:-1]]).T).type(torch.float)
        trial_mask[i, :n_trials, :] = 1
        
    best_model_idx = 0
    for i in range(len(all_models[str(subj)][model_name])):
        if all_models[str(subj)][model_name][i]['perf']['norm_llh'] > all_models[str(subj)][model_name][best_model_idx]['perf']['norm_llh']:
            best_model_idx = i    

    model = all_models[str(subj)][model_name][best_model_idx]['model'].clone()
    
    output, agent_states = th.run_model(model, inputs, output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
    betas = model.beta.weight[0].detach().numpy()
    
    # add to the agent states for hybrid value/state inference agents
    if isinstance(model.agents[0], agents.QValueStateInferenceAgent):
        value_hist = torch.stack(model.agents[0].v_hist[1:], dim=1).numpy()
        belief_hist = torch.stack(model.agents[0].belief_hist[1:], dim=1).numpy()
        agent_states = np.insert(agent_states, 1, value_hist, axis=3)
        agent_states = np.insert(agent_states, 2, belief_hist, axis=3)
        betas = np.insert(betas, 1, np.array([1,1]))

    if use_simple_plot:
        th.plot_simple_multi_val_fit_results(sess_data, output, agent_states, 
                                             agent_names, betas=betas, 
                                             title_prefix='Subj {}, {} - '.format(subj, model_name))
    else:
        th.plot_multi_val_fit_results(sess_data, output, agent_states, agent_names, 
                                      betas=betas, title_prefix='Subj {}, {} - '.format(subj, model_name))