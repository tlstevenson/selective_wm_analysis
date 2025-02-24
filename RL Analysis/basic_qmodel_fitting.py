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
from sys_neuro_tools import plot_utils
from models import agents
import models.training_helpers as th
import models.sim_helpers as sh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import matplotlib.pyplot as plt
import seaborn as sb
from os import path

# passed function for basic model
def generate_agents(alpha0 = None):
    return [agents.SingleValueAgent(alpha0), agents.PerseverativeAgent(alpha0), agents.FallacyAgent(alpha0)]
 
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

# %% Run Fitting

plot_fits = False
skip_existing_fits = True

for subj in subj_ids: 
    print("\nSubj", subj)
    
    if not str(subj) in all_models:
        all_models[str(subj)] = {}
        
    sess_data = all_sess[all_sess['subjid'] == subj]
    # filter out no responses
    sess_data = sess_data[sess_data['hit']==True]
    
    ## Create 3-D inputs tensor and 3-D labels tensor
    n_sess = len(sess_ids[subj])

    ## add a column to represent choice (+1 left and -1 right) and outcome (+1 rewarded and -1 unrewarded)
    choice_input_dict = {'left': 1, 'right': -1}
    outcome_dict = {False: -1, True: 1}
    
    # Add choice and outcome inputs as new columns to sess data
    sess_data['choice_inputs'] = sess_data['choice'].apply(lambda choice: choice_input_dict[choice] if choice in choice_input_dict.keys() else np.nan).to_numpy()
    sess_data['outcome_inputs'] = sess_data['rewarded'].apply(lambda reward: outcome_dict[reward] if reward in outcome_dict.keys() else np.nan).to_numpy()
    sess_data['chose_left'] = sess_data['chose_left'].astype(int)
    sess_data['chose_right'] = sess_data['chose_right'].astype(int)
    
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
  
    n_fits = 2
    n_steps = 4000
    
    basic_model_agents = [[agents.SingleValueAgent()], [agents.SingleValueAgent(), agents.PerseverativeAgent()], [agents.SingleValueAgent(), agents.FallacyAgent()],
                          [agents.SingleValueAgent(), agents.PerseverativeAgent(), agents.FallacyAgent()]]
    basic_model_names = ['Basic - Value', 'Basic - Value/Persev', 'Basic - Value/Fall', 'Basic - Value/Persev/Fall']
    
    for model_agents, model_name in zip(basic_model_agents, basic_model_names):
        if not model_name in all_models[str(subj)]:
            all_models[str(subj)][model_name] = []
            n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits

        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            # fit the model
            ##### EDIT ####### passed initial alpha=0.1
            #fit_model = agents.SummationModule(generate_agents())
            fit_model = agents.SummationModule(model_agents)
            fit_model.reset_params()
            
            loss = nn.BCEWithLogitsLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, left_choice_labels, n_steps, trial_mask=trial_mask)
                
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
                
                all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                
                agents.save_model(all_models, save_path)
                
                i += 1
                
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))
    
    ## Q-MODEL FITS

    ##### EDIT ####### Q-model
    #created new column based on whether a trial was rewarded or not
    sess_data['rewarded_int'] = sess_data['rewarded'].astype(int)
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
                
                'All Alpha Free, All K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
            
                # 'All Alpha Free, All K Fixed, Diff K=0.5': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                             'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
                # All Alphas shared, Different K Fixes
                'All Alpha Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                                                  'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                                                  'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'All Alpha Shared, Same K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                    'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                'All Alpha Shared, All K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_rew'}},
                
                # 'All Alpha Shared, All K Fixed, Diff K=0.5': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                #                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                #                                               'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
                
                # Models with limited different choice updating
                'Same Alpha Only Shared, K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                                               'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                'Same Alpha Only Shared, Counter K': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                                               'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Shared, K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 
                #                               'alpha_diff_unrew': {'fit': False, 'init': 0}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                'Same Alpha Only, K Fixed': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                # 'Same Alpha Only, K Free': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                #                             'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                'Same Alpha Only, Counter K': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
                                               'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 
                                               'k_diff_unrew': {'fit': False}},
                
                'No Alpha D/U, All K Fixed': {'alpha_diff_unrew': {'fit': False, 'init': 0}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                                               'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                'No Alpha D/R, All K Fixed': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
                                               'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                
                
                # Constrained Alpha Pairs
                # 'Alpha Same/Diff Shared, Same K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                'Alpha Same/Diff Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                                                         'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                                                         'k_diff_unrew': {'fit': False}},
                
                # 'Alpha Rew/Unrew Shared, Same K Fixed': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
                #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
                
                'Alpha Rew/Unrew Shared, All K Fixed': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
                                                         'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                                                         'k_diff_unrew': {'fit': False}},
                
                
                # Counterfactual models
                'All Alpha Shared, Counter D/U K=1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                                                      'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 
                                                      'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False, 'init': 1}},
                
                'Alpha Same/Diff Shared, Counter D/U K=1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                                                            'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                                                            'k_diff_unrew': {'fit': False, 'init': 1}},
                
                'All Alpha Shared, Counter S/U K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                                                      'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                                                      'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
                
                'Alpha Same/Diff Shared, Counter S/U K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
                                                            'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 
                                                            'k_diff_unrew': {'fit': False}},
                
                
                }

    for label, s in settings.items():
        model_name = 'Q - ' + label
        
        if not model_name in all_models[str(subj)]:
            all_models[str(subj)][model_name] = []
            n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits
        
        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            #fit the model formulated in the same way as the single value model
            q_agent = agents.QValueAgent(alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
                                          k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, 
                                          constraints=s)
            
            fit_model = agents.SummationModule([q_agent]) #, agents.PerseverativeAgent(n_vals=2), agents.FallacyAgent(n_vals=2)
            
            loss = nn.CrossEntropyLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                # need to reshape the output to work with the cross entropy loss function
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                                           output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1))
                
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
                                                  ['Value', 'Persev', 'Fallacy'], 
                                                  ['Left', 'Right'], betas=fit_model.beta.weight[0], 
                                                  title_prefix='Subi {}: Fit Model {} - '.format(subj, '{} {}'.format(model_name, i)))
                    plt.show(block=False)
                
                all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                
                agents.save_model(all_models, save_path)
                i += 1
                
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))
            
    ## STATE INFERENCE MODEL FITS

    settings = {'Shared Evidence': {'complement_c_rew':True, 'complement_c_diff':True}, 
                'Separate Same/Diff Evidence': {'complement_c_rew':True, 'complement_c_diff':False},
                'Separate Rew/Unrew Evidence': {'complement_c_rew':False, 'complement_c_diff':True}, 
                'All Separate Evidence': {'complement_c_rew':False, 'complement_c_diff':False}}
    
    for label, s in settings.items():
        model_name = 'SI - ' + label
        
        if not model_name in all_models[str(subj)]:
            all_models[str(subj)][model_name] = []
            n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[str(subj)][model_name])
            else:
                n_model_fits = n_fits
        
        i = 0
        while i < n_model_fits:
            print('\n{}, fit {}\n'.format(model_name, i))

            #fit the model formulated in the same way as the single value model
            si_agent = agents.StateInferenceAgent(complement_c_rew=s['complement_c_rew'], complement_c_diff=s['complement_c_diff'])
            
            fit_model = agents.SummationModule([si_agent]) #, agents.PerseverativeAgent(n_vals=2)
            
            loss = nn.CrossEntropyLoss(reduction='none')
            
            optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
            
            try:
                # need to reshape the output to work with the cross entropy loss function
                loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, n_steps, trial_mask=trial_mask,
                                           output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1))
            
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
                                                  ['Value', 'Persev', 'Fallacy'], 
                                                  ['Left', 'Right'], betas=fit_model.beta.weight[0], 
                                                  title_prefix='Subj {}: Fit Model {} - '.format(subj, '{} {}'.format(model_name, i)))
                    plt.show(block=False)
                
                all_models[str(subj)][model_name].append({'model': fit_model, 'perf': fit_perf})
                
                agents.save_model(all_models, save_path)
                
                i += 1
            except RuntimeError as e:
                print('Error: {}. \nTrying Again...'.format(e))
            


# %% Investigate Fit Results

# plot accuracy and total LL
subjids = list(all_models.keys())
if '182' in subjids:
    subjids.remove('182')

ignore_models = [] # ['basic - value only']

model_names = list(all_models[subjids[0]].keys())
model_names.sort(key=str.lower)

for name in ignore_models:
    model_names.remove(name)

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

model_names = fit_mets['model'].tolist()
model_names.sort(key=str.lower)

best_model_counts = {m: {'norm_llh': 0, 'acc': 0, 'bic': 0} for m in model_names}

# calculate percent difference from best fitting model per subject
fit_mets[['diff_ll_avg', 'diff_norm_llh', 'diff_acc', 'diff_bic']] = 0.0
for subj in subjids:
    subj_sel = fit_mets['subjid'] == subj
    subj_mets = fit_mets[subj_sel]
    best_ll_avg = subj_mets['ll_avg'].max()
    best_norm_llh = subj_mets['norm_llh'].max()
    best_acc = subj_mets['acc'].max()
    best_bic = subj_mets['bic'].min()

    fit_mets.loc[subj_sel, 'diff_ll_avg'] = -(subj_mets['ll_avg'] - best_ll_avg)/best_ll_avg*100
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

# Plot model fit performances
fig, axs = plt.subplots(2, 1, figsize=(10,10), layout='constrained')

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
fig, axs = plt.subplots(3, 1, figsize=(10,15), layout='constrained')

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
fig, axs = plt.subplots(3, 1, figsize=(10,15), layout='constrained')

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

# %%

# save_models = {}

# for subj in all_models.keys():
#     save_models[subj] = {}
#     for model_name in all_models[subj].keys():
#         save_models[subj][model_name] = []
        
#         for i in range(len(all_models[subj][model_name])):
#             old_model = all_models[subj][model_name][i]['model']
            
#             if model_name == 'basic':
#                 new_model = agents.SummationModule([agents.SingleValueAgent(old_model.agents[0].alpha_v.a.item()), 
#                                                     agents.PerseverativeAgent(old_model.agents[1].alpha_h.a.item()), 
#                                                     agents.FallacyAgent(old_model.agents[2].alpha_g.a.item())],
#                                                    betas=old_model.beta.weight.tolist(), bias=old_model.bias.item())
                
#                 assert torch.allclose(old_model.agents[0].alpha_v.a, new_model.agents[0].alpha.a), 'Value alphas are different'

#             else:
#                 old_q = old_model.agents[0]
#                 new_q = agents.QValueAgent(alpha_same_rew=old_q.alpha_same_rew.a.item(), alpha_same_unrew=old_q.alpha_same_unrew.a.item(), 
#                                            alpha_diff_rew=old_q.alpha_diff_rew.a.item(), alpha_diff_unrew=old_q.alpha_diff_unrew.a.item(), 
#                                            k_same_rew=old_q.k_same_rew.item(), k_same_unrew=old_q.k_same_unrew.item(), 
#                                            k_diff_rew=old_q.k_diff_rew.item(), k_diff_unrew=old_q.k_diff_unrew.item(), 
#                                            constraints=old_q.constraints)
                
#                 new_model = agents.SummationModule([new_q, 
#                                                     agents.PerseverativeAgent(old_model.agents[1].alpha_h.a.item(), n_vals=2), 
#                                                     agents.FallacyAgent(old_model.agents[2].alpha_g.a.item(), n_vals=2)],
#                                                    betas=old_model.beta.weight.tolist(), bias=old_model.bias.item())
                
#                 assert torch.allclose(old_q.alpha_same_rew.a, new_q.alpha_same_rew.a), 'Same/Rew alphas are different'
#                 assert torch.allclose(old_q.alpha_same_unrew.a, new_q.alpha_same_unrew.a), 'Same/Unrew alphas are different'
#                 assert torch.allclose(old_q.alpha_diff_rew.a, new_q.alpha_diff_rew.a), 'Diff/Rew alphas are different'
#                 assert torch.allclose(old_q.alpha_diff_unrew.a, new_q.alpha_diff_unrew.a), 'Diff/Unrew alphas are different'
                
#                 assert torch.allclose(old_q.k_same_rew, new_q.k_same_rew), 'Same/Rew kappas are different'
#                 assert torch.allclose(old_q.k_same_unrew, new_q.k_same_unrew), 'Same/Unrew kappas are different'
#                 assert torch.allclose(old_q.k_diff_rew, new_q.k_diff_rew), 'Diff/Rew kappas are different'
#                 assert torch.allclose(old_q.k_diff_unrew, new_q.k_diff_unrew), 'Diff/Unrew kappas are different'
                
#             assert torch.allclose(old_model.agents[1].alpha_h.a, new_model.agents[1].alpha.a), 'Persev alphas are different'
#             assert torch.allclose(old_model.agents[2].alpha_g.a, new_model.agents[2].alpha.a), 'Fallacy alphas are different'
#             assert torch.allclose(old_model.beta.weight, new_model.beta.weight), 'Betas are different'
#             assert torch.allclose(old_model.bias, new_model.bias), 'Bias is different'
                
#             save_models[subj][model_name].append({'model': new_model, 'perf': all_models[subj][model_name][i]['perf']})

# agents.save_model(save_models, r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\fit_models.json')

# loaded_models = agents.load_model(r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\fit_models.json')

