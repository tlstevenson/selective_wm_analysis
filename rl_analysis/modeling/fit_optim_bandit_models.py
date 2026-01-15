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
from models import agents
import models.training_helpers as th
import models.sim_helpers as sh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import time
from os import path
import copy
import matplotlib.pyplot as plt

# %% Declare models and simulation options for the fits

n_trials = 500
n_sims = 5

choice_input_dict = {'left': [1,0], 'right': [0,1], None: [0,0]}
outcome_dict = {False: [0], True: [1], None: [0]}

def input_formatter(choice, reward):
    return choice_input_dict[choice] + outcome_dict[reward]

# without random choices, the model will overfit and produce much better LL than the simulation model
# with random choices, the model may slightly overfit, but it is better bounded by the simulation LL
rand_choice = True
def output_choice(output):
    p_left = output[0,0]
    if rand_choice:
        if rand.random() < p_left:
            return 'left'
        else:
            return 'right'
    else:
        if p_left >= 0.5:
            return 'left'
        else:
            return 'right'
        
# define method to convert the sim trial information into input and lable tensors for model training
def io_formatter(sim_trials):
    sess_ids = sim_trials['sessid'].unique()
    inputs = torch.zeros(len(sess_ids), n_trials-1, 3)
    high_port_labels = torch.zeros(len(sess_ids), n_trials-1, 1).type(torch.long)

    for i in range(len(sess_ids)):
        trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
        inputs[i, :, :] = torch.from_numpy(np.array([trial_data['chose_left'][:-1], trial_data['chose_right'][:-1], trial_data['rewarded'].astype(int)[:-1]]).T).type(torch.float)
        # for double value models class labels are 0 for left choice (first element of output) and 1 for right choice
        high_port_labels[i, :, :] = torch.from_numpy(np.array(trial_data['high_side'][1:] == 'right')[:,None]).type(torch.long)
        
    return inputs, high_port_labels
        
# define method to optimize a model to choose the high port as often as possible
def optimize_model(model, sim_options, optimizer, n_cycles, print_time=True, eval_interval=100, loss_diff_exit_thresh=1e-6):
    ''' Optimizes a model parameters based on simulated behavior '''

    # just hardcode this to work with double value models
    loss = nn.CrossEntropyLoss(reduction='none')
    output_formatter = lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1)
    
    running_loss = 0
    loss_arr = np.zeros(n_cycles)

    start_t = time.perf_counter()
    loop_start_t = start_t

    for i in range(n_cycles):

        # simulate new trials based on current model
        sim_trials, _ = sh.simulate_behavior(model, n_trials, n_sims, input_formatter, output_choice, 
                                             output_transform=lambda x: torch.softmax(x, 1), **sim_options)
        
        # format inouts and labels from simulated data
        inputs, labels = io_formatter(sim_trials)

        # train the model 
        model.train()
        optimizer.zero_grad() # zero the gradient buffers
        output, _ = model(inputs)
        
        # format the output and labels to work with the loss function
        orig_shape = labels.shape
        output = output_formatter(output)
        labels = output_formatter(labels)
            
        err = loss(output, labels) # compute the loss
        
        # revert the formatting of the error to match the original output shape
        err = torch.reshape(err, orig_shape)

        err = err.mean()
        
        err.backward() # calculate gradients & store in the tensors
        optimizer.step() # update the network parameters based off stored gradients
        
        # print out the average loss every 100 trials
        loss_val = err.item()
        running_loss += loss_val
        loss_arr[i] = loss_val
        
        if i % eval_interval == eval_interval - 1:
            running_loss /= eval_interval
            if print_time:
                print('Step {}, Loss {:.5f}, elapsed time: {:.1f}s, time per step: {:.3f}s'.format(i+1, running_loss, time.perf_counter()-start_t, (time.perf_counter()-loop_start_t)/eval_interval))
                loop_start_t = time.perf_counter()
            else:
                print('Step {}, Loss {:.5f}'.format(i+1, running_loss))
            running_loss = 0
            
            print(full_model.print_params())
            
            if not loss_diff_exit_thresh is None:
                # check if we should end early
                loss_hist_max = max(loss_arr[i-eval_interval+1:i+1])
                loss_hist_min = min(loss_arr[i-eval_interval+1:i+1])
                if (loss_hist_max - loss_hist_min) < loss_diff_exit_thresh:
                    break

    return loss_arr

# %% Define models and options

models = [agents.QValueAgent(constraints={'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                              'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                              'k_same_unrew': {'fit': False, 'init': -1}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}), 
          agents.QValueAgent(), agents.StateInferenceAgent(), agents.StateInferenceAgent(complement_c_rew=False, complement_c_diff=False)]
model_names = ['Basic Value Agent', 'Full Q Agent', 'All Shared SI Agent', 'Full SI Agent']
sim_options = [{'block_gen_method': 'n_high_choice'}, {'block_gen_method': 'const_switch_p'}, {'block_gen_method': 'block_switch_p'}]
sim_names = ['Uniform Dist Block Length', 'Const Switch P', 'Two-state Switch P']


save_path = r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\optim_models_CELoss.json'
if path.exists(save_path):
    all_models = agents.load_model(save_path)
else:
    all_models = {}
        
# optimize models
n_cycles = 1000
n_fits = 1
n_eval_sims = 20

skip_existing_fits = True
plot_fits = True

for model, model_name in zip(models, model_names):
    for sim_option, sim_name in zip(sim_options, sim_names):
        
        full_name = '{} - {}'.format(model_name, sim_name)
        
        if not full_name in all_models:
            all_models[full_name] = []
            n_model_fits = n_fits
        else:
            if skip_existing_fits:
                n_model_fits = n_fits - len(all_models[full_name])
            else:
                n_model_fits = n_fits
        
        for i in range(n_model_fits):
            
            print('\n{}, fit {}\n'.format(full_name, i))

            model.reset_params()
            full_model = agents.SummationModule([model.clone()])
            
            print('\n{} Original Params:'.format(full_name))
            print(full_model.print_params())
            
            optimizer = optim.Adam(full_model.parameters(recurse=True), lr=0.01)
            optimize_model(full_model, sim_option, optimizer, n_cycles, print_time=True, eval_interval=100, loss_diff_exit_thresh=1e-6)
            
            print('\n{} Fit Params:'.format(full_name))
            print(full_model.print_params())

            # evaluate the fit
            sim_trials, sim_model_data = sh.simulate_behavior(full_model, n_trials, n_eval_sims, input_formatter, output_choice, 
                                                              output_transform=lambda x: torch.softmax(x, 1), **sim_option)
            
            tmp_choice_input_dict = {'left': 1, 'right': -1}
            sim_trials['choice_inputs'] = sim_trials['choice'].apply(lambda choice: tmp_choice_input_dict[choice])
            
            sess_ids = sim_trials['sessid'].unique()
            high_port_labels = np.zeros((n_eval_sims, n_trials-1))

            for j in range(n_eval_sims):
                trial_data = sim_trials[sim_trials['sessid'] == sess_ids[j]]
                # for double value models class labels are 0 for left choice (first element of output) and 1 for right choice
                high_port_labels[j, :] = np.array(trial_data['high_side'][1:] == 'left')
            
            # print true model performance based on random choices
            ll_tot, ll_avg = th.log_likelihood(high_port_labels, sim_model_data['model_output'][:,:,0])
            acc = th.accuracy(high_port_labels, sim_model_data['model_output'][:,:,0])

            print('\nSim Accuracy: {:.2f}%'.format(acc*100))
            print('Sim Total LL: {:.3f}'.format(ll_tot))
            print('Sim Avg LL: {:.5f}'.format(ll_avg))

            # plot fit results
            if plot_fits:
                th.plot_multi_val_fit_results(sim_trials, sim_model_data['model_output'], sim_model_data['agent_states'], 
                                              ['Value'], ['Left', 'Right'], betas=full_model.beta.weight[0], 
                                              title_prefix='{}, run {}, '.format(full_name, i))
                plt.show(block=False)
            
            all_models[full_name].append({'model': full_model, 'sim_options': sim_option, 'perf': {'acc': acc, 'll_tot': ll_tot, 'll_avg': ll_avg}})
            
            agents.save_model(all_models, save_path)


# # %% Investigate simulation results

# left_choice_labels = torch.zeros(n_sims, n_trials-1, 1)

# for i in range(n_sims):
#     trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
#     left_choice_labels[i, :, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)

# # plot results
# if use_single_val_sim:
#     th.plot_single_val_fit_results(sim_trials, sim_model_data['model_output'], sim_model_data['agent_states'][:,:,0,:], 
#                         ['Value', 'Persev', 'Fallacy'], betas=sim_model.beta.weight[0],
#                         title_prefix= 'Sim Model - ')
# else:
#     th.plot_multi_val_fit_results(sim_trials, sim_model_data['model_output'], sim_model_data['agent_states'], 
#                         ['Value', 'Persev'], ['Left', 'Right'], betas=sim_model.beta.weight[0],
#                         title_prefix= 'Sim Model - ')

# # print true model performance based on random choices
# ll_tot, ll_avg = th.log_likelihood(left_choice_labels.numpy(), sim_model_data['model_output'])
# acc = th.accuracy(left_choice_labels.numpy(), sim_model_data['model_output'])

# print('\nSim Accuracy: {:.2f}%'.format(acc*100))
# print('Sim Total LL: {:.3f}'.format(ll_tot))
# print('Sim Avg LL: {:.5f}'.format(ll_avg))

# # %% Fit a new model on the simulated data to recover model parameters

# inputs = torch.zeros(n_sims, n_trials-1, 2)

# for i in range(n_sims):
#     trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
#     inputs[i, :, :] = torch.from_numpy(np.array([trial_data['choice_inputs'][:-1], trial_data['outcome_inputs'][:-1]]).T).type(torch.float)

# n_fits = 1
# for i in range(n_fits):
#     # fit the model
#     fit_model = agents.SummationModule(generate_agents())
    
#     loss = nn.BCEWithLogitsLoss(reduction='none')
#     optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
    
#     loss_vals = th.train_model(fit_model, optimizer, loss, inputs, left_choice_labels, 1500)
    
#     # evaluate the fit
#     fit_output, agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, output_transform=lambda x: 1/(1+np.exp(-x)))
    
#     print('\nSim Model Params:')
#     print(sim_model.print_params())
    
#     print('\nFit Model Params:')
#     print(fit_model.print_params())
    
#     print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
#     print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
#     print('Avg LL: {:.5f}'.format(fit_perf['ll_avg']))
    
#     # plot fit results
#     th.plot_single_val_fit_results(sim_trials, fit_output, agent_states[:,:,0,:], 
#                         ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], betas=fit_model.beta.weight[0],
#                         title_prefix= 'Fit Model {} - '.format(i))

# # %% Fit two-state model to the same simulated results

# # make new outcome inputs as 1/0 instead of 1/-1
# sim_trials['rewarded_int'] = sim_trials['rewarded'].astype(int)

# inputs = torch.zeros(n_sims, n_trials-1, 3)
# choice_class_labels = torch.zeros(n_sims, n_trials-1, 1).type(torch.long)

# for i in range(n_sims):
#     trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
#     inputs[i, :, :] = torch.from_numpy(np.array([trial_data['chose_left'][:-1], trial_data['chose_right'][:-1], trial_data['rewarded_int'][:-1]]).T).type(torch.float)
#     # class labels are 0 for left choice (first element of output) and 1 for right choice
#     choice_class_labels[i, :, :] = torch.from_numpy(np.array(trial_data['chose_right'][1:])[:,None]).type(torch.long)

# # {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_rew'},
# #          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}

# # constrts = [{'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'},
# #          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
# #         {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}]

# n_fits = 2
# for i in range(n_fits):
#     # fit the model formulated in the same way as the single value model
#     # q_agent = agents.QValueAgent(alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
#     #                              k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, 
#     #                              constraints={})
    
#     # if i < 2:
#     #     q_agent = agents.DynamicQAgent(global_lam=True, inverse_update=i==0,
#     #                                    constraints={'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'},
#     #                                                 'gamma_diff_rew': {'share': 'gamma_same_rew'}, 'gamma_diff_unrew': {'share': 'gamma_same_unrew'},
#     #                                                 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}})
        
#     #     model_name = 'Dynamic Q'
#     #     if i == 0:
#     #         model_name += ' w/ Inverse Update'
#     # else:
#     #     q_agent = agents.UncertaintyDynamicQAgent(global_lam=True, shared_side_alphas=True, shared_outcome_alpha_update=False,
#     #                                               constraints={'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'},
#     #                                                            'gamma_alpha_diff_rew': {'share': 'gamma_alpha_same_rew'}, 'gamma_alpha_diff_unrew': {'share': 'gamma_alpha_same_unrew'},
#     #                                                            'gamma_lam_diff_rew': {'share': 'gamma_lam_same_rew'}, 'gamma_lam_diff_unrew': {'share': 'gamma_lam_same_unrew'},
#     #                                                            'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}})
        
#     #     model_name = 'Uncertainty Dynamic Q'
    
#     infer_agent = agents.StateInferenceAgent(complement_c_rew=False)
#     model_name = 'State Inference'
    
#     #fit_model = agents.SummationModule([infer_agent, agents.PerseverativeAgent(n_vals=2), agents.FallacyAgent(n_vals=2)])
#     fit_model = agents.SummationModule([infer_agent, agents.PerseverativeAgent(n_vals=2)])
    
#     loss = nn.CrossEntropyLoss(reduction='none')
#     optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
    
#     # need to reshape the output to work with the cross entropy loss function
#     loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, 2000,
#                                output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1))
    
#     # evaluate the fit
#     fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, 
#                                                           output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
    
#     print('\nSim Model Params:')
#     print(sim_model.print_params())
    
#     print('\nFit Model Params:')
#     print(fit_model.print_params())
    
#     print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
#     print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
#     print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
    
#     # plot fit results
#     th.plot_multi_val_fit_results(sim_trials, fit_output,  fit_agent_states, 
#                                   ['Value', 'Persev'], 
#                                   ['Left', 'Right'], betas=fit_model.beta.weight[0], 
#                                   title_prefix='Fit Model {} - '.format(model_name))
