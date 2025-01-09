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

# %% Create model for simulation and simulate trials

def generate_agents(alpha0 = None):
    if utils.is_scalar(alpha0):
        alpha0 = [alpha0 for i in range(3)]
    return [agents.SingleValueAgent(alpha0[0]), agents.PerseverativeAgent(alpha0[1]), agents.FallacyAgent(alpha0[2])]

sim_model = agents.SummationModule(generate_agents([0.4, 0.7, 0.9]), output_layer = nn.Sigmoid(), bias=-0.05, betas=[1.6,0.8,0.2])

choice_input_dict = {'left': 1, 'right': -1, None: 0}
outcome_dict = {False: -1, True: 1, None: 0}
choice_output_dict = {'left': 1, 'right': 0}

def input_formatter(choice, reward):
    return [choice_input_dict[choice], outcome_dict[reward]]

def output_choice(output):
    if rand.random() < output:
        return 'left'
    else:
        return 'right'
    
# simulate model
n_trials = 500
sim_trials, model_data = sh.simulate_behavior(sim_model, n_trials, input_formatter, output_choice)

# %% Plot simulation results

# these columns needed for plotting
sim_trials['sessid'] = 'Sim 1'
sim_trials['choice_inputs'] = sim_trials['choice'].apply(lambda choice: choice_input_dict[choice] if choice in choice_input_dict.keys() else np.nan).to_numpy()
sim_trials['outcome_inputs'] = sim_trials['rewarded'].apply(lambda reward: outcome_dict[reward] if reward in outcome_dict.keys() else np.nan).to_numpy()
sim_trials['choice_outputs'] = sim_trials['choice'].apply(lambda choice: choice_output_dict[choice] if choice in choice_output_dict.keys() else np.nan).to_numpy()

th.plot_fit_results(sim_trials, model_data['model_output'][:,1:,:], model_data['agent_states'][:,1:,:], 
                    ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], betas=sim_model.beta.weight[0],
                    title_prefix= 'Sim Model - ')

# print true model performance based on random choices

left_choice_labels = torch.zeros(1, n_trials-1, 1)
left_choice_labels[0, :, :] = torch.from_numpy(np.array(sim_trials['choice_outputs'][1:])[:,None]).type(torch.float)

ll_tot, ll_avg = th.log_likelihood(left_choice_labels.numpy(), model_data['model_output'][:,1:,:])
acc = th.accuracy(left_choice_labels.numpy(), model_data['model_output'][:,1:,:])

print('\nSim Accuracy: {:.2f}%'.format(acc*100))
print('Sim Total LL: {:.2f}'.format(ll_tot))
print('Sim Avg LL: {:.2f}'.format(ll_avg))

# %% Fit a new model on the simulated data to recover model parameters

inputs = torch.zeros(1, n_trials-1, 2)
inputs[0, :, :] = torch.from_numpy(np.array([sim_trials['choice_inputs'][:-1], sim_trials['outcome_inputs'][:-1]]).T).type(torch.float)

n_fits = 1
for i in range(n_fits):
    # fit the model
    fit_model = agents.SummationModule(generate_agents())
    
    loss = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
    
    loss_vals = th.train_model(fit_model, optimizer, loss, inputs, left_choice_labels, 1500)
    
    # evaluate the fit
    fit_output, fit_output_data, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, output_transform=lambda x: 1/(1+np.exp(-x)))
    
    print('\nSim Model Params:')
    print(sim_model.print_params())
    
    print('\nFit Model Params:')
    print(fit_model.print_params())
    
    print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
    print('Total LL: {:.2f}'.format(fit_perf['ll_total']))
    print('Avg LL: {:.2f}'.format(fit_perf['ll_avg']))
    
    # plot fit results
    th.plot_fit_results(sim_trials, fit_output, fit_output_data['agent_states'][:,:,0,:], 
                        ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], betas=fit_model.beta.weight[0],
                        title_prefix= 'Fit Model {} - '.format(i))

# %% Fit two-state model to the same simulated results

sim_trials['chose_left'] = (sim_trials['choice'].to_numpy() == 'left').astype(int)
sim_trials['chose_right'] = (sim_trials['choice'].to_numpy() == 'right').astype(int)

inputs = torch.zeros(1, n_trials-1, 3)
choice_class_labels = torch.zeros(1, n_trials-1, 1).type(torch.long)

inputs[0, :, :] = torch.from_numpy(np.array([sim_trials['chose_left'][:-1], sim_trials['chose_right'][:-1], sim_trials['outcome_inputs'][:-1]]).T).type(torch.float)
# class labels are 0 for left choice (first element of output) and 1 for right choice
choice_class_labels[0, :, :] = torch.from_numpy(np.array(sim_trials['chose_right'][1:])[:,None]).type(torch.long)

n_fits = 1
for i in range(n_fits):
    # fit the model
    fit_model = agents.SummationModule([agents.QValueAgent(), agents.PerseverativeAgent(n_vals=2), agents.FallacyAgent(n_vals=2)], 
                                       ) # output_layer=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2)
    
    #loss = nn.BCELoss(reduction='none')
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
    
    loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, 2000, output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1))
    
    #%% evaluate the fit
    fit_output, fit_output_data, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, 
                                                          output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
    
    print('\nSim Model Params:')
    print(sim_model.print_params())
    
    print('\nFit Model Params:')
    print(fit_model.print_params())
    
    print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
    print('Total LL: {:.2f}'.format(fit_perf['ll_total']))
    print('Avg LL: {:.2f}'.format(fit_perf['ll_avg']))
    
    # # plot fit results
    # th.plot_fit_results(sim_trials, fit_output, fit_output_data['agent_states'][:,:,0,:], 
    #                     ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], betas=fit_model.beta.weight[0],
    #                     title_prefix= 'Fit Model {} - '.format(i))