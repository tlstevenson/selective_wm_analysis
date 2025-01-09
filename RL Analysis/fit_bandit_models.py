# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:44:39 2024

@author: tanne
"""

# %% Imports

import init
import pandas as pd
from pyutils import utils
import hankslab_db.basicRLtasks_db as db
from hankslab_db import db_access
from models import agents
import models.training_helpers as mh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand

# %% Load Data

subj_ids = 179 #179, 188, 191, 207, 182]
sess_ids = db_access.get_subj_sess_ids(subj_ids, protocol='ClassicRLTasks', stage_num=2)

## Create 3-D inputs tensor and 3-D labels tensor
n_sess = len(utils.flatten(sess_ids))

# get session data
reload = False
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
sess_data = loc_db.get_behavior_data(utils.flatten(sess_ids), reload=reload)
# filter out no responses
sess_data = sess_data[sess_data['hit']==True]

## add a column to represent choice (+1 right and -1 left) and outcome (+1 rewarded and -1 unrewarded)
choice_input_dict = {'left': -1, 'right': 1}
choice_output_dict = {'left': 0, 'right': 1}
outcome_dict = {False: -1, True: 1}

# Add choice and outcome inputs as new columns to sess data
sess_data['choice_inputs'] = sess_data['choice'].apply(lambda choice: choice_input_dict[choice] if choice in choice_input_dict.keys() else np.nan).to_numpy()
sess_data['outcome_inputs'] = sess_data['rewarded'].apply(lambda reward: outcome_dict[reward] if reward in outcome_dict.keys() else np.nan).to_numpy()
sess_data['choice_outputs'] = sess_data['choice'].apply(lambda choice: choice_output_dict[choice] if choice in choice_output_dict.keys() else np.nan).to_numpy()

max_trials = np.max(sess_data.groupby('sessid').size())

input_size = 2 #choice and outcome
output_size = 1

#create empty tensors with the max number of trials across all sessions
inputs = torch.zeros(n_sess, max_trials-1, input_size)
right_choice_labels = torch.zeros(n_sess, max_trials-1, output_size)
both_choice_labels = torch.zeros_like(right_choice_labels)
trial_mask = torch.zeros_like(right_choice_labels)

# populate tensors from behavioral data
for i, sess_id in enumerate(utils.flatten(sess_ids)):
    trial_data = sess_data[sess_data['sessid'] == sess_id]
    n_trials = len(trial_data) - 1 # one less because we predict the next choice based on the prior choice
    
    inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['choice_inputs'][:-1], trial_data['outcome_inputs'][:-1]]).T).type(torch.float)
    right_choice_labels[i, :n_trials, :] = torch.from_numpy(np.array(trial_data['choice_outputs'][1:])[:,None]).type(torch.float)
    both_choice_labels[i, :n_trials, :] = torch.from_numpy(np.array(trial_data['choice_inputs'][1:])[:,None]).type(torch.float)
    
    trial_mask[i, :n_trials, :] = 1
    
# %% Compare different loss functions

def generate_agents(alpha0 = None):
    return [agents.SingleValueAgent(alpha0), agents.PerseverativeAgent(alpha0), agents.FallacyAgent(alpha0)]

batch_size = None
n_cycles = 3000

# Instantiate each agent and the summation class
print('MSE Loss, tanh:')
net_mse_tanh = agents.SummationModule(generate_agents(0.1), output_layer = nn.Tanh())
loss = nn.MSELoss(reduction='none')
optimizer_mse_tanh = optim.Adam(net_mse_tanh.parameters(recurse=True), lr=0.01)

mh.train_model(net_mse_tanh, optimizer_mse_tanh, loss, inputs, both_choice_labels, trial_mask, n_cycles, batch_size=batch_size)

tanh_transform = lambda x: (x+1)/2
[output_mse_tanh, output_data_mse_tanh, perf_mse_tanh] = mh.eval_model(net_mse_tanh, inputs, both_choice_labels, trial_mask, output_transform=tanh_transform, label_transform=tanh_transform)


print('MSE Loss, sigmoid:')
net_mse_sig = agents.SummationModule(generate_agents(0.1), output_layer = nn.Sigmoid())
loss = nn.MSELoss(reduction='none')
optimizer_mse_sig = optim.Adam(net_mse_sig.parameters(recurse=True), lr=0.01)

mh.train_model(net_mse_sig, optimizer_mse_sig, loss, inputs, right_choice_labels, trial_mask, n_cycles, batch_size=batch_size)

[output_mse_sig, output_data_mse_sig, perf_mse_sig] = mh.eval_model(net_mse_sig, inputs, right_choice_labels, trial_mask)


print('BCE Loss:')
net_bce = agents.SummationModule(generate_agents(0.1), output_layer = nn.Sigmoid())
loss = nn.BCELoss(reduction='none')
optimizer_bce = optim.Adam(net_bce.parameters(recurse=True), lr=0.01)

mh.train_model(net_bce, optimizer_bce, loss, inputs, right_choice_labels, trial_mask, n_cycles, batch_size=batch_size)

[output_bce, output_data_bce, perf_bce] = mh.eval_model(net_bce, inputs, right_choice_labels, trial_mask)


print('BCE with Logits Loss:')
net_bce_logits = agents.SummationModule(generate_agents(0.1))
loss = nn.BCEWithLogitsLoss(reduction='none')
optimizer_bce_logits = optim.Adam(net_bce_logits.parameters(recurse=True), lr=0.01)

mh.train_model(net_bce_logits, optimizer_bce_logits, loss, inputs, right_choice_labels, trial_mask, n_cycles, batch_size=batch_size)

[output_bce_logits, output_data_bce_logits, perf_bce_logits] = mh.eval_model(net_bce_logits, inputs, right_choice_labels, trial_mask, output_transform=lambda x: 1/(1+np.exp(-x)))


# Print Model Parameters and fit qualities

models = [net_mse_tanh, net_mse_sig, net_bce, net_bce_logits]
perfs = [perf_mse_tanh, perf_mse_sig, perf_bce, perf_bce_logits]
names = ['mse tanh', 'mse sigmoid', 'bce', 'bce w/ logits']

for model, perf, name in zip(models, perfs, names):
    print('{}:'.format(name))

    mh.print_params(model)

    print('Accuracy: {:.2}%'.format(perf['acc']))
    print('Total LL: {:.2}'.format(perf['ll_total']))
    print('Avg LL: {:.2}'.format(perf['ll_avg']))
    print('')
    
# Plot model outputs

mh.plot_fit_results(sess_data, output_mse_tanh, output_data_mse_tanh['agent_states'][:,:,0,:], ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], 
                    betas=net_mse_tanh.beta.weight[0], title_prefix='MSE Tanh Loss - ')

mh.plot_fit_results(sess_data, output_mse_sig, output_data_mse_sig['agent_states'][:,:,0,:], ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], 
                    betas=net_mse_sig.beta.weight[0], title_prefix='MSE Sigmoid Loss - ')

mh.plot_fit_results(sess_data, output_bce, output_data_bce['agent_states'][:,:,0,:], ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], 
                    betas=net_bce.beta.weight[0], title_prefix='BCE Loss - ')

mh.plot_fit_results(sess_data, output_bce_logits, output_data_bce_logits['agent_states'][:,:,0,:], ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], 
                    betas=net_bce_logits.beta.weight[0], title_prefix='BCE w/ Logits Loss - ')