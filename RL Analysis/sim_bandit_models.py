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
from modeling import agents
import modeling.training_helpers as th
import modeling.sim_helpers as sh
import beh_analysis_helpers as bah
import bandit_beh_helpers as bbh
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
from os import path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# %% Define reusable simulation methods

def generate_agents(alpha0 = None):
    if utils.is_scalar(alpha0):
        alpha0 = [alpha0 for i in range(3)]
    return [agents.SingleValueAgent(alpha0[0]), agents.PerseverativeAgent(alpha0[1]), agents.FallacyAgent(alpha0[2])]

choice_input_dict_single = {'left': 1, 'right': -1, None: 0}
outcome_dict_single = {False: -1, True: 1, None: 0}

choice_input_dict_double = {'left': [1,0], 'right': [0,1], None: [0,0]}
outcome_dict_double = {False: [0], True: [1], None: [0]}

def input_formatter_single(choice, reward):
    return [choice_input_dict_single[choice], outcome_dict_single[reward]]

def input_formatter_double(choice, rewarded, reward):
    return choice_input_dict_double[choice] + outcome_dict_double[rewarded]

# without random choices, the model will overfit and produce much better LL than the simulation model
# with random choices, the model may slightly overfit, but it is better bounded by the simulation LL
rand_choice = True
def output_choice_single(output):
    if rand_choice:
        if rand.random() < output:
            return 'left'
        else:
            return 'right'
    else:
        if output >= 0.5:
            return 'left'
        else:
            return 'right'
        
def output_choice_double(output):
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
        
# %% simulate model

use_single_val_sim = False

beh_options = {'block_gen_method': 'block_switch_rate', 'p_reward_high': [0.1, 0.75], 'p_reward_low': [0.1, 0.45]}

n_trials = 500
n_sims = 5

if use_single_val_sim:
    sim_model = agents.SummationModule(generate_agents([0.5, 0.3, 0.9]), output_layer = nn.Sigmoid(), bias=-0.05, betas=[1.2,0.7,0.01])
    
    sim_trials, sim_model_data = sh.simulate_behavior(sim_model, n_trials, n_sims, input_formatter_single, output_choice_single, 
                                                      block_gen_method='block_switch_p', p_switch=0.15, p_reward=[0.2, 0.7], p_min=0.03, p_max=0.15, p_drift=0.005)
else:
    # sim_model = agents.SummationModule([agents.StateInferenceAgent(p_stay=0.8, c_same_rew=0.7, c_same_unrew=0.1, complement_c_rew=False)],  # , agents.PerseverativeAgent(0.3, n_vals=2)
    #                                    output_layer=nn.Softmax(dim=1), bias=0, betas=[2])
    sim_model = agents.SummationModule([agents.QValueStateInferenceAgent(p_stay=0.36, alpha_high_rew=0.001, alpha_high_unrew=0.838, alpha_low_rew=0, alpha_low_unrew=0,
                                                                         k_high_rew=0.12, k_high_unrew=0.01, k_low_rew=0.155, k_low_unrew=0.155)],  # , agents.PerseverativeAgent(0.3, n_vals=2)
                                       output_layer=nn.Softmax(dim=1), bias=0, betas=[26])

    sim_trials, sim_model_data = sh.simulate_behavior(sim_model, n_trials, n_sims, input_formatter_double, output_choice_double, **beh_options)
    

sim_trials['choice_inputs'] = sim_trials['choice'].apply(lambda choice: choice_input_dict_single[choice] if choice in choice_input_dict_single.keys() else np.nan).to_numpy()
sim_trials['outcome_inputs'] = sim_trials['rewarded'].apply(lambda reward: outcome_dict_single[reward] if reward in outcome_dict_single.keys() else np.nan).to_numpy()
sim_trials['chose_left'] = sim_trials['chose_left'].astype(int)
sim_trials['chose_right'] = sim_trials['chose_right'].astype(int)
    
sess_ids = np.unique(sim_trials['sessid'])

# %% Investigate simulation results

left_choice_labels = torch.zeros(n_sims, n_trials-1, 1)

for i in range(n_sims):
    trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
    left_choice_labels[i, :, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)

# plot results
if use_single_val_sim:
    th.plot_single_val_fit_results(sim_trials, sim_model_data['model_output'], sim_model_data['agent_states'][:,:,0,:], 
                        ['Value', 'Persev', 'Fallacy'], betas=sim_model.beta.weight[0],
                        title_prefix= 'Sim Model - ')
else:
    agent_states = sim_model_data['agent_states']
    betas = sim_model.beta.weight[0].detach().numpy()
    agent_names = ['State']
    
    if isinstance(sim_model.agents[0], agents.QValueStateInferenceAgent):
        value_hist = torch.stack(sim_model.agents[0].v_hist[1:], dim=1).numpy()
        belief_hist = torch.stack(sim_model.agents[0].belief_hist[1:], dim=1).numpy()
        agent_states = np.insert(agent_states, 1, value_hist, axis=3)
        agent_states = np.insert(agent_states, 2, belief_hist, axis=3)
        betas = np.insert(betas, 1, np.array([1,1]))
        agent_names.extend(['Value', 'Belief'])
        
    th.plot_multi_val_fit_results(sim_trials, sim_model_data['model_output'], agent_states, 
                        agent_names, betas=betas, #, 'Persev'
                        title_prefix= 'Sim Model - ')

# print true model performance based on random choices
ll_tot, ll_avg = th.log_likelihood(left_choice_labels.numpy(), sim_model_data['model_output'])
acc = th.accuracy(left_choice_labels.numpy(), sim_model_data['model_output'])

print('\nSim Accuracy: {:.2f}%'.format(acc*100))
print('Sim Total LL: {:.3f}'.format(ll_tot))
print('Sim Avg LL: {:.5f}'.format(ll_avg))

# %% Fit a new model on the simulated data to recover model parameters

inputs = torch.zeros(n_sims, n_trials-1, 2)

for i in range(n_sims):
    trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
    inputs[i, :, :] = torch.from_numpy(np.array([trial_data['choice_inputs'][:-1], trial_data['outcome_inputs'][:-1]]).T).type(torch.float)

n_fits = 1
for i in range(n_fits):
    # fit the model
    fit_model = agents.SummationModule(generate_agents())
    
    loss = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
    
    loss_vals = th.train_model(fit_model, optimizer, loss, inputs, left_choice_labels, 1500)
    
    # evaluate the fit
    fit_output, agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, output_transform=lambda x: 1/(1+np.exp(-x)))
    
    print('\nSim Model Params:')
    print(sim_model.print_params())
    
    print('\nFit Model Params:')
    print(fit_model.print_params())
    
    print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
    print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
    print('Avg LL: {:.5f}'.format(fit_perf['ll_avg']))
    
    # plot fit results
    th.plot_single_val_fit_results(sim_trials, fit_output, agent_states[:,:,0,:], 
                        ['Value Agent (V)', 'Persev Agent (H)', 'Fallacy Agent (G)'], betas=fit_model.beta.weight[0],
                        title_prefix= 'Fit Model {} - '.format(i))

# %% Fit two-state model to the same simulated results

# make new outcome inputs as 1/0 instead of 1/-1
sim_trials['rewarded_int'] = sim_trials['rewarded'].astype(int)

inputs = torch.zeros(n_sims, n_trials-1, 3)
choice_class_labels = torch.zeros(n_sims, n_trials-1, 1).type(torch.long)

for i in range(n_sims):
    trial_data = sim_trials[sim_trials['sessid'] == sess_ids[i]]
    inputs[i, :, :] = torch.from_numpy(np.array([trial_data['chose_left'][:-1], trial_data['chose_right'][:-1], trial_data['rewarded_int'][:-1]]).T).type(torch.float)
    # class labels are 0 for left choice (first element of output) and 1 for right choice
    choice_class_labels[i, :, :] = torch.from_numpy(np.array(trial_data['chose_right'][1:])[:,None]).type(torch.long)

# {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_rew'},
#          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}

# constrts = [{'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'},
#          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
#         {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}]

n_fits = 2
for i in range(n_fits):
    # fit the model formulated in the same way as the single value model
    # q_agent = agents.QValueAgent(alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
    #                              k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, 
    #                              constraints={})
    
    # if i < 2:
    #     q_agent = agents.DynamicQAgent(global_lam=True, inverse_update=i==0,
    #                                    constraints={'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'},
    #                                                 'gamma_diff_rew': {'share': 'gamma_same_rew'}, 'gamma_diff_unrew': {'share': 'gamma_same_unrew'},
    #                                                 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}})
        
    #     model_name = 'Dynamic Q'
    #     if i == 0:
    #         model_name += ' w/ Inverse Update'
    # else:
    #     q_agent = agents.UncertaintyDynamicQAgent(global_lam=True, shared_side_alphas=True, shared_outcome_alpha_update=False,
    #                                               constraints={'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'},
    #                                                            'gamma_alpha_diff_rew': {'share': 'gamma_alpha_same_rew'}, 'gamma_alpha_diff_unrew': {'share': 'gamma_alpha_same_unrew'},
    #                                                            'gamma_lam_diff_rew': {'share': 'gamma_lam_same_rew'}, 'gamma_lam_diff_unrew': {'share': 'gamma_lam_same_unrew'},
    #                                                            'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}})
        
    #     model_name = 'Uncertainty Dynamic Q'
    
    infer_agent = agents.StateInferenceAgent(complement_c_rew=False)
    model_name = 'State Inference'
    
    #fit_model = agents.SummationModule([infer_agent, agents.PerseverativeAgent(n_vals=2), agents.FallacyAgent(n_vals=2)])
    fit_model = agents.SummationModule([infer_agent, agents.PerseverativeAgent(n_vals=2)])
    
    loss = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(fit_model.parameters(recurse=True), lr=0.01)
    
    # need to reshape the output to work with the cross entropy loss function
    loss_vals = th.train_model(fit_model, optimizer, loss, inputs, choice_class_labels, 2000,
                               output_formatter=lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1))
    
    # evaluate the fit
    fit_output, fit_agent_states, fit_perf = th.eval_model(fit_model, inputs, left_choice_labels, 
                                                           output_transform=lambda x: torch.softmax(x, 2)[:,:,0].unsqueeze(2))
    
    print('\nSim Model Params:')
    print(sim_model.print_params())
    
    print('\nFit Model Params:')
    print(fit_model.print_params())
    
    print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
    print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
    print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
    
    # plot fit results
    th.plot_multi_val_fit_results(sim_trials, fit_output,  fit_agent_states, 
                                  ['Value', 'Persev'], 
                                  ['Left', 'Right'], betas=fit_model.beta.weight[0], 
                                  title_prefix='Fit Model {} - '.format(model_name))

# %% Use fitted models to simulate behavior


save_path = r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\fit_models_local.json'
all_models = agents.load_model(save_path)

sim_models = ['Bayes - No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates']#, 'SI - Free Same/Diff Rew Evidence']
beh_options = {#'classic (20/50/80)': {'block_gen_method': 'n_high_choice', 'p_reward': [0.2, 0.5, 0.8]},
               #'classic (10/40/70)': {'block_gen_method': 'n_high_choice', 'p_reward': [0.1, 0.4, 0.7]},
               #'epoch rate switch (10/50/80)': {'block_gen_method': 'block_switch_rate', 'p_reward_high': [0.1, 0.8], 'p_reward_low': [0.1, 0.5]},
               #'epoch rate switch (10/45/75)': {'block_gen_method': 'block_switch_rate', 'p_reward_high': [0.1, 0.75], 'p_reward_low': [0.1, 0.45]},
               #'epoch rate switch (10/40/70)': {'block_gen_method': 'block_switch_rate', 'p_reward_high': [0.1, 0.7], 'p_reward_low': [0.1, 0.4]},
               #'epoch var switch (20/70)': {'block_gen_method': 'block_switch_p', 'p_reward': [0.2, 0.7]},
               #'epoch var switch (10/70)': {'block_gen_method': 'block_switch_p', 'p_reward': [0.1, 0.7]},
               'epoch var switch n high (10/75)': {'block_gen_method': 'block_n_high_choice', 'p_reward': [0.1, 0.75]},
               }

n_sims = 20
n_trials = 500
ind_subj = False
meta_subj = True

analyze_beh = True
analyze_rpe_update_corr = True

plot_model_output = False
use_betas = False
n_plots = 2

def get_model_data(model, sim_idx, info_dict):
    
    if isinstance(model.agents[0], agents.BayesianAgent):
        if len(info_dict) == 0: 
            info_dict.update({
                'point_rpe': np.empty((n_sims, n_trials-1, 2)), 'full_nll': np.empty((n_sims, n_trials-1, 2)), 
                'stay_nll': np.empty((n_sims, n_trials-1, 2)), 'rew_kl_div': np.empty((n_sims, n_trials-1, 2)), 
                'rew_js_div': np.empty((n_sims, n_trials-1, 2)), 'switch_kl_div': np.empty((n_sims, n_trials-1, 1)),
                'p_switch': np.empty((n_sims, n_trials-1, 1))})
    
        info_dict['point_rpe'][sim_idx,:,:] = torch.stack(model.agents[0].state_diff_hist, dim=1).numpy()
        info_dict['full_nll'][sim_idx,:,:] = torch.stack(model.agents[0].nll_hist_full, dim=1).numpy()
        info_dict['stay_nll'][sim_idx,:,:] = torch.stack(model.agents[0].nll_hist_stay, dim=1).numpy()
        info_dict['rew_kl_div'][sim_idx,:,:] = model.agents[0].get_information_update(metric='kl', p_dist='reward')
        info_dict['rew_js_div'][sim_idx,:,:] = model.agents[0].get_information_update(metric='js', p_dist='reward')
        info_dict['switch_kl_div'][sim_idx,:,:] = model.agents[0].get_information_update(metric='kl', p_dist='switch').T
        
        switch_dist_hist = torch.stack(model.agents[0].switch_prior_hist[1:], dim=1)
        switch_point_est = torch.matmul(switch_dist_hist, model.agents[0].prob_vals)
        info_dict['p_switch'][sim_idx,:,:] = switch_point_est.numpy().T

subj_ids = list(all_models.keys())
sim_perfs = {}
model_info = {}

for opt_name, options in beh_options.items():
    sim_perfs[opt_name] = {}
    model_info[opt_name] = {}
    
    for model_name in sim_models:
        sim_perfs[opt_name][model_name] = {}
        model_info[opt_name][model_name] = {}
        
        all_trials = []
        
        # use all subject's model fits
        for subj in subj_ids:
            
            if model_name not in all_models[subj] or len(all_models[subj][model_name]) == 0:
                continue
            
            # get model
            best_model_idx = 0
            for i in range(len(all_models[subj][model_name])):
                if all_models[subj][model_name][i]['perf']['norm_llh'] > all_models[str(subj)][model_name][best_model_idx]['perf']['norm_llh']:
                    best_model_idx = i    

            model = all_models[str(subj)][model_name][best_model_idx]['model'].model.clone()
            
            # collect RPE information
            model_info[opt_name][model_name][subj] = {}
            access_data = lambda model, i: get_model_data(model, i, model_info[opt_name][model_name][subj])

            subj_trials, subj_sim_data = sh.simulate_behavior(model, n_trials, n_sims, input_formatter_double, output_choice_double,
                                                              output_transform=lambda x: torch.softmax(x, 1), access_data=access_data, **options)

            subj_trials['subjid'] = subj
            subj_trials['sessid'] = subj_trials['sessid'] + '-' + subj_trials['subjid']
            subj_trials['chose_left'] = subj_trials['chose_left'].astype(int)
                
            sess_ids = subj_trials['sessid'].unique()
    
            left_choice_labels = torch.zeros(n_sims, n_trials-1, 1)
    
            for i in range(n_sims):
                trial_data = subj_trials[subj_trials['sessid'] == sess_ids[i]]
                left_choice_labels[i, :, :] = torch.from_numpy(np.array(trial_data['chose_left'][1:])[:,None]).type(torch.float)
    
            subj_trials['choice_inputs'] = subj_trials['choice'].apply(lambda choice: choice_input_dict_single[choice] if choice in choice_input_dict_single.keys() else np.nan).to_numpy()
            subj_trials['outcome_inputs'] = subj_trials['rewarded'].apply(lambda reward: outcome_dict_single[reward] if reward in outcome_dict_single.keys() else np.nan).to_numpy()
            
            # plot results
            if plot_model_output:
                
                agent_states = subj_sim_data['agent_states'].copy()
                agent_labels = ['Value']
                             
                if use_betas:
                    betas = model.beta.weight[0].detach().numpy()
                else:
                    betas = None
                    
                # Add switch point estimate to the states to plot
                if isinstance(model.agents[0], agents.BayesianAgent):
                    p_switch = model_info[opt_name][model_name][subj]['p_switch']
                    agent_states = np.concatenate([agent_states, np.tile(np.expand_dims(p_switch, -1), (1,1,2,1))], axis=3)
                    agent_labels.append('Switch')
                    if use_betas:
                        betas = np.insert(betas, 1, np.array([1]))

                th.plot_multi_val_fit_results(subj_trials, subj_sim_data['model_output'], agent_states, 
                                    agent_labels, ['Left', 'Right'], betas=betas, #, 'Persev'
                                    title_prefix= '{}, {} - '.format(model_name, opt_name), n_sess=n_plots)
                
                plt.show(block=False)
    
            if analyze_rpe_update_corr:
                if isinstance(model.agents[0], agents.BayesianAgent):
                    info_dict = model_info[opt_name][model_name][subj]
                    point_rpe = info_dict['point_rpe']
                    full_nll = info_dict['full_nll']
                    stay_nll = info_dict['stay_nll']
                    rew_kl_div = info_dict['rew_kl_div']
                    rew_js_div = info_dict['rew_js_div']
                    switch_kl_div = info_dict['switch_kl_div']
                    
                    side_p_reward = subj_sim_data['agent_states'].copy()[:,:,:,0]
                    
                    n_rows = 5
                    sides = ['Left', 'Right']
                    
                    plot_ids = sess_ids[:n_plots]
                    
                    for i, sess_id in enumerate(plot_ids):
                        trial_data = subj_trials[subj_trials['sessid'] == sess_id]
                        fig, axs = plt.subplots(n_rows, 1, figsize=(12, n_rows*3), layout='constrained')
                            
                        # get block transitions
                        block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
                        block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
                        block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
                            
                        fig.suptitle('RPE/Update Metrics - Session {}'.format(sess_id))

                        x = np.arange(len(trial_data)-1)+1
                        
                        # plot side reward probability versus actual choices
                        ax = axs[0]
                        
                        for k, side in enumerate(sides):
                            ax.plot(x, side_p_reward[i,:len(trial_data)-1,k], alpha=0.7, label='p(Reward) {}'.format(side))
                
                        ax.set_ylabel('p(Reward)')
                        ax.set_xlabel('Trial')
                        ax.set_title('Mean Estimated p(Reward)')
                        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                        ax.margins(x=0.01)
                        
                        th._draw_choices(trial_data, ax)
                        th._draw_blocks(block_switch_trials, block_rates, ax)
                        
                        # plot point RPE
                        ax = axs[1]
                        z_rpe = point_rpe[i,:len(trial_data)-1,:]
                        # rescale to have variance of 1 while keeping sign the same
                        #z_rpe = z_rpe/np.std(z_rpe)
                        
                        for k, side in enumerate(sides):
                            ax.plot(x, z_rpe[:,k], alpha=0.7, label='RPE {}'.format(side))

                        ax.set_ylabel('Z-score')
                        ax.set_xlabel('Trial')
                        ax.set_title('Point Estimate RPE')
                        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                        ax.margins(x=0.01)
                        
                        th._draw_choices(trial_data, ax)
                        th._draw_blocks(block_switch_trials, block_rates, ax)

                        # plot KL divergence
                        ax = axs[2]
                        
                        # rescale variance to 1
                        z_kld = rew_kl_div[i,:len(trial_data)-1,:]
                        #z_kld = z_kld/np.std(z_kld)
                        
                        for k, side in enumerate(sides):
                            ax.plot(x, z_kld[:,k], alpha=0.7, label='Rew KL Div. {}'.format(side))
                
                        ax.set_ylabel('Z-score')
                        ax.set_xlabel('Trial')
                        ax.set_title('KL Divergence')
                        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                        ax.margins(x=0.01)
                        
                        th._draw_choices(trial_data, ax)
                        th._draw_blocks(block_switch_trials, block_rates, ax)
                        
                        # plot log likelihood for stays and switches
                        ax = axs[3]

                        # z-score so scale is same 
                        z_nll_full = -full_nll[i,:len(trial_data)-1,:]
                        #z_nll_full = z_nll_full/np.std(z_nll_full)

                        for k, side in enumerate(sides):
                            ax.plot(x, z_nll_full[:,k], alpha=0.7, label='Outcome LL (full) {}'.format(side))
                
                        ax.set_ylabel('Z-score')
                        ax.set_xlabel('Trial')
                        ax.set_title('Outcome Log Likelihood - Stays and Switches')
                        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                        ax.margins(x=0.01)
                        
                        th._draw_choices(trial_data, ax)
                        th._draw_blocks(block_switch_trials, block_rates, ax)
                        
                        # plot log likelihood of outcome for just stays
                        ax = axs[4]

                        # z-score so scale is same 
                        z_nll_stay = -stay_nll[i,:len(trial_data)-1,:]
                        #z_nll_stay = z_nll_stay/np.std(z_nll_stay)

                        for k, side in enumerate(sides):
                            ax.plot(x, z_nll_stay[:,k], alpha=0.7, label='Outcome LL (stay) {}'.format(side))
                
                        ax.set_ylabel('Z-score')
                        ax.set_xlabel('Trial')
                        ax.set_title('Outcome Log Likelihood - Stays Only')
                        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
                        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                        ax.margins(x=0.01)
                        
                        th._draw_choices(trial_data, ax)
                        th._draw_blocks(block_switch_trials, block_rates, ax)
                    
                    plt.show(block=False)
    
    
            # collect 'best' possible fit model performance from the simulated output
            sim_perfs[opt_name][model_name][subj] = th.calc_model_performance(model, subj_sim_data['model_output'], left_choice_labels.numpy())
            all_trials.append(subj_trials)
        
        # aggregate all subject trials together
        all_trials = pd.concat(all_trials)
        
        if analyze_beh:
            # analyze behavioral results
            bah.calc_trial_hist(all_trials, 3)
            bbh.make_trial_hist_labels(all_trials, 3)
            bbh.make_rew_hist_labels(all_trials, 3)
            
            plt_suffix = ' - {}; {}'.format(model_name, opt_name)
            
            # aggregate count tables into dictionary
            count_columns = ['side_prob', 'block_prob', 'high_side']
            column_titles = ['Side Probability (L/R)', 'Block Probabilities', 'High Side']
            count_dict_pct = bah.get_count_dict(all_trials, 'subjid', count_columns, normalize=True)

            # plot bar charts of trial distribution

            fig, axs = plt.subplots(len(count_dict_pct.keys()), 1, layout='constrained',
                                    figsize=(3+0.25*len(subj_ids), 3*len(count_dict_pct.keys())))
            for i, (col_name, title) in enumerate(zip(count_columns, column_titles)):
                bah.plot_counts(count_dict_pct[col_name], axs[i], title, '% Trials', 'v')
            
            bbh.analyze_trial_hist_counts(all_trials, 3, plot_suffix=plt_suffix)
    
            # Count Block Lengths
            bbh.count_block_lengths(all_trials, ind_subj=ind_subj, meta_subj=meta_subj, plot_suffix=plt_suffix)
    
            # Analyze response metrics
    
            plot_simple_summary = False
    
            bbh.analyze_choice_behavior(all_trials, n_back_hist=3, plot_simple_summary=plot_simple_summary, meta_subj=meta_subj, ind_subj=ind_subj, plot_suffix=plt_suffix)
            bbh.analyze_trial_choice_behavior(all_trials, plot_simple_summary=plot_simple_summary, meta_subj=meta_subj, ind_subj=ind_subj, plot_suffix=plt_suffix)
    
            # Logistic regression of choice by past choices and trial outcomes
            
            separate_block_rates = True
            n_back = 5
    
            # whether to model the interaction as win-stay/lose switch or not
            include_winstay = False
            # whether to include reward as a predictor on its own
            include_reward = False
            # whether to have separate interaction terms for rewarded trials and unrewarded trials
            separate_unreward = False
            # whether to include all possible interaction combinations
            include_full_interaction = False
    
            plot_cis = True
    
            bbh.logit_regress_side_choice(all_trials, n_back=n_back, separate_block_rates=separate_block_rates, include_winstay=include_winstay, include_reward=include_reward, plot_suffix=plt_suffix,
                                          separate_unreward=separate_unreward, include_full_interaction=include_full_interaction, plot_cis=plot_cis, ind_subj=ind_subj, meta_subj=meta_subj)
                    
            # Logistic regression of stay/switch choice by past choices and trial outcomes
    
            separate_block_rates = True
            n_back = 5
    
            # whether to fit stays or switches
            fit_switches = False
            # whether to have reward predictors be 1/0 or +1/-1
            include_unreward = False
            # whether to include choice (same/diff of prior choice) as a predictor on its own
            include_choice = True
            # whether to model choice as 1/0 or +1/-1 for same/diff
            include_diff_choice = True
            # whether to include an interaction term of choice x reward
            include_interaction = True
            # whether to include all possible interaction combinations. Supersedes above options
            include_full_interaction = True
    
            plot_cis = True
    
            bbh.logit_regress_stay_choice(all_trials, n_back=n_back, separate_block_rates=separate_block_rates, fit_switches=fit_switches, include_unreward=include_unreward, 
                                          include_choice=include_choice, include_diff_choice=include_diff_choice, include_interaction=include_interaction, include_full_interaction=include_full_interaction, 
                                          plot_cis=plot_cis, ind_subj=ind_subj, meta_subj=meta_subj, plot_suffix=plt_suffix)
    
            plt.show(block=False)
            
        
                
# %% simulate value model

import modeling.agents_value as val_agents
import vol_bandit_beh_helpers as vbbh

def input_formatter_value(choice, reward):
    return choice_input_dict_double[choice] + [reward]

n_trials = 350
n_sims = 10

options = {'high_choice_epoch_range': [120, 200]}

sim_model = val_agents.SummationModule([val_agents.QValueAgent(alpha_same=0.5, alpha_diff=0.2, alpha_avg=0.2, v_init=16, k_diff=8, decay_to_avg=True)],
                                       output_layer=nn.Softmax(dim=1), bias=0, betas=[1])

sim_trials, sim_model_data = sh.simulate_var_rew_behavior(sim_model, n_trials, n_sims, input_formatter_value, output_choice_double, **options)
    

sim_trials['choice_inputs'] = sim_trials['choice'].apply(lambda choice: choice_input_dict_single[choice] if choice in choice_input_dict_single.keys() else np.nan).to_numpy()
sim_trials['outcome_inputs'] = sim_trials['rewarded'].apply(lambda reward: outcome_dict_single[reward] if reward in outcome_dict_single.keys() else np.nan).to_numpy()
sim_trials['chose_left'] = sim_trials['chose_left'].astype(int)
sim_trials['chose_right'] = sim_trials['chose_right'].astype(int)
sim_trials['subjid'] = 'sim'
    
sess_ids = np.unique(sim_trials['sessid'])

agent_states = sim_model_data['agent_states']
betas = sim_model.beta.weight[0].detach().numpy()
agent_names = ['State']

th.plot_multi_val_fit_results(sim_trials, sim_model_data['model_output'], agent_states, 
                    agent_names, betas=betas,
                    title_prefix= 'Sim Model - ')

# aggregate count tables into dictionary
count_columns = ['side_prob', 'block_prob', 'high_side']
column_titles = ['Side Volume (L/R)', 'Block Probabilities', 'High Side']
count_dict_pct = bah.get_count_dict(sim_trials, 'subjid', count_columns, normalize=True)

# plot bar charts of trial distribution

fig, axs = plt.subplots(len(count_dict_pct.keys()), 1, layout='constrained',
                        figsize=(3+0.25*len(subj_ids), 3*len(count_dict_pct.keys())))
for i, (col_name, title) in enumerate(zip(count_columns, column_titles)):
    bah.plot_counts(count_dict_pct[col_name], axs[i], title, '% Trials', 'v')

# Count Block Lengths
ind_subj = False
meta_subj = True

vbbh.count_block_lengths(sim_trials,  ind_subj=ind_subj, meta_subj=meta_subj)

vbbh.analyze_choice_behavior(sim_trials, meta_subj=meta_subj, ind_subj=ind_subj)
vbbh.analyze_trial_choice_behavior(sim_trials, meta_subj=meta_subj, ind_subj=ind_subj)

# %%

def sample_random_sum(val_min, val_max, count_min, count_max, size=100000):
    
    results = []
    for _ in range(size):
        n = np.random.randint(count_min, count_max+1, 1)[0]
        vals = np.random.randint(val_min, val_max+1, n)
        results.append(sum(vals))

    return results

# high_vol_choice_range = kwargs.get('high_vol_choice_range', [3, 8])
# low_vol_choice_range = kwargs.get('low_vol_choice_range', [15, 30])
# consec_blocks_range_low_vol = kwargs.get('consec_blocks_range_low_vol', [5,10])
# consec_blocks_range_high_vol = kwargs.get('consec_blocks_range_high_vol', [15,30])

low_vol = sample_random_sum(15, 25, 5, 8)
high_vol = sample_random_sum(3, 8, 15, 25)

# Optional: plot the empirical distribution
n_bins = 40
plt.hist(low_vol, bins=n_bins, density=True, alpha=0.5, label='low vol')
plt.hist(high_vol, bins=n_bins, density=True, alpha=0.5, label='high vol')
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()
