# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:02:02 2024

@author: tanne
"""

import torch
import torch.nn as nn
import torch.optim as optim
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pyutils import cluster_utils
from sys_neuro_tools import plot_utils
import beh_analysis_helpers as bah
import agents
import time
from filelock import FileLock
from os import path

on_cluster = cluster_utils.on_cluster()

# %% Fitting Setup Methods

def define_choice_outcome(sess_data):
    # add needed columns to the session data structure
    
    # for basic model fits tracking only one side, have 1/-1 
    choice_input_dict = {'left': 1, 'right': -1}
    outcome_dict = {False: -1, True: 1}

    # Add choice and outcome inputs as new columns to sess data
    sess_data['choice_inputs'] = sess_data['choice'].apply(lambda choice: choice_input_dict[choice] if choice in choice_input_dict.keys() else np.nan).to_numpy()
    sess_data['outcome_inputs'] = sess_data['rewarded'].apply(lambda reward: outcome_dict[reward] if reward in outcome_dict.keys() else np.nan).to_numpy()
    
    # for more sophisticated model fits, have binary variables
    sess_data['rewarded_int'] = sess_data['rewarded'].astype(int)
    sess_data['chose_left_int'] = sess_data['chose_left'].astype(int)
    sess_data['chose_right_int'] = sess_data['chose_right'].astype(int)
    
    return sess_data

def get_model_training_data(sess_data, basic_model, limit_mask=False, n_limit_hist=2):
    
    if limit_mask:
        if not bah.trial_hist_exists(sess_data):
            bah.calc_trial_hist(sess_data, n_limit_hist)
    
    # filter out no responses
    sess_data = sess_data[sess_data['hit'] == True]
    sess_ids = np.unique(sess_data['sessid'])
    
    sess_data = define_choice_outcome(sess_data)

    n_sess = len(sess_ids)
    max_trials = np.max(sess_data.groupby('sessid').size())
    
    # create input and label tensors
    basic_inputs = torch.zeros(n_sess, max_trials-1, 2).type(torch.float)
    two_side_inputs = torch.zeros(n_sess, max_trials-1, 3).type(torch.float)
    left_choice_labels = torch.zeros(n_sess, max_trials-1, 1).type(torch.float)
    # these need to be long for CrossEntropyLoss to work
    choice_class_labels = torch.zeros(n_sess, max_trials-1, 1).type(torch.long)
    
    trial_mask_train = torch.zeros(n_sess, max_trials-1, 1)
    trial_mask_eval = torch.zeros(n_sess, max_trials-1, 1)
    
    # populate tensors from behavioral data
    for i, sess_id in enumerate(sess_ids):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        n_trials = len(trial_data) - 1 # one less because we predict the next choice based on the prior choice
        
        basic_inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['choice_inputs'][:-1], trial_data['outcome_inputs'][:-1]]).T)
        two_side_inputs[i, :n_trials, :] = torch.from_numpy(np.array([trial_data['chose_left_int'][:-1], trial_data['chose_right_int'][:-1], trial_data['rewarded_int'][:-1]]).T)
        
        left_choice_labels[i, :n_trials, :] = torch.from_numpy(trial_data['chose_left_int'].to_numpy()[1:][:,None])
        # choice class labels are the correct indices of the choices, which corresponds to whether they chose right (1) or not (0)
        choice_class_labels[i, :n_trials, :] = torch.from_numpy(trial_data['chose_right_int'].to_numpy()[1:][:,None])
        
        mask = np.full((n_trials,1), 1)
        # always ignore forced choice trials
        if 'forced_choice' in trial_data.columns:
            mask[trial_data['forced_choice'].to_numpy()[1:]] = 0
            
        trial_mask_eval[i, :n_trials, :] = torch.from_numpy(mask)
            
        if limit_mask:
            # exclude trials that are stays after the animal has received repeated rewards on the same side
            choices = trial_data['choice'].to_numpy()
            stays = choices[:-1] == choices[1:]
            all_prev_reward = trial_data['rew_hist'].apply(lambda x: np.all(x[:n_limit_hist] == 1)).to_numpy()[1:]
            choice_hist_same = trial_data['choice_hist'].apply(lambda x: np.all(x[:n_limit_hist] == x[0])).to_numpy()[1:]
            
            mask[all_prev_reward & choice_hist_same & stays] = 0
            
        trial_mask_train[i, :n_trials, :] = torch.from_numpy(mask)
        
    if basic_model:
        return {'inputs': basic_inputs, 'labels': left_choice_labels,
                'trial_mask_train': trial_mask_train, 'trial_mask_eval': trial_mask_eval}
    else:
        return {'inputs': two_side_inputs, 'labels': choice_class_labels,
                'trial_mask_train': trial_mask_train, 'trial_mask_eval': trial_mask_eval}

def get_loss_output_transforms(basic_model):
    if basic_model:
        loss = nn.BCEWithLogitsLoss(reduction='none')
        train_output_formatter = None
        eval_output_transform = lambda x: 1/(1+np.exp(-x))
    else:
        loss = nn.CrossEntropyLoss(reduction='none')
        train_output_formatter = lambda x: torch.reshape(x, [-1, x.shape[-1]]).squeeze(-1)
        eval_output_transform = lambda x: torch.softmax(x, 2)[:,:,1].unsqueeze(2)
        
    return {'loss': loss, 'train_output_formatter': train_output_formatter, 'eval_output_transform': eval_output_transform}

# %% Fitting methods

def_n_fits = 2
def_n_steps = 10000
def_end_tol = 1e-6
        
def fit_model(model, model_name, inputs, labels, trial_mask_train, trial_mask_eval, loss, subj_name, save_path, n_fits=def_n_fits, 
              n_steps=def_n_steps, end_tol=def_end_tol, optim_generator=None, train_output_formatter=None, 
              eval_output_transform=None, skip_existing_fits=True, refit_existing=False, print_train_params=False, equal_sess_weight=False):

    lock = FileLock('fitting.lock')
    
    if optim_generator is None:
        optim_generator = lambda p: optim.Adam(p, lr=0.01)
        
    if path.exists(save_path):
        with lock:
            model_dict = agents.load_model(save_path)
    else:
        model_dict = {}
        
    if not str(subj_name) in model_dict:
        model_dict[str(subj_name)] = {}

    # determine the number of fit repeats
    if model_name not in model_dict[str(subj_name)]:
        model_dict[str(subj_name)][model_name] = []
        n_exist_fits = 0
    else:
        n_exist_fits = len(model_dict[str(subj_name)][model_name])
    
    # default number of fits 
    # only do 1 fit at a time on the cluster
    n_model_fits = 1 if on_cluster else n_fits
    
    if skip_existing_fits:
        if n_exist_fits >= n_fits:
            n_model_fits = 0
        else:
            n_model_fits = 1 if on_cluster else (n_fits - n_exist_fits)
    
    elif refit_existing:
        n_model_fits = n_exist_fits
    
    print('Fitting model for {}\n'.format(subj_name))

    i = 0
    while i < n_model_fits:
        print('\n{}, fit {}\n'.format(model_name, i))

        if refit_existing:
            model = model_dict[str(subj_name)][model_name][i]['model'].model.clone()
        else: 
            model.reset_params()

        optimizer = optim_generator(model.parameters(recurse=True))
        
        try:
            # train the model
            _ = train_model(model, optimizer, loss, inputs, labels, n_steps, trial_mask=trial_mask_train,
                            output_formatter=train_output_formatter, loss_diff_exit_thresh=end_tol,
                            print_params=print_train_params, equal_sess_weight=equal_sess_weight)
            
            # evaluate the fit
            _, _, fit_perf = eval_model(model, inputs, labels, trial_mask=trial_mask_eval,
                                        output_transform=eval_output_transform)
            
            print('\n{} Params:'.format(model_name))
            print(model.print_params())
            
            print('\nAccuracy: {:.2f}%'.format(fit_perf['acc']*100))
            print('Total LL: {:.3f}'.format(fit_perf['ll_total']))
            print('Avg LL: {:5f}'.format(fit_perf['ll_avg']))
        
            with lock:
                # if on the cluster, make sure to reload any changes made to the file by other processes
                if on_cluster and path.exists(save_path):
                    model_dict = agents.load_model(save_path)
                    
                    if not str(subj_name) in model_dict:
                        model_dict[str(subj_name)] = {}
                    if not model_name in model_dict[str(subj_name)]:
                        model_dict[str(subj_name)][model_name] = []
                
                if refit_existing:
                    model_dict[str(subj_name)][model_name][i] = {'model': model, 'perf': fit_perf}
                else:
                    model_dict[str(subj_name)][model_name].append({'model': model, 'perf': fit_perf})
                
                agents.save_model(model_dict, save_path)
            
            i += 1
            
        except RuntimeError as e:
            print('Error: {}. \nTrying Again...'.format(e))


def train_model(model, optimizer, loss, inputs, labels, n_cycles, trial_mask=None, batch_size=None, output_formatter=None, 
                print_time=True, eval_interval=100, loss_diff_exit_thresh=1e-6, print_params=False, equal_sess_weight=False):
    ''' A general-purpose method for training a network
       this assumes the batch is the first dimension of the tensor
       also the loss function must have reduction = 'none' 
    '''
    
    n_sess = inputs.shape[0]
    running_loss = 0
    loss_arr = np.zeros(n_cycles)
    
    model.train()
    
    if trial_mask is None:
        trial_mask = torch.zeros_like(labels)+1
        
    if print_params:
        print(model.print_params())
        
    start_t = time.perf_counter()
    loop_start_t = start_t

    for i in range(n_cycles):
        # stochastically train the network on randomly sampled trial batches
        if batch_size is None:
            batch_idxs = np.arange(n_sess)
        else:
            batch_idxs = rand.sample(range(n_sess), batch_size)
            
        batch_inputs = inputs[batch_idxs, :, :]
        batch_labels = labels[batch_idxs, :, :]

        optimizer.zero_grad() # zero the gradient buffers
        output, _ = model(batch_inputs)
        
        # format the output and labels to work with the loss function
        if not output_formatter is None:
            orig_shape = batch_labels.shape
            output = output_formatter(output)
            batch_labels = output_formatter(batch_labels)
            
        err = loss(output, batch_labels) # compute the loss
        
        # revert the formatting of the error to match the original output shape
        if not output_formatter is None:
            err = torch.reshape(err, orig_shape)
        
        # mask the loss to ignore padded elements
        err = err * trial_mask
        
        # manually normalize the errors by the different trial lengths
        if equal_sess_weight:
            err = (err.sum(dim=1)/trial_mask.sum(dim=1)).mean()
        else:
            err = err.sum()/trial_mask.sum()
        
        if torch.isnan(err).any():
            raise RuntimeError('Error was NaN')
        
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
                
            if print_params:
                print(model.print_params())
                
            running_loss = 0
            
            if not loss_diff_exit_thresh is None:
                # check if we should end early
                loss_hist_max = max(loss_arr[i-eval_interval+1:i+1])
                loss_hist_min = min(loss_arr[i-eval_interval+1:i+1])
                if (loss_hist_max - loss_hist_min) < loss_diff_exit_thresh:
                    break

    return loss_arr

# %% Evaluation Methods 

def eval_model(model, inputs, labels, trial_mask=None, output_transform=None):
    ''' run model to simulate outputs and evaluate model performance '''
    model.eval()
    
    if trial_mask is None:
        trial_mask = torch.zeros_like(labels)+1
    trial_mask = trial_mask.numpy()
    
    output, agent_states = run_model(model, inputs, output_transform=output_transform)
    labels = labels.detach().numpy()
    
    return output, agent_states, calc_model_performance(model, output, labels, trial_mask)


def calc_model_performance(model, output, labels, trial_mask=None):
    if trial_mask is None:
        trial_mask = np.zeros_like(labels)+1
    
    ll_tot, ll_avg = log_likelihood(labels, output, trial_mask)
    acc = accuracy(labels, output, trial_mask)
    n_params = count_params(model)
    n_trials = np.sum(trial_mask)
    bic = calc_bic(ll_tot, n_params, n_trials)
    
    return {'ll_total': ll_tot, 'll_avg': ll_avg, 'norm_llh': np.exp(ll_avg), 
            'acc': acc, 'bic': bic, 'n_params': n_params, 'n_trials': n_trials}
    

def run_model(model, inputs, output_transform=None):
    with torch.no_grad():
        output, agent_states = model(inputs)
        
        if not output_transform is None:
            output = output_transform(output)
            
    return output.detach().numpy(), agent_states.detach().numpy()
    

def log_likelihood(labels, outputs, trial_mask=None):
    
    if trial_mask is None:
        trial_mask = np.zeros_like(labels)+1
        
    # calculate log likelihood for binary classification
    # labels should be 0 or 1 and outputs should span [0,1]
    ll = labels*np.log(outputs) + (1-labels)*np.log(1-outputs)
    ll = ll * trial_mask
    
    ll_tot = np.sum(ll)
    
    return ll_tot, ll_tot/np.sum(trial_mask)
    

def accuracy(labels, outputs, trial_mask=None):
    
    if trial_mask is None:
        trial_mask = np.zeros_like(labels)+1
        
    # calculate choice accuracy for binary classification
    # labels should be 0 or 1 and outputs should span [0,1]
    output_choice = outputs.copy()
    output_choice[output_choice >= 0.5] = 1
    output_choice[output_choice < 0.5] = 0
    
    n_tot = np.sum(trial_mask)
    n_incorrect = np.sum(np.abs(labels[trial_mask.astype(bool)] - output_choice[trial_mask.astype(bool)]))
    
    return (n_tot - n_incorrect)/n_tot


def calc_bic(ll, n_params, n_trials):
    return n_params*np.log(n_trials) - 2*ll
    

def print_params(model):
    # print the parameters of a network
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('{}: {}'.format(name, param.data))
            
def count_params(model):
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    return total_params
    
# %% Plotting Methods

def plot_single_val_fit_results(sess_data, output, agent_activity, agent_names, betas=None, title_prefix=''):
    # common plotting method to plot output of fits
    
    # convert to numpy for plotting
    if type(output) is torch.Tensor:
        output = output.detach().numpy()
        
    if type(agent_activity) is torch.Tensor:
        agent_activity = agent_activity.detach().numpy()
    
    sess_ids = sess_data['sessid'].unique()
    
    for i, sess_id in enumerate(sess_ids):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        fig, axs = plt.subplots(2, 1, figsize=(12,6), layout='constrained')
            
        # get block transitions
        block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
        block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
        block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
            
        fig.suptitle('{}Session {}'.format(title_prefix, sess_id))

        # label trials from 1 to the last trial
        x = np.arange(len(trial_data)-1)+1
        
        ax = axs[0]
        #start plotting from trial 1 to the last trial
        ax.plot(x, output[i,:len(trial_data)-1,:], label='Model Output')
        ax.set_ylabel('p(Choose Left)')
        ax.set_xlabel('Trial')
        ax.set_title('Model Output vs Choices',  fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.axhline(y=0.5, color='black', linestyle='dashed')
        ax.margins(x=0.01)
        
        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)

        #Plot agent states over trials
        ax = axs[1]
        
        if not betas is None:
            if type(betas) is torch.Tensor:
                betas = betas.detach().numpy()
                
            plot_agent_vals = agent_activity*betas[None,None,None,:]
        else:
            plot_agent_vals = agent_activity
        
        for j, agent in enumerate(agent_names):
            agent_vals = plot_agent_vals[i,:len(trial_data)-1,0,j]
            ax.plot(x, agent_vals, alpha=0.7, label=agent)

        ax.set_ylabel('Agent State')
        ax.set_xlabel('Trial')
        ax.set_title('Model Agent States', fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.axhline(y=0, color='black', linestyle='dashed')
        ax.margins(x=0.01)
        
        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)

def plot_multi_val_fit_results(sess_data, output, agent_activity, agent_names, choice_names=['Left', 'Right'], 
                               betas=None, title_prefix='', n_sess=np.inf, trial_mask=None):
    # common plotting method to plot output of model fits maintaining values for more than one choice
    
    # convert to numpy for plotting
    if type(output) is torch.Tensor:
        output = output.detach().numpy()
        
    if type(agent_activity) is torch.Tensor:
        agent_activity = agent_activity.detach().numpy()
        
    if not betas is None and type(betas) is torch.Tensor:
        betas = betas.detach().numpy()
        
    if trial_mask is None:
        trial_mask = np.full_like(output, 1)
    elif type(trial_mask) is torch.Tensor:
        trial_mask = trial_mask.detach().numpy()
        
    n_agents = len(agent_names)
    
    sess_ids = sess_data['sessid'].unique()
    if n_sess < len(sess_ids):
        sess_ids = sess_ids[:n_sess]
    
    for i, sess_id in enumerate(sess_ids):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        fig, axs = plt.subplots(n_agents+1, 1, figsize=(12,(n_agents+1)*3), layout='constrained')
            
        # get block transitions
        block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
        block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
        block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
            
        fig.suptitle('{}Session {}'.format(title_prefix, sess_id))

        x = np.arange(len(trial_data))+1
        
        # plot model output versus actual choices
        ax = axs[0]
        _draw_mask(x[1:], trial_mask[i,:len(trial_data)-1], ax)
        #start plotting output from trial 2 to the last trial
        ax.plot(x[1:], output[i,:len(trial_data)-1,:], label='Model Output')
        ax.set_ylabel('p(Choose {})'.format(choice_names[0]))
        ax.set_xlabel('Trial')
        ax.set_title('Model Output vs Choices')
        ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.axhline(y=0.5, color='black', linestyle='dashed')
        ax.margins(x=0.01)

        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)

        # Plot agent states over trials
        for j, agent in enumerate(agent_names):
            ax = axs[j+1]
            
            _draw_mask(x[1:], trial_mask[i,:len(trial_data)-1], ax)
    
            for k, choice_name in enumerate(choice_names):
                agent_vals = agent_activity[i,:len(trial_data)-1,k,j]
                
                if not betas is None:
                    ax.plot(x[1:], agent_vals*betas[j], alpha=0.7, label='β*{} {}'.format(agent, choice_name))
                else:
                    ax.plot(x[1:], agent_vals, alpha=0.7, label='{} {}'.format(agent, choice_name), color='C{}'.format(k))
    
            ax.set_ylabel('{} Agent State'.format(agent))
            ax.set_xlabel('Trial')
            ax.set_title('{} Agent State'.format(agent))

            ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.axhline(y=0, color='black', linestyle='dashed')
            ax.margins(x=0.01)

            _draw_choices(trial_data, ax)
            _draw_blocks(block_switch_trials, block_rates, ax)
            
def plot_simple_multi_val_fit_results(sess_data, output, agent_activity, agent_names, choice_names=['Left', 'Right'], betas=None, title_prefix='', n_sess=np.inf, use_ratio=False):
    # common plotting method to plot output of model fits maintaining values for more than one choice
    # this method will plot the difference between side values for each agent on one axis
    
    # convert to numpy for plotting
    if type(output) is torch.Tensor:
        output = output.detach().numpy()
        
    if type(agent_activity) is torch.Tensor:
        agent_activity = agent_activity.detach().numpy()
        
    if not betas is None and type(betas) is torch.Tensor:
        betas = betas.detach().numpy()

    # compute the agent value differences over the choices
    agent_diffs = -np.diff(agent_activity, axis=2)[:,:,0,:]
    
    if use_ratio:
        new_names = []
        agent_ratios = []
        first_agent_vals = agent_diffs[:,:,0]
        # compute the ratio of the differences
        for i in range(len(agent_names)-1):
            comp_agent_vals = agent_diffs[:,:,i+1]
            agent_ratios.append(first_agent_vals/comp_agent_vals)
            new_names.append(agent_names[0]+'/'+agent_names[i+1])
        
        agent_names = new_names
        agent_diffs = np.stack(agent_ratios, axis=2)
    
    sess_ids = sess_data['sessid'].unique()
    if n_sess < len(sess_ids):
        sess_ids = sess_ids[:n_sess]
    
    for i, sess_id in enumerate(sess_ids):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        fig, axs = plt.subplots(2, 1, figsize=(12,6), layout='constrained')
            
        # get block transitions
        block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
        block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
        block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
            
        fig.suptitle('{}Session {}'.format(title_prefix, sess_id))

        x = np.arange(len(trial_data))+1
        
        # plot model output versus actual choices
        ax = axs[0]
        #start plotting from trial 1 to the last trial
        ax.plot(x[1:], output[i,:len(trial_data)-1,:], label='Model Output')
        ax.set_ylabel('p(Choose {})'.format(choice_names[0]))
        ax.set_xlabel('Trial')
        ax.set_title('Model Output vs Choices')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.axhline(y=0.5, color='black', linestyle='dashed')
        ax.margins(x=0.01)

        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)

        # Plot agent states over trials
        ax = axs[1]
        for j, agent in enumerate(agent_names):
            agent_vals = agent_diffs[i,:len(trial_data)-1,j]
                
            if not betas is None:
                ax.plot(x[1:], agent_vals*betas[j], alpha=0.7, label=agent)
                if use_ratio:
                    ax.set_ylabel('Agent Value Ratio')
                else:
                    ax.set_ylabel('Weighted State Difference')
            else:
                ax.plot(x[1:], agent_vals, alpha=0.7, label=agent)
                ax.set_ylabel('State Difference')

        ax.set_xlabel('Trial')
        if use_ratio:
            ax.set_title('Agent Value Difference Ratios ({}-{})'.format(choice_names[0], choice_names[1]))
        else:
            ax.set_title('Agent State Differences ({}-{})'.format(choice_names[0], choice_names[1]))

        ax.legend(title='Agents', fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), borderaxespad=0)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.axhline(y=0, color='black', linestyle='dashed')
        ax.margins(x=0.01)

        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)
            
def _draw_choices(trial_data, ax):
    choices = trial_data['choice_inputs']
    rewarded = trial_data['rewarded']
    choice_outcome_lines = np.vstack([np.zeros_like(choices), choices*0.1 + rewarded*choices*0.1])
    
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    # add choice lines
    # Lines will be a multiple of the y range and start at the current y min/max
    choice_outcome_lines[:,choices == -1] = choice_outcome_lines[:,choices == -1]*y_range + y_min
    choice_outcome_lines[:,choices == 1] = choice_outcome_lines[:,choices == 1]*y_range + y_max
    
    # draw choices starting at 1
    ax.vlines(x=np.arange(len(trial_data))+1, ymin=choice_outcome_lines[0,:], ymax=choice_outcome_lines[1,:], 
              color='gray', label='Choices')
    
    # add labels
    min_label_y = np.max(choice_outcome_lines[1,choices == -1])
    max_label_y = np.min(choice_outcome_lines[1,choices == 1])
    
    current_ticks = ax.get_yticks()
    # need to keep any exponential since the formatting gets whacky with a mix of strings and numbers
    current_labels = np.array(['{:.2}'.format(tick) for tick in current_ticks])
    # keep ticks that are between the base of the ticks
    keep_tick_sel = (current_ticks < y_max) & (current_ticks > y_min)
    new_ticks = [min_label_y] + current_ticks[keep_tick_sel].tolist() + [max_label_y]
    new_labels = ['Chose Right'] + current_labels[keep_tick_sel].tolist() + ['Chose Left']
    
    ax.set_yticks(new_ticks, labels=new_labels)

def _draw_blocks(block_switch_trials, block_rates, ax):
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    
    plot_utils.plot_dashlines(block_switch_trials[:-1], ax=ax, label='_', c='black')    
    block_switch_mids = np.diff(block_switch_trials)/2 + block_switch_trials[:-1]
    
    for idx, rate in zip(block_switch_mids, block_rates):
        ax.text(idx, y_max, rate, horizontalalignment='center', fontsize=8)
        
    ax.set_ylim(y_min, y_max + y_range*0.06)
    
def _draw_mask(x, trial_mask, ax):
    in_zero = False
    start = None
    for i, m in zip(x, trial_mask):
        if m == 0 and not in_zero:
            start = i
            in_zero = True
        elif m == 1 and in_zero:
            ax.axvspan(start, i, color='gray', alpha=0.3, zorder=0)
            in_zero = False
    
    # If ends with a 0 region, close it
    if in_zero:
        ax.axvspan(start, x[-1], color='gray', alpha=0.3, zorder=0)