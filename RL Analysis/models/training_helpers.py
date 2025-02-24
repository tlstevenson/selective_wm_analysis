# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 14:02:02 2024

@author: tanne
"""

import torch
import torch.nn as nn
import random as rand
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sys_neuro_tools import plot_utils
import time

def bin_accy_loss_penalty(output, labels, lam=0.2, steepness=50, mid=0.55):
    ''' Calculate a penalty based on binary classifier accuracy.
        The penalty will be lam if the difference between output and label
        is greater than 0.5 and 0 if less than 0.5''' 
    diff = torch.abs(output - labels)
    return lam/(1+torch.exp(-steepness*(diff-mid)))
    

def train_model(model, optimizer, loss, inputs, labels, n_cycles, trial_mask=None, batch_size=None, output_formatter=None, 
                print_time=True, eval_interval=100, loss_diff_exit_thresh=1e-6):
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
        err = (err.sum(dim=1)/trial_mask.sum(dim=1)).mean()
        
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
            running_loss = 0
            
            if not loss_diff_exit_thresh is None:
                # check if we should end early
                loss_hist_max = max(loss_arr[i-eval_interval+1:i+1])
                loss_hist_min = min(loss_arr[i-eval_interval+1:i+1])
                if (loss_hist_max - loss_hist_min) < loss_diff_exit_thresh:
                    break

    return loss_arr


def eval_model(model, inputs, labels, trial_mask=None, output_transform=None):
    ''' run model to simulate outputs and evaluate model performance '''
    model.eval()
    
    if trial_mask is None:
        trial_mask = torch.zeros_like(labels)+1
    trial_mask = trial_mask.numpy()
    
    with torch.no_grad():
        output, agent_states = model(inputs)
        
        if not output_transform is None:
            output = output_transform(output)
            
    output = output.detach().numpy()
    labels = labels.detach().numpy()
        
    # evaluate model
    ll_tot, ll_avg = log_likelihood(labels, output, trial_mask)
    acc = accuracy(labels, output, trial_mask)
    n_params = count_params(model)
    n_trials = np.sum(trial_mask)
    bic = calc_bic(ll_tot, n_params, n_trials)
    
    return output, agent_states.numpy(), {'ll_total': ll_tot, 'll_avg': ll_avg, 'norm_llh': np.exp(ll_avg), 
                                          'acc': acc, 'bic': bic, 'n_params': n_params, 'n_trials': n_trials}
    

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
        if betas is None:
            fig, axs = plt.subplots(2, 1, figsize=(12,6), layout='constrained')
        else:
            fig, axs = plt.subplots(3, 1, figsize=(12,9), layout='constrained')
            
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
        ax.set_title('Model vs True Output across trials',  fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
        ax.axhline(y=0.5, color='black', linestyle='dashed')
        ax.margins(x=0.01)
        
        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)

        #Plot agent states over trials
        ax = axs[1]
        
        for j, agent in enumerate(agent_names):
            agent_vals = agent_activity[i,:len(trial_data)-1,j]
            ax.plot(x, agent_vals, alpha=0.7, label=agent)

        ax.set_ylabel('Agent State')
        ax.set_xlabel('Trial')
        ax.set_title('Model Agent State across trials', fontsize=10)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-4:], labels[-4:], loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.16, 1))
        ax.axhline(y=0, color='black', linestyle='dashed')
        ax.margins(x=0.01)
        
        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)
        
        # plot weighted agent values
        if not betas is None:
            if type(betas) is torch.Tensor:
                betas = betas.detach().numpy()
            
            ax = axs[2]

            for j, agent in enumerate(agent_names):
                agent_vals = agent_activity[i,:len(trial_data)-1,j]*betas[j]
                ax.plot(x, agent_vals, alpha=0.7, label=agent)

            ax.plot(x, output[i,:len(trial_data)-1,:], label='Model Output')

            ax.set_ylabel('Weighted Agent State')
            ax.set_xlabel('Trial')
            ax.set_title('Weighted Model Agent State across trials', fontsize=10)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[-4:], labels[-4:], loc='upper right', fontsize=8,framealpha=0.5, bbox_to_anchor=(1.16, 1))
            ax.axhline(y=0, color='black', linestyle='dashed')
            ax.margins(x=0.01)
            
            _draw_choices(trial_data, ax)
            _draw_blocks(block_switch_trials, block_rates, ax)
            

def plot_multi_val_fit_results(sess_data, output, agent_activity, agent_names, choice_names, betas=None, title_prefix=''):
    # common plotting method to plot output of model fits maintaining values for more than one choice
    
    # convert to numpy for plotting
    if type(output) is torch.Tensor:
        output = output.detach().numpy()
        
    if type(agent_activity) is torch.Tensor:
        agent_activity = agent_activity.detach().numpy()
        
    if not betas is None and type(betas) is torch.Tensor:
        betas = betas.detach().numpy()
        
    n_agents = len(agent_names)
    
    sess_ids = sess_data['sessid'].unique()
    
    for i, sess_id in enumerate(sess_ids):
        trial_data = sess_data[sess_data['sessid'] == sess_id]
        fig, axs = plt.subplots(n_agents+1, 1, figsize=(12,(n_agents+1)*3), layout='constrained')
            
        # get block transitions
        block_switch_trials = trial_data[trial_data['block_trial'] == 1]['trial']
        block_switch_trials = np.append(block_switch_trials, trial_data.iloc[-1]['trial'])
        block_rates = trial_data[trial_data['trial'].isin(block_switch_trials[:-1])]['side_prob']
            
        fig.suptitle('{}Session {}'.format(title_prefix, sess_id))

        x = np.arange(len(trial_data)-1)+1
        
        # plot model output versus actual choices
        ax = axs[0]
        #start plotting from trial 1 to the last trial
        ax.plot(x, output[i,:len(trial_data)-1,:], label='Model Output')
        ax.set_ylabel('p(Choose Left)')
        ax.set_xlabel('Trial')
        ax.set_title('Model vs True Output across trials')
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[-2:], labels[-2:], loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.14, 1.))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        ax.axhline(y=0.5, color='black', linestyle='dashed')
        ax.margins(x=0.01)

        _draw_choices(trial_data, ax)
        _draw_blocks(block_switch_trials, block_rates, ax)

        # Plot agent states over trials
        for j, agent in enumerate(agent_names):
            ax = axs[j+1]
    
            for k, choice_name in enumerate(choice_names):
                agent_vals = agent_activity[i,:len(trial_data)-1,k,j]
                
                if not betas is None:
                    ax.plot(x, agent_vals*betas[j], alpha=0.7, label='β*{} {}'.format(agent, choice_name))
                else:
                    ax.plot(x, agent_vals, alpha=0.7, label='{} {}'.format(agent, choice_name), color='C{}'.format(k))
    
            ax.set_ylabel('{} Agent State'.format(agent))
            ax.set_xlabel('Trial')
            ax.set_title('{} Agent State across trials'.format(agent))

            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[-4:], labels[-4:], loc='upper right', fontsize=8, framealpha=0.5, bbox_to_anchor=(1.16, 1))
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
    
    # draw choices starting at 0 to align the model output with the appropriate line marker
    ax.vlines(x=np.arange(len(trial_data)), ymin=choice_outcome_lines[0,:], ymax=choice_outcome_lines[1,:], 
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
    
    # have block switches be drawn starting at 0
    plot_utils.plot_dashlines(block_switch_trials[:-1]-1, ax=ax, label='_', c='black')    
    block_switch_mids = np.diff(block_switch_trials)/2 + block_switch_trials[:-1]
    
    for idx, rate in zip(block_switch_mids, block_rates):
        ax.text(idx, y_max*1.03, rate, horizontalalignment='center', fontsize=8)
        
    ax.set_ylim(y_min, y_max*1.15)