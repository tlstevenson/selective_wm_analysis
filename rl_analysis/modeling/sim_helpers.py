# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:49:32 2024

@author: tanne
"""

import random as rand
import numpy as np
import pandas as pd
import torch
import scipy.stats as sts
from pyutils import utils
import matplotlib.pyplot as plt

def simulate_behavior(model, n_trials, n_sess, input_formatter, output_choice, output_transform=None, access_data=None, **kwargs):
    block_gen_method = kwargs.get('block_gen_method', 'n_high_choice') # 'n_high_choice', 'const_switch_p', 'block_switch_p'
    
    match block_gen_method:
        case 'n_high_choice':
            p_reward = kwargs.get('p_reward', [0.1, 0.45, 0.75])
            high_choice_range = kwargs.get('high_choice_range', [15, 25])
            p_switch = np.nan
        case 'block_n_high_choice':
            p_reward = kwargs.get('p_reward', [0.1, 0.75])
            block_switch_p = kwargs.get('block_switch_p', 0.2)
            min_consec_blocks = kwargs.get('min_consec_blocks', 4)
            high_vol_choice_range = kwargs.get('high_vol_choice_range', [5, 10])
            low_vol_choice_range = kwargs.get('low_vol_choice_range', [20, 30])
            p_switch = np.nan
            block_vols = np.array(['low', 'high'])
        case 'block_switch_rate':
            p_reward_high = kwargs.get('p_reward_high', [0.1, 0.75])
            p_reward_low = kwargs.get('p_reward_low', [0.1, 0.45])
            high_choice_range = kwargs.get('high_choice_range', [15, 25])
            block_switch_p = kwargs.get('block_switch_p', 0.2)
            min_consec_blocks = kwargs.get('min_consec_blocks', 4)
            block_rates = np.array(['low', 'high'])
            p_switch = np.nan
        case 'const_switch_p':
            p_reward = kwargs.get('p_reward', [0.1, 0.7])
            p_switch = kwargs.get('p_switch', 0.1)
            block_lims = kwargs.get('block_lims', [4, 40])
        case 'var_switch_p':
            p_reward = kwargs.get('p_reward', [0.1, 0.7])
            p_max = kwargs.get('p_max', 0.2)
            p_min = kwargs.get('p_min', 0.03)
            p_drift = kwargs.get('p_drift', (p_max - p_min)/10)
            lam_smooth = kwargs.get('lam_smooth', 0.6)
            bound_type = kwargs.get('bound_type', 'sticky') # 'sticky', 'reflective'
            block_lims = kwargs.get('block_lims', [4, 40])
        case 'block_switch_p':
            p_reward = kwargs.get('p_reward', [0.1, 0.7])
            p_high = kwargs.get('p_max', 0.15)
            p_low = kwargs.get('p_min', 0.03)
            block_switch_p = kwargs.get('block_switch_p', 0.2)
            block_lims = kwargs.get('block_lims', [4, 40])
            min_consec_blocks = kwargs.get('min_consec_blocks', 3)
            block_vols = np.array(['low', 'high'])
        

    trial_data = [] # list of dicts
    sides = np.array(['left', 'right'])
    
    model_output = []
    agent_states = []
    
    model.eval()
    
    for i in range(n_sess):
        
        model.reset_state()
        sess_model_output = []
        sess_agent_states = []
        prev_choice = None
        prev_reward = None
        new_block = True
        block = 0

        match block_gen_method:
            case 'var_switch_p':
                p_switch = np.random.uniform(p_min, p_max)
                smoothed_noise = 0
            case 'block_switch_p':
                block_vol = rand.choice(block_vols)
                p_switch = p_low if block_vol == 'low' else p_high
                consec_blocks = 0
            case 'block_n_high_choice':
                consec_blocks = 0
                block_vol = rand.choice(block_vols)
            case 'block_switch_rate':
                consec_blocks = 0
                block_rate = rand.choice(block_rates)

        for j in range(n_trials):
        
            if new_block:
                new_block = False
                block += 1
                block_trial = 1
                block_high_choices = 0
                
                match block_gen_method:
                    case 'n_high_choice':
                        block_high_choices_max = rand.randint(high_choice_range[0], high_choice_range[1])
                    case 'block_n_high_choice':
                        if block_vol == 'low':
                            high_choice_range = low_vol_choice_range
                        else:
                            high_choice_range = high_vol_choice_range
                            
                        block_high_choices_max = rand.randint(high_choice_range[0], high_choice_range[1])
                    case 'block_switch_rate':
                        block_high_choices_max = rand.randint(high_choice_range[0], high_choice_range[1])
                        p_reward = p_reward_low if block_rate == 'low' else p_reward_high

                # pick high port and port probabilities
                if block == 1:
                    high_port = rand.choice(sides)
                else:
                    high_port = sides[sides != high_port][0]

                reward_rate = rand.sample(p_reward, k=2)
                
                if high_port == 'left':
                   p_reward_right = min(reward_rate);
                   p_reward_left = max(reward_rate);
                else:
                   p_reward_right = max(reward_rate);
                   p_reward_left = min(reward_rate);
            
            # simulate choice and outcome
            # only step the model after the first trial
            if j > 0:
                inputs = torch.tensor(input_formatter(prev_choice, prev_reward)).unsqueeze(0).float()
                
                with torch.no_grad():
                    output, out_agent_states = model.step(inputs)
                    
                    if not output_transform is None:
                        output = output_transform(output)
                
                prev_choice = output_choice(output)
                
                sess_model_output.append(output.detach().numpy())
                sess_agent_states.append(out_agent_states.numpy())
            else:
                # randomly choose first choice
                prev_choice = rand.choice(sides)
            
            choice_prob = p_reward_left if prev_choice == 'left' else p_reward_right
            prev_reward = rand.random() < choice_prob
            
            chose_high = prev_choice == high_port
            block_prob = '{:.0f}/{:.0f}'.format(max(p_reward_left, p_reward_right)*100, min(p_reward_left, p_reward_right)*100)
            
            if block_gen_method in ['block_switch_p', 'block_n_high_choice']:
                block_prob = block_prob + '-{}'.format(block_vol)
            
            # Save trial data
            trial_data.append({'sessid': 'Sim {}'.format(i), 'trial': j+1, 'block_num': block, 'block_trial': block_trial, 'hit': True, 
                               'p_reward_right': p_reward_right, 'p_reward_left': p_reward_left, 'high_side': high_port,
                               'choice': prev_choice, 'rewarded': prev_reward, 'chose_high': chose_high, 'chose_left': prev_choice == 'left',
                               'chose_right': prev_choice == 'right', 'side_prob': '{:.0f}/{:.0f}'.format(p_reward_left*100, p_reward_right*100),
                               'block_prob': block_prob, 'choice_prob': choice_prob, 'choice_block_prob': '{:.0f} ({})'.format(choice_prob*100, block_prob),
                               'p_switch': p_switch})
            
            # prepare next trial
            if chose_high:
                block_high_choices += 1
                
            match block_gen_method:
                case 'n_high_choice':
                    if block_high_choices == block_high_choices_max:
                        new_block = True
                        
                case 'const_switch_p':
                    if chose_high:
                        new_block = (rand.random() < p_switch and block_high_choices >= min(block_lims)) or block_high_choices >= max(block_lims)
                        
                case 'var_switch_p':
                    if chose_high:
                        new_block = (rand.random() < p_switch and block_high_choices >= min(block_lims)) or block_high_choices >= max(block_lims)
                        
                        p_noise = np.random.normal(0, p_drift)
                        smoothed_noise = lam_smooth*smoothed_noise + (1-lam_smooth)*p_noise
                        
                        new_p = p_switch + smoothed_noise
                        match bound_type:
                            case 'sticky':
                                if new_p < p_min:
                                    new_p = p_min
                                elif new_p > p_max:
                                    new_p = p_max
                            case 'reflective':
                                # Reflective boundary handling
                                if new_p < p_min:
                                    new_p = p_min - (new_p - p_min)
                                elif new_p > p_max:
                                    new_p = p_max - (new_p - p_max)
                            
                        p_switch = new_p
                        
                case 'block_switch_p':
                    if chose_high:
                        new_block = (rand.random() < p_switch and block_high_choices >= min(block_lims)) or block_high_choices >= max(block_lims)
                        
                        if new_block:
                            # if we need to change the switching regime
                            if rand.random() < block_switch_p and consec_blocks >= min_consec_blocks:
                                block_vol = block_vols[block_vols != block_vol][0]
                                p_switch = p_low if block_vol == 'low' else p_high
                                consec_blocks = 0
                            else:
                                consec_blocks += 1
                                
                case 'block_n_high_choice':
                    if block_high_choices == block_high_choices_max:
                        new_block = True
                        
                    if new_block:
                        # if we need to change the switching regime
                        if rand.random() < block_switch_p and consec_blocks >= min_consec_blocks:
                            block_vol = block_vols[block_vols != block_vol][0]
                            consec_blocks = 0
                        else:
                            consec_blocks += 1
                            
                case 'block_switch_rate':
                    if block_high_choices == block_high_choices_max:
                        new_block = True
                        
                    if new_block:
                        # if we need to change the switching regime
                        if rand.random() < block_switch_p and consec_blocks >= min_consec_blocks:
                            block_rate = block_rates[block_rates != block_rate][0]
                            consec_blocks = 0
                        else:
                            consec_blocks += 1
                
            block_trial += 1
            
        model_output.append(np.stack(sess_model_output, axis=1))
        agent_states.append(np.stack(sess_agent_states, axis=1))
        
        if not access_data is None:
            access_data(model, i)
        
    return pd.DataFrame(trial_data), {'model_output': np.concatenate(model_output, axis=0)[:,:,0,:],
                                      'agent_states': np.concatenate(agent_states, axis=0)}
    
    
# %%

def simulate_var_rew_behavior(model, n_trials, n_sess, input_formatter, output_choice, output_transform=None, access_data=None, **kwargs):
    rew_mean_high = kwargs.get('rew_mean_high', 24)
    rew_mean_low = kwargs.get('rew_mean_low', 8)
    rew_var_high = kwargs.get('rew_var_high', 8)
    rew_var_low = kwargs.get('rew_var_low', 4)
    high_vol_choice_range = kwargs.get('high_vol_choice_range', [3, 8])
    low_vol_choice_range = kwargs.get('low_vol_choice_range', [20, 30])
    high_choice_epoch_range = kwargs.get('high_choice_epoch_range', [100, 200])
    use_double_epoch = kwargs.get('use_double_epoch', False)
    disc_step = kwargs.get('disc_rew_diff', 4)
    
    trial_data = [] # list of dicts
    sides = np.array(['left', 'right'])
    
    model_output = []
    agent_states = []
    
    model.eval()
    rng = np.random.default_rng()
    
    for i in range(n_sess):
        
        model.reset_state()
        sess_model_output = []
        sess_agent_states = []
        prev_choice = None
        prev_reward = None
        new_block = True
        block = 0
        epoch_high_choices = 0
        
        block_var, block_vol = _get_new_block_var_vol()
        max_epoch_high_choice = rand.randint(high_choice_epoch_range[0], high_choice_epoch_range[1])
        
        prev_choice = pd.NA
        prev_reward = pd.NA
        prev_high_port = pd.NA

        for j in range(n_trials):
        
            if new_block:
                if epoch_high_choices >= max_epoch_high_choice:
                    block = 0
                    epoch_high_choices = 0
                    
                    block_var, block_vol = _get_new_block_var_vol(block_var, block_vol)
                    max_epoch_high_choice = rand.randint(high_choice_epoch_range[0], high_choice_epoch_range[1])
                    model.reset_state()
                    
                    prev_choice = pd.NA
                    prev_reward = pd.NA
                    prev_high_port = pd.NA
                
                new_block = False
                block += 1
                block_trial = 1
                block_high_choices = 0
                
                if block_vol == 'low':
                    high_choice_range = low_vol_choice_range
                else:
                    high_choice_range = high_vol_choice_range
                            
                block_high_choices_max = rand.randint(high_choice_range[0], high_choice_range[1])
                
                if block_var == 'low':
                    rew_var = rew_var_low
                else:
                    rew_var = rew_var_high

                # pick high port and port probabilities
                if block == 1:
                    high_port = rand.choice(sides)
                else:
                    high_port = sides[sides != high_port][0]

                if high_port == 'left':
                   rew_mean_right = rew_mean_low;
                   rew_mean_left = rew_mean_high;
                else:
                   rew_mean_right = rew_mean_high;
                   rew_mean_left = rew_mean_low;
            
            # simulate choice and outcome
            # only step the model after the first trial
            if not pd.isna(prev_choice):
                inputs = torch.tensor(input_formatter(prev_choice, prev_reward)).unsqueeze(0).float()
                
                with torch.no_grad():
                    output, out_agent_states = model.step(inputs)
                    
                    if not output_transform is None:
                        output = output_transform(output)
                
                new_choice = output_choice(output)
                
                sess_model_output.append(output.detach().numpy())
                sess_agent_states.append(out_agent_states.numpy())
            else:
                # randomly choose first choice
                new_choice = rand.choice(sides)
                
                sess_model_output.append(np.full([1,2,1], np.nan))
                sess_agent_states.append(np.full([1,2,len(model.agents)], np.nan))
            
            choice_rew_mean = rew_mean_left if new_choice == 'left' else rew_mean_right
            
            chose_high = new_choice == high_port
            
            if chose_high:
                choice_rew_var = rew_var
            else:
                choice_rew_var = rew_var_low
                
            rew_vals = np.concatenate((np.flip(np.arange(choice_rew_mean, choice_rew_mean-4*choice_rew_var-disc_step, -disc_step)), 
                                       np.arange(choice_rew_mean+disc_step, choice_rew_mean+4*choice_rew_var+disc_step, disc_step)))
            rew_probs = sts.norm.pdf(rew_vals, loc=choice_rew_mean, scale=choice_rew_var)
            rew_probs = rew_probs/np.sum(rew_probs)
                
            new_reward = max(rng.choice(rew_vals, 1, p=rew_probs)[0], 0)
            
            if pd.isna(prev_high_port):
                chose_prev_high = pd.NA
            else:
                chose_prev_high = new_choice == prev_high_port
                
            block_prob = '{} var/{} vol'.format(block_var, block_vol)
            simplify_label = lambda x: x.replace('low', 'L').replace('high', 'H')
            block_label = '{}/{}'.format(simplify_label(block_var), simplify_label(block_vol))
            
            # Save trial data
            trial_data.append({'sessid': 'Sim {}'.format(i), 'trial': j+1, 'block_num': block, 'block_trial': block_trial, 'hit': True, 
                               'mean_reward_right': rew_mean_right, 'mean_reward_left': rew_mean_left, 'high_side': high_port,
                               'choice': new_choice, 'reward': new_reward, 'rewarded': new_reward >= np.mean([rew_mean_high, rew_mean_low]), 
                               'chose_high': chose_high, 'chose_left': new_choice == 'left', 'chose_right': new_choice == 'right', 
                               'side_prob': '{:.0f}/{:.0f}, {}'.format(rew_mean_left, rew_mean_right, block_var),
                               'block_prob': block_prob, 'choice_prob': choice_rew_mean, 'choice_block_prob': '{:.0f} ({})'.format(choice_rew_mean, block_prob),
                               'side_means': '{:.0f}/{:.0f}'.format(rew_mean_left, rew_mean_right), 
                               'epoch_block_label': '{}-{:.0f}/{:.0f}'.format(block_label, max(rew_mean_left, rew_mean_right), min(rew_mean_left, rew_mean_right)),
                               'epoch_side_label': '{}-{:.0f}/{:.0f}'.format(block_label, rew_mean_left, rew_mean_right),
                               'prev_high_side': prev_high_port, 'chose_prev_high': chose_prev_high, 'next_switch': pd.NA})
            
            # ammend previous trial's next switch
            if j > 0:
                trial_data[i*n_trials + j-1]['next_switch'] = prev_choice != new_choice
                
            prev_high_port = high_port
            prev_choice = new_choice
            prev_reward = new_reward
            
            # prepare next trial
            if chose_high:
                block_high_choices += 1
                epoch_high_choices += 1

                if block_high_choices == block_high_choices_max:
                    new_block = True
                            
            block_trial += 1
            
        model_output.append(np.stack(sess_model_output, axis=1)[:,1:,:,:])
        agent_states.append(np.stack(sess_agent_states, axis=1)[:,1:,:,:])
        
        if not access_data is None:
            access_data(model, i)
        
    return pd.DataFrame(trial_data), {'model_output': np.concatenate(model_output, axis=0)[:,:,0,:],
                                      'agent_states': np.concatenate(agent_states, axis=0)}
   
def _get_new_block_var_vol(prev_var=None, prev_vol=None, no_high_high=True):
    block_type_labels = ['low', 'high']
    current_labels = np.array([prev_var, prev_vol])
    poss_combs = utils.get_all_combinations(block_type_labels, block_type_labels)
    
    # remove current option
    poss_combs = poss_combs[~np.all(poss_combs == current_labels, axis=1)]
    
    # remove option where both change
    #poss_combs = poss_combs[~np.all(poss_combs == np.flip(current_labels), axis=1)]
    
    if no_high_high:
        poss_combs = poss_combs[~np.all(poss_combs == np.array(['high', 'high']), axis=1)]
    
    new_val_row = np.random.choice(poss_combs.shape[0], 1)[0]
    new_vals = poss_combs[new_val_row]
        
    return new_vals[0], new_vals[1]
    

# # %%
# d=6
# h=18
# l=6
# s_h=4
# s_l=4
# n_s=2

# x_h = np.concatenate((np.flip(np.arange(h, h-n_s*s_h-d, -d)), np.arange(h+d, h+n_s*s_h+d, d))) 
# x_l = np.concatenate((np.flip(np.arange(l, l-n_s*s_l-d, -d)), np.arange(l+d, l+n_s*s_l+d, d))) 
# plt.bar(x_h, sts.norm.pdf(x_h, loc=h, scale=s_h), alpha=0.5) 
# plt.bar(x_l, sts.norm.pdf(x_l, loc=l, scale=s_l), alpha=0.5)

# plt.title('μ high: {}, μ low: {}, σ high: {}, σ low: {}, step: {}'.format(h, l, s_h, s_l, d))


# %%
# d=5
# h=20
# l=5
# n_l=7
# n_h=9
# s_h=11
# s_l=9

# x_h = np.concatenate((np.flip(np.arange(h, h-np.ceil(n_h/2)*d, -d)), np.arange(h+d, h+np.ceil(n_h/2)*d, d))) 
# x_l = np.concatenate((np.flip(np.arange(l, l-np.ceil(n_l/2)*d, -d)), np.arange(l+d, l+np.ceil(n_l/2)*d, d))) 

# x_h = x_h[x_h >= 0]
# x_l = x_l[x_l >= 0]

# plt.bar(x_h, sts.norm.pdf(x_h, loc=h, scale=s_h), alpha=0.5) 
# plt.bar(x_l, sts.norm.pdf(x_l, loc=l, scale=s_l), alpha=0.5)

# plt.title('μ high: {}, μ low: {}, σ high: {}, σ low: {}, n high: {}, n low: {}, step: {}'.format(h, l, s_h, s_l, n_h, n_l, d))


# # %%
# r_h = [3,8]
# r_l = [20,30]
# s_mu = 2
# s_sig = 2

# def sample_block_length(block_range, size=100000):

#     n = np.random.randint(block_range[0], block_range[1]+1, size)
#     var = np.random.normal(s_mu, s_sig, size)
#     return n + np.maximum(var, 0)

# low_vol = sample_block_length(r_l)
# high_vol = sample_block_length(r_h)
# all_vol = np.concatenate([low_vol,high_vol])

# min_len = np.min(all_vol)
# max_len = np.max(all_vol)
# bins = np.arange(min_len, max_len+1, 1)
# hist_args = dict(histtype='bar', density=True, cumulative=False, bins=bins, alpha=0.5)

# plt.hist(low_vol, **hist_args, label='Low Vol')
# plt.hist(high_vol, **hist_args, label='High Vol')
    
# plt.xlabel('Block Length (trials)')
# plt.ylabel('Proportion')
# plt.legend()


# %%
# import matplotlib.pyplot as plt

# def random_walk(x_min, x_max, sig_drift, lam_smooth, n_steps, boundary):
#     lam_smooth = np.array(lam_smooth)
#     x = np.zeros((n_steps, len(lam_smooth)))
#     x[0,:] = np.random.uniform(x_min, x_max)
#     cum_noise = 0
#     for i in range(n_steps-1):
#         noise = np.random.normal(0, sig_drift)
#         cum_noise = lam_smooth*cum_noise + (1-lam_smooth)*noise
        
#         x_new = x[i,:] + cum_noise
        
#         match boundary:
#             case 'sticky':
#                 x_new[x_new < x_min] = x_min
#                 x_new[x_new > x_max] = x_max
#             case 'reflective':
#                 # Reflective boundary handling
#                 x_new[x_new < x_min] = x_min - (x_new[x_new < x_min] - x_min)
#                 x_new[x_new > x_max] = x_max - (x_new[x_new > x_max] - x_max)
#                 # if x_new < 0:
#                 #     x_new = -x_new  # Reflect back into range
#                 # elif x_new > x_max:
#                 #     x_new = x_max - (x_new - x_max)  # Reflect back into range
            
#         x[i+1,:] = x_new
        
#     return x
    
# x = random_walk(0.03, 0.15, 0.01, [0, 0.2, 0.4, 0.6, 0.8], 300, 'sticky')
# plt.plot(x)