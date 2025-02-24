# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:49:32 2024

@author: tanne
"""

import random as rand
import numpy as np
import pandas as pd
import torch

def simulate_behavior(model, n_trials, n_sess, input_formatter, output_choice, output_transform=None, **kwargs):
    block_gen_method = kwargs.get('block_gen_method', 'n_high_choice') # 'n_high_choice', 'const_switch_p', 'block_switch_p'
    
    match block_gen_method:
        case 'n_high_choice':
            p_reward = kwargs.get('p_reward', [0.1, 0.4, 0.7])
            high_choice_range = kwargs.get('high_choice_range', [10, 25])
            p_switch = np.nan
        case 'block_n_high_choice':
            p_reward = kwargs.get('p_reward', [0.2, 0.7])
            block_switch_p = kwargs.get('block_switch_p', 0.2)
            min_consec_blocks = kwargs.get('min_consec_blocks', 3)
            high_vol_choice_range = kwargs.get('high_vol_choice_range', [4, 10])
            low_vol_choice_range = kwargs.get('low_vol_choice_range', [15, 40])
            p_switch = np.nan
            block_vols = np.array(['low', 'high'])
        case 'const_switch_p':
            p_reward = kwargs.get('p_reward', [0.2, 0.7])
            p_switch = kwargs.get('p_switch', 0.1)
            block_lims = kwargs.get('block_lims', [4, 40])
        case 'var_switch_p':
            p_reward = kwargs.get('p_reward', [0.2, 0.7])
            p_max = kwargs.get('p_max', 0.2)
            p_min = kwargs.get('p_min', 0.03)
            p_drift = kwargs.get('p_drift', (p_max - p_min)/10)
            lam_smooth = kwargs.get('lam_smooth', 0.6)
            bound_type = kwargs.get('bound_type', 'sticky') # 'sticky', 'reflective'
            block_lims = kwargs.get('block_lims', [4, 40])
        case 'block_switch_p':
            p_reward = kwargs.get('p_reward', [0.2, 0.7])
            p_high = kwargs.get('p_max', 0.15)
            p_low = kwargs.get('p_min', 0.03)
            block_switch_p = kwargs.get('block_switch_p', 0.2)
            block_lims = kwargs.get('block_lims', [4, 40])
            min_consec_blocks = kwargs.get('min_consec_blocks', 3)

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
                p_switch = rand.choice([p_low, p_high])
                consec_blocks = 0
            case 'block_n_high_choice':
                consec_blocks = 0
                block_vol = rand.choice(block_vols)

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
            
            if prev_choice == 'left':
                prev_reward = rand.random() < p_reward_left
            else:
                prev_reward = rand.random() < p_reward_right
                
            chose_high = prev_choice == high_port
                
            # Save trial data
            trial_data.append({'sessid': 'Sim {}'.format(i), 'trial': j+1, 'block_num': block, 'block_trial': block_trial, 'hit': True, 
                               'p_reward_right': p_reward_right, 'p_reward_left': p_reward_left, 'high_side': high_port,
                               'choice': prev_choice, 'rewarded': prev_reward, 'chose_high': chose_high, 'chose_left': prev_choice == 'left',
                               'chose_right': prev_choice == 'right', 'side_prob': '{:.0f}/{:.0f}'.format(p_reward_left*100, p_reward_right*100),
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
                        new_block = (rand.random() < p_switch and block_high_choices >= min(block_lims)) or block_high_choices == max(block_lims)
                case 'var_switch_p':
                    if chose_high:
                        new_block = (rand.random() < p_switch and block_high_choices >= min(block_lims)) or block_high_choices == max(block_lims)
                        
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
                        new_block = (rand.random() < p_switch and block_high_choices >= min(block_lims)) or block_high_choices == max(block_lims)
                        
                        if new_block:
                            # if we need to change the switching regime
                            if rand.random() < block_switch_p and consec_blocks >= min_consec_blocks:
                                ps = np.array([p_low, p_high])
                                p_switch = ps[ps != p_switch][0]
                                consec_blocks = 0
                            else:
                                consec_blocks += 1
                case 'block_n_high_choice':
                    if block_high_choices == block_high_choices_max:
                        new_block = True
                        
                    if new_block:
                        # if we need to change the switching regime
                        if rand.random() < block_switch_p and consec_blocks >= min_consec_blocks:
                            block_vol = block_vols[block_vols != block_vol]
                            consec_blocks = 0
                        else:
                            consec_blocks += 1
                
            block_trial += 1
            
        model_output.append(np.stack(sess_model_output, axis=1))
        agent_states.append(np.stack(sess_agent_states, axis=1))
        
    return pd.DataFrame(trial_data), {'model_output': np.concatenate(model_output, axis=0)[:,:,0,:],
                                      'agent_states': np.concatenate(agent_states, axis=0)}
    
    
    
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