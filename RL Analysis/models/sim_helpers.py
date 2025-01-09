# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:49:32 2024

@author: tanne
"""

import random as rand
import numpy as np
import pandas as pd
import torch

def simulate_behavior(model, n_trials, input_formatter, output_choice, options={}):
    p_reward = options.get('p_reward', [0.2, 0.5, 0.8])
    high_choice_range = options.get('high_choice_range', [15, 25])
    new_block = True
    trial_data = [] # list of dicts
    sides = np.array(['left', 'right'])
    block = 0
    trial = 1
    prev_reward = None
    prev_choice = None

    model_output = []
    agent_states = []
    agent_input_state_diffs = []
    agent_state_deltas = []
    
    while trial <= n_trials:
    
        if new_block:
            new_block = False
            block += 1
            block_high_choices_max = rand.randint(high_choice_range[0], high_choice_range[1])
            block_high_choices = 0
            block_trial = 1
            
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
        inputs = torch.tensor(input_formatter(prev_choice, prev_reward)).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            output, output_data = model.step(inputs)
        
        prev_choice = output_choice(output)
        
        if prev_choice == 'left':
            prev_reward = rand.random() < p_reward_left
        else:
            prev_reward = rand.random() < p_reward_right
            
        chose_high = prev_choice == high_port
            
        # Save trial data
        trial_data.append({'trial': trial, 'block_num': block, 'block_trial': block_trial, 'hit': True, 
                           'p_reward_right': p_reward_right, 'p_reward_left': p_reward_left, 'high_side': high_port,
                           'choice': prev_choice, 'rewarded': prev_reward, 'chose_high': chose_high, 
                           'side_prob': '{:.0f}/{:.0f}'.format(p_reward_left*100, p_reward_right*100)})
    
        model_output.append(output.detach().numpy())
        agent_states.append(output_data['agent_states'])
        agent_input_state_diffs.append(output_data['agent_input_state_diffs'])
        agent_state_deltas.append(output_data['agent_state_deltas'])
        
        # prepare next trial
        if chose_high:
            block_high_choices += 1
        
        if block_high_choices == block_high_choices_max:
            new_block = True
            
        trial += 1
        block_trial += 1
        
    return pd.DataFrame(trial_data), {'model_output': np.stack(model_output, axis=1),
                                      'agent_states': np.stack(agent_states, axis=1), 
                                      'agent_input_state_diffs': np.stack(agent_input_state_diffs, axis=1), 
                                      'agent_state_deltas': np.stack(agent_state_deltas, axis=1)}
    
    
    
    
    