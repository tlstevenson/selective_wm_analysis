# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 13:49:32 2024

@author: tanne
"""

import random as rand
import numpy as np
import pandas as pd
import torch

def simulate_behavior(model, n_trials, n_sess, input_formatter, output_choice, options={}):
    p_reward = options.get('p_reward', [0.2, 0.5, 0.8])
    high_choice_range = options.get('high_choice_range', [10, 25])
    new_block = True
    trial_data = [] # list of dicts
    sides = np.array(['left', 'right'])
    
    model_output = []
    agent_states = []
    agent_input_state_diffs = []
    agent_state_deltas = []
    
    for i in range(n_sess):
        
        sess_model_output = []
        sess_agent_states = []
        sess_agent_input_state_diffs = []
        sess_agent_state_deltas = []
        
        block = 0
        prev_reward = None
        prev_choice = None

        for j in range(n_trials):
        
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
            trial_data.append({'sessid': 'Sim {}'.format(i), 'trial': j+1, 'block_num': block, 'block_trial': block_trial, 'hit': True, 
                               'p_reward_right': p_reward_right, 'p_reward_left': p_reward_left, 'high_side': high_port,
                               'choice': prev_choice, 'rewarded': prev_reward, 'chose_high': chose_high, 'chose_left': prev_choice == 'left',
                               'chose_right': prev_choice == 'right', 'side_prob': '{:.0f}/{:.0f}'.format(p_reward_left*100, p_reward_right*100)})
        
            sess_model_output.append(output.detach().numpy())
            sess_agent_states.append(output_data['agent_states'])
            sess_agent_input_state_diffs.append(output_data['agent_input_state_diffs'])
            sess_agent_state_deltas.append(output_data['agent_state_deltas'])
            
            # prepare next trial
            if chose_high:
                block_high_choices += 1
            
            if block_high_choices == block_high_choices_max:
                new_block = True
                
            block_trial += 1
            
        model_output.append(np.stack(sess_model_output, axis=1))
        agent_states.append(np.stack(sess_agent_states, axis=1))
        agent_input_state_diffs.append(np.stack(sess_agent_input_state_diffs, axis=1))
        agent_state_deltas.append(np.stack(sess_agent_state_deltas, axis=1))
        
    return pd.DataFrame(trial_data), {'model_output': np.concatenate(model_output, axis=0)[:,:,0,:],
                                      'agent_states': np.concatenate(agent_states, axis=0), 
                                      'agent_input_state_diffs': np.concatenate(agent_input_state_diffs, axis=0), 
                                      'agent_state_deltas': np.concatenate(agent_state_deltas, axis=0)}
    
    
    
    
    