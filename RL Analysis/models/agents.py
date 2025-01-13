# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:00:09 2024

@author: tanne
"""

# %% General Infrastructure

import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as functional

class UnitConstraint(nn.Module):
    # constrain values to be between 0 and 1
    def forward(self, x):
        return torch.sigmoid(x)
    
    def right_inverse(self, x):
        # convert an initialization between 0 and 1 into the inverse sigmoid space 
        return torch.log(x) - torch.log1p(-x)
    
class PositiveConstraint(nn.Module):
    # constrain values to be positive using softplus
    # having beta be 100 closely approximates the relu function
    def forward(self, x):
        return functional.softplus(x, beta=100)
    
    def right_inverse(self, x):
        # convert an initialization into positive values
        return functional.softplus(x, beta=100)
    
def _init_param(x=None):
    if x is None:
        x = torch.rand(1)
    else:
        # make sure this is a float
        x = torch.tensor(x).type(torch.float)
        
    return nn.Parameter(x)


class AlphaParam(nn.Module):
    
    def __init__(self, alpha0 = None):
        super().__init__()

        self.a = _init_param(alpha0)
        parametrize.register_parametrization(self, 'a', UnitConstraint())
        
    def forward(self, x):
        return self.a*x
    
    def to_string(self, fmt=':.5f'):
        fmt = '{'+fmt+'}'
        return fmt.format(self.a.item())
        

# define methods that allow for model simulation in addition to model fitting
class ModelAgent(nn.Module):
    
    def reset_state(self):
        pass
    
    def step(self, input):
        pass
    
    def print_params(self):
        pass

# %% Single Value Agents

class SingleValueAgent(ModelAgent):
    """ Reward-Seeking/Value Agent V

    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output_v: the output value of the Reward-Seeking/Value Agent V on each trial. Tensor of shape (n_sess, n_trials, 1)
    """

    def __init__(self, alpha0 = None):
        super().__init__()
            
        self.alpha_v = AlphaParam(alpha0)
        self.v = torch.tensor(0)
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 1).to(input.device)
        input_diff = torch.zeros_like(output)
        output_delta = torch.zeros_like(output)
        self.v = self.v.to(input.device)

        # Propogate input through the network
        for t in range(input.shape[1]):
            new_v, diff, delta = self.step(input[:,t,:])
            output[:,t,:] = new_v
            input_diff[:,t,:] = diff
            output_delta[:,t,:] = delta

        return output, input_diff, output_delta
    
    def reset_state(self):
        self.v = torch.tensor(0)
        
    def step(self, input):
        diff = input[:,[0]]*input[:,[1]] - self.v
        delta = self.alpha_v(diff)
        self.v = self.v + delta
        
        return self.v, diff, delta
    
    def print_params(self):
        return 'Value Agent: α = {}'.format(self.alpha_v.to_string()) 


class PerseverativeAgent(ModelAgent):
    """ Perseverative Agent H
    
    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output_h: the output value of the Perseverative Agent H on each trial. Tensor of shape (n_sess, n_trials, 1)
    
    """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha_h = AlphaParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals).to(input.device)
        input_diff = torch.zeros_like(output)
        output_delta = torch.zeros_like(output)
        self.h = self.h.to(input.device)

        # Propogate input through the network
        for t in range(input.shape[1]):
            new_h, diff, delta = self.step(input[:,t,:])
            output[:,t,:] = new_h
            input_diff[:,t,:] = diff
            output_delta[:,t,:] = delta

        return output, input_diff, output_delta
    
    def reset_state(self):
        self.h = torch.zeros(self.n_vals)
        
    def step(self, input):
        diff = input[:,:self.n_vals] - self.h
        delta = self.alpha_h(diff)
        self.h = self.h + delta
        
        return self.h, diff, delta
    
    def print_params(self):
        return 'Perseverative Agent: α = {}'.format(self.alpha_h.to_string())
    
    
class FallacyAgent(ModelAgent):
    """ Gambler Fallacy Agent G

    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output_v: the output value of the Gambler Fallacy Agent G on each trial. Tensor of shape (n_sess, n_trials, 1)
   """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha_g = AlphaParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals).to(input.device)
        input_diff = torch.zeros_like(output)
        output_delta = torch.zeros_like(output)
        self.g = self.g.to(input.device)

        # Propogate input through the network
        for t in range(input.shape[1]):
            new_g, diff, delta = self.step(input[:,t,:])
            output[:,t,:] = new_g
            input_diff[:,t,:] = diff
            output_delta[:,t,:] = delta

        return output, input_diff, output_delta
    
    def reset_state(self):
        self.g = torch.zeros(self.n_vals)
        
    def step(self, input):
        diff = input[:,:self.n_vals] - input[:,:self.n_vals]*input[:,[self.n_vals]] - self.g
        delta = self.alpha_g(diff)
        self.g = self.g + delta
        
        return self.g, diff, delta
    
    def print_params(self):
        return 'Fallacy Agent: α = {}'.format(self.alpha_g.to_string())
    

# %% Q Value Agent

class QValueAgent(ModelAgent):
    """ Q-learning Value Agent 

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output_v: the output value of the Q-Value Agent on each trial. Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=1, k_diff_unrew=0, constraints={}):
        
        super().__init__()
            
        self.alpha_same_rew = AlphaParam(alpha_same_rew)
        self.alpha_same_unrew = AlphaParam(alpha_same_unrew)
        self.alpha_diff_rew = AlphaParam(alpha_diff_rew)
        self.alpha_diff_unrew = AlphaParam(alpha_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        
        # apply parameter constraints
        for key, vals in constraints.items():
            # reassign parameter to be the same instance of another parameter
            if 'share' in vals:
                setattr(self, key, getattr(self, vals['share']))
            # change whether this parameter is being fit
            if 'fit' in vals:
                param = getattr(self, key)
                param.requires_grad = vals['fit']
        
        self.v = torch.tensor([0,0])
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2).to(input.device)
        input_diff = torch.zeros_like(output)
        output_delta = torch.zeros_like(output)
        self.v = self.v.to(input.device)

        # Propogate input through the network
        for t in range(input.shape[1]):
            new_v, diff, delta = self.step(input[:,t,:])
            output[:,t,:] = new_v
            input_diff[:,t,:] = diff
            output_delta[:,t,:] = delta

        return output, input_diff, output_delta
    
    def reset_state(self):
        self.v = torch.tensor([[0,0]])
        
    def step(self, input):
        left_diffs = torch.stack([input[:,0]*input[:,2]*(self.k_same_rew - self.v[:,0]), 
                                  input[:,0]*(1-input[:,2])*(self.k_same_unrew - self.v[:,0]),
                                  input[:,1]*input[:,2]*(self.k_diff_rew - self.v[:,0]),
                                  input[:,1]*(1-input[:,2])*(self.k_diff_unrew - self.v[:,0])], dim=1)
        
        right_diffs = torch.stack([input[:,1]*input[:,2]*(self.k_same_rew - self.v[:,1]), 
                                   input[:,1]*(1-input[:,2])*(self.k_same_unrew - self.v[:,1]),
                                   input[:,0]*input[:,2]*(self.k_diff_rew - self.v[:,1]),
                                   input[:,0]*(1-input[:,2])*(self.k_diff_unrew - self.v[:,1])], dim=1)
        
        left_deltas = torch.stack([self.alpha_same_rew(left_diffs[:,0]), self.alpha_same_unrew(left_diffs[:,1]),
                                   self.alpha_diff_rew(left_diffs[:,2]), self.alpha_diff_unrew(left_diffs[:,3])], dim=1)
        
        right_deltas = torch.stack([self.alpha_same_rew(right_diffs[:,0]), self.alpha_same_unrew(right_diffs[:,1]),
                                    self.alpha_diff_rew(right_diffs[:,2]), self.alpha_diff_unrew(right_diffs[:,3])], dim=1)
        
        diff = torch.stack([left_diffs.sum(dim=1), right_diffs.sum(dim=1)], dim=1)
        delta = torch.stack([left_deltas.sum(dim=1), right_deltas.sum(dim=1)], dim=1)
        self.v = self.v + delta
        
        return self.v, diff, delta
    
    def print_params(self):
        return '''Q Value Agent: \n\t α same, rew = {} \n\t α same, unrew = {} \n\t α diff, rew = {} \n\t α diff, unrew = {}
\t κ same, rew = {:.5f} \n\t κ same, unrew = {:.5f} \n\t κ diff, rew = {:.5f} \n\t κ diff, unrew = {:.5f}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_unrew.to_string(), self.alpha_diff_rew.to_string(), self.alpha_diff_unrew.to_string(),
                  self.k_same_rew.item(), self.k_same_unrew.item(), self.k_diff_rew.item(), self.k_diff_unrew.item()) 
    

# %% Summantion Agent

class SummationModule(ModelAgent):
    """ Sum agent outputs together to get an output value of the choice on the current trial

    Args:
        agents: instances of agent classes to include in the module.
        
    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs (in order)
        output_stack: the agent values on each trial
        output: the network output of choice
    """

    def __init__(self, agents, output_layer=None, bias=None, betas=None):
        super().__init__()
        #make a linear weight Beta (mx1) from agents to output
        #where m is the number of agents
        self.agents = nn.ModuleList(agents)
        self.beta = nn.Linear(len(agents), 1, bias=False)
        parametrize.register_parametrization(self.beta, 'weight', PositiveConstraint())
        if not betas is None:
            with torch.no_grad():
                self.beta.weight = torch.tensor(betas).reshape_as(self.beta.weight)
                
        # have separate bias to handle scenario with outputs for each choice
        # in this scenario, still only want one bias term
        if bias is None:
            bias = torch.rand(1)
        else:
            bias = torch.tensor([bias])
        self.bias = nn.Parameter(bias)
        self.output_layer = output_layer
       
    
    def forward(self, input): #first col of input-choice, second col of input-outcome
        self.reset_state()
        return self._gen_apply_agents(input, lambda agent, input: agent(input))
        
    
    def reset_state(self):
        for model in self.agents:
            model.reset_state()
        
    def step(self, input):
        return self._gen_apply_agents(input, lambda agent, input: agent.step(input))
    
    def print_params(self):
        print_str = ''
        for agent in self.agents:
            print_str += agent.print_params() + '\n'
        return print_str + 'Summation Agent: bias = {:.5f}; beta = [{}]'.format(self.bias.item(), ', '.join(['{:.5f}'.format(w.item()) for w in self.beta.weight.view(-1)]))
        
    
    def _gen_apply_agents(self, input, agent_method):
        
        # store outputs from every model's forward method
        output_stack = [] 
        input_diff_stack = []
        output_delta_stack = []
        
        # Propogate input through the network
        # dont want to call step here because we need to call forward method of each agent
        for agent in self.agents: 
            output, input_diff, output_delta = agent_method(agent, input)
            # store outputs together in lists
            output_stack.append(output) 
            input_diff_stack.append(input_diff) 
            output_delta_stack.append(output_delta) 
        
        # concatenate tensor outputs on the last dimension (parameter output)
        output_stack = torch.stack(output_stack, dim=-1)
        input_diff_stack = torch.stack(input_diff_stack, dim=-1)
        output_delta_stack = torch.stack(output_delta_stack, dim=-1)
        
        # Apply the beta linear weights
        output = self.beta(output_stack) 
        
        # output will always be scalar for every choice (3rd dimensions), so collapse the 4th dimension into the third for ease of use
        # ndim will be <4 when simulating data
        if output.ndim == 4:
            output = output.squeeze(-1)
        
        # configure the bias term based on the number of choices, now in the last position
        b = torch.cat([self.bias, torch.zeros(output.shape[-1]-1)])
        
        # Apply the bias
        output += output + b
        
        # Apply output layer if given
        if not self.output_layer is None:
            output = self.output_layer(output)

        return output, {'agent_states': output_stack.detach().numpy(), 
                        'agent_input_state_diffs': input_diff_stack.detach().numpy(), 
                        'agent_state_deltas': output_delta_stack.detach().numpy()}
        