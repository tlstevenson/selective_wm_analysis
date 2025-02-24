# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 13:00:09 2024

@author: tanne
"""

# %% General Infrastructure

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import torch.nn.functional as functional
import json
from pyutils import utils
import copy

def smooth_clamp(x, lower_bound=0, upper_bound=1):
    """
    Smoothly clamps inputs to [lower_bound, upper_bound], with a linear relationship between
    
    Parameters:
    - x: Input tensor.
    
    Returns:
    - Tensor with values constrained to [lower_bound, upper_bound].
    """
    beta = (upper_bound - lower_bound) * 100
    low_sel = x < (upper_bound + lower_bound)/2
    ret_val = torch.zeros_like(x)
    ret_val[low_sel] = lower_bound + functional.softplus(x[low_sel] - lower_bound, beta=beta)
    ret_val[~low_sel] = upper_bound - functional.softplus(upper_bound - x[~low_sel], beta=beta)

    return ret_val

class UnitConstraint(nn.Module):
    # constrain values to be between 0 and 1
    # Note: using smooth_clamp results in much slower fitting
    def forward(self, x):
        return torch.sigmoid(x)
    
    def right_inverse(self, x):
        # convert an initialization between 0 and 1 into the inverse sigmoid space 
        return torch.log(x) - torch.log1p(-x)
    
class PositiveConstraint(nn.Module):
    # constrain values to be positive using softplus
    # having beta be 100 closely approximates the relu function
    def __init__(self, beta=100, thresh=20):
        super().__init__()
        self.beta = beta
        self.thresh = thresh

    def forward(self, x):
        return functional.softplus(x, beta=self.beta, threshold=self.thresh)
    
    def right_inverse(self, x):
        # convert an initialization into the inverse softplus space
        below_thresh = x*self.beta < self.thresh
        new_vals = x.clone()
        new_vals[below_thresh] = torch.log(torch.expm1(self.beta*torch.abs(x[below_thresh])))/self.beta
        return new_vals
    
def _init_param(x=None):
    if x is None:
        x = torch.rand(1)
    else:
        # make sure this is a float
        x = torch.tensor([x]).type(torch.float)
        
    return nn.Parameter(x)


class AlphaParam(nn.Module):
    
    def __init__(self, alpha0 = None):
        super().__init__()

        self.a = _init_param(alpha0)
        parametrize.register_parametrization(self, 'a', UnitConstraint())
        
    def __repr__(self):
         return 'Alpha Param: {}'.format(self.to_string())
        
    def forward(self, x):
        return self.a*x
    
    @property
    def requires_grad(self):
        return self.parametrizations.a.original.requires_grad  # Get from the parameter

    @requires_grad.setter
    def requires_grad(self, value: bool):
        self.parametrizations.a.original.requires_grad = value  # Set for the parameter
    
    def to_string(self, fmt=':.5f'):
        fmt = '{'+fmt+'}'
        return fmt.format(self.a.item())
        

# define methods that allow for model simulation in addition to model fitting
class ModelAgent(nn.Module):
    
    def reset_params(self):
        pass
    
    def reset_state(self):
        pass
    
    def step(self, input):
        pass
    
    def print_params(self):
        pass
    
    def toJson(self):
        pass
    
    def formatJson(self, child_data):
        return {'type': type(self).__name__, 'data': child_data}
    
    @staticmethod
    def fromJson(json_str):
        pass
    
    def clone(self):
        pass

# %% Single Value Agents

class SingleValueAgent(ModelAgent):
    """ Reward-Seeking/Value Agent V

    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Reward-Seeking/Value Agent V on each trial. Tensor of shape (n_sess, n_trials, 1)
    """

    def __init__(self, alpha0 = None):
        super().__init__()
            
        self.alpha = AlphaParam(alpha0)
        self.reset_state()
        
    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 1)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_params(self):
        self.alpha = AlphaParam()
        self.reset_state()
    
    def reset_state(self):
        self.state = torch.tensor(0)
        self.state_hist = []
        self.diff_hist = []
        self.delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        diff = input[:,[0]]*input[:,[1]] - self.state
        delta = self.alpha(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.diff_hist.append(diff.detach())
        self.delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Value Agent: α = {}'.format(self.alpha.to_string())
    
    def toJson(self):
        return super().formatJson({'alpha': self.alpha.a.item()})
    
    def fromJson(data):
        return SingleValueAgent(alpha0=data['alpha'])


class PerseverativeAgent(ModelAgent):
    """ Perseverative Agent H
    
    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Perseverative Agent H on each trial. Tensor of shape (n_sess, n_trials, 1)
    
    """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha = AlphaParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
        
    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_params(self):
        self.alpha = AlphaParam()
        self.reset_state()
        
    def reset_state(self):
        self.state = torch.zeros(self.n_vals)
        self.state_hist = []
        self.diff_hist = []
        self.delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        diff = input[:,:self.n_vals] - self.state
        delta = self.alpha(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.diff_hist.append(diff.detach())
        self.delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Perseverative Agent: α = {}'.format(self.alpha.to_string())
    
    def toJson(self):
        return super().formatJson({'alpha': self.alpha.a.item(), 'n_vals': self.n_vals})
    
    def fromJson(data):
        return PerseverativeAgent(alpha0=data['alpha'], n_vals=data['n_vals'])
    
    
class FallacyAgent(ModelAgent):
    """ Gambler Fallacy Agent G

    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Gambler Fallacy Agent G on each trial. Tensor of shape (n_sess, n_trials, 1)
   """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha = AlphaParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
        
    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_params(self):
        self.alpha = AlphaParam()
        self.reset_state()
        
    def reset_state(self):
        self.state = torch.zeros(self.n_vals)
        self.state_hist = []
        self.diff_hist = []
        self.delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        diff = input[:,:self.n_vals] - input[:,:self.n_vals]*input[:,[self.n_vals]] - self.state
        delta = self.alpha(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.diff_hist.append(diff.detach())
        self.delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Fallacy Agent: α = {}'.format(self.alpha.to_string())
    
    def toJson(self):
        return super().formatJson({'alpha': self.alpha.a.item(), 'n_vals': self.n_vals})
    
    def fromJson(data):
        return FallacyAgent(alpha0=data['alpha'], n_vals=data['n_vals'])
    

# %% Q Value Agent

class QValueAgent(ModelAgent):
    """ Q-learning Value Agent 

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output: the output value of the Q-Value Agent on each trial. Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, constraints=None):
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
            
        self.alpha_same_rew = AlphaParam(alpha_same_rew)
        self.alpha_same_unrew = AlphaParam(alpha_same_unrew)
        self.alpha_diff_rew = AlphaParam(alpha_diff_rew)
        self.alpha_diff_unrew = AlphaParam(alpha_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        self.constraints = copy.deepcopy(constraints)
        
        self.apply_constraints()
        
        self.reset_state()
        
    def apply_constraints(self):
        # apply parameter constraints
        for key, vals in self.constraints.items():
            # reassign parameter to be the same instance of another parameter
            if 'share' in vals:
                setattr(self, key, getattr(self, vals['share']))
            # change whether this parameter is being fit
            if 'fit' in vals:
                param = getattr(self, key)
                param.requires_grad = vals['fit']
            # change whether this parameter is being fit
            if 'init' in vals:
                param = getattr(self, key)
                with torch.no_grad():
                    if isinstance(param, AlphaParam):
                        param.parametrizations.a.right_inverse(torch.tensor([vals['init']]))
                    else:
                        param.copy_(torch.tensor([vals['init']]))
        
    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_params(self):
        self.alpha_same_rew = AlphaParam()
        self.alpha_same_unrew = AlphaParam()
        self.alpha_diff_rew = AlphaParam()
        self.alpha_diff_unrew = AlphaParam()
        self.k_same_rew = _init_param(1)
        self.k_same_unrew = _init_param(0)
        self.k_diff_rew = _init_param(0)
        self.k_diff_unrew = _init_param(0)
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        self.state = torch.tensor([[0,0]])
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
        
        self.k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        self.alphas = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        
    def step(self, input):
        left_side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                           (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        right_side_outcome_sel = left_side_outcome_sel[:,[2,3,0,1]]
        
        left_diffs = self.k_vals - self.state[:,0].unsqueeze(1)
        right_diffs = self.k_vals - self.state[:,1].unsqueeze(1)
        
        left_state_diffs = left_side_outcome_sel*left_diffs
        right_state_diffs = right_side_outcome_sel*right_diffs
        
        # update state
        left_state_deltas = left_state_diffs * self.alphas
        right_state_deltas = right_state_diffs * self.alphas
        
        state_diff = torch.cat([left_state_diffs.sum(dim=1).unsqueeze(1), right_state_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        state_delta = torch.cat([left_state_deltas.sum(dim=1).unsqueeze(1), right_state_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.state = self.state + state_delta

        # record state histories
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())

        return self.state
    
    def print_params(self):
        return '''Q Value Agent: \n\t α same, rew = {}, fit: {} \n\t α same, unrew = {}, fit: {} \n\t α diff, rew = {}, fit: {} 
\t α diff, unrew = {}, fit: {} \n\t κ same, rew = {:.5f}, fit: {} \n\t κ same, unrew = {:.5f}, fit: {} 
\t κ diff, rew = {:.5f}, fit: {} \n\t κ diff, unrew = {:.5f}, fit: {}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_rew.requires_grad, self.alpha_same_unrew.to_string(), self.alpha_same_unrew.requires_grad,
                  self.alpha_diff_rew.to_string(), self.alpha_diff_rew.requires_grad, self.alpha_diff_unrew.to_string(), self.alpha_diff_unrew.requires_grad,
                  self.k_same_rew.item(), self.k_same_rew.requires_grad, self.k_same_unrew.item(), self.k_same_unrew.requires_grad, 
                  self.k_diff_rew.item(), self.k_diff_rew.requires_grad, self.k_diff_unrew.item(), self.k_diff_unrew.requires_grad) 

    def toJson(self):
        return super().formatJson({'alpha_same_rew': self.alpha_same_rew.a.item(), 'alpha_same_unrew': self.alpha_same_unrew.a.item(),
                                   'alpha_diff_rew': self.alpha_diff_rew.a.item(), 'alpha_diff_unrew': self.alpha_diff_unrew.a.item(),
                                   'k_same_rew': self.k_same_rew.item(), 'k_same_unrew': self.k_same_unrew.item(),
                                   'k_diff_rew': self.k_diff_rew.item(), 'k_diff_unrew': self.k_diff_unrew.item(),
                                   'constraints': self.constraints})
    
    def fromJson(data):
        return QValueAgent(alpha_same_rew=data['alpha_same_rew'], alpha_same_unrew=data['alpha_same_unrew'], 
                           alpha_diff_rew=data['alpha_diff_rew'], alpha_diff_unrew=data['alpha_diff_unrew'], 
                           k_same_rew=data['k_same_rew'], k_same_unrew=data['k_same_unrew'], 
                           k_diff_rew=data['k_diff_rew'], k_diff_unrew=data['k_diff_unrew'], constraints=data['constraints'])
    
    def clone(self):
        return QValueAgent.fromJson(self.toJson()['data'])
    
# %% Dynamic Learning Rate

class DynamicQAgent(ModelAgent):
    """ Q-learning Value Agent w/ dynamic learning rates based on history of RPE values
        Inspired by Mischanchuk et al. 2024

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output_v: the output value of the Q-Value Agent on each trial. Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None,
                 gamma_same_rew=None, gamma_same_unrew=None, gamma_diff_rew=None, gamma_diff_unrew=None,
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, inverse_update=False, global_lam=True, constraints=None):
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
        
        self.alpha_same_rew = AlphaParam(alpha_same_rew)
        self.alpha_same_unrew = AlphaParam(alpha_same_unrew)
        self.alpha_diff_rew = AlphaParam(alpha_diff_rew)
        self.alpha_diff_unrew = AlphaParam(alpha_diff_unrew)
        self.gamma_same_rew = AlphaParam(gamma_same_rew)
        self.gamma_same_unrew = AlphaParam(gamma_same_unrew)
        self.gamma_diff_rew = AlphaParam(gamma_diff_rew)
        self.gamma_diff_unrew = AlphaParam(gamma_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        self.inverse_update = inverse_update
        self.global_lam = global_lam
        self.constraints = copy.deepcopy(constraints)
        
        # automatically constrain gammas if the lambda multiplier is a global term for both sides
        if global_lam:
            constraints['gamma_diff_rew'] = {'share': 'gamma_same_rew'}
            constraints['gamma_diff_unrew'] = {'share': 'gamma_same_unrew'}
        
        # apply parameter constraints
        for key, vals in constraints.items():
            # reassign parameter to be the same instance of another parameter
            if 'share' in vals:
                setattr(self, key, getattr(self, vals['share']))
            # change whether this parameter is being fit
            if 'fit' in vals:
                param = getattr(self, key)
                param.requires_grad = vals['fit']
        
        self.reset_state()
        
    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_state(self):
        self.state = torch.zeros((1,2))
        self.lam = torch.zeros((1,2)) + 0.5            
        self.state_hist = []
        self.lam_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.lam_diff_hist = []
        self.lam_delta_hist = []
        self.state_hist.append(self.state.detach())
        self.lam_hist.append(self.lam.detach())
        
        self.k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        self.alphas = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        self.gammas = torch.cat([self.gamma_same_rew.a, self.gamma_same_unrew.a, self.gamma_diff_rew.a, self.gamma_diff_unrew.a]).unsqueeze(0)
        
    def step(self, input):
        
        left_side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                           (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        right_side_outcome_sel = left_side_outcome_sel[:,[2,3,0,1]]
        
        left_diffs = self.k_vals - self.state[:,0].unsqueeze(1)
        right_diffs = self.k_vals - self.state[:,1].unsqueeze(1)
        
        left_state_diffs = left_side_outcome_sel*left_diffs
        right_state_diffs = right_side_outcome_sel*right_diffs
        
        # update state
        lam_mult = self.lam
        if self.inverse_update:
            lam_mult = 1 - lam_mult

        left_state_deltas = left_state_diffs * self.alphas * lam_mult[:,0].unsqueeze(1)
        right_state_deltas = right_state_diffs * self.alphas * lam_mult[:,1].unsqueeze(1)
        
        state_diff = torch.cat([left_state_diffs.sum(dim=1).unsqueeze(1), right_state_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        state_delta = torch.cat([left_state_deltas.sum(dim=1).unsqueeze(1), right_state_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.state = self.state + state_delta
        
        # update update multiplier lambdas

        if self.global_lam:
            # reconfigure the diffs to only include the k_same values in the order: left/reward, left/unreward, right/reward, right/unreward (so it works with left_side_outcome_sel)
            global_diffs = torch.cat([left_diffs[:,[0,1]], right_diffs[:,[0,1]]], dim=1)
            left_lam_diffs = (torch.abs(global_diffs) - self.lam[:,0].unsqueeze(1))*left_side_outcome_sel
            right_lam_diffs = left_lam_diffs
    
            left_lam_deltas = left_lam_diffs*self.gammas
            right_lam_deltas = left_lam_deltas
        else:
            left_lam_diffs = (torch.abs(left_diffs) - self.lam[:,0].unsqueeze(1))*left_side_outcome_sel
            right_lam_diffs = (torch.abs(right_diffs) - self.lam[:,1].unsqueeze(1))*right_side_outcome_sel
    
            left_lam_deltas = left_lam_diffs*self.gammas
            right_lam_deltas = right_lam_diffs*self.gammas
        
        lam_diff = torch.cat([left_lam_diffs.sum(dim=1).unsqueeze(1), right_lam_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        lam_delta = torch.cat([left_lam_deltas.sum(dim=1).unsqueeze(1), right_lam_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.lam = self.lam + lam_delta
        
        # record state histories
        self.state_hist.append(self.state.detach())
        self.lam_hist.append(self.lam.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())
        self.lam_diff_hist.append(lam_diff.detach())
        self.lam_delta_hist.append(lam_delta.detach())
        
        return self.state
    
    def print_params(self):
        return '''Q Value Agent: \n\t α same, rew = {}, fit: {} \n\t α same, unrew = {}, fit: {} \n\t α diff, rew = {}, fit: {} \n\t α diff, unrew = {}, fit: {} 
\t γ same, rew = {}, fit: {} \n\t γ same, unrew = {}, fit: {} \n\t γ diff, rew = {}, fit: {} \n\t γ diff, unrew = {}, fit: {} 
\t κ same, rew = {:.5f}, fit: {} \n\t κ same, unrew = {:.5f}, fit: {} \n\t κ diff, rew = {:.5f}, fit: {} \n\t κ diff, unrew = {:.5f}, fit: {}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_rew.requires_grad, self.alpha_same_unrew.to_string(), self.alpha_same_unrew.requires_grad, 
                  self.alpha_diff_rew.to_string(), self.alpha_diff_rew.requires_grad, self.alpha_diff_unrew.to_string(), self.alpha_diff_unrew.requires_grad,
                  self.gamma_same_rew.to_string(), self.gamma_same_rew.requires_grad, self.gamma_same_unrew.to_string(), self.gamma_same_unrew.requires_grad, 
                  self.gamma_diff_rew.to_string(), self.gamma_diff_rew.requires_grad, self.gamma_diff_unrew.to_string(), self.gamma_diff_unrew.requires_grad,
                  self.k_same_rew.item(), self.k_same_rew.requires_grad, self.k_same_unrew.item(), self.k_same_unrew.requires_grad, 
                  self.k_diff_rew.item(), self.k_diff_rew.requires_grad, self.k_diff_unrew.item(), self.k_diff_unrew.requires_grad)

    def toJson(self):
        return super().formatJson({'alpha_same_rew': self.alpha_same_rew.a.item(), 'alpha_same_unrew': self.alpha_same_unrew.a.item(),
                                   'alpha_diff_rew': self.alpha_diff_rew.a.item(), 'alpha_diff_unrew': self.alpha_diff_unrew.a.item(),
                                   'gamma_same_rew': self.gamma_same_rew.a.item(), 'gamma_same_unrew': self.gamma_same_unrew.a.item(),
                                   'gamma_diff_rew': self.gamma_diff_rew.a.item(), 'gamma_diff_unrew': self.gamma_diff_unrew.a.item(),
                                   'k_same_rew': self.k_same_rew.item(), 'k_same_unrew': self.k_same_unrew.item(),
                                   'k_diff_rew': self.k_diff_rew.item(), 'k_diff_unrew': self.k_diff_unrew.item(),
                                   'inverse_update': self.inverse_update, 'global_lam': self.global_lam, 'constraints': self.constraints})
    
    def fromJson(data):
        return DynamicQAgent(alpha_same_rew=data['alpha_same_rew'], alpha_same_unrew=data['alpha_same_unrew'], 
                           alpha_diff_rew=data['alpha_diff_rew'], alpha_diff_unrew=data['alpha_diff_unrew'], 
                           gamma_same_rew=data['gamma_same_rew'], gamma_same_unrew=data['gamma_same_unrew'], 
                           gamma_diff_rew=data['gamma_diff_rew'], gamma_diff_unrew=data['gamma_diff_unrew'], 
                           k_same_rew=data['k_same_rew'], k_same_unrew=data['k_same_unrew'], 
                           k_diff_rew=data['k_diff_rew'], k_diff_unrew=data['k_diff_unrew'], 
                           inverse_update=data['inverse_update'], global_lam=data['global_lam'], constraints=data['constraints'])
    
# %% Uncertainty Dynamic Learning Rate 

# make another version of this that matches grossman et al. exactly
class UncertaintyDynamicQAgent(ModelAgent):
    """ Q-learning Value Agent w/ two different dynamic learning rate terms based on history of RPE values.
        One term is the same multiplier as the dynamic Q agent above (expected uncertainty) but only with an inverse update rule where higher expected uncertainty decreases learning rate
        The other term is a variable learning rate that increases with unexpected uncertainty (the difference between expected uncertainty and outcome)
        Inspired by Grossman et al. 2022

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output_v: the output value of the Q-Value Agent on each trial. Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None,
                 gamma_lam_same_rew=None, gamma_lam_same_unrew=None, gamma_lam_diff_rew=None, gamma_lam_diff_unrew=None,
                 gamma_alpha_same_rew=None, gamma_alpha_same_unrew=None, gamma_alpha_diff_rew=None, gamma_alpha_diff_unrew=None,
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, global_lam=True, shared_side_alphas=True, shared_outcome_alpha_update=True, constraints=None):
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
        
        self.alpha_same_rew = AlphaParam(alpha_same_rew)
        self.alpha_same_unrew = AlphaParam(alpha_same_unrew)
        self.alpha_diff_rew = AlphaParam(alpha_diff_rew)
        self.alpha_diff_unrew = AlphaParam(alpha_diff_unrew)
        self.gamma_lam_same_rew = AlphaParam(gamma_lam_same_rew)
        self.gamma_lam_same_unrew = AlphaParam(gamma_lam_same_unrew)
        self.gamma_lam_diff_rew = AlphaParam(gamma_lam_diff_rew)
        self.gamma_lam_diff_unrew = AlphaParam(gamma_lam_diff_unrew)
        self.gamma_alpha_same_rew = AlphaParam(gamma_alpha_same_rew)
        self.gamma_alpha_same_unrew = AlphaParam(gamma_alpha_same_unrew)
        self.gamma_alpha_diff_rew = AlphaParam(gamma_alpha_diff_rew)
        self.gamma_alpha_diff_unrew = AlphaParam(gamma_alpha_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        self.global_lam = global_lam
        self.shared_side_alphas = shared_side_alphas
        self.shared_outcome_alpha_update = shared_outcome_alpha_update # whether to update alphas separately based on the outcome
        self.constraints = copy.deepcopy(constraints)
        
        # automatically constrain alphas & gammas if the alpha multiplier is shared for both sides
        if shared_side_alphas:
            constraints['alpha_diff_rew'] = {'share': 'alpha_same_rew'}
            constraints['alpha_diff_unrew'] = {'share': 'alpha_same_unrew'}
            constraints['gamma_alpha_diff_rew'] = {'share': 'gamma_alpha_same_rew'}
            constraints['gamma_alpha_diff_unrew'] = {'share': 'gamma_alpha_same_unrew'}
            
        # automatically constrain gammas if the lambda multiplier is a global term for both sides
        if global_lam:
            constraints['gamma_lam_diff_rew'] = {'share': 'gamma_lam_same_rew'}
            constraints['gamma_lam_diff_unrew'] = {'share': 'gamma_lam_same_unrew'}
        
        # apply parameter constraints
        for key, vals in constraints.items():
            # reassign parameter to be the same instance of another parameter
            if 'share' in vals:
                setattr(self, key, getattr(self, vals['share']))
            # change whether this parameter is being fit
            if 'fit' in vals:
                param = getattr(self, key)
                param.requires_grad = vals['fit']
        
        self.reset_state()

    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_state(self):
        self.state = torch.zeros((1,2))
        self.lam = torch.zeros((1,2)) + 0.5
        # alpha is stored as left/reward, left/unreward, right/reward, right/unreward
        self.alpha = None        
        self.state_hist = []
        self.lam_hist = []
        self.alphaist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.lam_diff_hist = []
        self.lam_delta_hist = []
        self.alpha_diff_hist = []
        self.alpha_delta_hist = []
        self.state_hist.append(self.state.detach())
        self.lam_hist.append(self.lam.detach())
        
        self.k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        self.alpha0s = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        self.gamma_alphas = torch.cat([self.gamma_alpha_same_rew.a, self.gamma_alpha_same_unrew.a, self.gamma_alpha_diff_rew.a, self.gamma_alpha_diff_unrew.a]).unsqueeze(0)
        self.gamma_lams = torch.cat([self.gamma_lam_same_rew.a, self.gamma_lam_same_unrew.a, self.gamma_lam_diff_rew.a, self.gamma_lam_diff_unrew.a]).unsqueeze(0)

        
    def step(self, input):
        
        left_side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                           (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        right_side_outcome_sel = left_side_outcome_sel[:,[2,3,0,1]]
        
        left_diffs = self.k_vals - self.state[:,0].unsqueeze(1)
        right_diffs = self.k_vals - self.state[:,1].unsqueeze(1)
        
        # calculate unexpected uncertainty on current trial
        left_lam_diffs = torch.abs(left_diffs) - self.lam[:,0].unsqueeze(1)
        right_lam_diffs = torch.abs(right_diffs) - self.lam[:,1].unsqueeze(1)
        
        # calculate alpha on current trial
        
        # initialize alpha based of choice/outcome and parameter options
        if self.alpha is None:
            if self.shared_side_alphas and self.shared_outcome_alpha_update:
                # alphas are all the same value, starting at value determined by first outcome
                # doesn't matter which side is chosen because alpha same/diff is the same
                alphas = (self.alpha0s*left_side_outcome_sel).sum(dim=1).unsqueeze(1)
                self.alpha = alphas.repeat(1,4)
            else:
                # alphas can be different values so initialize to same rew/unrew 
                self.alpha = self.alpha0s[:,[0,1]].repeat(input.shape[0],2)
      
            alpha_diff = torch.zeros_like(left_side_outcome_sel)
            alpha_delta = torch.zeros_like(left_side_outcome_sel)

        else:
            
            # calculate all alpha diff values used in the different update rules (depending on global alpha and separate outcome alpha options), order: same, diff
            left_rew_alpha_diffs = self.alpha0s[:,[0,2]] + left_lam_diffs[:,[0,2]] - self.alpha[:,0].unsqueeze(1)
            left_unrew_alpha_diffs = self.alpha0s[:,[1,3]] + left_lam_diffs[:,[1,3]] - self.alpha[:,1].unsqueeze(1)
            right_rew_alpha_diffs = self.alpha0s[:,[0,2]] + right_lam_diffs[:,[0,2]] - self.alpha[:,2].unsqueeze(1)
            right_unrew_alpha_diffs = self.alpha0s[:,[1,3]] + right_lam_diffs[:,[1,3]] - self.alpha[:,3].unsqueeze(1)
            
            if self.shared_side_alphas:
                if self.shared_outcome_alpha_update:
                    # alphas are shared between sides and updates are combined for each outcome
                    left_rew_alpha_diffs = torch.cat([(left_rew_alpha_diffs[:,0]*left_side_outcome_sel[:,0]).unsqueeze(1),
                                                      (left_unrew_alpha_diffs[:,0]*left_side_outcome_sel[:,1]).unsqueeze(1),
                                                      (right_rew_alpha_diffs[:,0]*right_side_outcome_sel[:,0]).unsqueeze(1),
                                                      (right_unrew_alpha_diffs[:,0]*right_side_outcome_sel[:,1]).unsqueeze(1)], dim=1)
                    right_rew_alpha_diffs = left_rew_alpha_diffs
                    left_unrew_alpha_diffs = left_rew_alpha_diffs
                    right_unrew_alpha_diffs = left_rew_alpha_diffs
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*self.gamma_alphas
                    right_rew_alpha_deltas = left_rew_alpha_deltas
                    left_unrew_alpha_deltas = left_rew_alpha_deltas
                    right_unrew_alpha_deltas = left_rew_alpha_deltas
                else:
                    # alphas are shared between sides but updates are separated for each outcome
                    left_rew_alpha_diffs = torch.cat([(left_rew_alpha_diffs[:,0]*left_side_outcome_sel[:,0]).unsqueeze(1), 
                                                      (right_rew_alpha_diffs[:,0]*right_side_outcome_sel[:,0]).unsqueeze(1)], dim=1)
                    right_rew_alpha_diffs = left_rew_alpha_diffs
                    left_unrew_alpha_diffs = torch.cat([(left_unrew_alpha_diffs[:,0]*left_side_outcome_sel[:,1]).unsqueeze(1), 
                                                        (right_unrew_alpha_diffs[:,0]*right_side_outcome_sel[:,1]).unsqueeze(1)], dim=1)
                    right_unrew_alpha_diffs = left_unrew_alpha_diffs
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*self.gamma_alphas[:,[0,2]]
                    right_rew_alpha_deltas = left_rew_alpha_deltas
                    left_unrew_alpha_deltas = left_unrew_alpha_diffs*self.gamma_alphas[:,[1,3]]
                    right_unrew_alpha_deltas = left_unrew_alpha_deltas
                    
            else:
                if self.shared_outcome_alpha_update:
                    # alphas are incremented separately for each side but updates are combined for each outcome
                    # This is an analogous update rule as the classic q-value agent
                    left_rew_alpha_diffs = torch.cat([left_rew_alpha_diffs[:,0].unsqueeze(1), left_unrew_alpha_diffs[:,0].unsqueeze(1),
                                                      left_rew_alpha_diffs[:,1].unsqueeze(1), left_unrew_alpha_diffs[:,1].unsqueeze(1)], dim=1)*left_side_outcome_sel
                    right_rew_alpha_diffs = torch.cat([right_rew_alpha_diffs[:,0].unsqueeze(1), right_unrew_alpha_diffs[:,0].unsqueeze(1),
                                                       right_rew_alpha_diffs[:,1].unsqueeze(1), right_unrew_alpha_diffs[:,1].unsqueeze(1)], dim=1)*right_side_outcome_sel
                    left_unrew_alpha_diffs = left_rew_alpha_diffs
                    right_unrew_alpha_diffs = right_rew_alpha_diffs
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*self.gamma_alphas
                    right_rew_alpha_deltas = right_rew_alpha_diffs*self.gamma_alphas
                    left_unrew_alpha_deltas = left_rew_alpha_deltas
                    right_unrew_alpha_deltas = right_rew_alpha_deltas
                else:
                    # alphas are incremented separately for each side and outcome
                    left_rew_alpha_diffs = left_rew_alpha_diffs*left_side_outcome_sel[:,[0,2]]
                    right_rew_alpha_diffs = right_rew_alpha_diffs*right_side_outcome_sel[:,[0,2]]
                    left_unrew_alpha_diffs = left_unrew_alpha_diffs*left_side_outcome_sel[:,[1,3]]
                    right_unrew_alpha_diffs = right_unrew_alpha_diffs*right_side_outcome_sel[:,[1,3]]
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*self.gamma_alphas[:,[0,2]]
                    right_rew_alpha_deltas = right_rew_alpha_diffs*self.gamma_alphas[:,[0,2]]
                    left_unrew_alpha_deltas = left_unrew_alpha_diffs*self.gamma_alphas[:,[1,3]]
                    right_unrew_alpha_deltas = right_unrew_alpha_diffs*self.gamma_alphas[:,[1,3]]

        
            alpha_diff = torch.cat([left_rew_alpha_diffs.sum(dim=1).unsqueeze(1), left_unrew_alpha_diffs.sum(dim=1).unsqueeze(1), 
                                    right_rew_alpha_diffs.sum(dim=1).unsqueeze(1), right_unrew_alpha_diffs.sum(dim=1).unsqueeze(1)], dim=1)
            
            alpha_delta = torch.cat([left_rew_alpha_deltas.sum(dim=1).unsqueeze(1), left_unrew_alpha_deltas.sum(dim=1).unsqueeze(1), 
                                    right_rew_alpha_deltas.sum(dim=1).unsqueeze(1), right_unrew_alpha_deltas.sum(dim=1).unsqueeze(1)], dim=1)
    
            self.alpha = smooth_clamp(self.alpha + alpha_delta, lower_bound=0, upper_bound=1)
        
        # update state
        left_state_diffs = left_side_outcome_sel*left_diffs
        right_state_diffs = right_side_outcome_sel*right_diffs
        
        lam_mult = 1 - self.lam
        
        # handle state differences being in same/rew, same/unrew, diff/rew, diff/unrew while alphas are in left/rew, left/unrew, right/rew, right/unrew
        left_state_deltas = torch.cat([left_state_diffs[:,[0,2]]*self.alpha[:,0].unsqueeze(1), left_state_diffs[:,[1,3]]*self.alpha[:,1].unsqueeze(1)], dim=1)[:,[0,2,1,3]] * lam_mult[:,0].unsqueeze(1)
        right_state_deltas = torch.cat([right_state_diffs[:,[0,2]]*self.alpha[:,2].unsqueeze(1), right_state_diffs[:,[1,3]]*self.alpha[:,3].unsqueeze(1)], dim=1)[:,[0,2,1,3]] * lam_mult[:,1].unsqueeze(1)
        
        state_diff = torch.cat([left_state_diffs.sum(dim=1).unsqueeze(1), right_state_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        state_delta = torch.cat([left_state_deltas.sum(dim=1).unsqueeze(1), right_state_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.state = self.state + state_delta
        
        # update multiplier lambdas
        
        if self.global_lam:
            # reconfigure the diffs to only include the k_same values in the order: left/reward, left/unreward, right/reward, right/unreward (so it works with left_side_outcome_sel)
            global_lam_diffs = torch.cat([left_lam_diffs[:,[0,1]], right_lam_diffs[:,[0,1]]], dim=1)
            left_lam_diffs = global_lam_diffs*left_side_outcome_sel
            right_lam_diffs = left_lam_diffs
    
            left_lam_deltas = left_lam_diffs*self.gamma_lams
            right_lam_deltas = left_lam_deltas
        else:
            left_lam_diffs = left_lam_diffs*left_side_outcome_sel
            right_lam_diffs = right_lam_diffs*right_side_outcome_sel
    
            left_lam_deltas = left_lam_diffs*self.gamma_lams
            right_lam_deltas = right_lam_diffs*self.gamma_lams
        
        lam_diff = torch.cat([left_lam_diffs.sum(dim=1).unsqueeze(1), right_lam_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        lam_delta = torch.cat([left_lam_deltas.sum(dim=1).unsqueeze(1), right_lam_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.lam = self.lam + lam_delta
        
        # record state histories
        self.state_hist.append(self.state.detach())
        self.lam_hist.append(self.lam.detach())
        self.alphaist.append(self.alpha.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())
        self.lam_diff_hist.append(lam_diff.detach())
        self.lam_delta_hist.append(lam_delta.detach())
        self.alpha_diff_hist.append(alpha_diff.detach())
        self.alpha_delta_hist.append(alpha_delta.detach())
        
        return self.state
    
    def print_params(self):
        return '''Q Value Agent: \n\t α same, rew = {}, fit: {} \n\t α same, unrew = {}, fit: {} \n\t α diff, rew = {}, fit: {} \n\t α diff, unrew = {}, fit: {} 
\t γ_α same, rew = {}, fit: {} \n\t γ_α same, unrew = {}, fit: {} \n\t γ_α diff, rew = {}, fit: {} \n\t γ_α diff, unrew = {}, fit: {} 
\t γ_λ same, rew = {}, fit: {} \n\t γ_λ same, unrew = {}, fit: {} \n\t γ_λ diff, rew = {}, fit: {} \n\t γ_λ diff, unrew = {}, fit: {} 
\t κ same, rew = {:.5f}, fit: {} \n\t κ same, unrew = {:.5f}, fit: {} \n\t κ diff, rew = {:.5f}, fit: {} \n\t κ diff, unrew = {:.5f}, fit: {}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_rew.requires_grad, self.alpha_same_unrew.to_string(), self.alpha_same_unrew.requires_grad, 
                  self.alpha_diff_rew.to_string(), self.alpha_diff_rew.requires_grad, self.alpha_diff_unrew.to_string(), self.alpha_diff_unrew.requires_grad,
                  self.gamma_alpha_same_rew.to_string(), self.gamma_alpha_same_rew.requires_grad, self.gamma_alpha_same_unrew.to_string(), self.gamma_alpha_same_unrew.requires_grad, 
                  self.gamma_alpha_diff_rew.to_string(), self.gamma_alpha_diff_rew.requires_grad, self.gamma_alpha_diff_unrew.to_string(), self.gamma_alpha_diff_unrew.requires_grad,
                  self.gamma_lam_same_rew.to_string(), self.gamma_lam_same_rew.requires_grad, self.gamma_lam_same_unrew.to_string(), self.gamma_lam_same_unrew.requires_grad, 
                  self.gamma_lam_diff_rew.to_string(), self.gamma_lam_diff_rew.requires_grad, self.gamma_lam_diff_unrew.to_string(), self.gamma_lam_diff_unrew.requires_grad,
                  self.k_same_rew.item(), self.k_same_rew.requires_grad, self.k_same_unrew.item(), self.k_same_unrew.requires_grad, 
                  self.k_diff_rew.item(), self.k_diff_rew.requires_grad, self.k_diff_unrew.item(), self.k_diff_unrew.requires_grad)

    def toJson(self):
        return super().formatJson({'alpha_same_rew': self.alpha_same_rew.a.item(), 'alpha_same_unrew': self.alpha_same_unrew.a.item(),
                                   'alpha_diff_rew': self.alpha_diff_rew.a.item(), 'alpha_diff_unrew': self.alpha_diff_unrew.a.item(),
                                   'gamma_alpha_same_rew': self.gamma_alpha_same_rew.a.item(), 'gamma_alpha_same_unrew': self.gamma_alpha_same_unrew.a.item(),
                                   'gamma_alpha_diff_rew': self.gamma_alpha_diff_rew.a.item(), 'gamma_alpha_diff_unrew': self.gamma_alpha_diff_unrew.a.item(),
                                   'gamma_lam_same_rew': self.gamma_lam_same_rew.a.item(), 'gamma_lam_same_unrew': self.gamma_lam_same_unrew.a.item(),
                                   'gamma_lam_diff_rew': self.gamma_lam_diff_rew.a.item(), 'gamma_lam_diff_unrew': self.gamma_lam_diff_unrew.a.item(),
                                   'k_same_rew': self.k_same_rew.item(), 'k_same_unrew': self.k_same_unrew.item(),
                                   'k_diff_rew': self.k_diff_rew.item(), 'k_diff_unrew': self.k_diff_unrew.item(),
                                   'shared_side_alphas': self.shared_side_alphas, 'shared_outcome_alpha_update': self.shared_outcome_alpha_update,
                                   'global_lam': self.global_lam, 'constraints': self.constraints})
    
    def fromJson(data):
        return UncertaintyDynamicQAgent(alpha_same_rew=data['alpha_same_rew'], alpha_same_unrew=data['alpha_same_unrew'], 
                           alpha_diff_rew=data['alpha_diff_rew'], alpha_diff_unrew=data['alpha_diff_unrew'], 
                           gamma_alpha_same_rew=data['gamma_alpha_same_rew'], gamma_alpha_same_unrew=data['gamma_alpha_same_unrew'], 
                           gamma_alpha_diff_rew=data['gamma_alpha_diff_rew'], gamma_alpha_diff_unrew=data['gamma_alpha_diff_unrew'], 
                           gamma_lam_same_rew=data['gamma_lam_same_rew'], gamma_lam_same_unrew=data['gamma_lam_same_unrew'], 
                           gamma_lam_diff_rew=data['gamma_lam_diff_rew'], gamma_lam_diff_unrew=data['gamma_lam_diff_unrew'], 
                           k_same_rew=data['k_same_rew'], k_same_unrew=data['k_same_unrew'], 
                           k_diff_rew=data['k_diff_rew'], k_diff_unrew=data['k_diff_unrew'], 
                           shared_side_alphas=data['shared_side_alphas'], shared_outcome_alpha_update=data['shared_outcome_alpha_update'],
                           global_lam=data['global_lam'], constraints=data['constraints'])
    
# %% State Inference Agent

class StateInferenceAgent(ModelAgent):
    """ State Inference Agent from Mishchanchuk et al. 2024

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output: the belief state that the left or right port has higher reward probability. Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, p_stay=None, c_same_rew=None, c_same_unrew=None, c_diff_rew=None, c_diff_unrew=None, complement_c_rew=True, complement_c_diff=True, constraints=None):
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
            
        self.p_stay = AlphaParam(p_stay)
        self.c_same_rew = AlphaParam(c_same_rew)
        self.c_same_unrew = AlphaParam(c_same_unrew)
        self.c_diff_rew = AlphaParam(c_diff_rew)
        self.c_diff_unrew = AlphaParam(c_diff_unrew)
        self.constraints = copy.deepcopy(constraints)
        self.complement_c_rew = complement_c_rew
        self.complement_c_diff = complement_c_diff
        
        self.apply_constraints()
        
        self.reset_state()
        
    def apply_constraints(self):
        # make sure complemented parameters are appropriately shared
        if self.complement_c_rew and self.complement_c_diff:
            self.constraints['c_same_unrew'] = {'share': 'c_same_rew'}
            self.constraints['c_diff_rew'] = {'share': 'c_same_rew'}
            self.constraints['c_diff_unrew'] = {'share': 'c_same_rew'}
            
        elif self.complement_c_rew:
            self.constraints['c_same_unrew'] = {'share': 'c_same_rew'}
            self.constraints['c_diff_unrew'] = {'share': 'c_diff_rew'}
        
        elif self.complement_c_diff:
            self.constraints['c_diff_rew'] = {'share': 'c_same_rew'}
            self.constraints['c_diff_unrew'] = {'share': 'c_same_unrew'}
        
        # apply parameter constraints
        for key, vals in self.constraints.items():
            # reassign parameter to be the same instance of another parameter
            if 'share' in vals:
                setattr(self, key, getattr(self, vals['share']))
            # change whether this parameter is being fit
            if 'fit' in vals:
                param = getattr(self, key)
                param.requires_grad = vals['fit']
            # change whether this parameter is being fit
            if 'init' in vals:
                param = getattr(self, key)
                with torch.no_grad():
                    if isinstance(param, AlphaParam):
                        param.parametrizations.a.right_inverse(torch.tensor([vals['init']]))
                    else:
                        param.copy_(torch.tensor([vals['init']]))
        
        
    def __repr__(self):
         return self.print_params()
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_params(self):
        self.p_stay = AlphaParam()
        self.c_same_rew = AlphaParam()
        self.c_same_unrew = AlphaParam()
        self.c_diff_rew = AlphaParam()
        self.c_diff_unrew = AlphaParam()
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        self.state = torch.tensor([[0.5,0.5]])
        self.state_hist = []
        self.prior_outcome_diff_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
        
        # build the evidence matrix for current trial/outcome
        self.ev_mat = torch.cat([torch.cat([0.5*(1+self.c_same_rew.a), 0.5*(1-self.c_same_unrew.a), 0.5*(1-self.c_diff_rew.a), 0.5*(1+self.c_diff_unrew.a)]).unsqueeze(1),
                                 torch.cat([0.5*(1-self.c_diff_rew.a), 0.5*(1+self.c_diff_unrew.a), 0.5*(1+self.c_same_rew.a), 0.5*(1-self.c_same_unrew.a)]).unsqueeze(1)], axis=1)
            
        # build the transition matrix
        self.trans_mat = torch.cat([torch.cat([0.5*(1+self.p_stay.a), 0.5*(1-self.p_stay.a)]).unsqueeze(0),
                                    torch.cat([0.5*(1-self.p_stay.a), 0.5*(1+self.p_stay.a)]).unsqueeze(0)])
        
    def step(self, input):
        side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                      (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        
        # compute the evidence for each state based on the side/outcome
        evidence_vals = torch.matmul(side_outcome_sel, self.ev_mat)
        
        # compute likelihood of state given current evidence and previous history
        likelihood = evidence_vals*self.state
        like_sum = likelihood.sum(dim=1)
        
        # handle inputs of all 0s
        if torch.any(like_sum == 0):
            likelihood[like_sum == 0,:] = self.state[like_sum == 0,:]
            like_sum[like_sum == 0] = 1
        
        norm_likelihood = likelihood/like_sum.unsqueeze(1)
        
        # compute new state probabilities
        new_state = torch.matmul(norm_likelihood, self.trans_mat)
        
        state_diff = norm_likelihood - self.state
        state_delta = new_state - self.state
        
        self.state = new_state

        # record state histories
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())

        return self.state
    
    def print_params(self):
        return '''State Inference Agent: \n\t c same, rew = {}, fit: {} \n\t c same, unrew = {}, fit: {} \n\t c diff, rew = {}, fit: {} 
\t c diff, unrew = {}, fit: {} \n\t p(block stay) = {}, fit: {}'''.format(
                  self.c_same_rew.to_string(), self.c_same_rew.requires_grad, self.c_same_unrew.to_string(), self.c_same_unrew.requires_grad,
                  self.c_diff_rew.to_string(), self.c_diff_rew.requires_grad, self.c_diff_unrew.to_string(), self.c_diff_unrew.requires_grad,
                  self.p_stay.to_string(), self.p_stay.requires_grad) 

    def toJson(self):
        return super().formatJson({'c_same_rew': self.c_same_rew.a.item(), 'c_same_unrew': self.c_same_unrew.a.item(),
                                   'c_diff_rew': self.c_diff_rew.a.item(), 'c_diff_unrew': self.c_diff_unrew.a.item(),
                                   'p_stay': self.p_stay.a.item(), 'complement_c_rew': self.complement_c_rew,
                                   'complement_c_diff': self.complement_c_diff, 'constraints': self.constraints})
    
    def fromJson(data):
        return StateInferenceAgent(c_same_rew=data['c_same_rew'], c_same_unrew=data['c_same_unrew'], 
                           c_diff_rew=data['c_diff_rew'], c_diff_unrew=data['c_diff_unrew'], 
                           p_stay=data['p_stay'], complement_c_rew=data['complement_c_rew'],
                           complement_c_diff=data['complement_c_diff'], constraints=data['constraints'])
    
    def clone(self):
        return StateInferenceAgent.fromJson(self.toJson()['data'])

# %% Summation Agent

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
        
        # Doing this only works after registering the parametrization
        if not betas is None:
            with torch.no_grad():
                self.beta.weight = torch.tensor(betas).type(torch.float).reshape_as(self.beta.weight)

        # have separate bias to handle scenario with outputs for each choice
        # in this scenario, still only want one bias term
        self.bias = _init_param(bias)
        self.output_layer = output_layer
        
        self.reset_state()
        
    def __repr__(self):
         return self.print_params()
     
    def print_params(self):
        print_str = ''
        for agent in self.agents:
            print_str += agent.print_params() + '\n'
        return print_str + 'Summation Agent: bias = {:.5f}; beta = [{}]'.format(self.bias.item(), ', '.join(['{:.5f}'.format(w.item()) for w in self.beta.weight.view(-1)]))
        
    def forward(self, input): #first col of input-choice, second col of input-outcome
        self.reset_state()
        return self._gen_apply_agents(input, lambda agent, input: agent(input))
    
    def reset_params(self):
        self.beta = nn.Linear(len(self.agents), 1, bias=False)
        parametrize.register_parametrization(self.beta, 'weight', PositiveConstraint())
        self.bias = _init_param()
        
        for model in self.agents:
            model.reset_params()
    
        self.reset_state()
    
    def reset_state(self):
        for model in self.agents:
            model.reset_state()
            
        self.output_hist = [] 
        
    def step(self, input):
        return self._gen_apply_agents(input, lambda agent, input: agent.step(input))
    
    def _gen_apply_agents(self, input, agent_method):
        
        # store outputs from every model's forward method
        output_stack = [] 
        
        # Propogate input through the network
        for agent in self.agents: 
            # store outputs together in lists
            output_stack.append(agent_method(agent, input)) 
        
        # concatenate tensor outputs on the last dimension (parameter output)
        output_stack = torch.stack(output_stack, dim=-1)
        
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
            
        self.output_hist.append(output)

        return output, output_stack.detach()

    def toJson(self):
        return super().formatJson({'betas': self.beta.weight.tolist(), 'bias': self.bias.item(),
                                   'agents': [a.toJson() for a in self.agents]})
    
    def fromJson(data):
        return SummationModule(agents=data['agents'], betas=data['betas'], bias=data['bias'])
        
# %% Custom Save/Load Infrastructure

class ModelJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if the object is of type ModelAgent
        if isinstance(obj, ModelAgent):
            return obj.toJson()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        # For other objects, use the default behavior
        else:
            return super().default(obj)
    
def ModelJsonDecoder(obj):
    if 'type' in obj and 'data' in obj:
        return getattr(globals()[obj['type']], 'fromJson')(obj['data'])
    else:
        return obj
    
def save_model(data, save_path):
    utils.check_make_dir(save_path)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, cls=ModelJsonEncoder)
        
def load_model(save_path):
    with open(save_path, 'r') as f:
        return json.load(f, object_hook=ModelJsonDecoder)
    

# test_module = SummationModule(agents=[SingleValueAgent(0.25), PerseverativeAgent(0.4), FallacyAgent(0.9)], bias=0.5, betas=[0.9, 1.2, 0.33])
# test_dict = {'model': test_module, 'perf': {'acc': 0.8}}
# # test_json = json.dumps(test_dict, cls=ModelJsonEncoder)
# # test_decode = json.loads(test_json, object_hook=ModelJsonDecoder)

# save_path = r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\test.pth'
# save_model(test_dict, save_path)
# test_decode = load_model(save_path)

# print(test_decode['model'].print_params())
