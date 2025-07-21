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
import torch.distributions as dist
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

class UnitParam(nn.Module):
    
    def __init__(self, x = None):
        super().__init__()

        self.a = _init_param(x)
        parametrize.register_parametrization(self, 'a', UnitConstraint())
        
    def __repr__(self):
         return 'Unit Param: {}'.format(self.to_string())
        
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
    
    
class PositiveParam(nn.Module):
    
    def __init__(self, x = None, beta=100, thresh=20):
        super().__init__()

        self.a = _init_param(x)
        parametrize.register_parametrization(self, 'a', PositiveConstraint(beta=beta, thresh=thresh))
        
    def __repr__(self):
         return 'Positive Param: {}'.format(self.to_string())
        
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
    
    
def _init_val(x=None):
    if x is None:
        x = torch.rand(1)
    else:
        # make sure this is a float
        x = torch.tensor([x]).type(torch.float)
        
    return x
    
def _init_param(x=None):

    return nn.Parameter(_init_val(x))

def _apply_constraints(agent):
    # apply parameter constraints
    for key, vals in agent.constraints.items():
        # reassign parameter to be the same instance of another parameter
        if 'share' in vals:
            setattr(agent, key, getattr(agent, vals['share']))
        # change whether this parameter is being fit
        if 'fit' in vals:
            param = getattr(agent, key)
            param.requires_grad = vals['fit']
        # change whether this parameter is being fit
        if 'init' in vals:
            param = getattr(agent, key)
            with torch.no_grad():
                if isinstance(param, UnitParam) or isinstance(param, PositiveParam):
                    param.parametrizations.a.right_inverse(_init_val(vals['init']))
                else:
                    param.copy_(_init_val(vals['init']))
        

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
            
        self.alpha = UnitParam(alpha0)
        self.reset_state()
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        self.alpha = UnitParam()
        self.reset_state()
    
    def reset_state(self):
        self.state = torch.tensor(0)
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 1)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def step(self, input):
        diff = input[:,[0]]*input[:,[1]] - self.state
        delta = self.alpha(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(diff.detach())
        self.state_delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Value Agent: α = {}'.format(self.alpha.to_string())
    
    def toJson(self):
        return super().formatJson({'alpha': self.alpha.a.item()})
    
    def fromJson(data):
        return SingleValueAgent(alpha0=data['alpha'])
    
    def clone(self):
        return SingleValueAgent.fromJson(self.toJson()['data'])


class PerseverativeAgent(ModelAgent):
    """ Perseverative Agent H
    
    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Perseverative Agent H on each trial. Tensor of shape (n_sess, n_trials, 1)
    
    """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha = UnitParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
        
    def __repr__(self):
        return self.print_params()
     
    def reset_params(self):
        self.alpha = UnitParam()
        self.reset_state()
        
    def reset_state(self):
        self.state = torch.zeros(self.n_vals)
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def step(self, input):
        diff = input[:,:self.n_vals] - self.state
        delta = self.alpha(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(diff.detach())
        self.state_delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Perseverative Agent: α = {}'.format(self.alpha.to_string())
    
    def toJson(self):
        return super().formatJson({'alpha': self.alpha.a.item(), 'n_vals': self.n_vals})
    
    def fromJson(data):
        return PerseverativeAgent(alpha0=data['alpha'], n_vals=data['n_vals'])
    
    def clone(self):
        return PerseverativeAgent.fromJson(self.toJson()['data'])
    
    
class FallacyAgent(ModelAgent):
    """ Gambler Fallacy Agent G

    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Gambler Fallacy Agent G on each trial. Tensor of shape (n_sess, n_trials, 1)
   """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha = UnitParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
        
    def __repr__(self):
        return self.print_params()
     
    def reset_params(self):
        self.alpha = UnitParam()
        self.reset_state()
        
    def reset_state(self):
        self.state = torch.zeros(self.n_vals)
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
        
    def step(self, input):
        diff = input[:,:self.n_vals] - input[:,:self.n_vals]*input[:,[self.n_vals]] - self.state
        delta = self.alpha(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(diff.detach())
        self.state_delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Fallacy Agent: α = {}'.format(self.alpha.to_string())
    
    def toJson(self):
        return super().formatJson({'alpha': self.alpha.a.item(), 'n_vals': self.n_vals})
    
    def fromJson(data):
        return FallacyAgent(alpha0=data['alpha'], n_vals=data['n_vals'])
    
    def clone(self):
        return FallacyAgent.fromJson(self.toJson()['data'])

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
            
        self.alpha_same_rew = UnitParam(alpha_same_rew)
        self.alpha_same_unrew = UnitParam(alpha_same_unrew)
        self.alpha_diff_rew = UnitParam(alpha_diff_rew)
        self.alpha_diff_unrew = UnitParam(alpha_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        self.constraints = copy.deepcopy(constraints)
        
        self.apply_constraints()
        
        self.reset_state()
        
    def apply_constraints(self):
        _apply_constraints(self)
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        self.alpha_same_rew = UnitParam()
        self.alpha_same_unrew = UnitParam()
        self.alpha_diff_rew = UnitParam()
        self.alpha_diff_unrew = UnitParam()
        self.k_same_rew = _init_param(1)
        self.k_same_unrew = _init_param(0)
        self.k_diff_rew = _init_param(0)
        self.k_diff_unrew = _init_param(0)
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        # initialize state based on the kappa set-points
        self.state = torch.cat([(self.k_same_rew+self.k_same_unrew)/2, (self.k_same_rew+self.k_same_unrew)/2]).unsqueeze(0)
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
        
        self.k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        self.alphas = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output

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
        
        self.alpha_same_rew = UnitParam(alpha_same_rew)
        self.alpha_same_unrew = UnitParam(alpha_same_unrew)
        self.alpha_diff_rew = UnitParam(alpha_diff_rew)
        self.alpha_diff_unrew = UnitParam(alpha_diff_unrew)
        self.gamma_same_rew = UnitParam(gamma_same_rew)
        self.gamma_same_unrew = UnitParam(gamma_same_unrew)
        self.gamma_diff_rew = UnitParam(gamma_diff_rew)
        self.gamma_diff_unrew = UnitParam(gamma_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        self.inverse_update = inverse_update
        self.global_lam = global_lam
        self.constraints = copy.deepcopy(constraints)
        
        self.apply_constraints()
        
        self.reset_state()
       
    def apply_constraints(self):
        # automatically constrain gammas if the lambda multiplier is a global term for both sides
        if self.global_lam:
            self.constraints['gamma_diff_rew'] = {'share': 'gamma_same_rew'}
            self.constraints['gamma_diff_unrew'] = {'share': 'gamma_same_unrew'}
            
        _apply_constraints(self)
        
    def __repr__(self):
         return self.print_params()
     
    def reset_params(self):
        self.alpha_same_rew = UnitParam()
        self.alpha_same_unrew = UnitParam()
        self.alpha_diff_rew = UnitParam()
        self.alpha_diff_unrew = UnitParam()
        self.gamma_same_rew = UnitParam()
        self.gamma_same_unrew = UnitParam()
        self.gamma_diff_rew = UnitParam()
        self.gamma_diff_unrew = UnitParam()
        self.k_same_rew = _init_param(1)
        self.k_same_unrew = _init_param(0)
        self.k_diff_rew = _init_param(0)
        self.k_diff_unrew = _init_param(0)
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        self.state = torch.cat([[(self.k_same_rew+self.k_same_unrew)/2, (self.k_same_rew+self.k_same_unrew)/2]])
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
        
        
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    
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
        return '''Dynamic Q Agent: \n\t α same, rew = {}, fit: {} \n\t α same, unrew = {}, fit: {} \n\t α diff, rew = {}, fit: {} \n\t α diff, unrew = {}, fit: {} 
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
    
    def clone(self):
        return DynamicQAgent.fromJson(self.toJson()['data'])
    
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
        
        self.alpha_same_rew = UnitParam(alpha_same_rew)
        self.alpha_same_unrew = UnitParam(alpha_same_unrew)
        self.alpha_diff_rew = UnitParam(alpha_diff_rew)
        self.alpha_diff_unrew = UnitParam(alpha_diff_unrew)
        self.gamma_lam_same_rew = UnitParam(gamma_lam_same_rew)
        self.gamma_lam_same_unrew = UnitParam(gamma_lam_same_unrew)
        self.gamma_lam_diff_rew = UnitParam(gamma_lam_diff_rew)
        self.gamma_lam_diff_unrew = UnitParam(gamma_lam_diff_unrew)
        self.gamma_alpha_same_rew = UnitParam(gamma_alpha_same_rew)
        self.gamma_alpha_same_unrew = UnitParam(gamma_alpha_same_unrew)
        self.gamma_alpha_diff_rew = UnitParam(gamma_alpha_diff_rew)
        self.gamma_alpha_diff_unrew = UnitParam(gamma_alpha_diff_unrew)
        self.k_same_rew = _init_param(k_same_rew)
        self.k_same_unrew = _init_param(k_same_unrew)
        self.k_diff_rew = _init_param(k_diff_rew)
        self.k_diff_unrew = _init_param(k_diff_unrew)
        self.global_lam = global_lam
        self.shared_side_alphas = shared_side_alphas
        self.shared_outcome_alpha_update = shared_outcome_alpha_update # whether to update alphas separately based on the outcome
        self.constraints = copy.deepcopy(constraints)
        
        self.apply_constraints()
        
        self.reset_state()

    def apply_constraints(self):
        # automatically constrain alphas & gammas if the alpha multiplier is shared for both sides
        if self.shared_side_alphas:
            self.constraints['alpha_diff_rew'] = {'share': 'alpha_same_rew'}
            self.constraints['alpha_diff_unrew'] = {'share': 'alpha_same_unrew'}
            self.constraints['gamma_alpha_diff_rew'] = {'share': 'gamma_alpha_same_rew'}
            self.constraints['gamma_alpha_diff_unrew'] = {'share': 'gamma_alpha_same_unrew'}
            
        # automatically constrain gammas if the lambda multiplier is a global term for both sides
        if self.global_lam:
            self.constraints['gamma_lam_diff_rew'] = {'share': 'gamma_lam_same_rew'}
            self.constraints['gamma_lam_diff_unrew'] = {'share': 'gamma_lam_same_unrew'}
            
        _apply_constraints(self)
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        self.alpha_same_rew = UnitParam()
        self.alpha_same_unrew = UnitParam()
        self.alpha_diff_rew = UnitParam()
        self.alpha_diff_unrew = UnitParam()
        self.gamma_lam_same_rew = UnitParam()
        self.gamma_lam_same_unrew = UnitParam()
        self.gamma_lam_diff_rew = UnitParam()
        self.gamma_lam_diff_unrew = UnitParam()
        self.gamma_alpha_same_rew = UnitParam()
        self.gamma_alpha_same_unrew = UnitParam()
        self.gamma_alpha_diff_rew = UnitParam()
        self.gamma_alpha_diff_unrew = UnitParam()
        self.k_same_rew = _init_param(1)
        self.k_same_unrew = _init_param(0)
        self.k_diff_rew = _init_param(0)
        self.k_diff_unrew = _init_param(0)
        
        self.apply_constraints()
        self.reset_state()
       
    def reset_state(self):
        self.state = torch.cat([[(self.k_same_rew+self.k_same_unrew)/2, (self.k_same_rew+self.k_same_unrew)/2]])
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

        
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    

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
        return '''Uncertainty Dynamic Q Agent: \n\t α same, rew = {}, fit: {} \n\t α same, unrew = {}, fit: {} \n\t α diff, rew = {}, fit: {} \n\t α diff, unrew = {}, fit: {} 
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
    
    def clone(self):
        return UncertaintyDynamicQAgent.fromJson(self.toJson()['data'])
    
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
            
        self.p_stay = UnitParam(p_stay)
        self.c_same_rew = UnitParam(c_same_rew)
        self.c_same_unrew = UnitParam(c_same_unrew)
        self.c_diff_rew = UnitParam(c_diff_rew)
        self.c_diff_unrew = UnitParam(c_diff_unrew)
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
        
        _apply_constraints(self)
        
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        self.p_stay = UnitParam()
        self.c_same_rew = UnitParam()
        self.c_same_unrew = UnitParam()
        self.c_diff_rew = UnitParam()
        self.c_diff_unrew = UnitParam()
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        self.state = torch.tensor([[0.5,0.5]])
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
        
        # build the evidence matrix for current trial/outcome
        self.ev_mat = torch.cat([torch.cat([0.5*(1+self.c_same_rew.a), 0.5*(1-self.c_same_unrew.a), 0.5*(1-self.c_diff_rew.a), 0.5*(1+self.c_diff_unrew.a)]).unsqueeze(1),
                                 torch.cat([0.5*(1-self.c_diff_rew.a), 0.5*(1+self.c_diff_unrew.a), 0.5*(1+self.c_same_rew.a), 0.5*(1-self.c_same_unrew.a)]).unsqueeze(1)], axis=1)
            
        # build the transition matrix
        self.trans_mat = torch.cat([torch.cat([0.5*(1+self.p_stay.a), 0.5*(1-self.p_stay.a)]).unsqueeze(0),
                                    torch.cat([0.5*(1-self.p_stay.a), 0.5*(1+self.p_stay.a)]).unsqueeze(0)])
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def step(self, input):
        side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                      (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        
        # compute the evidence for each state based on the side/outcome
        evidence_vals = torch.matmul(side_outcome_sel, self.ev_mat)
        
        # compute posterior of state given current evidence and previous history
        posterior = evidence_vals*self.state
        post_sum = posterior.sum(dim=1)
        
        # handle inputs of all 0s to ject kep the same value
        if torch.any(post_sum == 0):
            posterior[post_sum == 0,:] = self.state[post_sum == 0,:]
            post_sum[post_sum == 0] = 1
        
        norm_posterior = posterior/post_sum.unsqueeze(1)
        
        # compute new state probabilities
        new_state = torch.matmul(norm_posterior, self.trans_mat)
        
        # RPE using evidence vals
        # state_diff = evidence_vals - self.state
        
        # RPE using diff between new norm posterior and old
        # state_diff = norm_posterior - self.state
        
        # RPE using a heuristic
        # outcome_heuristic = torch.tensor([[1,0.5,0,0.5],[0,0.5,1,0.5]]).t()
        # outcome_heuristic = torch.matmul(side_outcome_sel, outcome_heuristic)
        # state_diff = outcome_heuristic - self.state
        
        # RPE using a heuristic with evidence vals where RPE is the outcome minus the evidence value if they chose the high belief port and outcome minus 0 if they chose the low belief port
        # this is how Mishchanchuk et al. calculated RPE in their paper
        outcome = input[:,2].unsqueeze(1)
        max_values, _ = torch.max(self.state, dim=1, keepdim=True)
        high_state = (self.state == max_values).int()
        evidence_heuristic = high_state * evidence_vals
        state_diff = outcome - evidence_heuristic
        
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


class RLStateInferenceAgent(ModelAgent):
    """ RL State Inference Hybrid Agent from Qu et al. 2023

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output: the value for each port based on belief that the left or right port has higher reward probability weighted by a running estimate of the reward rate. 
                Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, p_stay=None, c_same_rew=None, c_same_unrew=None, c_diff_rew=None, c_diff_unrew=None, 
                 alpha_w=None, w_high_init=0.99, w_low_init=0.01, complement_c_rew=True, complement_c_diff=True, constraints=None):
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
            
        self.p_stay = UnitParam(p_stay)
        self.c_same_rew = UnitParam(c_same_rew)
        self.c_same_unrew = UnitParam(c_same_unrew)
        self.c_diff_rew = UnitParam(c_diff_rew)
        self.c_diff_unrew = UnitParam(c_diff_unrew)
        self.alpha_w = UnitParam(alpha_w)
        self.w_high_init = UnitParam(w_high_init)
        self.w_low_init = UnitParam(w_low_init)
        
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
        
        _apply_constraints(self)
        
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        self.p_stay = UnitParam()
        self.c_same_rew = UnitParam()
        self.c_same_unrew = UnitParam()
        self.c_diff_rew = UnitParam()
        self.c_diff_unrew = UnitParam()
        self.alpha_w = UnitParam()
        self.w_high_init = UnitParam(0.99)
        self.w_low_init = UnitParam(0.01)
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        self.w = torch.cat([self.w_high_init.a, self.w_low_init.a]).unsqueeze(0)
        self.belief = torch.tensor([[0.5, 0.5]])
        self.state = self._calc_state()
        
        self.w_hist = []
        self.belief_hist = []
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.w_hist.append(self.w.detach())
        self.belief_hist.append(self.belief.detach())
        self.state_hist.append(self.state.detach())
        
        # build the evidence matrix for current trial/outcome
        self.ev_mat = torch.cat([torch.cat([0.5*(1+self.c_same_rew.a), 0.5*(1-self.c_same_unrew.a), 0.5*(1-self.c_diff_rew.a), 0.5*(1+self.c_diff_unrew.a)]).unsqueeze(1),
                                 torch.cat([0.5*(1-self.c_diff_rew.a), 0.5*(1+self.c_diff_unrew.a), 0.5*(1+self.c_same_rew.a), 0.5*(1-self.c_same_unrew.a)]).unsqueeze(1)], axis=1)
            
        # build the transition matrix
        self.trans_mat = torch.cat([torch.cat([0.5*(1+self.p_stay.a), 0.5*(1-self.p_stay.a)]).unsqueeze(0),
                                    torch.cat([0.5*(1-self.p_stay.a), 0.5*(1+self.p_stay.a)]).unsqueeze(0)])
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def step(self, input):
        # compute RPE and update state weights
        # use choose right indicator as indices for state selection since it is 0 for a left choice and 1 for a right choice

        # need to expand the state and belief on the first trial
        if self.state.shape[0] != input.shape[0]:
            self.state = self.state.expand(input.shape[0], -1)
            self.belief = self.belief.expand(input.shape[0], -1)
            
        state_diff = (input[:,2] - self.state[torch.arange(input.shape[0]), input[:,1].type(torch.int)]).unsqueeze(1)
        # select belief values by swapping the left & right inputs so that the chosen side's belief is always first
        w_delta = self.alpha_w.a * state_diff * self.belief.gather(dim=1, index=input[:,[1,0]].type(torch.long))
        self.w = self.w + w_delta
        
        # compute new belief probabilities
        side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                      (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        
        # select the evidence for each state based on the side/outcome
        evidence_vals = torch.matmul(side_outcome_sel, self.ev_mat)
        
        # compute posterior of state given current evidence and previous state
        posterior = evidence_vals*self.belief
        post_sum = posterior.sum(dim=1)
        
        # handle inputs of all 0s to ject kep the same value
        if torch.any(post_sum == 0):
            posterior[post_sum == 0,:] = self.belief[post_sum == 0,:]
            post_sum[post_sum == 0] = 1
        
        norm_posterior = posterior/post_sum.unsqueeze(1)
        
        self.belief = torch.matmul(norm_posterior, self.trans_mat)
        
        # update state estimate
        new_state = self._calc_state()

        state_delta = new_state - self.state
        
        self.state = new_state

        # record state histories
        self.w_hist.append(self.w.detach())
        self.belief_hist.append(self.belief.detach())
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())

        return self.state
    
    def _calc_state(self):
        
        return torch.cat([(self.w*self.belief).sum(dim=1).unsqueeze(1), (self.w*self.belief[:,[1,0]]).sum(dim=1).unsqueeze(1)], dim=1)
    
    def print_params(self):
        return '''RL State Inference Agent: \n\t c same, rew = {}, fit: {} \n\t c same, unrew = {}, fit: {} \n\t c diff, rew = {}, fit: {} 
\t c diff, unrew = {}, fit: {} \n\t p(block stay) = {}, fit: {} \n\t alpha w = {}, fit: {} \n\t w high init = {}, fit: {} \n\t w low init = {}, fit: {}'''.format(
                  self.c_same_rew.to_string(), self.c_same_rew.requires_grad, self.c_same_unrew.to_string(), self.c_same_unrew.requires_grad,
                  self.c_diff_rew.to_string(), self.c_diff_rew.requires_grad, self.c_diff_unrew.to_string(), self.c_diff_unrew.requires_grad,
                  self.p_stay.to_string(), self.p_stay.requires_grad, self.alpha_w.to_string(), self.alpha_w.requires_grad,
                  self.w_high_init.to_string(), self.w_high_init.requires_grad, self.w_low_init.to_string(), self.w_low_init.requires_grad) 

    def toJson(self):
        return super().formatJson({'c_same_rew': self.c_same_rew.a.item(), 'c_same_unrew': self.c_same_unrew.a.item(),
                                   'c_diff_rew': self.c_diff_rew.a.item(), 'c_diff_unrew': self.c_diff_unrew.a.item(),
                                   'p_stay': self.p_stay.a.item(), 'alpha_w': self.alpha_w.a.item(), 'w_high_init': self.w_high_init.a.item(), 
                                   'w_low_init': self.w_low_init.a.item(), 'complement_c_rew': self.complement_c_rew, 
                                   'complement_c_diff': self.complement_c_diff, 'constraints': self.constraints})
    
    def fromJson(data):
        return RLStateInferenceAgent(c_same_rew=data['c_same_rew'], c_same_unrew=data['c_same_unrew'], 
                           c_diff_rew=data['c_diff_rew'], c_diff_unrew=data['c_diff_unrew'], 
                           p_stay=data['p_stay'], alpha_w=data['alpha_w'], w_high_init=data['w_high_init'],
                           w_low_init=data['w_low_init'], complement_c_rew=data['complement_c_rew'],
                           complement_c_diff=data['complement_c_diff'], constraints=data['constraints'])
    
    def clone(self):
        return RLStateInferenceAgent.fromJson(self.toJson()['data'])
    
# %% Hybrid Q and SI Agent
    
class QValueStateInferenceAgent(ModelAgent):
    """ Hybrid Q-value and State Inference Agent inspired by Qu et al. 2023
        The values of the high and low port are maintained via Q-learning while the high state probabilites for each side are updated in a bayesian manner using the maintained state values

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output: the value for each port based on belief that the left or right port has higher reward probability weighted by a running estimate of the reward rate. 
                Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, alpha_high_rew=None, alpha_high_unrew=None, alpha_low_rew=None, alpha_low_unrew=None, 
                 k_high_rew=0.99, k_high_unrew=None, k_low_rew=None, k_low_unrew=0.01, p_stay=None, constraints=None, update_order='simultaneous'):
        
        # note K's are set slightly off 1 and 0 to help with gradients
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
        
        self.alpha_high_rew = UnitParam(alpha_high_rew)
        self.alpha_high_unrew = UnitParam(alpha_high_unrew)
        self.alpha_low_rew = UnitParam(alpha_low_rew)
        self.alpha_low_unrew = UnitParam(alpha_low_unrew)
        # make k values alpha params so they are constrained from 0 to 1 since they define reward probabilities
        self.k_high_rew = UnitParam(k_high_rew)
        self.k_high_unrew = UnitParam(k_high_unrew)
        self.k_low_rew = UnitParam(k_low_rew)
        self.k_low_unrew = UnitParam(k_low_unrew)
        self.p_stay = UnitParam(p_stay)
        self.update_order = update_order
        
        self.constraints = copy.deepcopy(constraints)

        self.apply_constraints()
        
        self.reset_state()
        
    def apply_constraints(self):
        _apply_constraints(self)
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        self.alpha_high_rew = UnitParam()
        self.alpha_high_unrew = UnitParam()
        self.alpha_low_rew = UnitParam()
        self.alpha_low_unrew = UnitParam()
        self.k_high_rew = UnitParam(0.99)
        self.k_high_unrew = UnitParam()
        self.k_low_rew = UnitParam()
        self.k_low_unrew = UnitParam(0.01)
        self.p_stay = UnitParam()
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self):
        self.v = torch.cat([self.k_high_rew.a, self.k_low_unrew.a]).unsqueeze(0)
        self.belief = torch.tensor([[0.5, 0.5]])
        self.state = self._calc_state()
        
        self.v_hist = []
        self.belief_hist = []
        self.state_hist = []
        self.state_diff_hist = []
        self.v_diff_hist = []
        self.v_diff_belief_hist = []
        self.v_hist.append(self.v.detach())
        self.belief_hist.append(self.belief.detach())
        self.state_hist.append(self.state.detach())

        # build the alpha and k value matrices for q-learning
        # have unreward first so we can index appropriately by the reward column
        self.k_vals = torch.cat([torch.cat([self.k_high_unrew.a, self.k_low_unrew.a]).unsqueeze(0),
                                 torch.cat([self.k_high_rew.a, self.k_low_rew.a]).unsqueeze(0)])
        self.alphas = torch.cat([torch.cat([self.alpha_high_unrew.a, self.alpha_low_unrew.a]).unsqueeze(0),
                                 torch.cat([self.alpha_high_rew.a, self.alpha_low_rew.a]).unsqueeze(0)])

        # build the transition matrix
        self.trans_mat = torch.cat([torch.cat([0.5*(1+self.p_stay.a), 0.5*(1-self.p_stay.a)]).unsqueeze(0),
                                    torch.cat([0.5*(1-self.p_stay.a), 0.5*(1+self.p_stay.a)]).unsqueeze(0)])
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def step(self, input):
        # compute RPE and update state weights

        # need to expand the state and belief on the first trial
        if self.state.shape[0] != input.shape[0]:
            self.v = self.v.expand(input.shape[0], -1)
            self.belief = self.belief.expand(input.shape[0], -1)
            self.state = self.state.expand(input.shape[0], -1)
        
        # calc RPE based on current state value estimate
        # use choose right indicator as indices for state selection since it is 0 for a left choice and 1 for a right choice
        state_diff = input[:,2].unsqueeze(1) - self.state.gather(dim=1, index=input[:,1].type(torch.long).unsqueeze(1))
        
        # arrange the beliefs so the chosen side is first
        # select belief values by swapping the left & right inputs to make an index so that the chosen side's belief is always first
        chosen_side_idxs = input[:,[1,0]].type(torch.long)
        
        if self.update_order == 'simultaneous':
            chosen_belief = self.belief.gather(dim=1, index=chosen_side_idxs)
            
            rew_idxs = input[:,2].type(torch.long).unsqueeze(1).expand(-1,2)
            # update the state values. First element is high value, second is low value
            v_diff = self.k_vals.gather(dim=0, index=rew_idxs) - self.v
            v_diff_belief = v_diff * chosen_belief
            new_v = self.v + self.alphas.gather(dim=0, index=rew_idxs) * v_diff_belief

            # build the value tensor for current trial/outcome
            # shape is n_sessx2x2 where second dim is high/low value states and third dim is unrewarded/rewarded
            prior_tensor = torch.cat([1-self.v.unsqueeze(2), self.v.unsqueeze(2)], axis=2)
            # select the appropriate priors given the outcome
            prior_vals = prior_tensor.gather(dim=2, index=rew_idxs.unsqueeze(2)).squeeze(2)    
    
            # compute posterior of state given current prior and previous state estimation
            posterior = prior_vals * chosen_belief
            post_sum = posterior.sum(dim=1)
            
            # handle inputs of all 0s to just keep the same belief value
            if torch.any(post_sum == 0):
                posterior[post_sum == 0,:] = chosen_belief[post_sum == 0,:]
                post_sum[post_sum == 0] = 1
            
            norm_posterior = posterior/post_sum.unsqueeze(1)
            
            # compute beliefs and transfer the columns back from chosen order to side order
            self.belief = torch.matmul(norm_posterior, self.trans_mat).gather(dim=1, index=chosen_side_idxs)
            self.v = new_v
            
        elif self.update_order == 'value_first':
            chosen_belief = self.belief.gather(dim=1, index=chosen_side_idxs)
            
            rew_idxs = input[:,2].type(torch.long).unsqueeze(1).expand(-1,2)
            # update the state values. First element is high value, second is low value
            v_diff = self.k_vals.gather(dim=0, index=rew_idxs) - self.v
            v_diff_belief = v_diff * chosen_belief
            self.v = self.v + self.alphas.gather(dim=0, index=rew_idxs) * v_diff_belief

            # build the value tensor for current trial/outcome
            # shape is n_sessx2x2 where second dim is high/low value states and third dim is unrewarded/rewarded
            prior_tensor = torch.cat([1-self.v.unsqueeze(2), self.v.unsqueeze(2)], axis=2)
            # select the appropriate priors given the outcome
            prior_vals = prior_tensor.gather(dim=2, index=rew_idxs.unsqueeze(2)).squeeze(2)    
    
            # compute posterior of state given current prior and previous state estimation
            posterior = prior_vals * chosen_belief
            post_sum = posterior.sum(dim=1)
            
            # handle inputs of all 0s to just keep the same belief value
            if torch.any(post_sum == 0):
                posterior[post_sum == 0,:] = chosen_belief[post_sum == 0,:]
                post_sum[post_sum == 0] = 1
            
            norm_posterior = posterior/post_sum.unsqueeze(1)
            
            # compute beliefs and transfer the columns back from chosen order to side order
            self.belief = torch.matmul(norm_posterior, self.trans_mat).gather(dim=1, index=chosen_side_idxs)
            
        elif self.update_order == 'belief_first':

            rew_idxs = input[:,2].type(torch.long).unsqueeze(1).expand(-1,2)
            
            # build the value tensor for current trial/outcome
            # shape is n_sessx2x2 where second dim is high/low value states and third dim is unrewarded/rewarded
            prior_tensor = torch.cat([1-self.v.unsqueeze(2), self.v.unsqueeze(2)], axis=2)
            # select the appropriate priors given the outcome
            prior_vals = prior_tensor.gather(dim=2, index=rew_idxs.unsqueeze(2)).squeeze(2)    
    
            # compute posterior of state given current prior and previous state estimation
            chosen_belief = self.belief.gather(dim=1, index=chosen_side_idxs)
            posterior = prior_vals * chosen_belief
            post_sum = posterior.sum(dim=1)
            
            # handle inputs of all 0s to just keep the same belief value
            if torch.any(post_sum == 0):
                posterior[post_sum == 0,:] = chosen_belief[post_sum == 0,:]
                post_sum[post_sum == 0] = 1
            
            norm_posterior = posterior/post_sum.unsqueeze(1)
            
            # compute beliefs and transfer the columns back from chosen order to side order
            chosen_belief = torch.matmul(norm_posterior, self.trans_mat)
            self.belief = chosen_belief.gather(dim=1, index=chosen_side_idxs)

            # update the state values. First element is high value, second is low value
            v_diff = self.k_vals.gather(dim=0, index=rew_idxs) - self.v
            v_diff_belief = v_diff * chosen_belief
            self.v = self.v + self.alphas.gather(dim=0, index=rew_idxs) * v_diff_belief

        # update state estimate
        self.state = self._calc_state()

        # record state histories
        self.v_hist.append(self.v.detach())
        self.belief_hist.append(self.belief.detach())
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.v_diff_hist.append(v_diff.detach())
        self.v_diff_belief_hist.append(v_diff_belief.detach())

        return self.state
    
    def _calc_state(self):
        
        return torch.cat([(self.v*self.belief).sum(dim=1).unsqueeze(1), (self.v*self.belief[:,[1,0]]).sum(dim=1).unsqueeze(1)], dim=1)

    def print_params(self):
        return '''Q-Value State Inference Agent: \n\t alpha high rew = {}, fit: {} \n\t alpha high unrew = {}, fit: {} 
\t alpha low rew = {}, fit: {} \n\t alpha low unrew = {}, fit: {} \n\t k high rew = {}, fit: {} \n\t k high unrew = {}, fit: {} 
\t k low rew = {}, fit: {} \n\t k low unrew = {}, fit: {} \n\t p(block stay) = {}, fit: {}'''.format(
                  self.alpha_high_rew.to_string(), self.alpha_high_rew.requires_grad, self.alpha_high_unrew.to_string(), self.alpha_high_unrew.requires_grad,
                  self.alpha_low_rew.to_string(), self.alpha_low_rew.requires_grad, self.alpha_low_unrew.to_string(), self.alpha_low_unrew.requires_grad,
                  self.k_high_rew.to_string(), self.k_high_rew.requires_grad, self.k_high_unrew.to_string(), self.k_high_unrew.requires_grad,
                  self.k_low_rew.to_string(), self.k_low_rew.requires_grad, self.k_low_unrew.to_string(), self.k_low_unrew.requires_grad,
                  self.p_stay.to_string(), self.p_stay.requires_grad) 

    def toJson(self):
        return super().formatJson({'alpha_high_rew': self.alpha_high_rew.a.item(), 'alpha_high_unrew': self.alpha_high_unrew.a.item(),
                                   'alpha_low_rew': self.alpha_low_rew.a.item(), 'alpha_low_unrew': self.alpha_low_unrew.a.item(),
                                   'k_high_rew': self.k_high_rew.a.item(), 'k_high_unrew': self.k_high_unrew.a.item(),
                                   'k_low_rew': self.k_low_rew.a.item(), 'k_low_unrew': self.k_low_unrew.a.item(),
                                   'p_stay': self.p_stay.a.item(), 'update_order': self.update_order, 'constraints': self.constraints})
    
    def fromJson(data):
        return QValueStateInferenceAgent(alpha_high_rew=data['alpha_high_rew'], alpha_high_unrew=data['alpha_high_unrew'], 
                                         alpha_low_rew=data['alpha_low_rew'], alpha_low_unrew=data['alpha_low_unrew'], 
                                         k_high_rew=data['k_high_rew'], k_high_unrew=data['k_high_unrew'], 
                                         k_low_rew=data['k_low_rew'], k_low_unrew=data['k_low_unrew'], 
                                         p_stay=data['p_stay'], update_order=data['update_order'], constraints=data['constraints'])
    
    def clone(self):
        return QValueStateInferenceAgent.fromJson(self.toJson()['data'])
    
# %% Full Bayesian Agent

def normalize(x, dim=None):
    return x/x.sum(dim=dim, keepdim=True)

def norm_pdf(mu, sig, vals, min_val=1e-10):
    return torch.clamp(torch.exp(-0.5*((vals - mu)/sig)**2), min=min_val)

# transform the sigma variable bounded by 1 and 0 to a value between 2 and 0.01 distributed evenly in logarithmic space
def sig_transform(x, sig_max=2, sig_min=0.02):
    return torch.exp(torch.log(torch.tensor(sig_max))*x+torch.log(torch.tensor(sig_min))*(1-x))

class BayesianAgent(ModelAgent):
    """ Full Bayesian Agent inspired by Boorman et al. 2016 and Witkowski et al. 2022
        The reward probabilities for each side are maintained as a probability mass over all possible discretized reward probabilities
        and the switch probability is maintained as a separate probability mass over all possible switch probabilities

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output: the value for each port based on the reward belief probability mass distributions 
                Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, p_step=0.01, init_high_rew_mean=None, init_low_rew_mean=None, init_switch_mean=None, init_rew_sig=None, init_switch_sig=None, 
                 stay_bias_lam=None, outcome_inference_lam=None, unreward_inference_lam=None, imperfect_update_alpha=None, 
                 forget_alpha=None, switch_scatter_sig=None, update_p_switch_first=False, constraints=None):
        
        # note initial means are slightly off 1/0 to help with gradient calculation
        
        super().__init__()
        
        if constraints is None:
            constraints = {}
        
        self.p_step = p_step
        
        self.init_high_rew_mean = UnitParam(init_high_rew_mean)
        self.init_low_rew_mean = UnitParam(init_low_rew_mean)
        self.init_rew_sig = UnitParam(init_rew_sig)
        
        self.init_switch_mean = UnitParam(init_switch_mean)
        self.init_switch_sig = UnitParam(init_switch_sig)
        
        self.switch_scatter_sig = UnitParam(switch_scatter_sig)
        
        self.stay_bias_lam = UnitParam(stay_bias_lam)
        self.outcome_inference_lam = UnitParam(outcome_inference_lam)
        self.unreward_inference_lam = UnitParam(unreward_inference_lam)
        
        self.imperfect_update_alpha = UnitParam(imperfect_update_alpha)
        self.forget_alpha = UnitParam(forget_alpha)
        
        self.update_p_switch_first = update_p_switch_first
    
        self.constraints = copy.deepcopy(constraints)

        self.apply_constraints()
        
        self.reset_state()
        
    def apply_constraints(self):
        _apply_constraints(self)
        
    def __repr__(self):
        return self.print_params()
    
    def reset_params(self):
        
        self.init_rew_mean_sep = UnitParam()
        self.init_switch_mean = UnitParam()
        self.init_rew_sig = UnitParam()
        self.init_switch_sig = UnitParam()
        
        self.switch_scatter_sig = UnitParam()
        
        self.stay_bias_lam = UnitParam()
        self.outcome_inference_lam = UnitParam()
        self.unreward_inference_lam = UnitParam()
        
        self.imperfect_update_alpha = UnitParam()
        self.forget_alpha = UnitParam()
        
        self.apply_constraints()
        self.reset_state()
    
    def reset_state(self, n_sess=1):
        
        self.prob_vals = torch.arange(self.p_step, 1, self.p_step)
        self.n_p = len(self.prob_vals)
        
        # precompute probabilities for rewards/unrewards, stays/switches, and reward bin transition probabilities
        
        # build the outcome probabilities
        self.p_outcome_chosen = torch.empty((self.n_p, 2))
        # have unrewarded first for ease of indexing
        self.p_outcome_chosen[:,0] = 0.5*(1-self.unreward_inference_lam.a) + (1-self.prob_vals)*self.unreward_inference_lam.a
        self.p_outcome_chosen[:,1] = self.prob_vals
        
        self.p_outcome_unchosen = 0.5*(1-self.outcome_inference_lam.a) + (1-self.p_outcome_chosen)*self.outcome_inference_lam.a

        # build the stay/switch probabilities
        # apply the bias term here. Doesn't matter that it is not normalized because normalization happens in the update equations
        self.p_stay = (1+self.stay_bias_lam.a)*(1-self.prob_vals)
        self.p_switch = (1-self.stay_bias_lam.a)*self.prob_vals
        
        # reshape to facilitate batched computation
        self.p_stay = self.p_stay.reshape(1,1,-1)
        self.p_switch = self.p_switch.reshape(1,1,-1)
        
        # build the transition probabilities P(Pr_t | Pr_t-1, Switch) 
        # will model this distance as a gaussian: normal(Pr_t, mu=Pr_t-1, sig=switch_scatter_sig)
        if np.isclose(self.switch_scatter_sig.a, 0):
            self.p_rew_transition = torch.eye(len(self.prob_vals))
        else:
            self.p_rew_transition = norm_pdf(self.prob_vals.unsqueeze(0), sig_transform(self.switch_scatter_sig.a), self.prob_vals.unsqueeze(1))
        
            # normalize this matrix so that the sum of P(Pr_t | Pr_t-1) over all Pr_t-1 is 1
            self.p_rew_transition = normalize(self.p_rew_transition, dim=1)
            
        # expand to facilitate batched computation
        self.p_rew_transition = self.p_rew_transition.unsqueeze(0)

        # build the initial distributions
        rew_sig = sig_transform(self.init_rew_sig.a)
        side_prior = (norm_pdf(0.5*(1+self.init_high_rew_mean.a), rew_sig, self.prob_vals) + 
                      norm_pdf(0.5*self.init_low_rew_mean.a, rew_sig, self.prob_vals))
        self.init_side_prior = torch.tile(normalize(side_prior).unsqueeze(0), (2,1))
        
        # have switch mean be maximum at 0.5
        switch_prior = norm_pdf(0.5*self.init_switch_mean.a, sig_transform(self.init_switch_sig.a), self.prob_vals)
        self.init_switch_prior = normalize(switch_prior)
        
        # initialize current trial values and build structures to store values over trials
        self.side_prior = self.init_side_prior.unsqueeze(0).expand(n_sess, -1, -1)
        self.switch_prior = self.init_switch_prior.unsqueeze(0).expand(n_sess, -1)
        self.state = self._calc_state()
        
        self.side_prior_hist = []
        self.side_posterior_hist = []
        self.switch_prior_hist = []
        self.switch_posterior_hist = []
        self.state_hist = []
        self.state_diff_hist = []
        self.nll_hist_full = []
        self.nll_hist_stay = []

        self.side_prior_hist.append(self.side_prior.detach())
        self.switch_prior_hist.append(self.switch_prior.detach())
        self.state_hist.append(self.state.detach())
       
    def forward(self, input):
        self.reset_state(input.shape[0])
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def step(self, input):
        
        chosen_side_idxs = input[:,1].type(torch.long).unsqueeze(1).unsqueeze(2).expand(-1,-1,self.n_p)
        unchosen_side_idxs = input[:,0].type(torch.long).unsqueeze(1).unsqueeze(2).expand(-1,-1,self.n_p)
        n_sess = input.shape[0]
        
        chosen_prior = self.side_prior.gather(dim=1, index=chosen_side_idxs).squeeze(1)
        unchosen_prior = self.side_prior.gather(dim=1, index=unchosen_side_idxs).squeeze(1)

        # get all the outcome probabilities for the chosen and unchosen sides across all sessions
        outcome_idxs = input[:,2].type(torch.long).unsqueeze(1).expand(-1,self.n_p).unsqueeze(2)
        p_o_chosen = self.p_outcome_chosen.unsqueeze(0).expand(n_sess,-1,-1).gather(dim=2, index=outcome_idxs)
        p_o_unchosen = self.p_outcome_unchosen.unsqueeze(0).expand(n_sess,-1,-1).gather(dim=2, index=outcome_idxs)
        
        if self.update_p_switch_first:
            # compute the joint outcome likelihood distributions for each reward and switch probability based on the trial outcome for chosen and unchosen sides
            chosen_joint_outcome_pr_ps = p_o_chosen*(chosen_prior.unsqueeze(2)*self.p_stay + torch.sum(unchosen_prior.unsqueeze(1)*self.p_rew_transition, dim=2, keepdim=True)*self.p_switch)
            unchosen_joint_outcome_pr_ps = p_o_unchosen*(unchosen_prior.unsqueeze(2)*self.p_stay + torch.sum(chosen_prior.unsqueeze(1)*self.p_rew_transition, dim=2, keepdim=True)*self.p_switch)
    
            # now update switch posterior
            switch_prior_expand = self.switch_prior.unsqueeze(1)
            switch_posterior = normalize(switch_prior_expand*chosen_joint_outcome_pr_ps.sum(dim=1, keepdim=True), dim=2)
            
            # perform imperfect update of switch posterior
            imperfect_switch_posterior = normalize((1-self.imperfect_update_alpha.a)*switch_posterior + self.imperfect_update_alpha.a*switch_prior_expand, dim=2)

            # Compute the side reward posteriors
            unnorm_chosen_posterior = torch.sum(imperfect_switch_posterior*chosen_joint_outcome_pr_ps, dim=2)
            unnorm_unchosen_posterior = torch.sum(imperfect_switch_posterior*unchosen_joint_outcome_pr_ps, dim=2)
        else:
            # make this more efficient by performing switch prior multiplications only once
            
            switch_prior_expand = self.switch_prior.unsqueeze(1)
            
            # compute the joint outcome likelihood distributions for each reward and switch probability based on the trial outcome for chosen and unchosen sides
            chosen_joint_outcome_pr_ps = switch_prior_expand * p_o_chosen * (chosen_prior.unsqueeze(2)*self.p_stay + torch.sum(unchosen_prior.unsqueeze(1)*self.p_rew_transition, dim=2, keepdim=True)*self.p_switch)
            unchosen_joint_outcome_pr_ps = switch_prior_expand * p_o_unchosen * (unchosen_prior.unsqueeze(2)*self.p_stay + torch.sum(chosen_prior.unsqueeze(1)*self.p_rew_transition, dim=2, keepdim=True)*self.p_switch)
    
            # update switch posterior
            switch_posterior = normalize(chosen_joint_outcome_pr_ps.sum(dim=1, keepdim=True), dim=2)
            
            # perform imperfect update of switch posterior
            imperfect_switch_posterior = normalize((1-self.imperfect_update_alpha.a)*switch_posterior + self.imperfect_update_alpha.a*switch_prior_expand, dim=2)

            # Compute the side reward posteriors
            unnorm_chosen_posterior = torch.sum(chosen_joint_outcome_pr_ps, dim=2)
            unnorm_unchosen_posterior = torch.sum(unchosen_joint_outcome_pr_ps, dim=2)
        
        chosen_posterior = normalize(unnorm_chosen_posterior, dim=1)
        unchosen_posterior = normalize(unnorm_unchosen_posterior, dim=1)
        
        # regroup the posteriors
        side_posterior = torch.cat([chosen_posterior.unsqueeze(1), unchosen_posterior.unsqueeze(1)], axis=1)
        
        # switch the order back into left/right side
        gather_idxs = torch.cat([chosen_side_idxs, unchosen_side_idxs], dim=1)
        side_posterior = side_posterior.gather(dim=1, index=gather_idxs)
        
        # perform imperfect update of side posterior
        imperfect_side_posterior = normalize((1-self.imperfect_update_alpha.a)*side_posterior + self.imperfect_update_alpha.a*self.side_prior, dim=2)
        
        # perform forgetting of updated posteriors to make the new priors for the next choice
        new_switch_prior = normalize((1-self.forget_alpha.a)*imperfect_switch_posterior.squeeze(1) + self.forget_alpha.a*self.init_switch_prior, dim=1)
        new_side_prior = normalize((1-self.forget_alpha.a)*imperfect_side_posterior + self.forget_alpha.a*self.init_side_prior, dim=2)

        # calculate point estimate RPE and Bayesian Suprise (neg log likelihood)
        state_diff = input[:,2].unsqueeze(1) - self.state
        
        # chosen_llh = switch_prior_expand * p_o_chosen * chosen_prior.unsqueeze(2) * self.p_stay
        # unchosen_llh = switch_prior_expand * p_o_chosen * torch.sum(unchosen_prior.unsqueeze(1)*self.p_rew_transition, dim=2, keepdim=True) * self.p_switch
        # nll_full = torch.cat([-torch.log(chosen_llh.sum(dim=[1,2]).unsqueeze(-1)), -torch.log(unchosen_llh.sum(dim=[1,2]).unsqueeze(-1))], axis=1)
        # nll_gather_idxs = torch.cat([input[:,1].type(torch.long).unsqueeze(1), input[:,0].type(torch.long).unsqueeze(1)], dim=1)
        # nll_full = nll_full.gather(dim=1, index=nll_gather_idxs)
        
        # Calculate full neg log likelihood including switch probabilities
        nll_full = torch.cat([-torch.log(unnorm_chosen_posterior.sum(dim=1, keepdim=True)), -torch.log(unnorm_unchosen_posterior.sum(dim=1, keepdim=True))], axis=1)
        nll_gather_idxs = torch.cat([input[:,1].type(torch.long).unsqueeze(1), input[:,0].type(torch.long).unsqueeze(1)], dim=1)
        nll_full = nll_full.gather(dim=1, index=nll_gather_idxs)
        # calculate neg log likelihood only considering the current side's prior distribution (ignoring any likelihood of switching)
        nll_stay = torch.cat([-torch.log(torch.sum(p_o_chosen.squeeze(2)*chosen_prior, dim=1, keepdim=True)), 
                              -torch.log(torch.sum(p_o_unchosen.squeeze(2)*unchosen_prior, dim=1, keepdim=True))], axis=1)
        nll_stay = nll_stay.gather(dim=1, index=nll_gather_idxs)
        
        # update the state
        self.switch_prior = new_switch_prior
        self.side_prior = new_side_prior
        self.state = self._calc_state()
        
        # update histories
        self.switch_posterior_hist.append(imperfect_switch_posterior.squeeze(1).detach())
        self.switch_prior_hist.append(self.switch_prior.detach())
        self.side_posterior_hist.append(imperfect_side_posterior.detach())
        self.side_prior_hist.append(self.side_prior.detach())
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.nll_hist_full.append(nll_full.detach())
        self.nll_hist_stay.append(nll_stay.detach())
        
        return self.state
    
    def _calc_state(self):
        return torch.matmul(self.side_prior, self.prob_vals)
    
    def get_entropy(self, p_dist='reward', epsilon=1e-10):
        ''' Calculates the KL divergence between distributions p and q '''
        
        if p_dist == 'reward':
            prior = torch.stack(self.side_prior_hist[:-1], dim=1).numpy()
        else:
            prior = torch.stack(self.switch_prior_hist[:-1], dim=1).numpy()
        
        ent_mat = prior * np.log(prior + epsilon)
        
        return -np.sum(ent_mat, axis=-1)

    def get_kl_divergence(self, p_dist='reward', epsilon=1e-10):
        ''' Calculates the KL divergence between prior and posterior '''
        
        if p_dist == 'reward':
            prior = torch.stack(self.side_prior_hist[:-1], dim=1).numpy()
            posterior = torch.stack(self.side_posterior_hist, dim=1).numpy()
        else:
            prior = torch.stack(self.switch_prior_hist[:-1], dim=1).numpy()
            posterior = torch.stack(self.switch_posterior_hist, dim=1).numpy()
        
        kl_mat = posterior * (np.log(posterior + epsilon) - np.log(prior + epsilon))
        
        return np.sum(kl_mat, axis=-1)

    def print_params(self):
        return '''Full Bayesian Inference Agent: \n\t init high rew μ = {}, fit: {} \n\t init low rew μ = {}, fit: {} 
\t init rew σ = {}, fit: {} \n\t init switch μ = {}, fit: {} \n\t init switch σ = {}, fit: {} 
\t stay bias λ = {}, fit: {} \n\t outcome inference λ = {}, fit: {} \n\t unrewarded inference λ = {}, fit: {}
\t imperfect update α = {}, fit: {} \n\t forget α = {}, fit: {} \n\t switch scatter σ = {}, fit: {}'''.format(
                  self.init_high_rew_mean.to_string(), self.init_high_rew_mean.requires_grad, self.init_low_rew_mean.to_string(), self.init_low_rew_mean.requires_grad, 
                  self.init_rew_sig.to_string(), self.init_rew_sig.requires_grad, self.init_switch_mean.to_string(), self.init_switch_mean.requires_grad, 
                  self.init_switch_sig.to_string(), self.init_switch_sig.requires_grad, self.stay_bias_lam.to_string(), self.stay_bias_lam.requires_grad, 
                  self.outcome_inference_lam.to_string(), self.outcome_inference_lam.requires_grad, self.unreward_inference_lam.to_string(), self.unreward_inference_lam.requires_grad, 
                  self.imperfect_update_alpha.to_string(), self.imperfect_update_alpha.requires_grad, self.forget_alpha.to_string(), self.forget_alpha.requires_grad, 
                  self.switch_scatter_sig.to_string(), self.switch_scatter_sig.requires_grad) 

    def toJson(self):
        return super().formatJson({'p_step': self.p_step, 'init_high_rew_mean': self.init_high_rew_mean.a.item(),
                                   'init_low_rew_mean': self.init_low_rew_mean.a.item(), 'init_rew_sig': self.init_rew_sig.a.item(), 
                                   'init_switch_mean': self.init_switch_mean.a.item(), 'init_switch_sig': self.init_switch_sig.a.item(), 
                                   'switch_scatter_sig': self.switch_scatter_sig.a.item(), 'stay_bias_lam': self.stay_bias_lam.a.item(), 
                                   'outcome_inference_lam': self.outcome_inference_lam.a.item(), 'unreward_inference_lam': self.unreward_inference_lam.a.item(), 
                                   'imperfect_update_alpha': self.imperfect_update_alpha.a.item(), 'forget_alpha': self.forget_alpha.a.item(), 
                                   'update_p_switch_first': self.update_p_switch_first, 'constraints': self.constraints})
    
    def fromJson(data):
        return BayesianAgent(p_step=data['p_step'], init_high_rew_mean=data['init_high_rew_mean'], init_low_rew_mean=data['init_low_rew_mean'], 
                             init_rew_sig=data['init_rew_sig'], init_switch_mean=data['init_switch_mean'], init_switch_sig=data['init_switch_sig'], 
                             switch_scatter_sig=data['switch_scatter_sig'], stay_bias_lam=data['stay_bias_lam'], 
                             outcome_inference_lam=data['outcome_inference_lam'], unreward_inference_lam=data['unreward_inference_lam'], 
                             imperfect_update_alpha=data['imperfect_update_alpha'], forget_alpha=data['forget_alpha'], 
                             update_p_switch_first=data['update_p_switch_first'], constraints=data['constraints'])
    
    def clone(self):
        return BayesianAgent.fromJson(self.toJson()['data'])

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
        # cache any parameterized values (e.g. UnitParam) so that they don't have to be recomputed every time they are accessed
        with parametrize.cached():
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
        # load agents first from SaveModelAgents
        agents = []
        for save_model in data['agents']:
            agents.append(save_model.model)
        return SummationModule(agents=agents, betas=data['betas'], bias=data['bias'])
    
    def clone(self):
        return SummationModule.fromJson(json.loads(json.dumps(self.toJson()['data']), object_hook=ModelJsonDecoder))
        
# %% Custom Save/Load Infrastructure

# define save model object to streamline saving and loading dictionaries without fully instantiating agent classes
class SaveModelAgent():
    def __init__(self, model_data):
        self.data = model_data
        self.is_loaded = False
        self._model = None

    def _load(self):
        self._model = getattr(globals()[self.data['type']], 'fromJson')(self.data['data'])
        self.is_loaded = True
    
    @property
    def model(self):
        if not self.is_loaded:
            self._load()
            
        return self._model

class ModelJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Check if the object is of type ModelAgent
        if isinstance(obj, SaveModelAgent):
            return obj.data
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
        return SaveModelAgent(obj)
    else:
        return obj
    
def save_model(data, save_path):
    utils.check_make_dir(save_path)
    
    with open(save_path, 'w') as f:
        json.dump(data, f, cls=ModelJsonEncoder)
        
def load_model(save_path):
    
    with open(save_path, 'r') as f:
        data = json.load(f, object_hook=ModelJsonDecoder)
        
    return data
    

# test_module = SummationModule(agents=[SingleValueAgent(0.25), PerseverativeAgent(0.4), FallacyAgent(0.9)], bias=0.5, betas=[0.9, 1.2, 0.33])
# test_dict = {'model': test_module, 'perf': {'acc': 0.8}}
# # test_json = json.dumps(test_dict, cls=ModelJsonEncoder)
# # test_decode = json.loads(test_json, object_hook=ModelJsonDecoder)

# save_path = r'C:\Users\tanne\repos\python\selective_wm_analysis\RL Analysis\test.pth'
# save_model(test_dict, save_path)
# test_decode = load_model(save_path)

# print(test_decode['model'].print_params())
