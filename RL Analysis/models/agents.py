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
        x = torch.tensor([x]).type(torch.float)
        
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
        output: the output value of the Reward-Seeking/Value Agent V on each trial. Tensor of shape (n_sess, n_trials, 1)
    """

    def __init__(self, alpha0 = None):
        super().__init__()
            
        self.alpha_v = AlphaParam(alpha0)
        self.reset_state()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 1)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_state(self):
        self.state = torch.tensor(0)
        self.state_hist = []
        self.diff_hist = []
        self.delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        diff = input[:,[0]]*input[:,[1]] - self.state
        delta = self.alpha_v(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.diff_hist.append(diff.detach())
        self.delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Value Agent: α = {}'.format(self.alpha_v.to_string()) 


class PerseverativeAgent(ModelAgent):
    """ Perseverative Agent H
    
    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Perseverative Agent H on each trial. Tensor of shape (n_sess, n_trials, 1)
    
    """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha_h = AlphaParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_state(self):
        self.state = torch.zeros(self.n_vals)
        self.state_hist = []
        self.diff_hist = []
        self.delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        diff = input[:,:self.n_vals] - self.state
        delta = self.alpha_h(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.diff_hist.append(diff.detach())
        self.delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Perseverative Agent: α = {}'.format(self.alpha_h.to_string())
    
    
class FallacyAgent(ModelAgent):
    """ Gambler Fallacy Agent G

    Inputs:
        input: tensor of shape (n_sess, n_trials, [choice_t, outcome_t])

    Outputs:
        output: the output value of the Gambler Fallacy Agent G on each trial. Tensor of shape (n_sess, n_trials, 1)
   """

    def __init__(self, alpha0=None, n_vals=1):
        super().__init__()

        self.alpha_g = AlphaParam(alpha0)
        self.n_vals = n_vals
        self.reset_state()
       
    def forward(self, input):#first col of input-choice, second col of input-outcome
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], self.n_vals)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_state(self):
        self.state = torch.zeros(self.n_vals)
        self.state_hist = []
        self.diff_hist = []
        self.delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        diff = input[:,:self.n_vals] - input[:,:self.n_vals]*input[:,[self.n_vals]] - self.state
        delta = self.alpha_g(diff)
        self.state = self.state + delta
        
        self.state_hist.append(self.state.detach())
        self.diff_hist.append(diff.detach())
        self.delta_hist.append(delta.detach())
        
        return self.state
    
    def print_params(self):
        return 'Fallacy Agent: α = {}'.format(self.alpha_g.to_string())
    

# %% Q Value Agent

class QValueAgent(ModelAgent):
    """ Q-learning Value Agent 

    Inputs:
        input: tensor of shape (n_sess, n_trials, [chose_left, chose_right, outcome_t])

    Outputs:
        output: the output value of the Q-Value Agent on each trial. Tensor of shape (n_sess, n_trials, [left_value, right_value])
    """

    def __init__(self, alpha_same_rew=None, alpha_same_unrew=None, alpha_diff_rew=None, alpha_diff_unrew=None, 
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, constraints={}):
        
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
        
        self.reset_state()
       
    def forward(self, input):
        self.reset_state()
        output = torch.zeros(input.shape[0], input.shape[1], 2)

        # Propogate input through the network
        for t in range(input.shape[1]):
            output[:,t,:] = self.step(input[:,t,:])

        return output
    
    def reset_state(self):
        self.state = torch.tensor([[0,0]])
        self.state_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.state_hist.append(self.state.detach())
        
    def step(self, input):
        left_side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                           (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        right_side_outcome_sel = left_side_outcome_sel[:,[2,3,0,1]]
        
        k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        left_diffs = k_vals - self.state[:,0].unsqueeze(1)
        right_diffs = k_vals - self.state[:,1].unsqueeze(1)
        
        left_state_diffs = left_side_outcome_sel*left_diffs
        right_state_diffs = right_side_outcome_sel*right_diffs
        
        # update state
        alphas = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        
        left_state_deltas = left_state_diffs * alphas
        right_state_deltas = right_state_diffs * alphas
        
        state_diff = torch.cat([left_state_diffs.sum(dim=1).unsqueeze(1), right_state_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        state_delta = torch.cat([left_state_deltas.sum(dim=1).unsqueeze(1), right_state_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.state = self.state + state_delta

        # record state histories
        self.state_hist.append(self.state.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())
        
        # # update state        
        # left_deltas = torch.cat([self.alpha_same_rew(left_diffs[:,0]).unsqueeze(1), self.alpha_same_unrew(left_diffs[:,1]).unsqueeze(1),
        #                          self.alpha_diff_rew(left_diffs[:,2]).unsqueeze(1), self.alpha_diff_unrew(left_diffs[:,3]).unsqueeze(1)], dim=1)
        
        # right_deltas = torch.cat([self.alpha_same_rew(right_diffs[:,0]).unsqueeze(1), self.alpha_same_unrew(right_diffs[:,1]).unsqueeze(1),
        #                           self.alpha_diff_rew(right_diffs[:,2]).unsqueeze(1), self.alpha_diff_unrew(right_diffs[:,3]).unsqueeze(1)], dim=1)
        
        return self.state
    
    def print_params(self):
        return '''Q Value Agent: \n\t α same, rew = {} \n\t α same, unrew = {} \n\t α diff, rew = {} \n\t α diff, unrew = {}
\t κ same, rew = {:.5f} \n\t κ same, unrew = {:.5f} \n\t κ diff, rew = {:.5f} \n\t κ diff, unrew = {:.5f}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_unrew.to_string(), self.alpha_diff_rew.to_string(), self.alpha_diff_unrew.to_string(),
                  self.k_same_rew.item(), self.k_same_unrew.item(), self.k_diff_rew.item(), self.k_diff_unrew.item()) 
    
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
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, inverse_update=False, global_lam=True, constraints={}):
        
        super().__init__()
        
        # automatically constrain gammas if the lambda multiplier is a global term for both sides
        if global_lam:
            constraints['gamma_diff_rew'] = {'share': 'gamma_same_rew'}
            constraints['gamma_diff_unrew'] = {'share': 'gamma_same_unrew'}
            
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
        
    def step(self, input):
        
        left_side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                           (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        right_side_outcome_sel = left_side_outcome_sel[:,[2,3,0,1]]
        
        k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        left_diffs = k_vals - self.state[:,0].unsqueeze(1)
        right_diffs = k_vals - self.state[:,1].unsqueeze(1)
        
        left_state_diffs = left_side_outcome_sel*left_diffs
        right_state_diffs = right_side_outcome_sel*right_diffs
        
        # update state
        lam_mult = self.lam
        if self.inverse_update:
            lam_mult = 1 - lam_mult
        
        alphas = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        
        left_state_deltas = left_state_diffs * alphas * lam_mult[:,0].unsqueeze(1)
        right_state_deltas = right_state_diffs * alphas * lam_mult[:,1].unsqueeze(1)
        
        state_diff = torch.cat([left_state_diffs.sum(dim=1).unsqueeze(1), right_state_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        state_delta = torch.cat([left_state_deltas.sum(dim=1).unsqueeze(1), right_state_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.state = self.state + state_delta
        
        # update update multiplier lambdas
        
        gammas = torch.cat([self.gamma_same_rew.a, self.gamma_same_unrew.a, self.gamma_diff_rew.a, self.gamma_diff_unrew.a]).unsqueeze(0)
        
        if self.global_lam:
            # reconfigure the diffs to only include the k_same values in the order: left/reward, left/unreward, right/reward, right/unreward (so it works with left_side_outcome_sel)
            global_diffs = torch.cat([left_diffs[:,[0,1]], right_diffs[:,[0,1]]], dim=1)
            left_lam_diffs = (torch.abs(global_diffs) - self.lam[:,0].unsqueeze(1))*left_side_outcome_sel
            right_lam_diffs = left_lam_diffs
    
            left_lam_deltas = left_lam_diffs*gammas
            right_lam_deltas = left_lam_deltas
        else:
            left_lam_diffs = (torch.abs(left_diffs) - self.lam[:,0].unsqueeze(1))*left_side_outcome_sel
            right_lam_diffs = (torch.abs(right_diffs) - self.lam[:,1].unsqueeze(1))*right_side_outcome_sel
    
            left_lam_deltas = left_lam_diffs*gammas
            right_lam_deltas = right_lam_diffs*gammas
        
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
        return '''Q Value Agent: \n\t α same, rew = {} \n\t α same, unrew = {} \n\t α diff, rew = {} \n\t α diff, unrew = {}
\t γ same, rew = {} \n\t γ same, unrew = {} \n\t γ diff, rew = {} \n\t γ diff, unrew = {}
\t κ same, rew = {:.5f} \n\t κ same, unrew = {:.5f} \n\t κ diff, rew = {:.5f} \n\t κ diff, unrew = {:.5f}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_unrew.to_string(), self.alpha_diff_rew.to_string(), self.alpha_diff_unrew.to_string(),
                  self.gamma_same_rew.to_string(), self.gamma_same_unrew.to_string(), self.gamma_diff_rew.to_string(), self.gamma_diff_unrew.to_string(),
                  self.k_same_rew.item(), self.k_same_unrew.item(), self.k_diff_rew.item(), self.k_diff_unrew.item())
    
    
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
                 k_same_rew=1, k_same_unrew=0, k_diff_rew=0, k_diff_unrew=0, global_lam=True, shared_side_alphas=True, shared_outcome_alpha_update=True, constraints={}):
        
        super().__init__()
        
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
        self.alpha_hist = []
        self.state_diff_hist = []
        self.state_delta_hist = []
        self.lam_diff_hist = []
        self.lam_delta_hist = []
        self.alpha_diff_hist = []
        self.alpha_delta_hist = []
        self.state_hist.append(self.state.detach())
        self.lam_hist.append(self.lam.detach())
        
    def step(self, input):
        
        left_side_outcome_sel = torch.cat([(input[:,0]*input[:,2]).unsqueeze(1), (input[:,0]*(1-input[:,2])).unsqueeze(1), 
                                           (input[:,1]*input[:,2]).unsqueeze(1), (input[:,1]*(1-input[:,2])).unsqueeze(1)], dim=1)
        right_side_outcome_sel = left_side_outcome_sel[:,[2,3,0,1]]
        
        k_vals = torch.cat([self.k_same_rew, self.k_same_unrew, self.k_diff_rew, self.k_diff_unrew]).unsqueeze(0)
        left_diffs = k_vals - self.state[:,0].unsqueeze(1)
        right_diffs = k_vals - self.state[:,1].unsqueeze(1)
        
        # calculate unexpected uncertainty on current trial
        left_lam_diffs = torch.abs(left_diffs) - self.lam[:,0].unsqueeze(1)
        right_lam_diffs = torch.abs(right_diffs) - self.lam[:,1].unsqueeze(1)
        
        # calculate alpha on current trial
        alpha0s = torch.cat([self.alpha_same_rew.a, self.alpha_same_unrew.a, self.alpha_diff_rew.a, self.alpha_diff_unrew.a]).unsqueeze(0)
        
        # initialize alpha based of choice/outcome and parameter options
        if self.alpha is None:
            if self.shared_side_alphas and self.shared_outcome_alpha_update:
                # alphas are all the same value, starting at value determined by first outcome
                # doesn't matter which side is chosen because alpha same/diff is the same
                alphas = (alpha0s*left_side_outcome_sel).sum(dim=1).unsqueeze(1)
                self.alpha = alphas.repeat(1,4)
            else:
                # alphas can be different values so initialize to same rew/unrew 
                self.alpha = alpha0s[:,[0,1]].repeat(input.shape[0],2)
      
            alpha_diff = torch.zeros_like(left_side_outcome_sel)
            alpha_delta = torch.zeros_like(left_side_outcome_sel)

        else:
            
            # calculate all alpha diff values used in the different update rules (depending on global alpha and separate outcome alpha options), order: same, diff
            left_rew_alpha_diffs = alpha0s[:,[0,2]] + left_lam_diffs[:,[0,2]] - self.alpha[:,0].unsqueeze(1)
            left_unrew_alpha_diffs = alpha0s[:,[1,3]] + left_lam_diffs[:,[1,3]] - self.alpha[:,1].unsqueeze(1)
            right_rew_alpha_diffs = alpha0s[:,[0,2]] + right_lam_diffs[:,[0,2]] - self.alpha[:,2].unsqueeze(1)
            right_unrew_alpha_diffs = alpha0s[:,[1,3]] + right_lam_diffs[:,[1,3]] - self.alpha[:,3].unsqueeze(1)
            
            gamma_alphas = torch.cat([self.gamma_alpha_same_rew.a, self.gamma_alpha_same_unrew.a, self.gamma_alpha_diff_rew.a, self.gamma_alpha_diff_unrew.a]).unsqueeze(0)
    
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
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*gamma_alphas
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
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*gamma_alphas[:,[0,2]]
                    right_rew_alpha_deltas = left_rew_alpha_deltas
                    left_unrew_alpha_deltas = left_unrew_alpha_diffs*gamma_alphas[:,[1,3]]
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
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*gamma_alphas
                    right_rew_alpha_deltas = right_rew_alpha_diffs*gamma_alphas
                    left_unrew_alpha_deltas = left_rew_alpha_deltas
                    right_unrew_alpha_deltas = right_rew_alpha_deltas
                else:
                    # alphas are incremented separately for each side and outcome
                    left_rew_alpha_diffs = left_rew_alpha_diffs*left_side_outcome_sel[:,[0,2]]
                    right_rew_alpha_diffs = right_rew_alpha_diffs*right_side_outcome_sel[:,[0,2]]
                    left_unrew_alpha_diffs = left_unrew_alpha_diffs*left_side_outcome_sel[:,[1,3]]
                    right_unrew_alpha_diffs = right_unrew_alpha_diffs*right_side_outcome_sel[:,[1,3]]
                    
                    left_rew_alpha_deltas = left_rew_alpha_diffs*gamma_alphas[:,[0,2]]
                    right_rew_alpha_deltas = right_rew_alpha_diffs*gamma_alphas[:,[0,2]]
                    left_unrew_alpha_deltas = left_unrew_alpha_diffs*gamma_alphas[:,[1,3]]
                    right_unrew_alpha_deltas = right_unrew_alpha_diffs*gamma_alphas[:,[1,3]]

        
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
        gamma_lams = torch.cat([self.gamma_lam_same_rew.a, self.gamma_lam_same_unrew.a, self.gamma_lam_diff_rew.a, self.gamma_lam_diff_unrew.a]).unsqueeze(0)
        
        if self.global_lam:
            # reconfigure the diffs to only include the k_same values in the order: left/reward, left/unreward, right/reward, right/unreward (so it works with left_side_outcome_sel)
            global_lam_diffs = torch.cat([left_lam_diffs[:,[0,1]], right_lam_diffs[:,[0,1]]], dim=1)
            left_lam_diffs = global_lam_diffs*left_side_outcome_sel
            right_lam_diffs = left_lam_diffs
    
            left_lam_deltas = left_lam_diffs*gamma_lams
            right_lam_deltas = left_lam_deltas
        else:
            left_lam_diffs = left_lam_diffs*left_side_outcome_sel
            right_lam_diffs = right_lam_diffs*right_side_outcome_sel
    
            left_lam_deltas = left_lam_diffs*gamma_lams
            right_lam_deltas = right_lam_diffs*gamma_lams
        
        lam_diff = torch.cat([left_lam_diffs.sum(dim=1).unsqueeze(1), right_lam_diffs.sum(dim=1).unsqueeze(1)], dim=1)
        lam_delta = torch.cat([left_lam_deltas.sum(dim=1).unsqueeze(1), right_lam_deltas.sum(dim=1).unsqueeze(1)], dim=1)
        
        self.lam = self.lam + lam_delta
        
        # record state histories
        self.state_hist.append(self.state.detach())
        self.lam_hist.append(self.lam.detach())
        self.alpha_hist.append(self.alpha.detach())
        self.state_diff_hist.append(state_diff.detach())
        self.state_delta_hist.append(state_delta.detach())
        self.lam_diff_hist.append(lam_diff.detach())
        self.lam_delta_hist.append(lam_delta.detach())
        self.alpha_diff_hist.append(alpha_diff.detach())
        self.alpha_delta_hist.append(alpha_delta.detach())
        
        return self.state
    
    def print_params(self):
        return '''Q Value Agent: \n\t α same, rew = {} \n\t α same, unrew = {} \n\t α diff, rew = {} \n\t α diff, unrew = {}
\t γ_α same, rew = {} \n\t γ_α same, unrew = {} \n\t γ_α diff, rew = {} \n\t γ_α diff, unrew = {}
\t γ_λ same, rew = {} \n\t γ_λ same, unrew = {} \n\t γ_λ diff, rew = {} \n\t γ_λ diff, unrew = {}
\t κ same, rew = {:.5f} \n\t κ same, unrew = {:.5f} \n\t κ diff, rew = {:.5f} \n\t κ diff, unrew = {:.5f}'''.format(
                  self.alpha_same_rew.to_string(), self.alpha_same_unrew.to_string(), self.alpha_diff_rew.to_string(), self.alpha_diff_unrew.to_string(),
                  self.gamma_alpha_same_rew.to_string(), self.gamma_alpha_same_unrew.to_string(), self.gamma_alpha_diff_rew.to_string(), self.gamma_alpha_diff_unrew.to_string(),
                  self.gamma_lam_same_rew.to_string(), self.gamma_lam_same_unrew.to_string(), self.gamma_lam_diff_rew.to_string(), self.gamma_lam_diff_unrew.to_string(),
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
        self.bias = _init_param(bias)
        self.output_layer = output_layer
        
        self.reset_state()
       
    
    def forward(self, input): #first col of input-choice, second col of input-outcome
        self.reset_state()
        return self._gen_apply_agents(input, lambda agent, input: agent(input))
        
    
    def reset_state(self):
        for model in self.agents:
            model.reset_state()
            
        self.output_hist = [] 
        
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
        