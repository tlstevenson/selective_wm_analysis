# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:35:16 2024

@author: tanne
"""

# %% Imports

import init
from pyutils import utils, cluster_utils
import hankslab_db.basicRLtasks_db as db
from hankslab_db import db_access
import agents
import training_helpers as th
from path_helper import path_helper
import fit_bandit_models
from datetime import datetime
import json
from os import path
import time
import numpy as np

# %% Declare model(s) to fit

# build dictionary of model(s) to fit

all_main_agent_gens = {}
all_main_agent_settings = {}
all_add_agents = {}

## Additional Agents
# will be applied to all main agents
all_add_agents[''] = None
all_add_agents['Persev (free alpha)'] = [agents.PerseverativeAgent(n_vals=2)]
all_add_agents['Persev (fixed)'] = [agents.PerseverativeAgent(n_vals=2, constraints={'alpha': {'fit': False, 'init': 1}})]

## Q MODELS

all_main_agent_gens['Q'] = lambda s: agents.QValueAgent(constraints=s)

all_main_agent_settings['Q'] = {
        # All Alphas Free, Different K Fixes
         'All Free': {},
        
        # 'All Alpha Free, S/R K Fixed': {'k_same_rew': {'fit': False}},
        
        # 'All Alpha Free, Same K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
        
        'All Alpha Free, All K Fixed': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'All Alpha Free, D/R K Free': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
    
        # 'All Alpha Free, All K Fixed, Diff K=0.5': {'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
        #                                             'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
        
        # # All Alphas shared, Different K Fixes
        'All Alpha Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
                                          'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
                                          'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'All Alpha Shared, D/R K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
        #                                   'alpha_diff_unrew': {'share': 'alpha_same_rew'},'k_same_rew': {'fit': False}, 
        #                                   'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'All Alpha Shared, Same K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
        #                                    'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
        
        # 'All Alpha Shared, All K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_rew'}},
        
        # 'All Alpha Shared, All K Fixed, Diff K=0.5': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
        #                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
        #                                               'k_diff_rew': {'fit': False, 'init': 0.5}, 'k_diff_unrew': {'fit': False, 'init': 0.5}},
        
        # # Models with limited different choice updating
        # 'Same Alpha Only Shared, K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
        #                                'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Same Alpha Only Shared, Counter K': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
        #                                'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Same Alpha Shared, K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'fit': False, 'init': 0}, 
        #                               'alpha_diff_unrew': {'fit': False, 'init': 0}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Same Alpha Only, K Fixed': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
        #                              'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Same Alpha Only, K Free': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
        #                             'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Same Alpha Only, Counter K': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'alpha_diff_unrew': {'fit': False, 'init': 0},
        #                                'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 
        #                                'k_diff_unrew': {'fit': False}},
        
        # 'No Alpha D/U, All K Fixed': {'alpha_diff_unrew': {'fit': False, 'init': 0}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
        #                                'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'No Alpha D/R, All K Fixed': {'alpha_diff_rew': {'fit': False, 'init': 0}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False},
        #                                'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        
        # # Constrained Alpha Pairs
        # 'Alpha Same/Diff Shared, Same K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
        #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
        
        # 'Alpha Same/Diff Shared, All K Fixed': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
        #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
        #                                          'k_diff_unrew': {'fit': False}},
        
        # 'Alpha Same/Diff Shared, D/R K Free': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
        #                                        'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Alpha Rew/Unrew Shared, Same K Fixed': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
        #                                          'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}},
        
        'Alpha Rew/Unrew Shared, All K Fixed': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
                                                 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
                                                 'k_diff_unrew': {'fit': False}},
        
        # 'Alpha Rew/Unrew Shared, D/R K Free': {'alpha_diff_rew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_same_unrew'}, 
        #                                        'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        
        # # Counterfactual models
        # 'All Alpha Shared, Counter D/U K=1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
        #                                       'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 
        #                                       'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False, 'init': 1}},
        
        # 'Alpha Same/Diff Shared, Counter D/U K=1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
        #                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False}, 
        #                                             'k_diff_unrew': {'fit': False, 'init': 1}},
        
        # 'All Alpha Shared, Counter D/R K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
        #                                       'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 
        #                                       'k_diff_rew': {'fit': False, 'init': -1}, 'k_diff_unrew': {'fit': False}},
        
        # 'Alpha Same/Diff Shared, Counter D/R K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
        #                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False}, 'k_diff_rew': {'fit': False, 'init': -1}, 
        #                                             'k_diff_unrew': {'fit': False}},
        
        # 'All Alpha Shared, Counter S/U K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
        #                                       'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
        #                                       'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}},
        
        # 'Alpha Same/Diff Shared, Counter S/U K=-1': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_unrew': {'share': 'alpha_diff_rew'}, 
        #                                             'k_same_rew': {'fit': False}, 'k_same_unrew': {'fit': False, 'init':-1}, 'k_diff_rew': {'fit': False}, 
        #                                             'k_diff_unrew': {'fit': False}},
        
        }


## SI Models

all_main_agent_gens['SI'] = lambda s: agents.StateInferenceAgent(**s)

all_main_agent_settings['SI'] = {
        'Shared Evidence': {'complement_c_rew':True, 'complement_c_diff':True}, 
        
        # 'Separate Same/Diff Evidence': {'complement_c_rew':True, 'complement_c_diff':False},
        
        'Separate Rew/Unrew Evidence': {'complement_c_rew':False, 'complement_c_diff':True}, 
        
         'All Separate Evidence': {'complement_c_rew':False, 'complement_c_diff':False},
        
        # 'Free Same/Diff Rew Evidence': {'complement_c_rew':False, 'complement_c_diff':False,
        #                                 'constraints': {'c_same_unrew': {'fit': False, 'init': 0}, 
        #                                                 'c_diff_unrew': {'fit': False, 'init': 0}}}
        }

                    
## FULL BAYESIAN MODEL FITS

all_main_agent_gens['Bayes'] = lambda s: agents.BayesianAgent(p_step=0.01, **s)

all_main_agent_settings['Bayes'] = {
        # 'No Switch Scatter, Switch Update First': {'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False}}}, 
        
        'No Switch Scatter, Perfect Update, No Stay Bias, Simul Updates': {'update_p_switch_first': False,
                                                                           'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False},
                                                                                           'imperfect_update_alpha': {'init': 0, 'fit': False},
                                                                                           'stay_bias_lam': {'init': 0, 'fit': False}}}, 
            
        # 'Fixed 0.5 Prior Rew Mean, No Switch Scatter': {'constraints': {'switch_scatter_sig': {'init': 0, 'fit': False}, 
        #                                                          'init_high_rew_mean': {'init': 0, 'fit': False}, 
        #                                                          'init_low_rew_mean': {'init': 1, 'fit': False}}}
        }


### NOTE: Below calls have not been modified to reflect new code structure ###

## DYNAMIC Q-MODEL FITS

# # declare model fit settings
# settings = {
#             'All Alpha Shared, All K Fixed, Global λ': {'global_lam': True, 'constraints': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
#                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
#                                               'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}},
            
#             }

# agent_gen = lambda s: agents.DynamicQAgent(**s)

# th.fit_two_side_model(agent_gen, 'Dynamic Q', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
#                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
#                    print_train_params=print_train_params)

## UNCERTAINTY DYNAMIC Q-MODEL FITS

# # declare model fit settings
# settings = {
#             'All Alpha Shared, All K Fixed, Global λ': {'global_lam': True, 'constraints': {'alpha_same_unrew': {'share': 'alpha_same_rew'}, 'alpha_diff_rew': {'share': 'alpha_same_rew'}, 
#                                               'alpha_diff_unrew': {'share': 'alpha_same_rew'}, 'k_same_rew': {'fit': False}, 
#                                               'k_same_unrew': {'fit': False},'k_diff_rew': {'fit': False}, 'k_diff_unrew': {'fit': False}}},
            
#             }

# agent_gen = lambda s: agents.UncertaintyDynamicQAgent(**s)

# th.fit_two_side_model(agent_gen, 'Uncertainty Dynamic Q', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
#                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
#                    print_train_params=print_train_params)
        
## Q-VALUE STATE INFERENCE MODEL FITS

# settings = {'Alpha Free, K Free': {},
            
#             # 'Alpha Free, K Free, Value Update First': {'update_order': 'value_first'},
            
#             # 'Alpha Free, K Free, Belief Update First': {'update_order': 'belief_first'},
    
#             # 'Alpha Free, K Fixed': {'constraints': {'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False},
#             #                                         'k_low_rew': {'fit': False}, 'k_low_unrew': {'fit': False}}},
            
#             # 'All Alpha Shared, K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 'alpha_low_rew': {'share': 'alpha_high_rew'}, 
#             #                                              'alpha_low_unrew': {'share': 'alpha_high_rew'}}},
            
#             # 'Shared High Alpha, Const Low K, High K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
#             #                                                                 'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
#             #                                                                 'k_low_unrew': {'share': 'k_low_rew', 'init': None}}},
            
#             # 'Shared High Alpha, Fixed Low K, High K Free': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
#             #                                                                 'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
#             #                                                                 'k_low_unrew': {'share': 'k_low_rew', 'fit': False, 'init': 0.1}}},
            
#             # 'Separate High Alphas, Const Low K, High K Free': {'constraints': {'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
#             #                                                                    'k_low_unrew': {'share': 'k_low_rew', 'init': None}}},
            
#             # 'Separate Low Alphas, Const High K, Low K Free': {'constraints': {'alpha_high_rew': {'share': 'alpha_high_unrew', 'fit': False, 'init': 0}, 
#             #                                                                    'k_high_unrew': {'share': 'k_high_rew', 'init': None}}},
            
#             # 'Shared High Alpha, Const Low K, High K Fixed': {'constraints': {'alpha_high_unrew': {'share': 'alpha_high_rew'}, 
#             #                                                                  'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
#             #                                                                  'k_low_unrew': {'share': 'k_low_rew', 'init': None}, 
#             #                                                                  'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False}}},
            
#             # 'Separate High Alphas, Const Low K, High K Fixed': {'constraints': {'alpha_low_rew': {'share': 'alpha_low_unrew', 'fit': False, 'init': 0}, 
#             #                                                                     'k_low_unrew': {'share': 'k_low_rew', 'init': None}, 
#             #                                                                     'k_high_rew': {'fit': False}, 'k_high_unrew': {'fit': False}}}
#             }

# agent_gen = lambda s: agents.QValueStateInferenceAgent(**s)

# th.fit_two_side_model(agent_gen, 'Q SI', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
#                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
#                    print_train_params=print_train_params)


# ## RL STATE INFERENCE MODEL FITS

# settings = {
#             # 'Shared Evidence': {'complement_c_rew':True, 'complement_c_diff':True}, 
#             # 'Separate Same/Diff Evidence': {'complement_c_rew':True, 'complement_c_diff':False},
#             # 'Separate Rew/Unrew Evidence': {'complement_c_rew':False, 'complement_c_diff':True}, 
#             # 'All Separate Evidence': {'complement_c_rew':False, 'complement_c_diff':False},
#             'Free Same/Diff Rew Evidence': {'complement_c_rew':False, 'complement_c_diff':False,
#                                             'constraints': {'c_same_unrew': {'fit': False, 'init': 0}, 
#                                                             'c_diff_unrew': {'fit': False, 'init': 0}}}
#             }

# agent_gen = lambda s: agents.RLStateInferenceAgent(**s)

# th.fit_two_side_model(agent_gen, 'RL SI', settings, two_side_inputs, choice_class_labels, trial_mask, subj, save_path, 
#                    n_fits=n_fits, n_steps=n_steps, end_tol=end_tol, skip_existing_fits=skip_existing_fits, refit_existing=refit_existing, 
#                    print_train_params=print_train_params)
    

# %% Run Fitting

run_on_cluster = True

subj_ids = [198, 199, 274, 400, 402]
separate_rew_rates = True
fit_ind_subj = True
fit_meta_subj = True
equal_sess_weight = True

skip_existing_fits = True
refit_existing = False
print_train_params = False

# limit_mask = False
# n_limit_hist = 2

#limitations
n_fits = 3 
n_steps = 10000 
end_tol = 1e-6

meta_subj_name = 'meta'
model_beh_name = 'probabilistic_bandit'

reload_beh = False

# get group session ids
subj_sess_ids = db_access.get_fp_data_sess_ids(subj_ids=subj_ids, protocol='ClassicRLTasks', stage_num=2)

# make sure all the behavior data is loaded locally
loc_db = db.LocalDB_BasicRLTasks('twoArmBandit')
all_sess = loc_db.get_behavior_data(utils.flatten(subj_sess_ids), reload=reload_beh)

group_sess_ids = {}

if separate_rew_rates:
    save_file_name = 'fit_models_sep_rates.json'
    
    rew_rates = all_sess['block_prob'].unique()
    
    for rate in rew_rates:
        block_sess = all_sess[all_sess['block_prob'] == rate]
        for subj in subj_ids:
            group_sess_ids['{} ({})'.format(subj, rate)] = block_sess.loc[block_sess['subjid'] == subj, 'sessid'].unique().tolist()
            
        if fit_meta_subj:
            group_sess_ids['{} ({})'.format(meta_subj_name, rate)] = block_sess['sessid'].unique().tolist()
else:
    save_file_name = 'fit_models_all_sess.json'
    group_sess_ids = {k: v.tolist() for k,v in subj_sess_ids.items()}
    if fit_meta_subj:
        group_sess_ids[meta_subj_name] = utils.flatten(subj_sess_ids).tolist()

local_ph = path_helper(model_beh_name, False)

fit_config = {'fit_group_names': list(group_sess_ids.keys()), 'group_sess_ids': group_sess_ids, 'save_file_name': save_file_name, 'model_beh_name': model_beh_name, 
              'equal_sess_weight': equal_sess_weight, 'skip_existing_fits': skip_existing_fits, 'refit_existing': refit_existing, 
              'print_train_params': print_train_params, 'n_fits': n_fits, 'n_steps': n_steps, 'end_tol': end_tol}

#%%
# push behavioral data to cluster
if run_on_cluster:
    cluster_ph = path_helper(model_beh_name, True)
    clust_db = db.LocalDB_BasicRLTasks('twoArmBandit', data_dir=cluster_ph.data_path, save_locally=False)
    
    clust_data_path = path.dirname(clust_db._get_sess_beh_path(''))
    existing_files = cluster_utils.get_all_files(clust_data_path)
                
    for subj in subj_ids: 
        for sess_id in subj_sess_ids[subj]:
            _, sess_filename = path.split(clust_db._get_sess_beh_path(sess_id))
            
            if not sess_filename in existing_files:
                local_path = loc_db._get_sess_beh_path(sess_id)
                cluster_utils.push_to_cluster(local_path, clust_data_path)
                # sleep for a tick to not overload the command prompt
                time.sleep(0.01)

# Run the model fits
for agent_name in all_main_agent_gens.keys():
    for setting_name, setting_vals in all_main_agent_settings[agent_name].items():
        for add_agent_name, add_agent_list in all_add_agents.items():
            main_agent = all_main_agent_gens[agent_name](setting_vals)
            
            if add_agent_list is None:
                model_name = '{} - {}'.format(agent_name, setting_name)
                model = agents.SummationModule([main_agent])
            else:
                model_name = '{}/{} - {}'.format(agent_name, add_agent_name, setting_name)
                model = agents.SummationModule([main_agent]+add_agent_list)
                
            fit_config['model'] = agents.serialize_model(model)
            fit_config['model_name'] = model_name
            fit_config['basic_model'] = isinstance(main_agent, agents.SingleValueAgent)

            if run_on_cluster:

                # save config file and push to cluster
                current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
                config_name = 'config_{}.json'.format(current_datetime)
                
                config_path = path.join(local_ph.config_path, config_name)
                utils.check_make_dir(config_path)
                with open(config_path, 'w') as f:
                    json.dump(fit_config, f, indent=3)
                    
                cluster_utils.push_to_cluster(config_path, cluster_ph.config_path)

                # submit slurm job
                slurm_path = path.join(cluster_ph.cluster_home, 'code/selective_wm_analysis/rl_analysis/modeling/fit_bandit_models.slurm')
                options = '--array=1-{}'.format(len(group_sess_ids.keys())*n_fits)
                args = 'CONFIGPATH=\'{}\''.format(cluster_utils.clean_cluster_path(path.join(cluster_ph.config_path, config_name)))

                status, output = cluster_utils.run_slurm_job(slurm_path, sbatch_options=options, custom_args=args, print_out=False)
                if status == 0:
                    print('Submitted model {} for fitting: {}'.format(model_name, output))
                else:
                    print('Error submitting model {} for fitting: {}'.format(model_name, output))
                
                # wait 1 second before submitting next job so config file name is unique
                time.sleep(1)
                
            else:
                for subj_idx in range(len(group_sess_ids.keys())):
                    fit_bandit_models.perform_fit(fit_config, subj_idx)
