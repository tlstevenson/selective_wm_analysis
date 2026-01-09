# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 11:35:16 2024

@author: tanne
"""

# %% Imports

import init
from pyutils import utils, cluster_utils
import hankslab_db.basicRLtasks_db as db
import agents
import training_helpers as th
import os
from os import path
import json
import argparse
from path_helper import path_helper

on_cluster = cluster_utils.on_cluster()

# %% Model Fitting Method

def perform_fit_cluster(config_path, fit_group_idx):
    
    print('Config path: {}'.format(config_path))

    if path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # repeats are handled through multiples of the subject index
        fit_group_idx = (fit_group_idx-1) % len(config['fit_group_names'])
        perform_fit(config, fit_group_idx)
    else:
        print('Config file was not found. Exiting...')


def perform_fit(config, fit_group_idx):

    fit_group_names = config['fit_group_names']
    group_sess_ids = config['group_sess_ids']
    group_name = fit_group_names[fit_group_idx]

    ph = path_helper(config['model_beh_name'], on_cluster)
    
    # get session data
    loc_db = db.LocalDB_BasicRLTasks('twoArmBandit', data_dir=ph.data_path)
    sess_ids = group_sess_ids[group_name]
    sess_data = loc_db.get_behavior_data(sess_ids)

    # get model training information
    model = agents.deserialize_model(config['model'])

    basic_model = config['basic_model']
    
    save_file_name = config['save_file_name']
    
    training_data = th.get_model_training_data(sess_data, basic_model)
    loss_output_transforms = th.get_loss_output_transforms(basic_model)

    th.fit_model(model, config['model_name'], training_data['inputs'], training_data['labels'], training_data['trial_mask_train'], training_data['trial_mask_eval'], 
                 loss_output_transforms['loss'], group_name, ph.get_fit_save_path(save_file_name), train_output_formatter=loss_output_transforms['train_output_formatter'], 
                 eval_output_transform=loss_output_transforms['eval_output_transform'], n_fits=config['n_fits'], n_steps=config['n_steps'], 
                 equal_sess_weight=config['equal_sess_weight'], skip_existing_fits=config['skip_existing_fits'], 
                 refit_existing=config['refit_existing'], print_train_params=config['print_train_params'])
        
# %% main entry point for script called on cluster

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path')

    args = parser.parse_args()

    task_id_str = os.environ.get('SLURM_ARRAY_TASK_ID')

    if task_id_str is not None:
        task_id = int(task_id_str)
        print('Array task id: {}'.format(task_id))
    else:
        print('Array task id not found. Using 0')
        task_id = 0

    perform_fit_cluster(args.config_path, task_id)    