# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 13:58:40 2026

@author: Tanner
"""

import utils
from os import path
from hankslab_db import base_db

class path_helper():
    
    def __init__(self, model_name, on_cluster):
        self.model_name = model_name
        self.on_cluster = on_cluster
        
    @property
    def cluster_home(self):
        return '/group/thanksgrp/python'
    
    @property
    def base_cluster_path(self):
        return path.join(self.cluster_home, self.model_name)
    
    @property
    def base_local_path(self):
        return path.join(utils.get_user_home(), 'model_fits', self.model_name)
    
    @property
    def base_path(self):
        if self.on_cluster:
            return self.base_cluster_path
        else:
            return self.base_local_path
    
    @property
    def output_path(self):
        return path.join(self.base_path, 'fits')
    
    @property
    def config_path(self):
        return path.join(self.base_path, 'configs')
    
    @property
    def data_path(self):
        if self.on_cluster:
            return path.join(self.cluster_home, 'data')
        else:
            return base_db.default_data_dir()

    def get_fit_save_path(self, file_name):
        return path.join(self.output_path, file_name)