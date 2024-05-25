# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 23:30:18 2023

@author: tanne
"""

# %% Declare imports and methods

import init
from hankslab_db import package_fp_data as pkg

subj_id = 179
sess_id = 95631

region_dict = {'PL': 1, 'DMS': 2} #'PL': 1, 'DMS': 2
wavelength_dict = {490: 2, 420: 1, 405: 3} #420: 1, 465: 4, 490: 2,
target_dt = 0.005
new_format = True
print_struct = False
print_attr = False

pkg.package_doric_data(subj_id, sess_id, region_dict, wavelength_dict,
                       target_dt=target_dt, new_format=new_format,
                       print_file_struct=print_struct, print_attr=print_attr)
