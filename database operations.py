# -*- coding: utf-8 -*-
"""
Created on Mon May 20 15:38:47 2024

@author: tanne
"""

import init
from hankslab_db import db_access

# %% Add procedures and implants to the database

db_access.add_procedure(188, 'Optical fiber implant for dLight fiber photometry data collection', '2x 1mm tapered 400μm optical fiber', 'Prelimbic cortex and dorsomedial striatum')

db_access.add_fp_implant(188, 'PL', '1mm tapered 400μm optical fiber', 3.0, -0.7, -3.4, 'Injected 400nL 2x separated by 0.6mm. CAG-dLight3.8. Lost a lot of blood during durotomy.')
db_access.add_fp_implant(188, 'DMS', '1mm tapered 400μm optical fiber', 1.0, 2.3, -4.0, 'Injected 400nL 2x separated by 0.6mm. CAG-dLight3.8')

# %%
db_access.add_fp_data(179, 'PL', [], [], [], sess_id=92562)

# %%
db_access.get_fp_protocol_subj_sess_ids('ClassicRLTasks', 2, 179)

# %% Package data

subj_id = 179
sess_id = 95631

region_dict = {'PL': 1, 'DMS': 2} #'PL': 1, 'DMS': 2
wavelength_dict = {490: 2, 420: 1, 405: 3} #420: 1, 465: 4, 490: 2,
target_dt = 0.005
new_format = True
print_struct = False
print_attr = False
