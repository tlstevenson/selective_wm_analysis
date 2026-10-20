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

subjid = 274
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (1.2x10^13 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'Bilateral DMS & DLS/TS')
db_access.add_fp_implant(subjid, 'DMS-L', 'Flat 400μm optical fiber', 1.5, 2.1, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS-R', 'Flat 400μm optical fiber', 1.5, -2.1, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS-L', 'Flat 400μm optical fiber', 0, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS-R', 'Flat 400μm optical fiber', 0, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 400
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (1.2x10^13 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'PL, NAc, DMS, & DLS')
db_access.add_fp_implant(subjid, 'PL', 'Flat 400μm optical fiber', 3.0, -0.6, -3, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.6, 1.6, -7.2, 'Injected 500nl at tip. Implanted 0.2mm lower than intended. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1.1, -2.1, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS', 'Flat 400μm optical fiber', 0.6, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. May be slightly more posterior (~0.5mm). CAG-dLight3.8')

subjid = 402
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (1.2x10^13 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'PL, DMS, DLS, & TS')
db_access.add_fp_implant(subjid, 'PL', 'Flat 400μm optical fiber', 3.0, 0.6, -3, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1.2, -2.1, -3.6, 'Injected 500nl 0.2mm below tip. Mushrooming of cortex. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS', 'Flat 400μm optical fiber', 0.6, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. Lots of bleeding on implant of fiber. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS', 'Flat 400μm optical fiber', -1, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

# %%
subjid = 237
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'PL, NAc, DMS, & TS')
db_access.add_fp_implant(subjid, 'PL', 'Flat 400μm optical fiber', 3.0, -0.6, -3, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.5, 1.6, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1.1, -2, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS', 'Flat 400μm optical fiber', -1, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 424
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'PL, NAc, DMS, & DLS')
db_access.add_fp_implant(subjid, 'PL', 'Flat 400μm optical fiber', 3.0, 0.6, -3, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.5, -1.6, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1.1, 2, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS', 'Flat 400μm optical fiber', 0.4, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 238
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'NAc, DMS, DLS, & TS')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.6, -1.6, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1.1, 2, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS', 'Flat 400μm optical fiber', 0.4, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS', 'Flat 400μm optical fiber', -1, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 483
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'Bilateral NAc & TS')
db_access.add_fp_implant(subjid, 'NAc-L', 'Flat 400μm optical fiber', 1.6, 1.6, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8. Not perfectly symmetric')
db_access.add_fp_implant(subjid, 'NAc-R', 'Flat 400μm optical fiber', 1.6, -1.6, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8. Not perfectly symmetric')
db_access.add_fp_implant(subjid, 'TS-L', 'Flat 400μm optical fiber', -1, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8. Not perfectly symmetric')
db_access.add_fp_implant(subjid, 'TS-R', 'Flat 400μm optical fiber', -1, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8. Not perfectly symmetric')


# %%
subjid = 234
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'Bilateral NAc & DLS')
db_access.add_fp_implant(subjid, 'NAc-L', 'Flat 400μm optical fiber', 1.7, 1.7, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'NAc-R', 'Flat 400μm optical fiber', 1.7, -1.7, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS-L', 'Flat 400μm optical fiber', 0.3, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS-R', 'Flat 400μm optical fiber', 0.3, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 235
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'PL, NAc, DLS, & TS')
db_access.add_fp_implant(subjid, 'PL', 'Flat 400μm optical fiber', 3.0, -0.65, -3, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.6, 1.7, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS', 'Flat 400μm optical fiber', 0.4, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS', 'Flat 400μm optical fiber', -1, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 419
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'NAc, DMS, DLS, & TS')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.7, 1.7, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1.2, -1.95, -3.6, 'Injected 500nl 0.2mm below tip. Injected additional 300nl in DMS because I was unsure if the original 500nl was injected. I didn’t hear the beep. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DLS', 'Flat 400μm optical fiber', 0.3, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS', 'Flat 400μm optical fiber', -1, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 421
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'PL, NAc, DMS, & TS')
db_access.add_fp_implant(subjid, 'PL', 'Flat 400μm optical fiber', 3.0, 0.65, -3, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'NAc', 'Flat 400μm optical fiber', 1.6, -1.7, -7, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS', 'Flat 400μm optical fiber', 1, 2, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS', 'Flat 400μm optical fiber', -1, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

subjid = 422
db_access.add_procedure(subjid, 'Optical fiber implant for dLight fiber photometry data collection. AAV9 CAG-dLight3.8 (6.1x10^12 VP/ml)',
                        '4x flat 400μm, 1.25mm ferrule fibers', 'Bilateral DMS & TS')
db_access.add_fp_implant(subjid, 'DMS-L', 'Flat 400μm optical fiber', 1.2, 1.95, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'DMS-R', 'Flat 400μm optical fiber', 1.2, -1.95, -3.6, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS-L', 'Flat 400μm optical fiber', -1, 3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')
db_access.add_fp_implant(subjid, 'TS-R', 'Flat 400μm optical fiber', -1, -3.8, -3.8, 'Injected 500nl 0.2mm below tip. CAG-dLight3.8')

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

# %% Modify Behavioral Stages/Settings
