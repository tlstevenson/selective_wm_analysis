# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 12:23:33 2022

@author: tanne
"""

import sys
from os import path
sys.path.append(path.join(path.dirname(path.abspath(__file__)), '..'))

import hankslab_db.tonecatdelayresp_db as db

loc_db = db.LocalDB_ToneCatDelayResp()  # reload=True
sess_99 = loc_db.get_behavior_data([90005])
