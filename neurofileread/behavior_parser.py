# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:57:18 2026

@author: cns-th-lab
"""

from hankslab_db import db_access
from hankslab_db import tonecatdelayresp_db as wm_db
from hankslab_db import basicRLtasks_db as bandit_db

def load_wm_behavior(sess_ids):
    """Wraps the working memory local DB access."""
    wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
    return wm_loc_db.get_behavior_data(sess_ids)

def load_bandit_behavior(sess_ids):
    """Wraps the bandit local DB access."""
    bandit_loc_db = bandit_db.LocalDB_BasicRLTasks("twoArmBandit")
    return bandit_loc_db.get_behavior_data(sess_ids)