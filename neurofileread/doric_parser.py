# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:33:46 2026

@author: cns-th-lab
"""

from sys_neuro_tools import doric_utils as du
import fp_analysis_helpers as fpah

def get_doric_timestamps(doric_filepath):
    """Extracts camera/hardware timestamps from the .doric file."""
    time_in, time_in_info = du.h5read(
        doric_filepath,
        [
            "DataAcquisition",
            "BehaviorCamera",
            "Video",
            "Series0001",
            "DMK-33UX290",
            "Time",
        ],
    )
    return time_in

def load_fp(loc_db, subj_sess_ids):
    """Wraps the fiber photometry data extraction."""
    return fpah.load_fp_data(loc_db, subj_sess_ids)