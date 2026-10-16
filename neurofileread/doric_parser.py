# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 14:33:46 2026

@author: cns-th-lab
"""

from sys_neuro_tools import doric_utils as du
import fp_analysis_helpers as fpah
import os

def get_doric_timestamps_sess(sess_ids, par_dir):
    timestamps = {}
    rat_folders = [f for f in os.listdir(par_dir) if os.path.isdir(os.path.join(par_dir, f))]

    for sess_id in sess_ids:
        found_doric = False
        for rat_folder in rat_folders:
            doric_path = os.path.join(par_dir, rat_folder, "Videos", f"mov_{sess_id}.doric")
            
            if os.path.exists(doric_path):
                found_doric = True
                timestamps[sess_id] = get_doric_timestamps(doric_path)
                break
        
        #Raise an error if any sepcified session does not have a prediction
        if not found_doric:
            raise FileExistsError(f"Rat and port prediction files not found for session id: {sess_id}")
    return timestamps

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