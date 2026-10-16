# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:57:18 2026

@author: cns-th-lab
"""
import numpy as np

def get_trial_end_ts(sess_data):
    """Get the last state timestamp from a trial to determine its end relative to the start.

    Args:
        sess_data(TODO???): Takes a session data from db_access
    Returns:
        A list of time deltas relative to the start of the trial indicating relative trial end times.
    """
    
    """print(type(sess_data))
    print(sess_data["parsed_events"][0]["States"])
    print()
    print(type(sess_data["parsed_events"][0]["States"]))
    print()
    trial_end_ts_vect = []
    print(len(sess_data["parsed_events"]))
    print()
    print(sess_data.keys())
    print()
    print(sess_data["sessid"])"""
    
    trial_end_ts_dict = {}
    for sessid in np.unique(sess_data["sessid"]):
        #Get trials in current session
        curr_sess_df = sess_data[sess_data["sessid"] == sessid]
        trial_end_ts_vect = []
        #Iterate through the trials extracting values
        for trial in range(len(curr_sess_df["parsed_events"])):
            max_val = 0
            for key, value in sess_data["parsed_events"][trial]["States"].items():
                if value == [None, None]:
                    continue
                else:
                    max_val = max(max_val, value[1])
            trial_end_ts_vect.append(max_val)
        trial_end_ts_dict[sessid] = trial_end_ts_vect
    return trial_end_ts_dict