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
    print(sess_data["parsed_events"][0]["States"])
    print(type(sess_data["parsed_events"][0]["States"]))
    trial_end_ts_vect = []
    for trial in range(len(sess_data["parsed_events"])):
        max_val = 0
        for key, value in sess_data["parsed_events"][trial]["States"].items():
            if value == [None, None]:
                continue
            else:
                max_val = max(max_val, value[1])
        trial_end_ts_vect.append(max_val)
    return trial_end_ts_vect


def pose_in_intervals(frame_timestamps, coords, intervals):
    """A function that gets a non-homogenous list of coordinates by time intervals.

    It uses the time since the start from the intervals to find the correct frames
    in the video and get a list of pose_data for each interval

    Args:
        frame_timestamps (float[]): 1d array of timestamps for each frame
        coords (float[,,,]): frames x nodes x 2 x tracks SLEAP array
        intervals (List<(float, float)>): intervals in which to get the data

    Returns:
        A non-homogenous list of slices corresponding to the intervals."""
    pose_data_list = []
    idx_intervals = []
    print(np.shape(frame_timestamps))
    print(np.shape(coords))
    print(np.shape(intervals))
    for start_time, end_time in intervals:
        print(start_time)
        print(end_time)
        start_idx = np.searchsorted(frame_timestamps, start_time, side="left")
        end_idx = np.searchsorted(frame_timestamps, end_time, side="right")
        idx_intervals.append([start_idx, end_idx])

        # Slice the coordinates array using the found frame indices
        interval_coords = coords[start_idx:end_idx]
        pose_data_list.append(interval_coords)

    return pose_data_list, idx_intervals