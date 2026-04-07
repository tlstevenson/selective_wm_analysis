# -*- coding: utf-8 -*-
"""
Created on Wed Feb 25 23:10:22 2026

@author: cns-th-lab
"""
import os
import sys

import sleap_io as sio
from pathlib import Path

def convert_slp_to_analysis_h5(slp_path, output_path=None):
    """
    Converts a SLEAP prediction file with no tracks into a 
    valid Analysis HDF5 file for DISK.
    """
    if not os.path.exists(slp_path):
        raise Warning(f"The file {slp_path} does not exist or could not be found. Track assignment aborted.")
        return
    
    #TEMPORARY: Dont reformat if it already exists as an analysis file
    if os.path.exists(output_path):
        return str(output_path)
    
    # 1. Load the labels (very fast with sleap-io)
    labels = sio.load_slp(slp_path)
    
    # 2. Create a dummy track if none exist
    if len(labels.tracks) == 0:
        single_track = sio.Track(name="track_0")
        labels.tracks.append(single_track)
    else:
        single_track = labels.tracks[0]

    # 3. Force every instance into this track
    for lf in labels.labeled_frames:
        for inst in lf.instances:
            inst.track = single_track

    # 4. Define output path if not provided
    if output_path is None:
        output_path = Path(slp_path).with_suffix(".analysis.h5")

    # 5. Save using the analysis-specific function
    # all_frames=True is vital for DISK to maintain the time-series continuity
    sio.io.main.save_analysis_h5(
        labels=labels, 
        filename=str(output_path), 
        all_frames=True
    )
    
    print(f"Successfully created: {output_path}")
    return str(output_path)

print(convert_slp_to_analysis_h5(sys.argv[1], output_path=sys.argv[2]))
