#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:16:56 2026

@author: alex
"""

import h5py
import numpy as np
import pandas as pd
import shutil
from pathlib import Path

def interpolate_and_save_h5(filepath, max_gap_length=None):
    """
    Loads position data from a SLEAP h5 file, linearly interpolates missing frames,
    and saves the cleaned data to a new file ending in '_basic.h5'.
    
    Parameters:
    - filepath (str): Path to the original .h5 file.
    - max_gap_length (int or None): Maximum consecutive NaN frames to fill.
    
    Returns:
    - locations (np.array): Squeezed, interpolated shape (frames, nodes, 2)
    - node_names (list): List of node names.
    - new_filepath (str): The path to the newly saved file.
    """
    # 1. Generate the new filename (e.g., video1.h5 -> video1_basic.h5)
    path_obj = Path(filepath)
    new_filepath = path_obj.with_name(f"{path_obj.stem}_basic{path_obj.suffix}")
    
    # 2. Copy the original file to preserve all SLEAP metadata/structure
    shutil.copy(filepath, new_filepath)
    print(f"Created copy for interpolation: {new_filepath}")
    
    # 3. Open the NEW file in read/write mode ("r+")
    with h5py.File(new_filepath, "r+") as f:
        # Read the tracks and transpose to: (frames, nodes, coords, tracks)
        locations = f["tracks"][:].T 
        node_names = [n.decode() for n in f["node_names"][:]]
        
        # Copy the array so we can modify it safely
        interpolated_locations = np.copy(locations)
        frames, nodes, coords, tracks = interpolated_locations.shape
        
        # 4. Iterate through every track, node, and coordinate
        for track_idx in range(tracks):
            for node_idx in range(nodes):
                for coord_idx in range(coords): 
                    
                    series = pd.Series(interpolated_locations[:, node_idx, coord_idx, track_idx])
                    
                    interpolated_series = series.interpolate(
                        method='linear', 
                        limit=max_gap_length, 
                        limit_area='inside'
                    )
                    
                    interpolated_locations[:, node_idx, coord_idx, track_idx] = interpolated_series.to_numpy()
                    
        # 5. Overwrite the dataset in the new file!
        # We MUST transpose it back (.T) so it returns to SLEAP's native shape: 
        # (tracks, coords, nodes, frames)
        f["tracks"][...] = interpolated_locations.T
        
    # --- Format for Python Returns ---
    # Just like before, we squeeze out the tracks dimension if there's only 1 animal 
    # so it plays nicely with your plotting functions.
    if interpolated_locations.shape[-1] == 1:
        return_locations = np.squeeze(interpolated_locations, axis=-1)
    else:
        return_locations = interpolated_locations[:, :, :, 0]
        print(f"Warning: Multiple tracks. Returning Track 0 to Python environment.")
        
    return return_locations, node_names, str(new_filepath)

# --- Example Usage ---
# locs, names, saved_path = interpolate_and_save_h5("my_video.h5", max_gap_length=5)
# print(f"Successfully saved to: {saved_path}")