# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 15:57:18 2026

@author: cns-th-lab
"""

import numpy as np
import pandas as pd


def NodePositionsLocal(row, target_column, node_names, right_ortho=True):
    """
    Returns the node positions in a local coordinate system.

    The first basis vector is from the body to the neck.
    The second basis vector is orthogonal and on the right side of the body.
    """
    # Ensure session data is a NumPy array: shape (frames, nodes, 2, 1)
    session = row[target_column]

    origin_idx = node_names.index("spine_2")
    basis_idx = node_names.index("spine_1")

    # 1. Extract origin and basis points (Shape: frames, 2, 1)
    p_origin = session[:, origin_idx, :, :]
    p_basis = session[:, basis_idx, :, :]

    # 2. Center nodes relative to the origin
    # We add a new axis for 'nodes' so it broadcasts to (frames, nodes, 2, 1)
    centered_pos = session - p_origin[:, np.newaxis, :, :]

    # 3. Calculate basis vectors
    # We index [..., 0] to temporarily ignore the trailing 1, making math easier (Shape: frames, 2)
    b1 = p_basis[..., 0] - p_origin[..., 0]

    # Calculate norms and protect against divide-by-zero
    norms = np.linalg.norm(b1, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)

    # Normalize to unit vectors
    u = b1 / norms

    # Extract X and Y unit components
    u_x = u[:, 0]
    u_y = u[:, 1]

    # 4. Build the rotation matrices for all frames (Shape: frames, 2, 2)
    frames = session.shape[0]
    R = np.empty((frames, 2, 2))

    R[:, 0, 0] = u_x
    R[:, 0, 1] = u_y
    R[:, 1, 0] = -u_y
    R[:, 1, 1] = u_x

    # 5. Apply rotations
    # f = frames, n = nodes, i = new coord (2), j = old coord (2), k = trailing dim (1)
    # This precisely multiplies the 2x2 matrix into the (2, 1) vector for every node in every frame.
    local_locations = np.einsum("fij, fnjk -> fnik", R, centered_pos)

    # Convert back to a list of arrays (each array being shape (nodes, 2, 1))
    return list(local_locations)

def get_sequence_intervals(boolean_mask):
    """A function to return the start and end indices of NaN sequences
    
    Args:
        boolean_mask (bool[]): a 1d array with potential NaNs
        
    Returns:
        A list of (start, end) int tuples for the interval indexes"""

    # Pad with False at both ends so sequences starting at frame 0 or
    # ending at the last frame are detected perfectly by np.diff
    padded = np.pad(boolean_mask, (1, 1), mode="constant", constant_values=False)
    diffs = np.diff(padded.astype(int))

    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] # Exclusive end index (standard Python slice logic)

    sequences = list(zip(starts, ends))
    return sequences

def interpolate_cubic(series, max_dist):
    """
    Interpolates NaN gaps in coordinate series using cubic polynomial (pchip).
    Only fills gaps that are less than or equal to max_dist in length.
    
    Args:
        series (np.ndarray): Coordinate array of shape (frames, nodes, 2, 1).
        max_dist (int): Maximum gap length (in frames) to interpolate.
        
    Returns:
        np.ndarray: A new array with qualified gaps interpolated.
    """
    if series.shape[3] != 1:
        raise ValueError("Inputted series must have exactly one track.")
        
    interpolated_series = np.copy(series)
    n_nodes = series.shape[1]
    
    for node in range(n_nodes):
        for coord in range(2):  # x=0, y=1
            # 1. Extract the 1D signal
            signal = np.squeeze(series)[:, node, coord]
            
            # 2. Find all NaN gaps
            nan_mask = np.isnan(signal)
            gaps = get_sequence_intervals(nan_mask)
            
            # 3. Filter for gaps smaller than or equal to the frame limit
            valid_gaps = [(s, e) for s, e in gaps if (e - s) <= max_dist]
            
            # Skip full interpolation if there are no small gaps to fill
            if not valid_gaps:
                continue
                
            # 4. Fully interpolate the signal (no limit applied here)
            # limit_area="inside" ensures we don't extrapolate beyond the first/last valid frames
            full_interp = pd.Series(signal).interpolate(
                method="pchip", 
                limit_area="inside"
            )
            
            # 5. Selectively patch only the valid gaps back into the array
            for start, end in valid_gaps:
                interpolated_series[start:end, node, coord, 0] = full_interp.iloc[start:end]
                
    return interpolated_series