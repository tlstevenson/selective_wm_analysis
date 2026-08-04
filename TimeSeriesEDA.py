# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:32:08 2026

@author: cns-th-lab
"""

import init
import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt

#Imports for model training
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity

#Imports for behavioral data
from hankslab_db import db_access
from hankslab_db import tonecatdelayresp_db as wm_db, basicRLtasks_db as bandit_db

#Imports for combination thresholding
from itertools import combinations

#Imports for convolution and visualization
from scipy.ndimage import convolve1d
import PredictionViewerSeqVis as pvsq

#%% File management functions
def get_h5_files_dir(label_dir_paths, sess_ids = []):
    """Get a list of h5 files in the specified directory either by session ids or in bulk
    
    Args:
        label_dir_paths (string[]): A list of strings corresponding to directories with labels
        
        sess_ids (string[]): A list of session ids to check for in the specified directories
        
    Returns:
        All h5 files in the label directory paths with specified ids"""
    #Ensure all directory paths exist
    for path in label_dir_paths:
        if not os.path.exists(path):
            raise ValueError(f"Path {path} does not exist for h5 searching.")
            
    
    label_files = []
    #Use specific sess_ids provided
    if len(sess_ids) > 0:
        for folder in label_dir_paths:
            for file in os.listdir(folder):
                name, ext = os.path.splitext(file)
                if ext == ".h5" and str.removeprefix(name, "mov_") in sess_ids:
                    label_files.append(os.path.join(folder, file))
    #Get all h5 files
    else:
        label_files = [os.path.join(folder, file) for folder in label_dir_paths for file in os.listdir(folder) if file.endswith(".h5")]
        
    return label_files

def get_port_file(rat_label_file, port_label_dirs):
    """Return the path of the port file corresponding to the rat label file.
    
    Assumes that both the rat and port video are labeled as mov_{sess_id}.slp.
    
    Args:
        rat_label_file (string): The specific path to the rat label file.
        
        port_label_dirs (string[]): A list of paths to port label directories.
        
    Returns:
        The corresponding port prediction for the given label_file"""
    port_file = None
    name_no_ext = os.path.splitext(os.path.basename(rat_label_file))[0]
    my_sess = str.removeprefix(name_no_ext, "mov_")
    for port_dir in port_label_dirs:
        for filename in os.listdir(port_dir):
            if my_sess in filename and ".h5" in filename:
                port_file = os.path.join(port_dir, filename)
                print(port_file)
                break
        if port_file != None:
            break
    
    if port_file == None:
        raise ValueError(f"File {os.path.basename(rat_label_file)} has no corresponding port label in provided port path directories.")
    return port_file

#%% Metric generation functions
def NodePositionsLocal(row, target_column, node_names, right_ortho=True):
    '''
    Returns the node positions in a local coordinate system.
    
    The first basis vector is from the body to the neck.
    The second basis vector is orthogonal and on the right side of the body.
    '''
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
    local_locations = np.einsum('fij, fnjk -> fnik', R, centered_pos)
    
    # Convert back to a list of arrays (each array being shape (nodes, 2, 1))
    return list(local_locations)

def CubicInterpolation(series, max_dist):
    """Interpolate with cubic polynomial equations.
    
    Args:
        series (float[]): a series of values where first axis is interpolated (frames x nodes x 2 x 1)
        
        max_dist (int): the maximum number of frames to interpolate
        
    Returns:
        The target column interpolated across time"""
    data_shape = np.shape(series)
    interpolated_series = np.copy(series)
    if data_shape[3] != 1:
        raise ValueError("Inputted series must have exactly one track.")
    for node in range(data_shape[1]): #Node
        for i in range(2): #2d points 
            my_series = pd.Series(np.squeeze(series)[:,node,i])
            interpolated_series[:,node,i,0] = my_series.interpolate(method="polynomial", order=3, limit=max_dist)
    return interpolated_series

def calc_velocity(coords):
    """Calculates Euclidean velocity from x and y coordinates.
    
    Args:
        coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        
    Returns:
        An array of velocities of size frames-1 x nodes x 1
    """
    dx = np.diff(coords[:, :, 0, :], axis=0)
    dy = np.diff(coords[:, :, 1, :], axis=0)
    velocities = np.sqrt(dx**2 + dy**2)
    return np.squeeze(velocities) #Removes the tracks dimension

def calculate_nan_proportions(video_row, node_names, curr_thresh_methods, all_mask_key_dict):
    """
    Calculates the proportion of NaNs/outliers for raw data and all combinations 
    of outlier detection methods for a single video.
    
    Args:
        video_row: A row from labels_df (dict or pd.Series)
        node_names: List of node names (strings)
        curr_thresh_methods (list[string]): name of the current methods as specified in all_thresh_key_dict
        all_thresh_key_dict (dict{string, string}): Dictionary for indexing df mask by threshold method   
        
    Returns:
        pd.DataFrame containing the proportions.
    """
    # 1. Extract raw tracks and determine dimensions
    tracks = np.array(video_row["tracks"])
    n_frames = tracks.shape[0]
    n_nodes = len(node_names)
    
    # Helper function to collapse masks/tracks safely to (frames, nodes)
    def collapse_to_node(array_data, is_raw=False):
        if is_raw:
            # Check for NaNs in raw data
            mask = np.isnan(array_data)
        else:
            # It's already a boolean mask, just ensure it's numpy
            mask = np.array(array_data, dtype=bool)
            
        # Reshape to flatten coordinate/track dimensions, then apply .any()
        return mask.reshape(n_frames, n_nodes, -1).any(axis=-1)

    # 2. Get the base masks (frames, nodes)
    raw_mask = collapse_to_node(tracks, is_raw=True)
    
    masks_dict = {}
    for threshold_method in curr_thresh_methods:
        mask_key = all_mask_key_dict[threshold_method]
        masks_dict[threshold_method] = collapse_to_node(video_row[mask_key])
    
    # 3. Initialize results dictionary with raw NaN proportions
    #TODO: Uncomment below
    proportions = {
        #"Raw Data": raw_mask.sum(axis=0) / n_frames
    }
    
    method_keys = list(masks_dict.keys())
    
    # 4. Loop through combinations of sizes 1 to 4
    for combo_size in range(1, len(method_keys) + 1):
        for combo in combinations(method_keys, combo_size):
            combo_name = " + ".join(combo)
            
            # Start with the first mask in the combination
            combined_mask = masks_dict[combo[0]]
            
            # Find the intersection (&) with the remaining masks in the combo
            for method in combo[1:]:
                combined_mask = combined_mask | masks_dict[method]
                
            # Calculate proportion and store
            proportions[combo_name] = combined_mask.sum(axis=0) / n_frames
            
    # 5. Convert to DataFrame (Rows = Nodes, Columns = Methods/Combinations)
    df_proportions = pd.DataFrame(proportions, index=node_names)
    
    return df_proportions

def calculate_post_interpolation_nans(interpolated_results, node_names):
    """
    Calculates the proportion of NaNs remaining in the signals AFTER interpolation.
    
    Args:
        interpolated_results(dict{string:float[,,,]}): Mapping of method to 
        interpolated locations (often from threshold_and_interpolate combinations)
        
        node_names (list[string]): node names in order as they appear in labels
    
    Returns:
        pd.DataFrame (Rows = Nodes, Columns = Method Combinations)
    """
    first_key = list(interpolated_results.keys())[0]
    n_frames = interpolated_results[first_key].shape[0]
    n_nodes = len(node_names)
    
    proportions = {}
    
    for combo_name, interp_tracks in interpolated_results.items():
        # Check which coordinates are still NaN (meaning gap was > max_dist)
        is_nan = np.isnan(interp_tracks) 
        
        # Collapse back to (frames, nodes) to get node-level proportions
        node_nan_mask = is_nan.reshape(n_frames, n_nodes, -1).any(axis=-1)
        
        # Calculate proportion
        proportions[combo_name] = node_nan_mask.sum(axis=0) / n_frames
        
    return pd.DataFrame(proportions, index=node_names)

def convolve_nans(nan_mask, window_size, sum_nodes=False):
    """
    Convolves a boolean NaN mask to calculate missing data density over time.
    
    Args:
        nan_mask (np.ndarray): Boolean array of shape (frames, nodes).
        window_size (int): Size of the rolling window (in frames).
        sum_nodes (bool): If True, sums NaNs across all nodes before convolving.
        
    Returns:
        np.ndarray: 1D array (frames,) if sum_nodes=True, or 2D array (frames, nodes) if False.
                    Values represent the count of NaNs within the window.
    """
    # Convert boolean mask to floats (1.0 for NaN, 0.0 for valid)
    data = np.array(nan_mask, dtype=float)
    
    # A boxcar window of 1s will count the exact number of NaNs in the window
    window = np.ones(window_size)
    
    if sum_nodes:
        # Sum across nodes first: shape becomes (frames,)
        #TODO: I think this assumes that all x NaNs are also y NaNs as there is no cap
        data_sum = np.sum(data, axis=1) 
        # mode='same' keeps the output size perfectly aligned with frame indices
        convolved = np.convolve(data_sum, window, mode='same')
        return convolved
    else:
        # Apply 1D convolution independently along the frames axis (axis=0) for each node
        convolved = convolve1d(data, window, axis=0, mode='constant', cval=0.0)
        return convolved
    
def extract_thresholded_sequences(convolved_data, thresholds):
    """
    Finds continuous sequences where the convolved NaN density exceeds thresholds.
    
    Args:
        convolved_data (np.ndarray): 1D or 2D convolved array from `convolve_nans`.
        thresholds (list of floats): The thresholds to evaluate.
        
    Returns:
        A nested dictionary mapped by threshold (and node index if 2D), 
        containing a list of sequence tuples (start, end) and percentage included.
    """
    n_frames = convolved_data.shape[0]
    
    # Helper to calculate sequences for a single 1D array
    def get_1d_sequences(signal_1d, thresh):
        exceeds = signal_1d > thresh
        
        # Pad with False at both ends so sequences starting at frame 0 or 
        # ending at the last frame are detected perfectly by np.diff
        padded = np.pad(exceeds, (1, 1), mode='constant', constant_values=False)
        diffs = np.diff(padded.astype(int))
        
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0] # Exclusive end index (standard Python slice logic)
        
        sequences = list(zip(starts, ends))
        included_frames = np.sum(exceeds)
        percentage = (included_frames / n_frames) * 100.0
        
        return {"sequences": sequences, "percentage": percentage}

    results = {}
    
    # Check if 1D (summed nodes) or 2D (per node)
    is_2d = convolved_data.ndim == 2
    
    for thresh in thresholds:
        if is_2d:
            node_results = {}
            for n_idx in range(convolved_data.shape[1]):
                node_results[n_idx] = get_1d_sequences(convolved_data[:, n_idx], thresh)
            results[thresh] = node_results
        else:
            results[thresh] = get_1d_sequences(convolved_data, thresh)
            
    return results
#%% Masking and thresholding functions

def ThresholdedPositions(positions, scores, threshold):
    """Return the mask and position values given by a certain score threshold
    
    Args:
        positions(float[,,,]): node positions in array frames x nodes x 2 x tracks
        
        scores(float[,,,]): scores for all predictions frames x nodes x 1 x tracks
        
        threshold: minimum score to include as valid in the final positions
        
    Returns:
        Thresholded positions in the same shape as input and NaN mask"""
    mask = scores < threshold
    mask = np.stack([mask, mask], axis = 2)
    
    #Isolate newly thresholded positions
    raw_mask = np.isnan(positions)
    exclusive_score_mask = mask & ~raw_mask
    
    thresholded_positions = np.copy(positions)
    thresholded_positions[exclusive_score_mask] = np.nan
    print(np.shape(mask))
    print(np.shape(thresholded_positions))
    return thresholded_positions, mask

def ThresholdedByVelocity(positions, threshold):
    """Get the node positions with velocities greater than the threshold removed, along with the mask that filters it
    
    Args:
        positions(float[,,,]): node positions in array frames x nodes x 2 x tracks 
        
        threshold (float /  np.array(float)): array for thresholding velocity
            Should be same shape as velocity or 1 less than the number of frames in positions if not float
        
    Returns:
        Thresholded positions in the same shape and a NaN mask"""
        
    thresholded_positions = np.copy(positions)
    velocities = calc_velocity(positions)
    
    mask = velocities > threshold
    print(np.shape(mask))
    mask = np.stack([mask, mask], axis = 2)
    print(np.shape(mask))
    mask = np.expand_dims(mask, axis=len(np.shape(mask)))
    print(np.shape(mask))
    #Add all false to the first (since missing 1 and velocity can't remove first)
    mask = np.insert(mask, 0, False, axis=0)
    print(np.shape(mask))
    thresholded_positions[mask] = np.nan
    
    return thresholded_positions, mask #Not is to transform nan mask -> valid

def get_iso_outliers(node_data, contamination=0.01):
    """Uses isolation forest (non-parametric) to determine local position outlier mask.
    
    Args:
        node_data (float[]): Position of the node over time for one track  (frames x 2)
        
        contamination (float): Expected proportion of observations that are outliers
    
    Returns:
        A boolean outlier mask of shape (frames,) for a single node. True = Outlier
    """
    valid_mask = ~np.isnan(node_data).any(axis=1)
    clean_data = node_data[valid_mask]
    
    # Initialize full mask with False (defaulting to inlier/NaN)
    outlier_mask = np.zeros(len(node_data), dtype=bool) 
    
    if len(clean_data) > 0:
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        preds = iso_forest.fit_predict(clean_data)
        
        # Isolation forest returns 1 for inliers, -1 for outliers
        outlier_mask[valid_mask] = (preds == -1)
        
    return outlier_mask

def get_kde_outliers(node_data, percentile_threshold=1.0, bandwidth=1.0):
    """Uses kernel density estimation (KDE) to get the outlier mask for 2d data.
    
    Args:
        node_data (float[]): Position of the node over time for one track  (frames x 2)
        
        percentile_threshold(float): Percent below which to consider an outlier
        
        bandwidth (float): spread of the kernel function (gaussian)
    Returns:
        Boolean outlier mask of shape (frames,) for a single node. True = Outlier.
    """
        
    valid_mask = ~np.isnan(node_data).any(axis=1)
    clean_data = node_data[valid_mask]
    
    outlier_mask = np.zeros(len(node_data), dtype=bool)
    
    if len(clean_data) > 0:
        kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
        kde.fit(clean_data)
        log_density_scores = kde.score_samples(clean_data)
        
        threshold = np.percentile(log_density_scores, percentile_threshold)
        
        outlier_mask[valid_mask] = (log_density_scores < threshold)
        
    return outlier_mask

def calculate_node_outliers(local_coords_param, method="iso", **kwargs):
    """
    Applies node-level outlier detection, thresholds the positions, and returns both.
    
    Args:
        local_coords_param (float[,,,]): a numpy array of local positions (from rotated_tracks)
        
        method (string): "iso" or "kde" whether you want isolated forest or KDE outliers
        
        **kwargs: 
            contamination (float): Expected proportion of observations that are outliers (iso)
            
            percentile_threshold(float): Percent below which to consider an outlier (kde)
            
            bandwidth (float): spread of the kernel function (gaussian) (kde)     
    Returns:
        Thresholded positions and the outlier mask obtained by the chosen method (kde/iso)
    """
    
    print(f"Starting node outlier detection with method: {method}")
    # Squeeze out the tracks dimension for calculation
    local_coords = np.squeeze(local_coords_param) 
    n_frames, n_nodes, _ = local_coords.shape
    
    # 1. Calculate 2D Node Mask (frames, nodes)
    node_outlier_mask = np.zeros((n_frames, n_nodes), dtype=bool)
    
    for n_idx in range(n_nodes):
        node_data = local_coords[:, n_idx, :]
        
        if method == "iso":
            node_outlier_mask[:, n_idx] = get_iso_outliers(node_data, **kwargs)
        elif method == "kde":
            node_outlier_mask[:, n_idx] = get_kde_outliers(node_data, **kwargs)
            
    # 2. Expand mask to match coordinate dimensions (frames, nodes, 2)
    full_mask = np.stack([node_outlier_mask, node_outlier_mask], axis=2)
    
    # 3. Apply to original positions array (preserving original shape)
    thresholded_positions = np.copy(local_coords_param)
    
    # If the original parameter was 4D (frames, nodes, 2, tracks), expand mask again
    if len(thresholded_positions.shape) == 4:
        full_mask = np.expand_dims(full_mask, axis=-1)
        
    thresholded_positions[full_mask] = np.nan
            
    return thresholded_positions, full_mask

#TODO: This is useless currently because it removes both edges out of caution
def calculate_edge_outliers(local_coords_param, edge_names, node_names, iqr_multiplier=1.5):
    """
    Removes outliers in edge length by IQR multiplier
    
    Args:
        local_coords_param (float[,,,]): a numpy array of local positions (from rotated_tracks)
        
        edge_names ([[string, string]]): pairs of names for edges extracted from labels file
        
        node_names (string[]): node_names in order as they appear in the labels file
        
        iqr_multiplier (float): multiplier beyond which a point is considered an outlier
    Returns:
        An outlier mask for the local_coords_param that can also be applied to raw data.
        NOTE: The current mask is overly cautious and removes both edge nodes
    """
    local_coords = np.squeeze(local_coords_param)
    n_frames, n_nodes, _ = local_coords.shape
    
    edge_idx = [[node_names.index(n1), node_names.index(n2)] for n1, n2 in edge_names]
    
    # Track outliers at the NODE level (frames, nodes)
    node_outlier_mask = np.zeros((n_frames, n_nodes), dtype=bool)
    
    for (n1, n2) in edge_idx:
        node_1_data = local_coords[:, n1, :]
        node_2_data = local_coords[:, n2, :]
        
        edge_vects = node_1_data - node_2_data
        edge_lengths = np.linalg.norm(edge_vects, axis=1) 
        
        valid_mask = ~np.isnan(edge_lengths)
        
        if np.any(valid_mask):
            valid_lengths = edge_lengths[valid_mask]
            
            q1, q3 = np.percentile(valid_lengths, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - (iqr_multiplier * iqr)
            upper_bound = q3 + (iqr_multiplier * iqr)
            
            # Find points outside the IQR bounds
            outliers = (edge_lengths < lower_bound) | (edge_lengths > upper_bound)
            outliers = outliers & valid_mask
            
            # Flag BOTH nodes connected to the invalid edge
            node_outlier_mask[:, n1] = node_outlier_mask[:, n1] | outliers
            node_outlier_mask[:, n2] = node_outlier_mask[:, n2] | outliers
            
    # Expand mask to coordinate dimensions (frames, nodes, 2)
    full_mask = np.stack([node_outlier_mask, node_outlier_mask], axis=2)
    
    thresholded_positions = np.copy(local_coords_param)
    
    # Re-add tracks dimension if needed to match original array
    if len(thresholded_positions.shape) == 4:
        full_mask = np.expand_dims(full_mask, axis=-1)
        
    thresholded_positions[full_mask] = np.nan
            
    return thresholded_positions, full_mask

def threshold_and_interpolate_combinations(video_row, node_names, curr_thresh_methods, all_mask_keys_dict, max_dist=15):
    """
    Applies every combination of outlier masks to the raw data, sets outliers to NaN,
    and applies cubic interpolation.
    
    Args:
        video_row (labels_df row): A video row from the video dataframe
        node_names (string[]): A list of node names in the labels
        curr_thresh_methods (string[]): All thresholding methods that should be used (See all_mask_keys_dict)
        all_mask_keys_dict(dict{string,string}): A dictionary of thresholding types and the mask locations in df
        max_dist (float): Largest frame gap to interpolate
    
    Returns:
        dict: A dictionary mapping the combination name to the (frames, nodes, 2, 1) interpolated array.
    """
    tracks = np.array(video_row["tracks"])
    n_frames = tracks.shape[0]
    n_nodes = len(node_names)
    
    # Helper to ensure masks are (frames, nodes) boolean arrays
    def get_node_mask(mask_data):
        return np.array(mask_data, dtype=bool).reshape(n_frames, n_nodes, -1).any(axis=-1)
    
    masks_dict = {}
    for threshold_method in curr_thresh_methods:
        mask_key = all_mask_keys_dict[threshold_method]
        masks_dict[threshold_method] = get_node_mask(video_row[mask_key])
    
    method_keys = list(masks_dict.keys())
    interpolated_results = {}
    
    # 1. Base Case: Interpolate raw data without any outlier masking
    interpolated_results["Raw Data"] = CubicInterpolation(tracks, max_dist)
    
    # 2. Iterate through all combinations
    for combo_size in range(1, len(method_keys) + 1):
        for combo in combinations(method_keys, combo_size):
            combo_name = " + ".join(combo)
            
            # Use Logical OR (|): If ANY method in the combo flags it, treat it as an outlier
            combined_mask = np.zeros((n_frames, n_nodes), dtype=bool)
            for method in combo:
                combined_mask = combined_mask | masks_dict[method]
                
            # Expand the 2D mask to match the 4D raw tracks (frames, nodes, 2, 1)
            # Broadcasting automatically applies the mask to both x and y coordinates
            expanded_mask = np.expand_dims(combined_mask, axis=(2, 3))
            expanded_mask = np.broadcast_to(expanded_mask, tracks.shape)
            
            # Mask the raw tracks
            tracks_thresh = np.copy(tracks)
            tracks_thresh[expanded_mask] = np.nan
            
            # Interpolate and store
            interpolated_results[combo_name] = CubicInterpolation(tracks_thresh, max_dist)
            
    return interpolated_results
#%% Plotting functions
def plot_velocity_time_traces(all_node_velocity, node_names=None):
    """Plots velocity time traces for each node.
    
    Args:
        all_node_velocity (float[,,]): Raw (including NaNs) velocity array [often output of calc_velocity()]
        
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
    """
    if node_names is None:
        node_names = np.arange(np.shape(all_node_velocity)[1])
    
    fig, ax = plt.subplots(len(node_names), 1, figsize=(10, 2 * len(node_names)), sharex=True)
    if len(node_names) == 1:
        ax = [ax]
        
    for i in range(len(node_names)):
        ax[i].plot(np.arange(np.shape(all_node_velocity)[0]), all_node_velocity[:, i])
        ax[i].set_title(node_names[i])
    plt.tight_layout()
    plt.show()

def plot_velocity_boxplots(all_node_velocity, node_names=None, vel_threshold=None):
    """Plots velocity boxplots by node and prints distribution stats.
    
    Args:
        all_node_velocity (float[,,]): Raw (including NaNs) velocity array [often output of calc_velocity()]
        
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        
        vel_threshold (float): An optional visualization for potential velocity outlier cutoff
    """
    if node_names is None:
        node_names = np.arange(np.shape(all_node_velocity)[1])
        
    cleaned_velocities = [col[~np.isnan(col)] for col in np.transpose(all_node_velocity)]
    x = np.arange(np.shape(all_node_velocity)[1]) #Use all_node_velocity since cleaned is non-homogenous
    
    plt.figure(figsize=(10, 5))
    if vel_threshold is not None:
        plt.axhline(y=vel_threshold, color="red", linestyle="--", label=f"Threshold ({vel_threshold})")
        
    plt.boxplot(cleaned_velocities, positions=x, tick_labels=node_names)
    if vel_threshold is not None:
        plt.legend()
    plt.ylabel("Velocity")
    plt.title("Velocity Distributions by Node")
    plt.show()
    
    # Print statistics
    std_vel = np.nanstd(all_node_velocity, axis=0)
    mean_vel = np.nanmean(all_node_velocity, axis=0)
    median_vel = np.nanmedian(all_node_velocity, axis=0)
    
    print("Velocity distributions:")
    for i in range(len(node_names)):
        print(f"{node_names[i]}: Mean({mean_vel[i]:.2f}) Median({median_vel[i]:.2f}) STD({std_vel[i]:.2f})")

def plot_nan_proportions(df_proportions):
    """
    Generates a grouped bar chart from the proportions DataFrame.
    
    Args:
        df_proportions: pandas dataframe from calculate_nan_proportions {method: proportion} -> dataframe
    """
    # Create the plot. 16 categories per node requires a wider figure.
    df_proportions.plot(kind='bar', figsize=(16, 8), width=0.85)
    
    plt.title("Proportion of Outliers by Node and Method Combination", fontsize=16)
    plt.ylabel("Proportion of Total Frames", fontsize=12)
    plt.xlabel("Nodes", fontsize=12)
    
    # Move the legend outside the plot so it doesn't cover the bars
    plt.legend(title="Methods (Intersection)", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    
    plt.xticks(rotation=45, ha="right")
    plt.ylim((0,1))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_post_interpolation_nans(df_proportions, interpolation_limit):
    """
    Plots a grouped bar chart of the remaining NaN proportions.
    
    Args:
        df_proportions (Dataframe): dataframe obtained from threshold_and_interpolate_combinations
        interpolation_limit (int): largest gap that was interpolated (should be stored and shared with previous steps)
    """
    df_proportions.plot(kind='bar', figsize=(16, 8), width=0.85)
    
    plt.title(f"Proportion of Gaps Remaining AFTER Interpolation (Limit={interpolation_limit})", fontsize=16)
    plt.ylabel("Proportion of Total Frames", fontsize=12)
    plt.xlabel("Nodes", fontsize=12)
    
    plt.legend(title="Filters Applied", bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
    
def plot_threshold_percentages(sequence_results, node_names=None):
    """
    Plots the percentage of the video included across thresholds.
    Handles both summed (1D) and per-node (2D) sequence results.
    """
    plt.figure(figsize=(10, 6))
    
    thresholds = sorted(list(sequence_results.keys()))
    
    # Check structure to see if it's node-level or network-level
    first_thresh = sequence_results[thresholds[0]]
    is_per_node = (0 in first_thresh) and ("percentage" not in first_thresh)
    
    if is_per_node:
        n_nodes = len(first_thresh.keys())
        for n_idx in range(n_nodes):
            pcts = [sequence_results[t][n_idx]["percentage"] for t in thresholds]
            label_name = node_names[n_idx] if node_names else f"Node {n_idx}"
            plt.plot(thresholds, pcts, marker='o', label=label_name)
            
        plt.title("Percentage of Video Included by Threshold (Per Node)", fontsize=14)
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    else:
        pcts = [sequence_results[t]["percentage"] for t in thresholds]
        plt.plot(thresholds, pcts, marker='o', color='b', linewidth=2)
        plt.title("Percentage of Video Included by Threshold (Network-Wide Sum)", fontsize=14)
        
    plt.xlabel("Threshold (NaNs in window)", fontsize=12)
    plt.ylabel("% of Total Frames Exceeding Threshold", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_convolved_signal(convolved_data, window_size, fps=None, node_names=None, threshold=None):
    """
    Plots the convolved NaN density over time.
    
    Args:
        convolved_data (np.ndarray): 1D or 2D convolved array from `convolve_nans`.
        window_size (int): The window size used (for title formatting).
        fps (float, optional): Frames per second. If provided, X-axis is in seconds.
        node_names (list, optional): List of node names for 2D data legend.
        threshold (float, optional): If provided, draws a horizontal threshold line.
    """
    plt.figure(figsize=(14, 6))
    
    n_frames = convolved_data.shape[0]
    
    # Determine X-axis (frames vs time)
    if fps is not None:
        x_axis = np.arange(n_frames) / fps
        x_label = "Time (seconds)"
    else:
        x_axis = np.arange(n_frames)
        x_label = "Frames"
        
    # Check if 1D (network-wide) or 2D (per node)
    is_2d = convolved_data.ndim == 2
    
    if is_2d:
        n_nodes = convolved_data.shape[1]
        for n_idx in range(n_nodes):
            label_name = node_names[n_idx] if node_names else f"Node {n_idx}"
            # Adding alpha makes it easier to read when many nodes overlap
            plt.plot(x_axis, convolved_data[:, n_idx])
            plt.title(f"{label_name} Missing Data Density (Rolling Window = {window_size} frames)", fontsize=14)
            plt.show()
        #plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
    else:
        plt.plot(x_axis, convolved_data, color='blue', linewidth=1.5)
        plt.title(f"Network-Wide Missing Data Density (Rolling Window = {window_size} frames)", fontsize=14)
        
        # Shade the area under the curve for readability
        plt.fill_between(x_axis, convolved_data, alpha=0.2, color='blue')

    # Draw threshold line if provided
    if threshold is not None:
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
        if not is_2d:
            plt.legend()
            
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("NaN Count in Window", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
#%% Path settings and execution start
sess_ids = ["116498"]#"116543"]#,"116498"] #Currently manually specified
label_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               ]
port_label_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260716_port_model"
                    ]
label_files = get_h5_files_dir(label_paths, sess_ids)
    
#%% Extract project wide data
with h5py.File(label_files[0], "r") as f:
    project_dict = {"node_names": [n.decode('utf-8') for n in f['node_names'][:]], 
                    "edge_names": [[n1.decode('utf-8'), n2.decode('utf-8')] for n1, n2 in f["edge_names"][:]]}
    project_dict["edge_inds"] = [[project_dict["node_names"].index(name_1),project_dict["node_names"].index(name_2)] 
                                 for name_1, name_2 in project_dict["edge_names"]]
    
with h5py.File(get_port_file(label_files[0], port_label_paths), "r") as g:
    project_dict["port_names"] = [n.decode('utf-8') for n in g['node_names'][:]]

#%% Extract session specific data
labels_dict = {"sess": [],
               "scores": [],
               "tracks": [],
               "port_tracks":[]}

for file in label_files:   
    with h5py.File(file, "r") as f:
        # Check that project shares structure
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        edge_names = [[n1.decode('utf-8'), n2.decode('utf-8')] for n1, n2 in f["edge_names"][:]]
        with h5py.File(get_port_file(file, port_label_paths), "r") as g:
            port_names = [n.decode('utf-8') for n in g['node_names'][:]]
            
        if project_dict["node_names"] != node_names:
            raise LookupError("Project structure not identical in nodes and indexing will fail.")
        if project_dict["edge_names"] != edge_names:
            raise LookupError("Project structure not identical in edges and indexing will fail.")
        if project_dict["port_names"] != port_names:
            raise LookupError("Project structure not identical in port names and indexing will fail.")
        
        # Get Prediction Scores 
        # Raw shape: (tracks, nodes, frames) -> Transposed: (frames, nodes, tracks)
        scores = np.transpose(f['point_scores'][:], (2, 1, 0)) 
        
        # Get Coordinates
        # Raw shape: (tracks, nodes, 2, frames) -> Transposed: (frames, nodes, 2, tracks)
        tracks_coords = np.transpose(f['tracks'][:])
        
        # Set Dictionary Values
        my_sess = str.removeprefix(os.path.splitext(os.path.basename(file))[0], "mov_")
        labels_dict["sess"].append(my_sess)
        labels_dict["scores"].append(scores)
        labels_dict["tracks"].append(tracks_coords)
        
        # Get corresponding port file info
        port_file = get_port_file(file, port_label_paths)
        # Read the data into port_names and a column of the df
        with h5py.File(port_file, "r") as g:
            print("Getting port tracks")
            port_tracks_coords = np.transpose(g['tracks'][:])
            print("Setting port tracks")
            labels_dict["port_tracks"].append(port_tracks_coords)
            
labels_df = pd.DataFrame(labels_dict)
labels_df = labels_df.set_index('sess')
#%% Threshold node labels by prediction scores
score_thresh = .3
labels_df[["score_thresh_tracks","score_nan_mask"]] = labels_df.apply(
    lambda row: ThresholdedPositions(row["tracks"], row["scores"], score_thresh), 
    axis=1, 
    result_type="expand")

#%% Threshold node labels by velocity
velocity_thresh = 50
labels_df[["vel_thresh_tracks","vel_nan_mask"]] = labels_df.apply(
    lambda row: ThresholdedByVelocity(row["tracks"], velocity_thresh), 
    axis=1,
    result_type="expand")

#%% Interpolate it (doesnt account for large gaps)
max_interpol_dist = 15
labels_df["cubic_interpol_tracks"] = labels_df.apply(
    lambda row: CubicInterpolation(row["score_thresh_tracks"], max_interpol_dist), 
    axis=1)

#%% Center and rotate it
labels_df["rotated_tracks"] = labels_df.apply(lambda row:NodePositionsLocal(row, "score_thresh_tracks", project_dict["node_names"]), axis=1)

#%% Isolation Forest Thresholding
labels_df[["iso_thresh_tracks", "iso_nan_mask"]] = labels_df.apply(
    lambda row: calculate_node_outliers(row["rotated_tracks"], method="iso", contamination=0.01), 
    axis=1, 
    result_type="expand")

#%% KDE Thresholding TAKES LONG
labels_df[["kde_thresh_tracks", "kde_nan_mask"]] = labels_df.apply(
    lambda row: calculate_node_outliers(row["rotated_tracks"], method="kde", percentile_threshold=1.0, bandwidth=2.0), 
    axis=1, 
    result_type="expand")

#%% Edge Length IQR Thresholding (using kde_thresh_tracks as input) NOT IMPLEMENTED PROPERLY
#TODO: Turn on once the bonelength thresholding works
labels_df[["edge_thresh_tracks", "edge_nan_mask"]] = labels_df.apply(
    lambda row: calculate_edge_outliers(row["rotated_tracks"], edge_names, node_names, iqr_multiplier=1.5), 
    axis=1,
    result_type="expand")

#%% Single vid thresholding analysis
methods_mask_keys = {"Score": "score_nan_mask",
                     "Velocity": "vel_nan_mask",
                     "ISO": "iso_nan_mask",
                     "KDE": "kde_nan_mask",
                     "Edge": "edge_nan_mask"}

#Plot various thresholding algorithms on first video (based on df entries)
first_video_row = labels_df.iloc[0]
proportions_df = calculate_nan_proportions(first_video_row, node_names, ["Score", "Velocity", "ISO"], methods_mask_keys)
plot_nan_proportions(proportions_df)

#%% Single vid interpolation analysis
first_video_row = labels_df.iloc[0]

# 1. Apply masks and interpolate
interpolation_limit = 15
interpolated_signals = threshold_and_interpolate_combinations(first_video_row, 
                                                              node_names, 
                                                              ["Score", "Velocity", "ISO"], 
                                                              methods_mask_keys, 
                                                              max_dist=interpolation_limit)

# 2. Calculate remaining NaNs
df_post_interp = calculate_post_interpolation_nans(interpolated_signals, node_names)

# 3. Plot the results
plot_post_interpolation_nans(df_post_interp, interpolation_limit)

#%% High NaN Sequence Extractor

# 1. Get a boolean (frames, nodes) mask. 
# Here we'll pretend we are using the Raw Data mask from the previous steps
tracks = np.array(labels_df.iloc[0]["tracks"])
n_frames, n_nodes = tracks.shape[0], len(node_names)
nan_mask = np.isnan(tracks).reshape(n_frames, n_nodes, -1).any(axis=-1)

# Set parameters
window_size = 60 # E.g., 30 frames (1 second if 30fps)
thresholds = np.linspace(0,window_size,60) # How many NaNs in the window triggers the threshold

# --- Path A: Sum across all nodes ---
convolved_sum = convolve_nans(nan_mask, window_size, sum_nodes=True)
seqs_sum = extract_thresholded_sequences(convolved_sum, thresholds)
plot_threshold_percentages(seqs_sum)

# --- Path B: Evaluate per node independently ---
convolved_nodes = convolve_nans(nan_mask, window_size, sum_nodes=False)
seqs_nodes = extract_thresholded_sequences(convolved_nodes, thresholds)
plot_threshold_percentages(seqs_nodes, node_names)
#%% Plot convolved NaN sequences

# Using the variables from the previous step:
# window_size = 30
# convolved_sum = convolve_nans(nan_mask, window_size, sum_nodes=True)
# convolved_nodes = convolve_nans(nan_mask, window_size, sum_nodes=False)

print("Plotting Network-Wide Density...")
plot_convolved_signal(
    convolved_sum, 
    window_size=window_size, 
    fps=30.0, 
    threshold=15
)

print("Plotting Per-Node Density...")
plot_convolved_signal(
    convolved_nodes, 
    window_size=window_size, 
    fps=30.0, 
    node_names=node_names, 
    threshold=5
)
#%% Run Prediction Viewer on Bad Nodes
# Assuming seqs_sum is the dictionary generated in the previous step
target_threshold = 15

# Get the list of (start, end) tuples for that specific threshold
my_bad_sequences = seqs_sum[target_threshold]["sequences"]

pvsq.RunApp(
    video_path="my_video.mp4", 
    inference_path="my_labels.h5", 
    output_window="Tracker", 
    fps=30,
    bad_sequences=my_bad_sequences # Pass the list here
)

#%% Single vid angle and local position analysis
#%%% Clean a video data
def generate_clean_batches(local_coords_param):
    """Extract valid local positions of nodes from a video tracking numpy array.
    
    Args:
        local_coords_param (float): A video tracking position array (frames, nodes, 2, tracks{should be 1})
        
    Returns:
        A non-homogenous list of lists by node of valid vectors (for use in K-means)
    """
    #Get local_coordinates for this video and remove tracks dimension
    local_coords = np.squeeze(local_coords_param)
    print(f"Initial shape: {np.shape(local_coords)}")
    
    valid_nodes_pos = []
    #Iterate through nodes, dropping NaN values
    for n_idx in range(np.shape(local_coords)[1]):
        node_data = local_coords[:,n_idx,:]
        print(f"Node data shape {n_idx}: {np.shape(node_data)}")
        nan_mask = np.isnan(node_data).any(axis=1)
        valid_mask = ~np.isnan(node_data).any(axis=1)
        print(f"Mask data shape {n_idx}: {np.shape(valid_mask)}")
        clean_data = node_data[valid_mask]
        print(f"Masked data shape {n_idx}: {np.shape(clean_data)}")
        valid_nodes_pos.append(clean_data)
        print(f"Number NaN entries: {np.sum(nan_mask)}")
        print(f"Non-NaN + NaN = Total: {np.sum(nan_mask) + np.shape(clean_data)[0] == np.shape(node_data)[0]}")
        print()
        
    return valid_nodes_pos

def scatter_node_local_pos(node_name, node_names, valid_local_positions):
    """Visualize the distribution of a node in local space
    
    Args:
        node_name: The node of interest
        node_names: The array of node names corresponding to the indices of 
            valid_local_positions
        valid_local_positions[[float]]: a non-homogenous list of lists of valid 
            local positions by node"""
    
    node_data = valid_local_positions[node_names.index(node_name)]
    plt.scatter(node_data[:,0], node_data[:,1], label=node_name)
    plt.title(f"{node_name}")
    #plt.show()

#TODO: REMOVE dependency on global variable edge_names
def hist_edge(edge_names, node_names, local_coords_param):
    """Plotting function for drawing histograms of limb length
    
    Args:
        edge_names [[string, string]]: List of node name pairs
        node_names [string]: list of node names
        all_nodes_valid_pos: non-homogenous list of valid (x,y) node positions by node
    
    Returns:
        Summary statistics by node"""
        
    #Convert edge name pairs to edge index pairs
    edge_idx = [[node_names.index(name_1), node_names.index(name_2)] for name_1, name_2 in edge_names]
    
    #Get local_coordinates for this video and remove tracks dimension
    local_coords = np.squeeze(local_coords_param)
    print(f"Initial shape: {np.shape(local_coords)}")
        
    #Iterate through edges, dropping NaN values where either is NaN
    for n1, n2 in edge_idx:
        node_1_data = local_coords[:,n1,:]
        node_2_data = local_coords[:,n2,:]
        
        valid_mask_1 = ~np.isnan(node_1_data).any(axis=1)
        valid_mask_2 = ~np.isnan(node_2_data).any(axis=1)
        valid_mask = valid_mask_1 & valid_mask_2
        print(f"Mask data shape: {np.shape(valid_mask)}")
        edge_vects = node_1_data[valid_mask] - node_2_data[valid_mask]
        print(f"Masked data shape: {np.shape(edge_vects)}")
        edge_lengths = np.linalg.norm(edge_vects, axis=1)
        
        plt.hist(edge_lengths, bins = 20)
        plt.title(f"{node_names[n1]} -> {node_names[n2]} Edge Length Histogram")
        plt.show()        
        
        plt.boxplot(edge_lengths)
        plt.title(f"{node_names[n1]} -> {node_names[n2]} Edge Length Boxplot")
        plt.show()
#%%%
hist_edge(edge_names, node_names, np.array(labels_df["rotated_tracks"].iloc[0]))
# %%% Outlier Detection Functions

def detect_outliers_isolation_forest(node_data, contamination=0.01):
    """
    Applies Isolation Forest to a single node's valid 2D positions.
    
    Returns:
        boolean mask (True means the point is an outlier)
    """
    # Initialize the model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit and predict (-1 for outliers, 1 for inliers)
    preds = iso_forest.fit_predict(node_data)
    
    # Convert to a boolean mask (True for outliers)
    outlier_mask = (preds == -1)
    return outlier_mask

def detect_outliers_kde(node_data, percentile_threshold=1.0, bandwidth=1.0):
    """
    Applies Kernel Density Estimation to a single node's valid 2D positions.
    
    Returns:
        boolean mask (True means the point is an outlier)
    """
    # Initialize and fit the KDE model
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(node_data)
    
    # Get log-density scores for all points
    log_density_scores = kde.score_samples(node_data)
    
    # Determine the density threshold based on the specified percentile (e.g., bottom 1%)
    threshold = np.percentile(log_density_scores, percentile_threshold)
    
    # Points with a density strictly less than the threshold are outliers
    outlier_mask = log_density_scores < threshold
    return outlier_mask

# %%% Visualization Function

def scatter_node_outliers(node_name, node_data, outlier_mask, algorithm_name):
    """
    Visualizes the inliers and outliers for a specific node.
    """
    inliers = node_data[~outlier_mask]
    outliers = node_data[outlier_mask]
    
    plt.figure(figsize=(8, 6))
    
    # Plot inliers (blue, slightly transparent)
    plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', s=5, alpha=0.3, label='Inliers')
    
    # Plot outliers (red, larger, opaque)
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=15, alpha=1.0, label='Outliers')
    
    plt.title(f"{node_name} - {algorithm_name} Outliers", fontsize=14)
    plt.xlabel("Local X", fontsize=12)
    plt.ylabel("Local Y", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

#%%% Plots scatterplots of local positions for nodes across a video

first_vid_clean_batches = generate_clean_batches(labels_df["rotated_tracks"].iloc[0])
for node in node_names:
    scatter_node_local_pos(node, node_names, first_vid_clean_batches)
plt.legend()
plt.show()
    
#%%% Plots scatterplots of local positions for nodes across a video (outliers)
print("Running Outlier Detection...\n")

for i, node in enumerate(node_names):
    # Get the (N, 2) array of clean data for this specific node
    node_data = first_vid_clean_batches[i]
    
    if len(node_data) == 0:
        print(f"Skipping {node}: No valid data points.")
        continue
        
    print(f"Processing {node} ({len(node_data)} points)...\n")
    
    # --- 1. Isolation Forest ---
    print(f"Running Isolation Forest...")
    iso_mask = detect_outliers_isolation_forest(node_data, contamination=0.01)
    
    # --- 2. Kernel Density Estimation ---
    print(f"Running KDE...")
    # bandwidth may need tuning depending on the scale of your local coordinates
    kde_mask = detect_outliers_kde(node_data, percentile_threshold=1.0, bandwidth=2.0)
    
    # Plot the results side-by-side to compare
    scatter_node_outliers(node, node_data, iso_mask, algorithm_name="Isolation Forest")
    scatter_node_outliers(node, node_data, kde_mask, algorithm_name="Kernel Density Estimation")

#TODO: Might be a good idea to add above/below port for nose to be able to tell which direction it's coming from
#%%% Plot whole frame with nose port angle labeled
for n in range(len(node_names)):
    print(n)
    #Plot body (inverted y to account for image coordinates)
    plt.scatter(labels_df["cubic_interpol_tracks"].iloc[0][0][n][0][0], labels_df["cubic_interpol_tracks"].iloc[0][0][n][1][0], color="red")
#Plot ports
plt.scatter(labels_df["port_tracks"].iloc[0][0][0][0][0], labels_df["port_tracks"].iloc[0][0][0][1][0], color="purple")
plt.annotate("Left Port", (labels_df["port_tracks"].iloc[0][0][0][0][0], labels_df["port_tracks"].iloc[0][0][0][1][0]))
plt.scatter(labels_df["port_tracks"].iloc[0][0][1][0][0], labels_df["port_tracks"].iloc[0][0][1][1][0], color="purple")
plt.annotate("Center Port", (labels_df["port_tracks"].iloc[0][0][1][0][0], labels_df["port_tracks"].iloc[0][0][1][1][0]))
plt.scatter(labels_df["port_tracks"].iloc[0][0][2][0][0], labels_df["port_tracks"].iloc[0][0][2][1][0], color="purple")
plt.annotate("Right Port", (labels_df["port_tracks"].iloc[0][0][2][0][0], labels_df["port_tracks"].iloc[0][0][2][1][0]))

nose_idx = project_dict["node_names"].index("nose")
implant_idx = project_dict["node_names"].index("implant")
#Plot arrows and write angle for visualization
nose_x = np.array(labels_df["cubic_interpol_tracks"].iloc[0])[0,nose_idx,0,0]
nose_y = np.array(labels_df["cubic_interpol_tracks"].iloc[0])[0,nose_idx,1,0]
implant_x = np.array(labels_df["cubic_interpol_tracks"].iloc[0])[0,implant_idx,0,0]
implant_y = np.array(labels_df["cubic_interpol_tracks"].iloc[0])[0,implant_idx,1,0]
port_0_x = np.array(labels_df["port_tracks"].iloc[0])[0,0,0,0]
port_0_y = np.array(labels_df["port_tracks"].iloc[0])[0,0,1,0]
port_1_x = np.array(labels_df["port_tracks"].iloc[0])[0,1,0,0]
port_1_y = np.array(labels_df["port_tracks"].iloc[0])[0,1,1,0]
port_2_x = np.array(labels_df["port_tracks"].iloc[0])[0,2,0,0]
port_2_y = np.array(labels_df["port_tracks"].iloc[0])[0,2,1,0]
#Implant-nose
plt.quiver(implant_x, implant_y, 
           nose_x-implant_x, nose_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='blue')
#Implant-port_x
plt.quiver(implant_x, implant_y,
           port_0_x-implant_x, port_0_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='orange', label=angles_0[0])
plt.quiver(implant_x, implant_y,
           port_1_x-implant_x, port_1_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='yellow', label=angles_1[0])
plt.quiver(implant_x, implant_y,
           port_2_x-implant_x, port_2_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='green', label=angles_2[0])
plt.xlim(0,1440)
plt.ylim(1080,0) #Inverted for image
plt.legend()
plt.show()
#%% Access fp and behavioral data
#%%% Read trial data from database 
wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

wm_sess_data = wm_loc_db.get_behavior_data(sess_ids)
bandit_sess_data = bandit_loc_db.get_behavior_data(sess_ids)
#%%% Helper functions for trial ends and intervals
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
        start_idx = np.searchsorted(frame_timestamps, start_time, side='left')
        end_idx = np.searchsorted(frame_timestamps, end_time, side='right')
        idx_intervals.append([start_idx, end_idx])
        
        # Slice the coordinates array using the found frame indices
        interval_coords = coords[start_idx:end_idx]
        pose_data_list.append(interval_coords)
    
    return pose_data_list, idx_intervals
#%%% Print wm_sess data
print(sess_ids)
print(wm_sess_data.head())
print(wm_sess_data.columns.tolist())
print(wm_sess_data.iloc[0])
#%%% Print bandit sess data
print(sess_ids)
print(bandit_sess_data.head())
print(bandit_sess_data.columns.tolist())
#%%% Print trial start times (No NaNs)
trial_starts_plus_last = db_access.get_fp_trial_start_ts(sess_ids)[int(sess_ids[0])]
trial_starts = trial_starts_plus_last[:-1]
trial_ends_rel_trial_start = get_trial_end_ts(wm_sess_data)
trial_ends = trial_starts + trial_ends_rel_trial_start
print(np.shape(trial_starts))
print(np.shape(trial_ends))
print(f"Num NaNs: {np.sum(np.isnan(trial_starts))}")
print(f"Num NaNs: {np.sum(np.isnan(trial_ends))}")
#%%% Print center poke in times (NaNs for invalid)
print(len(wm_sess_data["cpoke_in_time"]))
print(f"Num NaNs: {np.sum(np.isnan(wm_sess_data['cpoke_in_time']))}")
print(wm_sess_data["cpoke_in_time"].tolist())
cpoke_in_times_vid = trial_starts + wm_sess_data["cpoke_in_time"].tolist()
print(f"Num NaNs: {np.sum(np.isnan(cpoke_in_times_vid))}")
print(cpoke_in_times_vid)
cpoke_in_times_vid_f = 30 * cpoke_in_times_vid #TODO: Paramterize frame rate at the top

#%%% Read video doric times
from sys_neuro_tools import doric_utils as du
active_sess_vid_doric = r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\mov_116498.doric"
du.h5print(active_sess_vid_doric)
time_in, time_in_info = du.h5read(active_sess_vid_doric,['DataAcquisition','BehaviorCamera','Video','Series0001','DMK-33UX290','Time']);
print(time_in)
print(time_in_info)
#%%% Various Intervals
#%%%% Between the end of one and start of next
intervals = np.transpose(np.stack((trial_ends[:-1], trial_starts[1:])))
#%%%% Between start and end
intervals = np.transpose(np.stack((trial_ends[:-1], trial_starts[1:])))
#%%%% Between response cue and response poke
response_cue_abs_time = trial_starts + wm_sess_data["response_cue_time"].tolist()
response_abs_time = trial_starts + wm_sess_data["response_time"].tolist()
intervals = np.transpose(np.stack((response_cue_abs_time, response_abs_time)))
print(intervals)
#%%% Plot interval duration histogram
interval_dur = [end - start for start, end in intervals]
plt.hist(interval_dur,bins=30)
plt.show()

#%%% Use the intervals to slice the pose data
print(f"Shape of intervals: {np.shape(intervals)}")
segmented_poses, segment_idxs = pose_in_intervals(time_in, labels_df["cubic_interpol_tracks"][0], intervals)
for segment in segmented_poses:
    print(np.shape(segment))
#%%% 2d scatter plot of nose position in interval color coded by time
x_sub_1 = []
y_sub_1 = []
t_sub_1 = []
x_1_2 = []
y_1_2 = []
t_1_2 = []
x_2_m = []
y_2_m = []
t_2_m = []
for i in range(len(segmented_poses)):
    abs_t = np.array(pd.Series(time_in[segment_idxs[i][0]:segment_idxs[i][1]]).interpolate())
    if np.shape(abs_t)[0] == 0:
        continue
    rel_t = abs_t - abs_t[0]
    print(rel_t[-1])
    if rel_t[-1] < 1:
        x_sub_1 = x_sub_1 + pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
        y_sub_1 = y_sub_1 + pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
        t_sub_1 = t_sub_1 + rel_t.tolist()
    elif rel_t[-1] > 2:
        x_1_2 = x_1_2 + pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
        y_1_2 = y_1_2 + pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
        t_1_2 = t_1_2 + rel_t.tolist()
    else:
        x_2_m = x_2_m + pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
        y_2_m = y_2_m + pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
        t_2_m = t_2_m + rel_t.tolist()
fig, ax = plt.subplots(nrows=2, ncols=2)
a = ax[0][0].scatter(x_sub_1,y_sub_1,c=t_sub_1, cmap='inferno')
ax[0][0].set_xlim(0,1440)
ax[0][0].set_ylim(1080, 0) #Inverted for image
b = ax[0][1].scatter(x_1_2, y_1_2, c=t_1_2, cmap='inferno')
ax[0][1].set_xlim(0,1440)
ax[0][1].set_ylim(1080, 0) #Inverted for image
c = ax[1][0].scatter(x_2_m, y_2_m, c=t_2_m, cmap='inferno')
ax[1][0].set_xlim(0,1440)
ax[1][0].set_ylim(1080, 0) #Inverted for image
fig.colorbar(a, ax=ax)
fig.colorbar(b, ax=ax)
fig.colorbar(c, ax=ax)
plt.show()
#%%% 3d scatter plot of nose position in interval color coded by time and 
#separated on z axis by time
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(segmented_poses)):
    abs_t = np.array(pd.Series(time_in[segment_idxs[i][0]:segment_idxs[i][1]]).interpolate())
    if np.shape(abs_t)[0] == 0:
        continue
    rel_t = abs_t - abs_t[0]
    x = pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
    y = pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
    t = rel_t.tolist()
    ax.scatter(x,y,t)
ax.set_xlim(0,1440)
ax.set_ylim(1080, 0) #Inverted for image
plt.show()
#%%% Align trajectories
#%%%% Function
def simple_align(segmented_poses, segment_idxs, n_bins=50):
    original_shape = np.shape(segmented_poses[0])
    aligned_trajectories = np.zeros((len(segmented_poses), n_bins, original_shape[1], original_shape[2]))
    for i in range(len(segmented_poses)):
        print(f"Segment #{i+1}")
        if segment_idxs[i][0] >= len(time_in) or segment_idxs[i][1] >= len(time_in):
            continue
        t_start = time_in[segment_idxs[i][0]]
        t_end = time_in[segment_idxs[i][1]]
        time_cut = time_in[segment_idxs[i][0]:segment_idxs[i][1]]
        pose_cut = segmented_poses[i]
        
        t_est = np.linspace(t_start, t_end, n_bins)
        if i==0:
            print(t_start)
            print(t_end)
            print(t_est)
            print()
            print(np.shape(t_est))
            print(np.shape(pose_cut[:,0,i,0]))
            print(np.shape(time_cut))
            print()
        for node in range(np.shape(pose_cut)[1]):
            for j in range(2):
                aligned_trajectories[i,:,node,j] = np.interp(t_est, time_cut, pose_cut[:,node, j, 0])
                print(pose_cut[:,node, j, 0])
                print(aligned_trajectories[i,:,node,j])
    return aligned_trajectories
    
aligned = simple_align(segmented_poses, segment_idxs) #Trials x bins(norm_time) x nodes x 2
for segment in aligned:
    plt.scatter(segment[:,0,0], segment[:,0,1], c=np.arange(0,50))
plt.show()
#%%%% Execution
from sys_neuro_tools import fp_utils as fpu
data = np.array(labels_df["cubic_interpol_tracks"].iloc[0])
data_timestamps = time_in
start_timestamps = intervals[:,0]
end_timestamps = intervals[:,1]
n_bins = 100
aligned = fpu.build_time_norm_signal_matrix(data, data_timestamps, start_timestamps, end_timestamps, n_bins)
print(np.shape(aligned))