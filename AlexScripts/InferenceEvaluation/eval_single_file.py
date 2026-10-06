#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1: Single Inference File Analysis
Profiles missing data, calculates kinematics, transforms coordinates, 
flags geometric outliers, and applies time-bounded/static interpolation for a single video.
"""

import numpy as np
import pandas as pd
import h5py
import json
import os
import math
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d, gaussian_filter1d
from tkinter import Tk, filedialog, messagebox

# =============================================================================
# 1. Missing Data & Gap Profiling
# =============================================================================
def measure_nan_gaps(s: pd.Series):
    out = pd.Series(0, index=s.index, dtype=int)
    is_nan = s.isna()
    starts = is_nan & ~is_nan.shift(1, fill_value=False)
    blocks = (~is_nan).cumsum()
    gap_sizes = is_nan.groupby(blocks).sum().astype(int)
    gap_sizes = gap_sizes[gap_sizes > 0]
    
    if len(out.loc[starts]) == len(gap_sizes.values):
        out.loc[starts] = gap_sizes.values
    return out, np.where(starts)[0], gap_sizes.values

def nan_gap_spike_graph(my_coords, node_names=None, columns=4):
    if node_names is None: node_names = np.arange(np.shape(my_coords)[1])
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns, figsize=(15, 8))
    ax = np.array(ax).flatten()
    for i in range(len(node_names)):
        nan_out, nan_starts, _ = measure_nan_gaps(pd.Series(my_coords[:, i, 0]))
        ax[i].bar(np.arange(np.shape(my_coords)[0])[nan_starts], nan_out.iloc[nan_starts], label=node_names[i])
        ax[i].legend()
        ax[i].set_yscale('log')
        ax[i].set_xlim(0, len(my_coords))
    plt.suptitle("Spike Graph: NaN Gap Starts and Lengths")
    plt.tight_layout()
    plt.show()

def nan_heatplot(my_coords, node_names=None, columns=4):
    if node_names is None: node_names = np.arange(np.shape(my_coords)[1])
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns, figsize=(15, 8))
    ax = np.array(ax).flatten()
    for i in range(len(node_names)):
        is_nan = np.isnan(my_coords[:, i, 0])
        nan_x_points = np.arange(np.shape(my_coords)[0])[is_nan]
        ax[i].bar(nan_x_points, np.ones(len(nan_x_points)))
        ax[i].set_title(node_names[i])
        ax[i].set_xlim(0, len(my_coords))
    plt.suptitle("Heatplot: NaN Locations")
    plt.tight_layout()
    plt.show()

def plot_distr_nan_gaps(my_coords, node_names=None, columns=4):
    if node_names is None: node_names = np.arange(np.shape(my_coords)[1])
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns, figsize=(15, 8))
    ax = np.array(ax).flatten()
    for i in range(len(node_names)):
        _, _, nan_gap_sizes = measure_nan_gaps(pd.Series(my_coords[:, i, 0]))
        if len(nan_gap_sizes) > 0:
            ax[i].hist(nan_gap_sizes, label=node_names[i], bins=20)
        ax[i].legend()
    plt.suptitle("Distribution of Gap Lengths")
    plt.tight_layout()
    plt.show()

def analyze_nan_binned_zscores(coords, bin_length_frames=30):
    num_nan_per_node = np.sum(np.isnan(coords[:, :, 0]), axis=0)
    tot_prop_nan = num_nan_per_node / np.shape(coords)[0]
    
    prop_bins = []
    lower_bound = 0
    while lower_bound < np.shape(coords)[0]:
        bin_coords = coords[lower_bound:min(lower_bound + bin_length_frames, np.shape(coords)[0]), :, 0]
        prop_bins.append(np.sum(np.isnan(bin_coords), axis=0) / bin_length_frames)
        lower_bound += bin_length_frames
        
    prop_bins = np.array(prop_bins)
    std_prop_nan = np.std(prop_bins, axis=0)
    std_prop_nan = np.where(std_prop_nan == 0, 1e-8, std_prop_nan)
    bin_zs = (prop_bins - tot_prop_nan) / std_prop_nan
    
    plt.figure(figsize=(8, 6))
    plt.imshow(bin_zs.T, cmap="cividis", aspect="auto")
    plt.colorbar(label="Z-score")
    plt.title(f"NaN Binned Z-Scores (Bin={bin_length_frames} frames)")
    plt.xlabel("Bins")
    plt.ylabel("Nodes")
    plt.show()

def plot_single_file_nans(coords, node_names):
    nan_counts = np.sum(np.isnan(coords[:, :, 0]), axis=0)
    prop_nans = nan_counts / coords.shape[0]
    
    plt.figure(figsize=(12, 5))
    plt.bar(node_names, prop_nans, color="steelblue")
    plt.title("Proportion of Missing Frames by Node (Single Video)")
    plt.ylabel("Proportion of NaNs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_single_file_scores(scores, node_names):
    valid_scores = [scores[:, i][~np.isnan(scores[:, i])] for i in range(len(node_names))]
    plt.figure(figsize=(14, 6))
    plt.boxplot(valid_scores, tick_labels=node_names, patch_artist=True,
                boxprops=dict(facecolor="#a1c9f4", color="black"),
                medianprops=dict(color="red", linewidth=1.5))
    plt.title("Prediction Confidence Scores by Node (Single Video)")
    plt.ylabel("Confidence Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. Kinematic & Spatial Computations
# =============================================================================
def calc_velocity(coords):
    dx = np.diff(coords[:, :, 0], axis=0)
    dy = np.diff(coords[:, :, 1], axis=0)
    return np.sqrt(dx**2 + dy**2)

def plot_velocity_time_traces(all_node_velocity, node_names=None):
    if node_names is None: node_names = np.arange(np.shape(all_node_velocity)[1])
    fig, ax = plt.subplots(len(node_names), 1, figsize=(10, 2 * len(node_names)), sharex=True)
    if len(node_names) == 1: ax = [ax]
    for i in range(len(node_names)):
        ax[i].plot(np.arange(np.shape(all_node_velocity)[0]), all_node_velocity[:, i])
        ax[i].set_title(node_names[i])
    plt.tight_layout()
    plt.show()

def NodePositionsLocal(coords, node_names):
    origin_idx, basis_idx = node_names.index("spine_2"), node_names.index("spine_1")
    p_origin = coords[:, origin_idx, :]
    p_basis = coords[:, basis_idx, :]
    centered = coords - p_origin[:, np.newaxis, :]
    
    b1 = p_basis - p_origin
    norms = np.linalg.norm(b1, axis=1, keepdims=True)
    u = b1 / np.where(norms == 0, 1.0, norms)
    
    R = np.empty((coords.shape[0], 2, 2))
    R[:, 0, 0], R[:, 0, 1] = u[:, 0], u[:, 1]
    R[:, 1, 0], R[:, 1, 1] = -u[:, 1], u[:, 0]
    return np.einsum("fij, fnj -> fni", R, centered)

def get_angle(p_origin, p1, p2):
    v1, v2 = p1 - p_origin, p2 - p_origin
    v1_n = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    dot_product = np.sum(v1_n * v2_n, axis=1)
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

def hist_edge(edge_names, node_names, local_coords):
    edge_idx = [[node_names.index(n1), node_names.index(n2)] for n1, n2 in edge_names]
    for n1, n2 in edge_idx:
        edge_vects = local_coords[:, n1, :] - local_coords[:, n2, :]
        valid_mask = ~np.isnan(edge_vects).any(axis=1)
        edge_lengths = np.linalg.norm(edge_vects[valid_mask], axis=1)
        if len(edge_lengths) > 0:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.hist(edge_lengths, bins=20)
            plt.title(f"{node_names[n1]} -> {node_names[n2]} Histogram")
            plt.subplot(1, 2, 2)
            plt.boxplot(edge_lengths)
            plt.title(f"{node_names[n1]} -> {node_names[n2]} Boxplot")
            plt.show()

# =============================================================================
# 3. Outlier Detection & Anomaly Categorization
# =============================================================================
def remove_anatomical_outliers(coords, head_idx, nose_idx, neck_idx):
    angles = get_angle(coords[:, head_idx], coords[:, nose_idx], coords[:, neck_idx])
    q25, q75 = np.nanpercentile(angles, [25, 75])
    iqr = q75 - q25
    outlier_mask = (angles < (q25 - 3 * iqr)) | (angles > (q75 + 3 * iqr))
    coords_clean = np.copy(coords)
    coords_clean[outlier_mask, nose_idx, :] = np.nan
    return coords_clean, outlier_mask

def plot_outlier_scatter_optimized(local_coords, outlier_mask, node_name):
    inliers = local_coords[~outlier_mask]
    outliers = local_coords[outlier_mask]
    plt.figure(figsize=(6, 6))
    plt.plot(inliers[:, 0], inliers[:, 1], linestyle='none', marker='.', color="blue", markersize=2, alpha=0.3, label="Inliers")
    if len(outliers) > 0:
        plt.plot(outliers[:, 0], outliers[:, 1], linestyle='none', marker='.', color="red", markersize=6, alpha=1.0, label="Outliers")
    plt.title(f"Local Spatial Outliers: {node_name}")
    plt.xlabel("Local X")
    plt.ylabel("Local Y")
    plt.legend()
    plt.axis("equal")
    plt.show()

# =============================================================================
# 4. Smoothing & Interpolation Testing
# =============================================================================
def interpolate_cubic_bounded(coords, max_gap_frames=15):
    interp_coords = np.copy(coords)
    for node in range(coords.shape[1]):
        for axis in range(2):
            series = pd.Series(coords[:, node, axis])
            is_nan = series.isna()
            blocks = (~is_nan).cumsum()
            gap_sizes = is_nan.groupby(blocks).transform('sum')
            valid_gaps = is_nan & (gap_sizes <= max_gap_frames)
            full_interp = series.interpolate(method="pchip", limit_area="inside")
            interp_coords[valid_gaps, node, axis] = full_interp[valid_gaps]
    return interp_coords

def find_qualified_static_gaps(coords, max_flank_dist=50.0):
    qualified_gaps = {n_idx: [] for n_idx in range(coords.shape[1])}
    for n_idx in range(coords.shape[1]):
        is_nan = np.isnan(coords[:, n_idx, 0])
        starts = np.where(is_nan & ~np.roll(is_nan, 1))[0]
        stops = np.where(~is_nan & np.roll(is_nan, 1))[0]
        if is_nan[0]: starts = np.insert(starts, 0, 0)
        if is_nan[-1]: stops = np.append(stops, len(is_nan))
        for s, e in zip(starts, stops):
            if s > 0 and e < coords.shape[0]:
                p_before, p_after = coords[s-1, n_idx], coords[e, n_idx]
                if not np.isnan(p_before).any() and not np.isnan(p_after).any():
                    if np.linalg.norm(p_after - p_before) <= max_flank_dist:
                        qualified_gaps[n_idx].append((s, e))
    return qualified_gaps

def interpolate_qualified_gaps(coords, qualified_gaps):
    interp_coords = np.copy(coords)
    for n_idx, gaps in qualified_gaps.items():
        if not gaps: continue
        for ax in range(2):
            series = pd.Series(coords[:, n_idx, ax])
            full_interp = series.interpolate(method="pchip", limit_area="inside")
            for s, e in gaps:
                interp_coords[s:e, n_idx, ax] = full_interp.iloc[s:e]
    return interp_coords

# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    PREFS_FILE = "script1_prefs.json"
    prefs = {}
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f: prefs = json.load(f)

    root = Tk()
    root.withdraw()
    
    h5_path = None
    saved_file = prefs.get("last_h5_file")
    
    if saved_file and os.path.exists(saved_file):
        if messagebox.askyesno("Use Saved Preference", f"Use previously selected file?\n\n{saved_file}"):
            h5_path = saved_file
            
    if not h5_path:
        h5_path = filedialog.askopenfilename(
            title="Select Inference .h5 File",
            initialdir=prefs.get("last_h5_dir", os.getcwd()),
            filetypes=[("HDF5 Files", "*.h5")]
        )
        
        if h5_path:
            prefs["last_h5_file"] = h5_path
            prefs["last_h5_dir"] = os.path.dirname(h5_path)
            if messagebox.askyesno("Save Preferences", "Overwrite saved default file for next time?"):
                with open(PREFS_FILE, "w") as f: json.dump(prefs, f)
    
    if h5_path:
        print(f"Loading {h5_path}...")
        with h5py.File(h5_path, "r") as f:
            coords = f['tracks'][:].T
            if coords.ndim == 4: coords = coords[..., 0]
            scores = np.transpose(f['point_scores'][:], (2, 1, 0))[..., 0]
            node_names = [n.decode('utf-8') for n in f['node_names'][:]]
            
        print("Executing Analyses...")
        
        # Whole Skeleton Summaries
        plot_single_file_scores(scores, node_names)
        plot_single_file_nans(coords, node_names)
        
        # Missing Data Profiling
        nan_gap_spike_graph(coords, node_names)
        nan_heatplot(coords, node_names)
        plot_distr_nan_gaps(coords, node_names)
        analyze_nan_binned_zscores(coords)
        
        # Kinematics
        velocities = calc_velocity(coords)
        plot_velocity_time_traces(velocities, node_names)
        
        if "spine_2" in node_names and "spine_1" in node_names:
            local_coords = NodePositionsLocal(coords, node_names)
            if "nose" in node_names and "implant" in node_names:
                hist_edge([["implant", "nose"]], node_names, local_coords)
            
            # Outlier Spatial Visualizations
            if all(n in node_names for n in ["nose", "implant", "neck"]):
                clean_coords, out_mask = remove_anatomical_outliers(
                    coords, node_names.index("implant"), node_names.index("nose"), node_names.index("neck")
                )
                plot_outlier_scatter_optimized(local_coords[:, node_names.index("nose"), :], out_mask, "nose")

        # Interpolation
        q_gaps = find_qualified_static_gaps(coords, 50.0)
        static_interp = interpolate_qualified_gaps(coords, q_gaps)
        bounded_interp = interpolate_cubic_bounded(static_interp)
        
        print("Analysis complete. Processed tensors kept in memory.")