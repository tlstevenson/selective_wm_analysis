# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 21:52:56 2026

@author: cns-th-lab
"""

import cv2
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks
import sleap_io as sio

# ==========================================
# 1. Visualization Function
# ==========================================
def plot_frame_predictions(frame_idx, slp_path, stripped_h5, interp_h5):
    """
    Plots the predictions from the original .slp, stripped .h5, and interpolated .h5 
    over the original video frame.
    """
    print(f"Generating plot for frame {frame_idx}...")
    
    # --- A. Load the Video Frame ---
    labels = sio.load_slp(str(slp_path))
    video_path = labels.videos[0].filename
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"  -> Error: Could not read frame {frame_idx} from {video_path}")
        return

    # Convert BGR (OpenCV default) to RGB (Matplotlib default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- B. Extract Points from SLP (Original) ---
    orig_x, orig_y = [], []
    for lf in labels.find(video=labels.videos[0], frame_idx=frame_idx):
        for inst in lf.instances:
            for node in inst.skeleton.nodes:
                try:
                    # Try to grab the point directly, bypassing the broken 'in' operator
                    pt = inst[node]
                    
                    # Check that the point exists, is visible, and isn't a NaN
                    if getattr(pt, 'visible', True) and not np.isnan(pt.x):
                        orig_x.append(pt.x)
                        orig_y.append(pt.y)
                except (KeyError, IndexError, ValueError, TypeError):
                    # If the point doesn't exist or isn't formatted right, skip it
                    continue

    # --- C. Extract Points from Stripped H5 ---
    with h5py.File(stripped_h5, 'r') as f:
        strp_x = f['tracks'][0, 0, :, frame_idx]
        strp_y = f['tracks'][0, 1, :, frame_idx]

    # --- D. Extract Points from Interpolated H5 ---
    with h5py.File(interp_h5, 'r') as f:
        int_x = f['tracks'][0, 0, :, frame_idx]
        int_y = f['tracks'][0, 1, :, frame_idx]

    # --- E. Plotting ---
    plt.figure(figsize=(12, 10))
    plt.imshow(frame_rgb)
    
    # Layer 1: Interpolated (Red Stars, Plotted first so it acts as a background highlight)
    plt.scatter(int_x, int_y, c='red', marker='*', s=250, label='Interpolated (Spline)', zorder=1)
    
    # Layer 2: Original (Gray Circles, semi-transparent)
    plt.scatter(orig_x, orig_y, c='lightgray', marker='o', s=100, alpha=0.7, edgecolors='black', label='Original (.slp)', zorder=2)
    
    # Layer 3: Stripped (Cyan Dots, plotted last to show confident points clearly)
    plt.scatter(strp_x, strp_y, c='cyan', marker='o', s=30, label='Stripped (Clean)', zorder=3)

    plt.title(f"Pose Predictions - Frame {frame_idx}", fontsize=16)
    plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. Priority Scoring & Maxima Finder
# ==========================================
def find_resolved_worst_gaps_and_plot(slp_path, stripped_h5, interp_h5, node_priorities, window_size=15):
    """
    Calculates a penalty score based on missing nodes (2*high + 1*low).
    Finds the top 5 local maxima of these scores where the interpolated data
    contains ZERO NaNs, then plots them.
    """
    print("Calculating missing data scores based on node priorities...")
    
    # --- A. Load Data ---
    with h5py.File(stripped_h5, 'r') as f:
        strp_x = f['tracks'][0, 0, :, :]
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        
    with h5py.File(interp_h5, 'r') as f:
        int_x = f['tracks'][0, 0, :, :]

    # Mask of where data is missing (NaN)
    missing_mask = np.isnan(strp_x)           # Shape: (nodes, frames)
    interp_missing_mask = np.isnan(int_x)     # Shape: (nodes, frames)
    
    # --- B. Apply Weights & Calculate Labels ---
    weights = np.zeros(len(node_names))
    for i, name in enumerate(node_names):
        prio = node_priorities.get(name, 'low').lower()
        weights[i] = 2 if prio == 'high' else 1

    # Raw label per frame: Sum of (is_missing * weight)
    raw_labels = np.sum(missing_mask * weights[:, None], axis=0)

    # Average of the labels (rolling mean to find sustained bad regions)
    smoothed_labels = pd.Series(raw_labels).rolling(window=window_size, center=True, min_periods=1).mean().to_numpy()

    # --- C. Find Local Maxima ---
    peaks, _ = find_peaks(smoothed_labels)

    # --- D. Filter & Sort ---
    # Condition: The interpolated data at this frame index must have NO missing data
    resolved_peaks = [p for p in peaks if not interp_missing_mask[:, p].any()]

    if not resolved_peaks:
        print("Warning: Could not find any local maxima frames where interpolation perfectly resolved all NaNs.")
        return

    # Sort peaks by their smoothed label (highest missing data score first)
    resolved_peaks.sort(key=lambda p: smoothed_labels[p], reverse=True)
    
    top_5_frames = resolved_peaks[:5]
    print(f"Top 5 successfully resolved frames with worst missing data: {top_5_frames}\n" + "-"*40)

    # --- E. Plot ---
    for frame_idx in top_5_frames:
        plot_frame_predictions(frame_idx, slp_path, stripped_h5, interp_h5)

# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    
    # 1. Define Paths
    base_dir = Path(r"C:/Users/cns-th-lab/SLEAP_Projects/EENI6CT")
    slp_file = base_dir / "single.199_1.600_model.slp"
    stripped_file = base_dir / "single.199_1.600_model_STRIPPED.h5"
    interp_file = base_dir / "single.199_1.600_model_INTERPOLATED.h5"
    
    # 2. Designate Node Priorities
    # Replace these keys with the actual node names from your SLEAP project
    my_priorities = {
        'nose': 'high',
        'implant': 'high',
        'ear_l': 'low',
        'ear_r': 'low',
        'cheek_l': 'low',
        'cheek_r': 'low',
        'body_end': 'low',
        'contour_1_l': 'high',
        'contour_1_r': 'high',
        'contour_2_l': 'high',
        'contour_2_r': 'high',
        'contour_3_l': 'high',
        'contour_3_r': 'high',
        'spine_1': 'high',
        'spine_2': 'high',
        'spine_3': 'high',
        'tail_base': 'high',
        'tail_mid': 'low',
        'tail_end': 'low'
    }
    
    # 3. Run the finder
    if slp_file.exists() and stripped_file.exists() and interp_file.exists():
        find_resolved_worst_gaps_and_plot(
            slp_path=slp_file, 
            stripped_h5=stripped_file, 
            interp_h5=interp_file, 
            node_priorities=my_priorities,
            window_size=15 # Rolling average window size
        )
    else:
        print("Error: Could not find one or more of the required files.")