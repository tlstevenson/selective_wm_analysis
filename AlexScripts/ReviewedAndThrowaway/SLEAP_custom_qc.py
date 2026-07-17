import os
import h5py
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================
# Helper Functions
# ==========================================
def get_gap_mask(series, min_gap, max_gap):
    is_nan = series.isna()
    gap_ids = (is_nan != is_nan.shift()).cumsum()
    gap_sizes = is_nan.groupby(gap_ids).transform('sum')
    return is_nan & (gap_sizes >= min_gap) & (gap_sizes <= max_gap)

def collapse_instances(tracks, point_scores):
    """
    Scans all instances and collapses them into a single identity by selecting
    the X/Y coordinates with the highest confidence score for each node, per frame.
    """
    frames, nodes, dims, instances = tracks.shape
    
    # Replace NaNs with -1 so we can safely use argmax to find the highest score
    safe_scores = np.nan_to_num(point_scores, nan=-1.0)
    
    # Find the instance index with the highest score per frame and node
    best_inst_idx = np.argmax(safe_scores, axis=2) # Shape: (frames, nodes)
    
    # Create advanced indexing grids
    f_idx, n_idx = np.indices((frames, nodes))
    
    # Extract the winning tracks and scores
    collapsed_tracks = tracks[f_idx, n_idx, :, best_inst_idx]
    collapsed_scores = point_scores[f_idx, n_idx, best_inst_idx]
    
    # Re-add the instance dimension (size 1) at the end so the rest of the pipeline works
    collapsed_tracks = np.expand_dims(collapsed_tracks, axis=-1)
    collapsed_scores = np.expand_dims(collapsed_scores, axis=-1)
    
    return collapsed_tracks, collapsed_scores

# ==========================================
# Math Operations
# ==========================================
def strip_outliers(tracks, point_scores, conf_threshold=0.5, vel_threshold=30.0):
    cleaned_tracks = tracks.copy()
    frames, nodes, dims, instances = cleaned_tracks.shape
    print(instances)
    
    # Confidence Filter
    low_conf_mask = point_scores < conf_threshold
    for d in range(dims):
        cleaned_tracks[:, :, d, :][low_conf_mask] = np.nan
        
    # Velocity Filter
    diff = np.diff(cleaned_tracks, axis=0)
    velocity = np.linalg.norm(diff, axis=2)
    high_vel_mask = velocity > vel_threshold
    full_high_vel_mask = np.zeros((frames, nodes, instances), dtype=bool)
    full_high_vel_mask[1:] = high_vel_mask
    
    for d in range(dims):
        cleaned_tracks[:, :, d, :][full_high_vel_mask] = np.nan
        
    return cleaned_tracks

def interpolate_and_smooth(tracks, instance=0):
    interp_tracks = tracks.copy()
    frames, nodes, dims, instances = interp_tracks.shape
    
    # Spline (< 10 frames)
    for n in range(nodes):
        for d in range(dims):
            series = pd.Series(interp_tracks[:, n, d, instance])
            valid_gaps = get_gap_mask(series, min_gap=1, max_gap=9)
            if valid_gaps.any():
                interp_series = series.interpolate(method='cubicspline', limit_direction='both')
                interp_tracks[valid_gaps, n, d, instance] = interp_series[valid_gaps]

    # Global Smoothing
    """
    window_size = 5 
    for n in range(nodes):
        for d in range(dims):
            series = pd.Series(interp_tracks[:, n, d, instance])
            smoothed = series.rolling(window=window_size, min_periods=1, center=True).mean()
            valid_mask = ~np.isnan(smoothed)
            interp_tracks[valid_mask, n, d, instance] = smoothed[valid_mask]"""
            
    return interp_tracks

# ==========================================
# File Saving Operation
# ==========================================
def save_in_place(original_h5_path, new_h5_path, new_tracks_array):
    """Duplicates the original h5 file and overwrites the tracks dataset."""
    shutil.copy2(original_h5_path, new_h5_path)
    
    with h5py.File(new_h5_path, 'r+') as f:
        del f['tracks']
        # Transpose back to SLEAP's native shape: (instances, 2, nodes, frames)
        f.create_dataset('tracks', data=new_tracks_array.T)

# ==========================================
# Batch Processing Execution
# ==========================================
if __name__ == "__main__":
    
    # predictions_dir = Path(r"C:\Users\cns-th-lab\SLEAP_Projects\predictions")
    
    # Grab all .slp files (ignoring ones with CLEANED/STRIPPED suffixes if any exist)
    #slp_files = [f for f in predictions_dir.iterdir() if f.is_file() and f.suffix == ".slp" and "CLEANED" not in f.name]
    slp_files = [Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-07-28.mov_0001.proj.slp"),
                 Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-07-29.mov_0002.proj.slp"),
                 Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-07-30.mov_0003.proj.slp"),
                 Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-07-31.mov_0004.proj.slp"),
                 Path(r"C:\Users\cns-th-lab\SLEAP_Labels_4_29_26\198\Videos\198.2025-08-01.mov_0005.proj.slp")]
    
    print(f"Found {len(slp_files)} .slp projects to process.\n" + "="*40)
    
    for slp_path in slp_files:
        print(f"\nProcessing: {slp_path.name}")
        
        # Define expected paths
        analysis_h5_path = slp_path.parent / f"{slp_path.name}.analysis.h5" 
        stripped_h5_path = slp_path.parent / f"{slp_path.stem}_STRIPPED.h5"
        interp_h5_path = slp_path.parent / f"{slp_path.stem}_INTERPOLATED.h5"
        
        # 1. Run uv sleap export via subprocess
        print(f"  -> Exporting to {analysis_h5_path.name} via CLI...")
        command = ["uv", "run", "sleap", "export", str(slp_path), "-o", str(analysis_h5_path)]
        
        try:
            # shell=False is safer, check=True ensures it throws an error if it fails
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as e:
            print(f"  -> CLI Export failed for {slp_path.name}. Skipping.")
            continue
            
        if not analysis_h5_path.exists():
            print(f"  -> Error: Exported file {analysis_h5_path.name} not found. Skipping.")
            continue
            
        # 2. Load Raw Arrays from the newly generated file
        print("  -> Loading arrays and applying math...")
        with h5py.File(analysis_h5_path, 'r') as f:
            raw_tracks = f['tracks'][:].T.astype(float)  
            point_scores = f['point_scores'][:].T
            
        # 2.5 Collapse 120 fragmented instances down to 1
        raw_tracks, point_scores = collapse_instances(raw_tracks, point_scores)
            
        # 3. Strip Outliers
        stripped_tracks = strip_outliers(raw_tracks, point_scores, conf_threshold=0.2, vel_threshold=50.0)
        
        # 4. Save File 1 (Stripped)
        save_in_place(analysis_h5_path, stripped_h5_path, stripped_tracks)
        print(f"  -> Saved: {stripped_h5_path.name}")
        
        # 5. Interpolate & Smooth
        interpolated_tracks = interpolate_and_smooth(stripped_tracks, instance=0)
        
        # 6. Save File 2 (Interpolated)
        save_in_place(analysis_h5_path, interp_h5_path, interpolated_tracks)
        print(f"  -> Saved: {interp_h5_path.name}")
        
        # 7. Cleanup the intermediate analysis.h5 file so you just have the final two
        os.remove(analysis_h5_path)

    print("\nBatch processing complete!")
    