# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:44:11 2026

@author: cns-th-lab
"""

import os
import glob
import json
import cv2
import numpy as np
import pandas as pd
import deeplabcut

def run_dlc_inference(vid_path, dlc_config_path):
    print("Running DeepLabCut Inference...")
    deeplabcut.analyze_videos(dlc_config_path, [vid_path], save_as_csv=False)
    
    vid_dir = os.path.dirname(vid_path)
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    h5_files = glob.glob(os.path.join(vid_dir, f"{vid_name}*.h5"))
    
    if not h5_files:
        raise FileNotFoundError("DLC output .h5 file not found.")
    
    latest_h5 = max(h5_files, key=os.path.getctime)
    df = pd.read_hdf(latest_h5)
    df.columns = df.columns.droplevel('scorer') # Simplify dataframe access
    return df

def calculate_normalized_mismatch(sleap_data, dlc_df, shared_nodes):
    print("Calculating Normalized Discrepancies...")
    mismatch_scores = {}
    
    # Variance normalization 
    node_variances = {}
    for node in shared_nodes:
        if node in dlc_df.columns.levels[0]:
            x_var = dlc_df[node]['x'].var()
            y_var = dlc_df[node]['y'].var()
            node_variances[node] = np.sqrt(x_var + y_var) + 1e-6 
    
    for str_frame_idx, s_data in sleap_data.items():
        frame_idx = int(str_frame_idx) # JSON keys are always strings
        if frame_idx >= len(dlc_df):
            continue 
            
        dlc_frame = dlc_df.iloc[frame_idx]
        frame_distance = 0
        valid_nodes = 0
        
        for node in shared_nodes:
            if node in s_data['coords'] and node in dlc_frame:
                sx, sy = s_data['coords'][node]
                dx, dy = dlc_frame[node]['x'], dlc_frame[node]['y']
                
                dist = np.sqrt((sx - dx)**2 + (sy - dy)**2)
                norm_dist = dist / node_variances[node]
                
                frame_distance += norm_dist
                valid_nodes += 1
                
        if valid_nodes > 0:
            mismatch_scores[frame_idx] = frame_distance / valid_nodes
            
    return mismatch_scores

def extract_and_save_frames(vid_path, frames_to_extract, dlc_project_path):
    vid_name = os.path.splitext(os.path.basename(vid_path))[0]
    target_dir = os.path.join(dlc_project_path, "labeled-data", vid_name)
    os.makedirs(target_dir, exist_ok=True)
    
    print(f"Extracting {len(frames_to_extract)} unique frames to {target_dir}...")
    cap = cv2.VideoCapture(vid_path)
    extracted_count = 0
    
    for frame_idx in sorted(list(frames_to_extract)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            img_name = f"img{str(frame_idx).zfill(4)}.png"
            cv2.imwrite(os.path.join(target_dir, img_name), frame)
            extracted_count += 1
            
    cap.release()
    print(f"Successfully extracted {extracted_count} edge-case frames!")

if __name__ == "__main__":
    # --- Parameters to Adjust ---
    DLC_PROJECT_PATH = r"C:\Path\To\Your\DLC_Project"
    DLC_CONFIG = os.path.join(DLC_PROJECT_PATH, "config.yaml")
    
    # Point this to the exact JSON file Script 1 just generated
    SLEAP_JSON_PATH = r"C:\Path\To\Your\Raw_Videos\my_random_video_sleap_results.json"
    
    SHARED_NODES = ['snout', 'left_ear', 'right_ear', 'tail_base'] 
    
    N_SLEAP = 15   # Worst SLEAP predictions
    M_DLC = 15     # Worst DLC predictions
    P_DIST = 20    # Highest discrepancy between models
    # -----------------------------

    # Step 1: Load SLEAP Data
    with open(SLEAP_JSON_PATH, 'r') as f:
        loaded_data = json.load(f)
    vid_path = loaded_data['video_path']
    sleap_data = loaded_data['data']
    
    # Step 2: Run DLC Inference
    dlc_df = run_dlc_inference(vid_path, DLC_CONFIG)
    
    # Step 3: Find Worst SLEAP Frames 
    sleap_sorted = sorted(sleap_data.items(), key=lambda x: x[1]['score'])
    worst_sleap = [int(f[0]) for f in sleap_sorted[:N_SLEAP]]
    
    # Step 4: Find Worst DLC Frames 
    dlc_likelihoods = dlc_df.xs('likelihood', level='coords', axis=1).mean(axis=1)
    worst_dlc = dlc_likelihoods.nsmallest(M_DLC).index.tolist()
    
    # Step 5: Find Mismatches 
    mismatches = calculate_normalized_mismatch(sleap_data, dlc_df, SHARED_NODES)
    mismatch_sorted = sorted(mismatches.items(), key=lambda x: x[1], reverse=True)
    worst_mismatch = [f[0] for f in mismatch_sorted[:P_DIST]]
    
    # Step 6: Combine & Extract
    all_target_frames = set(worst_sleap + worst_dlc + worst_mismatch)
    extract_and_save_frames(vid_path, all_target_frames, DLC_PROJECT_PATH)