# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "numpy",
#     "sleap",
#     "pandas"
# ]
# ///

# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:43:29 2026

@author: cns-th-lab
"""

import os
import random
import subprocess
import numpy as np
import pandas as pd
import sleap
from sleap.qc import LabelQCDetector
#import logging


def get_random_video(parent_dir, extension=".mp4"):
    """Finds and returns a random video from the specified directory tree."""
    video_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith(extension):
                video_files.append(os.path.join(root, file))
    
    if not video_files:
        raise FileNotFoundError(f"No {extension} videos found in {parent_dir}")
    
    selected_vid = random.choice(video_files)
    print(f"Randomly selected video: {selected_vid}")
    return selected_vid

"""def run_sleap_inference(vid_path, sleap_centroid_model, sleap_centered_model, output_slp):
    Runs SLEAP tracking on the full video, forcing low-confidence predictions, and extracts scores.
    print("\n--- Running SLEAP Top-Down Inference CLI ---")
        
    # Command to run on the full video with a lowered peak threshold
    command = f'sleap track -i {vid_path} -m {sleap_centroid_model} -m {sleap_centered_model} -o {output_slp} -t --peak_threshold 0.01'
    
    print("Starting SLEAP tracking inference subprocess...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,  
        stderr=subprocess.PIPE,  
        text=True                
    )
    
    stdout, stderr = process.communicate()
    
    if process.returncode != 0:
        print(f"SLEAP process failed with exit code {process.returncode}")
        print(stderr)
        raise RuntimeError("SLEAP tracking failed.")
    
    print(f"Tracking complete. Loading predicted data from {output_slp}...")
    labels = sleap.load(output_slp)
    
    # Extract prediction scores into a list of dictionaries
    frame_data_list = []
    for frame in labels:
        if not frame.instances:
            continue
        
        instance = frame.instances[0]
        row_data = {'frame_idx': frame.frame_idx}
        
        # Add the score for every tracked body part
        for node, p in instance.points.items():
            row_data[node.name] = float(p.score) if p.score is not None else np.nan
            
        frame_data_list.append(row_data)
        
    # Convert to DataFrame
    df = pd.DataFrame(frame_data_list)
    if not df.empty:
        df.set_index('frame_idx', inplace=True)
        df.sort_index(inplace=True)
        
    return df, output_slp"""

def run_sleap_inference(vid_path, centroid_model_path, centered_model_path, output_path):
    print(f"\n--- Running SLEAP PyTorch CLI ---")
    
    # Build the EXACT string that you confirmed works in your command line.
    # Wrapping the variables in double quotes handles any spaces in your file paths.
    command_str = (
        rf'uv run sleap track '
        rf'--data_path "{vid_path}" '
        rf'--model_paths "{centroid_model_path}" '
        rf'--model_paths "{centered_model_path}" '
        rf'--output_path "{output_path}" '
        rf'--tracking'
    )
    
    print(f"Executing Command:\n{command_str}\n")
    
    try:
        # shell=True takes the raw string and runs it exactly as if you pasted it into cmd
        process = subprocess.Popen(command_str, shell=True)
        process.wait()
        
        if process.returncode != 0:
            print(f"\nSLEAP tracking failed with exit code {process.returncode}")
            return None, None
        else:
            print("\nInference completed successfully!")
            
    except Exception as e:
        print(f"Subprocess launch failed: {e}")
        return None, None

    # (Replace this placeholder with your actual Pandas extraction logic)
    scores_df = "Placeholder_Scores" 
    
    return scores_df, output_path

def run_quality_control(gt_path, pred_path):
    """Uses SLEAP's LabelQCDetector to find the worst predicted frames based on ground-truth data."""
    print("\n--- Running SLEAP Quality Control ---")
    print(f"Loading user labels from: {gt_path}")
    user_labels = sleap.load(gt_path)

    print(f"Loading predictions from: {pred_path}")
    pred_labels = sleap.load(pred_path)

    print("Initializing LabelQCDetector and fitting the GMM to user-labeled data...")
    qc_detector = LabelQCDetector()
    qc_detector.fit(user_labels)

    print("Scanning predictions for anomalies (jitter, gross misses, swaps)...")
    qc_results = qc_detector.detect(pred_labels)

    # Convert results to DataFrame and sort by the worst scores
    qc_df = qc_results.to_dataframe()
    worst_frames_df = qc_df.sort_values(by="score", ascending=False)
    
    return worst_frames_df

#%%
if __name__ == "__main__":
    # --- Paths ---
    PARENT_VID_DIR = r"C:\Users\cns-th-lab\Tanner_Alex_Vids"
    SLEAP_CENTROID_MODEL = r"C:/Users/cns-th-lab/SLEAP_Projects/models/260410_165103.centroid.n=318"
    SLEAP_CENTERED_MODEL = r"C:/Users/cns-th-lab/SLEAP_Projects/models/260410_182314.centered_instance.n=318"
    PARENT_SLEAP_ANALYSIS_DIR = r"C:\Users\cns-th-lab\Training_Pipeline"
    print(SLEAP_CENTERED_MODEL)
    print(os.path.exists(SLEAP_CENTERED_MODEL))
    print(SLEAP_CENTROID_MODEL)
    print(os.path.exists(SLEAP_CENTROID_MODEL))
    
    
    # NOTE: Ensure this path points to the original project file you used to train the model
    USER_LABELS_PATH = r"C:\Users\cns-th-lab\SLEAP_Projects\labels.v002.slp"
        
    # 1. Select Video and Run Inference
    target_vid = get_random_video(PARENT_VID_DIR)
    print(target_vid)
    quit()
    output_slp = os.path.splitext(target_vid)[0] + ".predictions.slp" #Set output path
    parent_analysis_dir = os.path.join(PARENT_SLEAP_ANALYSIS_DIR, os.path.dirname(os.path.dirname(os.path.relpath(target_vid, start=PARENT_VID_DIR))))
    os.makedirs(parent_analysis_dir, exist_ok=True) #Mirror vid structure
    output_slp = os.path.join(parent_analysis_dir, os.path.basename(output_slp))
    print(output_slp)
    print(os.path.exists(os.path.dirname(output_slp)))
    scores_df, generated_slp_path = run_sleap_inference(target_vid, SLEAP_CENTROID_MODEL, SLEAP_CENTERED_MODEL, output_slp)
    
    # 2. Save Prediction Scores to CSV
    output_scores_csv = os.path.splitext(target_vid)[0] + "_sleap_scores.csv"
    scores_df.to_csv(output_scores_csv, index=True)
    print(f"\nPrediction scores saved to: {output_scores_csv}")
    
    #%% 3. Run Quality Control to flag worst frames
    worst_frames_df = run_quality_control(USER_LABELS_PATH, generated_slp_path)
    
    # 4. Save QC report to CSV
    output_qc_csv = os.path.splitext(generated_slp_path)[0] + "_worst_frames_qc.csv"
    worst_frames_df.to_csv(output_qc_csv, index=False)
    
    print(f"\nQC anomaly report saved to: {output_qc_csv}")
    print("\n=== Top 5 Worst Frames Flagged by QC ===")
    print(worst_frames_df[['frame_idx', 'error_type', 'score']].head(5))
    
    print("\nPipeline pass complete! You can now review the CSVs or open the .slp file in the GUI.")

# uv run C:\Users\cns-th-lab\Github_Repos\selective_wm_analysis\Training_Pipeline\SLEAP_prediction_score_extractor.py