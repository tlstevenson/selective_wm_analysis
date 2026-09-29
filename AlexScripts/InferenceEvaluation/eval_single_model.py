import numpy as np
import pandas as pd
import h5py
import json
import os
from pathlib import Path
from tkinter import Tk, filedialog
import sleap_nn
import sleap_io as sio

# --- 1. Missing Data Profiling ---
def aggregate_nan_proportions(h5_directory):
    files = list(Path(h5_directory).glob("*.h5"))
    total_frames = 0
    nan_counts = None
    node_names = []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if not node_names:
                node_names = [n.decode() for n in f['node_names'][:]]
                nan_counts = np.zeros(len(node_names))
            coords = f['tracks'][:].T
            if coords.ndim == 4: coords = coords[..., 0] 
            total_frames += coords.shape[0]
            frame_node_nans = np.isnan(coords).any(axis=2)
            nan_counts += np.sum(frame_node_nans, axis=0)

    return pd.DataFrame({"Node": node_names, "Prop_NaN": nan_counts / total_frames})

# --- 2. Outlier Thresholding Evaluations ---
def evaluate_thresholds(coords, scores, score_thresh=0.3, vel_thresh=50):
    score_mask = scores < score_thresh
    dx = np.diff(coords[:, :, 0], axis=0)
    dy = np.diff(coords[:, :, 1], axis=0)
    vel = np.sqrt(dx**2 + dy**2)
    vel_mask = np.pad((vel > vel_thresh), ((1, 0), (0, 0)), constant_values=False)
    
    return {
        "score_dropped_pct": np.mean(score_mask),
        "velocity_dropped_pct": np.mean(vel_mask),
        "combined_dropped_pct": np.mean(score_mask | vel_mask)
    }

# --- 3. Model Evaluation vs. Ground Truth ---
def evaluate_ground_truth(gt_slp_path, model_prediction_path):
    labels_gt = sio.load_slp(gt_slp_path)
    labels_pr = sleap_nn.predict(gt_slp_path, model_paths=[model_prediction_path])
    evaluator = sleap_nn.evaluation.Evaluator(labels_gt, labels_pr)
    metrics = evaluator.evaluate()
    return {
        "p50_error_px": metrics["distance_metrics"]["p50"],
        "p90_error_px": metrics["distance_metrics"]["p90"],
        "p95_error_px": metrics["distance_metrics"]["p95"],
        "mAP": metrics["voc_metrics"]["oks_voc.mAP"],
        "mAR": metrics["voc_metrics"]["oks_voc.mAR"]
    }

# --- Execution Block ---
if __name__ == "__main__":
    PREFS_FILE = "script2_prefs.json"
    prefs = {}
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f:
            prefs = json.load(f)

    root = Tk()
    root.withdraw()
    
    h5_dir = filedialog.askdirectory(
        title="Select Directory of Model .h5 Inference Files",
        initialdir=prefs.get("last_h5_dir", os.getcwd())
    )

    if h5_dir:
        prefs["last_h5_dir"] = h5_dir
        with open(PREFS_FILE, "w") as f:
            json.dump(prefs, f)
            
        print("Executing Aggregation...")
        
        # Reusable lists for dataframes
        all_nan_records = []
        all_score_records = []
        model_name = os.path.basename(h5_dir)
        
        files = list(Path(h5_dir).glob("*.h5"))
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                nodes = [n.decode('utf-8') for n in f['node_names'][:]]
                coords = f['tracks'][:].T
                if coords.ndim == 4: coords = coords[..., 0]
                scores = np.transpose(f['point_scores'][:], (2, 1, 0))[..., 0]
                
                for n_idx, node in enumerate(nodes):
                    # Compile NaNs
                    nan_count = np.sum(np.isnan(coords[:, n_idx, :]).any(axis=1))
                    all_nan_records.append({
                        "model_name": model_name, "node": node, 
                        "nan_count": nan_count, "total_frames": coords.shape[0]
                    })
                    
                    # Compile valid scores
                    valid_scores = scores[:, n_idx][~np.isnan(scores[:, n_idx])]
                    for s in valid_scores:
                        all_score_records.append({"model_name": model_name, "node": node, "score": s})

        # Save NaNs to CSV
        nan_df = pd.DataFrame(all_nan_records)
        agg_nan = nan_df.groupby(["model_name", "node"]).sum().reset_index()
        agg_nan["prop_nan"] = agg_nan["nan_count"] / agg_nan["total_frames"]
        nan_out_path = os.path.join(h5_dir, f"{model_name}_nan_props.csv")
        agg_nan.to_csv(nan_out_path, index=False)
        
        # Save Scores to CSV
        score_df = pd.DataFrame(all_score_records)
        score_out_path = os.path.join(h5_dir, f"{model_name}_scores.csv")
        score_df.to_csv(score_out_path, index=False)
        
        print(f"Saved aggregated metrics to {h5_dir}")