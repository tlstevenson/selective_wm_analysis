#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 2: Single Model Dataset Aggregation
Aggregates tracking performance across a full dataset for a single model 
(spanning one or multiple directories), evaluates thresholding impacts, 
and compares against ground truth annotations.
"""

import numpy as np
import pandas as pd
import h5py
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from itertools import combinations
from tkinter import Tk, filedialog, messagebox, simpledialog

try:
    import sleap_nn
except ImportError:
    print("Warning: sleap_nn not found. Ground truth evaluation will not be available.")

# =============================================================================
# 1. Model-Level Missing Data & Gap Profiling
# =============================================================================
def aggregate_model_gaps_and_dx(files):
    """Aggregates gaps and dx across a provided list of file paths."""
    node_names = None
    all_gap_lengths = []
    all_gap_dx = []
    total_nans = None
    total_frames = 0
    
    for fpath in files:
        with h5py.File(fpath, "r") as f:
            if not node_names:
                node_names = [n.decode('utf-8') for n in f['node_names'][:]]
                all_gap_lengths = [[] for _ in node_names]
                all_gap_dx = [[] for _ in node_names]
                total_nans = np.zeros(len(node_names))
                
            coords = f['tracks'][:].T
            if coords.ndim == 4: coords = coords[..., 0]
            total_frames += coords.shape[0]
            
            for n_idx in range(len(node_names)):
                node_coords = coords[:, n_idx, :]
                is_nan = np.isnan(node_coords[:, 0])
                total_nans[n_idx] += np.sum(is_nan)
                
                starts = np.where(is_nan & ~np.roll(is_nan, 1))[0]
                stops = np.where(~is_nan & np.roll(is_nan, 1))[0]
                if is_nan[0]: starts = np.insert(starts, 0, 0)
                if is_nan[-1]: stops = np.append(stops, len(is_nan))
                
                for s, e in zip(starts, stops):
                    all_gap_lengths[n_idx].append(e - s)
                    if s > 0 and e < len(node_coords):
                        p0, pf = node_coords[s-1], node_coords[e]
                        if not np.isnan(p0).any() and not np.isnan(pf).any():
                            all_gap_dx[n_idx].append(np.linalg.norm(pf - p0))
                            
    return node_names, total_nans/total_frames, all_gap_lengths, all_gap_dx

def plot_model_gap_violins(all_gap_lengths, all_gap_dx, node_names, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    
    axes[0].violinplot([g if g else [0] for g in all_gap_lengths], showmedians=True)
    axes[0].set_xticks(np.arange(1, len(node_names) + 1))
    axes[0].set_xticklabels(node_names, rotation=45)
    axes[0].set_title(f"Average Gap Length in Frames: {model_name}")
    axes[0].set_ylabel("Frames")
    
    axes[1].violinplot([dx if dx else [0] for dx in all_gap_dx], showmedians=True)
    axes[1].set_xticks(np.arange(1, len(node_names) + 1))
    axes[1].set_xticklabels(node_names, rotation=45)
    axes[1].set_title(f"Average Euclidean Displacement During Gaps: {model_name}")
    axes[1].set_ylabel("Pixels")
    
    plt.tight_layout()
    plt.show()

def plot_aggregate_nans(nan_df, model_name):
    plt.figure(figsize=(12, 5))
    sns.barplot(data=nan_df, x="node", y="prop_nan", color="steelblue")
    plt.title(f"Aggregate NaN Proportion by Node: {model_name}")
    plt.ylabel("Proportion of Missing Frames")
    plt.xlabel("Node")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_aggregate_scores(score_df, model_name):
    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=score_df, x="node", y="score", 
        color="mediumseagreen", inner="quartile", density_norm="width"
    )
    plt.title(f"Prediction Confidence Scores by Node: {model_name}")
    plt.ylabel("Confidence Score")
    plt.xlabel("Node")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# =============================================================================
# 2. Dataset Threshold Combo Evaluations
# =============================================================================
def evaluate_threshold_combinations(coords, scores, score_thresh=0.3, vel_thresh=50):
    n_frames, n_nodes, _ = coords.shape
    masks_dict = {}
    
    masks_dict["Score"] = scores < score_thresh
    dx = np.diff(coords[:, :, 0], axis=0)
    dy = np.diff(coords[:, :, 1], axis=0)
    vel = np.sqrt(dx**2 + dy**2)
    vel_mask = vel > vel_thresh
    masks_dict["Velocity"] = np.pad(vel_mask, ((1, 0), (0, 0)), constant_values=False)
    
    proportions = {"Raw": np.isnan(coords[:, :, 0]).sum(axis=0) / n_frames}
    method_keys = list(masks_dict.keys())
    
    for combo_size in range(1, len(method_keys) + 1):
        for combo in combinations(method_keys, combo_size):
            combo_name = " + ".join(combo)
            combined_mask = masks_dict[combo[0]]
            for method in combo[1:]:
                combined_mask = combined_mask | masks_dict[method]
            proportions[combo_name] = combined_mask.sum(axis=0) / n_frames
            
    return pd.DataFrame(proportions)

def plot_nan_prop_across_thresh(df_proportions):
    df_proportions.plot(kind="bar", figsize=(16, 8), width=0.85)
    plt.title("Proportion of Outliers by Node and Method Combination", fontsize=16)
    plt.ylabel("Proportion of Total Frames")
    plt.xlabel("Nodes")
    plt.legend(title="Methods (Intersection)", bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# =============================================================================
# 3. Model Evaluation vs. Ground Truth
# =============================================================================
def evaluate_ground_truth(gt_slp_path, model_prediction_path):
    import sleap_io as sio
    labels_gt = sio.load_slp(gt_slp_path)
    labels_pr = sleap_nn.predict(gt_slp_path, model_paths=[model_prediction_path])
    evaluator = sleap_nn.evaluation.Evaluator(labels_gt, labels_pr)
    metrics = evaluator.evaluate()
    
    plt.figure(figsize=(6, 3), dpi=150)
    sns.histplot(metrics["voc_metrics"]["oks_voc.match_scores"].flatten(), 
                 binrange=(0, 1), kde=True, stat="probability")
    plt.xlabel("Object Keypoint Similarity")
    plt.title("OKS Match Scores")
    plt.show()
    
    plt.figure(figsize=(4, 4), dpi=150)
    for precision, thresh in zip(metrics["voc_metrics"]['oks_voc.precisions'][::2], 
                                 metrics["voc_metrics"]["oks_voc.match_score_thresholds"][::2]):
        plt.plot(metrics["voc_metrics"]["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.show()
    
    print("mAP:", metrics["voc_metrics"]["oks_voc.mAP"])
    print("mAR:", metrics["voc_metrics"]["oks_voc.mAR"])
    print("Error (p50):", metrics["distance_metrics"]["p50"])
    print("Error (p90):", metrics["distance_metrics"]["p90"])
    return metrics

# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    PREFS_FILE = "script2_prefs.json"
    prefs = {}
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f: prefs = json.load(f)

    root = Tk()
    root.withdraw()
    
    h5_dirs = []
    gt_path, pred_path = None, None
    saved_dirs = prefs.get("last_h5_dirs", [])
    valid_saved_dirs = [d for d in saved_dirs if os.path.exists(d)]
    
    if valid_saved_dirs:
        dirs_str = "\n".join([os.path.basename(d) for d in valid_saved_dirs])
        if messagebox.askyesno("Use Saved Preference", f"Use previously selected model directories?\n\n{dirs_str}"):
            h5_dirs = valid_saved_dirs
            gt_path = prefs.get("last_gt_path")
            pred_path = prefs.get("last_pred_path")
            if gt_path and not os.path.exists(gt_path): gt_path = None
            if pred_path and not os.path.exists(pred_path): pred_path = None
            
    if not h5_dirs:
        while True:
            d = filedialog.askdirectory(
                title=f"Select Directory {len(h5_dirs)+1} of Model .h5 Files (Cancel to finish)",
                initialdir=prefs.get("last_h5_dir", os.getcwd())
            )
            if not d: break
            h5_dirs.append(d)
            prefs["last_h5_dir"] = d
            
        if h5_dirs:
            if messagebox.askyesno("Ground Truth", "Do you want to select a Ground Truth .slp file for CV metrics?"):
                gt_path = filedialog.askopenfilename(title="Select Ground Truth .slp", filetypes=[("SLEAP", "*.slp")])
                pred_path = filedialog.askopenfilename(title="Select Prediction .slp")

            new_prefs = prefs.copy()
            new_prefs["last_h5_dirs"] = h5_dirs
            if gt_path: new_prefs["last_gt_path"] = gt_path
            if pred_path: new_prefs["last_pred_path"] = pred_path
            
            if messagebox.askyesno("Save Preferences", "Save these directories/paths as default for next time?"):
                with open(PREFS_FILE, "w") as f: json.dump(new_prefs, f)

    if h5_dirs:
        print("Executing Aggregation...")
        
        # Ask for a unified model name to group all directories under
        model_name = simpledialog.askstring("Model Name", "Enter the name of this model:", initialvalue=os.path.basename(h5_dirs[0]))
        if not model_name:
            model_name = os.path.basename(h5_dirs[0])
            
        all_nan_records = []
        all_score_records = []
        
        files = []
        for d in h5_dirs:
            files.extend(list(Path(d).glob("*.h5")))
            
        if not files:
            print("No .h5 files found in the selected directories.")
            exit()
            
        for fpath in files:
            with h5py.File(fpath, "r") as f:
                nodes = [n.decode('utf-8') for n in f['node_names'][:]]
                coords = f['tracks'][:].T
                if coords.ndim == 4: coords = coords[..., 0]
                scores = np.transpose(f['point_scores'][:], (2, 1, 0))[..., 0]
                
                for n_idx, node in enumerate(nodes):
                    nan_count = np.sum(np.isnan(coords[:, n_idx, :]).any(axis=1))
                    all_nan_records.append({
                        "model_name": model_name, "node": node, 
                        "nan_count": nan_count, "total_frames": coords.shape[0]
                    })
                    valid_scores = scores[:, n_idx][~np.isnan(scores[:, n_idx])]
                    for s in valid_scores:
                        all_score_records.append({"model_name": model_name, "node": node, "score": s})

        nan_df = pd.DataFrame(all_nan_records)
        agg_nan = nan_df.groupby(["model_name", "node"]).sum().reset_index()
        agg_nan["prop_nan"] = agg_nan["nan_count"] / agg_nan["total_frames"]
        
        # Save combined CSVs into the first selected directory
        save_dir = h5_dirs[0]
        agg_nan.to_csv(os.path.join(save_dir, f"{model_name}_nan_props.csv"), index=False)
        
        score_df = pd.DataFrame(all_score_records)
        score_df.to_csv(os.path.join(save_dir, f"{model_name}_scores.csv"), index=False)
        
        print(f"Saved aggregated metrics to {save_dir}")
        
        plot_aggregate_nans(agg_nan, model_name)
        plot_aggregate_scores(score_df, model_name)
        
        # Dataset Gap & DX violin plots
        node_names, prop_nans, gap_len, gap_dx = aggregate_model_gaps_and_dx(files)
        plot_model_gap_violins(gap_len, gap_dx, node_names, model_name)
        
        # Combo Thresholding on a sample file
        if files:
            with h5py.File(files[0], "r") as f:
                coords = f['tracks'][:].T
                if coords.ndim == 4: coords = coords[..., 0]
                scores = np.transpose(f['point_scores'][:], (2, 1, 0))[..., 0]
                sample_nodes = [n.decode('utf-8') for n in f['node_names'][:]]
            df_combo = evaluate_threshold_combinations(coords, scores)
            df_combo.index = sample_nodes
            plot_nan_prop_across_thresh(df_combo)
        
        if gt_path and pred_path:
            print("\nEvaluating Ground Truth Metrics...")
            try:
                evaluate_ground_truth(gt_path, pred_path)
            except Exception as e:
                print("Could not complete GT evaluation:", e)