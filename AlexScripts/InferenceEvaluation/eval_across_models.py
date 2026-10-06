#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 3: Inference Across Models (Cross-Model Comparison)
Scans multiple model directories, extracts universal metrics (strictly SLEAP .h5 formats),
and generates full-screen seaborn violins and boxplots across models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from itertools import groupby
import h5py
import json
import os
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

try:
    import sleap_nn
except ImportError:
    print("Warning: sleap_nn not found. CV Ground truth evaluation will not be available.")

# =============================================================================
# 1. HDF5 Loading Logic (Strictly SLEAP)
# =============================================================================
def load_sleap_data(filepath):
    with h5py.File(filepath, "r") as f:
        locations = f["tracks"][:].T
        node_names = [n.decode('utf-8') for n in f["node_names"][:]]
        
    if locations.ndim == 4:
        if locations.shape[-1] == 1:
            locations = np.squeeze(locations, axis=-1)
        else:
            locations = locations[:, :, :, 0] 
            print(f"Warning: Multiple tracks found in {filepath}. Defaulting to Track 0.")
            
    return locations, node_names

# =============================================================================
# 2. Core Extraction Logic
# =============================================================================
def extract_node_metrics(locations):
    num_frames, num_nodes, _ = locations.shape
    node_metrics = {}

    for node_idx in range(num_nodes):
        x_coords = locations[:, node_idx, 0]
        y_coords = locations[:, node_idx, 1]
        is_nan_mask = np.isnan(x_coords)
        
        prop_nans = float(np.sum(is_nan_mask)) / num_frames
        seq_lengths = [sum(1 for _ in group) for key, group in groupby(is_nan_mask) if key]
        num_seqs = len(seq_lengths)
        
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        velocities = np.sqrt(dx**2 + dy**2)
        valid_velocities = velocities[~np.isnan(velocities)]
        
        node_metrics[node_idx] = {
            "prop_nans": prop_nans,            
            "num_seqs": num_seqs,            
            "seq_lengths": seq_lengths,      
            "velocities": valid_velocities   
        }
    return node_metrics

# =============================================================================
# 3. Statistical Calculation & Plotting
# =============================================================================
def calculate_stats(df):
    stats = df.groupby(["Metric", "Node", "Group"])["Value"].agg(
        Average='mean', Median='median', Std='std',
        IQR=lambda x: iqr(x, nan_policy='omit')
    ).reset_index()
    return stats

def calculate_outlier_counts(df):
    outlier_data = []
    for (metric, node, group), group_df in df.groupby(['Metric', 'Node', 'Group']):
        values = group_df['Value'].dropna()
        if len(values) > 1:
            q1, q3 = values.quantile(0.25), values.quantile(0.75)
            iqr_val = q3 - q1
            lower_bound, upper_bound = q1 - 1.5 * iqr_val, q3 + 1.5 * iqr_val
            num_outliers = ((values < lower_bound) | (values > upper_bound)).sum()
        else:
            num_outliers = 0
            
        outlier_data.append({
            'Metric': metric, 'Node': node, 'Group': group, 'Outlier Count': num_outliers
        })
    return pd.DataFrame(outlier_data)

def plot_distributions_all_nodes(df, stats_df):
    metrics = ["Proportion NaNs", "Num NaN Sequences", "NaN Sequence Lengths", "Velocity"]
    groups = df["Group"].unique()
    palette = sns.color_palette("viridis", n_colors=len(groups))
    color_dict = dict(zip(groups, palette))

    fig, axes = plt.subplots(2, 2, figsize=(20, 12)) 
    fig.suptitle('Metrics Distributions Across All Nodes & Models', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_df = df[df["Metric"] == metric]
        if metric_df.empty: continue

        sns.violinplot(
            data=metric_df, x="Node", y="Value", hue="Group", 
            palette=color_dict, inner=None, ax=ax, alpha=0.6, legend=False, density_norm='width' 
        )
        sns.boxplot(
            data=metric_df, x="Node", y="Value", hue="Group", 
            palette=color_dict, width=0.2, boxprops={'zorder': 2}, ax=ax, legend=False, dodge=True 
        )
        
        ax.set_title(metric, fontsize=16)
        ax.set_xlabel("Node", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis='x', rotation=45) 
        if i == 0: ax.legend(title="Model", loc='upper right', fontsize=12, title_fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_outlier_bar_graphs(outlier_df):
    metrics_with_outliers = ["NaN Sequence Lengths", "Velocity"]
    groups = outlier_df["Group"].unique()
    palette = sns.color_palette("viridis", n_colors=len(groups))
    color_dict = dict(zip(groups, palette))

    for metric in metrics_with_outliers:
        metric_outliers = outlier_df[outlier_df["Metric"] == metric]
        if metric_outliers.empty: continue

        fig, ax = plt.subplots(figsize=(16, 6)) 
        fig.suptitle(f'{metric} - Comparative Outliers', fontsize=20, fontweight='bold')
        sns.barplot(data=metric_outliers, x="Node", y="Outlier Count", hue="Group", palette=color_dict, ax=ax)
        
        ax.set_xlabel("Node", fontsize=14)
        ax.set_ylabel("Total Outlier Count", fontsize=14)
        ax.tick_params(axis='x', rotation=45) 
        sns.move_legend(ax, "upper right", title="Model", fontsize=12, title_fontsize=14)
        plt.tight_layout()
        plt.show()

# =============================================================================
# 4. CV Metrics Comparison (from .npz)
# =============================================================================
def plot_cv_metrics_comparison(cv_metrics_dict):
    df = pd.DataFrame(cv_metrics_dict).T.reset_index().rename(columns={'index': 'Model'})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.barplot(data=df, x='Model', y='mAP', ax=axes[0], color='steelblue')
    axes[0].set_title('Mean Average Precision (mAP)')
    axes[0].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x='Model', y='mAR', ax=axes[1], color='mediumseagreen')
    axes[1].set_title('Mean Average Recall (mAR)')
    axes[1].tick_params(axis='x', rotation=45)
    
    sns.barplot(data=df, x='Model', y='p90_error_px', ax=axes[2], color='indianred')
    axes[2].set_title('Localization Error (p90 in pixels)')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Execution Block
# =============================================================================
if __name__ == "__main__":
    PREFS_FILE = "script3_prefs.json"
    prefs = {}
    if os.path.exists(PREFS_FILE):
        with open(PREFS_FILE, "r") as f: prefs = json.load(f)

    root = Tk()
    root.withdraw()
    
    model_dirs = []
    saved_dirs = prefs.get("last_model_dirs", [])
    valid_saved_dirs = [d for d in saved_dirs if os.path.exists(d)]
    
    if valid_saved_dirs:
        dirs_str = "\n".join([os.path.basename(d) for d in valid_saved_dirs])
        if messagebox.askyesno("Use Saved Preference", f"Use previously selected model directories?\n\n{dirs_str}"):
            model_dirs = valid_saved_dirs

    if not model_dirs:
        while True:
            d = filedialog.askdirectory(
                title=f"Select Directory for Model {len(model_dirs)+1} (Cancel to finish)",
                initialdir=prefs.get("last_h5_dir", os.getcwd())
            )
            if not d: break
            model_dirs.append(d)
            prefs["last_h5_dir"] = d
            
        if model_dirs:
            prefs["last_model_dirs"] = model_dirs
            if messagebox.askyesno("Save Preferences", "Save these model directories as default for next time?"):
                with open(PREFS_FILE, "w") as f: json.dump(prefs, f)

    if model_dirs:
        print("Extracting metrics across selected models...")
        all_records = []
        node_names_master = []
        
        for m_dir in model_dirs:
            model_name = os.path.basename(m_dir)
            files = list(Path(m_dir).glob("*.h5"))
            
            for fpath in files:
                locations, node_names = load_sleap_data(fpath)
                if not node_names_master: node_names_master = node_names
                metrics = extract_node_metrics(locations)
                
                for node_idx, node_name in enumerate(node_names):
                    node_data = metrics[node_idx]
                    
                    all_records.append({"Group": model_name, "Node": node_name, "Metric": "Proportion NaNs", "Value": node_data["prop_nans"]})
                    all_records.append({"Group": model_name, "Node": node_name, "Metric": "Num NaN Sequences", "Value": node_data["num_seqs"]})
                    
                    for length in node_data["seq_lengths"]:
                        all_records.append({"Group": model_name, "Node": node_name, "Metric": "NaN Sequence Lengths", "Value": length})
                    for vel in node_data["velocities"]:
                        all_records.append({"Group": model_name, "Node": node_name, "Metric": "Velocity", "Value": vel})

        master_df = pd.DataFrame(all_records)
        stats_df = calculate_stats(master_df)
        outlier_df = calculate_outlier_counts(master_df)
        
        print("Generating Comparative Plots...")
        plot_distributions_all_nodes(master_df, stats_df)
        plot_outlier_bar_graphs(outlier_df)

    npz_paths = []
    saved_npz_paths = prefs.get("last_npz_paths", [])
    valid_saved_npz = [p for p in saved_npz_paths if os.path.exists(p)]
    
    if valid_saved_npz:
        paths_str = "\n".join([os.path.basename(p) for p in valid_saved_npz])
        if messagebox.askyesno("Use Saved Preference", f"Use previously selected CV metric (.npz) files?\n\n{paths_str}"):
            npz_paths = valid_saved_npz
            
    if not npz_paths and messagebox.askyesno("CV Metrics", "Compare SLEAP validation_metrics.npz files?"):
        paths = filedialog.askopenfilenames(
            title="Select validation_metrics.npz for all models",
            initialdir=prefs.get("last_npz_dir", os.getcwd()),
            filetypes=[("NPZ Files", "*.npz")]
        )
        if paths:
            npz_paths = list(paths)
            prefs["last_npz_paths"] = npz_paths
            prefs["last_npz_dir"] = os.path.dirname(npz_paths[0])
            if messagebox.askyesno("Save Preferences", "Save these .npz paths as default for next time?"):
                with open(PREFS_FILE, "w") as f: json.dump(prefs, f)

    if npz_paths:
        print("\nExtracting and Plotting Real CV Metrics...")
        try:
            real_cv_data = {}
            for path in npz_paths:
                model_name = os.path.basename(os.path.dirname(path))
                metrics = sleap_nn.evaluation.load_metrics(path)
                real_cv_data[model_name] = {
                    'mAP': metrics["voc_metrics"]["oks_voc.mAP"],
                    'mAR': metrics["voc_metrics"]["oks_voc.mAR"],
                    'p90_error_px': metrics["distance_metrics"]["p90"]
                }
            plot_cv_metrics_comparison(real_cv_data)
        except NameError:
            print("CV Metrics comparison failed: 'sleap_nn' package could not be imported.")