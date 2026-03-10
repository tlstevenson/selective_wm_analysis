import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from itertools import groupby
import h5py
from pathlib import Path

# ---------------------------------------------------------
# 1. Configuration & Grouping Setup
# ---------------------------------------------------------
def get_file_paths(directory_path):
    """Returns a list of strings containing the paths of all .h5 files in a directory."""
    path_obj = Path(directory_path)
    # .is_file() ensures we don't include subdirectories
    # .suffix == '.h5' ensures we don't accidentally try to load hidden files like .DS_Store
    return [str(file) for file in path_obj.iterdir() if file.is_file() and file.suffix == '.h5']

# Define your groups by folder
h5_folder_paths = [
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/199/h5", "199"), 
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/274/h5", "274"), 
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/400/h5", "400"),
    ("/Users/alex/Documents/HanksLab/HanksLabVideos/Analysis_3_10_2026/Analysis/402/h5", "402")
]

# Dynamically build the configuration list
VIDEO_CONFIG = []
for folder in h5_folder_paths:
    for file in get_file_paths(folder[0]):
        # Note: mapped to "filepath" to match downstream variables
        VIDEO_CONFIG.append({"filepath": file, "group": folder[1]})

# ---------------------------------------------------------
# 2. HDF5 Loading Logic
# ---------------------------------------------------------
def load_sleap_data(filepath):
    """
    Loads SLEAP h5 data and formats the locations array.
    Returns: locations (frames, nodes, 2) and a list of node_names.
    """
    with h5py.File(filepath, "r") as f:
        # Transpose to get shape: (frames, nodes, 2, tracks)
        locations = f["tracks"][:].T
        node_names = [n.decode() for n in f["node_names"][:]]
        
    # Handle the 'tracks' (animals) dimension
    if locations.ndim == 4:
        if locations.shape[-1] == 1:
            # Squeeze out the track dimension if there's only 1 animal
            locations = np.squeeze(locations, axis=-1)
        else:
            # If there are multiple tracks, default to the first animal (Track 0)
            locations = locations[:, :, :, 0] 
            print(f"Warning: Multiple tracks found in {filepath}. Defaulting to Track 0.")
            
    return locations, node_names

# ---------------------------------------------------------
# 3. Core Extraction Logic
# ---------------------------------------------------------
def extract_node_metrics(locations):
    """
    Extracts metrics for a single video's locations array.
    Expects locations shape: (frames, nodes, 2)
    Returns a dictionary mapping node_idx to its metrics.
    """
    num_frames, num_nodes, _ = locations.shape
    node_metrics = {}

    for node_idx in range(num_nodes):
        x_coords = locations[:, node_idx, 0]
        y_coords = locations[:, node_idx, 1]
        is_nan_mask = np.isnan(x_coords)
        
        # 1. Number of missing frames
        num_nans = int(np.sum(is_nan_mask))
        
        # 2 & 3. Number and lengths of NaN sequences
        seq_lengths = [
            sum(1 for _ in group) 
            for key, group in groupby(is_nan_mask) if key
        ]
        num_seqs = len(seq_lengths)
        
        # 4. Velocity
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        velocities = np.sqrt(dx**2 + dy**2)
        valid_velocities = velocities[~np.isnan(velocities)]
        
        node_metrics[node_idx] = {
            "num_nans": num_nans,            
            "num_seqs": num_seqs,            
            "seq_lengths": seq_lengths,      
            "velocities": valid_velocities   
        }
        
    return node_metrics

# ---------------------------------------------------------
# 4. Data Processing & Aggregation
# ---------------------------------------------------------
def process_all_videos(config):
    """
    Processes all videos and aggregates them into a Pandas DataFrame.
    Automatically extracts node names from the file.
    """
    all_records = []
    
    for video in config:
        print(f"Processing: {video['filepath']}...")
        locations, node_names = load_sleap_data(video["filepath"])
        metrics = extract_node_metrics(locations)
        
        for node_idx, node_name in enumerate(node_names):
            node_data = metrics[node_idx]
            
            # Scalars (One value per video)
            all_records.append({"Group": video["group"], "Node": node_name, "Metric": "Total NaNs", "Value": node_data["num_nans"]})
            all_records.append({"Group": video["group"], "Node": node_name, "Metric": "Num NaN Sequences", "Value": node_data["num_seqs"]})
            
            # Arrays (Explode them so each value is a row for seaborn distributions)
            for length in node_data["seq_lengths"]:
                all_records.append({"Group": video["group"], "Node": node_name, "Metric": "NaN Sequence Lengths", "Value": length})
            for vel in node_data["velocities"]:
                all_records.append({"Group": video["group"], "Node": node_name, "Metric": "Velocity", "Value": vel})

    return pd.DataFrame(all_records), node_names

# ---------------------------------------------------------
# 5. Statistical Calculation & Plotting
# ---------------------------------------------------------
def calculate_stats(df):
    """Calculates Average, Median, Std, and IQR."""
    stats = df.groupby(["Metric", "Node", "Group"])["Value"].agg(
        Average='mean',
        Median='median',
        Std='std',
        IQR=lambda x: iqr(x, nan_policy='omit')
    ).reset_index()
    return stats

def plot_distributions_all_nodes(df, stats_df):
    """
    Generates a single figure with 4 subplots (one for each metric).
    All nodes are plotted on the x-axis, grouped by experimental condition.
    """
    metrics = ["Total NaNs", "Num NaN Sequences", "NaN Sequence Lengths", "Velocity"]
    
    groups = df["Group"].unique()
    palette = sns.color_palette("viridis", n_colors=len(groups))
    color_dict = dict(zip(groups, palette))

    fig, axes = plt.subplots(2, 2, figsize=(20, 12)) 
    fig.suptitle('Metrics Distributions Across All Nodes', fontsize=20, fontweight='bold')
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        metric_df = df[df["Metric"] == metric]
        
        if metric_df.empty:
            continue

        sns.violinplot(
            data=metric_df, x="Node", y="Value", hue="Group", 
            palette=color_dict, inner=None, ax=ax, alpha=0.6, legend=False,
            density_norm='width' 
        )
        sns.boxplot(
            data=metric_df, x="Node", y="Value", hue="Group", 
            palette=color_dict, width=0.2, boxprops={'zorder': 2}, ax=ax, legend=False,
            dodge=True 
        )
        
        ax.set_title(metric, fontsize=16)
        ax.set_xlabel("Node", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis='x', rotation=45) 
        
        if i == 0:
            ax.legend(title="Group", loc='upper right', fontsize=12, title_fontsize=14)

    plt.tight_layout()
    plt.show()
    
    csv_filename = "node_statistics_summary.csv"
    stats_df.round(2).to_csv(csv_filename, index=False)
    print(f"\nSaved detailed statistical table to: {csv_filename}")

def plot_boxplots_separate_figures(df):
    """
    Generates a separate, full-sized Box Plot figure for each metric.
    All nodes are plotted on the x-axis, grouped by experimental condition.
    """
    metrics = ["Total NaNs", "Num NaN Sequences", "NaN Sequence Lengths", "Velocity"]
    
    groups = df["Group"].unique()
    palette = sns.color_palette("viridis", n_colors=len(groups))
    color_dict = dict(zip(groups, palette))

    for metric in metrics:
        print(metric)
        metric_df = df[df["Metric"] == metric]
        
        if metric_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 8)) 
        fig.suptitle(f'{metric} Box Plot Across All Nodes', fontsize=20, fontweight='bold')

        # Draw only the box plot
        sns.boxplot(
            data=metric_df, x="Node", y="Value", hue="Group", 
            palette=color_dict, ax=ax, legend=True
        )
        
        ax.set_xlabel("Node", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis='x', rotation=45) 
        
        sns.move_legend(ax, "upper right", title="Group", fontsize=12, title_fontsize=14)

        plt.tight_layout()
        plt.show()


def plot_violinplots_separate_figures(df):
    """
    Generates a separate, full-sized Violin Plot figure for each metric.
    All nodes are plotted on the x-axis, grouped by experimental condition.
    """
    metrics = ["Total NaNs", "Num NaN Sequences", "NaN Sequence Lengths", "Velocity"]
    
    groups = df["Group"].unique()
    palette = sns.color_palette("viridis", n_colors=len(groups))
    color_dict = dict(zip(groups, palette))

    for metric in metrics:
        print(metric)
        metric_df = df[df["Metric"] == metric]
        
        if metric_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(16, 8)) 
        fig.suptitle(f'{metric} Violin Plot Across All Nodes', fontsize=20, fontweight='bold')

        # Draw only the violin plot, but add inner='box' to show quartiles
        sns.violinplot(
            data=metric_df, x="Node", y="Value", hue="Group", 
            palette=color_dict, inner='box', ax=ax, alpha=0.8, legend=True,
            density_norm='width' 
        )
        
        ax.set_xlabel("Node", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis='x', rotation=45) 
        
        sns.move_legend(ax, "upper right", title="Group", fontsize=12, title_fontsize=14)

        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------
# 6. Execution
# ---------------------------------------------------------
#%%Processing steps
if __name__ == "__main__":
    
    print("Extracting data and building DataFrame...")
    df, extracted_node_names = process_all_videos(VIDEO_CONFIG)
    
    print("Calculating statistics...")
    stats_df = calculate_stats(df)
    
    # Save the stats table
    csv_filename = "node_statistics_summary.csv"
    stats_df.round(2).to_csv(csv_filename, index=False)
    print(f"\nSaved detailed statistical table to: {csv_filename}")
#%%Plotting steps
if __name__ == "__main__":

    print("Generating separate plots for each metric...")
    plot_boxplots_separate_figures(df)