# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:25:58 2026

@author: cns-th-lab

env: neuropy
"""

#%% I begin inference evaluation by overlaying poses with video.
# This gives a subjective understanding of the data
import pandas as pd
import numpy as np
import h5py
import init
import file_select_ui as fsui
import os
import PredictionViewer as pv
import subprocess
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math

# --- Configuration ---
VIDEO_PATH = ''#fsui.GetFile("Please select a video file")
INFERENCE_PATH = fsui.GetFile("Please select the corresponding .h5 path")  # Or .csv
OUTPUT_WINDOW = "Keypoint Inspector"
FPS = 30 # Defined conversion rate

if os.path.splitext(os.path.basename(VIDEO_PATH))[0] != os.path.splitext(os.path.basename(INFERENCE_PATH))[0]:
    print("Session id of video and data do not match by current naming conventions.")
#%% Run App
pv.RunApp(VIDEO_PATH, INFERENCE_PATH, OUTPUT_WINDOW, FPS)

#%% Currently homeless functions for slp -> h5 and h5 -> coords, names, scores
def slp_to_analysis_h5(slp_path, h5_path):
    """
    Converts a SLEAP .slp file to a standard analysis .h5 file using sleap-io.
    """
    print(f"  -> Exporting to {h5_path} via CLI...")
    command = ["uv", "run", "sleap", "export", str(slp_path), "-o", str(h5_path)]
    
    try:
        if not os.path.exists(h5_path):
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e)
    return h5_path

def ExtractH5RawData(inference_path):
    with h5py.File(inference_path, "r") as f:
        # Decode node names
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        
        # 1. Get Prediction Scores 
        # Raw shape: (tracks, nodes, frames) -> Transposed: (frames, nodes, tracks)
        scores = np.transpose(f['point_scores'][:], (2, 1, 0)) 
        
        # 2. Get Coordinates
        # Raw shape: (tracks, nodes, 2, frames) -> Transposed: (frames, nodes, 2, tracks)
        tracks_coords = np.transpose(f['tracks'][:])
        
        return tracks_coords, node_names, scores

#%% Visualize where model is missing
#%%% Get raw data
coords, names, scores = ExtractH5RawData(INFERENCE_PATH)

#%%% Get thresholded data
def ThresholdedPositions(positions, scores, threshold):
    positions_copy = np.copy(positions)
    mask = scores < threshold
    positions_copy[:, :, 0,:][mask] = np.nan
    positions_copy[:, :, 1,:][mask] = np.nan
    return positions_copy

thresh_coords = ThresholdedPositions(coords, scores, 0.3)

#%%% Get interpolated data
def InterpolateCoordsCubic(coords, limit_arg):
    deconstr_dict = {}
    coords_copy = np.copy(coords)
    
    # Turn each x and y series into a column
    for i in range(np.shape(coords)[1]):
        deconstr_dict[f"{i}_x"] = coords_copy[:, i, 0, 0] 
        deconstr_dict[f"{i}_y"] = coords_copy[:, i, 1, 0]
        
    my_df = pd.DataFrame(deconstr_dict)
    
    my_df = my_df.interpolate(method="polynomial", order=3, limit=limit_arg, limit_area='inside', axis=0)
    
    for i in range(np.shape(coords)[1]):
        coords_copy[:,i,0,0] = my_df[f"{i}_x"]
        coords_copy[:,i,1,0] = my_df[f"{i}_y"]
        
    return coords_copy

cube_coords = InterpolateCoordsCubic(thresh_coords, 30)
#%%% Function definitions
def measure_nan_gaps(s: pd.Series) -> pd.Series:
    """
    Takes a pandas Series and returns a Series of the same length where the 
    start of each NaN gap contains the length of that gap, and all other 
    values are 0.
    """
    # Create an output series initialized with zeros
    out = pd.Series(0, index=s.index, dtype=int)
    
    is_nan = s.isna()
    
    # 1. Identify the starting index of each NaN gap
    starts = is_nan & ~is_nan.shift(1, fill_value=False)
    
    # 2. Group the data into blocks (increments every time a non-NaN is seen)
    blocks = (~is_nan).cumsum()
    
    # 3. Count the number of NaNs in each block
    gap_sizes = is_nan.groupby(blocks).sum().astype(int)
    
    # 4. Filter for only the blocks that actually contain NaNs
    gap_sizes = gap_sizes[gap_sizes > 0]
    
    # 5. Assign the computed lengths to the starting positions
    if len(out.loc[starts]) != len(gap_sizes.values):
        raise IndexError("The indexes of the gap starts and calculated gap lengths do not match in length")
    else:
        out.loc[starts] = gap_sizes.values
    return out, starts, gap_sizes

#%%% Visualize starts of long NaN gaps by node
#%%% Individual Plotting Functions
def nan_gap_spike_graph(my_coords, node_names=None, columns=4):
    """A function to plot a spike of height gap_length at each position where a NaN gap begins
    
    Args:
        my_coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        columns (int): The number of columns in the graph grid.
    """
    if node_names == None:
        node_names = np.arange(0,np.shape(my_coords)[1])
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns)
    for i in range(len(names)):
        nan_out, nan_starts, _ = measure_nan_gaps(pd.Series(my_coords[:,i, 0, 0]))
        print(node_names[i])
        ax[i//columns, i%columns].bar(np.arange(0, np.shape(my_coords)[0])[nan_starts],nan_out[nan_starts], label=node_names[i])
        ax[i//columns, i%columns].legend()
        ax[i//columns, i%columns].set_yscale('log')
        ax[i//columns, i%columns].set_xlim(0,len(my_coords))
    plt.show()
    
def nan_heatplot(my_coords, node_names=None, columns=2):
    """A function to plot heatmap of NaN locations by node
    
    Args:
        my_coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        columns (int): The number of columns in the graph grid.
    """
    if node_names == None:
        node_names = np.arange(0,np.shape(my_coords)[1])
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns)
    for i in range(len(node_names)):
        print(node_names[i])
        is_nan = np.isnan(my_coords[:,i, 0, 0])
        nan_x_points = np.arange(0, np.shape(my_coords)[0])[is_nan]
        ax[i//columns, i%columns].bar(nan_x_points,np.ones(len(nan_x_points)))
        ax[i//columns, i%columns].set_title(node_names[i])
        ax[i//columns, i%columns].set_xlim(0,len(my_coords))
    plt.show()
    
def nan_prop(my_coords, node_names=None):
    """A function to plot proportion of NaNs by node
    
    Args:
        my_coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
    """
    if node_names == None:
        node_names = np.arange(0,np.shape(my_coords)[1])
    prop_nan = []
    for i in range(len(node_names)):
        print(node_names[i])
        nan_out, nan_starts, _ = measure_nan_gaps(pd.Series(my_coords[:,i, 0, 0]))
        prop_nan_i = np.sum(nan_out) / len(nan_out)
        prop_nan.append(prop_nan_i)
    plt.bar(names, prop_nan)
    plt.show()
    
def plot_distr_nan_gaps(my_coords, node_names=None, columns=2):
    """A function to plot distribution of gap lengths by node
    
    Args:
        my_coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
    """
    if node_names == None:
        node_names = np.arange(0,np.shape(my_coords)[1])
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns)
    for i in range(len(node_names)):
        print(node_names[i])
        nan_out, nan_starts, nan_gap_sizes = measure_nan_gaps(pd.Series(my_coords[:,i, 0, 0]))
        ax[i//columns, i%columns].hist(nan_gap_sizes,label=node_names[i])
        ax[i//columns, i%columns].legend()
        ax[i//columns, i%columns].set_xlim(left=0, right=None)
    plt.show()
#%%% Combined Plots
# Group your arrays into a dictionary
dataset_dict = {
    "Raw": coords, 
    "Thresholded": thresh_coords, 
    "Interpolated": cube_coords
}

def nan_gap_spike_graph_d(coords_dict, node_names=None, columns=4):
    """A function to plot a spike of height gap_length at each position where a NaN gap begins
    
    Args:
        coords_dict (dict{string:float[,,,]}): dictionary with several SLEAP arrays of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        columns (int): The number of columns in the graph grid.
    """
    first_coords = list(coords_dict.values())[0]
    if node_names is None:
        node_names = np.arange(0, np.shape(first_coords)[1])
        
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns, figsize=(15, 8))
    for i in range(len(node_names)):
        row, col = i//columns, i%columns
        for label, my_coords in coords_dict.items():
            nan_out, nan_starts, _ = measure_nan_gaps(pd.Series(my_coords[:,i, 0, 0]))
            # alpha=0.5 makes the overlapping bars transparent so you can see all three
            ax[row, col].bar(np.arange(0, np.shape(my_coords)[0])[nan_starts], nan_out[nan_starts], label=label, alpha=0.5)
        
        ax[row, col].set_title(node_names[i])
        ax[row, col].set_yscale('log')
        ax[row, col].set_xlim(0, len(first_coords))
        ax[row, col].legend()
    fig.suptitle("NaN Gap Starts and Lengths")
    plt.tight_layout()
    plt.show()

def nan_heatplot_d(coords_dict, node_names=None, columns=4):
    """A function to plot heatmap of NaN locations by node
    
    Args:
        coords_dict (dict{string:float[,,,]}): dictionary with several SLEAP arrays of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        columns (int): The number of columns in the graph grid.
    """
    first_coords = list(coords_dict.values())[0]
    if node_names is None:
        node_names = np.arange(0, np.shape(first_coords)[1])
        
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns, figsize=(15, 8))
    for i in range(len(node_names)):
        row, col = i//columns, i%columns
        y_offset = 1 # We will stack the heatmaps on the Y axis
        
        for label, my_coords in coords_dict.items():
            is_nan = np.isnan(my_coords[:,i, 0, 0])
            nan_x_points = np.arange(0, np.shape(my_coords)[0])[is_nan]
            # Use scatter with vertical lines so they stack cleanly
            ax[row, col].scatter(nan_x_points, np.ones(len(nan_x_points)) * y_offset, label=label, marker='|')
            y_offset += 1
            
        ax[row, col].set_title(node_names[i])
        ax[row, col].set_xlim(0, len(first_coords))
        ax[row, col].set_yticks([1, 2, 3])
        ax[row, col].set_yticklabels(list(coords_dict.keys()))
    fig.suptitle("NaNs across video")
    plt.tight_layout()
    plt.show()

def nan_prop_d(coords_dict, node_names=None):
    """A function to plot proportion of NaNs by node
    
    Args:
        coords_dict (dict{string:float[,,,]}): dictionary with several SLEAP arrays of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
    """
    first_coords = list(coords_dict.values())[0]
    if node_names is None:
        node_names = np.arange(0, np.shape(first_coords)[1])
        
    df_data = {}
    for label, my_coords in coords_dict.items():
        prop_nan = []
        for i in range(len(node_names)):
            nan_out, _, _ = measure_nan_gaps(pd.Series(my_coords[:,i, 0, 0]))
            prop_nan.append(np.sum(nan_out) / len(nan_out))
        df_data[label] = prop_nan
        
    # Pandas handles side-by-side grouped bar charts automatically
    df = pd.DataFrame(df_data, index=node_names)
    df.plot(kind="bar", figsize=(12, 5))
    plt.ylabel("NaN Fraction")
    plt.show()

def plot_distr_nan_gaps_d(coords_dict, node_names=None, columns=4):
    """A function to plot distribution of gap lengths by node
    
    Args:
        coords_dict (dict{string:float[,,,]}): dictionary with several SLEAP arrays of size frames x nodes x 2 x tracks
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        columns (int): The number of columns in the graph grid.
    """
    first_coords = list(coords_dict.values())[0]
    if node_names is None:
        node_names = np.arange(0, np.shape(first_coords)[1])
        
    fig, ax = plt.subplots(math.ceil(len(node_names)/columns), columns, figsize=(15, 8))
    for i in range(len(node_names)):
        row, col = i//columns, i%columns
        hist_data = []
        labels = []
        
        for label, my_coords in coords_dict.items():
            _, _, nan_gap_sizes = measure_nan_gaps(pd.Series(my_coords[:,i, 0, 0]))
            hist_data.append(nan_gap_sizes)
            labels.append(label)
            
        # Passing a list of arrays to ax.hist automatically plots them side-by-side
        ax[row, col].hist(hist_data, label=labels)
        ax[row, col].set_title(node_names[i])
        ax[row, col].legend()
    plt.tight_layout()
    plt.show()

#%%% Generate Combined Plots
nan_gap_spike_graph_d(dataset_dict, node_names=names)
nan_heatplot_d(dataset_dict, node_names=names)
#%%
nan_prop_d(dataset_dict, node_names=names)
#%%
plot_distr_nan_gaps_d(dataset_dict, node_names=names)

#%%% Plotting Spike Graph
nan_gap_spike_graph(coords, node_names=names)
nan_gap_spike_graph(thresh_coords, node_names=names)
nan_gap_spike_graph(cube_coords, node_names=names)

#%%% Plotting Heatmap by node
nan_heatplot(coords, node_names=names, columns=4)
nan_heatplot(thresh_coords, node_names=names, columns=4)
nan_heatplot(cube_coords, node_names=names, columns=4)

#%%% Plotting Proportion of NaNs by Node
nan_prop(coords, node_names=names)
nan_prop(thresh_coords, node_names=names)
nan_prop(cube_coords, node_names=names)

#%%% Plotting Distribution of Gap Lengths by Node
plot_distr_nan_gaps(coords, node_names=names, columns=4)
plot_distr_nan_gaps(thresh_coords, node_names=names, columns=4)
plot_distr_nan_gaps(cube_coords, node_names=names, columns=4)

#%%% Diagnose NaN areas by frequency
num_nan_per_node = np.sum(np.isnan(coords[:,:,0,0]), axis = 0) #(14,)
tot_prop_nan = num_nan_per_node/np.shape(coords)[0]
bin_length_frames = 30
print(tot_prop_nan)
#%%

#TODO: Vectorize logic
prop_bins = []
lower_bound = 0
while lower_bound < np.shape(coords)[0]:
    bin_coords = coords[lower_bound:min(lower_bound + bin_length_frames, np.shape(coords)[0]), :, 0, 0] #All nodes c1 coords for c2 frames
    bin_prop_nan = np.sum(np.isnan(bin_coords[:,:]), axis=0) / bin_length_frames
    prop_bins.append(bin_prop_nan)
    lower_bound = lower_bound + bin_length_frames

prop_bins = np.array(prop_bins) #(bins, nodes)
print(prop_bins[0,:])

#%%
std_prop_nan = np.std(prop_bins, axis=0) #()
print(std_prop_nan)

#%%
bin_zs = (prop_bins - tot_prop_nan)/std_prop_nan
print(bin_zs[0,:])

#%%
print(np.shape(bin_zs))
plt.imshow(bin_zs, cmap="cividis")
plt.gca().set_aspect(1/(np.shape(bin_zs)[0] / np.shape(bin_zs)[1]))
plt.colorbar()
plt.show()
#%% Visualize Velocity Performance
def CalcVelocity(coords):
    print(np.shape(coords))
    dx = np.diff(coords[:,:,0,:], axis = 0)
    dy = np.diff(coords[:,:,1,:], axis = 0)
    velocities = np.sqrt(dx**2 + dy**2)
    print(np.shape(np.squeeze(velocities)))
    return np.squeeze(velocities)
all_node_velocity = np.squeeze(CalcVelocity(coords))

#%%% Visualize velocity time trace
fig4, ax4 = plt.subplots(len(names))
for i in range(len(names)):
    print(names[i])
    ax4[i].plot(np.arange(np.shape(all_node_velocity)[0]), all_node_velocity[:,i])
    ax4[i].set_title(names[i])
plt.show()
#%%% Visualize boxplot of valid velocities
cleaned_all_node_velocity = [col[~np.isnan(col)] for col in np.transpose(all_node_velocity)] #Cleans by node (needed for list of nodes)
x = np.arange(np.shape(all_node_velocity)[1])
for i in range(len(x)):
    plt.boxplot(cleaned_all_node_velocity[i][:], positions=[x[i]], tick_labels=[names[i]])
plt.show()

std_vel = np.nanstd(all_node_velocity, axis=0)
mean_vel = np.nanmean(all_node_velocity, axis=0)
median_vel = np.nanmedian(all_node_velocity, axis=0)
print("Velocity distributions") 
for i in range(len(names)):
    print(f"{names[i]}: Mean({mean_vel[i]}) Median({median_vel[i]}) STD({std_vel[i]})")


#%%% Observe what various thresholds would do to data

#%%%% Boxplot with threshold
vel_threshold = 30
x = np.arange(np.shape(all_node_velocity)[1])
y = np.ones(len(x)) * vel_threshold
plt.plot(x,y, color = "red")
for i in range(len(x)):
    plt.boxplot(cleaned_all_node_velocity[i][:], positions=[x[i]], tick_labels=[names[i]])
plt.show()
#%%%% Proportion of valid velocities removed
prop_valid_vel = []
for i in range(len(x)):
    num_above_thresh = len(cleaned_all_node_velocity[i][:][cleaned_all_node_velocity[i][:] > vel_threshold])
    prop_valid_vel_i = num_above_thresh / len(cleaned_all_node_velocity[i][:])
    prop_valid_vel.append(prop_valid_vel_i)
plt.bar(names, prop_valid_vel)
plt.show()

#%%%% Diagnose velocity outlier areas by frequency

#!!!Get the prop_valid_vel on average from previous cell
bin_length_frames = 30

#TODO: Vectorize logic
prop_bins_vel = []
lower_bound = 0
while lower_bound < np.shape(all_node_velocity)[0]:
    bin_vels = all_node_velocity[lower_bound:min(lower_bound + bin_length_frames, np.shape(all_node_velocity)[0])] #All nodes c1 coords for c2 frames
    bin_prop_vel = []
    #Safeguard if sum is 0
    if np.sum(~np.isnan(bin_vels)) != 0:
        bin_prop_vel = np.sum(~np.isnan(bin_vels) & (bin_vels > vel_threshold), axis=0) / np.sum(~np.isnan(bin_vels))
    else:
        bin_prop_vel = 0
    if lower_bound == 0:
        print(np.shape(bin_vels))
        print(np.shape(bin_prop_vel))
        print(bin_prop_vel)
    prop_bins_vel.append(bin_prop_vel)
    lower_bound = lower_bound + bin_length_frames

prop_bins_vel = np.array(prop_bins_vel) #(bins, nodes)
print(np.shape(prop_bins_vel))

std_prop_vel = np.std(prop_bins_vel, axis=0) #()
print(std_prop_vel)

bin_vel_zs = (prop_bins_vel -np.array(prop_valid_vel))/std_prop_vel
print(f"Prop valid velocities removed: {prop_valid_vel}")
print(np.shape(bin_vel_zs))

plt.imshow(bin_vel_zs, cmap="cividis")
plt.gca().set_aspect(1/(np.shape(bin_vel_zs)[0] / np.shape(bin_vel_zs)[1]))
plt.colorbar()
plt.show()

#%%Z scores and velocity functions (CHECK)
def analyze_nan_binned_zscores(coords, bin_length_frames=30):
    """Calculates z scores of (default) 30 frame bins across the video by node.
    Z scores are relative to the total proportion of NaNs.
    Values are plotted as a heatmap of z scores
    
    Args:
        coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        bin_length_frames (int): The number of frames in one z score bin
        
    Returns:
        The bin z score values as a 2d array of size nodes x num_bins (or transpose CHECK)
    """
    num_nan_per_node = np.sum(np.isnan(coords[:, :, 0, 0]), axis=0)
    tot_prop_nan = num_nan_per_node / np.shape(coords)[0]
    
    prop_bins = []
    lower_bound = 0
    while lower_bound < np.shape(coords)[0]:
        bin_coords = coords[lower_bound:min(lower_bound + bin_length_frames, np.shape(coords)[0]), :, 0, 0]
        bin_prop_nan = np.sum(np.isnan(bin_coords), axis=0) / bin_length_frames
        prop_bins.append(bin_prop_nan)
        lower_bound += bin_length_frames
        
    prop_bins = np.array(prop_bins)
    std_prop_nan = np.std(prop_bins, axis=0)
    std_prop_nan = np.where(std_prop_nan == 0, 1e-8, std_prop_nan)  # Prevent division by zero
    
    bin_zs = (prop_bins - tot_prop_nan) / std_prop_nan
    
    plt.figure(figsize=(8, 6))
    plt.imshow(bin_zs, cmap="cividis", aspect="auto")
    plt.gca().set_aspect(1 / (np.shape(bin_zs)[0] / np.shape(bin_zs)[1]))
    plt.colorbar(label="Z-score")
    plt.title("NaN Binned Z-Scores")
    plt.show()
    return bin_zs

def calc_velocity(coords):
    """Calculates Euclidean velocity from x and y coordinates.
    
    Args:
        coords (float array[,,,]): SLEAP array of size frames x nodes x 2 x tracks
        
    Returns:
        An array of velocities of size 3d frames x nodes x 1
    """
    dx = np.diff(coords[:, :, 0, :], axis=0)
    dy = np.diff(coords[:, :, 1, :], axis=0)
    velocities = np.sqrt(dx**2 + dy**2)
    return np.squeeze(velocities) #Removes the tracks dimension

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

def clean_velocity(all_node_velocity):
    """Removes NaNs from velocity for visualization.
    
    Args:
        all_node_velocity (float[,,]): Raw (including NaNs) velocity array [often output of calc_velocity()]
    
    Returns:
        A list of lists for each node with all non-NaN velocities (non-homogenous)
    """
    return [col[~np.isnan(col)] for col in np.transpose(all_node_velocity)]

def get_prop_valid_vel(cleaned_velocities, velocity_threshold):
    """Calculate the proportion of valid velocities still above a threshold.
    
    
    Args:
        cleaned_velocities (float[][]): A list of lists for each node with all non-NaN velocities (non-homogenous)
        velocity_threshold (float): The upper bound for valid velocities
        
    Returns:
        Proportion of valid velocities exceeding threshold by node
    """
    
    prop_valid_vel = []
    
    for col in cleaned_velocities:
        if len(col) == 0:
            prop_valid_vel.append(0.0)
        else:
            num_above = np.sum(col > velocity_threshold)
            prop_valid_vel.append(num_above / len(col))
    return prop_valid_vel
    

def plot_velocity_threshold_proportions(all_node_velocity, node_names=None, vel_threshold=30):
    """Pplots the proportion of valid velocities exceeding a threshold.
    
    Args:
        all_node_velocity (float[,,]): Raw (including NaNs) velocity array [often output of calc_velocity()]
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        vel_threshold (float): An optional visualization for potential velocity outlier cutoff
    """
    if node_names is None:
        node_names = np.arange(np.shape(all_node_velocity)[1])
        
    cleaned_velocities = clean_velocity(all_node_velocity)
    prop_valid_vel = get_prop_valid_vel(cleaned_velocities, vel_threshold)
            
    plt.figure(figsize=(10, 4))
    plt.bar(node_names, prop_valid_vel)
    plt.ylabel("Proportion Above Threshold")
    plt.title(f"Velocity Outliers Removed (Threshold > {vel_threshold})")
    plt.xticks(rotation=45)
    plt.show()
    

def analyze_velocity_outlier_zscores(all_node_velocity, vel_threshold=30, bin_length_frames=30):
    """Analyzes and plots binned velocity outlier frequency Z-scores.
    
    Args:
        all_node_velocity (float[,,]): Raw (including NaNs) velocity array [often output of calc_velocity()]
        
        node_names (string[]): An optional list of node names. Alternatively uses 1-n.
        vel_threshold (float): An optional visualization for potential velocity outlier cutoff"""
    
    cleaned_velocities = clean_velocity(all_node_velocity)
    prop_valid_vel = get_prop_valid_vel(cleaned_velocities,vel_threshold)
    
    prop_bins_vel = []
    lower_bound = 0
    
    while lower_bound < np.shape(all_node_velocity)[0]:
        bin_vels = all_node_velocity[lower_bound:min(lower_bound + bin_length_frames, np.shape(all_node_velocity)[0])]
        valid_mask = ~np.isnan(bin_vels)
        denom = np.sum(valid_mask, axis=0)
        
        bin_prop_vel = np.where(
            denom > 0,
            np.sum(valid_mask & (bin_vels > vel_threshold), axis=0) / denom,
            0.0
        )
        prop_bins_vel.append(bin_prop_vel)
        lower_bound += bin_length_frames
        
    prop_bins_vel = np.array(prop_bins_vel)
    std_prop_vel = np.std(prop_bins_vel, axis=0)
    std_prop_vel = np.where(std_prop_vel == 0, 1e-8, std_prop_vel)
    
    bin_vel_zs = (prop_bins_vel - np.array(prop_valid_vel)) / std_prop_vel
    
    plt.figure(figsize=(8, 6))
    plt.imshow(bin_vel_zs, cmap="cividis", aspect="auto")
    plt.gca().set_aspect(1 / (np.shape(bin_vel_zs)[0] / np.shape(bin_vel_zs)[1]))
    plt.colorbar(label="Z-score")
    plt.title("Velocity Outlier Binned Z-Scores")
    plt.show()
    
    return bin_vel_zs
#%% Runs binned zs and velocity funcitons
# 1. NaN Binned Z-Score Analysis
bin_zs = analyze_nan_binned_zscores(coords, bin_length_frames=30)

#%% 2. Velocity Performance Calculation & Time Traces
all_node_velocity = calc_velocity(coords)
plot_velocity_time_traces(all_node_velocity, node_names=names)

#%% 3. Velocity Boxplots
plot_velocity_boxplots(all_node_velocity, node_names=names)

#%% 4. Threshold & Proportion Analysis
vel_threshold = 30
plot_velocity_boxplots(all_node_velocity, node_names=names, vel_threshold=vel_threshold)
plot_velocity_threshold_proportions(all_node_velocity, node_names=names, vel_threshold=vel_threshold)

#%% 5. Velocity Outlier Z-Score Heatmap
bin_vel_zs = analyze_velocity_outlier_zscores(all_node_velocity, vel_threshold=vel_threshold)

#%% Purely model evaluation NOT evluation of inference
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sleap_nn.evaluation import load_metrics, Evaluator
from pathlib import Path
import sleap_nn
#%%% Path definitions

model_path = r"C:\Users\cns-th-lab\SLEAP_Projects\models\260523_198_199x_237x_238x_274x_400x_402x_424x_483x.centered_instance.n=222"
validation_metrics_path = r"C:\Users\cns-th-lab\SLEAP_Projects\models\260523_198_199x_237x_238x_274x_400x_402x_424x_483x.centered_instance.n=222\metrics.val.0.npz"

#%%% Metrics loading
metrics = sleap_nn.evaluation.load_metrics(validation_metrics_path)
print("\n".join(metrics.keys()))

print("Error distance (50%):", metrics["distance_metrics"]["p50"])
print("Error distance (90%):", metrics["distance_metrics"]["p90"])
print("Error distance (95%):", metrics["distance_metrics"]["p95"])

#%%% Visualize localization error
plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
sns.histplot(metrics["distance_metrics"]["dists"].flatten(), binrange=(0, 20), kde=True, kde_kws={"clip": (0, 20)}, stat="probability")
plt.xlabel("Localization error (px)");
plt.show()

#%%% Plot OKS Scores
plt.figure(figsize=(6, 3), dpi=150, facecolor="w")
sns.histplot(metrics["voc_metrics"]["oks_voc.match_scores"].flatten(), binrange=(0, 1), kde=True, kde_kws={"clip": (0, 1)}, stat="probability")
plt.xlabel("Object Keypoint Similarity");
plt.show()

plt.figure(figsize=(4, 4), dpi=150, facecolor="w")
for precision, thresh in zip(metrics["voc_metrics"]['oks_voc.precisions'][::2], metrics["voc_metrics"]["oks_voc.match_score_thresholds"][::2]):
    plt.plot(metrics["voc_metrics"]["oks_voc.recall_thresholds"], precision, "-", label=f"OKS @ {thresh:.2f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower left");
plt.show()

#%%% Want these to be close to 1
print("mAP:", metrics["voc_metrics"]["oks_voc.mAP"])
print("mAR:", metrics["voc_metrics"]["oks_voc.mAR"])

#%%% Can generate more ground truth and reevaluate with the following
from sleap_nn.predict import run_inference
import sleap_io as sio
from sleap_nn.evaluation import Evaluator

#Generate new prediction for ground truth
new_ground_truth_labels = "test.pkg.slp" #Must be .pkg.slp to include images
labels_gt = sio.load_slp(new_ground_truth_labels)
labels_pr = run_inference(data_path=new_ground_truth_labels, model_paths=[model_path])

evals = Evaluator(labels_gt, labels_pr)
metrics = evals.evaluate()

print("Error distance (50%):", metrics["distance_metrics"]["p50"])
print("Error distance (90%):", metrics["distance_metrics"]["p90"])
print("Error distance (95%):", metrics["distance_metrics"]["p95"])
print("mAP:", metrics["voc_metrics"]["oks_voc.mAP"])
print("mAR:", metrics["voc_metrics"]["oks_voc.mAR"])