# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 10:32:08 2026

@author: cns-th-lab
"""

import init
import pandas as pd
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
#%% A way to specify all sess ids
sess_ids = ["116498"]#"116543"]#,"116498"] #Currently manually specified
#%% A way to get all data from sess_ids
label_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               ]
port_label_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260716_port_model"
                    ]
label_files = []
mode = "sess" #all, sess, idx
if mode == "sess": #Get all files belonging to sess_ids
    label_files = [os.path.join(folder, file) for folder in label_paths for file in os.listdir(folder) if os.path.splitext(file)[1] == ".h5" and os.path.splitext(file)[0] in sess_ids]
    for folder in label_paths:
        for file in os.listdir(folder):
            name, ext = os.path.splitext(file)
            if ext == ".h5" and str.removeprefix(name, "mov_") in sess_ids:
                label_files.append(os.path.join(folder, file))
elif mode == "all":
    label_files = [os.path.join(folder, file) for folder in label_paths for file in os.listdir(folder) if os.path.splitext(".h5")]

#%% Thresholding functions
def ThresholdedPositions(df_row, threshold):
    print(np.shape(df_row["scores"]))
    print(np.shape(df_row["tracks"]))
    mask = df_row["scores"] < threshold
    #positions[mask] = np.nan
    #return positions
    df_row["tracks"][:, :, 0,:][mask] = np.nan
    df_row["tracks"][:, :, 1,:][mask] = np.nan
    return df_row["tracks"]
    
#%% Extract raw positions
labels_dict = {"sess": [],
               "scores": [],
               "tracks": [],
               "port_tracks":[]}
node_names = []
edge_names = []
port_names = []
for file in label_files:   
    with h5py.File(file, "r") as f:
        # Decode node names
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        edge_names = [[n1.decode('utf-8'), n2.decode('utf-8')] for n1, n2 in f["edge_names"][:]]
        if(label_files.index(file) ==0):
            print(edge_names)
            #edge_names = [e.decode('utf-8') for e in f['edge_names'][:]]
            #print(edge_names)
        
        # 1. Get Prediction Scores 
        # Raw shape: (tracks, nodes, frames) -> Transposed: (frames, nodes, tracks)
        scores = np.transpose(f['point_scores'][:], (2, 1, 0)) 
        
        # 2. Get Coordinates
        # Raw shape: (tracks, nodes, 2, frames) -> Transposed: (frames, nodes, 2, tracks)
        tracks_coords = np.transpose(f['tracks'][:])
        
        # 4. Set Dictionary Values
        my_sess = str.removeprefix(os.path.splitext(os.path.basename(file))[0], "mov_")
        labels_dict["sess"].append(my_sess)
        labels_dict["scores"].append(scores)
        labels_dict["tracks"].append(tracks_coords)
        
        # Get corresponding port file info
        print("Reset port file name.")
        port_file = None
        for port_dir in port_label_paths:
            for filename in os.listdir(port_dir):
                if my_sess in filename and ".slp" not in filename:
                    port_file = os.path.join(port_dir, filename)
                    print(port_file)
                    break
            if port_file != None:
                break
        print(port_file)
        # Read the data into port_names and a column of the df
        with h5py.File(port_file, "r") as g:
            print("Getting port names")
            port_names = [n.decode('utf-8') for n in g['node_names'][:]]
            print("Getting port tracks")
            port_tracks_coords = np.transpose(g['tracks'][:])
            print("Setting port tracks")
            labels_dict["port_tracks"].append(port_tracks_coords)
            

labels_df = pd.DataFrame(labels_dict)
labels_df.set_index('sess')
#%%
print(np.shape(np.array(labels_df["port_tracks"][0])))

#%% Filter it

# 3. Get Thresholded Coordinates (Only 1 score for both so must index separately)
labels_df["thresh_tracks"] = labels_df.apply(lambda row: ThresholdedPositions(row, .3), axis=1) #Throws error if no valid
#%% Interpolate it (doesnt account for large gaps)
def CubicInterpolation(row, target_column, max_dist):
    data_shape = np.shape(row[target_column])
    for node in range(data_shape[1]):
        for i in range(2):
            for t in range(data_shape[3]):
                my_series = pd.Series(row[target_column][:,node,i,t])
                row[target_column][:,node,i,t] = my_series.interpolate(method="polynomial", order=3, limit=max_dist)
    return row[target_column]
labels_df["cubic_interpol_tracks"] = labels_df.apply(lambda row: CubicInterpolation(row, "thresh_tracks", 15), axis=1)
#%% Center and rotate it
def NodePositionsLocal(row, target_column, right_ortho=True):
    '''
    Returns the node positions in a local coordinate system.
    The first basis vector is from the body to the neck.
    The second basis vector is orthogonal and on the right side of the body.
    '''
    # Ensure session data is a NumPy array: shape (frames, nodes, 2, 1)
    session = np.array(row[target_column])
    
    origin_idx = node_names.index("spine_2")
    basis_idx = node_names.index("spine_1")
    
    # 1. Extract origin and basis points (Shape: frames, 2, 1)
    p_origin = session[:, origin_idx, :, :]
    p_basis = session[:, basis_idx, :, :]
    
    # 2. Center nodes relative to the origin
    # We add a new axis for 'nodes' so it broadcasts to (frames, nodes, 2, 1)
    centered_pos = session - p_origin[:, np.newaxis, :, :]
    
    # 3. Calculate basis vectors
    # We index [..., 0] to temporarily ignore the trailing 1, making math easier (Shape: frames, 2)
    b1 = p_basis[..., 0] - p_origin[..., 0]
    
    # Calculate norms and protect against divide-by-zero
    norms = np.linalg.norm(b1, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    
    # Normalize to unit vectors
    u = b1 / norms  
    
    # Extract X and Y unit components
    u_x = u[:, 0]
    u_y = u[:, 1]
    
    # 4. Build the rotation matrices for all frames (Shape: frames, 2, 2)
    frames = session.shape[0]
    R = np.empty((frames, 2, 2))
    
    R[:, 0, 0] = u_x
    R[:, 0, 1] = u_y
    R[:, 1, 0] = -u_y
    R[:, 1, 1] = u_x
    
    # 5. Apply rotations
    # f = frames, n = nodes, i = new coord (2), j = old coord (2), k = trailing dim (1)
    # This precisely multiplies the 2x2 matrix into the (2, 1) vector for every node in every frame.
    local_locations = np.einsum('fij, fnjk -> fnik', R, centered_pos)
    
    # Convert back to a list of arrays (each array being shape (nodes, 2, 1))
    return list(local_locations)

labels_df["rotated_tracks"] = labels_df.apply(lambda row:NodePositionsLocal(row, "cubic_interpol_tracks"), axis=1)
#%% Outlier detection pipeline
#%%% Clean a video data
def generate_clean_batches(local_coords_param):
    """Extract valid local positions of nodes from a video tracking numpy array.
    
    Args:
        local_coords_param (float): A video tracking position array (frames, nodes, 2, tracks{should be 1})
        
    Returns:
        A non-homogenous list of lists by node of valid vectors (for use in K-means)
    """
    #Get local_coordinates for this video and remove tracks dimension
    local_coords = np.squeeze(local_coords_param)
    print(f"Initial shape: {np.shape(local_coords)}")
    
    valid_nodes_pos = []
    #Iterate through nodes, dropping NaN values
    for n_idx in range(np.shape(local_coords)[1]):
        node_data = local_coords[:,n_idx,:]
        print(f"Node data shape {n_idx}: {np.shape(node_data)}")
        nan_mask = np.isnan(node_data).any(axis=1)
        valid_mask = ~np.isnan(node_data).any(axis=1)
        print(f"Mask data shape {n_idx}: {np.shape(valid_mask)}")
        clean_data = node_data[valid_mask]
        print(f"Masked data shape {n_idx}: {np.shape(clean_data)}")
        valid_nodes_pos.append(clean_data)
        print(f"Number NaN entries: {np.sum(nan_mask)}")
        print(f"Non-NaN + NaN = Total: {np.sum(nan_mask) + np.shape(clean_data)[0] == np.shape(node_data)[0]}")
        print()
        
    return valid_nodes_pos

def scatter_node_local_pos(node_name, node_names, valid_local_positions):
    """Visualize the distribution of a node in local space
    
    Args:
        node_name: The node of interest
        node_names: The array of node names corresponding to the indices of 
            valid_local_positions
        valid_local_positions[[float]]: a non-homogenous list of lists of valid 
            local positions by node"""
    
    node_data = valid_local_positions[node_names.index(node_name)]
    plt.scatter(node_data[:,0], node_data[:,1], label=node_name)
    plt.title(f"{node_name}")
    #plt.show()

#TODO: REMOVE dependency on global variable edge_names
def hist_edge(edge_names, node_names, local_coords_param):
    """Plotting function for drawing histograms of limb length
    
    Args:
        edge_names [[string, string]]: List of node name pairs
        node_names [string]: list of node names
        all_nodes_valid_pos: non-homogenous list of valid (x,y) node positions by node
    
    Returns:
        Summary statistics by node"""
        
    #Convert edge name pairs to edge index pairs
    edge_idx = [[node_names.index(name_1), node_names.index(name_2)] for name_1, name_2 in edge_names]
    
    #Get local_coordinates for this video and remove tracks dimension
    local_coords = np.squeeze(local_coords_param)
    print(f"Initial shape: {np.shape(local_coords)}")
        
    #Iterate through edges, dropping NaN values where either is NaN
    for n1, n2 in edge_idx:
        node_1_data = local_coords[:,n1,:]
        node_2_data = local_coords[:,n2,:]
        
        valid_mask_1 = ~np.isnan(node_1_data).any(axis=1)
        valid_mask_2 = ~np.isnan(node_2_data).any(axis=1)
        valid_mask = valid_mask_1 & valid_mask_2
        print(f"Mask data shape: {np.shape(valid_mask)}")
        edge_vects = node_1_data[valid_mask] - node_2_data[valid_mask]
        print(f"Masked data shape: {np.shape(edge_vects)}")
        edge_lengths = np.linalg.norm(edge_vects, axis=1)
        
        plt.hist(edge_lengths, bins = 20)
        plt.title(f"{node_names[n1]} -> {node_names[n2]} Edge Length Histogram")
        plt.show()        
        
        plt.boxplot(edge_lengths)
        plt.title(f"{node_names[n1]} -> {node_names[n2]} Edge Length Boxplot")
        plt.show()
#%%
hist_edge(edge_names, node_names, np.array(labels_df["rotated_tracks"][0]))

#%%% Importts for model training
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KernelDensity
import numpy as np
import matplotlib.pyplot as plt

# %%% Outlier Detection Functions

def detect_outliers_isolation_forest(node_data, contamination=0.01):
    """
    Applies Isolation Forest to a single node's valid 2D positions.
    
    Returns:
        boolean mask (True means the point is an outlier)
    """
    # Initialize the model
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    
    # Fit and predict (-1 for outliers, 1 for inliers)
    preds = iso_forest.fit_predict(node_data)
    
    # Convert to a boolean mask (True for outliers)
    outlier_mask = (preds == -1)
    return outlier_mask

def detect_outliers_kde(node_data, percentile_threshold=1.0, bandwidth=1.0):
    """
    Applies Kernel Density Estimation to a single node's valid 2D positions.
    
    Returns:
        boolean mask (True means the point is an outlier)
    """
    # Initialize and fit the KDE model
    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth)
    kde.fit(node_data)
    
    # Get log-density scores for all points
    log_density_scores = kde.score_samples(node_data)
    
    # Determine the density threshold based on the specified percentile (e.g., bottom 1%)
    threshold = np.percentile(log_density_scores, percentile_threshold)
    
    # Points with a density strictly less than the threshold are outliers
    outlier_mask = log_density_scores < threshold
    return outlier_mask

# %%% Visualization Function

def scatter_node_outliers(node_name, node_data, outlier_mask, algorithm_name):
    """
    Visualizes the inliers and outliers for a specific node.
    """
    inliers = node_data[~outlier_mask]
    outliers = node_data[outlier_mask]
    
    plt.figure(figsize=(8, 6))
    
    # Plot inliers (blue, slightly transparent)
    plt.scatter(inliers[:, 0], inliers[:, 1], c='blue', s=5, alpha=0.3, label='Inliers')
    
    # Plot outliers (red, larger, opaque)
    plt.scatter(outliers[:, 0], outliers[:, 1], c='red', s=15, alpha=1.0, label='Outliers')
    
    plt.title(f"{node_name} - {algorithm_name} Outliers", fontsize=14)
    plt.xlabel("Local X", fontsize=12)
    plt.ylabel("Local Y", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

#%%% Plots scatterplots of local positions for nodes across a video

first_vid_clean_batches = generate_clean_batches(labels_df["rotated_tracks"][0])
for node in node_names:
    scatter_node_local_pos(node, node_names, first_vid_clean_batches)
plt.legend()
plt.show()
    
#%%% Plots scatterplots of local positions for nodes across a video (outliers)
print("Running Outlier Detection...\n")

for i, node in enumerate(node_names):
    # Get the (N, 2) array of clean data for this specific node
    node_data = first_vid_clean_batches[i]
    
    if len(node_data) == 0:
        print(f"Skipping {node}: No valid data points.")
        continue
        
    print(f"Processing {node} ({len(node_data)} points)...\n")
    
    # --- 1. Isolation Forest ---
    print(f"Running Isolation Forest...")
    iso_mask = detect_outliers_isolation_forest(node_data, contamination=0.01)
    
    # --- 2. Kernel Density Estimation ---
    print(f"Running KDE...")
    # bandwidth may need tuning depending on the scale of your local coordinates
    kde_mask = detect_outliers_kde(node_data, percentile_threshold=1.0, bandwidth=2.0)
    
    # Plot the results side-by-side to compare
    scatter_node_outliers(node, node_data, iso_mask, algorithm_name="Isolation Forest")
    scatter_node_outliers(node, node_data, kde_mask, algorithm_name="Kernel Density Estimation")

#TODO: Might be a good idea to add above/below port for nose to be able to tell which direction it's coming from
#%% Plot whole frame with nose port angle labeled
for n in range(len(node_names)):
    print(n)
    #Plot body (inverted y to account for image coordinates)
    plt.scatter(labels_df["cubic_interpol_tracks"][0][0][n][0][0], labels_df["cubic_interpol_tracks"][0][0][n][1][0], color="red")
#Plot ports
plt.scatter(labels_df["port_tracks"][0][0][0][0][0], labels_df["port_tracks"][0][0][0][1][0], color="purple")
plt.annotate("Left Port", (labels_df["port_tracks"][0][0][0][0][0], labels_df["port_tracks"][0][0][0][1][0]))
plt.scatter(labels_df["port_tracks"][0][0][1][0][0], labels_df["port_tracks"][0][0][1][1][0], color="purple")
plt.annotate("Center Port", (labels_df["port_tracks"][0][0][1][0][0], labels_df["port_tracks"][0][0][1][1][0]))
plt.scatter(labels_df["port_tracks"][0][0][2][0][0], labels_df["port_tracks"][0][0][2][1][0], color="purple")
plt.annotate("Right Port", (labels_df["port_tracks"][0][0][2][0][0], labels_df["port_tracks"][0][0][2][1][0]))
#Plot arrows and write angle for visualization
nose_x = np.array(labels_df["cubic_interpol_tracks"][0])[0,nose_idx,0,0]
nose_y = np.array(labels_df["cubic_interpol_tracks"][0])[0,nose_idx,1,0]
implant_x = np.array(labels_df["cubic_interpol_tracks"][0])[0,implant_idx,0,0]
implant_y = np.array(labels_df["cubic_interpol_tracks"][0])[0,implant_idx,1,0]
port_0_x = np.array(labels_df["port_tracks"][0])[0,0,0,0]
port_0_y = np.array(labels_df["port_tracks"][0])[0,0,1,0]
port_1_x = np.array(labels_df["port_tracks"][0])[0,1,0,0]
port_1_y = np.array(labels_df["port_tracks"][0])[0,1,1,0]
port_2_x = np.array(labels_df["port_tracks"][0])[0,2,0,0]
port_2_y = np.array(labels_df["port_tracks"][0])[0,2,1,0]
#Implant-nose
plt.quiver(implant_x, implant_y, 
           nose_x-implant_x, nose_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='blue')
#Implant-port_x
plt.quiver(implant_x, implant_y,
           port_0_x-implant_x, port_0_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='orange', label=angles_0[0])
plt.quiver(implant_x, implant_y,
           port_1_x-implant_x, port_1_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='yellow', label=angles_1[0])
plt.quiver(implant_x, implant_y,
           port_2_x-implant_x, port_2_y-implant_y,
           angles='xy', scale_units='xy', scale=1, color='green', label=angles_2[0])
plt.xlim(0,1440)
plt.ylim(1080,0) #Inverted for image
plt.legend()
plt.show()
#%% Access fp and behavioral data (+ imports)
#%%%Imports
from hankslab_db import db_access
#import doric_utils as du
import numpy as np
import os
#from pathlib import Path
from hankslab_db import tonecatdelayresp_db as wm_db, basicRLtasks_db as bandit_db
#from sys_neuro_tools import sleap_utils

#from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
#from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
#%%% Read trial data from database 
wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

wm_sess_data = wm_loc_db.get_behavior_data(sess_ids)
bandit_sess_data = wm_loc_db.get_behavior_data(sess_ids)
#%%% Helper functions for trial ends and intervals
def get_trial_end_ts(sess_data):
    """Get the last state timestamp from a trial to determine its end relative to the start.
    
    Args:
        sess_data(TODO???): Takes a session data from db_access
    Returns:
        A list of time deltas relative to the start of the trial indicating relative trial end times.
    """
    print(sess_data["parsed_events"][0]["States"])
    print(type(sess_data["parsed_events"][0]["States"]))
    trial_end_ts_vect = []
    for trial in range(len(sess_data["parsed_events"])):
        max_val = 0
        for key, value in sess_data["parsed_events"][trial]["States"].items():
            if value == [None, None]:
                continue
            else:
                max_val = max(max_val, value[1])
        trial_end_ts_vect.append(max_val)
    return trial_end_ts_vect

def pose_in_intervals(frame_timestamps, coords, intervals):
    """A function that gets a non-homogenous list of coordinates by time intervals.
    
    It uses the time since the start from the intervals to find the correct frames
    in the video and get a list of pose_data for each interval
    
    Args:
        frame_timestamps (float[]): 1d array of timestamps for each frame
        coords (float[,,,]): frames x nodes x 2 x tracks SLEAP array
        intervals (List<(float, float)>): intervals in which to get the data
        
    Returns:
        A non-homogenous list of slices corresponding to the intervals."""
    pose_data_list = []
    idx_intervals = []
    print(np.shape(frame_timestamps))
    print(np.shape(coords))
    print(np.shape(intervals))
    for start_time, end_time in intervals:
        print(start_time)
        print(end_time)
        start_idx = np.searchsorted(frame_timestamps, start_time, side='left')
        end_idx = np.searchsorted(frame_timestamps, end_time, side='right')
        idx_intervals.append([start_idx, end_idx])
        
        # Slice the coordinates array using the found frame indices
        interval_coords = coords[start_idx:end_idx]
        pose_data_list.append(interval_coords)
    
    return pose_data_list, idx_intervals
#%%% Print wm_sess data
print(sess_ids)
print(wm_sess_data.head())
print(wm_sess_data.columns.tolist())
print(wm_sess_data.iloc[0])
#%%% Print bandit sess data
print(sess_ids)
print(bandit_sess_data.head())
print(bandit_sess_data.columns.tolist())
#%%% Print trial start times (No NaNs)
trial_starts_plus_last = db_access.get_fp_trial_start_ts(sess_ids)[int(sess_ids[0])]
trial_starts = trial_starts_plus_last[:-1]
trial_ends_rel_trial_start = get_trial_end_ts(wm_sess_data)
trial_ends = trial_starts + trial_ends_rel_trial_start
print(np.shape(trial_starts))
print(np.shape(trial_ends))
print(f"Num NaNs: {np.sum(np.isnan(trial_starts))}")
print(f"Num NaNs: {np.sum(np.isnan(trial_ends))}")
#%%% Print center poke in times (NaNs for invalid)
print(len(wm_sess_data["cpoke_in_time"]))
print(f"Num NaNs: {np.sum(np.isnan(wm_sess_data['cpoke_in_time']))}")
print(wm_sess_data["cpoke_in_time"].tolist())
cpoke_in_times_vid = trial_starts + wm_sess_data["cpoke_in_time"].tolist()
print(f"Num NaNs: {np.sum(np.isnan(cpoke_in_times_vid))}")
print(cpoke_in_times_vid)
cpoke_in_times_vid_f = 30 * cpoke_in_times_vid #TODO: Paramterize frame rate at the top

#%%% Read video doric times
from sys_neuro_tools import doric_utils as du
active_sess_vid_doric = r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\mov_116498.doric"
du.h5print(active_sess_vid_doric)
time_in, time_in_info = du.h5read(active_sess_vid_doric,['DataAcquisition','BehaviorCamera','Video','Series0001','DMK-33UX290','Time']);
print(time_in)
print(time_in_info)
#%%% Various Intervals
#%%%% Between the end of one and start of next
intervals = np.transpose(np.stack((trial_ends[:-1], trial_starts[1:])))
#%%%% Between start and end
intervals = np.transpose(np.stack((trial_ends[:-1], trial_starts[1:])))
#%%%% Between response cue and response poke
response_cue_abs_time = trial_starts + wm_sess_data["response_cue_time"].tolist()
response_abs_time = trial_starts + wm_sess_data["response_time"].tolist()
intervals = np.transpose(np.stack((response_cue_abs_time, response_abs_time)))
print(intervals)
#%%% Plot interval duration histogram
interval_dur = [end - start for start, end in intervals]
plt.hist(interval_dur,bins=30)
plt.show()

#%%% Use the intervals to slice the pose data
print(f"Shape of intervals: {np.shape(intervals)}")
segmented_poses, segment_idxs = pose_in_intervals(time_in, labels_df["cubic_interpol_tracks"][0], intervals)
for segment in segmented_poses:
    print(np.shape(segment))
#%%% 2d scatter plot of nose position in interval color coded by time
x_sub_1 = []
y_sub_1 = []
t_sub_1 = []
x_1_2 = []
y_1_2 = []
t_1_2 = []
x_2_m = []
y_2_m = []
t_2_m = []
for i in range(len(segmented_poses)):
    abs_t = np.array(pd.Series(time_in[segment_idxs[i][0]:segment_idxs[i][1]]).interpolate())
    if np.shape(abs_t)[0] == 0:
        continue
    rel_t = abs_t - abs_t[0]
    print(rel_t[-1])
    if rel_t[-1] < 1:
        x_sub_1 = x_sub_1 + pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
        y_sub_1 = y_sub_1 + pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
        t_sub_1 = t_sub_1 + rel_t.tolist()
    elif rel_t[-1] > 2:
        x_1_2 = x_1_2 + pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
        y_1_2 = y_1_2 + pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
        t_1_2 = t_1_2 + rel_t.tolist()
    else:
        x_2_m = x_2_m + pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
        y_2_m = y_2_m + pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
        t_2_m = t_2_m + rel_t.tolist()
fig, ax = plt.subplots(nrows=2, ncols=2)
a = ax[0][0].scatter(x_sub_1,y_sub_1,c=t_sub_1, cmap='inferno')
ax[0][0].set_xlim(0,1440)
ax[0][0].set_ylim(1080, 0) #Inverted for image
b = ax[0][1].scatter(x_1_2, y_1_2, c=t_1_2, cmap='inferno')
ax[0][1].set_xlim(0,1440)
ax[0][1].set_ylim(1080, 0) #Inverted for image
c = ax[1][0].scatter(x_2_m, y_2_m, c=t_2_m, cmap='inferno')
ax[1][0].set_xlim(0,1440)
ax[1][0].set_ylim(1080, 0) #Inverted for image
fig.colorbar(a, ax=ax)
fig.colorbar(b, ax=ax)
fig.colorbar(c, ax=ax)
plt.show()
#%%% 3d scatter plot of nose position in interval color coded by time and 
#separated on z axis by time
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
for i in range(len(segmented_poses)):
    abs_t = np.array(pd.Series(time_in[segment_idxs[i][0]:segment_idxs[i][1]]).interpolate())
    if np.shape(abs_t)[0] == 0:
        continue
    rel_t = abs_t - abs_t[0]
    x = pd.Series(segmented_poses[i][:,0,0,0]).interpolate().tolist()
    y = pd.Series(segmented_poses[i][:,0,1,0]).interpolate().tolist()
    t = rel_t.tolist()
    ax.scatter(x,y,t)
ax.set_xlim(0,1440)
ax.set_ylim(1080, 0) #Inverted for image
plt.show()
#%%% Align trajectories
#%%%% Function
def simple_align(segmented_poses, segment_idxs, n_bins=50):
    original_shape = np.shape(segmented_poses[0])
    aligned_trajectories = np.zeros((len(segmented_poses), n_bins, original_shape[1], original_shape[2]))
    for i in range(len(segmented_poses)):
        print(f"Segment #{i+1}")
        if segment_idxs[i][0] >= len(time_in) or segment_idxs[i][1] >= len(time_in):
            continue
        t_start = time_in[segment_idxs[i][0]]
        t_end = time_in[segment_idxs[i][1]]
        time_cut = time_in[segment_idxs[i][0]:segment_idxs[i][1]]
        pose_cut = segmented_poses[i]
        
        t_est = np.linspace(t_start, t_end, n_bins)
        if i==0:
            print(t_start)
            print(t_end)
            print(t_est)
            print()
            print(np.shape(t_est))
            print(np.shape(pose_cut[:,0,i,0]))
            print(np.shape(time_cut))
            print()
        for node in range(np.shape(pose_cut)[1]):
            for j in range(2):
                aligned_trajectories[i,:,node,j] = np.interp(t_est, time_cut, pose_cut[:,node, j, 0])
                print(pose_cut[:,node, j, 0])
                print(aligned_trajectories[i,:,node,j])
    return aligned_trajectories
    
aligned = simple_align(segmented_poses, segment_idxs) #Trials x bins(norm_time) x nodes x 2
for segment in aligned:
    plt.scatter(segment[:,0,0], segment[:,0,1], c=np.arange(0,50))
plt.show()
#%%%% Execution

from sys_neuro_tools import fp_utils as fpu
data = np.array(labels_df["cubic_interpol_tracks"][0])
data_timestamps = time_in
start_timestamps = intervals[:,0]
end_timestamps = intervals[:,1]
n_bins = 100
aligned = fpu.build_time_norm_signal_matrix(data, data_timestamps, start_timestamps, end_timestamps, n_bins)
print(np.shape(aligned))
