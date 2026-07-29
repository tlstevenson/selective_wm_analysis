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
port_names = []
for file in label_files:   
    with h5py.File(file, "r") as f:
        # Decode node names
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        
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
#%% Calculate head port angles
def get_angle(origin_pos, p1, p2):
    v1 = p1-origin_pos
    v2 = p2 - origin_pos #Port vector
    
    v1_n = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
    v2_n = v2 / np.linalg.norm(v2, axis=1, keepdims=True)
    print(np.shape(v1_n))
    print(np.shape(v2_n))
    
    # Calculate dot product
    dot_product = np.sum(v1_n * v2_n, axis=1)
    print(np.shape(dot_product))
    
    # Clip to prevent floating point domain errors
    clipped_dot = np.clip(dot_product, -1.0, 1.0)
    
    return np.degrees(np.arccos(clipped_dot))

nose_idx = node_names.index("nose")
implant_idx = node_names.index("implant")
print(np.shape(labels_df["cubic_interpol_tracks"]))
angles_0 = get_angle(np.array(labels_df["cubic_interpol_tracks"][0])[:,implant_idx,:,0], 
                   np.array(labels_df["cubic_interpol_tracks"][0])[:,nose_idx,:,0], 
                   np.array(labels_df["port_tracks"][0])[:,0,:,0])
angles_1 = get_angle(np.array(labels_df["cubic_interpol_tracks"][0])[:,implant_idx,:,0], 
                   np.array(labels_df["cubic_interpol_tracks"][0])[:,nose_idx,:,0], 
                   np.array(labels_df["port_tracks"][0])[:,1,:,0])
angles_2 = get_angle(np.array(labels_df["cubic_interpol_tracks"][0])[:,implant_idx,:,0], 
                   np.array(labels_df["cubic_interpol_tracks"][0])[:,nose_idx,:,0], 
                   np.array(labels_df["port_tracks"][0])[:,2,:,0])
#TODO: Might be a good idea to add above/below port for nose to be able to tell which direction it's coming from
#%%
print(labels_df["port_tracks"][0][0][0])
#%%
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
#%% Helper function for extraction of skeleton data by interval
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
    print(np.shape(frame_timestamps))
    print(np.shape(coords))
    print(np.shape(intervals))
    for start_time, end_time in intervals:
        print(start_time)
        print(end_time)
        start_idx = np.searchsorted(frame_timestamps, start_time, side='left')
        end_idx = np.searchsorted(frame_timestamps, end_time, side='right')
        
        # Slice the coordinates array using the found frame indices
        interval_coords = coords[start_idx:end_idx]
        pose_data_list.append(interval_coords)
    
    return pose_data_list
#%% Access fp and behavioral data (+ imports)
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
#%% Read trial data from database 
wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
bandit_loc_db = bandit_db.LocalDB_BasicRLTasks('twoArmBandit')

wm_sess_data = wm_loc_db.get_behavior_data(sess_ids)
bandit_sess_data = wm_loc_db.get_behavior_data(sess_ids)
#%%
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
#%%
print(sess_ids)
print(wm_sess_data.head())
print(wm_sess_data.columns.tolist())
print(wm_sess_data.iloc[0])
#%%
print(sess_ids)
print(bandit_sess_data.head())
print(bandit_sess_data.columns.tolist())
#%%Print trial start times (No NaNs)
trial_starts_plus_last = db_access.get_fp_trial_start_ts(sess_ids)[int(sess_ids[0])]
trial_starts = trial_starts_plus_last[:-1]
trial_ends_rel_trial_start = get_trial_end_ts(wm_sess_data)
trial_ends = trial_starts + trial_ends_rel_trial_start
print(np.shape(trial_starts))
print(np.shape(trial_ends))
print(f"Num NaNs: {np.sum(np.isnan(trial_starts))}")
print(f"Num NaNs: {np.sum(np.isnan(trial_ends))}")
#%%Print center poke in times (NaNs for invalid)
print(len(wm_sess_data["cpoke_in_time"]))
print(f"Num NaNs: {np.sum(np.isnan(wm_sess_data['cpoke_in_time']))}")
print(wm_sess_data["cpoke_in_time"].tolist())
#%%
cpoke_in_times_vid = trial_starts + wm_sess_data["cpoke_in_time"].tolist()
print(f"Num NaNs: {np.sum(np.isnan(cpoke_in_times_vid))}")
print(cpoke_in_times_vid)
cpoke_in_times_vid_f = 30 * cpoke_in_times_vid #TODO: Paramterize frame rate at the top

#%%Read video doric times
from sys_neuro_tools import doric_utils as du
active_sess_vid_doric = r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\mov_116498.doric"
du.h5print(active_sess_vid_doric)
time_in, time_in_info = du.h5read(active_sess_vid_doric,['DataAcquisition','BehaviorCamera','Video','Series0001','DMK-33UX290','Time']);
print(time_in)
print(time_in_info)
#%%Various Intervals
#%%% Between the end of one and start of next
intervals = np.transpose(np.stack((trial_ends[:-1], trial_starts[1:])))
#%%% Between start and end
intervals = np.transpose(np.stack((trial_ends[:-1], trial_starts[1:])))
#%%% Between response cue and response poke
response_cue_abs_time = trial_starts + wm_sess_data["response_cue_time"].tolist()
response_abs_time = trial_starts + wm_sess_data["response_time"].tolist()
#%% Use the intervals to slice the pose data
print(f"Shape of intervals: {np.shape(intervals)}")
segmented_poses = pose_in_intervals(time_in, labels_df["cubic_interpol_tracks"][0], intervals)
for segment in segmented_poses:
    print(np.shape(segment))
