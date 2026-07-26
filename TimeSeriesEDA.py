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
sess_ids = ["116543"]#,"116498"] #Currently manually specified
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
label_files = []
mode = "sess" #all, sess, idx
if mode == "sess":
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
               "tracks": []}
node_names = []
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
        labels_dict["sess"].append(str.removeprefix(os.path.splitext(os.path.basename(file))[0], "mov_"))
        labels_dict["scores"].append(scores)
        labels_dict["tracks"].append(tracks_coords)

labels_df = pd.DataFrame(labels_dict)
labels_df.set_index('sess')

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
#%%
for n in range(len(node_names)):
    print(n)
    plt.scatter(labels_df["cubic_interpol_tracks"][0][0][n][0][0], labels_df["cubic_interpol_tracks"][0][0][n][1][0], color="red")
    plt.scatter(labels_df["rotated_tracks"][0][0][n][0][0], labels_df["rotated_tracks"][0][0][n][1][0], color="blue")
    #print(np.shape(labels_df["rotated_tracks"][0]))
    ##print(labels_df["rotated_tracks"][0][0,n,0,0])
    #print(labels_df["rotated_tracks"][0][0,n,1,0])
    #plt.scatter(labels_df["rotated_tracks"][0][0,n,0,0], labels_df["rotated_tracks"][0][0,n,1,0], color="blue")
plt.show()

#%% Port label paths and dataframe assignment

port_label_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260716_port_model",
                    r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260716_port_model"]
                    
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
print(sess_ids)
print(wm_sess_data.head())
print(wm_sess_data.columns.tolist())
print(wm_sess_data.iloc[0])
#%%
print(wm_sess_data.iloc[1])

#%%
print(sess_ids)
print(bandit_sess_data.head())
print(bandit_sess_data.columns.tolist())
#%%

#for sess_id in sess_ids:
sess_start = wm_sess_data["starttime"]
trial_abs_time = wm_sess_data["trialtime"]

print(sess_start[0])
print(trial_abs_time[0])
trial_time_rel_start = trial_abs_time - sess_start
#%%
print(trial_time_rel_start[0].time().microsecond)

#%%Relative center poke in times with None placeholders for invalid trials
cpoke_in_times_rel = [(trial_time_rel_start[i] + pd.Timedelta(wm_sess_data["cpoke_in_time"][i], unit = "s")).time()
                      if not np.isnan(wm_sess_data["cpoke_in_time"][i]) 
                      else None
                      for i in range(len(trial_abs_time))]
print(cpoke_in_times_rel)

#%%Relative center poke in times without None placeholders for invalid trials
cpoke_in_times_rel = [(trial_time_rel_start[i] + pd.Timedelta(wm_sess_data["cpoke_in_time"][i], unit = "s")).time()
                      for i in range(len(trial_abs_time))
                      if not np.isnan(wm_sess_data["cpoke_in_time"][i])]
print(cpoke_in_times_rel)

#%%Read video doric times
from sys_neuro_tools import doric_utils as du
active_sess_vid_doric = r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\mov_116543.doric"
du.h5print(active_sess_vid_doric)
time_in, time_in_info = du.h5read(active_sess_vid_doric,['DataAcquisition','BehaviorCamera','Video','Series0001','DMK-33UX290','Time']);
print(time_in)
print(time_in_info)
#%%
print(np.shape(labels_df.iloc[0]["tracks"])[0])
print(len(time_in))
