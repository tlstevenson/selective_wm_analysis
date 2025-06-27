#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:13:20 2025

@author: alex
"""

import inference_to_outliers_lib as itol
import file_select_ui as fsui
import inference_to_clusters_lib as itcl 
import matplotlib.pyplot as plt
import numpy as np
import math

#Initial file reading
hdf5_file_path  = fsui.GetFile("Select an Analysis File")
processed_dict  = itcl.process_hdf5_data(hdf5_file_path)
frame_bounds= [0,60]
#region Plot position
fig, ax = plt.subplots(nrows= 6, ncols=2)
for node_idx in range(processed_dict["locations"].shape[1]):
    #Extract positions for the node
    x=processed_dict["locations"][frame_bounds[0]:min(frame_bounds[1],processed_dict["locations"].shape[0]),node_idx,0,:]
    y=processed_dict["locations"][frame_bounds[0]:min(frame_bounds[1],processed_dict["locations"].shape[0]),node_idx,1,:] 
    f_axis = range(len(x))
    ax[node_idx//2, node_idx%2].plot(f_axis, x, color="blue", linestyle='--', label="x position")        
    ax[node_idx//2, node_idx%2].plot(f_axis, y, color="red", linestyle='--', label="y position")
plt.legend()
plt.show()

#region Plot velocity NOT OPTIMAL 2 FOR LOOPS
fig, ax = plt.subplots(nrows= 6, ncols=2)
for node_idx in range(processed_dict["locations"].shape[1]):
    #Extract positions for the node
    x=processed_dict["locations"][frame_bounds[0]:min(frame_bounds[1],processed_dict["locations"].shape[0]),node_idx,0,:]
    y=processed_dict["locations"][frame_bounds[0]:min(frame_bounds[1],processed_dict["locations"].shape[0]),node_idx,1,:] 
    x_diff = itol.GetDiff(x)
    y_diff = itol.GetDiff(y)
    f_axis = range(len(x_diff))
    ax[node_idx//2, node_idx%2].plot(f_axis, x_diff, color="blue", linestyle='--', label="x velocity")        
    ax[node_idx//2, node_idx%2].plot(f_axis, y_diff, color="red", linestyle='--', label="y velocity")
plt.legend()
plt.show()
    

"""
#region Plot position outliers by velocity
mask = itol.VelocityOutlierDetection(processed_dict["locations"])
mask= mask==1
fig, ax = plt.subplots(nrows=processed_dict["locations"].shape[1], ncols=1)
for node_idx in range(processed_dict["locations"].shape[1]):
    #Plot each node's position over time
    x=processed_dict["locations"][frame_bounds[0]:min(frame_bounds[1],processed_dict["locations"].shape[0]),node_idx,0,:]
    y=processed_dict["locations"][frame_bounds[0]:min(frame_bounds[1],processed_dict["locations"].shape[0]),node_idx,1,:] 
    f_axis = np.linspace(1, len(x), len(x))
    ax[node_idx].plot(f_axis, x, color="blue", linestyle='--')        
    ax[node_idx].plot(f_axis, y, color="red", linestyle='--')
    for frame_idx in range(frame_bounds[0], frame_bounds[1]):
        #x is an outlier
        if mask[frame_idx, node_idx,0,:]:
            x_val = frame_idx+2 #One since it starts at 1, one since the second point is outlier
            y_val_x = processed_dict["locations"][frame_idx+1,node_idx, 0,:]
            ax[node_idx].scatter(x_val, y_val_x, color="purple")
        #y is an outlier
        if mask[frame_idx, node_idx,1,:]:
            x_val = frame_idx+2 #One since it starts at 1, one since the second point is outlier
            y_val_y = processed_dict["locations"][frame_idx+1,node_idx, 1,:]
            ax[node_idx].scatter(x_val, y_val_y, color="purple")
plt.show()
print(np.sum(mask))"""