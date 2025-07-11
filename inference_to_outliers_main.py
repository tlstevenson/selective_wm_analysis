#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:13:20 2025

@author: alex
"""

import init
from sys_neuro_tools import sleap_utils
from sys_neuro_tools import math_utils
from pyutils import file_select_ui as fsui
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import pandas as pd

#Initial file reading
hdf5_file_path  = fsui.GetFile("Select an Analysis File")
processed_dict  = sleap_utils.process_hdf5_data(hdf5_file_path)
print(f'Processed dict locations has shape {np.shape(processed_dict["locations"])}')
locations=None
min_frame=0
max_frame=200000
#Remove the dimension for multiple animals
try:
    locations = np.squeeze(processed_dict["locations"][min_frame:max_frame])
except:
    locations = np.squeeze(processed_dict["locations"])
    
#Remove all NaN values with pandas
for node_idx in range(locations.shape[1]):
    locations[:,node_idx,0] = pd.Series(locations[:,node_idx,0]).interpolate()
    locations[:,node_idx,1] = pd.Series(locations[:,node_idx,1]).interpolate()

#Smooth the position data
sigma = 3
smoothed_locations = np.array(locations)
for n_idx in range(smoothed_locations.shape[1]):
    smoothed_locations[:,n_idx,0] = gaussian_filter(smoothed_locations[:,n_idx,0], sigma)
    smoothed_locations[:,n_idx,1] = gaussian_filter(smoothed_locations[:,n_idx,0], sigma)

#region Plot position
fig, ax = plt.subplots(nrows= 6, ncols=2)
fig_s, ax_s = plt.subplots(nrows= 6, ncols=2)
for node_idx in range(locations.shape[1]):
    #Extract positions for the node
    x=locations[:,node_idx,0]
    y=locations[:,node_idx,1]
    
    x_s = smoothed_locations[:,node_idx,0]
    y_s = smoothed_locations[:,node_idx,1]
    
    f_axis = range(len(x))
    
    ax[node_idx//2, node_idx%2].plot(f_axis, x, color="blue", linestyle='--', label="x position")        
    ax[node_idx//2, node_idx%2].plot(f_axis, y, color="red", linestyle='--', label="y position")
    fig.suptitle("Position x(bllue) y(red) Unsmoothed")
    
    ax_s[node_idx//2, node_idx%2].plot(f_axis, x_s, color="blue", linestyle='--', label="x position")        
    ax_s[node_idx//2, node_idx%2].plot(f_axis, y_s, color="red", linestyle='--', label="y position")
    fig_s.suptitle("Position x(bllue) y(red) Smoothed")


#region Plot velocity
fig2, ax2 = plt.subplots(nrows= 6, ncols=2)
fig2_s, ax2_s = plt.subplots(nrows= 6, ncols=2)
v = np.zeros((locations.shape[1], locations.shape[0]-1))
v_s = np.zeros((locations.shape[1], locations.shape[0]-1))
for node_idx in range(locations.shape[1]):
    #Extract positions for the node
    x=locations[:,node_idx,0]
    y=locations[:,node_idx,1]
    
    #Extract smoothed positions for the node
    x_s=smoothed_locations[:,node_idx,0]
    y_s=smoothed_locations[:,node_idx,1]
    
    #Get x and y velocities
    x_diff = math_utils.GetDiff(x)
    y_diff = math_utils.GetDiff(y)
    
    #Get smoothed x and y velocities
    x_diff_s = math_utils.GetDiff(x_s)
    y_diff_s = math_utils.GetDiff(y_s)
    
    #Get overall velocities not smoothed and smoothed
    for i in range(len(x_diff)):
        v[node_idx, i] = np.sqrt(np.square(x_diff[i]) + np.square(y_diff[i]))
        v_s[node_idx, i] = np.sqrt(np.square(x_diff_s[i]) + np.square(y_diff_s[i]))
    f_axis = range(len(x_diff))
    
    ax2[node_idx//2, node_idx%2].plot(f_axis, v[node_idx], color="purple", linestyle='--')
    fig2.suptitle("Velocity Unsmoothed")
    ax2_s[node_idx//2, node_idx%2].plot(f_axis, v_s[node_idx], color="purple", linestyle='--')
    fig2_s.suptitle("Velocity Smoothed")

#region Plot acceleration
fig3, ax3 = plt.subplots(nrows= 6, ncols=2)
fig3_s, ax3_s = plt.subplots(nrows= 6, ncols=2)
accel = np.zeros((v.shape[0], v.shape[1]-1))
accel_s = np.zeros((v_s.shape[0], v_s.shape[1]-1))
for node_idx in range(v.shape[0]):
    a = math_utils.GetDiff(v[node_idx]) 
    a_s = math_utils.GetDiff(v_s[node_idx])
    f_axis = range(len(a))
    accel[node_idx,:] = a
    accel_s[node_idx,:] = a_s
    ax3[node_idx//2, node_idx%2].plot(f_axis, a, color="red", linestyle='--')
    fig3.suptitle("Acceleration Unsmoothed")
    ax3_s[node_idx//2, node_idx%2].plot(f_axis, a_s, color="red", linestyle='--')
    fig3_s.suptitle("Acceleration Smoothed")
    
#See how many outliers each threshold produces
for accel_thresh in range(1,101, 10):
    print(f"Accel thresh: {accel_thresh}")
    mask = accel > accel_thresh
    mask_s = accel_s > accel_thresh
    print(f"Shape mask: {np.shape(mask)}")
    print("Sum Shape")
    print(np.shape(np.sum(mask, axis=0)))
    #print(len(np.argwhere(np.sum(mask, axis=0))))
    a_idxs = np.argwhere(np.sum(mask, axis=0)) #Get frame indices where there is at least one outlier
    a_idxs_s = np.argwhere(np.sum(mask_s, axis=0))
    #a0 comes from v0 and v1 which come from x0, x1, and x
    #If a0 is outlier then plot 0,1,2 for position
    print(f"Number Outliers: {len(a_idxs)}")
    print(f"Number Outliers Smoothed: {len(a_idxs_s)}")

accel_thresh = 20
mask = accel > accel_thresh
a_idxs = np.argwhere(np.sum(mask, axis=0)) #Get frame indices where there is at least one outlier

#NOTE: Must be run with inline plots
"""for a_idx in a_idxs:
    fig4, ax4 = plt.subplots()
    #print(locations[a_idx])
    sleap_utils.PlotSkeleton(np.squeeze(locations[a_idx]), processed_dict, skeleton_color="green", nodes_mark=mask[:,a_idx],ax=ax4)
    sleap_utils.PlotSkeleton(np.squeeze(locations[a_idx+1]), processed_dict, skeleton_color="blue", nodes_mark=mask[:,a_idx],ax=ax4)
    sleap_utils.PlotSkeleton(np.squeeze(locations[a_idx+2]), processed_dict, skeleton_color="purple", nodes_mark=mask[:,a_idx],ax=ax4)
    ax4.set_title(f"{a_idx},{a_idx+1},{a_idx+2}, Time: {a_idx/60} seconds")
    #plt.show()"""