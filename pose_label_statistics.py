#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 15:17:22 2025

@author: alex
"""

#%%import statements
import init
from sys_neuro_tools import sleap_utils
from sys_neuro_tools import math_utils
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import pandas as pd
#%%Initial file reading and processing
hdf5_file_paths = ["/Users/alex/Downloads/mov_0001_r_labels.hdf5",
                   "/Users/alex/Downloads/mov_0001_r_labels.hdf5"] #ADD FILEPATHS
processed_dict_list = [0 for i in range(len(hdf5_file_paths))]
file_labels = [0 for i in range(len(hdf5_file_paths))]
locations = [0 for i in range(len(hdf5_file_paths))]

for i in range(len(hdf5_file_paths)):
    processed_dict_list[i] = sleap_utils.process_hdf5_data(hdf5_file_paths[i])
    file_labels[i] = input("Enter file label: ")

#%%Remove the dimension for multiple animals and cut if needed
min_frame=0
max_frame=None
try:
    for i in range(len(processed_dict_list)):
        locations[i] = np.squeeze(processed_dict_list[i]["locations"][min_frame:max_frame])
except:
    for i in range(len(processed_dict_list)):
        locations[i] = np.squeeze(processed_dict_list[i]["locations"])

#%%Define colors for every node (consistent throughout)
# 1. The name of the colormap you want to use (e.g., 'viridis', 'plasma', 'coolwarm', 'jet').
colormap_name = 'viridis' 

# 2. The number of distinct colors you want to generate.
num_colors = len(hdf5_file_paths)

# Get the colormap object from Matplotlib.
cmap = plt.get_cmap(colormap_name)

# Generate a list of colors from the colormap.
# np.linspace(0, 1, num_colors) creates evenly spaced numbers from 0 to 1.
# The colormap object (cmap) is then called like a function to map these numbers to colors.
colors = cmap(np.linspace(0, 1, num_colors))

#%%Plot number of missing values per node
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
num_nan_per_node = np.zeros((len(hdf5_file_paths), locations[0].shape[1],))
for file_idx in range(len(hdf5_file_paths)):
    for node_idx in range(locations[file_idx].shape[1]):
        node_loc = locations[file_idx][:,node_idx, 0]
        nan_count_sum = np.sum(np.isnan(node_loc))
        num_nan_per_node[file_idx, node_idx] = nan_count_sum

#Easy: units_btw_nodes = (num_files + 1) * width
units_btw_nodes = 1
x = np.arange(num_nan_per_node.shape[1]) * units_btw_nodes # the label locations
width = 0.25  # the width of the bars

fig,ax = plt.subplots(layout="constrained")
for node_i in range(num_nan_per_node.shape[1]):
    for file_i in range(num_nan_per_node.shape[0]):
        x_pos = x[node_i] + width * file_i
        if node_i==0:
            ax.bar(x_pos, num_nan_per_node[file_i,node_i], width, label=file_labels[file_i],color=colors[file_i])
        else:
            ax.bar(x_pos, num_nan_per_node[file_i,node_i], width, color=colors[file_i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Num NaN")
ax.set_title('Num NaN by Node')
ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show()    
    
#%% Plot number of gaps per node
num_gaps_per_node =  np.zeros((len(hdf5_file_paths), locations[0].shape[1],))
for file_idx in range(len(hdf5_file_paths)):
    for node_idx in range(locations[file_idx].shape[1]):
        # We can find the start of each interval by looking for a False -> True transition
        # in the boolean mask.
        is_nan_array = np.isnan(locations[file_idx][:,node_idx,0]) #POTENTIAL ERROR: Only checking x
        
        # Convert boolean to int (False=0, True=1) and find the difference.
        # A change from 0 to 1 (a number to a NaN) will result in a diff of 1.
        nan_starts = np.diff(is_nan_array.astype(int)) == 1
        
        # The count is the number of times a new interval starts.
        interval_count = np.sum(nan_starts)
        
        # Edge Case: We must also check if the very first value is a NaN.
        # np.diff won't catch this, as there's no preceding value.
        if is_nan_array[0]:
            interval_count += 1
        
        num_gaps_per_node[file_idx][node_idx] = interval_count
        
#Easy: units_btw_nodes = (num_files + 1) * width
units_btw_nodes = 1
x = np.arange(num_nan_per_node.shape[1]) * units_btw_nodes # the label locations
width = 0.25  # the width of the bars

fig,ax = plt.subplots(layout="constrained")
for node_i in range(num_nan_per_node.shape[1]):
    for file_i in range(num_nan_per_node.shape[0]):
        x_pos = x[node_i] + width * file_i
        if node_i==0:
            ax.bar(x_pos, num_gaps_per_node[file_i,node_i], width, label=file_labels[file_i],color=colors[file_i])
        else:
            ax.bar(x_pos, num_gaps_per_node[file_i,node_i], width, color=colors[file_i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Num Gaps")
ax.set_title('Num Gaps by Node')
ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show()    

#%%Gap length and dx for each gap (Mean and pooled confidence interval)
#Gap length is number to number not number of NaN
#The four are length mean, length std, dx mean, dx std
gap_length_dx = [[[] for j in range(locations[0].shape[1])] for i in range(len(hdf5_file_paths))]
gap_length_f = [[[] for j in range(locations[0].shape[1])] for i in range(len(hdf5_file_paths))]

for file_idx in range(len(hdf5_file_paths)):
    for node_idx in range(locations[file_idx].shape[1]):
        x_pos_node = locations[file_idx][:,node_idx,0]
        y_pos_node = locations[file_idx][:,node_idx,1]
        
        
        # 1. Correctly identify where EITHER x OR y is NaN
        nan_comb = np.isnan(x_pos_node) | np.isnan(y_pos_node)
        
        # Convert boolean to int (False=0, True=1) and find the indices of start and stop
        # A change from 0 to 1 (a number to a NaN) will result in a diff of 1.
        # A change from 1 to 0 (a NaN to a number) will result in a diff of -1.
        nan_starts = np.where(np.diff(nan_comb.astype(int)) == 1)[0]
        nan_stops = np.where(np.diff(nan_comb.astype(int)) == -1)[0]
        
        nan_stops = nan_stops + 1 #Makes sure it lands on first number not last NaN
        
        if nan_comb[0] == True:
            nan_stops = np.delete(nan_stops,0)
            print("Starts with a nan")
            
        if nan_comb[-1] == True:
            nan_starts = np.delete(nan_starts, -1)
            print("Ends with a nan")
        
        if len(nan_starts) != len(nan_stops):
            raise ValueError("The number of starts doesnt match number of stops")
        
        intervals = [(nan_starts[i], nan_stops[i]) for i in range(len(nan_starts))]
        interval_lengths = np.array([i[1] - i[0] for i in intervals])
        interval_dx = np.zeros((len(intervals),))
        for i in range(len(intervals)):
            x0 = locations[file_idx][intervals[i][0],node_idx,0]
            xf = locations[file_idx][intervals[i][1],node_idx,0]
            y0 = locations[file_idx][intervals[i][0],node_idx,1]
            yf = locations[file_idx][intervals[i][1],node_idx,1]
            diff_x = xf - x0
            diff_y = yf - y0
            interval_dx[i] = math.sqrt(diff_x*diff_x + diff_y*diff_y)        
        #gap_length_dx_matrix[file_idx, node_idx, 0] = np.mean(interval_lengths)
        #gap_length_dx_matrix[file_idx, node_idx, 1] = np.std(interval_lengths)
        #gap_length_dx_matrix[file_idx, node_idx, 2] = np.mean(interval_dx)
        #gap_length_dx_matrix[file_idx, node_idx, 3] = np.std(interval_dx)
        gap_length_dx[file_idx][node_idx] = interval_dx
        gap_length_f[file_idx][node_idx] = interval_lengths

#Visualize the results
#Easy: units_btw_nodes = (num_files + 1) * width
units_btw_nodes = 1
x = np.arange(locations[0].shape[1]) * units_btw_nodes # the label locations(num nodes)
width = 0.25  # the width of the bars

#Visualize average length in frames
fig,ax = plt.subplots(layout="constrained")
for file_i in range(len(hdf5_file_paths)):
    for node_i in range(locations[file_i].shape[1]):
        x_pos = x[node_i] + width * file_i
        print(np.shape(gap_length_f[file_idx][node_i]))
        parts = ax.violinplot(gap_length_f[file_idx][node_i],
                [x_pos], widths=[width], showmeans=False, showmedians=True,
                showextrema=True)
        if node_i==0:
            for pc in parts['bodies']:
                pc.set_label(file_labels[file_i])
                pc.set_facecolor(colors[file_i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
        else:
            for pc in parts['bodies']:
                pc.set_facecolor(colors[file_i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Average Length Gaps (frames)")
ax.set_title('Average Length Gaps by Node')
#ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show()   
        
#Visualize average dx
fig,ax = plt.subplots(layout="constrained")
for file_i in range(len(hdf5_file_paths)):
    for node_i in range(locations[file_i].shape[1]):
        x_pos = x[node_i] + width * file_i
        parts = ax.violinplot(gap_length_dx[file_idx][node_i],
                [x_pos], widths=[width], showmeans=False, showmedians=True,
                showextrema=True)
        if node_i==0:
            for pc in parts['bodies']:
                pc.set_label(file_labels[file_i])
                pc.set_facecolor(colors[file_i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
        else:
            for pc in parts['bodies']:
                pc.set_facecolor(colors[file_i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Average Change During Gaps (pixels)")
ax.set_title('Average Change During Gaps by Node')
#ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show()   

#%% Count velocity and acceleration outliers

#Calculate Velocity and Acceleration
# np.diff calculates the difference between adjacent elements
# axis=0 computes the difference between consecutive frames

velocities_mask = [0 for i in range(len(hdf5_file_paths))]
accelerations_mask = [0 for i in range(len(hdf5_file_paths))]
velocities = [[] for i in range(len(hdf5_file_paths))]
              
for file_idx in range(len(hdf5_file_paths)):
    # Calculate change in x and y coordinates between frames
    dx = np.diff(locations[file_idx][:, :, 0], axis=0)
    dy = np.diff(locations[file_idx][:, :, 1], axis=0)

    print(f"Shape dx: {np.shape(dx)}")
    print(f"Shape dy: {np.shape(dy)}")

    # Calculate velocity using the Pythagorean theorem (vector magnitude)
    # This vectorized approach is much faster than a for loop.
    velocity = np.array(np.sqrt(dx**2 + dy**2)) # Shape: (files, frames-1, nodes)
    velocities[file_idx] = velocity

    # Calculate acceleration (rate of change of velocity)
    acceleration = np.array(np.abs(np.diff(velocity, axis=0))) # Shape: (files, frames-2, nodes)

    velocity_q = np.nanquantile(velocity, [.05,.25,.50,.75,.95], axis=0) #Shape: (5, files, nodes)
    acceleration_q = np.nanquantile(acceleration, [.05,.25,.50,.75,.95], axis=0) #Shape: (5, files, nodes)

    print(f"Shape v_q: {np.shape(velocity_q)}")
    print(f"Shape a_q: {np.shape(acceleration_q)}")

    v_iqr = velocity_q[3,:] - velocity_q[1,:]
    a_iqr = acceleration_q[3,:] - acceleration_q[1,:]

    upper_bounds_v = np.array(velocity_q[3,:] + 2*v_iqr)
    lower_bounds_v = np.array(velocity_q[1,:] - 2*v_iqr)
    upper_bounds_a = np.array(acceleration_q[3,:] + 2*a_iqr)
    lower_bounds_a = np.array(acceleration_q[1,:] - 2*a_iqr)

    print(f"Upper Bounds V: {np.shape(upper_bounds_v)}")
    print(f"Lower Bounds V: {np.shape(lower_bounds_v)}")
    print(f"Upper Bounds A: {np.shape(upper_bounds_a)}")
    print(f"Lower Bounds A: {np.shape(lower_bounds_a)}")

    #Create a mask that says which frame node positions are outliers
    v_out_mask = np.zeros(velocity.shape)
    a_out_mask = np.zeros(acceleration.shape)

    #Sets velocity and acceleration mask. Only to -1 since velocity longer than acceleration
    for frame_idx in range(velocity.shape[0]-1):
        v_out_mask[frame_idx,:] =  (velocity[frame_idx,:] > upper_bounds_v) | (velocity[frame_idx,:] < lower_bounds_v)
        a_out_mask[frame_idx,:] = (acceleration[frame_idx,:] > upper_bounds_a) | (acceleration[frame_idx,:] < lower_bounds_a)
    v_out_mask[-1,:] =  (velocity[-1,:] > upper_bounds_v) | (velocity[-1,:] < lower_bounds_v)

    
    velocities_mask[file_idx] = v_out_mask
    accelerations_mask[file_idx] = a_out_mask
    
    #Easy: units_btw_nodes = (num_files + 1) * width
    units_btw_nodes = 1
    x = np.arange(num_nan_per_node.shape[1]) * units_btw_nodes # the label locations
    width = 0.25  # the width of the bars

#Visualize velocity distributions (FIX)
fig,ax = plt.subplots(layout="constrained")
for file_i in range(len(hdf5_file_paths)):
    for node_i in range(locations[file_i].shape[1]):
        x_pos = x[node_i] + width * file_i
        parts = ax.violinplot(velocities[file_i][node_i],
                [x_pos], widths=[width], showmeans=False, showmedians=True,
                showextrema=True)
        if node_i==0:
            for pc in parts['bodies']:
                pc.set_label(file_labels[file_i])
                pc.set_facecolor(colors[file_i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
        else:
            for pc in parts['bodies']:
                pc.set_facecolor(colors[file_i])
                pc.set_edgecolor('black')
                pc.set_alpha(1)
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Velocities (pixels/frame)")
ax.set_title('Velocity Distributions Per Node')
ax.legend(loc='upper left', ncols=3)
plt.show() 

#Visualize outlier results (Velocity)
fig,ax = plt.subplots(layout="constrained")
for file_i in range(len(hdf5_file_paths)):
    node_sums = np.sum(velocities_mask[file_i], axis=0)
    print(np.shape(node_sums))
    for node_i in range(locations[0].shape[1]):
        x_pos = x[node_i] + width * file_i
        if node_i==0:
            ax.bar(x_pos, node_sums[node_i], width, label=file_labels[file_i],color=colors[file_i])
        else:
            ax.bar(x_pos, node_sums[node_i], width, color=colors[file_i]) #Error

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Num Outliers")
ax.set_title('Num Outliers Per Node (Velocity)')
ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show() 

#Visualize results (Acceleration)
fig,ax = plt.subplots(layout="constrained")
for file_i in range(len(hdf5_file_paths)):
    node_sums = np.sum(accelerations_mask[file_i], axis=0)
    print(np.shape(node_sums))
    for node_i in range(locations[0].shape[1]):
        x_pos = x[node_i] + width * file_i
        if node_i==0:
            ax.bar(x_pos, node_sums[node_i], width, label=file_labels[file_i],color=colors[file_i])
        else:
            ax.bar(x_pos, node_sums[node_i], width, color=colors[file_i]) #Error

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Num Outliers")
ax.set_title('Num Outliers Per Node (Acceleration)')
ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show()      
    
 
#%% Remove Nose Tracking Outliers Based on Head Angle
# This step identifies and removes frames where the nose position is physiologically
# unlikely based on its angle relative to the head and neck.
node_names = processed_dict_list[0]["node_names"]

head_idx = node_names.index("implant")
nose_idx = node_names.index("nose")
neck_idx = node_names.index("neck")

# Calculate the angle between (head-to-nose) and (head-to-neck) vectors for each frame
head_nose_angle = [np.zeros(locations[file_idx].shape[0]) for file_idx in range(len(hdf5_file_paths))]
for f_idx in range(len(hdf5_file_paths)):
    for frame in range(locations[f_idx].shape[0]):
        head_pt = locations[f_idx][frame, head_idx, :]
        nose_pt = locations[f_idx][frame, nose_idx, :]
        neck_pt = locations[f_idx][frame, neck_idx, :]
        
        # Check for NaN values before calculation
        if np.any(np.isnan([head_pt, nose_pt, neck_pt])):
            head_nose_angle[f_idx][frame] = np.nan
            continue
        
        # Calculate vectors
        head_nose_vec = nose_pt - head_pt
        head_neck_vec = neck_pt - head_pt
        
        # Calculate dot product and norms
        dot_product = np.dot(head_nose_vec, head_neck_vec)
        norm_hno = np.linalg.norm(head_nose_vec)
        norm_hne = np.linalg.norm(head_neck_vec)
        
        # Avoid division by zero
        if norm_hno * norm_hne == 0:
            head_nose_angle[f_idx][frame] = np.nan
        else:
            # Calculate angle in radians
            cos_angle = dot_product / (norm_hno * norm_hne)
            head_nose_angle[f_idx][frame] = math.acos(np.clip(cos_angle, -1.0, 1.0))

# --- Use interquartile range (IQR) to define angle thresholds --- MAKE SURE THIS USES ALL FILES
nose_outlier_mask_all = [0 for i in range(len(hdf5_file_paths))]
for f_idx in range(len(hdf5_file_paths)):
    q75, q25 = np.nanpercentile(head_nose_angle[f_idx], [75, 25])
    iqr = q75 - q25
    # An outlier is defined as being 3 * IQR away from the quartiles
    low_bound = q25 - 3 * iqr
    high_bound = q75 + 3 * iqr
    
    # --- Set nose locations to NaN if the angle is an outlier ---
    nose_outlier_mask = (head_nose_angle[f_idx] < low_bound) | (head_nose_angle[f_idx] > high_bound)
    nose_outlier_mask_all[f_idx] = nose_outlier_mask
    num_nose_nodes_removed = np.sum(nose_outlier_mask)
    print(num_nose_nodes_removed)
    #locations[nose_outlier_mask, nose_idx, :] = np.nan

fig,ax = plt.subplots() 
num_nose_nodes_removed_all = [np.sum(nose_outlier_mask_all[i]) for i in range(len(nose_outlier_mask_all))]

x = np.arange(len(hdf5_file_paths))
ax.bar(x, height=num_nose_nodes_removed, width=1, label=file_labels, color=colors)
ax.legend(loc='upper left')
ax.set_ylabel("Num Nose Outliers (frames)")
#%% Plot new total number of outliers per node

#Apply velocity mask
for v in range(len(velocities_mask)):
    print(velocities_mask[v].astype(bool))
    locations[v][1:][velocities_mask[v].astype(bool)] = np.nan
#Apply angle mask
for d in range(len(nose_outlier_mask_all)):
    locations[d][nose_outlier_mask_all[d]] = np.nan
    
#Plot new total outliers
#https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
num_nan_per_node = np.zeros((len(hdf5_file_paths), locations[0].shape[1],))
for file_idx in range(len(hdf5_file_paths)):
    for node_idx in range(locations[file_idx].shape[1]):
        node_loc = locations[file_idx][:,node_idx, 0]
        nan_count_sum = np.sum(np.isnan(node_loc))
        num_nan_per_node[file_idx, node_idx] = nan_count_sum

#Easy: units_btw_nodes = (num_files + 1) * width
units_btw_nodes = 1
x = np.arange(num_nan_per_node.shape[1]) * units_btw_nodes # the label locations
width = 0.25  # the width of the bars

fig,ax = plt.subplots(layout="constrained")
for node_i in range(num_nan_per_node.shape[1]):
    for file_i in range(num_nan_per_node.shape[0]):
        x_pos = x[node_i] + width * file_i
        if node_i==0:
            ax.bar(x_pos, num_nan_per_node[file_i,node_i], width, label=file_labels[file_i],color=colors[file_i])
        else:
            ax.bar(x_pos, num_nan_per_node[file_i,node_i], width, color=colors[file_i])

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel("Num NaN")
ax.set_title('Num NaN by Node (Post-Processing)')
ax.set_xticks(x + width, processed_dict_list[0]["node_names"])
ax.legend(loc='upper left', ncols=3)
plt.show()    