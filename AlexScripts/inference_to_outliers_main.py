#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 26 12:13:20 2025

@author: alex

Takes an hdf5 file and calculates velocity and acceleration of all nodes
Uses the acceleration to set a threshold for outliers
Clusters the outliers by local position or by jumpy node
Plots some of the skeletons of each outlier group
"""
#import statements
import init
from sys_neuro_tools import sleap_utils
from sys_neuro_tools import math_utils
from pyutils import file_select_ui as fsui
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import pandas as pd
import random

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#import cv2

#Initial file reading
hdf5_file_path  = fsui.GetFile("Select an Analysis File")
processed_dict  = sleap_utils.process_hdf5_data(hdf5_file_path)
print(f'Processed dict locations has shape {np.shape(processed_dict["locations"])}')
locations=None
min_frame=0
max_frame=None

#Remove the dimension for multiple animals
try:
    locations = np.squeeze(processed_dict["locations"][min_frame:max_frame])
except:
    locations = np.squeeze(processed_dict["locations"])
    
#Exclude leap nose values
#Calculate average angle btw head and neck and nose and set quartile threshold

head_nose_angle = np.zeros((locations.shape[0]))
head_idx = processed_dict["node_names"].index("implant")
nose_idx = processed_dict["node_names"].index("nose")
neck_idx = processed_dict["node_names"].index("neck")
for frame in range(locations.shape[0]):
    if not np.isnan(locations[frame,nose_idx,0]) and not np.isnan(locations[frame,nose_idx,1]):
        head_nose = locations[frame,nose_idx,:] - locations[frame,head_idx,:]
        head_neck = locations[frame,neck_idx,:] - locations[frame,head_idx,:]
        dot = np.dot(head_nose, head_neck)
        #print(f"dot: {dot}")
        norm_hno = np.linalg.norm(head_nose)
        #print(f"hno: {norm_hno}")
        norm_hne = np.linalg.norm(head_neck)
        #print(f"hne: {norm_hne}")
        ans = math.acos(dot / (norm_hno * norm_hne))
        #print(f"ans: {ans}")
        head_nose_angle[frame] = ans
    else:
        head_nose_angle[frame]=np.nan
        
plt.hist(head_nose_angle)
print(np.nanmean(head_nose_angle))
print(np.nanstd(head_nose_angle))

q75, q50, q25 = np.nanpercentile(head_nose_angle, [75,50,25])
iqr = q75 - q25
low_bound = q25 - 3 * iqr
high_bound = q75 + 3 * iqr
print(f"Median statistics: {(q25, q50, q75)}")
print(f"Bounds: {(low_bound, high_bound)}")

nose_outlier_count = 0
for frame in range(locations.shape[0]):
    if not np.isnan(locations[frame,nose_idx,0]) and not np.isnan(locations[frame,nose_idx,1]):
        if head_nose_angle[frame] < low_bound or head_nose_angle[frame] > high_bound:
            locations[frame,nose_idx,0] = np.nan
            locations[frame,nose_idx,1] = np.nan 
            nose_outlier_count = nose_outlier_count + 1
#Exclude remaining velocity outliers

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
#Automatically sets a threshold where 5 percent or less are outliers
accel_thresh = None

for accel_thresh_temp in range(1,101, 5):
    print(f"Accel thresh: {accel_thresh_temp}")
    mask = accel > accel_thresh_temp
    mask_s = accel_s > accel_thresh_temp
    a_idxs = np.argwhere(np.sum(mask, axis=0)) #Get frame indices where there is at least one outlier
    a_idxs_s = np.argwhere(np.sum(mask_s, axis=0))
    #a0 comes from v0 and v1 which come from x0, x1, and x
    #If a0 is outlier then plot 0,1,2 for position
    print(f"Number Outliers: {len(a_idxs)}")
    print(f"Number Outliers Smoothed: {len(a_idxs_s)}")
    if accel_thresh == None and len(a_idxs) < .01 * mask.shape[1]:
        accel_thresh = accel_thresh_temp
        break

#Visualizes outliers if a threshold was found
if accel_thresh != None:
    mask = accel > accel_thresh
    a_idxs = np.argwhere(np.sum(mask, axis=0)) #Get frame indices where there is at least one outlier
    
    name_classify = False
    cluster_classify = True
    
    if name_classify:
        #Create a figure for each node
        for node_idx in range(mask.shape[0]):
            #Gets outliers and skips node if there aren't any
            outlier_frames= [i for i in range(len(mask[node_idx])) if mask[node_idx][i]]
            #outlier_frames=np.argwhere(mask[node_idx])
            if len(outlier_frames)==0:
                print(f'No outliers for {processed_dict["node_names"][node_idx]}')
                continue
            
            #Plots max five random outliers
            num_plots = min(6, len(outlier_frames))
            node_fig, node_ax = plt.subplots(nrows=math.ceil(num_plots/2), ncols=2)
            node_fig.suptitle(f'{processed_dict["node_names"][node_idx]} Outliers')
            chosen_frames = random.sample(outlier_frames, num_plots)
            plot_idx = 0
            for frame in chosen_frames:
                sleap_utils.PlotSkeleton(locations[frame], processed_dict, ax=node_ax[plot_idx//2,plot_idx%2])
                plot_idx = plot_idx + 1
            plt.show()
    if cluster_classify:
        #Find all the frames where any node is an outlier
        mask = accel > accel_thresh_temp
        outlier_frames = np.argwhere(np.sum(mask, axis=0)) #Get frame indices where there is at least one outlier
        print(np.shape(locations[outlier_frames[0]]))
        local_pos = np.array([sleap_utils.NodePositionsLocal(np.squeeze(locations[frame_idx]), processed_dict) for frame_idx in outlier_frames])
        
        #Need an array such that the index is the 10,000 points
        #However, it should be nose x, nose y, body x, body y, etc. not 3d
        #Normalize all the positions
        raw_df = sleap_utils.LocationToDataframe(local_pos, processed_dict["node_names"])
        no_nan = raw_df.dropna()
        scaler = StandardScaler()
        segmentation_std = scaler.fit_transform(no_nan)
        
        #Fit PCA on local position df
        pca = PCA()
        pca.fit(segmentation_std)
        
        #Check how many components are needed to account for 80% of variability
        #Use that number to train the final pca model
        num_pcs = np.argmax(pca.explained_variance_ratio_.cumsum() > .80)
        pca = PCA(n_components=num_pcs)
        pca.fit(segmentation_std)
        scores_pca = pca.transform(segmentation_std)
        #Check which number of clusters works best
        wcss = []
        num_test = 20
        for i in range(1,num_test):
            kmeans_pca = KMeans(n_clusters=i, init = 'k-means++', random_state = 42)
            kmeans_pca.fit(scores_pca)
            wcss.append(kmeans_pca.inertia_)
        
        fig_k, ax_k = plt.subplots()
        ax_k.plot(range(1,num_test), wcss, marker='o')
        ax_k.set_xlabel("Number of Clusters")
        ax_k.set_ylabel("Within Cluster Sum of Squares")
        ax_k.set_title("K-means with PCA Clustering")
        plt.show()
        
        num_clusters = int(input("Best Num Clusters: "))
        kmeans_pca = KMeans(n_clusters=num_clusters, init = 'k-means++', random_state = 42)
        kmeans_pca.fit(scores_pca)
        
        #Add new data to dataframe
        df_scores = pd.DataFrame(scores_pca, columns=[f"component {i}" for i in range(num_pcs)])
        print(df_scores.head())
        df_segmentation_std_kmeans = pd.concat([no_nan, df_scores], axis=1)
        #Rename the columns
        print(df_segmentation_std_kmeans.head())
        df_clusters = pd.DataFrame(kmeans_pca.labels_, columns=["cluster"])
        print(df_clusters.head())
        df_segmentation_std_kmeans = pd.concat([df_segmentation_std_kmeans, df_clusters], axis=1)
        
        vid_path = fsui.GetFile("Select Video File")
        cap = cv2.VideaCapture(vid_path)
        
        #Plot a random sample of six frames from each cluster
        for i in range(num_clusters):
            #Get all instances of a cluster 
            cluster = df_segmentation_std_kmeans[df_segmentation_std_kmeans["cluster"] == i]
            #Sample six of them
            num_plots = min(6, len(cluster))
            print(cluster.index)
            sample_idx = random.sample(cluster.index.tolist(), num_plots)
            print(sample_idx)
            #Plot each on a seperate subplot
            group_fig, group_ax = plt.subplots(nrows=math.ceil(num_plots/2), ncols=2)
            group_fig.suptitle(f'Cluster {i} Outliers')
            plot_idx = 0
            for entry in sample_idx:
                sleap_utils.PlotSkeleton(locations[entry], processed_dict, ax=group_ax[plot_idx//2,plot_idx%2])
                plot_idx = plot_idx + 1
                
                #Also show video
                cap.set(cv2.CAP_PROP_POS_FRAMES, entry)
                ret, frame = cap.read()
                
                if ret:
                    cv2.imshow("Specific Frame", frame)
                else:
                    print(f"Error: Could not read frame {entry}")
            plt.show()
        cap.release()
else:
    print("Could not find a reasonable threshold in supplied range.")
    
#NOTE: Must be run with inline plots (Plots all outliers)
"""for a_idx in a_idxs:
    fig4, ax4 = plt.subplots()
    #print(locations[a_idx])
    sleap_utils.PlotSkeleton(np.squeeze(locations[a_idx]), processed_dict, skeleton_color="green", nodes_mark=mask[:,a_idx],ax=ax4)
    sleap_utils.PlotSkeleton(np.squeeze(locations[a_idx+1]), processed_dict, skeleton_color="blue", nodes_mark=mask[:,a_idx],ax=ax4)
    sleap_utils.PlotSkeleton(np.squeeze(locations[a_idx+2]), processed_dict, skeleton_color="purple", nodes_mark=mask[:,a_idx],ax=ax4)
    ax4.set_title(f"{a_idx},{a_idx+1},{a_idx+2}, Time: {a_idx/60} seconds")
    #plt.show()"""