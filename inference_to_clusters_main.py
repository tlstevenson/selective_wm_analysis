#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 14:55:33 2025

@author: alex
#https://365datascience.com/tutorials/python-tutorials/pca-k-means/
"""

#%% Imports

import inference_to_clusters_lib as itcl
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import file_select_ui as fsu
import pandas as pd
import pickle


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

#Finds the hdf5 file based on previous selection. 
#If it doesn't exist, it asks for a new location
json_path = "file_loc_settings.json"
file_path = None
if os.path.exists(json_path) and input("Do you want to select a new location (t/f): ") != "t":
    with open(json_path, 'r') as file:
        data = json.load(file)
        file_path = data["json_path"]
else:
    with open("file_loc_settings.json", "w") as file:
        new_path = fsu.GetFile()
        data = {"json_path": new_path}
        json.dump(data, file)
        file_path = new_path
#filename = r"/Users/alex/Downloads/ResultInference.hdf5"
# Process the hdf5 to a dictionary 
print(file_path)
new_dict = itcl.process_hdf5_data(file_path)
pickle.dump(new_dict["edge_inds"], open("Skeleton_Edges.bin","wb"))


"""
port_pos_list = np.array([[969, 1068.5, 974],
                          [330.5, 484.5, 641]])
overallAngles = np.zeros((port_pos_list.shape[1], len(new_dict["locations"])))

print(itcl.AngleToPorts(new_dict["locations"][0], new_dict, port_pos_list))


for frame_idx in range(len(new_dict["locations"])):
    itcl.PlotSkeleton(new_dict["locations"][frame_idx], new_dict)
    itcl.PlotPorts(port_pos_list)
    angles = itcl.AngleToPorts(new_dict["locations"][frame_idx], new_dict, port_pos_list)
    offset = [20,0]
    itcl.PlotAnglesToPorts(angles, port_pos_list, offset)
    plt.xlim((0,1280))
    plt.ylim(960,0)

    plt.show()"""
    

local_pos = np.zeros(np.shape(new_dict["locations"]))
print(np.shape(local_pos))
for frame_idx in range(len(new_dict["locations"])):
    local_pos[frame_idx,:] = itcl.NodePositionsLocal(new_dict["locations"][frame_idx], new_dict)
    
print(np.shape(local_pos))
#itcl.PlotLocalPosNode(new_dict, "nose", local_pos)

def LocationToDataframe(locations_data, node_names):
    df = pd.DataFrame()
    for name in node_names:
        df[f"{name}_x"] = []
        df[f"{name}_y"] = []
    #print(df)
    for frame_idx in range(len(locations_data)):
        row_data = np.zeros((df.shape[1],))
        for n in range(len(locations_data[frame_idx])):
            #print(locations_data[frame_idx][n][0])
            #print(locations_data[frame_idx,n,0,:])
            row_data[2*n] = float(locations_data[frame_idx,n,0,0])
            row_data[2*n+1] = float(locations_data[frame_idx,n,1,0])
        df.loc[len(df)] = row_data
    return df

def ColFill(dataframe, column_name):
    time_series = dataframe[column_name]
    #Currently assume that the nose dosn't move while undetected
    print(np.shape(time_series))
    last_value = None
    replacement_count = 0
    for t_idx in range(len(time_series)):
        if dataframe[column_name][t_idx] == None or dataframe[column_name][t_idx] == np.nan:
            replacement_count = replacement_count + 1
            dataframe[column_name] = last_value
        else:
            last_value = dataframe[column_name][t_idx]
    print(f"Filled {replacement_count} locations")
        
                

def PCA_analysis(local_pos, processed_dict, num_test = 10):
    #Need an array such that the index is the 10,000 points
    #However, it should be nose x, nose y, body x, body y, etc. not 3d
    #Normalize all the positions
    raw_df = LocationToDataframe(local_pos, processed_dict["node_names"])
    no_nan = raw_df.dropna()
    #ColFill(raw_df, "nose_x")
    #ColFill(raw_df, "nose_y")
    scaler = StandardScaler()
    segmentation_std = scaler.fit_transform(no_nan)
    
    pca = PCA()
    pca.fit(segmentation_std)
    
    #Check how many components are needed to account for 80% of variability
    plt.plot(range(22), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title("Explained Variance By Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.show()
    
    #Results showed four or five pcs work best
    pca = PCA(n_components=4)
    pca.fit(segmentation_std)
    scores_pca = pca.transform(segmentation_std)
    #Check which number of clusters works best
    wcss = []
    for i in range(1,num_test):
        kmeans_pca = KMeans(n_clusters=i, init = 'k-means++', random_state = 42)
        kmeans_pca.fit(scores_pca)
        wcss.append(kmeans_pca.inertia_)
        
    plt.plot(range(1,num_test), wcss, marker='o')
    plt.xlabel("Number of Clusters")
    plt.ylabel("Within Cluster Sum of Squares")
    plt.title("K-means with PCA Clustering")
    plt.show()
    
    #Three clusters works best
    kmeans_pca = KMeans(n_clusters=3, init = 'k-means++', random_state = 42)
    kmeans_pca.fit(scores_pca)
    
    print(np.shape(scores_pca))
    print(scores_pca[1:5,:])
    
    #Add new data to dataframe
    df_scores = pd.DataFrame(scores_pca, columns=["component 1", "component 2", "component 3", "component 4"])
    print(df_scores.head())
    df_segmentation_std_kmeans = pd.concat([no_nan, df_scores], axis=1)
    #Rename the columns
    print(df_segmentation_std_kmeans.head())
    df_clusters = pd.DataFrame(kmeans_pca.labels_, columns=["cluster"])
    print(df_clusters.head())
    df_segmentation_std_kmeans = pd.concat([df_segmentation_std_kmeans, df_clusters], axis=1)
    #df_segmentation_std_kmeans['cluster'] = kmeans_pca.labels_
    df_segmentation_std_kmeans.head()
    return df_segmentation_std_kmeans

new_dict["locations"] = local_pos
for frame in new_dict["locations"]:
    itcl.PlotSkeleton(frame, new_dict)
    plt.show()
#df_final_kmeans = PCA_analysis(local_pos, new_dict)
#pickle.dump(df_final_kmeans, open("DataframeKMeans.bin", "wb"))