# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:30:23 2026

@author: cns-th-lab
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import subprocess
import itertools

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
#%%Define file paths and create analysis h5s
#Create a dataframe
# Define file paths and create analysis h5s directly into a DataFrame
data = [
    # Model 198_402
    ["198_402", 198, "in", r"C:/Users/cns-th-lab/SLEAP_Labels_198_402/198/Videos/198.2025-07-28.mov_0001.proj.slp", ""],
    ["198_402", 198, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\198\Videos\198.2025-08-23.mov_0014.proj.slp", ""],
    ["198_402", 199, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\199\Videos\199.2025-07-28.mov_0001.proj.slp", ""],
    ["198_402", 199, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\199\Videos\199.2025-08-23.mov_0014.proj.slp", ""],
    ["198_402", 237, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\237\Videos\237.2026-03-31.mov_0001.proj.slp", ""],
    ["198_402", 237, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\237\Videos\237.2026-04-03.mov_0004.proj.slp", ""],
    ["198_402", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\238\Videos\238.2026-03-31.mov_0001.proj.slp", ""],
    ["198_402", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_402\238\Videos\238.2026-04-03.mov_0004.proj.slp", ""],

    # Model 198_237x_402
    ["198_237x_402", 198, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\198\Videos\198.2025-07-28.mov_0001.proj.slp", ""],
    ["198_237x_402", 198, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\198\Videos\198.2025-08-23.mov_0014.proj.slp", ""],
    ["198_237x_402", 199, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\199\Videos\199.2025-07-28.mov_0001.proj.slp", ""],
    ["198_237x_402", 199, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\199\Videos\199.2025-08-23.mov_0014.proj.slp", ""],
    ["198_237x_402", 237, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\237\Videos\237.2026-03-31.mov_0001.proj.slp", ""],
    ["198_237x_402", 237, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\237\Videos\237.2026-04-03.mov_0004.proj.slp", ""],
    ["198_237x_402", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\238\Videos\238.2026-03-31.mov_0001.proj.slp", ""],
    ["198_237x_402", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402\238\Videos\238.2026-04-03.mov_0004.proj.slp", ""],

    # Model 198_199x_237x_402
    ["198_199x_237x_402", 198, "in", r"C:/Users/cns-th-lab/SLEAP_Labels_198_199x_237x_402/198/Videos/198.2025-07-28.mov_0001.proj.slp", ""],
    ["198_199x_237x_402", 198, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\198\Videos\198.2025-08-23.mov_0014.proj.slp", ""],
    ["198_199x_237x_402", 199, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\199\Videos\199.2025-07-28.mov_0001.proj.slp", ""],
    ["198_199x_237x_402", 199, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\199\Videos\199.2025-08-23.mov_0014.proj.slp", ""],
    ["198_199x_237x_402", 237, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\237\Videos\237.2026-03-31.mov_0001.proj.slp", ""],
    ["198_199x_237x_402", 237, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\237\Videos\237.2026-04-03.mov_0004.proj.slp", ""],
    ["198_199x_237x_402", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\238\Videos\238.2026-03-31.mov_0001.proj.slp", ""],
    ["198_199x_237x_402", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402\238\Videos\238.2026-04-03.mov_0004.proj.slp", ""],
    
    # Model All
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 198, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\198\Videos\198.2025-07-28.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 198, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\198\Videos\198.2025-08-23.mov_0014.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 199, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\199\Videos\199.2025-07-28.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 199, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\199\Videos\199.2025-08-23.mov_0014.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 237, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\237\Videos\237.2026-03-31.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 237, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\237\Videos\237.2026-04-03.mov_0004.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 238, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\238\Videos\238.2026-03-31.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 238, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\238\Videos\238.2026-04-03.mov_0004.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 274, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\274\Videos\274.2025-09-25.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 274, "out", r"C:/Users/cns-th-lab/SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x/274/Videos/274.2025-10-24.mov_0015.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 400, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\400\Videos\400.2025-09-25.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 400, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\400\Videos\400.2025-10-24.mov_0015.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 402, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\402\Videos\402.2025-09-25.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 402, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\402\Videos\402.2025-10-24.mov_0015.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 424, "in", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\424\Videos\424.2026-03-31.mov_0001.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 424, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\424\Videos\424.2026-04-03.mov_0004.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 483, "in", r"C:/Users/cns-th-lab/SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x/483/Videos/483.2026-04-01.mov_0002.proj.slp", ""],
    ["198_199x_237x_238x_274x_400x_402x_424x_483x", 483, "out", r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x\483\Videos\483.2026-04-03.mov_0004.proj.slp", ""]
]

df = pd.DataFrame(data, columns=["model", "rat", "incl_train", "slp_path", "h5_path"])
# Replace the string and assign it to the h5_path column
df["h5_path"] = df["slp_path"].str.replace(".slp", "_analysis.h5", regex=False)
df.to_csv("sleap_model_paths.csv", index=False)

#%% Convert h5 files
# Load permanently restructured data
df = pd.read_csv("sleap_model_paths.csv")
num_files = 0

for row in df.itertuples():
    # Access elements using dot notation
    if not os.path.exists(row.slp_path):
        print(f"File not found: {row.slp_path}. Please check the path.")
    elif os.path.exists(row.h5_path):
        print(f"File already converted: {row.h5_path}")
    else:
        print(f"Converting file {row.slp_path} to analysis h5")
        slp_to_analysis_h5(row.slp_path, row.h5_path)
#%% Add Score and NaN columns
import h5py
import numpy as np
import pandas as pd
import os
node_names = ""

def get_node_names(h5_path):
    with h5py.File(h5_path, "r") as f:
        # Decode node names
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
    return node_names

def get_node_metrics(h5_path, use_score_threshold=False, threshold=0.2):
    """
    Reads an h5 file and returns a tuple of two dictionaries:
    (nan_proportions, valid_scores) for each node.
    """
    global node_names
    # Return two empty dictionaries if the path is invalid
    if pd.isna(h5_path) or not h5_path or not os.path.exists(h5_path):
        return {}, {}

    nan_props = {}
    valid_scores_dict = {}
    
    with h5py.File(h5_path, "r") as f:
        # Decode node names
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        
        # 1. Get Prediction Scores 
        # Raw shape: (tracks, nodes, frames) -> Transposed: (frames, nodes, tracks)
        scores = np.transpose(f['point_scores'][:], (2, 1, 0)) 
        
        # 2. Get Coordinates
        # Raw shape: (tracks, nodes, 2, frames) -> Transposed: (frames, nodes, 2, tracks)
        tracks_coords = np.transpose(f['tracks'][:])
        
        for i, node in enumerate(node_names):
            # Extract X-coordinates and scores for this specific node
            node_x_coords = tracks_coords[:, i, 0, :]
            node_scores = scores[:, i, :]
            
            # Base condition: Is it naturally a NaN in the output?
            is_missing = np.isnan(node_x_coords) | np.isnan(node_scores)
            
            # Optional condition: Add frames where the model's confidence is too low
            if use_score_threshold:
                is_missing = is_missing | (node_scores < threshold)
            
            # Calculate proportion and extract valid scores
            nan_props[node] = is_missing.mean()
            valid_scores_dict[node] = list(node_scores[~is_missing]) #Must be list not numpy list for concatenation later
            
    return nan_props, valid_scores_dict

# 1. Apply the extraction function to the h5_path column
# This returns a Series where each element is a tuple: (nan_props_dict, valid_scores_dict)
metrics_series = df["h5_path"].apply(get_node_metrics)

# 2. Unpack the Series of tuples into two separate lists of dictionaries
nan_props_list, scores_list = zip(*metrics_series.tolist())

# 3. Convert the lists of dictionaries into two separate DataFrames
# Maintaining the original DataFrame's index ensures they align perfectly with your original df
nan_props_df = pd.DataFrame(list(nan_props_list), index=df.index)
scores_df = pd.DataFrame(list(scores_list), index=df.index)

# 1. Apply the extraction function to the h5_path column
# This returns a Series where each element is a tuple: (nan_props_dict, valid_scores_dict)
metrics_series = df["h5_path"].apply(get_node_metrics)

# 2. Unpack the Series of tuples into two separate lists of dictionaries
nan_props_list, scores_list = zip(*metrics_series.tolist())

# 3. Convert the lists of dictionaries into two separate DataFrames
# Maintaining the original DataFrame's index ensures they align perfectly with your original df
nan_props_df = pd.DataFrame(list(nan_props_list), index=df.index)
scores_df = pd.DataFrame(list(scores_list), index=df.index)

# 4. Extract the grouping columns from the original DataFrame
# We use a list comprehension just in case one of the columns is missing, 
# preventing a KeyError while grabbing what's available.
grouping_cols = ["model", "rat", "incl_train"]
meta_df = df[[col for col in grouping_cols if col in df.columns]]

# 5. Concatenate the metadata columns with the new metrics
nan_props_final = pd.concat([meta_df, nan_props_df], axis=1)
scores_final = pd.concat([meta_df, scores_df], axis=1)

#%% View the results
print(df.head())
print(df.columns)

#%% Group by one variable and average rest (by metric)

#Drop all the other grouping columns to allow mathematical operations on columns
filtered_df_nan = nan_props_final
filtered_df_scores = scores_final
group_col = "model"
for g_c in grouping_cols:
    if g_c != group_col:
        filtered_df_nan = filtered_df_nan.drop(columns=g_c)
        filtered_df_scores = filtered_df_scores.drop(columns=g_c)

#Group scores by grouping variable
nan_df_by_group = filtered_df_nan.groupby(group_col)
score_df_by_group = filtered_df_scores.groupby(group_col)

#Flatten scores by group
score_group_distributions = {}
for group_name, group_df in score_df_by_group:
    score_group_distributions[group_name] = {
        node: list(itertools.chain.from_iterable(group_df[node])) 
        for node in node_names
    }

#%% Plot group results (NaNs)
fig, ax = plt.subplots(1,4)
plot_num = 0
fig.suptitle("Proportion of NaNs by Node Between Models")
for group_name, group_df in nan_df_by_group:
    ax[plot_num].set_title(group_name, fontsize=4)
    group_df.boxplot(ax = ax[plot_num], rot=45, fontsize=4)
    ax[plot_num].set_ylim((0,1))
    plot_num = plot_num + 1


#%% Plot group results (scores)
fig_score, ax_score = plt.subplots() #TODO: Remove hardcode 4 for model count
fig.suptitle("Score Distribution per Node By Model")
plot_num_score = 0
group_names = list(score_group_distributions.keys())
ax_score.tick_params(axis='x', rotation=90,labelsize=4)

#Create a colormap
cmap = plt.colormaps['viridis'] 
color_indices = np.linspace(0, 1, len(group_names))
unique_colors = cmap(color_indices)

for i in range(len(group_names)):
    for j in range(len(node_names)):
        node_score_data = score_group_distributions[group_names[i]][node_names[j]]
# %%
        if(j == 0):

            b = ax_score.boxplot(node_score_data, positions=[i+j*len(group_names)], label=group_names[i], tick_labels=[node_names[j]], patch_artist=True)
            plt.setp(b['boxes'], color=unique_colors[i])
        else:
            b = ax_score.boxplot(node_score_data, positions=[i+j*len(group_names)], label="_nolegend_", tick_labels=[node_names[j]], patch_artist=True)
            plt.setp(b['boxes'], color=unique_colors[i])
        
ax_score.legend()
plt.show()
