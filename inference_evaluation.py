# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:25:58 2026

@author: cns-th-lab
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


# --- Configuration ---
VIDEO_PATH = ''#fsui.GetFile("Please select a video file")
INFERENCE_PATH = fsui.GetFile("Please select the corresponding .h5 path")  # Or .csv
OUTPUT_WINDOW = "Keypoint Inspector"
FPS = 30 # Defined conversion rate

if os.path.splitext(os.path.basename(VIDEO_PATH))[0] != os.path.splitext(os.path.basename(INFERENCE_PATH))[0]:
    raise Warning("Session id of video and data do not match by current naming conventions.")
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
    mask = scores < threshold
    positions[:, :, 0,:][mask] = np.nan
    positions[:, :, 1,:][mask] = np.nan
    return positions

coords = ThresholdedPositions(coords, scores, 0.3)

#%%% Get interpolated data
def InterpolateCoordsCubic(coords, limit_arg):
    deconstr_dict = {}
    
    #Turn each x and y series into a column
    for i in range(np.shape(coords)[1]):
        deconstr_dict[f"{i}_x"] = coords[:, i, 0, 0] 
        deconstr_dict[f"{i}_y"] = coords[:, i, 1, 0]
    my_df = pd.DataFrame(deconstr_dict)
    my_df.interpolate(method="polynomial", order=3, limit=limit_arg, axis=0)
    
    #Reverse of the deconstruction indexing by column
    for i in range(np.shape(coords)[1]):
        coords[:,i,0,0] = deconstr_dict[f"{i}_x"]
        coords[:,i,1,0] = deconstr_dict[f"{i}_y"]
    return coords

coords = InterpolateCoordsCubic(coords, 15)
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
fig, ax = plt.subplots(len(names)//2, 2)
for i in range(len(names)):
    nan_out, nan_starts, _ = measure_nan_gaps(pd.Series(coords[:,i, 0, 0]))
    print(names[i])
    ax[i//2, i%2].bar(np.arange(0, np.shape(coords)[0])[nan_starts],nan_out[nan_starts])
    #ax[i//2, i%2].set_title(names[i])
plt.show()

#%%% NaN heatplot across frames
fig2, ax2 = plt.subplots(len(names)//2, 2)
for i in range(len(names)):
    print(names[i])
    is_nan = np.isnan(coords[:,i, 0, 0])
    nan_x_points = np.arange(0, np.shape(coords)[0])[is_nan]
    ax2[i//2, i%2].bar(nan_x_points,np.ones(len(nan_x_points)))
    ax2[i//2, i%2].set_title(names[i])
plt.show()

#%%% Proportion of NaN by node
prop_nan = []
for i in range(len(names)):
    nan_out, nan_starts, _ = measure_nan_gaps(pd.Series(coords[:,i, 0, 0]))
    prop_nan_i = np.sum(nan_out) / len(nan_out)
    prop_nan.append(prop_nan_i)
print(prop_nan)
plt.bar(names, prop_nan)
plt.show()

#%%% Distribution of gap lengths by node
fig3, ax3 = plt.subplots(len(names)//2, 2)
for i in range(len(names)):
    print(names[i])
    nan_out, nan_starts, nan_gap_sizes = measure_nan_gaps(pd.Series(coords[:,i, 0, 0]))
    ax3[i//2, i%2].hist(nan_gap_sizes)
    #ax[i//2, i%2].set_title(names[i])
plt.show()
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
    bin_prop_vel = np.sum(~np.isnan(bin_vels) & (bin_vels > vel_threshold), axis=0) / np.sum(~np.isnan(bin_vels))
    
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