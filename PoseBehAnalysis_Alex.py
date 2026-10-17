# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 10:59:20 2026

@author: cns-th-lab
"""
#%% Import block

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from scipy.interpolate import interp1d

# Scikit-Learn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Behavioral Data Custom Modules
import init
from hankslab_db import db_access
from hankslab_db import (
    tonecatdelayresp_db as wm_db,
    basicRLtasks_db as bandit_db,
)
import neurofileread as nfr
import sleapstats as ss
import supplemental_behavioral_functions as sbf

#%%

#Parameters to set
sess_ids = [116498]
parent_dir = r"C:\Users\cns-th-lab\TannerVidsRenamed"
model_basename = "260731_198_199x_234x_235x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x_occin"
port_model_basename = "260716_port_model"

#Iterate through rat folders for each session id
rat_label_paths, port_label_paths = nfr.rat_port_file_paths_by_sess(sess_ids, parent_dir, model_basename, port_model_basename)
#%%%% Extract project and session specific data
labels_df = nfr.generate_sleap_dataframe_f(rat_label_paths, port_label_paths)
wm_sess_data = nfr.load_wm_behavior(sess_ids)
bandit_beh = nfr.load_bandit_behavior(sess_ids)
timestamps = nfr.get_doric_timestamps_sess(sess_ids, parent_dir)
#%%
print(type(wm_sess_data["sessid"][0]))
print(wm_sess_data["sessid"])
# %%% Extract trial start stop frames in video
trial_starts_plus_last = db_access.get_fp_trial_start_ts(sess_ids) #last is interrputed by end of the recordings
trial_ends_rel_trial_start = sbf.get_trial_end_ts(wm_sess_data)
trial_starts = trial_starts_plus_last #Setup for for loop to remove last
#TODO: Decide whether discarding last is a good thing to do

start_end_int = {}
start_end_int_vid_f = {}

cue_resp_int = {}
cue_resp_int_vid_f = {}

coords_resp_cue_resp = {}
switch_per_sess = {}
#Extract intervals and relevant trial information
for key in sess_ids:
    print(key)
    doric_series = timestamps[key] #Of the video
    session_coords = labels_df[labels_df["sess"]==str(key)]["tracks"].item()
    print("Coords shape")
    print(np.shape(session_coords))

    #Start-stop intervals
    trial_starts[key] = trial_starts[key][:-1] #Removes interrupted last trial
    trial_ends = trial_starts[key] + trial_ends_rel_trial_start[key] #Transforms relative to absolute time
    start_end_int[key] = np.transpose(np.stack((trial_starts[key], trial_ends))) #Creates intervals by stacking
    start_end_coords, intervals_vid_f_sess = ss.pose_in_intervals(doric_series, session_coords, start_end_int[key])
    
    #Response cue to response
    #Grab vectors from session data
    sess_resp_cue = wm_sess_data[wm_sess_data["sessid"]==key]["response_cue_time"]
    sess_resp = wm_sess_data[wm_sess_data["sessid"]==key]["response_time"]
    switch_sess = wm_sess_data[wm_sess_data["sessid"]==key]["switch"]
    
    if len(sess_resp_cue)!=len(sess_resp) or len(sess_resp_cue)!=len(trial_starts[key]) or len(trial_starts[key]) !=len(sess_resp):
        raise Warning(f"The length of response cue ({len(sess_resp_cue)}), responses ({len(sess_resp)}), and trial_starts ({len(trial_starts[key])}) are mismatched.")
    
    #Stack them to make intervals and extract coordinates
    #Remove any NaN ones
    resp_cue_resp_int = np.transpose(np.stack((sess_resp_cue + trial_starts[key], sess_resp + trial_starts[key])))
    valid_mask = ~np.isnan(resp_cue_resp_int).any(axis=1)
    resp_cue_resp_int = resp_cue_resp_int[valid_mask]
    print(resp_cue_resp_int)
    resp_cue_resp_coords, intervals_vid_f_sess = ss.pose_in_intervals(doric_series, session_coords, resp_cue_resp_int)
    
    #Assign current session data to larger dictionary by session id (int)
    cue_resp_int[key] = resp_cue_resp_int
    cue_resp_int_vid_f[key] = intervals_vid_f_sess
    coords_resp_cue_resp[key] = resp_cue_resp_coords
    switch_per_sess[key] = switch_sess[valid_mask]
#%%
def plot_all_trajectories(coords_resp_cue_resp, sess_ids, labels_df):
    """Plots raw 2D trajectories for all points across multiple intervals."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    node_count = np.shape(coords_resp_cue_resp[sess_ids[0]][0])[1] 
    colors = plt.cm.tab20(np.linspace(0, 1, node_count))

    for sess in sess_ids:
        print(f"Plotting trajectories for session: {sess}")
        for i, interval in enumerate(coords_resp_cue_resp[sess]):
            interval = np.squeeze(interval)
            for point_idx in range(node_count):
                x = interval[:, point_idx, 0]
                y = interval[:, point_idx, 1]
                
                # Prevent legend duplication
                label = labels_df[labels_df["sess"] == str(sess)]["node_names"].item()[point_idx] if i == 0 else None
                ax.plot(x, y, color=colors[point_idx], alpha=0.6, linewidth=1.5, label=label)
                ax.scatter(x[0], y[0], color=colors[point_idx], marker='o', s=30, zorder=3)

    ax.set_title("Trajectories of 14 2D Points Across Multiple Intervals")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(title="Point ID", loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def extract_displacements(coords_resp_cue_resp):
    """Squeezes intervals and extracts flat start-to-end displacement features."""
    intervals = [
        np.squeeze(interval) 
        for session_intervals in coords_resp_cue_resp.values() 
        for interval in session_intervals
    ]
    
    displacements = [interval[-1, :, :] - interval[0, :, :] for interval in intervals]
    displacements_array = np.array(displacements)
    
    # Flatten the 14x2 matrix into a 28-element feature vector per interval
    X = displacements_array.reshape(displacements_array.shape[0], -1)
    return intervals, X


def perform_pca_and_plot_variance(X):
    """Fits full PCA to calculate variance, plots scree plot, and returns 2D PCA."""
    pca_full = PCA()
    pca_full.fit(X)
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='--', linewidth=2, markersize=5)
    ax1.axhline(y=0.90, color='red', linestyle=':', label='90% Variance Threshold')
    ax1.set_title("Cumulative Explained Variance by Number of PCs")
    ax1.set_xlabel("Number of Principal Components")
    ax1.set_ylabel("Cumulative Explained Variance Ratio")
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()
    plt.tight_layout()
    plt.show()
    
    pca_2d = PCA(n_components=2)
    X_pca = pca_2d.fit_transform(X)
    return X_pca, pca_2d


def perform_kmeans_and_plot(X_pca, pca_model, k=4):
    """Runs K-Means on 2D PCA data and plots the resulting clusters."""
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(X_pca)
    centroids = kmeans.cluster_centers_

    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', 
                         alpha=0.7, edgecolor='k', s=60)
    centroid_scatter = ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
                                  s=200, linewidths=2, label='Cluster Centroids')

    ax.set_title(f"K-Means Clustering of Interval Displacements\n(PCA Space: {pca_model.explained_variance_ratio_.sum()*100:.1f}% variance explained)")
    ax.set_xlabel(f"Principal Component 1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"Principal Component 2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)")

    handles, _ = scatter.legend_elements(prop="colors")
    legend_labels = [f"Cluster {i}" for i in range(k)]
    handles.append(centroid_scatter)
    legend_labels.append("Centroids")
    
    ax.legend(handles, legend_labels, title="Cluster ID", loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    
    return labels, centroids


def plot_cluster_visualizations(intervals, labels, ports_data=None, k=3, num_points=14):
    """Plots quiver, time-normalized trajectories, and hexbin density heatmaps."""
    if ports_data is not None:
        squeezed_ports = np.squeeze(ports_data) 
        avg_ports = np.nanmean(squeezed_ports, axis=0)
    else:
        avg_ports = None

    clusters = {i: [] for i in range(k)}
    for interval, label in zip(intervals, labels):
        clusters[label].append(interval)

    def plot_ports(ax):
        if avg_ports is not None:
            ax.scatter(avg_ports[:, 0], avg_ports[:, 1], c='limegreen', marker='s', 
                       s=120, edgecolor='black', linewidth=1.5, zorder=5, label='Stationary Ports')

    # Option 1: Mean Displacement Vectors (Quiver Plot)
    fig1, axes1 = plt.subplots(1, k, figsize=(5 * k, 5), sharex=True, sharey=True)
    fig1.suptitle("Option 1: Mean Displacement Vectors per Cluster", fontsize=14)
    for c_id in range(k):
        cluster_data = clusters[c_id]
        if not cluster_data: continue
        starts = np.mean([seq[0, :, :] for seq in cluster_data], axis=0) 
        ends = np.mean([seq[-1, :, :] for seq in cluster_data], axis=0)  
        deltas = ends - starts
        ax = axes1[c_id] if k > 1 else axes1
        ax.quiver(starts[:, 0], starts[:, 1], deltas[:, 0], deltas[:, 1], 
                  angles='xy', scale_units='xy', scale=1, color='b', alpha=0.7)
        ax.scatter(starts[:, 0], starts[:, 1], c='red', s=30, label="Start Points", zorder=3)
        plot_ports(ax)
        ax.set_title(f"Cluster {c_id} (n={len(cluster_data)})")
        if c_id == 0: ax.legend()

    # Option 2: Time-Normalized Mean Trajectories
    fig2, axes2 = plt.subplots(1, k, figsize=(5 * k, 5), sharex=True, sharey=True)
    fig2.suptitle("Option 2: Time-Normalized Average Trajectories", fontsize=14)
    colors = plt.cm.tab20(np.linspace(0, 1, num_points))
    for c_id in range(k):
        cluster_data = clusters[c_id]
        if not cluster_data: continue
        ax = axes2[c_id] if k > 1 else axes2
        normalized_trajectories = []
        for seq in cluster_data:
            T = seq.shape[0]
            original_time = np.linspace(0, 1, T)
            standard_time = np.linspace(0, 1, 100)
            interpolator = interp1d(original_time, seq, axis=0, kind='linear')
            normalized_trajectories.append(interpolator(standard_time))
            
        mean_trajectory = np.mean(normalized_trajectories, axis=0)
        for p in range(num_points):
            ax.plot(mean_trajectory[:, p, 0], mean_trajectory[:, p, 1], color=colors[p], linewidth=2, zorder=2)
            ax.scatter(mean_trajectory[0, p, 0], mean_trajectory[0, p, 1], color='black', s=20, zorder=3)
        plot_ports(ax)
        ax.set_title(f"Cluster {c_id} Average Path")
        if c_id == 0 and avg_ports is not None: 
            ax.plot([], [], color='gray', label='Mean Trajectory') 
            ax.legend()

    # Option 3: Spatial Density Heatmap
    fig3, axes3 = plt.subplots(1, k, figsize=(5 * k, 5), sharex=True, sharey=True)
    fig3.suptitle("Option 3: Point Density Heatmaps", fontsize=14)
    for c_id in range(k):
        cluster_data = clusters[c_id]
        if not cluster_data: continue
        ax = axes3[c_id] if k > 1 else axes3
        all_coords = np.vstack(cluster_data).reshape(-1, 2)
        hb = ax.hexbin(all_coords[:, 0], all_coords[:, 1], gridsize=30, cmap='inferno', mincnt=1, zorder=1)
        plot_ports(ax)
        ax.set_title(f"Cluster {c_id} Density")
        if c_id == 0 and avg_ports is not None: ax.legend()
        fig3.colorbar(hb, ax=ax, label="Coordinate Frequency")

    plt.tight_layout()
    plt.show()


def plot_behavioral_switch_overlay(X_pca, labels, switch_array, centroids):
    """Plots aligned 2D PCA scatter mapping Clusters to Shapes and Switch to Color."""
    color_map = {False: '#1f77b4', True: '#e65c00'} 
    marker_map = {0: 'o', 1: 's', 2: '^', 3: 'D', 4: 'v'}

    fig, ax = plt.subplots(figsize=(10, 8))

    for c_id in np.unique(labels):
        for s_val in [True, False]:
            mask = (labels == c_id) & (switch_array == s_val)
            if np.any(mask):
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color_map[s_val], 
                           marker=marker_map[c_id], s=100, alpha=0.8, edgecolor='w')

    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', 
               s=200, linewidths=2, zorder=5, label='Centroids')

    # Custom Legends
    switch_handles = [
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=color_map[False], markersize=12, label='False (Stay)'),
        mlines.Line2D([], [], color='w', marker='o', markerfacecolor=color_map[True], markersize=12, label='True (Switch)')
    ]
    cluster_handles = [
        mlines.Line2D([], [], color='w', marker=marker_map[c], markerfacecolor='gray', markersize=12, label=f'Cluster {c}')
        for c in np.unique(labels)
    ]
    cluster_handles.append(mlines.Line2D([], [], color='w', marker='X', markeredgecolor='red', markerfacecolor='red', markersize=12, label='Centroids'))

    leg_switch = ax.legend(handles=switch_handles, title="Switch Status (Color)", 
                           loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=11, title_fontsize=12)
    ax.add_artist(leg_switch) 
    ax.legend(handles=cluster_handles, title="Movement Clusters (Shape)", 
              loc='upper left', bbox_to_anchor=(1.02, 0.8), fontsize=11, title_fontsize=12)

    ax.set_title("Behavioral Switch Overlay by Movement Clusters\n(2D PCA Space)", fontsize=14, fontweight='bold')
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def evaluate_svm_dashboard(X_features, y_true, model, X_pca, labels, centroids, title_prefix="SVM"):
    """
    Executes Cross-Validation on the provided model/features and plots a 
    standardized 3-panel dashboard (Predictions, Confusion Matrix, ROC).
    """
    sns.set_theme(style="darkgrid")
    
    # Generate cross-validated predictions and probabilities
    y_pred = cross_val_predict(model, X_features, y_true, cv=10)
    y_prob = cross_val_predict(model, X_features, y_true, cv=10, method='predict_proba')[:, 1]
    
    # Print basic metrics
    cv_50 = StratifiedKFold(n_splits=50, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_features, y_true, cv=cv_50, scoring='accuracy')
    print(f"--- 50-Fold SVM Results ({title_prefix}) ---")
    print(f"Mean Accuracy:      {cv_scores.mean() * 100:.2f}%")
    print(f"Standard Deviation: ±{cv_scores.std() * 100:.2f}%\n")

    # Categorize outcomes
    outcome_labels = []
    for t_val, p_val in zip(y_true, y_pred):
        if t_val == 1 and p_val == 1: outcome_labels.append('True Positive (Correct Switch)')
        elif t_val == 0 and p_val == 0: outcome_labels.append('True Negative (Correct Stay)')
        elif t_val == 1 and p_val == 0: outcome_labels.append('False Negative (Missed Switch)')
        else: outcome_labels.append('False Positive (False Alarm)')

    outcome_palette = {
        'True Positive (Correct Switch)': '#2ca02c',  
        'True Negative (Correct Stay)': '#1f77b4',    
        'False Negative (Missed Switch)': '#d62728',  
        'False Positive (False Alarm)': '#ff7f0e'     
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Plot 1: PCA Prediction Scatterplot ---
    sns.scatterplot(
        x=X_pca[:, 0], y=X_pca[:, 1], hue=outcome_labels, palette=outcome_palette,
        style=labels, markers={0: 'o', 1: 's', 2: '^', 3: 'D'}, s=80, alpha=0.8, ax=axes[0]
    )
    axes[0].scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=150, zorder=5)
    axes[0].set_title(f"{title_prefix} Classifications\n(Projected on PCA Space)", fontsize=14, fontweight='bold')
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:5], legend_labels[:5], loc='best', fontsize=9)

    # --- Plot 2: Confusion Matrix ---
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={"size": 14, "weight": "bold"}, ax=axes[1])
    axes[1].set_title(f"Confusion Matrix ({title_prefix})", fontsize=14, fontweight='bold')
    axes[1].set_xlabel("Predicted Label (SVM)", fontsize=12)
    axes[1].set_ylabel("True Label (Behavior)", fontsize=12)
    axes[1].set_xticklabels(['Stay (0)', 'Switch (1)'])
    axes[1].set_yticklabels(['Stay (0)', 'Switch (1)'])

    # --- Plot 3: ROC Curve ---
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'{title_prefix} (AUC = {roc_auc:.2f})')
    axes[2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
    axes[2].set_title("Receiver Operating Characteristic (ROC)", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[2].set_ylabel("True Positive Rate (Sensitivity)")
    axes[2].legend(loc="lower right", fontsize=12)

    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN EXECUTABLE SCRIPT
# =============================================================================

if __name__ == "__main__":
    
    # [!] ASSUMING THESE VARIABLES ARE ALREADY LOADED INTO THE ENVIRONMENT:
    # coords_resp_cue_resp, sess_ids, labels_df, switch_per_sess
    
    # 1. Plot raw multi-interval trajectories
    plot_all_trajectories(coords_resp_cue_resp, sess_ids, labels_df)
    
    # 2. Extract features and switches
    intervals, X = extract_displacements(coords_resp_cue_resp)
    switches = [s for sess in switch_per_sess for s in switch_per_sess[sess]]
    switch_array = np.array(switches)
    y_true = switch_array.astype(int)

    # 3. Principal Component Analysis
    X_pca, pca_model = perform_pca_and_plot_variance(X)

    # 4. K-Means Clustering (k=4)
    # K=4 is instructive for how large body movements impact switch behavior
    k = 4
    labels, centroids = perform_kmeans_and_plot(X_pca, pca_model, k=k)

    # 5. Cluster Visualizations
    port_pos = labels_df[labels_df["sess"] == str(sess_ids[0])]["port_tracks"].item()
    plot_cluster_visualizations(intervals, labels, ports_data=port_pos, k=k)

    # 6. Aligned Cluster & Switch Scatter
    plot_behavioral_switch_overlay(X_pca, labels, switch_array, centroids)

    # ---------------------------------------------------------
    # MACHINE LEARNING EVALUATIONS
    # ---------------------------------------------------------
    
    # MODEL 1: Predict Switch based on K-Means Clusters
    # We one-hot encode the clusters to avoid numerical scaling bias
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    X_cluster_encoded = encoder.fit_transform(labels.reshape(-1, 1))
    svm_cluster = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
    
    evaluate_svm_dashboard(
        X_features=X_cluster_encoded, y_true=y_true, model=svm_cluster, 
        X_pca=X_pca, labels=labels, centroids=centroids, title_prefix="Cluster-Based SVM"
    )
    
    # MODEL 2: Predict Switch based on Raw Displacement Vectors (28D)
    # Continuous spatial coordinates must be standardized for RBF kernels
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    svm_delta = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)
    
    evaluate_svm_dashboard(
        X_features=X_scaled, y_true=y_true, model=svm_delta, 
        X_pca=X_pca, labels=labels, centroids=centroids, title_prefix="Raw-Delta SVM"
    )