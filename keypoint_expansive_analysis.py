#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mutual_info_score
from collections import Counter
from itertools import groupby
#from prefixspan import PrefixSpan

# --- User's Database & System Imports ---
# Adjust these imports based on where your custom modules actually live
import init
from hankslab_db import db_access
from hankslab_db import tonecatdelayresp_db as wm_db
#from hankslab_db import basicRLtasks_db as bandit_db
from sys_neuro_tools import doric_utils as du

# --- Configuration Paths ---
KPMS_PROJECT_DIR = r"C:\Users\cns-th-lab\keypoint_tapus\all_rat_data_config_project"
MODEL_NAME = "full_kappa5000_latent_dim7"
VIDEO_BASE_DIR = r"C:\Users\cns-th-lab\TannerVidsRenamed"


# ==========================================
# 1. BEHAVIORAL DATA & TIMESTAMP ALIGNMENT
# ==========================================

def get_trial_end_ts(sess_data):
    """Get the last state timestamp from a trial to determine its end relative to the start."""
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


def load_aligned_session_data(rat_id, sess_id, event_key="cpoke_in_time"):
    """
    Queries the DB for a session, reads the Doric timestamps, loads KPMS syllables,
    and returns the syllables along with exact frame indices for the requested event.
    """
    print(f"\n--- Loading Data for Session: {sess_id} ---")
    sess_ids = [str(sess_id)] # DB access seems to expect a list
    
    # 1. Fetch DB Data
    wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
    wm_sess_data = wm_loc_db.get_behavior_data(sess_ids)
    
    # Calculate absolute trial start times
    trial_starts_plus_last = db_access.get_fp_trial_start_ts(sess_ids)[int(sess_ids[0])]
    trial_starts = trial_starts_plus_last[:-1]
    
    # Calculate absolute event times (e.g., center poke in)
    event_relative_times = wm_sess_data[event_key].values
    event_abs_times = trial_starts + event_relative_times
    
    # Filter out NaNs (invalid/missed events)
    valid_event_times = event_abs_times[~np.isnan(event_abs_times)]
    print(f"Found {len(valid_event_times)} valid '{event_key}' events.")

    # 2. Fetch Doric Frame Timestamps
    doric_path = os.path.join(VIDEO_BASE_DIR, str(rat_id), "Videos", f"mov_{sess_id}.doric")
    if not os.path.exists(doric_path):
        raise FileNotFoundError(f"Could not find Doric file at {doric_path}")
        
    time_in, _ = du.h5read(
        doric_path,
        ["DataAcquisition", "BehaviorCamera", "Video", "Series0001", "DMK-33UX290", "Time"],
    )
    
    # 3. Align Events to Frames
    # Find the exact frame index where the timestamp exceeds the event time
    event_frames = np.searchsorted(time_in, valid_event_times, side="left")
    
    # Ensure we don't grab frames out of bounds
    event_frames = event_frames[event_frames < len(time_in)]

    # 4. Fetch Keypoint MoSeq Syllables
    # Assumes kpms `extract_results` created a CSV named mov_sessid.csv
    moseq_csv_path = os.path.join(KPMS_PROJECT_DIR, MODEL_NAME, "results", f"mov_{sess_id}.csv")
    if not os.path.exists(moseq_csv_path):
        raise FileNotFoundError(f"Could not find KPMS results at {moseq_csv_path}")
        
    moseq_df = pd.read_csv(moseq_csv_path)
    # The exact column name depends on how you exported it, usually 'syllable' or 'state'
    syllables = moseq_df['syllable'] 
    
    # Check alignment integrity
    if len(syllables) != len(time_in):
        print(f"Warning: MoSeq frames ({len(syllables)}) do not perfectly match Doric frames ({len(time_in)}).")

    return syllables, event_frames


# ==========================================
# 2. ANALYSIS FUNCTIONS
# ==========================================

def plot_categorical_psth(syllables, event_frames, pre_frames=30, post_frames=60, title="Categorical PSTH"):
    n_syllables = int(syllables.max()) + 1
    time_window = pre_frames + post_frames + 1
    psth_counts = np.zeros((n_syllables, time_window))
    valid_events = 0
    
    for event_f in event_frames:
        start_f = event_f - pre_frames
        end_f = event_f + post_frames
        
        if start_f >= 0 and end_f < len(syllables):
            window_syllables = syllables.iloc[start_f:end_f+1].values
            for t, syll in enumerate(window_syllables):
                psth_counts[int(syll), t] += 1
            valid_events += 1
            
    with np.errstate(divide='ignore', invalid='ignore'):
        psth_probs = psth_counts / psth_counts.sum(axis=0, keepdims=True)
        psth_probs = np.nan_to_num(psth_probs)
        
    plt.figure(figsize=(12, 6))
    sns.heatmap(psth_probs, cmap="viridis", cbar_kws={'label': 'Probability'})
    plt.axvline(x=pre_frames, color='red', linestyle='--', linewidth=2)
    plt.xticks(ticks=np.arange(0, time_window, 10), labels=np.arange(-pre_frames, post_frames + 1, 10))
    plt.xlabel("Frames relative to event")
    plt.ylabel("Syllable ID")
    plt.title(f"{title} (n={valid_events} events)")
    plt.tight_layout()
    plt.show()


def plot_time_lagged_mi(syllables, event_frames, max_lag=60, max_step=1):
    event_vector = np.zeros(len(syllables))
    event_vector[event_frames] = 1
    
    lags = np.arange(-max_lag, max_lag + 1,max_step)
    mi_scores = []
    
    for lag in lags:
        if lag < 0:
            shifted_events = event_vector[:lag]
            aligned_sylls = syllables[-lag:]
        elif lag > 0:
            shifted_events = event_vector[lag:]
            aligned_sylls = syllables[:-lag]
        else:
            shifted_events = event_vector
            aligned_sylls = syllables
            
        mi_scores.append(mutual_info_score(aligned_sylls, shifted_events))
        
    plt.figure(figsize=(10, 4))
    plt.plot(lags, mi_scores, color='blue', linewidth=2)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.xlabel("Lag (Frames)")
    plt.ylabel("Mutual Information (bits)")
    plt.title("Time-Lagged MI (Event vs Syllable)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_sequence_mining(syllables, event_frames, pre_f=30, post_f=60):
    # 1. N-Grams on fully collapsed session data
    collapsed_syllables = [k for k, g in groupby(syllables)]
    print("\n--- Top Collapsed Trigrams (Whole Session) ---")
    trigrams = zip(*[collapsed_syllables[i:] for i in range(3)])
    for ngram, count in Counter(trigrams).most_common(5):
        print(f"Pattern: {ngram} | Count: {count}")
        
    # 2. PrefixSpan Motif Mining on Event Windows
    """
    print("\n--- PrefixSpan Motif Mining (Event Windows) ---")
    event_sequences = []
    
    for event_f in event_frames:
        start_f = max(0, event_f - pre_f)
        end_f = min(len(syllables), event_f + post_f)
        window_sylls = syllables.iloc[start_f:end_f].values
        
        # Collapse syllables within the window (removes duration noise)
        collapsed_window = [k for k, g in groupby(window_sylls)]
        event_sequences.append(collapsed_window)
        
    # Motifs must be present in at least 15% of events
    min_sup = max(2, int(len(event_frames) * 0.15)) 
    ps = PrefixSpan(event_sequences)
    motifs = ps.frequent(min_sup)
    motifs.sort(key=lambda x: x[0], reverse=True)
    
    complex_motifs = [m for m in motifs if len(m[1]) > 1]
    for support, pattern in complex_motifs[:10]:
        print(f"Support: {support}/{len(event_frames)} events | Motif: {pattern}")"""

def normalize_categorical_intervals(syllables, interval_frames, target_length=100):
    """
    Stretches or shrinks categorical syllable sequences to a fixed target length,
    preserving the proportional duration of each state.
    """
    print(f"\nNormalizing {len(interval_frames)} intervals to {target_length} bins...")
    normalized_intervals = []
    
    for start_f, end_f in interval_frames:
        if start_f >= end_f:
            continue
            
        # Get the raw sequence of syllables for this specific interval
        raw_seq = syllables.iloc[start_f:end_f].values
        L = len(raw_seq)
        
        # Create proportionally spaced indices
        # Example: mapping a 50-frame sequence to a 100-bin array will sample each frame twice.
        indices = np.floor(np.linspace(0, L - 1, target_length)).astype(int)
        
        # Sample the raw sequence using the proportional indices
        norm_seq = raw_seq[indices]
        normalized_intervals.append(norm_seq)
        
    return np.array(normalized_intervals)


def plot_normalized_interval_psth(normalized_intervals, title="Time-Normalized Interval PSTH"):
    """
    Plots a probability heatmap of syllables across a normalized 0-100% time scale.
    """
    if len(normalized_intervals) == 0:
        print("No intervals to plot.")
        return
        
    num_intervals, target_length = normalized_intervals.shape
    n_syllables = int(np.max(normalized_intervals)) + 1
    
    # Initialize a count matrix: (num_syllables, target_length)
    psth_counts = np.zeros((n_syllables, target_length))
    
    # Count the occurrences of each syllable at every normalized time bin
    for t in range(target_length):
        # bincount calculates frequency of each ID in this specific time column
        counts = np.bincount(normalized_intervals[:, t], minlength=n_syllables)
        psth_counts[:, t] = counts
        
    # Convert to probabilities across trials
    psth_probs = psth_counts / num_intervals
    
    plt.figure(figsize=(12, 6))
    sns.heatmap(psth_probs, cmap="viridis", cbar_kws={'label': 'Probability'})
    
    # Format x-axis as percentages (0% to 100% of the interval)
    num_ticks = 11
    plt.xticks(
        ticks=np.linspace(0, target_length, num_ticks), 
        labels=[f"{int(x)}%" for x in np.linspace(0, 100, num_ticks)]
    )
    
    plt.xlabel("Normalized Interval Progression")
    plt.ylabel("Syllable ID")
    plt.title(f"{title} (n={num_intervals} bounded trials)")
    plt.tight_layout()
    plt.show()
    
    return psth_probs

def load_aligned_interval_data(rat_id, sess_id, start_key="response_cue_time", end_key="response_time"):
    """
    Loads MoSeq syllables and finds the exact start/end frame indices for a specified interval.
    """
    print(f"\n--- Loading Interval Data: {start_key} -> {end_key} ---")
    sess_ids = [str(sess_id)]
    
    # 1. Fetch DB Data
    wm_loc_db = wm_db.LocalDB_ToneCatDelayResp()
    wm_sess_data = wm_loc_db.get_behavior_data(sess_ids)
    
    trial_starts = db_access.get_fp_trial_start_ts(sess_ids)[int(sess_ids[0])][:-1]
    
    # Calculate absolute start and end times
    start_abs_times = trial_starts + wm_sess_data[start_key].values
    end_abs_times = trial_starts + wm_sess_data[end_key].values
    
    # 2. Filter valid intervals (neither start nor end can be NaN)
    valid_mask = ~np.isnan(start_abs_times) & ~np.isnan(end_abs_times)
    valid_starts = start_abs_times[valid_mask]
    valid_ends = end_abs_times[valid_mask]
    print(f"Found {len(valid_starts)} valid bounded intervals.")

    # 3. Fetch Doric Frame Timestamps
    doric_path = os.path.join(VIDEO_BASE_DIR, str(rat_id), "Videos", f"mov_{sess_id}.doric")
    time_in, _ = du.h5read(
        doric_path,
        ["DataAcquisition", "BehaviorCamera", "Video", "Series0001", "DMK-33UX290", "Time"],
    )
    
    # 4. Align Events to Frames
    start_frames = np.searchsorted(time_in, valid_starts, side="left")
    end_frames = np.searchsorted(time_in, valid_ends, side="left")
    
    # Stack into a list of [start, end] pairs, clipping to max video length
    max_frame = len(time_in) - 1
    interval_frames = np.clip(np.column_stack((start_frames, end_frames)), 0, max_frame)

    # 5. Fetch Keypoint MoSeq Syllables
    moseq_csv_path = os.path.join(KPMS_PROJECT_DIR, MODEL_NAME, "results", f"mov_{sess_id}.csv")
    syllables = pd.read_csv(moseq_csv_path)['syllable'] 

    return syllables, interval_frames

# ==========================================
# %%3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Define your subject and session
    RAT_ID = "198"
    SESS_ID = "116543"
    
    # You can change this to "response_time", "response_cue_time", etc.
    TARGET_EVENT = "cpoke_in_time" 
    
    try:
        # Load and align
        syllables, event_frames = load_aligned_session_data(
            rat_id=RAT_ID, 
            sess_id=SESS_ID, 
            event_key=TARGET_EVENT
        )
        
        # Run Analytics
        plot_categorical_psth(syllables, event_frames, pre_frames=180, post_frames=30, title=f"PSTH Aligned to {TARGET_EVENT}")
        plot_time_lagged_mi(syllables, event_frames, max_lag=500, max_step=5)
        run_sequence_mining(syllables, event_frames, pre_f=30, post_f=60)
        
    except Exception as e:
        print(f"Error processing session {SESS_ID}: {e}")
        

        
#%%
if __name__ == "__main__":
    RAT_ID = "198"
    SESS_ID = "116543"
    
    try:
        # 1. Load the exact start and end frames for the behavioral interval
        syllables, interval_frames = load_aligned_interval_data(
            rat_id=RAT_ID, 
            sess_id=SESS_ID, 
            start_key="response_cue_time", 
            end_key="response_time"
        )
        
        # 2. Extract sequences for motif mining (un-normalized, just collapsed)
        #run_interval_sequence_mining(syllables, interval_frames, min_support_ratio=0.15)
        
        # 3. Normalize intervals to exactly 100 bins (0 to 100% completion)
        normalized_matrix = normalize_categorical_intervals(
            syllables, 
            interval_frames, 
            target_length=100
        )
        
        # 4. Plot the warped categorical PSTH
        plot_normalized_interval_psth(
            normalized_matrix, 
            title="Syllable Probabilities (Cue -> Response)"
        )
        
    except Exception as e:
        print(f"Error processing interval session {SESS_ID}: {e}")