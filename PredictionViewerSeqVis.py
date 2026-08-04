# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 11:42:52 2026

@author: cns-th-lab
"""
import cv2
import pandas as pd
import numpy as np
import h5py

# --- Extraction Function ---
def ExtractH5RawData(inference_path):
    with h5py.File(inference_path, "r") as f:
        node_names = [n.decode('utf-8') for n in f['node_names'][:]]
        scores = np.transpose(f['point_scores'][:], (2, 1, 0)) 
        tracks_coords = np.transpose(f['tracks'][:])
        return tracks_coords, node_names, scores

# --- Updated RunApp ---
def RunApp(video_path, inference_path, output_window, fps, bad_sequences=None, transformations=None):
    """
    Args:
        ...
        bad_sequences: A list of tuples containing (start_frame, end_frame).
        transformations: A list of numpy arrays (frames, nodes, 2, 1) or (frames, nodes, 2)
                         representing the time series to overlay.
    """
    if bad_sequences is None:
        bad_sequences = []
    if transformations is None:
        transformations = []
        
    # Get base raw coordinates
    tracks_coords, node_names, scores = ExtractH5RawData(inference_path)
        
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    frame_interval = 1
    seq_idx = -1 
    
    # State tracker for which transformations are currently visible
    active_transformations = [False] * len(transformations)
    
    # Pre-defined list of distinct BGR colors for up to 10 transformations
    trans_colors = [
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (0, 165, 255),  # Orange
        (255, 255, 0),  # Cyan
        (255, 255, 255),# White
        (128, 0, 128),  # Purple
        (0, 0, 255),    # Red
        (128, 128, 0),  # Teal
        (0, 128, 255)   # Amber
    ]
    
    print("--- Controls ---")
    print("[D] / Right Arrow : Next Frame")
    print("[A] / Left Arrow  : Previous Frame")
    print("[I]               : Change Interval")
    print("[F]               : Jump to specific Frame")
    print("[S]               : Jump to specific Second")
    print("[N]               : Jump to Next Bad Sequence")
    print("[P]               : Jump to Previous Bad Sequence")
    print("[Q]               : Quit")
    
    if len(transformations) > 0:
        print("\n--- Transformations ---")
        for i in range(min(len(transformations), 10)):
            print(f"[{i}] : Toggle Series {i} (Color index {i})")
    print("----------------")
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
    
        # 1. Draw Base Raw Tracking (Green)
        try:
            current_pos = tracks_coords[frame_idx]
            current_scores = scores[frame_idx]
            
            for i in range(np.shape(current_pos)[0]):
                x, y = current_pos[i, 0], current_pos[i, 1]
                # Handle potential 4th dimension safely
                if isinstance(x, np.ndarray): 
                    x, y = x[0], y[0]
                    
                if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                    x, y = int(x), int(y)
                    cv2.circle(frame, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
                    cv2.putText(frame, f"{node_names[i]}: {round(current_scores[i][0],2)}", (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        except IndexError:
            pass 
    
        # 2. Draw Active Transformations
        for t_idx, trans_series in enumerate(transformations):
            if active_transformations[t_idx]:
                try:
                    curr_trans = trans_series[frame_idx]
                    color = trans_colors[t_idx % len(trans_colors)]
                    
                    for i in range(np.shape(curr_trans)[0]):
                        x, y = curr_trans[i, 0], curr_trans[i, 1]
                        
                        # Handle potential 4th dimension safely
                        if isinstance(x, np.ndarray):
                            x, y = x[0], y[0]
                            
                        if not np.isnan(x) and not np.isnan(y) and x > 0 and y > 0:
                            x, y = int(x), int(y)
                            # Draw slightly offset so they don't perfectly cover the base tracking
                            cv2.circle(frame, center=(x + 2, y + 2), radius=5, color=color, thickness=-1)
                except IndexError:
                    pass 
    
        # 3. Overlay frame, time, and sequence info
        current_time = frame_idx / fps
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames - 1} | Time: {current_time:.2f}s", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if bad_sequences and seq_idx != -1:
            curr_seq = bad_sequences[seq_idx]
            cv2.putText(frame, f"Seq {seq_idx + 1}/{len(bad_sequences)}: Frames {curr_seq[0]}-{curr_seq[1]}", 
                        (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(output_window, frame)
        
        # 4. Keyboard Navigation
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d') or key == 83: # Next
            if frame_idx < total_frames - frame_interval:
                frame_idx += frame_interval
                
        elif key == ord('a') or key == 81: # Previous
            if frame_idx > frame_interval:
                frame_idx -= frame_interval
                
        elif key == ord('n'): # NEXT SEQUENCE
            if len(bad_sequences) > 0:
                seq_idx = (seq_idx + 1) % len(bad_sequences)
                frame_idx = bad_sequences[seq_idx][0]
                print(f"Jumped to Sequence {seq_idx + 1}: Frame {frame_idx}")

        elif key == ord('p'): # PREVIOUS SEQUENCE
            if len(bad_sequences) > 0:
                if seq_idx == -1: 
                    seq_idx = len(bad_sequences) - 1
                else:
                    seq_idx = (seq_idx - 1) % len(bad_sequences)
                frame_idx = bad_sequences[seq_idx][0]
                print(f"Jumped to Sequence {seq_idx + 1}: Frame {frame_idx}")
                
        elif key == ord('i'):
            try:
                target_interval = int(input("\nEnter new amount of frames to skip by (1-300): "))
                if 1 <= target_interval < 300:
                    frame_interval = target_interval
            except ValueError:
                pass
                
        elif key == ord('f'):
            try:
                target_frame = int(input(f"\nEnter target frame (0 to {total_frames - 1}): "))
                if 0 <= target_frame < total_frames:
                    frame_idx = target_frame
                    seq_idx = -1
            except ValueError:
                pass
                
        elif key == ord('s'):
            try:
                target_sec = float(input(f"\nEnter target second (0 to {(total_frames - 1) / fps:.2f}): "))
                target_frame = int(target_sec * fps)
                if 0 <= target_frame < total_frames:
                    frame_idx = target_frame
                    seq_idx = -1
            except ValueError:
                pass
                
        elif key == ord('q'): # Quit
            break
            
        # Dynamically map keys '0' through '9' to the transformation list
        elif ord('0') <= key <= ord('9'):
            trans_idx = key - ord('0')
            if trans_idx < len(transformations):
                active_transformations[trans_idx] = not active_transformations[trans_idx]
                print(f"Transformation [{trans_idx}] toggled: {active_transformations[trans_idx]}")
    
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)