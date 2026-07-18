# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:09:42 2026

@author: cns-th-lab
"""

import cv2
import pandas as pd
import numpy as np
import h5py

# --- Transformation and Filtering Functions ---

def ThresholdedPositions(positions, scores, threshold):
    mask = scores < threshold
    positions[mask] = np.nan
    return positions

def ThresholdedInterpolatedPositions(positions, scores, threshold, interpol_type, max_gap):
    mask = scores < threshold
    positions[:, :, 0,:][mask] = np.nan
    positions[:, :, 1,:][mask] = np.nan
    
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

def RunApp(video_path, inference_path, output_window, fps):
    #Get Thresholded Coordinates (Only 1 score for both so must index separately)
    tracks_coords, node_names, scores = ExtractH5RawData(inference_path)
    thresh_coords_x = ThresholdedPositions(tracks_coords[:,:,0,:], scores, .3)
    thresh_coords_y = ThresholdedPositions(tracks_coords[:,:,1,:], scores, .3)
        
    # --- Setup and main loop
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    frame_interval = 1
    
    print("--- Controls ---")
    print("[D] or Right Arrow : Next Frame")
    print("[A] or Left Arrow  : Previous Frame")
    print("[I]                : Change Interval")
    print("[F]                : Jump to specific Frame")
    print("[S]                : Jump to specific Second")
    print("[Q]                : Quit")
    print("[0-9]              : Toggle Transformation Slots")
    print("----------------")
    
    transformation_0 = False
    transformation_1 = False
    transformation_2 = False
    transformation_3 = False
    transformation_4 = False
    transformation_5 = False
    
    while True:
        # Seek to the current target frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
    
        # Extract coordinates for the current frame
        try:
            current_pos = tracks_coords[frame_idx, :, :, 0]
            current_scores = scores[frame_idx, :]
            
            # Iterating through keypoints
            for i in range(np.shape(current_pos)[0]):
                if np.isnan(current_pos[i,0]) or np.isnan(current_pos[i,1]):
                    continue
                
                x = int(current_pos[i,0])
                y = int(current_pos[i,1])
                
                # Draw the keypoint if coordinates are valid
                if x > 0 and y > 0:
                    cv2.circle(frame, center=(x, y), radius=5, color=(0, 255, 0), thickness=-1)
                    cv2.putText(frame, f"{node_names[i]}: {round(current_scores[i][0],2)}", (x + 8, y - 8), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        except IndexError:
            print("No tracking data this frame")
            pass # No tracking data for this frame
    
        #Transformation Handling
        if transformation_0:
            # Extract coordinates for the current frame
            try:
                print(np.shape(thresh_coords_x))
                curr_thresh_coords_x = thresh_coords_x[frame_idx,:,0]
                curr_thresh_coords_y = thresh_coords_y[frame_idx,:,0]
                print(np.shape(curr_thresh_coords_x))
                
                # Iterating through keypoints
                for i in range(np.shape(curr_thresh_coords_x)[0]):
                    if np.isnan(curr_thresh_coords_x[i]) or np.isnan(curr_thresh_coords_y[i]):
                        continue
                    
                    x = int(curr_thresh_coords_x[i])
                    y = int(curr_thresh_coords_y[i])
                    
                    # Draw the keypoint if coordinates are valid
                    if x > 0 and y > 0:
                        cv2.circle(frame, center=(x+2, y), radius=5, color=(255, 0, 0), thickness=-1)
                        #cv2.putText(frame, f"{node_names[i]}: {round(current_scores[i][0],2)}", (x + 8, y - 8), 
                        #            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            except IndexError:
                print("No tracking data this frame")
                pass # No tracking data for this frame
    
        # Overlay frame and time index text
        current_time = frame_idx / fps
        cv2.putText(frame, f"Frame: {frame_idx}/{total_frames - 1} | Time: {current_time:.2f}s", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(output_window, frame)
        
        # Keyboard Navigation
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d') or key == 83: # 'd' or Right Arrow
            if frame_idx < total_frames - frame_interval:
                frame_idx += frame_interval
                
        elif key == ord('a') or key == 81: # 'a' or Left Arrow
            if frame_idx > frame_interval:
                frame_idx -= frame_interval
        
        elif key == ord('i'): # Select new interval
            try:
                target_interval = int(input("\nEnter new amount of frames to skip by (1-300): "))
                if 1 <= target_interval < 300:
                    frame_interval = target_interval
                else:
                    print("Target interval too big or less than zero.")
            except ValueError:
                print("Invalid input. Please enter a whole number.")
                
        elif key == ord('f'): # Jump to Frame
            try:
                target_frame = int(input(f"\nEnter target frame (0 to {total_frames - 1}): "))
                if 0 <= target_frame < total_frames:
                    frame_idx = target_frame
                else:
                    print("Frame out of bounds.")
            except ValueError:
                print("Invalid input. Please enter a whole number.")
                
        elif key == ord('s'): # Jump to Second
            try:
                target_sec = float(input(f"\nEnter target second (0 to {(total_frames - 1) / fps:.2f}): "))
                target_frame = int(target_sec * fps)
                if 0 <= target_frame < total_frames:
                    frame_idx = target_frame
                else:
                    print("Time out of bounds.")
            except ValueError:
                print("Invalid input. Please enter a number.")
                
        elif key == ord('q'): # Quit
            break
        
        #Transformation toggles
        elif key == ord('0'):
            transformation_0 = not transformation_0
            print(f"Threshold coords toggled: {transformation_0}")
        elif key == ord('1'):
            transformation_1 = not transformation_1
        elif key == ord('2'):
            transformation_2 = not transformation_2
        elif key == ord('3'):
            transformation_3 = not transformation_3
        elif key == ord('4'):
            transformation_4 = not transformation_4
        elif key == ord('0'):
            transformation_5 = not transformation_5
    
        
    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    
#%% Potential old code for plotting actual edges if necessary
"""def PlotSkeleton(frame, processed_dict, skeleton_color="black", nodes_mark=[], ax=None):
    '''Uses processed dict from process_hdf5_data to visualize the skeleton
    in matplotlib figure. DOES NOT SHOW GRAPH AUTOMATICALLY (use plt.show()).
    ---
    Params: 
    frame: the current frame in the video that needs to be plotted
    processed_dict: result of processing hdf5 file
    skeleton_color: color of the edges
    nodes_mark: boolean mask for which nodes may be outliers
    ---
    Returns: None
    '''
    #plt.gca().invert_yaxis() #Images have 0,0 at top left and positive down
    for node_idx in range(len(frame)):
        if len(nodes_mark) != 0  and nodes_mark[node_idx] == 1:
            node_color="red"
        else:
            node_color=skeleton_color
        if ax != None:
            ax.scatter(frame[node_idx,0], frame[node_idx,1], color=node_color)
        else:
            plt.scatter(frame[node_idx,0], frame[node_idx,1], color=node_color)
        offset_x=20
        if skeleton_color == "black" or ((skeleton_color=="green") and (nodes_mark[node_idx] == 1)):
            if ax != None:
                ax.text(frame[node_idx,0]+offset_x, frame[node_idx,1],processed_dict["node_names"][node_idx])
            else:
                plt.text(frame[node_idx,0]+offset_x, frame[node_idx,1],processed_dict["node_names"][node_idx])
        #plt.legend()

    for edge_ind in processed_dict["edge_inds"]:
        #Get the first and second indices and use them to get the x position
        x = [frame[edge_ind[0]][0], frame[edge_ind[1]][0]]
        #Get the first and second indices and use them to get the y position
        y = [frame[edge_ind[0]][1], frame[edge_ind[1]][1]]
        #Plot the current edge
        if ax != None:
            ax.plot(x,y,color=skeleton_color)
        else:
            plt.plot(x, y, color = skeleton_color)"""