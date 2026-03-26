# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 11:19:47 2026

@author: cns-th-lab
"""

import cv2
import os
import torch
import numpy as np
from sam2.build_sam import build_sam2_video_predictor

# ==========================================
#%% 1. CONFIGURATION
# ==========================================
VIDEO_PATH = r"E:\Tanner_Vids\ReformattedVideos\199\mov_0001_r.mp4"
FRAMES_DIR = r"E:\Tanner_Vids\RatFramesTest"

# SAM 2 Paths
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"
CHECKPOINT = r"C:/Users/hankslab/Downloads/sam2.1_hiera_small.pt"

# You MUST find these exact X, Y pixel coordinates in your 00000.jpg frame!

# Provide multiple coordinates for the rat
points = np.array([
    [447, 838], # Rat Head (Positive)
    [662, 656], # Rat Torso (Positive)
    [854, 326], # Rat Tail Base (Positive)
    [265, 955], # The Cable (Negative)
], dtype=np.float32) #Consider adding floor or walls as negative

# ==========================================
# 2. DEFINING THE SAVING FUNCTION
# ==========================================
def save_isolated_frame(raw_frame, boolean_mask, frame_idx, output_dir):
    """
    Takes the raw frame and SAM 2 mask, isolates the rat on a black background, 
    and saves it sequentially to the specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the SAM 2 True/False mask into a 3D OpenCV mask (0s and 1s)
    mask_uint8 = (boolean_mask * 1).astype(np.uint8)
    mask_3d = np.repeat(mask_uint8[:, :, np.newaxis], 3, axis=2)
    
    # Apply the mask (everything outside the rat becomes [0,0,0] pure black)
    isolated_frame = raw_frame * mask_3d
    
    # Format the filename to match standard tracking software (e.g., 00005.jpg)
    filename = os.path.join(output_dir, f"{frame_idx:05d}.jpg")
    cv2.imwrite(filename, isolated_frame)

# ==========================================
#%% 3. FRAME EXTRACTION (OpenCV)
# ==========================================
print(f"Extracting first 500 frames to {FRAMES_DIR}/ ...")
os.makedirs(FRAMES_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0
while frame_count < 500:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_filename = os.path.join(FRAMES_DIR, f"{frame_count:05d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_count += 1

cap.release()
print(f"Successfully extracted {frame_count} frames.\n")

# ==========================================
#%% 4. SAM 2 INITIALIZATION & PROMPTING
# ==========================================
OUTPUT_DIR = r"E:\Tanner_Vids\CleanRatFramesTest"
print("Loading SAM 2 into GPU memory...")
predictor = build_sam2_video_predictor(MODEL_CFG, CHECKPOINT)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    inference_state = predictor.init_state(video_path=FRAMES_DIR)

    # Setup Positive (Rat) and Negative (Cable) prompts
    labels = np.array([1, 1, 1, 0], np.int32)
    
    print("Sending prompt to Frame 0...")
    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=1, 
        points=points,
        labels=labels,
    )
    print("Rat identified, cable excluded.")

    # ==========================================
    # 5. PROPAGATING & SAVING
    # ==========================================
    print(f"Tracking and saving cleaned frames to {OUTPUT_DIR}/ ...")
    
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        
        # 1. Get the binary mask from SAM 2 (True/False)
        mask = (out_mask_logits[0, 0] > 0.0).cpu().numpy()
        
        # 2. Read the original frame from your extracted folder
        original_frame_path = os.path.join(FRAMES_DIR, f"{out_frame_idx:05d}.jpg")
        original_frame = cv2.imread(original_frame_path)
        
        # 3. Call our custom function to clean and save the frame
        if original_frame is not None:
            save_isolated_frame(original_frame, mask, out_frame_idx, OUTPUT_DIR)
        
        # Print progress every 50 frames
        if out_frame_idx % 50 == 0:
            print(f"Successfully cleaned and saved frame {out_frame_idx}")
            
print("\nPipeline complete! Check the 'rat_frames_cleaned' folder.")