# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 21:15:00 2026
@author: cns-th-lab
"""

import cv2
import numpy as np
import os
import pandas as pd
import shutil # Added for moving files and folders

# --- CONFIGURATION ---
img_folder = r"C:\Users\cns-th-lab\ToOrganizeAndTrash\labeled-data-20260404T015449Z-3-001\labeled-data\mov_0003"
video_dir = r"C:\Users\cns-th-lab\Tanner_Alex_Vids"
csv_path = r"C:\Users\cns-th-lab\ToOrganizeAndTrash\labeled-data-20260404T015449Z-3-001\labeled-data\mov_0003\CollectedData_AITapus.csv"
output_dir = r"C:\Users\cns-th-lab\ToOrganizeAndTrash\labeled-data-20260404T015449Z-3-001\labeled-data\mov_0003"

target_rats = [274, 400, 402]
all_rats = [198, 199, 274, 400, 402, 237, 238, 424, 483]

# --- FUNCTIONS ---

def get_matching_rat(image_path, candidates_dict, frame_index):
    """Compares pixel data between the labeled PNG and candidate MP4 frames."""
    target_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        return None

    for rat_id, video_path in candidates_dict.items():
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Match dimensions if they differ slightly
            if gray_frame.shape != target_img.shape:
                gray_frame = cv2.resize(gray_frame, (target_img.shape[1], target_img.shape[0]))
            
            # MSE calculation
            err = np.mean((target_img.astype("float") - gray_frame.astype("float")) ** 2)
            
            # Threshold 5.0 accounts for MP4 compression artifacts
            if err < 5.0: 
                return rat_id
    return None

# --- MAIN EXECUTION ---

# 1. Load the CSV with the 3-line multi-index header
print("Loading master CSV...")
df = pd.read_csv(csv_path, header=[0, 1, 2])

# 2. Build the Image -> Rat Mapping
print("Identifying video sources for images...")
img_to_rat_map = {}
vid_folder_name = os.path.basename(img_folder) # e.g., 'mov_0001'

for img_name in os.listdir(img_folder):
    if img_name.lower().endswith(".png"):
        img_path = os.path.join(img_folder, img_name)
        
        # Extract frame index (img001320.png -> 1320)
        img_index = int(''.join(filter(str.isdigit, img_name)))

        # Find potential video files in the Tanner_Alex_Vids directory structure
        candidates = {}
        for rat in all_rats:
            rat_vid_dir = os.path.join(video_dir, str(rat), "Videos")
            if os.path.exists(rat_vid_dir):
                for vid_file in os.listdir(rat_vid_dir):
                    # Match video filename with the 'mov_XXXX' folder name
                    if vid_file.endswith(".mp4") and vid_folder_name in vid_file:
                        candidates[rat] = os.path.join(rat_vid_dir, vid_file)

        # Run pixel comparison
        found_rat = get_matching_rat(img_path, candidates, img_index)
        
        if found_rat:
            img_to_rat_map[img_name] = found_rat
            print(f"Match: {img_name} belongs to Rat {found_rat}")

# 3. Create Folders, Filter CSVs, and Move Files
print("\nCreating folders, splitting CSV, and moving frames...")
for rat in target_rats:
    # Get the list of filenames identified for this rat
    valid_filenames = [img for img, r_id in img_to_rat_map.items() if r_id == rat]
    
    if not valid_filenames:
        print(f"Skipping Rat {rat}: No matches found.")
        continue

    # Define the new reformatted folder and file name
    base_name = f"CollectedData_AITapus_Rat{rat}"
    rat_output_folder = os.path.join(output_dir, base_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(rat_output_folder, exist_ok=True)

    # Filter the dataframe based on Column C (Index 2 in Python)
    filtered_df = df[df.iloc[:, 2].str.strip().isin(valid_filenames)]
    
    # Save CSV inside the new folder
    output_csv_path = os.path.join(rat_output_folder, f"{base_name}.csv")
    filtered_df.to_csv(output_csv_path, index=False)
    print(f"Created CSV: {output_csv_path} ({len(filtered_df)} rows)")

    # Move the physical .png files into the new folder
    moved_count = 0
    for img_name in valid_filenames:
        src_img_path = os.path.join(img_folder, img_name)
        dst_img_path = os.path.join(rat_output_folder, img_name)
        
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dst_img_path)
            moved_count += 1
            
    print(f"Moved {moved_count} images to: {rat_output_folder}\n")

print("Process Complete.")