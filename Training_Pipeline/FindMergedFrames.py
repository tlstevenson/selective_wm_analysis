# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:30:00 2026
@author: cns-th-lab
"""

import cv2
import numpy as np
import os
import shutil

# --- CONFIGURATION ---
ratnum = "04"
img_folder = rf"C:\Users\cns-th-lab\ToOrganizeAndTrash\labeled-data-20260404T015449Z-3-001\labeled-data\mov_00{ratnum}"
video_dir = r"C:\Users\cns-th-lab\Tanner_Alex_Vids"

output_dir = rf"C:\Users\cns-th-lab\ToOrganizeAndTrash\labeled-data-20260404T015449Z-3-001\labeled-data\mov_00{ratnum}"

# OPTIONAL: Set to your CSV path, or set to None or "" to skip CSV processing
csv_path = rf"C:\Users\cns-th-lab\ToOrganizeAndTrash\labeled-data-20260404T015449Z-3-001\labeled-data\mov_00{ratnum}\CollectedData_AITapus.csv"
# csv_path = "" # Example of how to turn CSV processing off

target_rats = [274, 400, 402]
all_rats = [ 274, 400, 402]

# --- FUNCTIONS ---

def get_matching_video_info(image_path, candidates_dict, frame_index):
    """Compares pixel data and returns both the matching Rat ID and Video Name."""
    target_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if target_img is None:
        return None, None

    for rat_id, video_path in candidates_dict.items():
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if gray_frame.shape != target_img.shape:
                gray_frame = cv2.resize(gray_frame, (target_img.shape[1], target_img.shape[0]))
            
            err = np.mean((target_img.astype("float") - gray_frame.astype("float")) ** 2)
            
            if err < 5.0: 
                vid_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
                return rat_id, vid_name_no_ext
                
    return None, None

# --- MAIN EXECUTION ---

# 1. Determine if we are processing a CSV
process_csv = True
df = None
if csv_path and os.path.isfile(csv_path):
    import pandas as pd
    process_csv = True
    print(f"Valid CSV found. Will split and save CSV data.")
    df = pd.read_csv(csv_path, header=[0, 1, 2])
else:
    print("No valid CSV path provided. Operating in IMAGE-ONLY mode.")

# 2. Scan and Match Images
print("\nIdentifying video sources for images...")
img_to_match_data = {} 
vid_folder_name = os.path.basename(img_folder)

for img_name in os.listdir(img_folder):
    if img_name.lower().endswith(".png"):
        img_path = os.path.join(img_folder, img_name)
        try:
            img_index = int(''.join(filter(str.isdigit, img_name)))
        except ValueError:
            continue

        candidates = {}
        for rat in all_rats:
            rat_vid_dir = os.path.join(video_dir, str(rat), "Videos")
            if os.path.exists(rat_vid_dir):
                for vid_file in os.listdir(rat_vid_dir):
                    if vid_file.endswith(".mp4") and vid_folder_name in vid_file:
                        candidates[rat] = os.path.join(rat_vid_dir, vid_file)
        print(candidates)

        found_rat, found_vid_name = get_matching_video_info(img_path, candidates, img_index)
        
        if found_rat:
            img_to_match_data[img_name] = {'rat_id': found_rat, 'vid_name': found_vid_name}
            print(f"Match: {img_name} -> {found_vid_name} (Rat {found_rat})")

# 3. Create Folders, Filter CSV (if applicable), and Move Frames
print("\nOrganizing files...")
total_moved = 0

for rat in target_rats:
    valid_images = {img: data for img, data in img_to_match_data.items() if data['rat_id'] == rat}
    
    if not valid_images:
        continue

    # Get the target folder name (the video name without .mp4)
    target_folder_name = list(valid_images.values())[0]['vid_name']
    rat_output_folder = os.path.join(output_dir, target_folder_name)
    os.makedirs(rat_output_folder, exist_ok=True)
    
    valid_filenames = list(valid_images.keys())

    # --- CSV Logic (Optional) ---
    if process_csv and df is not None:
        filtered_df = df[df.iloc[:, 2].str.strip().isin(valid_filenames)]
        output_csv_path = os.path.join(rat_output_folder, f"CollectedData_{target_folder_name}.csv")
        filtered_df.to_csv(output_csv_path, index=False)
        print(f"Created CSV: {output_csv_path} ({len(filtered_df)} rows)")

    # --- File Moving Logic ---
    moved_count = 0
    for img_name in valid_filenames:
        src_img_path = os.path.join(img_folder, img_name)
        dst_img_path = os.path.join(rat_output_folder, img_name)
        
        if os.path.exists(src_img_path):
            shutil.move(src_img_path, dst_img_path)
            moved_count += 1
            total_moved += 1
            
    print(f"Moved {moved_count} images to: {rat_output_folder}")

print(f"\nProcess Complete. Successfully organized {total_moved} images.")

#%% Sort file names
file_names = ["mov_0001", "mov_0002", "mov_0003", "mov_0004",]
files = []
par_folder = r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos"
for file in os.listdir(par_folder):
    print(file)
    root, ext = os.path.splitext(file)
    print(ext)
    root, vid_name = os.path.splitext(root)
    vid_name = vid_name[1:]
    print(vid_name)
    print(ext==".mp4" )
    print(vid_name in file_names)
    if ext == ".mp4" and vid_name in file_names:
        files.append(os.path.join(par_folder, file))
for f in files:
    print(f)

#%% Combine path into one column
import os
import csv

# Your specific project variables
project_path = r"C:\Users\cns-th-lab\DeepLabCut_Projects\AllRatsBulky-AITapus-2026-04-08"
config_path = os.path.join(project_path, "config.yaml")
scorer_name = "AITapus"
labeled_data_dir = os.path.join(project_path, "labeled-data")

print("Starting CSV formatting...\n")

# Iterate through every video folder inside labeled-data
for folder_name in os.listdir(labeled_data_dir):
    folder_path = os.path.join(labeled_data_dir, folder_name)
    
    # Ensure it's a directory, not a hidden file
    if not os.path.isdir(folder_path):
        continue
        
    csv_path = os.path.join(folder_path, f"CollectedData_{scorer_name}.csv")
    
    if os.path.exists(csv_path):
        print(f"Checking: {folder_name}")
        
        # Read the raw lines of the CSV
        with open(csv_path, 'r', newline='') as file:
            reader = csv.reader(file)
            lines = list(reader)
            
        # Check if the file contains the Excel-corrupted "Unnamed" columns
        # (Looking at row 0, column 1)
        if len(lines[0]) > 1 and ("Unnamed" in lines[0][1] or lines[0][1] == ""):
            new_lines = []
            
            # 1. Fix the Headers (Rows 0, 1, and 2)
            for i in range(3):
                # Keep the first column (e.g., 'scorer'), drop columns 1 and 2, keep the rest
                fixed_header = [lines[i][0]] + lines[i][3:]
                new_lines.append(fixed_header)
                
            # 2. Fix the Data Rows (Rows 3 and beyond)
            for row in lines[3:]:
                if len(row) < 3:
                    continue # Skip any completely empty rows
                    
                image_filename = row[2] # Grab the 'img00xxxx.png' from column C
                
                # Reconstruct the exact path DeepLabCut expects using the current parent folder name
                # Example: labeled-data\274.2025-09-25.mov_0001\img004380.png
                correct_path = f"labeled-data\\{folder_name}\\{image_filename}"
                
                # Combine the newly built path with all the X/Y coordinates (skipping the old broken paths)
                fixed_row = [correct_path] + row[3:]
                new_lines.append(fixed_row)
                
            # 3. Overwrite the corrupted CSV with the clean data
            with open(csv_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(new_lines)
                
            print("  -> Fixed! Columns merged and paths updated to match folder.\n")
        else:
            print("  -> Already properly formatted. Skipping.\n")

print("-" * 40)
print("All CSVs repaired! Generating .h5 files...\n")

#python -c "import deeplabcut; deeplabcut.convertcsv2h5(r'C:\Users\cns-th-lab\DeepLabCut_Projects\AllRatsBulky-AITapus-2026-04-08\config.yaml', scorer='AITapus')"