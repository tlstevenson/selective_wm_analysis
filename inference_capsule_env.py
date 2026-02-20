#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 20:15:24 2026

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import sleap 
import json
import ffmpeg

def get_mp4_creation_date_ffmpeg(filepath: str) -> str:
    if not os.path.exists(filepath): return f"Error: File not found at {filepath}"
    try:
        probe_data = ffmpeg.probe(filepath)
        tags = probe_data.get('format', {}).get('tags', {})
        date_str = tags.get('creation_time')
        return date_str if date_str else "Creation date metadata not found."
    except Exception as e:
        return f"An unexpected error occurred: {e}"

def RunInference(vid_path, single_path, centroid_path, centered_path, write_path):
    if single_path != "NoFile":
        predictor = sleap.load_model([single_path], batch_size=16)
    else:
        predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    video = sleap.load_video(vid_path)
    print(f"Video loaded: {video.shape}, {video.dtype}", flush=True)
    predictions = predictor.predict(video)
    predictions.export(write_path)
    print(f"Predictions exported to {write_path}", flush=True)

def RunInferenceList(video_paths, single_path, centroid_path, centered_path, analysis_folder):
    """Iterates through a specific list of provided video paths."""
        
    for file_path in video_paths:
        filename = os.path.basename(file_path)
        name_without_ext, ext = os.path.splitext(filename)
        
        # 1. Extract animal name (parent folder) and video name
        animal_name = os.path.basename(os.path.dirname(file_path))
        video_name = name_without_ext
        
        # 2. Get the date (FIX TO BE CURRENT DATE)
        date_str = get_mp4_creation_date_ffmpeg(file_path)
        date_part = date_str.split('T')[0] if "Error" not in date_str and "not found" not in date_str else "UnknownDate"
        
        # 3. Construct the nested folder structure: analysis_folder / animal_name / video_name
        target_dir = os.path.join(analysis_folder, animal_name, video_name)
        os.makedirs(target_dir, exist_ok=True)
        
        # 4. Construct the output file name: videoname_date.h5
        new_labels_name = f"{video_name}_{date_part}.h5"
        write_path = os.path.join(target_dir, new_labels_name)
        
        # 5. Skip inference if this specific analysis file already exists
        if os.path.exists(write_path):
            print(f"\nSkipping video: {filename} (Analysis file already exists at {write_path})", flush=True)
            continue
            
        print(f"\nFound video to analyze: {file_path}", flush=True)
        print(f"  > Output labels will be saved to: {write_path}", flush=True)

        try:
            print(f"  > Running inference...", flush=True)
            RunInference(file_path, single_path, centroid_path, centered_path, write_path)
        except Exception as e:
            print(f"  > FAILED to analyze the video: {filename}. Error: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Missing arguments. Expected config path and at least one video path.")
        sys.exit(1)

    config_path = sys.argv[1]
    video_list = sys.argv[2:] 
    
    with open(config_path, "r") as file:
        config = json.load(file)
        
        analysis_folder = config.get("analysis_folder")
        single_path = config.get("single_model_path", "NoFile")
        centroid_path = config.get("centroid_model_path", "NoFile")
        centered_path = config.get("centered_model_path", "NoFile")
        
    RunInferenceList(video_list, single_path, centroid_path, centered_path, analysis_folder)
    print("\nProcessing complete.", flush=True)