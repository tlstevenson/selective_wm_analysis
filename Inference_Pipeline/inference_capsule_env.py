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
import logging

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
    if os.path.exists(single_path):
        predictor = sleap.load_model([single_path], batch_size=16)
    elif os.path.exists(centered_path) and os.path.exists(centroid_path):
        predictor = sleap.load_model([centroid_path, centered_path], batch_size=16)
    else:
        raise Exception("The path to the models does not exist. Please replace it with a valid path.")
    video = sleap.load_video(vid_path)
    print(f"Video loaded: {video.shape}, {video.dtype}", flush=True)
    # Turn on INFO logging to reveal SLEAP's internal progress updates
    logging.getLogger("sleap").setLevel(logging.INFO)
    predictions = predictor.predict(video)
    predictions.export(write_path)
    print(f"Predictions exported to {write_path}", flush=True)

def RunInferenceList(video_paths, single_path, centroid_path, centered_path, write_paths):
    """Iterates through a specific list of provided video paths."""
        
    for i in range(len(video_paths)):
        # 5. Skip inference if this specific analysis file already exists
        if os.path.exists(write_paths[i]):
            print(f"\nSkipping video: {video_paths[i]} (Analysis file already exists at {write_paths[i]})", flush=True)
            continue
            
        print(f"\nFound video to analyze: {video_paths[i]}", flush=True)
        print(f"  > Output labels will be saved to: {write_paths[i]}", flush=True)

        try:
            print("  > Running inference...", flush=True)
            RunInference(video_paths[i], single_path, centroid_path, centered_path, write_paths[i])
        except Exception as e:
            print(f"  > FAILED to analyze the video: {video_paths[i]}. Error: {e}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Missing arguments. Expected config path and at least one video path.")
        sys.exit(1)

    config_path = sys.argv[1]
    num_vids = int(sys.argv[2])
    video_list = sys.argv[2:2+num_vids+1]
    write_path_list = sys.argv[2+num_vids+1:]
    
    with open(config_path, "r") as file:
        config = json.load(file)
        
        analysis_folder = config.get("analysis_folder")
        single_path = config.get("single_model_path", "NoFile")
        centroid_path = config.get("centroid_model_path", "NoFile")
        centered_path = config.get("centered_model_path", "NoFile")
        
    RunInferenceList(video_list, single_path, centroid_path, centered_path, analysis_folder)
    print("\nProcessing complete.", flush=True)