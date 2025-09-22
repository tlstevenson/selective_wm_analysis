#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:38:36 2025

@author: alex
DEPRACATED: 9/18/2025
"""

import os
import sys
import subprocess
import datetime

#First the program mirrors the directory structure of the original videos
#It does not copy the videos to these locations

# --- Configuration ---
source_dir = r"/Users/alex/Documents/HanksLabGithub/selective_wm_analysis/Bob"
dest_dir = r"/Users/alex/Documents/HanksLabGithub/selective_wm_analysis/BobStructureOnly"
# ---------------------

# 1. Check if the destination directory already exists.
if os.path.exists(dest_dir):
    print(f"Error: Destination directory '{dest_dir}' already exists. Aborting. 🛑")
    sys.exit() # Exit the script
try:
    print(f"Destination does not exist. Starting to mirror directory structure...")
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_dir}")
    print("-" * 20)

    # 2. Walk through the source directory tree.
    for dirpath, dirnames, filenames in os.walk(source_dir):
        relative_path = os.path.relpath(dirpath, source_dir)
        new_dir_path = os.path.join(dest_dir, relative_path)
        os.makedirs(new_dir_path, exist_ok=False)
        print(f"Created: {new_dir_path}")

    print("-" * 20)
    print("Directory structure mirrored successfully! ✨")

except FileNotFoundError:
    print(f"Error: The source directory '{source_dir}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    
#Next the file iterates through all the videos in the original parent directory
#It checks the destination directory for a corresponding file ending in _r
#If no file is found, it creates a reformated video under the video convention
#Naming convention: mov_001_get_date()_r

def process_video(origin_vid_path, new_vid_path):
    # Define the output file path for the processed video

    # Define the FFmpeg command
    command = [
    'ffmpeg',
    '-y', # Overwrite output files without asking
    '-i', origin_vid_path,
    '-c:v', 'libx264',
    '-pix_fmt', 'yuv420p',
    '-preset', 'superfast',
    '-crf', '23',
    new_vid_path
    ]

    print(f"Processing video: {origin_vid_path}")
    try:
        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Successfully processed {origin_vid_path} -> {new_vid_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {origin_vid_path} : {e}")
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in your system's PATH.")
        print("Please install FFmpeg to run this script.")

    
def get_date(file_path):
    # Get the creation time as a timestamp
    creation_timestamp = os.path.getctime(file_path)
    # Convert the timestamp to a datetime object
    creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
    return str(creation_datetime)

# os.walk() generates the file paths in a directory tree
for dirpath, dirnames, filenames in os.walk(source_dir):
    relative_path = os.path.relpath(dirpath, source_dir)
    new_dir_path = os.path.join(dest_dir, relative_path)
    for filename in filenames:        
        # Check for a corresponding video ending with '_r'
        name_without_ext, ext = os.path.splitext(filename)
        reformat_name = f"{name_without_ext}_{get_date()}__r{ext}"
        new_vid_path = os.path.join(new_dir_path, reformat_name)
        if not os.path.exists(new_vid_path):
            process_video(filename, new_vid_path)
            print(f"Processed {new_vid_path}")