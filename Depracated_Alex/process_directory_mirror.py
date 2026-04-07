#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 11:43:41 2025

@author: alex
"""

import os
import subprocess
import sys
import datetime

# --- Configuration ---

# 1. SET YOUR FOLDER PATHS
# The parent directory containing your source videos and animal subfolders.
parent_dir = '/Users/your_username/Desktop/Source_Videos'

# The destination directory where the mirrored structure and processed files will be saved.
dest_dir = '/Users/your_username/Desktop/Processed_Videos'

# 2. DEFINE YOUR COMMAND
# This is the command to run on each new file.
# Use '{input}' and '{output}' as placeholders. They will be replaced
# with the actual file paths during the script's execution.
#
# EXAMPLE using ffmpeg to re-encode a video:
cmd = [
    'ffmpeg',
    '-i', '{input}',   # Input file placeholder
    '-c:v', 'libx264',
    '-preset', 'slow',
    '-crf', '22',
    '-c:a', 'aac',
    '-b:a', '128k',
    '{output}'         # Output file placeholder
]
# ---------------------
def get_date(file_path):
    # Get the creation time as a timestamp
    creation_timestamp = os.path.getctime(file_path)
    # Convert the timestamp to a datetime object
    creation_datetime = datetime.datetime.fromtimestamp(creation_timestamp)
    return str(creation_datetime)

def reformatted_name(file_path):
    """
    Defines the name of the output file based on the input filename.
    
    --- CUSTOMIZE THIS FUNCTION ---
    For example, this function adds '_processed' and changes the extension to .mp4.
    """
    # os.path.splitext() splits 'my_video.mov' into ('my_video', '.mov')
    filename = os.path.basename(file_path)
    name, ext = os.path.splitext(filename)
    reformat_name = f"{name}_{get_date(file_path)}__r{ext}"
    return reformat_name


def process_videos():
    """
    Main function to mirror directories and process new files.
    """
    # Ensure the parent directory exists before starting
    if not os.path.isdir(parent_dir):
        print(f"Error: Source directory not found at '{parent_dir}'. Aborting. 🛑")
        sys.exit()

    print("Starting script...")
    
    # Walk through the entire source directory tree
    for dirpath, dirnames, filenames in os.walk(parent_dir):
        # --- Part 1: Mirror Directory Structure ---
        relative_path = os.path.relpath(dirpath, parent_dir)
        dest_dir_path = os.path.join(dest_dir, relative_path)
        
        # This will create the directory if it doesn't exist and do nothing if it does.
        os.makedirs(dest_dir_path, exist_ok=True)
        
        # --- Part 2: Process Files ---
        if not filenames:
            continue # Skip directories that have no files
            
        print(f"\nScanning directory: {dirpath}")
        for filename in filenames:
            # Generate the full path for the expected output file
            output_filename = reformatted_name(os.path.join(dirpath ,filename))
            output_file_path = os.path.join(dest_dir_path, output_filename)
            
            # THE CORE LOGIC: Check if the processed file already exists
            if not os.path.exists(output_file_path):
                input_file_path = os.path.join(dirpath, filename)
                
                # Replace placeholders in the command with actual file paths
                final_cmd = [arg.format(input=input_file_path, output=output_file_path) for arg in cmd]
                
                print(f"  -> Processing '{filename}'...")
                try:
                    # Run the command. check=True will raise an error if the command fails.
                    subprocess.run(final_cmd, check=True, capture_output=True, text=True)
                    print(f"     Success: Saved to '{output_file_path}'")
                except subprocess.CalledProcessError as e:
                    print(f"     Error processing '{filename}'.")
                    print(f"     Command failed with exit code {e.returncode}")
                    print(f"     Stderr: {e.stderr}")
            else:
                # If the file exists, skip it
                print(f"  -> Skipping '{filename}', already processed.")

    print("\nScript finished! ✨")


# Run the main function
if __name__ == "__main__":
    process_videos()