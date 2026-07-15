# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 13:26:04 2025

@author: hankslab

"""
import os
import subprocess

def check_processed_vid_exists(file_path, processed_vids_folder, animal_num=None):
    """Given a video file path, it checks if the video has been reformatted before"""
    # Extract just the file name (e.g., 'vid1.mp4' from 'C:/input/vid1.mp4')
    if animal_num == None:
        animal_num = os.path.basename(os.path.dirname(file_path))
    base_name = os.path.basename(file_path)
    name_without_ext, ext = os.path.splitext(base_name)
    
    # Determine the target filename to look for
    if name_without_ext.endswith("_r"):
        target_file_name = base_name
    else:
        # Add the _r to the name before the extension (e.g., .mp4)
        target_file_name = f"{name_without_ext}_r{ext}"
        
    # Construct the full path to check: processed_vids_folder/animal_num/target_file_name
    # Using os.path.join ensures the slashes are correct for your operating system
    target_path = os.path.join(processed_vids_folder, str(animal_num), target_file_name)
    
    # os.path.exists returns True if the file exists, False otherwise
    return os.path.exists(target_path)

def process_video(file_path, processed_vids_folder, animal_num=None):
    """Processes a video using FFmpeg if it hasn't been processed yet.
       Returns the path to the processed video, or None if it failed/skipped."""
    
    if animal_num == None:
        animal_num = os.path.basename(os.path.dirname(file_path))
    filename = os.path.basename(file_path)

    # Check if the file is a video
    if not filename.endswith('.mp4'):
        print(f"Skipping file: {filename} (not an .mp4)")
        return None

    # Determine the output file name and path first
    name_without_ext, ext = os.path.splitext(filename)
    if name_without_ext.endswith('_r'):
        target_file_name = filename
    else:
        target_file_name = f"{name_without_ext}_r{ext}"

    output_dir = os.path.join(processed_vids_folder, str(animal_num), target_file_name)
    print(output_dir)

    # Call our helper function to see if the video is already processed
    if check_processed_vid_exists(file_path, processed_vids_folder, animal_num=animal_num):
        print(f"Skipping video: {filename} (processed version already exists)")
        return output_dir # Return the path since it already exists!
    
    print(output_dir)
    # Ensure the output directory exists before FFmpeg tries to write to it
    os.makedirs(output_dir, exist_ok=True)

    # Define the FFmpeg command
    command = [
        'ffmpeg',
        '-y',               # Overwrite output files without asking
        '-i', file_path,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'superfast',
        '-crf', '23',
         output_dir
    ]

    print(f"Processing video: {filename}")
    try:
        # Run the FFmpeg command
        subprocess.run(command, check=True)
        print(f"Successfully processed {filename} -> {output_dir}")
        return output_dir # Return the path for the newly created video!
        
    except subprocess.CalledProcessError as e:
        print(f"Error processing {filename}: {e}")
        return None
        
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in your system's PATH.")
        print("Please install FFmpeg to run this script.")
        return None