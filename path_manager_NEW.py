#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 19:09:21 2026

@author: alex
"""
import os

def vid_to_h5_raw(path):
    path_without_ext, ext = os.path.splitext(path)
    return f"{path_without_ext[:-2]}_raw.h5" #Replace _r.mp4 with _raw.h5

def h5_raw_to_any(path, ending):
    return f"{path[:path.rfind('_')]}_{ending}.h5"

def get_mirrored_path(parent_folder, child_file, new_folder):
    """
    Finds the mirrored path of a child file in a new destination folder.
    """
    
    try:
        # Extract the relative path (e.g., 'subfolder/file.txt')
        vid_rel_path_in_dir = os.path.relpath(parent_folder, child_file)
        
        # Append the relative path to the new destination folder
        mirrored_path = os.path.join(new_folder, vid_rel_path_in_dir)
        return vid_to_h5_raw(mirrored_path) #Replace _r.mp4 with _raw.h5
        
    except ValueError:
        # This triggers if the child file isn't actually inside the parent folder
        raise ValueError(f"The file '{child_file}' is not inside '{parent_folder}'")

def get_manual_path(analysis_folder, animal_num, video_file):
    filename = os.path.basename(video_file)
    return os.path.join(analysis_folder, str(animal_num), vid_to_h5_raw(filename)) #Replace _r.mp4 with _raw.h5
