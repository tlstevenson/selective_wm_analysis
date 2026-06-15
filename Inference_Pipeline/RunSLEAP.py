# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:56:14 2026

@author: cns-th-lab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1: Model Selection and Labeling
"""

#%% Import statements
from pathlib import Path
import os
import subprocess
import sys

#%% Define traversal function
def get_file_paths(directory_path, extension="None"):
    """Returns a list of strings containing the paths of all files in a directory."""
    path_obj = Path(directory_path)
    
    if extension=="None":
        return [str(file) for file in path_obj.iterdir() if file.is_file()]
    else:
        return[str(file) for file in path_obj.iterdir() if file.is_file() and file.suffix==extension]

#%% Select new videos by directory
vid_par_dir = r"C:\Users\cns-th-lab\Tanner_Alex_Vids"
vid_folders = []
curr_vids = []
for vid_folder in vid_folders:
    curr_vids = curr_vids + get_file_paths(vid_folder, ".mp4")
print("Current videos from directories:", curr_vids)

#%% Select new videos by hand
r"""additional_vids = [r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-07-28.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-08-23.mov_0014.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\237\Videos\237.2026-03-31.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\237\Videos\237.2026-04-03.mov_0004.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\199\Videos\199.2025-07-28.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\199\Videos\199.2025-08-23.mov_0014.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\238\Videos\238.2026-03-31.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\238\Videos\238.2026-04-03.mov_0004.mp4"
                   ]"""
    
additional_vids = [r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-07-28.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\199\Videos\199.2025-07-28.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\237\Videos\237.2026-03-31.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\238\Videos\238.2026-03-31.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\274\Videos\274.2025-09-25.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\400\Videos\400.2025-09-25.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\402\Videos\402.2025-09-25.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\424\Videos\424.2026-03-31.mov_0001.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\483\Videos\483.2026-04-01.mov_0002.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\198\Videos\198.2025-08-23.mov_0014.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\199\Videos\199.2025-08-23.mov_0014.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\237\Videos\237.2026-04-03.mov_0004.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\238\Videos\238.2026-04-03.mov_0004.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\274\Videos\274.2025-10-24.mov_0015.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\400\Videos\400.2025-10-24.mov_0015.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\402\Videos\402.2025-10-24.mov_0015.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\424\Videos\424.2026-04-03.mov_0004.mp4",
                   r"C:\Users\cns-th-lab\Tanner_Alex_Vids\483\Videos\483.2026-04-03.mov_0004.mp4"
                   ]
for vid_path in additional_vids:
    curr_vids.append(vid_path)
    
if len(additional_vids) > 0:
    print("Current videos :")
    print(curr_vids);
#%% Set up inference write paths
def vid_to_slp(path):
    #.mp4 -> .proj.slp
    path_without_ext, ext = os.path.splitext(path)
    return f"{path_without_ext}.proj.slp"

def get_mirrored_path_slp(parent_folder, child_file, new_folder):
    """
    Finds the mirrored path of a child file in a new destination folder.
    """
    
    try:
        # Extract the relative path (e.g., 'subfolder/file.txt')
        vid_rel_path_in_dir = os.path.relpath(child_file, parent_folder)
        
        # Append the relative path to the new destination folder
        mirrored_path = os.path.join(new_folder, vid_rel_path_in_dir)
        return vid_to_slp(mirrored_path)
        
    except ValueError:
        # This triggers if the child file isn't actually inside the parent folder
        raise ValueError(f"The file '{child_file}' is not inside '{parent_folder}'")

#%% Command to run inference on all files
def run_inference(video_list, write_path_list, model_path):
    """Launches the SLEAP inference pipeline, streaming real-time feedback to the console."""
    if not video_list:
        print("No videos provided for inference. Skipping.")
        return False

    print(f"\nLaunching SLEAP inference on {len(video_list)} videos...\n")
    print("=" * 50)
    
    for i in range(len(video_list)):
        #Skips if inference already exists
        if(os.path.exists(write_path_list[i])):
            print(f"{write_path_list[i]} already exists. Skipping inference.")
            continue
        command = []
        if(len(model_path)==2):
            centroid_model = model_path[0]
            centered_instance = model_path[1]
            if os.path.exists(centroid_model) and os.path.exists(centered_instance):
                command = ["sleap", "track", "-i", video_list[i], "-m", centroid_model, "-m", centered_instance, "-o", write_path_list[i]]
            else:
                print(f"Error: Could not find model paths. Skipping video {video_list[i]}.")
                return False
        elif len(model_path) == 1:
            single_instance_model = model_path[0]
            if os.path.exists(single_instance_model):
                command = ["sleap", "track", "-i", video_list[i], "-m", single_instance_model, "-o", write_path_list[i], "--tracking"]
            else:
                print(f"Error: Could not find model path. Skipping video {video_list[i]}.")
                return False
        else:
            raise Exception("The path to the models does not exist. Please replace it with a valid path.")
        #Run the inference command
        try:
            os.makedirs(os.path.dirname(write_path_list[i]), exist_ok=True)
            # Popen streams the output line-by-line
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, shell=False) as process:                # Iterate through the output as it is generated and print to console
                for line in process.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                # Ensure the process is fully complete before checking the exit code
                process.wait()
            if process.returncode == 0:
                print("=" * 50)
                print("Inference completed successfully!")
            else:
                print("=" * 50)
                print(f"Inference failed with exit code {process.returncode}.")
        except Exception as e:
            print(f"Failed to launch subprocess: {e}")
            return False
    print("\nAll videos processed!")
    return True

r"""folder_model_dict = {r"C:\Users\cns-th-lab\SLEAP_Labels_198_402": [r"C:\Users\cns-th-lab\SLEAP_Projects\EENII4C\models\STABLE.260429.centroid.n=72", r"C:/Users/cns-th-lab/SLEAP_Projects/EENII4C/models/STABLE.260429.centered_instance.n=72"],
                     r"C:\Users\cns-th-lab\SLEAP_Labels_198_237x_402":[r"C:\Users\cns-th-lab\SLEAP_Projects\models\260502_198_402_237x.centroid.n=92", r"C:/Users/cns-th-lab/SLEAP_Projects/models/260502_198_402_237x.centered_instance.n=92"],
                     r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402":[r"C:\Users\cns-th-lab\SLEAP_Projects\models\260504_198_199x_237x_402.centroid.n=112", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260504_198_199x_237x_402.centered_instance.n=112"]}
"""
#folder_model_dict = {r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_402":[r"C:\Users\cns-th-lab\SLEAP_Projects\models\260504_198_199x_237x_402.centroid.n=112", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260504_198_199x_237x_402.centered_instance.n=112"]}
folder_model_dict = {r"C:\Users\cns-th-lab\SLEAP_Labels_198_199x_237x_238x_274x_400x_402x_424x_483x":[r"C:\Users\cns-th-lab\SLEAP_Projects\models\260523_198_199x_237x_238x_274x_400x_402x_424x_483x.centroid.n=222", 
                                                                                                      r"C:\Users\cns-th-lab\SLEAP_Projects\models\260523_198_199x_237x_238x_274x_400x_402x_424x_483x.centered_instance.n=222"]}
for analysis_folder in folder_model_dict.keys():
    write_paths = []
    for i in range(len(curr_vids)):
        write_paths.append(get_mirrored_path_slp(vid_par_dir, curr_vids[i], analysis_folder))

    centroid_model_path = folder_model_dict[analysis_folder][0]
    centered_instance_model_path = folder_model_dict[analysis_folder][1]
    run_inference(curr_vids, write_paths, [centroid_model_path, centered_instance_model_path])
print("Labeling complete. Outputs saved to write_paths.")