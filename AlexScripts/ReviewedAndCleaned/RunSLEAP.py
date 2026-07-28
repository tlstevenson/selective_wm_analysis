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
import shutil

#%% Define traversal function
def get_file_paths(directory_path, extension="None"):
    """Returns a list of strings containing the paths of all files in a directory."""
    path_obj = Path(directory_path)
    
    if extension=="None":
        return [str(file) for file in path_obj.iterdir() if file.is_file()]
    else:
        return[str(file) for file in path_obj.iterdir() if file.is_file() and file.suffix==extension]

#%% Select new videos by directory
vid_par_dir = r"C:\Users\cns-th-lab\TannerVidsRenamed"
vid_folders = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\234\Videos",
               #r"C:\Users\cns-th-lab\TannerVidsRenamed\235\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\419\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\421\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\422\Videos",]
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
    return f"{path_without_ext}.slp"

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
#%%Main execution

#Define model
centroid_model_loc = r"C:\Users\cns-th-lab\SLEAP_Projects\models\260716_port_model.centroid.n=40"
centered_model_loc = r"C:\Users\cns-th-lab\SLEAP_Projects\models\260716_port_model.centered_instance.n=40"
centroid_model_name = os.path.basename(centroid_model_loc)
centered_model_name = os.path.basename(centered_model_loc)
model_name = os.path.basename(os.path.splitext(os.path.splitext(centroid_model_loc)[0])[0])
par_models_folder = os.path.join(vid_par_dir, "models")

#Define a folder to copy the model to
centroid_model_folder_loc = os.path.join(vid_par_dir, "models", centroid_model_name)
centered_model_folder_loc = os.path.join(vid_par_dir, "models", centered_model_name)

#Copy the model to that location
try:
    shutil.copytree(centroid_model_loc, centroid_model_folder_loc)
    shutil.copytree(centered_model_loc, centered_model_folder_loc)
except Exception as e:
    print(e)
    print("Model probably already exists")
print(f"Centroid model new location: {centroid_model_folder_loc}")
print(f"Centered model new location: {centered_model_folder_loc}")
print("Make sure to update these in the program")

#%% Define write paths
write_paths = []
for video in curr_vids:
    try:
        write_paths.append(os.path.join(os.path.dirname(video), "predictions", model_name, vid_to_slp(os.path.basename(video))))
    except:
        print(f"Could not append path for video {video}")
        
#%% Main execiton
run_inference(curr_vids, write_paths, [centroid_model_folder_loc, centered_model_folder_loc])

#%% Convertion to analysis h5 files
def slp_to_analysis_h5(slp_path, h5_path):
    """
    Converts a SLEAP .slp file to a standard analysis .h5 file using sleap-io.
    """
    print(f"  -> Exporting to {h5_path} via CLI...")
    command = ["uv", "run", "sleap", "export", str(slp_path), "-o", str(h5_path)]
    
    try:
        if not os.path.exists(h5_path):
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(e)
    return h5_path

label_folder_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               ]
for folder in label_folder_paths:
    for file in os.listdir(folder):
        file_full_path = os.path.join(folder, file)
        root, ext = os.path.splitext(file_full_path)
        if ext == ".slp":
            h5_path_name = f"{root}.h5"
            if not os.path.exists(h5_path_name):
                print(f"Converting {file_full_path} to {h5_path_name}")
            else:
                print(f"File at location {h5_path_name} already exists. Skipping.")
            slp_to_analysis_h5(file_full_path, h5_path_name)

#%% TODO: Edit below if you want to update this to a json config settings file
"""
import os
import json
import sys
from pathlib import Path

#%% Config default setup

def load_or_create_config(config_name="hanks_pose_config.json"):
    #Loads the JSON config if it exists. 
    #If not, creates a default template and halts execution.
    user_home = Path.home();
    # Define the default paths you want in your template
    default_config = {
        "processed_vids_folder": os.path.join(user_home, "ReformattedVideos"),
        "analysis_folder": os.path.join(user_home, "Analysis"),
        "conda_env_path": "C:/path/to/sleap_env",
        "inference_script_path": os.path.join(Path(__file__).parent.resolve(), "inference_capsule_env.py"),
        "single_model_path": "C:/path/to/model",
        "centroid_model_path": "C:/path/to/model",
        "centered_model_path": "C:/path/to/model",
        "disk_env_path": "C:/path/to/disk_env",
        "disk_files_path": "C:/path/to/disk_parent_folder"
    }
    config_path = os.path.join(user_home, config_name)
    # Check if the file already exists
    if not os.path.exists(config_path):
        print(f"Warning: Configuration file not found at '{config_path}'.")
        print("Creating a default template...")
        
        # Write the default dictionary to the file
        with open(config_path, 'w') as file:
            # indent=4 makes the JSON file readable with line breaks and spacing
            json.dump(default_config, file, indent=4)
            
        print("Template created! Please open 'config.json', update it with your actual paths, and run this script again.")
        # Exit the script so it doesn't try to run with dummy "C:/path/to/..." variables
        sys.exit(0)

    # If it does exist, load and return it normally
    manual_fields = ["conda_env_path", "single_model_path", "centroid_model_path", "centered_model_path"]
    defaults = ["C:/path/to/sleap_env", "C:/path/to/model", "C:/path/to/model", "C:/path/to/model"]
    with open(config_path, 'r') as file:
        config = json.load(file)
        for i in range(len(manual_fields)):
            if config[manual_fields[i]] == defaults[i]:
                raise Exception(f"Field {manual_fields[i]} is set to default value {defaults[i]}. Make sure that this and all other manual fields are valid.")
                sys.exit(0)
        return config"""