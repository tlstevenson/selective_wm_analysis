# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:56:14 2026

@author: cns-th-lab
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1: Model Selection and Labeling
Environment: sleappost
"""

#%% Import statements
from pathlib import Path
import os
import subprocess
import sys
import shutil
import random

#%% Define traversal function
def get_file_paths(directory_path, extension="None"):
    """Returns a list of strings containing the paths of all files in a directory."""
    path_obj = Path(directory_path)
    
    if extension=="None":
        return [str(file) for file in path_obj.iterdir() if file.is_file()]
    else:
        return[str(file) for file in path_obj.iterdir() if file.is_file() and file.suffix==extension]
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
                command = ["sleap", "track", "-i", video_list[i], "-m", centroid_model, "-m", centered_instance, "-o", write_path_list[i], "--max_instances", "1"]
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
#%%Copy model to location near prediction path
def create_write_path(model_location_pairs, vid_par_dir_param):
    all_write_paths = []
    for model_location_pair in model_location_pairs:
        centroid_model_loc, centered_model_loc = model_location_pair
        centroid_model_name = os.path.basename(centroid_model_loc)
        centered_model_name = os.path.basename(centered_model_loc)
        model_name = os.path.basename(os.path.splitext(os.path.splitext(centroid_model_loc)[0])[0])
    
        #Define a folder to copy the model to
        centroid_model_folder_loc = os.path.join(vid_par_dir_param, "models", centroid_model_name)
        centered_model_folder_loc = os.path.join(vid_par_dir_param, "models", centered_model_name)
    
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
    
        #Define write paths
        write_paths = []
        for video in curr_vids:
            try:
                write_paths.append(os.path.join(os.path.dirname(video), "predictions", model_name, vid_to_slp(os.path.basename(video))))
            except:
                print(f"Could not append path for video {video}")
        all_write_paths.append(write_paths)
    return all_write_paths
        
#%% Convert sleap projects to analysis h5 files
def slp_to_analysis_h5(slp_path, h5_path):
    """
    Converts a SLEAP .slp file to a standard analysis .h5 file using sleap-io.
    """
    print(f"  -> Exporting to {h5_path} via CLI...")
    command = ["uv", "run", "sleap", "export", str(slp_path), "-o", str(h5_path)]
    
    try:
        if not os.path.exists(h5_path):
            print(f"Converting {slp_path} to {h5_path}")
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        else:
            print(f"File at location {h5_path} already exists. Skipping.")
    except subprocess.CalledProcessError as e:
        print(e)
    return h5_path
#%% Main execiton
#%%% Select new videos by directory
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
               r"C:\Users\cns-th-lab\TannerVidsRenamed\235\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\419\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\421\Videos",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\422\Videos"]
#%%% Select all videos in the video folders
curr_vids = []
for vid_folder in vid_folders:
    curr_vids = curr_vids + get_file_paths(vid_folder, ".mp4")
print("Current videos from directories:", curr_vids)
#%%% Select n random videos as test vids from ceach animal
curr_vids = []
training_vid_paths = []
test_vid_paths = []
training_videos = ["mov_129089.mp4",
                   "mov_129117.mp4",
                   "mov_129104.mp4",
                   "mov_129096.mp4",
                   "mov_129081.mp4",
                   "mov_124606.mp4",
                   "mov_124605.mp4",
                   "mov_119009.mp4",
                   "mov_119000.mp4",
                   "mov_118992.mp4",
                   "mov_124589.mp4",
                   "mov_116507.mp4",
                   "mov_124598.mp4",
                   "mov_116498.mp4"]
num_rand_vids = 1
for vid_folder in vid_folders:
    folder_vids = get_file_paths(vid_folder, ".mp4")
    #Add training videos
    training_video = [video for video in folder_vids if os.path.basename(video) in training_videos]
    curr_vids = curr_vids + training_video
    folder_vids.remove(training_video[0])
    training_vid_paths = training_vid_paths + training_video
    
    #Add test videos
    test_vids_chosen = random.sample(folder_vids, num_rand_vids)
    curr_vids = curr_vids + test_vids_chosen
    test_vid_paths = test_vid_paths + test_vids_chosen
print("Current videos from directories:", curr_vids)
for video in curr_vids:
    print(os.path.basename(video))
print()
for video in training_vid_paths:
    print(os.path.basename(video))
print()
for video in test_vid_paths:
    print(os.path.basename(video))
print()
#%%% Define model locations
#In the form of [centroid_model_path, centered_instance_model_path]
model_locations = [[r"C:\Users\cns-th-lab\SLEAP_Projects\models\260430_182335.centroid.n=11", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260430_183010.centered_instance.n=11"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260502_198_402_237x.centroid.n=92", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260502_198_402_237x.centered_instance.n=92"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260504_198_199x_237x_402.centroid.n=112", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260504_198_199x_237x_402.centered_instance.n=112"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260523_198_199x_237x_238x_274x_400x_402x_424x_483x.centroid.n=222", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260523_198_199x_237x_238x_274x_400x_402x_424x_483x.centered_instance.n=222"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260729_198_199x_234x_237x_238x_274x_400x_402x_424x_483x.centroid.n=243", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260729_198_199x_234x_237x_238x_274x_400x_402x_424x_483x.centered_instance.n=243"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260729_198_199x_234x_237x_238x_274x_400x_402x_419x_424x_483x.centroid.n=263", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260729_198_199x_234x_237x_238x_274x_400x_402x_419x_424x_483x.centered_instance.n=263"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260730_198_199x_234x_237x_238x_274x_400x_402x_419x_421x_424x_483x.centroid.n=283", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260730_198_199x_234x_237x_238x_274x_400x_402x_419x_421x_424x_483x.centered_instance.n=283"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260730_198_199x_234x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x.centroid.n=303", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260730_198_199x_234x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x.centered_instance.n=303"],
                   [r"C:\Users\cns-th-lab\SLEAP_Projects\models\260731_198_199x_234x_235x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x.centroid.n=323", r"C:\Users\cns-th-lab\SLEAP_Projects\models\260731_198_199x_234x_235x_237x_238x_274x_400x_402x_419x_421x_422x_424x_483x.centered_instance.n=323"]]
model_locations = [model_locations[-(i+1)] for i in range(len(model_locations))] #Inverts models to recent first

#%%% Select new videos by hand
additional_vids = []
for vid_path in additional_vids:
    curr_vids.append(vid_path)
    
if len(additional_vids) > 0:
    print("Current videos :")
    print(curr_vids);
#%% Create the write paths
model_write_paths = create_write_path(model_locations, vid_par_dir)
print(model_write_paths)
if len(model_write_paths) != len(model_locations):
    raise ValueError("Number of models and model write paths do not match")

#%% Run inference and convert to h5 files
for m_idx in range(len(model_write_paths)):
    run_inference(curr_vids, model_write_paths[m_idx], model_locations[m_idx])
    for file in model_write_paths[m_idx]:
        root, ext = os.path.splitext(file)
        h5_path_name = f"{root}.h5"
        slp_to_analysis_h5(file, h5_path_name)
#%% Convertion to analysis h5 files (DEPRACATED)


label_folder_paths = [r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260523_198_199x_237x_238x_274x_400x_402x_424x_483x",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\198\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\199\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\237\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\238\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\274\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\400\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\402\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\424\Videos\predictions\260716_port_model",
               r"C:\Users\cns-th-lab\TannerVidsRenamed\483\Videos\predictions\260716_port_model"]
for folder in label_folder_paths:
    for file in os.listdir(folder):
        file_full_path = os.path.join(folder, file)
        root, ext = os.path.splitext(file_full_path)
        if ext == ".slp":
            h5_path_name = f"{root}.h5"
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